import torch
from tqdm import tqdm

class QuadratureIntegrator():

    def __init__(self, logger, sampler, encoder_dict):
        self.logger = logger
        self.sampler = sampler
        self.pos_encoder = encoder_dict["pos_encoder"]
        self.dir_encoder = encoder_dict["dir_encoder"]

    def render_rays(self, xyz, dir, delta, model, batch=64):
        """
        Given XYZ and view direction, returns pixel rgb, weight, sigma and radiance.
        Quadratic integration used, for more details check (https://arxiv.org/abs/2003.08934).
        
        Args:
            xyz: 3D points. (number of pixels x number of z-samples x 3)
            dir: 3D direction. (number of pixels x number of z-samples x 3)
            delta: Distance between two adjacent 3D points in z-direction. (number of pixels x number of z-samples)

        Returns:
            rgb: RGB of the pixel (number of pixels x 3)
            weight: Weight distribution of each ray. (number of pixels x number of z-samples)
            sigma: Density of queried 3D points (number of pixels x number of z-samples)
            radiance: Radiance of queried 3D points (number of pixels x number of z-samples x 3)
        """

        rgb = []
        weights = []
        sigma = []
        radiance = []

        batch_size, sample_size = batch, xyz.shape[1]
        for i in range(0, xyz.shape[0], batch):
            xyz_flat = torch.reshape(xyz[i:i+batch], (-1, xyz.shape[-1]))
            dir_flat = torch.reshape(dir[i:i+batch], (-1, dir.shape[-1]))
            delta_batch = delta[i:i+batch]

            xyz_flat_enc = self.pos_encoder.encode(xyz_flat)
            dir_flat_enc = self.dir_encoder.encode(dir_flat)

            radiance_batch, sigma_batch = model(xyz_flat_enc, dir_flat_enc)
            radiance_batch = radiance_batch.reshape(batch_size, sample_size, -1)
            sigma_batch = sigma_batch.reshape(batch_size, sample_size)

            rgb_batch, weight_batch = self._integrate(radiance_batch, sigma_batch, delta_batch)

            rgb.append(rgb_batch)
            weights.append(weight_batch)
            sigma.append(sigma_batch)
            radiance.append(radiance_batch)

        rgb = torch.cat(rgb, 0)
        weights = torch.cat(weights, 0)
        sigma = torch.cat(sigma, 0)
        radiance = torch.cat(radiance, 0)

        return rgb, weights, sigma, radiance
    
        
    def _integrate(self, radiance_batch, sigma_batch, delta):
        """
        Integrate(Quadrature) samples along the ray.
        
        Args:
            radiance_batch: batch of radiance samples. (B x S x 3)
            sigma_batch: batch of density samples. (B x S)
            delta: bins along the z-direction (B x S)
        
        Returns:
            rgb_batch: RGB of pixels (B x 3)
            weight_Batch: weights along the ray. (B x S)
        """

        alpha = 1. - torch.exp(-sigma_batch * delta)
        transmittance = torch.exp(-torch.cumsum(torch.cat([torch.zeros(sigma_batch.shape[0], 1), sigma_batch * delta], -1), -1))[..., :-1]

        weight_batch = transmittance * alpha

        rgb_batch = torch.sum(radiance_batch * weight_batch.unsqueeze(-1), 1)

        return rgb_batch, weight_batch