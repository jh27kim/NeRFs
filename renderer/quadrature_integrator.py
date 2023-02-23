import torch

class QuadratureIntegrator():

    def __init__(self, logger, sampler, encoder_dict):
        self.logger = logger
        self.sampler = sampler
        self.pos_encoder = encoder_dict["pos_encoder"]
        self.dir_encoder = encoder_dict["dir_encoder"]

    def render_rays(self, xyz, dir, delta, model, batch=1):
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

        for i in range(0, xyz.shape[0], batch):
            xyz_flat = torch.reshape(xyz[i:i+batch], (-1, xyz.shape[-1]))
            dir_flat = torch.reshape(dir[i:i+batch], (-1, dir.shape[-1]))

            xyz_flat_enc = self.pos_encoder.encode(xyz_flat)
            dir_flat_enc = self.dir_encoder.encode(dir_flat)

            radiance_batch, sigma_batch = model(xyz_flat_enc, dir_flat_enc)
            rgb_batch, weight_batch = self._integrate(radiance_batch, sigma_batch, delta)

            rgb.append(rgb_batch)
            weights.append(weight_batch)
            sigma.append(sigma_batch)
            radiance.append(radiance_batch)

        return rgb, weights, sigma, radiance
    
        
    def _integrate(self, radiance_batch, sigma_batch, delta):
        alpha = 1. - torch.exp(sigma_batch * delta)
        transmittance = torch.exp(-torch.cumsum(sigma_batch * delta))

        weight_batch = transmittance * alpha
        rgb_batch = radiance_batch * weight_batch

        return rgb_batch, weight_batch