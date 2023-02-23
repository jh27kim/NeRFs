import torch
import numpy as np

"""
Sample random pixels given an image.
Cast rays from the selected pixels, and sample along the rays (z-direction) bounded within z-bound
Most of the codes are from (https://github.com/DveloperY0115/torch-NeRF/)
"""

class StratifiedSampler():
    """
    Cast rays from target pixels and sample along the ray

    Args:
        cfg: configuration for sampling
        w: image width
        h: image height
    
    Returns:
        ray_o (num_samples x 3)
        ray_d (num_samples x 3)

    """
    def __init__(self, cfg, w, h, f, logger):
        self.target_pts = None
        self.width = w
        self.height = h
        self.focal = f
        self.logger = logger

        pts_2d = self._sample_2d_points()

        if cfg.test:
            target_idx = torch.arange(0, self.camera.img_height * self.camera.img_width)
        else:
            target_idx = torch.tensor(np.random.choice(self.height * self.width, size=[cfg.sampler.num_pixels], replace=False))
        self.target_pts = pts_2d[target_idx, :]


    def _sample_2d_points(self):
        """
        Generate 2D points given image height and width
        """
        ys = torch.arange(0, self.height)
        xs = torch.arange(0, self.width)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = (self.height - 1) - grid_y  # [0, H-1] -> [H-1, 0]

        coords = torch.cat(
            [grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],
            dim=-1,
        ).reshape(self.height * self.width, -1)

        return coords
    

    def sample_rays(self, ext, z_bound, num_samples_coarse, num_samples_refine=None, weights=None, refine=False):
        """
        Arg:
            ext: camera extrinsic parameter
            z_bound: (near, far) tuple of z-depth 
            num_samples: samples to be drawn from each ray
            weight: use inverse sampling to extract more points corresponding to the weight

        Returns:
            ray_o: 3D points along the ray (num rays x num samples x 3)
            ray_d: Cartesian direction corresponding to ray direction (num rays x num samples x 3)
        """

        ray_o, ray_d = self._cast_rays(ext)

        z_near, z_far = z_bound
        z_bins = torch.linspace(z_near, z_far, num_samples_coarse+1)[:-1]

        dist = (z_far - z_near) / num_samples_coarse

        z_samples_coarse = z_bins + (dist * torch.rand_like(z_bins))
        z_samples_coarse = z_samples_coarse.repeat(ray_o.shape[0], 1)

        if refine:
            if weights is None or num_samples_refine is None:
                if weights is None:
                    raise Exception("Weights not defined for refine sampling.")
                else:
                    raise Exception("Num samples refine not defined for refine sampling.")
            else:
                z_samples_refine = self._inverse_sampling(z_bins, weights, dist, num_samples_refine)
                z_samples, _ = torch.sort(torch.cat([z_samples_coarse, z_samples_refine], -1), -1)
        
        else:
            z_samples = z_samples_coarse
            self.logger.info(f"Hierarchial sampling done. Total {z_samples.shape} depth samples to be extracted.")
            

        delta = torch.diff(torch.cat([z_samples, 1e8 * torch.ones((z_samples.shape[0], 1))], dim=-1), n=1, dim=-1)

        ray_o = ray_o.unsqueeze(1)
        ray_o = ray_o.repeat(1, z_samples.shape[1], 1)
        
        ray_d = ray_d.unsqueeze(1)
        ray_d = ray_d.repeat(1, z_samples.shape[1], 1)

        z_samples = z_samples.unsqueeze(2)

        xyz = ray_o + z_samples * ray_d

        self.logger.info(f"Total rays {ray_o.shape[0]}. Sampled {z_samples.shape[1]} points from each ray. Sampled 3D point dimension: {xyz.shape}")

        return xyz, ray_d, delta
            

    def _cast_rays(self, ext):
        # Convert u, v -> x, y (applying inverse intrinsic)
        self.target_pts = self.target_pts.float()
        self.target_pts[:, 0]  = (self.target_pts[:, 0] - self.width//2) / self.focal
        self.target_pts[:, 1] = (self.target_pts[:, 1] - self.height//2) / self.focal

        # Cast rays, adding -1 in z-dimension
        _ray_img_plane = torch.cat([self.target_pts, -torch.ones(self.target_pts.shape[0], 1)], -1)

        # Apply c2w rotation matrix -> world coordinate rays
        ray_d = _ray_img_plane@ext[:3, :3].t()
        
        ray_o = torch.zeros_like(ray_d) + ext[:3, -1]

        assert ray_d.shape[0] == ray_o.shape[0]
        self.logger.info(f"Ray direction shape: {ray_d.shape}  -  ray origin shape: {ray_o.shape}")
        
        return ray_o, ray_d
    

    def _inverse_sampling(self, z_bins, weights, dist, num_samples_refine):
        z_bins_refine = z_bins.repeat(weights.shape[0], 1)
        weights += 1e-5

        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf[..., :-1]], -1)

        uniform_dist = torch.rand(weights.shape[0] ,num_samples_refine)
        uniform_dist.contiguous()

        idx = torch.searchsorted(cdf, uniform_dist, right=True) - 1
        z_start = torch.gather(z_bins_refine, 1, idx)
        samples = z_start + dist * torch.rand_like(z_start)

        return samples


