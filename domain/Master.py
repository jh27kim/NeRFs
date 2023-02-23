import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from loader.llff_loader import LLFFDataset
from encoder.positional_encoder import PositionalEncoder
from sampler.strafied_sampler import StratifiedSampler
from renderer.quadrature_integrator import QuadratureIntegrator


class Master():
    def __init__(self):
        self.initialize()
        self.load_data()
        self.encode_input()

        self.encoder_dict = {
            "pos_encoder": self.pos_encoder,
            "dir_encoder": self.dir_encoder
        }
        
        if self.cfg.sampler.sampler_type == "stratified":
            self.sampler = StratifiedSampler(self.cfg, self.dataset.img_height, self.dataset.img_width, self.dataset.focal_length, self.logger)
        else:
            raise NotImplementedError("Sampler not implemented. ", self.cfg.sampler.sampler_type)
        
        if self.cfg.rendering.renderer_type == "quadrature":
            self.renderer = QuadratureIntegrator(self.logger, self.sampler, self.encoder_dict)
        else:
            raise NotImplementedError("Renderer not implemented. ", self.cfg.rendering.renderer_type)

    
    def initialize(self):
        if torch.cuda.is_available():
            device_id = self.cfg.cuda

        if device_id > torch.cuda.device_count() - 1:
            self.logger.warn("Invalid device ID. " f"There are {torch.cuda.device_count()} devices but got index {device_id}.")

            device_id = 0
            self.cfg.cuda.device_id = device_id

            self.logger.info(f"Set device ID to {self.cfg.cuda.device_id} by default.")
            torch.cuda.set_device(self.cfg.cuda.device_id)
            self.logger.info(f"CUDA device detected. Using device {torch.cuda.current_device()}.")

        else:
            self.logger.warn("CUDA is not supported on this system. Using CPU by default.")

    def load_data(self):
        if self.cfg.data.dataset_type == "nerf_llff":
            self.dataset = LLFFDataset(self.cfg.data.data_root, self.logger, scene_name=self.cfg.data.scene_name, factor=self.cfg.data.factor, recenter=self.cfg.data.recenter, bd_factor=self.cfg.data.bd_factor, spherify=self.cfg.data.spherify)
            self.loader = data.DataLoader(self.dataset, batch_size=self.cfg.data.batch_size,shuffle=self.cfg.data.shuffle, num_workers=4)

            if self.cfg.rendering.project_to_ndc:
                self.cfg.rendering.t_near = 0.0
                self.cfg.rendering.t_far = 1.0
                self.logger.info("Using NDC projection for LLFF scene. " f"Set (t_near, t_far) to ({self.cfg.rendering.t_near}, {self.cfg.rendering.t_far}).")
            else:
                self.cfg.rendering.t_near = float(torch.min(self.dataset.z_bounds) * 0.9)
                self.cfg.rendering.t_far = float(torch.max(self.dataset.z_bounds) * 1.0)
                self.logger.info("Proceeding without NDC projection. " f"Set (t_near, t_far) to ({self.cfg.rendering.t_near}, {self.cfg.rendering.t_far}).")

        else:
            raise NotImplementedError("Dataset currently not available. ", self.cfg.data.datatset_type)
    
    def encode_input(self):
        if self.cfg.encoding.encoding_type == "positional":
            self.pos_encoder = PositionalEncoder(self.cfg.encoding.pos_dim, self.cfg.encoding.pos_encoding, self.cfg.encoding.include_input)
            self.dir_encoder = PositionalEncoder(self.cfg.encoding.view_dir_dim, self.cfg.encoding.view_dir_encoding, self.cfg.encoding.include_input)

        else:
            raise NotImplementedError("Encoding function not implemented. ", self.cfg.encoding.encoding_type)

    def run(self):
        for _img, _pose in tqdm(self.loader):
            img = _img.squeeze()
            pose = _pose.squeeze()

            xyz_coarse, ray_d_coarse, delta_coarse = self.sampler.sample_rays(pose, (self.cfg.rendering.t_near, self.cfg.rendering.t_far), self.cfg.sampler.num_samples_coarse)
            rgb_coarse, weight_coarse, sigma_coarse, radiance_coarse = self.renderer.render_rays(xyz_coarse, ray_d_coarse, delta_coarse, self.model_coarse)

            if self.cfg.sampler.hierarchial_sampling:
                xyz_refine, ray_d_refine, delta_refine = self.sampler.sample_rays(pose, (self.cfg.rendering.t_near, self.cfg.rendering.t_far), self.cfg.sampler.num_samples_coarse, self.cfg.sampler.num_samples_refine, weight_coarse, self.cfg.sampler.hierarchial_sampling)
                rgb_refine, weight_refine, sigma_refine, radiance_refine = self.renderer.render_rays(xyz_refine, ray_d_refine, delta_refine, self.model_refine)

            