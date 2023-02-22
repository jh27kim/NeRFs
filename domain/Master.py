import torch
import torch.utils.data as data

from loader.llff_loader import LLFFDataset
from encoder.positional_encoder import PositionalEncoder
from sampler.strafied_sampler import StratifiedSampler
from renderer.quadrature_integrator import QuadratureIntegrator


class Master():
    def __init__(self):
        self.initialize()
        self.load_data()
        self.encode_input()
        
        if self.cfg.sampler.sampler_type == "stratified":
            self.sampler = StratifiedSampler()
        else:
            raise NotImplementedError("Sampler not implemented. ", self.cfg.sampler.sampler_type)
        
        if self.cfg.rendering.renderer_type == "quadrature":
            self.renderer = QuadratureIntegrator()
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
        # Train / Test Model
        pass