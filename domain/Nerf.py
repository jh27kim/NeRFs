from .Master import Master
import sys

sys.path.append("..")

from model.Nerf import Nerf


class NeRF(Master):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.model_refine = None

        super().__init__()

        self.model_coarse = Nerf(self.cfg, self.pos_encoder.get_out_dim(), self.dir_encoder.get_out_dim())
        if self.cfg.sampler.hierarchial_sampling:
            self.model_refine = Nerf(self.cfg, self.pos_encoder.get_out_dim(), self.dir_encoder.get_out_dim())