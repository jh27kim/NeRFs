from .Master import Master
import sys

sys.path.append("..")

from model.Nerf import Nerf


class NeRF(Master):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        super().__init__()

        self.model = Nerf(self.cfg, self.pos_encoder.get_out_dim(), self.dir_encoder.get_out_dim())
        
