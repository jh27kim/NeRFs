from .Master import Master
import sys

sys.path.append("..")

from model.Nerf import Nerf


class NeRF(Master):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.enc_fn, self.enc_pos_dim, self.enc_dir_dim = self.encode_input()
        self.model = Nerf(self.cfg, self.enc_pos_dim, self.enc_dir_dim)

        super().__init__(cfg)

    def encode_input(self):
        # TODO
        # Positional encoding
        

        return None, 60, 24
    
