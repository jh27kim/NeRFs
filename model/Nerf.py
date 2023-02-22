import torch
import torch.nn as nn

class Nerf(nn.Module):
    def __init__(self, cfg, pos_input, dir_input):
        super().__init__()

        self.pos_input = pos_input
        self.pos_depth = cfg.network.pos_dim
        self.pos_width = cfg.network.pos_width

        self.dir_input = dir_input
        self.dir_depth = cfg.network.view_dir_depth
        self.dir_width = cfg.network.view_dir_width

        
        
