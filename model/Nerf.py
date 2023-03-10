import torch
import torch.nn as nn
import torch.nn.functional as F

class Nerf(nn.Module):
    def __init__(self, cfg, pos_input, dir_input):
        super().__init__()

        self.pos_input = pos_input
        self.pos_depth = cfg.network.pos_depth
        self.pos_width = cfg.network.pos_width

        self.dir_input = dir_input
        self.dir_depth = cfg.network.view_dir_depth
        self.dir_width = cfg.network.view_dir_width

        self.density_dim = cfg.network.density_dim
        self.rgb_dim = cfg.network.rgb_dim

        self.skip_layer = cfg.network.skip_layer

        self.pos_layer_lst = nn.ModuleList([nn.Linear(pos_input, self.pos_width)] + \
                         [nn.Linear(self.pos_width, self.pos_width) if i != self.skip_layer else nn.Linear(self.pos_width + self.pos_input, self.pos_width) for i in range(self.pos_depth - 1)])
        
        self.feature_layer = nn.Linear(self.pos_width, self.pos_width + self.density_dim)

        self.dir_layer = nn.Linear(self.dir_input + self.pos_width, self.dir_width)
        self.rgb_layer = nn.Linear(self.dir_width, self.rgb_dim)


    def forward(self, pos, dir):
        p = pos
        for i, _ in enumerate(self.pos_layer_lst):            
            p = self.pos_layer_lst[i](p)
            p = F.relu(p)
            
            if i == self.skip_layer:
                p = torch.cat([pos, p], -1)
        
        p = self.feature_layer(p)

        sigma = F.relu(p[:, 0])
        
        p = self.dir_layer(torch.cat([p[:, 1:], dir], -1))
        p = F.relu(p)

        p = self.rgb_layer(p)
        rgb = torch.sigmoid(p)

        return rgb, sigma


        