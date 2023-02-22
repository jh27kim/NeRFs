from loader.llff_loader import LLFFDataset
import torch
import torch.utils.data as data


class Master():
    def __init__(self, cfg):
        self.initialize()
        self.dataset = self.load_data()
    
    def initialize(self):
        pass

    def load_data(self):
        if self.cfg.data.dataset_type == "nerf_llff":
            self.dataset = LLFFDataset(self.cfg.data.data_root, scene_name=self.cfg.data.scene_name, factor=self.cfg.data.factor, recenter=self.cfg.data.recenter, bd_factor=self.cfg.data.bd_factor, spherify=self.cfg.data.spherify)
            self.loader = loader = data.DataLoader(self.dataset, batch_size=self.cfg.data.batch_size,shuffle=self.cfg.data.shuffle, num_workers=4)
            if self.cfg.rendering.project_to_ndc:
                self.cfg.rendering.t_near = 0.0
                self.cfg.rendering.t_far = 1.0
                print("Using NDC projection for LLFF scene. " f"Set (t_near, t_far) to ({self.cfg.rendering.t_near}, {self.cfg.rendering.t_far}).")
            else:
                self.cfg.rendering.t_near = float(torch.min(self.dataset.z_bounds) * 0.9)
                self.cfg.rendering.t_far = float(torch.max(self.dataset.z_bounds) * 1.0)
                print("Proceeding without NDC projection. " f"Set (t_near, t_far) to ({self.cfg.rendering.t_near}, {self.cfg.rendering.t_far}).")

        else:
            raise Exception("Dataset currently not available. ", self.cfg.data.datatset_type)
    

    def run(self):
        # Train / Test Model
        pass