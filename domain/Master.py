import torch
import torch.utils.data as data
from tqdm import tqdm
import os 
import torchvision.utils

from loader.llff_loader import LLFFDataset
from encoder.positional_encoder import PositionalEncoder
from sampler.strafied_sampler import StratifiedSampler
from renderer.quadrature_integrator import QuadratureIntegrator

"""
PyTorch implementation of NeRF (https://arxiv.org/abs/2003.08934)

Most of the implementations inspired from 
    - https://github.com/DveloperY0115/torch-NeRF/
    - https://github.com/yenchenlin/nerf-pytorch/

"""


class Master():
    def __init__(self):
        self.log_dir = self.cfg.logs.dir
        self.ckpt_dir = os.path.join(self.log_dir, self.cfg.data.dataset_type, self.cfg.data.scene_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(os.path.join(self.ckpt_dir))

        self.output_dir = os.path.join(self.cfg.output, self.cfg.data.dataset_type, self.cfg.data.scene_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(os.path.join(self.output_dir))

        self.initialize()
        self.load_data()
        self.encode_input()

        self.encoder_dict = {
            "pos_encoder": self.pos_encoder,
            "dir_encoder": self.dir_encoder
        }
        
        if self.cfg.sampler.sampler_type == "stratified":
            self.sampler = StratifiedSampler(self.cfg, self.dataset.img_height, self.dataset.img_width, self.dataset.focal_length, self.logger, self.device)
        else:
            raise NotImplementedError("Sampler not implemented. ", self.cfg.sampler.sampler_type)
        
        if self.cfg.rendering.renderer_type == "quadrature":
            self.renderer = QuadratureIntegrator(self.logger, self.sampler, self.encoder_dict, self.device)
        else:
            raise NotImplementedError("Renderer not implemented. ", self.cfg.rendering.renderer_type)

    
    def initialize(self):
        if torch.cuda.is_available():
            device_id = self.cfg.cuda

            if device_id > torch.cuda.device_count() - 1:
                self.logger.warn("Invalid device ID. " f"There are {torch.cuda.device_count()} devices but got index {device_id}.")

                device_id = 0
                self.cfg.cuda = device_id

                self.logger.info(f"Set device ID to {self.cfg.cuda.device_id} by default.")
                
            torch.cuda.set_device(self.cfg.cuda)
            self.device = torch.cuda.current_device()
            self.logger.info(f"CUDA device detected. Using device {torch.cuda.current_device()}.")

        else:
            self.logger.warn("CUDA is not supported on this system. Using CPU by default.")


    def init_optim(self):
        self.optimizer = None
        self.scheduler = None

        init_lr = self.cfg.train.optim.init_lr
        end_lr = self.cfg.train.optim.end_lr
        num_iter = self.cfg.train.optim.num_iter
        eps = self.cfg.train.optim.eps

        params = list(self.model_coarse.parameters())
        if self.cfg.sampler.hierarchial_sampling:
            params += list(self.model_refine.parameters())
        
        if self.cfg.train.optim.optim_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=init_lr, eps=eps)
        else:
            raise Exception("Selected optimizer is not available.", self.cfg.train.optim.optim_type)
        
        if self.cfg.train.optim.scheduler_type == "exp":
            gamma = pow(end_lr / init_lr, 1 / num_iter)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        else:
            raise Exception("Selected scheduler is not available.", self.cfg.train.optim.scheduler_type)
        

    def init_loss(self):
        self.loss_fn = None
        if self.cfg.network.model == "nerf":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise Exception("Selected model is not available.", self.cfg.network.models)


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
        self.init_optim()
        self.init_loss()
        start = self._load_ckpt()

        if self.cfg.test:
            intrinsics = {"f_x": self.dataset.focal_length, "f_y": self.dataset.focal_length, "img_width": self.dataset.img_width, "img_height": self.dataset.img_height}
            extrinsics = self.dataset.render_poses
            img_res = (self.dataset.img_height, self.dataset.img_width)
            save_dir = os.path.join(self.output_dir, f"/test")

            self._visualize(intrinsics, extrinsics, img_res, save_dir)

            return 0

        for epoch in tqdm(range(start, self.cfg.train.optim.num_iter//len(self.dataset))):
            total_loss = 0.
            total_coarse_loss = 0.
            total_refine_loss = 0.

            _img_cnt = 1
            for _img, _pose in self.loader:
                self.optimizer.zero_grad()

                gt_img = _img.squeeze()
                gt_img = torch.reshape(gt_img, (-1, 3))
                pose = _pose.squeeze()

                xyz_coarse, ray_d_coarse, delta_coarse = self.sampler.sample_rays(pose, (self.cfg.rendering.t_near, self.cfg.rendering.t_far), self.cfg.sampler.num_samples_coarse)
                rgb_coarse, weight_coarse, sigma_coarse, radiance_coarse = self.renderer.render_rays(xyz_coarse, ray_d_coarse, delta_coarse, self.model_coarse)

                target_pixel_index = self.sampler.get_target_index
                gt_target_pixel = gt_img[target_pixel_index, ...].cuda()

                coarse_loss = self.loss_fn(gt_target_pixel, rgb_coarse)
                total_coarse_loss += coarse_loss.item()

                self.logger.info(f"Coarse Done {_img_cnt} / {len(self.loader)}. Coarse loss: {coarse_loss.item()}")

                if self.cfg.sampler.hierarchial_sampling:
                    xyz_refine, ray_d_refine, delta_refine = self.sampler.sample_rays(pose, (self.cfg.rendering.t_near, self.cfg.rendering.t_far), self.cfg.sampler.num_samples_coarse, self.cfg.sampler.num_samples_refine, weight_coarse, self.cfg.sampler.hierarchial_sampling, target_pixel_index)
                    rgb_refine, weight_refine, sigma_refine, radiance_refine = self.renderer.render_rays(xyz_refine, ray_d_refine, delta_refine, self.model_refine)
                    
                    refine_loss = self.loss_fn(gt_target_pixel, rgb_refine)
                    coarse_loss += refine_loss
                    total_refine_loss += refine_loss.item()
                
                    self.logger.info(f"Refine Done {_img_cnt} / {len(self.loader)}. Refine loss: {refine_loss.item()}")

                _img_cnt += 1

                total_loss += coarse_loss.item()
                coarse_loss.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                self.logger.info(f"Image: {_img_cnt}. Total loss: {coarse_loss.item()}")

            total_loss /= len(self.loader)
            total_coarse_loss /= len(self.loader)
            total_refine_loss /= len(self.loader)

            self.logger.info(f"Epoch: {epoch+1} / {self.cfg.train.optim.num_iter//len(self.dataset)}. Total loss: {total_loss} | Total coarse loss {total_coarse_loss} | Total refine loss {total_refine_loss}")

            if (epoch + 1) % self.cfg.train.log.epoch_btw_ckpt == 0:
                self._save_ckpt(epoch)

            if (epoch + 1) % self.cfg.train.log.epoch_btw_vis == 0:
                intrinsics = {"f_x": self.dataset.focal_length, "f_y": self.dataset.focal_length, "img_width": self.dataset.img_width, "img_height": self.dataset.img_height}
                extrinsics = self.dataset.render_poses
                img_res = (self.dataset.img_height, self.dataset.img_width)
                save_dir = os.path.join(self.log_dir, f"vis/epoch_{epoch}")

                self._visualize(intrinsics, extrinsics, img_res, save_dir)


    def _visualize(self, intrinsics, extrinsics, img_res, save_dir, num_imgs=None):
        test_sampler = StratifiedSampler(self.cfg, intrinsics["img_width"], intrinsics["img_height"], intrinsics["f_x"], self.logger)
        test_renderer = QuadratureIntegrator(self.logger, test_sampler, self.encoder_dict)

        with torch.no_grad():
            for pose_idx, pose in tqdm(enumerate(extrinsics)):
                if num_imgs is not None:
                    if pose_idx >= num_imgs:
                        break

            img_height, img_width = img_res
            
            xyz_coarse, ray_d_coarse, delta_coarse = test_sampler.sample_rays(pose, (self.cfg.rendering.t_near, self.cfg.rendering.t_far), self.cfg.sampler.num_samples_coarse)
            _, weight_coarse, _, _ = test_renderer.render_rays(xyz_coarse, ray_d_coarse, delta_coarse, self.model_coarse)

            if self.cfg.sampler.hierarchial_sampling:
                xyz_refine, ray_d_refine, delta_refine = test_sampler.sample_rays(pose, (self.cfg.rendering.t_near, self.cfg.rendering.t_far), self.cfg.sampler.num_samples_coarse, self.cfg.sampler.num_samples_refine, weight_coarse, self.cfg.sampler.hierarchial_sampling)
                rgb, _, _, _ = test_renderer.render_rays(xyz_refine, ray_d_refine, delta_refine, self.model_refine)

            pixel_pred = rgb.reshape(img_height, img_width, -1)
            pixel_pred = pixel_pred.permute(2, 0, 1)

            # save the image
            torchvision.utils.save_image(
                pixel_pred,
                os.path.join(save_dir, f"{str(pose_idx).zfill(5)}.png"),
            )


    def _save_ckpt(self, epoch):
        if os.path.exists(self.ckpt_dir):
            self.logger.error("Checkpoint directory is not created. Skipping process.", self.ckpt_dir)

        ckpt_file = os.path.join(self.ckpt_dir, f"ckpt_{str(epoch).zfill(6)}.pth")

        ckpt = {
            "epoch": epoch,
        }

        ckpt["coarse_model"] = self.model_coarse.radiance_field.state_dict()
        if self.model_refine is not None:
            ckpt["refine_model"] = self.model_refine.radiance_field.state_dict()

        if self.optimizer is not None:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(ckpt, ckpt_file)       


    def _load_ckpt(self):
        epoch = 0
        ckpt_file = os.path.join(self.ckpt_dir, self.cfg.train.ckpt.filename)

        if not os.path.exists(ckpt_file):
            self.logger.warn("Checkpoint not found, proceeding at epoch 0")
            return epoch
        
        self.logger.info(f"Checkpoint found. Loading {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location="cpu")

        # load epoch
        epoch = ckpt["epoch"]

        # load scene(s)
        self.model_coarse.load_state_dict()(ckpt["coarse_model"]).to(torch.cuda.current_device())
        if self.model_refine is not None:
            self.model_refine.load_state_dict(ckpt["refine_model"]).to(torch.cuda.current_device())

        # load optimizer and scheduler states
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.logger.info("Check point loaded. Epoch at : ", epoch)

        return epoch
