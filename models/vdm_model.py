import torch
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange
from tqdm import tqdm
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
import sys 
import os
from gaussian_renderer import render, compute_target_scale_per_frame
from utils.general import build_rotation, rotation2normal
# Get the absolute path of the current file
sys.path.append("./GeometryCrafter")
sys.path.append("./MoGe")
from geo_infer import *
from moge.model.v1 import *
from models.models_vdm import *
from util.utils import convert_pt3d_cam_to_3dgs_cam



import utils3d
from util.segment_utils import create_mask_generator_repvit
from torchvision.transforms import ToPILImage

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')


class VideoFrameProcessor(FrameSyn):
    """Process video frames to generate point cloud with depth estimation and refinement.
    
    This class inherits from FrameSyn and adds functionality for processing video frames,
    estimating depth using MoGe model, and refining depth estimates.
    """
    
    def __init__(self, config, normal_estimator, segment_model=None, segment_processor=None, vggt =None, moge=None, mask_generator=None, grounded_sam=None):
        """Initialize the VideoFrameProcessor.
        
        Args:
            config: Configuration dictionary containing parameters for processing
            normal_estimator: Model for normal estimation
        """
        super().__init__(config, None, None, normal_estimator)
        self.frame_buffer = []
        self.depth_buffer = []
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.camera_buffer = []
        self.points_3d_l = []
        self.colors_l = []
        self.radius_l = []
        self.normals_l = []
        self.focal_length_l = []
        self.is_sky_l = []
        
        # Object label related attributes
        self.has_object_points = False
        self.object_points_start_idx = 0
        self.object_label_name = None
        
        # Initialize MoGe and related models
        self.moge, self.pipe, self.point_map_vae = get_moge_geo_model(model_type="diff")
        # self.moge = get_moge()
        # self.vggt = create_vggt_api(device="cuda")
        # self.moge = moge
        # self.vggt = vggt
                
        # Initialize SAM mask generator
        # self.mask_generator = create_mask_generator_repvit()
        self.mask_generator = mask_generator
        self.grounded_sam = grounded_sam
    def get_sam_masks(self, image_latest, min_mask_area=500, no_refine_mask=None):
        """Get SAM masks for the input image.
        
        Args:
            image_latest: Input image tensor
            min_mask_area: Minimum area for a mask to be considered
            no_refine_mask: Ground mask to exclude segments
            
        Returns:
            list: Sorted list of SAM masks
        """
        image = ToPILImage()(image_latest.squeeze())
        image_np = np.array(image)
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]
        return sorted_mask
    
    @torch.no_grad()
    def get_camera_at_origin(self):
        height, width = self.config["orig_H"], self.config["orig_W"]
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.init_focal_length
        K[0, 1, 1] = self.init_focal_length
        K[0, 0, 2] = width // 2  # Principal point x (center of the image)
        K[0, 1, 2] = height // 2  # Principal point y (center of the image)
        K[0, 2, 3] = 1  # Homogeneous scaling
        K[0, 3, 2] = 1  # Homogeneous scaling
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((height, width),), device=self.device)
        return camera

    
    
    def convert_to_3dgs_traindata_from_images(self,imgs, gaussians = None): 
        pass

class VideoGaussianProcessor(VideoFrameProcessor):
    """Process video frames and generate Gaussian splatting model.
    
    This class inherits from VideoFrameProcessor and adds functionality for
    generating and optimizing Gaussian splatting models from video frames.
    """
    
    def __init__(self, config, normal_estimator, **dicts_keys):
        """Initialize the VideoGaussianProcessor.
        
        Args:
            config: Configuration dictionary containing parameters for processing
            normal_estimator: Model for normal estimation
        """
        super().__init__(config, normal_estimator, **dicts_keys)
        self.gaussians = None
        self.scene = None
        self.opt = None
        
        
    @torch.no_grad()
    def convert_to_3dgs_traindata(self,points,colors,normals, images, cameras, xyz_scale=1.0, remove_threshold=None, use_no_loss_mask=False):
        """
        args:
        images: [b,c,h,w]
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
            remove_threshold: Since 3DGS does not optimize very distant points well, we remove points whose distance to scene origin is greater than this threshold.
        """
        if not isinstance(images,list):
            images = [images[i] for  i in range(images.shape[0])]
            
        W, H = self.config["orig_W"], self.config["orig_H"]
        camera_angle_x = 2*np.arctan(W / (2*cameras[0].K[0,0,0].item()))
        current_pc = {"xyz":points,"rgb": colors, 'normals': normals}
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()

        if remove_threshold is not None:
            remove_threshold_scaled = remove_threshold * xyz_scale
            mask = np.linalg.norm(pcd_points, axis=0) >= remove_threshold_scaled
            pcd_points = pcd_points[:, ~mask]
            pcd_colors = pcd_colors[~mask]

        frames = []
        for i, img in enumerate(images):
            camera_angle_x = 2*np.arctan(W / (2*cameras[i].K[0,0,0].item()))
            image = ToPILImage()(img.squeeze())
            no_loss_mask = 1 if use_no_loss_mask else None
            transform_matrix_pt3d = cameras[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
            
            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
            
            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask, 'camera_angle_x': camera_angle_x}
            frames.append(frame)
            # break
        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'W': W, 'H': H}
        return train_data
        
    def generate_sky_mask(self,input_image=None, return_sem_seg=False,sky_erode_kernel_size=7,dilation_kernel_size=0):
        image = ToPILImage()(input_image.squeeze())
        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                segment_output, target_sizes=[image.size[::-1]])[0]
        sky_mask = pred_semantic_map == 2  # 2 for ade20k, 119 for coco
        if sky_erode_kernel_size > 0:
            sky_mask = erosion(sky_mask.float()[None, None], 
                            kernel=torch.ones(sky_erode_kernel_size, sky_erode_kernel_size).to(self.device)
                            ).squeeze() > 0.5
        # if dilation_kernel_size > 0:
        #     sky_mask = sky_mask.float()[None, None].to("cuda")
        #     sky_mask = F.max_pool2d(sky_mask, kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size//2)
        #     sky_mask = sky_mask.squeeze() > 0.5
        if return_sem_seg:
            return sky_mask, pred_semantic_map
        else:
            return sky_mask                
        

    def finetune_depth_model(self, model, target_depth, input_image, a,b, mask_align=None, mask_cutoff=None, cutoff_depth=None):
        print(f"Model in training mode: {model.training}")
        target_depth.requires_grad_(True)
        input_image.requires_grad_(True)
        # a.requires_grad_(True)
        # b.requires_grad_(True)
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"Parameter {name} does not require grad!")
        #     break  # Only check the first parameter
            
        
        # print(f"target_depth requires_grad: {target_depth.requires_grad}")
        # print(f"input_image requires_grad: {input_image.requires_grad}")
        # print(f"a requires_grad: {a.requires_grad if isinstance(a, torch.Tensor) else 'not tensor'}")
        # print(f"b requires_grad: {b.requires_grad if isinstance(b, torch.Tensor) else 'not tensor'}")
        
        # Check model parameters

        
        params = [{"params": model.parameters(), "lr": self.config["depth_model_learning_rate"]}]
        optimizer = torch.optim.Adam(params)
        model.train()
        model.requires_grad_(True)
        if mask_align is None:
            mask_align = target_depth > 0

        progress_bar = tqdm(range(self.config["num_finetune_depth_model_steps"]), leave=False)
        min_loss = torch.inf 
        # model.device = self.device
        model.to(self.device)
        for _ in progress_bar:
            optimizer.zero_grad()
            
            next_depth, loss = finetune_depth_model_step(
                model,
                target_depth,
                input_image,
                a, b,
                mask_align=mask_align,
                mask_cutoff=mask_cutoff,
                cutoff_depth=cutoff_depth,
            )
            progress_bar.set_postfix(loss=loss.item(),a=a,b=b)  # Update the progress bar with loss at each step
            loss.backward()
            optimizer.step()
            if loss < min_loss:
                min_loss = loss
                best_depth = next_depth
                
            # except RuntimeError:
            #     print('No valid pixels to compute depth fine-tuning loss. Skip this step.')
            #     return
        model.eval()
        model.requires_grad_(False)
        return best_depth
    
    @torch.no_grad()
    def get_camera_by_js_view_matrix(self, view_matrix, fx_wonder = None, fy_wonder = None, xyz_scale=1.0, big_view=False):
        """
        args:
            view_matrix: list of 16 elements, representing the view matrix of the camera
            xyz_scale: This was used to scale the x, y, z coordinates of the camera when converting to 3DGS.
                Need to convert it back.
        return:
            camera: PyTorch3D camera object
        """
        view_matrix = torch.tensor(view_matrix, device=self.device, dtype=torch.float).reshape(4, 4)
        xy_negate_matrix = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=self.device, dtype=torch.float)
        view_matrix_negate_xy = view_matrix @ xy_negate_matrix
        R = view_matrix_negate_xy[:3, :3].unsqueeze(0)
        T = view_matrix_negate_xy[3, :3].unsqueeze(0)
        camera = self.get_camera_at_origin()
        camera.R = R
        camera.T = T / xyz_scale
        if fx_wonder is not None:
            camera.K[0,0,0] = fx_wonder
        if fy_wonder is not None:
            camera.K[0,1,1] = fy_wonder
        return camera
    
    @torch.no_grad()
    def generate_grad_magnitude(self, disparity, threshold=2):
        vmin, vmax = disparity.min(), disparity.max()
        normalized_disparity = (disparity - vmin) / (vmax - vmin)
        cmap = plt.get_cmap('viridis')
        if isinstance(normalized_disparity, torch.Tensor):
            normalized_disparity = normalized_disparity.detach().cpu()
        rgb_image = cmap(normalized_disparity)
        rgb_image = rgb_image[...,1]
        disparity = np.uint8(rgb_image * 255)


        # Compute gradients along the x and y axis
        grad_x = cv2.Sobel(disparity, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask = torch.from_numpy(grad_magnitude > threshold)
        return mask
    
    @torch.no_grad()
    def process_single_img(
        self,
        roots,
        # cameras,
        num=1,
        total_num=1,
        background_hard_depth=1.0,
    ):
        """Process video frames to generate point cloud.
        
        Args:
            roots: List of image paths
        """
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depth_l = []
            depth_align_masks_l = []
            
            points = rearrange(points_xy, "h w c -> (h w) c")
            cameras = [self.get_camera_at_origin()]

            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
                
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            for idx in idxs:
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                depth = self.moge.model.infer(img.to(self.device))['depth'].detach().cpu()
                if depth.median() == torch.inf:
                    numer = depth[depth!=torch.inf].median()
                    depth = depth / numer /100. + 1e-4
                else:
                    depth = depth / depth.median() /100. + 1e-4
                            
                inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()        
                # inf_mask = self.generate_sky_mask(img).cpu()
                is_sky = inf_mask.cpu()            
                            
                depth[inf_mask] = 100.
                # depth[depth == torch.inf] = 0.
                normals = self.get_normal( img[None])
                normals[:, 1:] *= -1
                    
                # import pdb; pdb.set_trace()
                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                 
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(self.device)
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(), threshold=self.config.get('grad_magnitude_threshold_single', 10)) #|(depth < 1e-3)
                grad_magnitude_mask[:] = grad_magnitude_mask & self.config.get('use_grad_single', False)
                
                depth_align_masks_l.append(~grad_magnitude_mask)
                grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                # Compute now_scale for points corresponding to current camera
                filtered_points_3d = points_3d[~grad_magnitude_mask]
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=self.device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=self.device) * 0.01
                    
                    # Use current camera to compute now_scale
                    xyz_scale = 1000  # Use the default xyz_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=self.device)
                
                colors_l.append(colors[~grad_magnitude_mask])
                points_3d_l.append(filtered_points_3d)
                radius_l.append(radius[~grad_magnitude_mask[None]][None])
                normals_l.append(new_normals[~grad_magnitude_mask])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[~grad_magnitude_mask].shape[0], device=self.device))
                is_sky_l.append(is_sky[~grad_magnitude_mask])
                now_scale_l.append(now_scale_batch)
                all_depth_l.append(depth.squeeze())
                    
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=self.device)
                    
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l    

            return self.points_3d, self.colors, self.radius, self.normals, imgs, cameras, self.focal_length, self.is_sky, all_depth_l, depth_align_masks_l, self.now_scale

    @torch.no_grad()
    def process_single_img_sky(
        self,
        roots,
        # cameras,
        num=1,
        total_num=1,
        background_hard_depth=1.0,
    ):
        """Process video frames to generate point cloud.
        
        Args:
            roots: List of image paths
        """
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depth_l = []
            depth_align_masks_l = []
            
            points = rearrange(points_xy, "h w c -> (h w) c")
            cameras = [self.get_camera_at_origin()]

            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
                
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            for idx in idxs:
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                depth = self.moge.model.infer(img.to(self.device))['depth'].detach().cpu()
                if depth.median() == torch.inf:
                    numer = depth[depth!=torch.inf].median()
                    depth = depth / numer /100. + 1e-4
                else:
                    depth = depth / depth.median() /100. + 1e-4
                            
                inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()        
                # inf_mask = self.generate_sky_mask(img).cpu()
                is_sky = inf_mask.cpu()            
                            
                depth[inf_mask] = 0.9
                # depth[depth == torch.inf] = 0.
                normals = self.get_normal( img[None])
                normals[:, 1:] *= -1
                    
                # import pdb; pdb.set_trace()
                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                 
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(self.device)
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(), threshold=self.config.get('grad_magnitude_threshold_single', 10)) #|(depth < 1e-3)
                grad_magnitude_mask[:] = grad_magnitude_mask & self.config.get('use_grad_single', False)
                
                depth_align_masks_l.append(~grad_magnitude_mask)
                grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                # Compute now_scale for points corresponding to current camera
                filtered_points_3d = points_3d[((~grad_magnitude_mask)&is_sky)]
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=self.device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=self.device) * 0.01
                    
                    # Use current camera to compute now_scale
                    xyz_scale = 1000  # Use the default xyz_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=self.device)
                
                colors_l.append(colors[(~grad_magnitude_mask)&is_sky])
                points_3d_l.append(filtered_points_3d)
                radius_l.append(radius[((~grad_magnitude_mask)&is_sky)[None]][None])
                normals_l.append(new_normals[(~grad_magnitude_mask)&is_sky])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[(~grad_magnitude_mask)&is_sky].shape[0], device=self.device))
                is_sky_l.append(is_sky[(~grad_magnitude_mask)&is_sky])
                now_scale_l.append(now_scale_batch)
                all_depth_l.append(depth.squeeze())
                    
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=self.device)
                    
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l    

            return self.points_3d, self.colors, self.radius, self.normals, imgs, cameras, self.focal_length, self.is_sky, all_depth_l, depth_align_masks_l, self.now_scale

  

    @torch.no_grad()
    def process_single_img_mask(
        self,
        roots,
        input_mask,  # New: input mask for filtering the final point cloud
        cameras,
        target_depth,
        num=1,
        total_num=1,
        background_hard_depth=1.0,
    ):
        """Process video frames with input mask to generate filtered point cloud.

        Args:
            roots: List of image paths
            input_mask: Input mask (H, W) or (1, H, W) to filter final points, True=keep, False=discard
        """
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depth_l = []
            depth_align_masks_l = []

            points = rearrange(points_xy, "h w c -> (h w) c")
            # cameras = [self.get_camera_at_origin()]

            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)

            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)

            # Process input_mask: ensure correct shape and convert to bool

            # Convert mask to shape matching the point cloud (w*h,)
            
            
            for idx in idxs:
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                depth = self.moge.model.infer(img.to(self.device))['depth'].detach().cpu()
                # depth = depth / depth.median() /100. + 1e-4
                            
                inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()        
                is_sky = inf_mask.cpu()            
                            
                depth[inf_mask] = 10.
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(), threshold=10)
                grad_magnitude_mask[:] = grad_magnitude_mask & self.config.get('use_grad_single_mask', False)
                
                # mask_align = ((depth<5.) & (depth>0) & (~inf_mask.cpu()) & (input_mask.cpu()) & (target_depth.cpu()>0.) & (target_depth.cpu()<0.8)).float()
                # with torch.enable_grad():
                #     a, b = compute_scale_and_shift_full(depth.detach().cpu(), target_depth.detach().cpu(), mask_align.float())
                #     self.moge.model.train()
                #     self.moge.model.requires_grad_(True)
                #     best_depth = self.finetune_depth_model(self.moge.model, depth, img, a, b, mask_align)
                #     self.moge.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)                             
                #     self.moge = self.moge.to(self.device).eval()   
                #     self.moge.model.requires_grad_(False)
                #     self.moge.model.eval()
                    
                # depth = best_depth.detach().cpu()
                # 
                # # print(a*0,b)
                # import pdb; pdb.set_trace()
                # print(a,b)
                # depth = depth * a + b

                # import pdb; pdb.set_trace()
                
                # grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                input_mask = input_mask.cpu().squeeze() & (~grad_magnitude_mask)
                masked_depth = depth[input_mask]
                threshold = torch.quantile(masked_depth, 0.999)
                input_mask = input_mask & (depth <= threshold)
                a = depth[input_mask].max() - depth[input_mask].min()
                print(a)
                amount = self.config.get('amount', 2)
                depth /= (a.item()*50000*amount)
                plt.imsave("./cache/input_mask.png", (input_mask.cpu().float().numpy()*255).astype(np.uint8), cmap="gray")
                
                if self.config.get('use_tilt', True):
                    print("🎯 Applying 2D tilt transform with depth constraint...")
                    depth = self._apply_constrained_tilt_transform(depth, input_mask, target_depth)
                # max_now = depth[input_mask.cpu()].max().detach().cpu()
                # min_tar = target_depth[input_mask.cpu()].min().detach().cpu()
                
                # depth[input_mask.cpu()] -= (max_now-min_tar)
                target_depth[target_depth.cpu()<0.] = 1e3
                depth_diff = target_depth.cpu()[input_mask.cpu()] - depth.cpu()[input_mask.cpu()]
                depth[input_mask.cpu()] += depth_diff.min()
                
                # Apply 2D coordinate tilt transform on the object region to fit the background while keeping it in front
                
                
                # mask_np = input_mask.numpy().astype(np.uint8) * 255

                # # Define kernel, 3x3
                # kernel = np.ones((9, 9), np.uint8)

                # # Erosion
                # eroded_np = cv2.erode(mask_np, kernel, iterations=1)

                # # Convert back to bool tensor
                # eroded = torch.from_numpy(eroded_np > 0)
                # plt.imsave("./cache/input_mask_eroded.png", (eroded.cpu().float().numpy()*255).astype(np.uint8))
                # import pdb; pdb.set_trace()
                
                normals = self.get_normal( img[None])
                normals[:, 1:] *= -1
                    

                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(self.device)

                # Combine all masks: gradient mask + input_mask
                # Only keep points that are neither on gradient edges nor outside input_mask
                input_mask_flat = rearrange(input_mask.bool().squeeze()[None,None], "b c h w -> (w h b) c")
                # input_mask_flat = rearrange(eroded.bool().squeeze()[None,None], "b c h w -> (w h b) c")
                combined_mask = input_mask_flat.cpu().squeeze()
                # combined_mask = eroded.cpu().squeeze()

                # Compute now_scale for points corresponding to current camera
                filtered_points_3d = points_3d[combined_mask]
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=self.device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=self.device) * 0.01
                    
                    # Use current camera to compute now_scale
                    xyz_scale = 1000  # Use the default xyz_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=self.device)

                colors_l.append(colors[combined_mask])
                points_3d_l.append(filtered_points_3d)
                radius_l.append(radius[combined_mask[None]][None])
                normals_l.append(new_normals[combined_mask])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[combined_mask].shape[0], device=self.device))
                is_sky_l.append(is_sky[combined_mask])
                now_scale_l.append(now_scale_batch)
                all_depth_l.append(depth.squeeze())
                    
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=self.device)
                    
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l

            print(f"Point cloud size after mask filtering: {self.points_3d.shape[0]} points")
            return self.points_3d, self.colors, self.radius, self.normals, imgs, cameras, self.focal_length, self.is_sky, self.now_scale


    def _apply_constrained_tilt_transform(self, depth, object_mask, target_depth):
        """
        Apply a 2D coordinate tilt transform on the object region to minimize the difference with target_depth,
        while ensuring the object is in front of the background: depth[mask].max() <= target_depth[mask].min()

        Rigid transform form: depth_new[y,x] = depth_old[y,x] + a*x + b*y + c
        Objective: minimize ||depth_new[mask] - target_depth[mask]||^2
        Constraint: depth_new[mask].max() <= target_depth[mask].min()

        Args:
            depth: Current depth [H, W]
            object_mask: Object mask [H, W], True indicates the object region
            target_depth: Target depth (background depth) [H, W]

        Returns:
            transformed_depth: Transformed depth [H, W]
        """
        # Ensure on the correct device
        depth = depth.to(self.device)
        object_mask = object_mask.to(self.device) 
        target_depth = target_depth.to(self.device)
        
        # Get the object region
        if not object_mask.any():
            return depth
            
        # Get coordinates and depth values of the object region
        object_coords = torch.where(object_mask)
        if len(object_coords[0]) == 0:
            return depth
            
        obj_y = object_coords[0].float()  # [N]
        obj_x = object_coords[1].float()  # [N] 
        obj_depths = depth[object_mask]   # [N]
        target_depths = target_depth[object_mask]  # [N]
        
        n_points = len(obj_y)
        print(f"🎯 Optimizing 2D tilt transform for {n_points} object points")
        
        # Build design matrix A = [x, y, 1] [N x 3]
        A = torch.stack([obj_x, obj_y, torch.ones_like(obj_x)], dim=1)  # [N, 3]
        
        # Build target vector B = target_depth - current_depth [N]
        B = target_depths - obj_depths  # [N]
        
        # Step 1: Unconstrained least squares to solve for optimal parameters
        try:
            AtA = torch.matmul(A.T, A)  # [3, 3]
            AtB = torch.matmul(A.T, B)  # [3]
            
            # Solve for parameters [a, b, c]^T
            X = torch.linalg.solve(AtA, AtB)  # [3]
            a, b, c = X[0], X[1], X[2]
            
            print(f"🎯 Unconstrained optimal parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            
            # Apply transform
            depth_correction = a * obj_x + b * obj_y + c
            new_obj_depths = obj_depths + depth_correction
            
            # Check constraint: object.max() <= target.min()
            obj_max = new_obj_depths.max()
            target_min = target_depths.min()
            
            print(f"🎯 Object depth range after transform: [{new_obj_depths.min():.6f}, {obj_max:.6f}]")
            print(f"🎯 Target depth range: [{target_min:.6f}, {target_depths.max():.6f}]")
            
            # If constraint is violated, add extra depth offset to keep object in front
            # if obj_max > target_min:
            #     safety_margin = 1e-9  # Safety margin
            #     extra_offset = target_min - obj_max - safety_margin
            #     print(f"🎯 Constraint violated! Adding extra offset: {extra_offset:.6f}")
            #     c += extra_offset
            #     new_obj_depths = obj_depths + (a * obj_x + b * obj_y + c)
            
            # Apply transform to the object region
            depth[object_mask] = new_obj_depths
            
            # Compute final residual and constraint satisfaction
            final_residual = torch.mean((new_obj_depths - target_depths)**2).sqrt()
            final_max = new_obj_depths.max()
            constraint_satisfied = final_max <= target_depths.min()
            
            print(f"🎯 Final parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            print(f"🎯 Final RMS error: {final_residual:.6f}")
            print(f"🎯 Object max depth: {final_max:.6f}, Target min depth: {target_depths.min():.6f}")
            print(f"🎯 Constraint satisfied: {constraint_satisfied}")
            
        except Exception as e:
            print(f"⚠️  Tilt transform optimization failed: {e}, using original depth")
            
        return depth

    @torch.no_grad()
    def process_single_img_mask_wt_guide(
        self,
        roots,
        input_mask,  # New: input mask for filtering the final point cloud
        cameras,
        target_depth,
        in_sam_masks,
        num=1,
        total_num=1,
        background_hard_depth=1.0,
    ):
        """Process video frames with input mask to generate filtered point cloud.
        
        Args:
            roots: List of image paths
            input_mask: Input mask (H, W) or (1, H, W) to filter final points, True=keep, False=discard
        """
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depth_l = []
            depth_align_masks_l = []
            
            points = rearrange(points_xy, "h w c -> (h w) c")
            # cameras = [self.get_camera_at_origin()]

            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
                
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            # Process input_mask: ensure correct shape and convert to bool

            # Convert mask to shape matching the point cloud (w*h,)
            
            
            for idx in idxs:
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                depth = self.moge.model.infer(img.to(self.device))['depth'].detach().cpu()
                depth = depth / depth.median() /100. + 1e-4
                            
                inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()        
                is_sky = inf_mask.cpu()            
                            
                depth[inf_mask] = 10.
                mask_align = ((depth<5.) & (depth>0) & (~inf_mask.cpu()) & (~input_mask.cpu()) & (target_depth.cpu()>0.) & (target_depth.cpu()<0.2)).float()
                a, b = compute_scale_and_shift_full(depth, target_depth, mask_align.float())
                # with torch.enable_grad():
                #     self.moge.model.train()
                #     self.moge.model.requires_grad_(True)
                #     best_depth = self.finetune_depth_model(self.moge.model, depth, img, a, b, mask_align)
                #     self.moge.model.requires_grad_(False)
                #     self.moge.model.eval()
                    
                print(a,b)
                depth = depth * a + b
                shift = compute_shift_only(depth, target_depth, mask_align.float())
                depth += shift
                # import pdb; pdb.set_trace()
                # depth = best_depth.detach().cpu()
                print("refining depth")
                mask_align = ((depth<5.) & (depth>0) & (~inf_mask.cpu()) & (target_depth.cpu()>0.) & (target_depth.cpu()<1.2)).float()

                for segment in in_sam_masks:
                    sam_mask = torch.tensor(segment)
                    sam_mask_zbuf = sam_mask #& mask_align.bool() 
                    # if sam_mask_zbuf.sum() <20:
                    #     continue
                    depths_now = depth[sam_mask_zbuf] 
                    target_now = target_depth[sam_mask_zbuf]
                    thra_now = torch.quantile(depths_now, 0.01)
                    thra_tar = torch.quantile(target_now, 0.99)
                    # diff = target_now.max() - depths_now.min()
                    diff = thra_tar - thra_now
                    # if torch.abs(diff) > 1e-5:
                    depth[sam_mask] += diff.to(depth.device) + 1e-4#.clip(min=0) 
                    # if diff < - 1e-4 :
                    #     depth[sam_mask] += diff.to(depth.device) + 1e-4
                
                # diff = target_depth[input_mask.cpu()].max() - depth[input_mask.cpu()].min()
                # depth[input_mask.cpu()] += diff.to(depth.device).clip(min=0)
                
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(),20)
                # grad_magnitude_mask[:] = False
                # grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                input_mask = input_mask.cpu().squeeze() & (~grad_magnitude_mask)


                normals = self.get_normal( img[None])
                normals[:, 1:] *= -1
                    

                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(self.device)

                # Combine all masks: gradient mask + input_mask
                # Only keep points that are neither on gradient edges nor outside input_mask
                input_mask_flat = rearrange(input_mask.bool().squeeze()[None,None], "b c h w -> (w h b) c")
                combined_mask = input_mask_flat.cpu().squeeze()

                # Compute now_scale for points corresponding to current camera
                filtered_points_3d = points_3d[combined_mask]
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=self.device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=self.device) * 0.01
                    
                    # Use current camera to compute now_scale
                    xyz_scale = 1000  # Use the default xyz_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=self.device)

                colors_l.append(colors[combined_mask])
                points_3d_l.append(filtered_points_3d)
                radius_l.append(radius[combined_mask[None]][None])
                normals_l.append(new_normals[combined_mask])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[combined_mask].shape[0], device=self.device))
                is_sky_l.append(is_sky[combined_mask])
                now_scale_l.append(now_scale_batch)
                all_depth_l.append(depth.squeeze())
                    
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=self.device)
                    
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l

            print(f"Point cloud size after mask filtering: {self.points_3d.shape[0]} points")
            return self.points_3d, self.colors, self.radius, self.normals, imgs, cameras, self.focal_length, self.is_sky, self.now_scale

    
    @torch.no_grad()
    def process_zoomin_frames_rewrite(
        self,
        roots,
        cameras,
        gaussians, 
        xyz_scale,
        opt, 
        total_num=49,
        background_hard_depth=1.0,
        BG_COLOR=[0.7, 0.7, 0.7],
    ):
        device = self.device 
        num = 3
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            bg_mask = None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            now_scale_l = []
            points = rearrange(points_xy, "h w c -> (h w) c")
            
            # Maintain the original order for depth estimation
            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            # Store point cloud data for each frame
            frame_data = []
            
            # First perform depth estimation in 0->1->2 order
            for iter_idx, idx in enumerate(idxs):
                if idx != 0:
                    scaling = True 
                    print(idx)
                else:
                    scaling = False
                    
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                with torch.no_grad():
                    if len(points_3d_l) > 0:
                        points_3d_now = torch.cat(points_3d_l, dim=0)
                        colors_now = torch.cat(colors_l, dim=0)
                        radius_now = torch.cat(radius_l, dim=-1)   
                        raster_settings = PointsRasterizationSettings(
                            image_size=[self.config["orig_H"], self.config["orig_W"]],
                            radius = radius_now.to(device),
                            points_per_pixel = 8,
                            bin_size = 0
                        )

                        renderer = PointsRenderer(
                            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                        )
                        point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                        zbuf = zbuf.cpu().squeeze().cpu()[...,0]
                        bg_mask = bg_mask.squeeze() 
                        bg_mask = rearrange(bg_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                with torch.no_grad():
                    if zbuf is None:
                        print("Using Gaussians for zoom-in depth estimation")
                        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                        render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                        bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.8
                        bg_mask_gaussian=rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                        if bg_mask is not None:
                            bg_mask = bg_mask & bg_mask_gaussian
                        else:
                            bg_mask = bg_mask_gaussian
                        depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                    else:
                        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                        render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                        bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.8
                        bg_mask_gaussian=rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                        if bg_mask is not None:
                            bg_mask = bg_mask & bg_mask_gaussian
                        else:
                            bg_mask = bg_mask_gaussian
                        depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                                                
                        depth = self.moge.model.infer(img.to(device))['depth'].detach().cpu()
                        inf_mask = (depth == torch.inf)
                        depth[inf_mask] = 0
                        ####################### 
                        # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                        # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                        # depth_gau = render_pkg["median_depth"][0:1].squeeze().cpu() / xyz_scale
                        # # zbuf = depth_gau
                        # zbuf = depth_gau.clone().detach().requires_grad_(True)  

                        
                        ############################
                        mask_align = ((zbuf<9.) & (zbuf>0) & (~inf_mask)).float()
                        # mask_align = ((zbuf < 5.) & (zbuf > 0) & (~inf_mask)).float()
                        # import pdb; pdb.set_trace()
                        if mask_align.sum() == 0:
                            print("mask_align is empty, skipping depth fine-tune")
                            scaling = False  
                        zbuf = zbuf.clip(0., 1e9)
                        a, b = compute_scale_and_shift_full(depth, zbuf, ((zbuf<5.) & (zbuf>0) & (~inf_mask)).float())
                        mask_align = mask_align.bool()
                        with torch.enable_grad():
                            if scaling and self.config['num_finetune_depth_model_steps'] > 0:
                                self.moge.model.train()
                                self.moge.model.requires_grad_(True)
                                best_depth = self.finetune_depth_model(self.moge.model, zbuf, img, a, b, mask_align)
                                self.moge.model.requires_grad_(False)
                                self.moge.model.eval()
                    
                    if best_depth is not None:                
                        depth = best_depth.detach().cpu()                
                        best_depth = None   
                    else:     
                        # try:
                        #     depth = depth*a*0.9 +b
                        # except:
                            depth = depth_gau
                    
                    inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()
                              
                    if scaling:
                        self.moge.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             
                        self.moge = self.moge.to(device).eval()   
                    
                    use_scaling_pull = self.config.get('use_scaling_pull', True)
                    if zbuf is not None and scaling and use_scaling_pull:
                        print("refining depth")
                        segments = self.get_sam_masks(img, min_mask_area=1)
                        for segment in segments:
                            sam_mask = torch.tensor(segment['segmentation'])
                            sam_mask_zbuf = sam_mask & mask_align 
                            depths_now = depth[sam_mask_zbuf] 
                            depths_now_zbuf = zbuf.squeeze()[sam_mask_zbuf]
                            diff = depths_now_zbuf.median() - depths_now.median() 
                            if torch.abs(diff) > 1e-5:
                                depth[sam_mask] += diff.to(depth.device)
                    ############################################
                    if zbuf is not None and self.config.get('pull_foreground_depth_rewrite', False):
                        foreground_list, background_description = self.grounded_sam.extract_foreground_background(root)
                        print(f"   Foreground objects: {foreground_list}")
                        print(f"   Background description: {background_description}")
                    
                        # Step 2: Use GroundedSAM to extract foreground mask
                        print("🎯 Step 2: Extracting foreground mask with GroundedSAM...")

                        
                        foreground_mask, combined_mask, masks = self.grounded_sam.get_combined_foreground_mask(root, foreground_list, kernel_size=21)
                        print(f"   Foreground mask shape: {foreground_mask.shape}")
                        for mask in masks:
                            mask = mask.squeeze()
                            depth[mask] += 0.7*(depth[mask].median() - depth[mask])

                            # depth = self._apply_constrained_tilt_transform(depth, mask&mask_align, zbuf)
                    
                        # print("pulling depth all")
                    depth_gau_np = depth_gau.squeeze().cpu().numpy().astype(np.float32)
                    # Create mask for holes (where opacity is low - these are the holes!)
                    opacity_mask = render_pkg["depth"].detach().cpu()[0] < 0.
                    hole_mask = opacity_mask.squeeze().numpy().astype(bool)
                    
                    # Fill holes if there are any
                    if hole_mask.sum() > 0:
                        depth_filled = inpaint_nearest_bilateral_preserve(depth_gau_np, hole_mask, bilateral=True)
                        depth_filled = torch.tensor(depth_filled, dtype=torch.float32)
                    else:
                        depth_filled = depth_gau
                        
                        # diff = depth_filled - depth
                        # depth += diff.to(depth.device)*0.7


                    
                    ############################################
                    inf_mask = inf_mask & (depth_filled>110.)
                    is_sky = inf_mask.cpu()  
                    if inf_mask.any():
                        depth[inf_mask] = 100.
                    
                    hole_mask = (~inf_mask) & (depth>90.)
                    # if hole_mask.any():
                    #     depth_filled = inpaint_nearest_bilateral_preserve(depth.squeeze().cpu().numpy(), hole_mask, bilateral=True)
                    #     depth_filled = torch.tensor(depth_filled, dtype=torch.float32).to(depth.device)
                    #     depth=depth_filled
                    if hole_mask.any():
                        depth[hole_mask] = depth_filled[hole_mask]
                    # if idx == 1:
                    #     import pdb; pdb.set_trace()
                    if idx==0:
                        depth = depth_gau
                    normals = self.get_normal(img[None])
                    normals[:, 1:] *= -1
                    print("min_depth",depth.min(),depth.max(), "depth_filled", depth_filled.min(), depth_filled.max())

                    
                    depth_mask = (depth >1e-3)
                    depth_mask = rearrange(depth_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(device)
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(), threshold=list(self.config.get('rewrite_grad_thra', [5,5]))[iter_idx-1] )
                grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                if not self.config.get('use_grad_rewrite', False):
                    grad_magnitude_mask[:] = False
                depth_mask = depth_mask & (~grad_magnitude_mask)
               
                # Store all point cloud data for current frame
                frame_data.append({
                    'points_3d': points_3d,
                    'colors': colors,
                    'radius': radius,
                    'normals': new_normals,
                    'depth_mask': depth_mask,
                    'camera': camera,
                    'focal_length': camera.K[0,0,0],
                    'is_sky': is_sky
                })

                # Handle gaussians deletion
    # Handle gaussians deletion
                if True:
                    if idx > 0:
                        bg_mask[:] = depth_mask
                    else:
                        # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cameras[idxs[1]], xyz_scale=xyz_scale, config=self.config)
                        # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                        # visible_mask = render_pkg["visibility_filter"]
                        visible_mask = gaussians.get_inscreen_points(cameras[idxs[1]])
                        # gaussians.delete_mask_all |= visible_mask
                        # gaussians.delete_all_points(gaussians.delete_mask_all)
                        gaussians.merge_all_to_trainable()
                        label_mask = gaussians.point_labels 
                        label_mask[visible_mask&(label_mask == 0)] = int(1e5)
                        gaussians.point_labels = label_mask
                        bg_mask[:] = depth_mask

                # Maintain lists like points_3d_l during the depth estimation stage
                filtered_points_3d = points_3d[depth_mask]
                
                # Compute now_scale for points corresponding to current camera
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=device) * 0.01
                    
                    # Use current camera to compute now_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=device)
                
                points_3d_l.append(filtered_points_3d)
                colors_l.append(colors[depth_mask])
                radius_l.append(radius[depth_mask[None]][None])
                normals_l.append(new_normals[depth_mask])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[depth_mask].shape[0], device=device))
                now_scale_l.append(now_scale_batch)

            # Add point clouds in 2->1->0 order
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            now_scale_l = []
            is_sky_l = []
            
            # Reverse iterate frame_data (2->1->0)
            for i in range(len(frame_data)-1, -1, -1):
                if i == 0 :
                    continue
                frame = frame_data[i]
                
                # If this is the first frame (idx=2), add all points directly
                if i == len(frame_data)-1:
                    filtered_points_3d = frame['points_3d'][frame['depth_mask']]
                    
                    # Compute now_scale for points corresponding to current frame
                    if len(filtered_points_3d) > 0:
                        # Create default rotations and scales for the current batch of point cloud
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        # Use current frame's camera to compute now_scale
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            frame['camera'],
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                            min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    print(frame['points_3d'][frame['depth_mask']].shape)
                    points_3d_l.append(filtered_points_3d)
                    colors_l.append(frame['colors'][frame['depth_mask']])
                    radius_l.append(frame['radius'][frame['depth_mask'][None]][None])
                    normals_l.append(frame['normals'][frame['depth_mask']])
                    focal_length_l.append(torch.tensor([frame['focal_length']]*frame['colors'][frame['depth_mask']].shape[0], device=device))
                    is_sky_l.append(frame['is_sky'][frame['depth_mask']])
                    now_scale_l.append(now_scale_batch)
                    continue
                    
                # For other frames, check if already rendered by previous frames
                points_3d_now = torch.cat(points_3d_l, dim=0)
                colors_now = torch.cat(colors_l, dim=0)
                radius_now = torch.cat(radius_l, dim=-1)
                # if set_front:
                #     raster_settings = PointsRasterizationSettings(
                #         image_size=[self.config["orig_H"], self.config["orig_W"]],
                #         radius = radius_now.to(device),
                #         points_per_pixel = 8,
                #     )
                #     renderer = PointsRenderer(
                #         rasterizer=PointsRasterizer(cameras=frame['camera'], raster_settings=raster_settings),
                #         compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                #     )
                #     point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                #     _, _, rendered_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                #     rendered_mask = rearrange(rendered_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                # else:
                #     rendered_mask = torch.ones_like(frame['depth_mask'],dtype=torch.bool,device=device)
                # # Only add points that haven't been rendered
                
                new_mask = frame['depth_mask'] # & (rendered_mask)
                filtered_points_3d = frame['points_3d'][new_mask]
                
                # Compute now_scale for points corresponding to current frame
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=device) * 0.01
                    
                    # Use current frame's camera to compute now_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        frame['camera'],
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=device)
                
                print(frame['colors'][new_mask].shape)
                points_3d_l.append(filtered_points_3d)
                colors_l.append(frame['colors'][new_mask])
                radius_l.append(frame['radius'][new_mask[None]][None])
                normals_l.append(frame['normals'][new_mask])
                focal_length_l.append(torch.tensor([frame['focal_length']]*frame['colors'][new_mask].shape[0], device=device))
                is_sky_l.append(frame['is_sky'][new_mask])
                now_scale_l.append(now_scale_batch)
                
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=device)
            
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l

            imgs_train = [imgs[idx] for idx in idxs]
            cameras_train = [cameras[idx] for idx in idxs]
            self.cameras_train = cameras_train
            self.imgs_train = imgs_train
            
            # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cameras[idxs[1]], xyz_scale=xyz_scale, config=self.config)
            # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
            # visible_mask = render_pkg["visibility_filter"]
            # gaussians.delete_mask_all |= visible_mask
            # gaussians.delete_all_points(gaussians.delete_mask_all)
        return self.points_3d, self.colors, self.radius, self.normals, imgs_train, cameras_train, self.focal_length, self.is_sky, self.now_scale        
      
    @torch.no_grad()
    def process_zoomin_frames_overwrite(
        self,
        roots,
        cameras,
        gaussians, 
        xyz_scale,
        opt, 
        total_num=49,
        use_gaussian_as_bg=False,
        background_hard_depth=1.0,
        BG_COLOR=[0.7, 0.7, 0.7],
        no_grad_mask=True,
    ):
        device = self.device 
        num = 3
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            bg_mask = None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            now_scale_l = []
            points = rearrange(points_xy, "h w c -> (h w) c")
            
            # Maintain the original order for depth estimation
            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            # Store point cloud data for each frame
            frame_data = []
            
            # First perform depth estimation in 0->1->2 order
            for idx in idxs:
                if idx != 0:
                    scaling = True 
                    print(idx)
                else:
                    scaling = False
                    
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                with torch.no_grad():
                    if len(points_3d_l) > 0:
                        points_3d_now = torch.cat(points_3d_l, dim=0)
                        colors_now = torch.cat(colors_l, dim=0)
                        radius_now = torch.cat(radius_l, dim=-1)   
                        raster_settings = PointsRasterizationSettings(
                            image_size=[self.config["orig_H"], self.config["orig_W"]],
                            radius = radius_now.to(device),
                            points_per_pixel = 8,
                            bin_size = 0
                        )

                        renderer = PointsRenderer(
                            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                        )
                        point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                        # import pdb; pdb.set_trace()
                        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                        zbuf = zbuf.cpu().squeeze().cpu()[...,0]
                        bg_mask = bg_mask.squeeze() 
                        bg_mask = rearrange(bg_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                with torch.no_grad():
                    if True:
                        print("Using Gaussians for zoom-in depth estimation")
                        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                        render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, render_normals=True, config=self.config)
                        bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.8
                        bg_mask_gaussian=rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                        if bg_mask is not None:
                            bg_mask = bg_mask & bg_mask_gaussian
                        else:
                            bg_mask = bg_mask_gaussian
                        depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                        # depth_gau = render_pkg["depth"][0:1].squeeze().detach().cpu() / xyz_scale
                        
                    # else:
                    #     depth = self.moge.model.infer(img.to(device))['depth'].detach().cpu()
                    #     inf_mask = (depth == torch.inf)
                    #     depth[inf_mask] = 0
                    #     mask_align = ((zbuf<5.) & (zbuf>0) & (~inf_mask)).float()
                    #     zbuf = zbuf.clip(0., 1e9)
                    #     a, b = compute_scale_and_shift_full(depth, zbuf, ((zbuf<5.) & (zbuf>0) & (~inf_mask)).float())
                    #     mask_align = mask_align.bool()
                    #     with torch.enable_grad():
                    #         if scaling:
                    #             self.moge.model.train()
                    #             self.moge.model.requires_grad_(True)
                    #             best_depth = self.finetune_depth_model(self.moge.model, zbuf, img, a, b, mask_align)
                    #             self.moge.model.requires_grad_(False)
                    #             self.moge.model.eval()
                    
                    # if best_depth is not None:                
                    #     depth = best_depth.detach().cpu()                
                    #     best_depth = None   
                    # else:     
                    depth = depth_gau 
                    
                    inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()
                    is_sky = inf_mask.cpu()            

                    # if scaling:
                    #     self.moge.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             
                    #     self.moge = self.moge.to(device).eval()   
                        
                    # if zbuf is not None and scaling:
                    #     print("refining depth")
                    #     segments = self.get_sam_masks(img, min_mask_area=1)
                    #     for segment in segments:
                    #         sam_mask = torch.tensor(segment['segmentation'])
                    #         sam_mask_zbuf = sam_mask & mask_align 
                    #         depths_now = depth[sam_mask_zbuf] 
                    #         depths_now_zbuf = zbuf.squeeze()[sam_mask_zbuf]
                    #         diff = depths_now_zbuf.median() - depths_now.median() 
                    #         if torch.abs(diff) > 1e-5:
                    #             depth[sam_mask] += diff.to(depth.device)
                                
                    # depth[inf_mask] = 1e2
                    # if idx==0:
                    # depth = depth_gau
                    # depth_np  = depth_gau.cpu().numpy().astype(np.float32)
                    # mask_np   = (depth_np <= 0)

                    # Method 1: OpenCV Navier-Stokes / Telea (fast)
                    # depth_inpaint = cv2.inpaint(depth_np, mask_np.astype(np.uint8), 3, cv2.INPAINT_NS)

                    # Method 2: skimage biharmonic (slow, but smooth)
                    # from skimage.restoration import inpaint_biharmonic
                    # depth_inpaint = inpaint_biharmonic(depth_np, mask_np)

                    # depth_filled = torch.from_numpy(depth_inpaint).to(depth_gau.device)
                    depth_gau_np = depth_gau.squeeze().cpu().numpy().astype(np.float32)
                    
                    # Create mask for holes (where opacity is low - these are the holes!)
                    opacity_mask = render_pkg["depth"].detach().cpu()[0] < 0.
                    hole_mask = opacity_mask.squeeze().numpy().astype(bool)
                    
                    # Fill holes if there are any
                    if hole_mask.sum() > 0:
                        depth_filled = inpaint_nearest_bilateral_preserve(depth_gau_np, hole_mask, bilateral=True)
                        depth = torch.tensor(depth_filled, dtype=torch.float32)
                        print(f"✅ Filled {hole_mask.sum()} hole pixels using nearest neighbor + bilateral filter")
                    else:
                        depth = depth_gau

                    # depth = depth_filled 
                    # Use normals from gaussian render, already in world coordinate system!
                    if "render_normal" in render_pkg and render_pkg["render_normal"] is not None:
                        normals_gau = render_pkg["render_normal"]  # [3, H, W] world-space normals
                        normals = normals_gau  # Use directly, no coordinate conversion needed
                        print("using gaussian normals")
                    else:
                        # Fallback: use the original method
                        normals = self.get_normal(img[None])
                        normals[:, 1:] *= -1
                        # Convert to world coordinate system
                        current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                        normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                        normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                    
                    print("min_depth",depth.min())
                    
                    depth_mask = (depth >1e-3)
                    depth_mask = rearrange(depth_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                # Normals are already in world coordinate system, just reshape
                new_normals = rearrange(normals.squeeze()[None], "b c h w -> (w h b) c")
                
                # Need current_camera for unproject
                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(device)
                
                depth_moge = self.moge.model.infer(img.to(device))['depth'].detach().cpu()
                depth_moge = depth_moge / depth_moge.median() /100. + 1e-4
                
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth_moge.squeeze(),5)
                grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                if no_grad_mask:
                    grad_magnitude_mask[:] = False
                depth_mask = depth_mask & (~grad_magnitude_mask)
               
                # Store all point cloud data for current frame
                frame_data.append({
                    'points_3d': points_3d,
                    'colors': colors,
                    'radius': radius,
                    'normals': new_normals,
                    'depth_mask': depth_mask,
                    'camera': camera,
                    'focal_length': camera.K[0,0,0],
                    'is_sky': is_sky
                })

                # Handle gaussians deletion
    # Handle gaussians deletion
                if True:
                    # Combine depth_mask and bg_mask, do not overwrite bg_mask
                    if idx > 0:
                        bg_mask[:] = depth_mask  #& bg_mask
                    else:
                        bg_mask[:] = depth_mask #& bg_mask

                # Maintain lists like points_3d_l during the depth estimation stage 
                # Use the combined bg_mask to filter point cloud (exclude object region)
                filtered_points_3d = points_3d[bg_mask]
                
                # Compute now_scale for points corresponding to current camera
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=device) * 0.01
                    
                    # Use current camera to compute now_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=device)
                
                points_3d_l.append(filtered_points_3d)
                colors_l.append(colors[bg_mask])
                radius_l.append(radius[bg_mask[None]][None])
                normals_l.append(new_normals[bg_mask])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[bg_mask].shape[0], device=device))
                now_scale_l.append(now_scale_batch)

            # Add point clouds in 2->1->0 order
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            now_scale_l = []
            is_sky_l = []
            
            # Reverse iterate frame_data (2->1->0)
            
            for i in range(len(frame_data)-1, -1, -1):
                if i == 0:
                    continue
                frame = frame_data[i]
                
                # If this is the first frame (idx=2), add all points directly
                if i == len(frame_data)-1 or i == 0:
                    filtered_points_3d = frame['points_3d'][frame['depth_mask']]
                    
                    # Compute now_scale for points corresponding to current frame
                    if len(filtered_points_3d) > 0:
                        # Create default rotations and scales for the current batch of point cloud
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        # Use current frame's camera to compute now_scale
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            frame['camera'],
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                            min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    print(frame['points_3d'][frame['depth_mask']].shape)
                    points_3d_l.append(filtered_points_3d)
                    colors_l.append(frame['colors'][frame['depth_mask']])
                    radius_l.append(frame['radius'][frame['depth_mask'][None]][None])
                    normals_l.append(frame['normals'][frame['depth_mask']])
                    focal_length_l.append(torch.tensor([frame['focal_length']]*frame['colors'][frame['depth_mask']].shape[0], device=device))
                    is_sky_l.append(frame['is_sky'][frame['depth_mask']])
                    now_scale_l.append(now_scale_batch)
                    continue
                    
                # For other frames, check if already rendered by previous frames
                points_3d_now = torch.cat(points_3d_l, dim=0)
                colors_now = torch.cat(colors_l, dim=0)
                radius_now = torch.cat(radius_l, dim=-1)
                
                # raster_settings = PointsRasterizationSettings(
                #     image_size=[self.config["orig_H"], self.config["orig_W"]],
                #     radius = radius_now.to(device),
                #     points_per_pixel = 8,
                # )
                # renderer = PointsRenderer(
                #     rasterizer=PointsRasterizer(cameras=frame['camera'], raster_settings=raster_settings),
                #     compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                # )
                # point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                # _, _, rendered_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                # rendered_mask = rearrange(rendered_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                
                # Only add points that haven't been rendered
                new_mask = frame['depth_mask'] #& (rendered_mask)
                filtered_points_3d = frame['points_3d'][new_mask]
                
                # Compute now_scale for points corresponding to current frame
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=device) * 0.01
                    
                    # Use current frame's camera to compute now_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        frame['camera'],
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=device)
                
                points_3d_l.append(filtered_points_3d)
                colors_l.append(frame['colors'][new_mask])
                radius_l.append(frame['radius'][new_mask[None]][None])
                normals_l.append(frame['normals'][new_mask])
                focal_length_l.append(torch.tensor([frame['focal_length']]*frame['colors'][new_mask].shape[0], device=device))
                is_sky_l.append(frame['is_sky'][new_mask])
                now_scale_l.append(now_scale_batch)
                
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=device)
            
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l

            self.frame_data = frame_data
            imgs_train = [imgs[idx] for idx in idxs]
            cameras_train = [cameras[idx] for idx in idxs]
            self.cameras_train = cameras_train
            self.imgs_train = imgs_train
            
            if True:
                # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cameras[idxs[1]], xyz_scale=xyz_scale, config=self.config)
                # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                # visible_mask = render_pkg["visibility_filter"]
                visible_mask = gaussians.get_inscreen_points(cameras[idxs[1]])
                # gaussians.delete_mask_all |= visible_mask
                # gaussians.delete_all_points(gaussians.delete_mask_all)
                gaussians.merge_all_to_trainable()
                label_mask = gaussians.point_labels 
                label_mask[visible_mask&(label_mask == 0)] = int(1e5)
                gaussians.point_labels = label_mask
            
        # If object_label_to_filter is specified, re-estimate depth for the object region
        # pakage = None
        # if object_label_to_filter is not None and detected_camera_idx is not None:
        #     print(f"🔍 Re-estimating depth for object '{object_label_to_filter}' area")
            
        #     # Use the provided detected_camera_idx
        #     detected_camera = cameras[detected_camera_idx]
        #     detected_img_path = roots[detected_camera_idx]
            
        #     # Get the object's screen mask
        #     tdgs_cam_detected = convert_pt3d_cam_to_3dgs_cam(detected_camera, xyz_scale=xyz_scale, config=self.config)
        #     object_mask = gaussians.get_label_mask(object_label_to_filter)
            
        #     if object_mask.any():
        #         # Use deepcopy to copy original gaussians, then delete non-object points
        #         import copy
        #         gaussians_temp = copy.deepcopy(gaussians)
                
        #         # Create deletion mask: delete all non-object points
        #         non_object_mask = ~object_mask
                
        #         print(f"Re-estimation: original point count: {gaussians_temp.get_xyz_all.shape[0]}")
        #         print(f"Re-estimation: object point count: {object_mask.sum()}")
        #         print(f"Re-estimation: points to delete: {non_object_mask.sum()}")
                
        #         # Delete non-object points
        #         gaussians_temp.delete_mask_all = non_object_mask
        #         gaussians_temp.prune_points(non_object_mask)
                
        #         print(f"Re-estimation: remaining points after deletion: {gaussians_temp.get_xyz_all.shape[0]}")
                
        #         # Render to get the object screen mask
        #         render_pkg_obj = render(tdgs_cam_detected, gaussians_temp, opt, torch.tensor([0.7, 0.7, 0.7]).to(device), render_visible=True)
        #         object_opacity = render_pkg_obj["final_opacity"].detach().cpu()[0]
        #         object_screen_mask = (object_opacity > 0.6).squeeze()  # H,W format
                
        #         # Get median_depth
        #         render_pkg_full = render(tdgs_cam_detected, gaussians, opt, torch.tensor([0.7, 0.7, 0.7]).to(device), render_visible=True)
        #         median_depth = render_pkg_full['median_depth'][0]/xyz_scale
                
        #         # Use process_single_img_mask to re-estimate the object region
        #         print(f"📍 Object occupies {object_screen_mask.sum().item()} pixels, re-estimating depth")
        #         obj_points_3d, obj_colors, obj_radius, obj_normals, obj_imgs, obj_cameras, obj_focal, obj_is_sky = self.process_single_img_mask(
        #             [detected_img_path], 
        #             object_screen_mask,
        #             [detected_camera], 
        #             median_depth, 
        #             num=1, 
        #             total_num=1
        #         )
                
        #         # Merge the new point cloud for the object region
                
        #         if obj_points_3d is not None and len(obj_points_3d) > 0:
        #             print(f"🔗 Adding {len(obj_points_3d)} object points to {len(points_3d3)} existing points")
                    
        #             # Mark these as object point cloud for later labeling (record before cat)
        #             self.object_points_start_idx = len(points_3d3)  # Record the starting position of object points
        #             self.object_label_name = object_label_to_filter
        #             self.has_object_points = True
                    
        #             pakage = [obj_points_3d.detach().cpu() , obj_colors.detach().cpu(), obj_radius.detach().cpu(), obj_normals.detach().cpu(), obj_imgs, obj_cameras, obj_focal.detach().cpu(), obj_is_sky.detach().cpu()]
        #             self.points_3d = torch.cat([points_3d3 ], dim=0)
        #             self.colors = torch.cat([colors3], dim=0)
        #             self.radius = torch.cat([radius3], dim=-1)
        #             self.normals = torch.cat([normals3], dim=0)
        #             self.focal_length = torch.cat([focal_length3], dim=0)
        #             self.is_sky = torch.cat([is_sky3], dim=0)
                    
        #             print(f"✅ Object depth re-estimation complete, marked {len(obj_points_3d)} points for label '{object_label_to_filter}'")
        #         else:
        #             print("⚠️ No object points generated")
        #     else:
        #         print(f"⚠️ No points found for object '{object_label_to_filter}'")
            
            return self.points_3d, self.colors, self.radius, self.normals, imgs_train, cameras_train, self.focal_length, self.is_sky, self.now_scale        
    
    @torch.no_grad()
    def process_zoomin_frames_overwrite_obj_mask(
        self,
        roots,
        cameras,
        gaussians, 
        xyz_scale,
        opt, 
        loss_masks,
        total_num=3,
        background_hard_depth=1.0,
        BG_COLOR=[0.7, 0.7, 0.7],
        no_grad_mask=True,
    ):
        device = self.device 
        num = 3
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            bg_mask = None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            now_scale_l = []
            points = rearrange(points_xy, "h w c -> (h w) c")
            
            # Maintain the original order for depth estimation
            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            # Store point cloud data for each frame
            frame_data = []
            
            # First perform depth estimation in 0->1->2 order
            for idx in idxs:
                if idx != 0:
                    scaling = True 
                    print(idx)
                else:
                    scaling = False
                    
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                loss_mask = loss_masks[idx]
                
                with torch.no_grad():
                    if len(points_3d_l) > 0:
                        points_3d_now = torch.cat(points_3d_l, dim=0)
                        colors_now = torch.cat(colors_l, dim=0)
                        radius_now = torch.cat(radius_l, dim=-1)   
                        raster_settings = PointsRasterizationSettings(
                            image_size=[self.config["orig_H"], self.config["orig_W"]],
                            radius = radius_now.to(device),
                            points_per_pixel = 8,
                            bin_size = 0
                        )

                        renderer = PointsRenderer(
                            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                        )
                        point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                        # import pdb; pdb.set_trace()
                        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                        zbuf = zbuf.cpu().squeeze().cpu()[...,0]
                        bg_mask = bg_mask.squeeze() 
                        bg_mask = rearrange(bg_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                with torch.no_grad():
                    if True:
                        print("Using Gaussians for zoom-in depth estimation")
                        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                        render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, render_normals=True, config=self.config)
                        bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.8
                        bg_mask_gaussian=rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                        if bg_mask is not None:
                            bg_mask = bg_mask & bg_mask_gaussian
                        else:
                            bg_mask = bg_mask_gaussian
                        depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                        # depth_gau = render_pkg["depth"][0:1].squeeze().detach().cpu() / xyz_scale
                        

                    depth = depth_gau 
                    
                    inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()
                    is_sky = inf_mask.cpu()            

                    depth_gau_np = depth_gau.squeeze().cpu().numpy().astype(np.float32)
                    
                    # Create mask for holes (where opacity is low - these are the holes!)
                    opacity_mask = render_pkg["depth"].detach().cpu()[0] < 0.
                    hole_mask = opacity_mask.squeeze().numpy().astype(bool)
                    
                    # Fill holes if there are any
                    if hole_mask.sum() > 0:
                        depth_filled = inpaint_nearest_bilateral_preserve(depth_gau_np, hole_mask, bilateral=True)
                        depth = torch.tensor(depth_filled, dtype=torch.float32)
                        print(f"✅ Filled {hole_mask.sum()} hole pixels using nearest neighbor + bilateral filter")
                    else:
                        depth = depth_gau

                    # depth = depth_filled 
                    # Use normals from gaussian render, already in world coordinate system!
                    if "render_normal" in render_pkg and render_pkg["render_normal"] is not None:
                        normals_gau = render_pkg["render_normal"]  # [3, H, W] world-space normals
                        normals = normals_gau  # Use directly, no coordinate conversion needed
                        print("using gaussian normals")
                    else:
                        # Fallback: use the original method
                        normals = self.get_normal(img[None])
                        normals[:, 1:] *= -1
                        # Convert to world coordinate system
                        current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                        normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                        normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                    
                    print("min_depth",depth.min())
                    
                    depth_mask = (depth >1e-3) & (loss_mask.squeeze().cpu())
                    depth_mask = rearrange(depth_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                # Normals are already in world coordinate system, just reshape
                new_normals = rearrange(normals.squeeze()[None], "b c h w -> (w h b) c")
                
                # Need current_camera for unproject
                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(device)
                
                depth_moge = self.moge.model.infer(img.to(device))['depth'].detach().cpu()
                depth_moge = depth_moge / depth_moge.median() /100. + 1e-4
                
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth_moge.squeeze(),0.7)
                grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                if no_grad_mask:
                    grad_magnitude_mask[:] = False
                depth_mask = depth_mask & (~grad_magnitude_mask)
               
                # Store all point cloud data for current frame
                frame_data.append({
                    'points_3d': points_3d,
                    'colors': colors,
                    'radius': radius,
                    'normals': new_normals,
                    'depth_mask': depth_mask,
                    'camera': camera,
                    'focal_length': camera.K[0,0,0],
                    'is_sky': is_sky
                })

                # Handle gaussians deletion
                if True:
                    # Combine depth_mask and bg_mask, do not overwrite bg_mask
                    if idx > 0:
                        bg_mask[:] = depth_mask  #& bg_mask
                    else:
                        bg_mask[:] = depth_mask #& bg_mask

                # Maintain lists like points_3d_l during the depth estimation stage 
                # Use the combined bg_mask to filter point cloud (exclude object region)
                filtered_points_3d = points_3d[bg_mask]
                
                # Compute now_scale for points corresponding to current camera
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=device) * 0.01
                    
                    # Use current camera to compute now_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=device)
                
                points_3d_l.append(filtered_points_3d)
                colors_l.append(colors[bg_mask])
                radius_l.append(radius[bg_mask[None]][None])
                normals_l.append(new_normals[bg_mask])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[bg_mask].shape[0], device=device))
                now_scale_l.append(now_scale_batch)

            # Add point clouds in 2->1->0 order
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            now_scale_l = []
            is_sky_l = []
            
            # Reverse iterate frame_data (2->1->0)
            
            for i in range(len(frame_data)-1, -1, -1):
                if i == 0:
                    continue
                frame = frame_data[i]
                
                # If this is the first frame (idx=2), add all points directly
                if i == len(frame_data)-1 or i == 0:
                    filtered_points_3d = frame['points_3d'][frame['depth_mask']]
                    
                    # Compute now_scale for points corresponding to current frame
                    if len(filtered_points_3d) > 0:
                        # Create default rotations and scales for the current batch of point cloud
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        # Use current frame's camera to compute now_scale
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            frame['camera'],
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                            min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    print(frame['points_3d'][frame['depth_mask']].shape)
                    points_3d_l.append(filtered_points_3d)
                    colors_l.append(frame['colors'][frame['depth_mask']])
                    radius_l.append(frame['radius'][frame['depth_mask'][None]][None])
                    normals_l.append(frame['normals'][frame['depth_mask']])
                    focal_length_l.append(torch.tensor([frame['focal_length']]*frame['colors'][frame['depth_mask']].shape[0], device=device))
                    is_sky_l.append(frame['is_sky'][frame['depth_mask']])
                    now_scale_l.append(now_scale_batch)
                    continue
                    
                # For other frames, check if already rendered by previous frames
                points_3d_now = torch.cat(points_3d_l, dim=0)
                colors_now = torch.cat(colors_l, dim=0)
                radius_now = torch.cat(radius_l, dim=-1)
                
                # raster_settings = PointsRasterizationSettings(
                #     image_size=[self.config["orig_H"], self.config["orig_W"]],
                #     radius = radius_now.to(device),
                #     points_per_pixel = 8,
                # )
                # renderer = PointsRenderer(
                #     rasterizer=PointsRasterizer(cameras=frame['camera'], raster_settings=raster_settings),
                #     compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                # )
                # point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                # _, _, rendered_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                # rendered_mask = rearrange(rendered_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                
                # Only add points that haven't been rendered
                new_mask = frame['depth_mask'] #& (rendered_mask)
                filtered_points_3d = frame['points_3d'][new_mask]
                
                # Compute now_scale for points corresponding to current frame
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=device) * 0.01
                    
                    # Use current frame's camera to compute now_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        frame['camera'],
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=device)
                
                points_3d_l.append(filtered_points_3d)
                colors_l.append(frame['colors'][new_mask])
                radius_l.append(frame['radius'][new_mask[None]][None])
                normals_l.append(frame['normals'][new_mask])
                focal_length_l.append(torch.tensor([frame['focal_length']]*frame['colors'][new_mask].shape[0], device=device))
                is_sky_l.append(frame['is_sky'][new_mask])
                now_scale_l.append(now_scale_batch)
                
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=device)
            
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l

            self.frame_data = frame_data
            imgs_train = [imgs[idx] for idx in idxs]
            cameras_train = [cameras[idx] for idx in idxs]
            self.cameras_train = cameras_train
            self.imgs_train = imgs_train
            
            if True:
                # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cameras[idxs[1]], xyz_scale=xyz_scale, config=self.config)
                # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                # visible_mask = render_pkg["visibility_filter"]
                visible_mask = gaussians.get_inscreen_points(cameras[idxs[1]])
                # gaussians.delete_mask_all |= visible_mask
                # gaussians.delete_all_points(gaussians.delete_mask_all)
                gaussians.merge_all_to_trainable()
                label_mask = gaussians.point_labels 
                label_mask[visible_mask&(label_mask == 0)] = int(1e5)
                gaussians.point_labels = label_mask
            
            
            return self.points_3d, self.colors, self.radius, self.normals, imgs_train, cameras_train, self.focal_length, self.is_sky, self.now_scale        
    
    
    
    @torch.no_grad()
    def process_video_frames(
        self,
        roots,
        cameras,
        gaussians, 
        xyz_scale,
        opt, 
        num=49,
        total_num=49,
        dep_thra=9e9,
        use_gaussian_as_bg=True,
        background_hard_depth=1.0,
        BG_COLOR=[0.7, 0.7, 0.7],
        revert = True,
        open_align = False,
        return_1 = False ,
        no_grad_mask = True,
        jump_bg_mask = True,
    ):
        """Process video frames to generate point cloud.
        
        Args:
            roots: List of image paths
            cameras: List of camera parameters
            device: Device to run on
            num: Number of frames to process
            total_num: Total number of frames
            rev: Whether to process frames in reverse
            use_video_depth: Whether to use video depth estimation
            zoom_in: Whether to apply depth finetuning and gathering
            dep_thra: Depth threshold
            background_hard_depth: Background depth normalization factor
            BG_COLOR: Background color for rendering
            
        Returns:
            tuple: (points_3d, colors, radius, normals) containing the generated point cloud data
        """
        device = self.device 
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            bg_mask = None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depths = []
            depth_align_masks = []
            used_bg_masks = []
            points = rearrange(points_xy, "h w c -> (h w) c")
            # flag = -1 if rev else 1
            
            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
            idxs_original = copy.deepcopy(idxs) 
                
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"], directly_resize=True) for root in roots_now], dim=0)
            
            if revert:
                t=idxs[1]
                idxs[1]=idxs[-1]
                idxs[-1]=t 
               
            for idx in idxs:

                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                
                with torch.no_grad():
                    if len(points_3d_l) > 0:
                        points_3d_now = torch.cat(points_3d_l, dim=0)
                        colors_now = torch.cat(colors_l, dim=0)
                        radius_now = torch.cat(radius_l, dim=-1)   
                        # points_3d_now[..., :2] = - points_3d_now[..., :2]
                        raster_settings = PointsRasterizationSettings(
                            image_size=[self.config["orig_H"], self.config["orig_W"]],
                            radius = radius_now.to(device),
                            points_per_pixel = 8,
                        )

                        renderer = PointsRenderer(
                            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                        )
                        point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                        zbuf = zbuf.cpu().squeeze().cpu()[...,0]
                        # points_3d_now[..., :2] = - points_3d_now[..., :2]
                        bg_mask = bg_mask.squeeze() 
                        bg_mask = rearrange(bg_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                        # print(idx,"bg_mask",bg_mask.float().sum())
                    
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                    # image_s, viewspace_point_tensor, visibility_filter, radii = (
                    #     render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
                    bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.6
                    bg_mask_gaussian = rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    # print(idx,"bg_mask_gaussian",bg_mask_gaussian.float().sum())
                    if bg_mask is not None:
                        if use_gaussian_as_bg:
                            bg_mask = bg_mask & bg_mask_gaussian
                    else:
                        bg_mask = bg_mask_gaussian
                    if bg_mask.float().sum()<720*480*0.01 and idx!=0 and jump_bg_mask:
                        continue 
                    
                    depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale


                with torch.no_grad():
                    if idx == 0:
                        print("Using Gaussians for video depth estimation")
                        # camera_final = cameras[idxs_original[0]]  # Use the first frame's camera
                        # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera_final, xyz_scale=xyz_scale, config=self.config)
                        # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                        # depth_gau_final = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                        
                        # Use VGGT for depth estimation
                        # print("Using VGGT for video depth estimation")
                        # # Convert images to the format required by VGGT
                        # # imgs_vggt = imgs[:].permute(0, 2, 3, 1)  # (N, H, W, 3)
                        
                        # # Run VGGT inference
                        # with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16):
                        #     vggt_predictions = self.vggt.infer(imgs.to(self.device))  # Convert back to (N, 3, H, W)
                        
                        # # Extract depth and confidence
                        # depth_video = torch.nn.functional.interpolate(vggt_predictions["depth"].detach().squeeze()[:,None], size=(self.config["orig_H"], self.config["orig_W"]), mode='bilinear').cpu().squeeze()  # (N, H, W, 1)
                        # confidence_video = torch.nn.functional.interpolate(vggt_predictions["depth_conf"].detach().squeeze()[:,None], size=(self.config["orig_H"], self.config["orig_W"]), mode='bilinear').cpu().squeeze()  # (N, H, W)
                        
                        # # Process confidence filtering following demo_viser's approach
                        # # Flatten confidence from all frames, compute the overall 75% threshold
                        # confidence_flat = confidence_video.reshape(-1)  # Flatten confidence from all frames
                        # k = int(0.2 * confidence_flat.numel())
                        # confi_threshold = torch.kthvalue(confidence_flat.flatten(), k).values # Compute confidence threshold

                        # # Create valid_video mask
                        # confidence_mask_all = confidence_video >= confi_threshold
                        # valid_video = (confidence_mask_all) & (confidence_video > 1e-5)
                        
                        # # Process depth data
                        # depth_video = depth_video.squeeze(-1)  # Remove the last dimension, becomes (N, H, W)
                        # depth_video[depth_video == torch.inf] = 1e6
                        # depth_video[~valid_video] = 1e6
                        
                        # # Align depth - using the first frame
                        # mask_align = ((depth_gau_final > 1e-4) & (depth_gau_final < 0.8) & (depth_video[0] < 50) & valid_video[0]).float()
                        # a, b = compute_scale_and_shift_full(depth_video.squeeze()[0], depth_gau_final.squeeze(), mask_align)
                        # print(depth_video.shape, depth_video[0][mask_align.bool()].max())
                        # depth_video = a * depth_video + b
                        # print("video", a, b)
                        # confidence_mask_all = rearrange(confidence_mask_all.squeeze(), "b h w -> b (h w)").detach().cpu().squeeze()
                        # depthu = self.moge.model.infer(, img[None].to(device))['depth'].detach().cpu()
                        # print("Using Gaussians for video depth estimation")
                        # camera_final = cameras[idxs_original[-1]]
                        # # camera_final = cameras[-1]
                        # tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera_final, xyz_scale=xyz_scale, config=self.config)
                        # render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                        # bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.6
                        # bg_mask_gaussian=rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                        # depth_gau_final = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                        
                        depth_video, valid_video = inf_geometry(self.moge, self.pipe, self.point_map_vae, 
                                                                imgs[:].permute(0,2,3,1).numpy(),
                                                                height=576, width=1024)
                        depth_video, valid_video = depth_video.cpu(), valid_video.cpu()
                        mask_align = ((depth_gau>1e-4) & (depth_gau<0.8) & (depth_video[0]<50) & valid_video[0]).float()
                        depth_video[depth_video==torch.inf] = 1e6
                        depth_video[~valid_video] = 1e6
                        a, b = compute_scale_and_shift_full(depth_video.squeeze()[0], depth_gau.squeeze(), mask_align)
                        print(depth_video.shape, depth_video[0][mask_align.bool()].max())
                        depth_video = a * depth_video + b 
                        # import pdb; pdb.set_trace()
                        print("video", a, b)

                            
                        
                    # depth = self.moge.model.infer( img.to(device))['depth'].detach().cpu()
                    # inf_mask = (depth == torch.inf)
                    # depth[inf_mask] = 0
                                
                    # inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()
                    inf_mask =  self.generate_sky_mask(img).cpu()            
                    depth = depth_video[idx]
                    is_sky = inf_mask
                    # if zbuf is not None and open_align:
                    #     print("refining depth")
                    #     mask_align = ((zbuf<5.) & (zbuf>0) & (~inf_mask))
                    #     segments = self.get_sam_masks(img, min_mask_area=1)
                    #     for segment in segments:
                    #         sam_mask = torch.tensor(segment['segmentation'])
                    #         sam_mask_zbuf = sam_mask & mask_align 
                    #         depths_now = depth[sam_mask_zbuf] 
                    #         depths_now_zbuf = zbuf.squeeze()[sam_mask_zbuf]
                    #         diff = depths_now_zbuf.median() - depths_now.median() 
                    #         if torch.abs(diff) > 1e-5:
                    #             depth[sam_mask] += diff.to(depth.device)
                    depth[inf_mask] = 1e2
                    
                    grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(),20)
                    grad_magnitude_mask_hw = ~grad_magnitude_mask.squeeze()
                    grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    if no_grad_mask:
                        grad_magnitude_mask[:] = False
                    normals = self.get_normal(img[None])
                    normals[:, 1:] *= -1
                    
                    # depth_mask = (depth <= dep_thra) & (depth)
                    depth_mask = depth > 1e-3
                    depth_align_mask = grad_magnitude_mask_hw & depth_mask & (~is_sky)
                    depth_mask = rearrange(depth_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(device)
                
                # if idx == idxs[1]:
                #     import pdb; pdb.set_trace() 
                
                if len(points_3d_l) > 0:
    
                    # bg_mask = bg_mask & confidence_mask_all[idx] & depth_mask
                    bg_mask = bg_mask & depth_mask & (~grad_magnitude_mask)
                    if not jump_bg_mask:
                        bg_mask = bg_mask_gaussian.clone()
                    all_depths.append(depth.squeeze().detach().cpu())
                    depth_align_masks.append(depth_align_mask.squeeze().detach().cpu())
                    print(colors[bg_mask].shape)
                    
                    # Compute now_scale for points corresponding to current camera
                    filtered_points_3d = points_3d[bg_mask]
                    if len(filtered_points_3d) > 0:
                        # Create default rotations and scales for the current batch of point cloud
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        # Use current camera to compute now_scale
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            camera,
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                            min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    used_bg_masks.append(bg_mask)
                    colors_l.append(colors[bg_mask])
                    points_3d_l.append(filtered_points_3d)
                    radius_l.append(radius[bg_mask[None]][None])
                    normals_l.append(new_normals[bg_mask])
                    focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[bg_mask].shape[0], device=device))
                    is_sky_l.append(is_sky[bg_mask])
                    now_scale_l.append(now_scale_batch)
                else:
                    # depth_mask = depth_mask & confidence_mask_all[idx]
                    all_depths.append(depth.squeeze().detach().cpu())
                    depth_align_masks.append(depth_align_mask.squeeze().detach().cpu())
                    if not jump_bg_mask:
                        bg_mask = bg_mask_gaussian.clone()
                    # Compute now_scale for points corresponding to current camera
                    filtered_points_3d = points_3d[depth_mask]
                    if len(filtered_points_3d) > 0:
                        # Create default rotations and scales for the current batch of point cloud
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        # Use current camera to compute now_scale
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            camera,
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                                min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    used_bg_masks.append(bg_mask)
                    colors_l.append(colors[depth_mask])
                    points_3d_l.append(filtered_points_3d)
                    radius_l.append(radius[depth_mask[None]][None])      
                    normals_l.append(new_normals[depth_mask])
                    focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[depth_mask].shape[0], device=device)    )
                    is_sky_l.append(is_sky[depth_mask])
                    now_scale_l.append(now_scale_batch)
            self.points_3d = torch.cat(points_3d_l[:], dim=0) if return_1 else torch.cat(points_3d_l[1:], dim=0)
            self.colors = torch.cat(colors_l[:], dim=0) if return_1 else torch.cat(colors_l[1:], dim=0)
            self.radius = torch.cat(radius_l[:], dim=-1) if return_1 else torch.cat(radius_l[1:], dim=-1) 
            self.normals = torch.cat(normals_l[:], dim=0) if return_1 else torch.cat(normals_l[1:], dim=0)
            self.focal_length = torch.cat(focal_length_l[:], dim=0) if return_1 else torch.cat(focal_length_l[1:], dim=0)
            self.is_sky = torch.cat(is_sky_l[:], dim=0) if return_1 else torch.cat(is_sky_l[1:], dim=0)
            self.now_scale = torch.cat(now_scale_l[:], dim=0) if (now_scale_l and return_1) else (torch.cat(now_scale_l[1:], dim=0) if (now_scale_l and len(now_scale_l) > 1) else torch.empty(0, device=device))
            
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.is_sky_l = is_sky_l
            self.imgs = imgs
            self.cameras = cameras
            depth_align_masks = []
            all_depths = []
            # for idx in range(depth_video.shape[0]):
            #     depth = depth_video[idx]
            #     grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze())
            #     grad_magnitude_mask_hw = ~grad_magnitude_mask.squeeze()
            #     depth_mask = depth > 1e-3
            #     is_sky = self.generate_sky_mask(imgs[idx]).cpu()
            #     depth_align_mask = grad_magnitude_mask_hw & depth_mask & (~is_sky)
            #     depth_align_masks.append(depth_align_mask.squeeze().detach().cpu())
            #     all_depths.append(depth.squeeze().detach().cpu())
            # num_points_cameras = [0] + [x.shape[0] for x in points_3d_l[1:]]
            
            # imgs_train = [imgs[idx] for idx in idxs_original[:]]
            # cameras_train = [cameras[idx] for idx in idxs_original[:]]
            self.depth_align_masks = depth_align_masks
            self.all_depths = all_depths
            imgs_train = imgs
            cameras_train = cameras
            self.imgs_train = imgs
            self.cameras_train = cameras
            self.used_bg_masks = used_bg_masks
            return self.points_3d, self.colors, self.radius, self.normals, imgs_train, cameras_train, self.focal_length, self.is_sky, all_depths, depth_align_masks, self.now_scale

    @torch.no_grad()
    def process_frames_inpainting(
        self,
        roots,
        cameras,
        gaussians, 
        xyz_scale,
        opt, 
        num=49,
        total_num=49,
        use_gaussian_as_bg=True,
        background_hard_depth=1.0,
        BG_COLOR=[0.7, 0.7, 0.7],
        revert = True,
        return_1 = False,
        no_grad_mask = True,
        jump_bg_mask = True,
    ):
        """Process video frames using inpainted depth from gaussians.
        
        Uses depth directly from gaussians rendering and fills holes with nearest neighbor + bilateral filtering.
        No depth estimation is performed, making it more stable for HQ mode.
        
        Args:
            roots: List of image paths
            cameras: List of camera parameters
            gaussians: Existing gaussian model to render depth from
            xyz_scale: Scale factor
            opt: Optimization parameters
            num: Number of frames to process
            total_num: Total number of frames
            use_gaussian_as_bg: Whether to use gaussian opacity for background mask
            background_hard_depth: Background depth normalization factor
            BG_COLOR: Background color for rendering
            revert: Whether to revert frame order
            return_1: Whether to return first frame
            no_grad_mask: Whether to disable gradient mask
            jump_bg_mask: Whether to enable bg mask jumping
            
        Returns:
            tuple: (points_3d, colors, radius, normals, imgs_train, cameras_train, focal_length, is_sky, all_depths, depth_align_masks, now_scale)
        """
        device = self.device 
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            bg_mask = None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depths = []
            depth_align_masks = []
            points = rearrange(points_xy, "h w c -> (h w) c")
            
            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
            idxs_original = copy.deepcopy(idxs) 
                
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"], directly_resize=True) for root in roots_now], dim=0)
            
            if revert:
                t=idxs[1]
                idxs[1]=idxs[-1]
                idxs[-1]=t 
               
            for idx in idxs:
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                with torch.no_grad():
                    # Render existing point cloud if available
                    if len(points_3d_l) > 0:
                        points_3d_now = torch.cat(points_3d_l, dim=0)
                        colors_now = torch.cat(colors_l, dim=0)
                        radius_now = torch.cat(radius_l, dim=-1)   
                        raster_settings = PointsRasterizationSettings(
                            image_size=[self.config["orig_H"], self.config["orig_W"]],
                            radius = radius_now.to(device),
                            points_per_pixel = 8,
                        )

                        renderer = PointsRenderer(
                            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                        )
                        point_cloud = Pointclouds(points=[points_3d_now.to(device)], features=[colors_now.to(device)])
                        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
                        zbuf = zbuf.cpu().squeeze().cpu()[...,0]
                        bg_mask = bg_mask.squeeze() 
                        bg_mask = rearrange(bg_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    
                    # Render gaussians to get depth
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
                    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
                    
                    # Get depth from gaussians
                    depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                    bg_mask_gaussian = render_pkg["final_opacity"].detach().cpu()[0]<0.6
                    bg_mask_gaussian = rearrange(bg_mask_gaussian.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    
                    if bg_mask is not None:
                        if use_gaussian_as_bg:
                            bg_mask = bg_mask & bg_mask_gaussian
                    else:
                        bg_mask = bg_mask_gaussian
                    
                    if jump_bg_mask and bg_mask.float().sum()<1 and idx!=0 and idx!=1:
                        continue 

                with torch.no_grad():
                    # Fill holes using nearest neighbor + bilateral filtering (ChatGPT's method)
                    
                    depth_gau_np = depth_gau.squeeze().cpu().numpy().astype(np.float32)
                    
                    # Create mask for holes (where opacity is low - these are the holes!)
                    opacity_mask = render_pkg["final_opacity"].detach().cpu()[0] < 0.6
                    hole_mask = opacity_mask.squeeze().numpy().astype(bool)
                    
                    # Fill holes if there are any
                    if hole_mask.sum() > 0:
                        depth_filled = inpaint_nearest_bilateral_preserve(depth_gau_np, hole_mask, bilateral=True)
                        depth = torch.tensor(depth_filled, dtype=torch.float32)
                        print(f"✅ Filled {hole_mask.sum()} hole pixels using nearest neighbor + bilateral filter")
                    else:
                        depth = depth_gau
                    
                    # Generate sky mask
                    inf_mask = self.generate_sky_mask(img).cpu()            
                    is_sky = inf_mask
                    depth[inf_mask] = 0.9
                    # Generate gradient magnitude mask
                    grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze())
                    grad_magnitude_mask_hw = ~grad_magnitude_mask.squeeze()
                    grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    if no_grad_mask:
                        grad_magnitude_mask[:] = False
                    
                    # Get normals
                    normals = self.get_normal(img[None])
                    normals[:, 1:] *= -1
                    
                    # Create depth masks
                    depth_mask = depth > 1e-3
                    depth_align_mask = grad_magnitude_mask_hw & depth_mask & (~is_sky)
                    depth_mask = rearrange(depth_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                    is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                # Convert to 3D points
                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                
                # Calculate radius
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(device)
  
                    
                if len(points_3d_l) > 0:
                    # Apply background mask
                    bg_mask = bg_mask & depth_mask & (~grad_magnitude_mask)
                    all_depths.append(depth.squeeze().detach().cpu())
                    depth_align_masks.append(depth_align_mask.squeeze().detach().cpu())
                    print(f"Frame {idx}: {colors[bg_mask].shape[0]} points added")
                    
                    # Calculate now_scale for current camera
                    filtered_points_3d = points_3d[bg_mask]
                    if len(filtered_points_3d) > 0:
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            camera,
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                            min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    colors_l.append(colors[bg_mask])
                    points_3d_l.append(filtered_points_3d)
                    radius_l.append(radius[bg_mask[None]][None])
                    normals_l.append(new_normals[bg_mask])
                    focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[bg_mask].shape[0], device=device))
                    is_sky_l.append(is_sky[bg_mask])
                    now_scale_l.append(now_scale_batch)
                else:
                    # First frame
                    all_depths.append(depth.squeeze().detach().cpu())
                    depth_align_masks.append(depth_align_mask.squeeze().detach().cpu())
                    print(f"Frame {idx} (first): {colors[depth_mask].shape[0]} points added")
                    
                    # Calculate now_scale for current camera
                    filtered_points_3d = points_3d[depth_mask]
                    if len(filtered_points_3d) > 0:
                        num_points = len(filtered_points_3d)
                        default_rotations = torch.zeros(num_points, 4, device=device)
                        default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                        default_scales = torch.ones(num_points, 3, device=device) * 0.01
                        
                        means3D_scaled = filtered_points_3d * xyz_scale
                        
                        q, s_target = compute_target_scale_per_frame(
                            camera,
                            means3D_scaled,
                            default_rotations,
                            default_scales,
                            rotation2normal_fn=rotation2normal,
                                min_cos=0.1,
                            config=self.config
                        )
                        now_scale_batch = s_target.detach()
                    else:
                        now_scale_batch = torch.empty(0, device=device)
                    
                    colors_l.append(colors[depth_mask])
                    points_3d_l.append(filtered_points_3d)
                    radius_l.append(radius[depth_mask[None]][None])      
                    normals_l.append(new_normals[depth_mask])
                    focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[depth_mask].shape[0], device=device))
                    is_sky_l.append(is_sky[depth_mask])
                    now_scale_l.append(now_scale_batch)
                    
            # Combine all points
            self.points_3d = torch.cat(points_3d_l[:], dim=0) if return_1 else torch.cat(points_3d_l[1:], dim=0)
            self.colors = torch.cat(colors_l[:], dim=0) if return_1 else torch.cat(colors_l[1:], dim=0)
            self.radius = torch.cat(radius_l[:], dim=-1) if return_1 else torch.cat(radius_l[1:], dim=-1) 
            self.normals = torch.cat(normals_l[:], dim=0) if return_1 else torch.cat(normals_l[1:], dim=0)
            self.focal_length = torch.cat(focal_length_l[:], dim=0) if return_1 else torch.cat(focal_length_l[1:], dim=0)
            self.is_sky = torch.cat(is_sky_l[:], dim=0) if return_1 else torch.cat(is_sky_l[1:], dim=0)
            self.now_scale = torch.cat(now_scale_l[:], dim=0) if (now_scale_l and return_1) else (torch.cat(now_scale_l[1:], dim=0) if (now_scale_l and len(now_scale_l) > 1) else torch.empty(0, device=device))
            
            # Store intermediate results
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.is_sky_l = is_sky_l
            self.imgs = imgs
            self.cameras = cameras
            
            # Final processing for depth align masks
            depth_align_masks = []
            all_depths = []
            # for idx in range(len(cameras)):
            #     camera = cameras[idx]
            #     tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=self.config)
            #     render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True)
            #     depth_gau = render_pkg["median_depth"][0:1].squeeze().detach().cpu() / xyz_scale
                
            #     # Apply nearest neighbor + bilateral filtering for holes
            #     depth_gau_np = depth_gau.cpu().numpy().astype(np.float32)
            #     opacity_mask = render_pkg["final_opacity"].detach().cpu()[0] < 0.6
            #     hole_mask = opacity_mask.squeeze().numpy().astype(bool)
            #     if hole_mask.sum() > 0:
            #         # Use the same inpainting function as in main loop
            #         from scipy.ndimage import distance_transform_edt
            #         import cv2
            #         dist, inds = distance_transform_edt(hole_mask, return_indices=True)
            #         nearest_y, nearest_x = inds
            #         filled = depth_gau_np[nearest_y, nearest_x]
            #         smooth = cv2.bilateralFilter(filled.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=3)
            #         depth_filled = depth_gau_np.copy()
            #         depth_filled[hole_mask] = smooth[hole_mask]
            #         depth = torch.tensor(depth_filled, dtype=torch.float32)
            #     else:
            #         depth = depth_gau
                
            #     grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze())
            #     grad_magnitude_mask_hw = ~grad_magnitude_mask.squeeze()
            #     depth_mask = depth > 1e-3
            #     is_sky = self.generate_sky_mask(imgs[idx]).cpu()
            #     depth_align_mask = grad_magnitude_mask_hw & depth_mask & (~is_sky)
            #     depth_align_masks.append(depth_align_mask.squeeze().detach().cpu())
            #     all_depths.append(depth.squeeze().detach().cpu())
            
            self.depth_align_masks = depth_align_masks
            self.all_depths = all_depths
            imgs_train = imgs
            cameras_train = cameras
            self.imgs_train = imgs
            self.cameras_train = cameras
            
            print(f"✅ Inpainting-based processing complete: {self.points_3d.shape[0] if self.points_3d.numel() > 0 else 0} total points generated")
            
            return self.points_3d, self.colors, self.radius, self.normals, imgs_train, cameras_train, self.focal_length, self.is_sky, all_depths, depth_align_masks, self.now_scale

    @torch.no_grad()
    def process_single_img_target(
        self,
        roots,
        # cameras,
        target_depth,
        mask_no_filter,
        num=1,
        total_num=1,
        background_hard_depth=1.0,
    ):
        """Process video frames to generate point cloud for spark web rendering.
        
        Args:
            roots: List of image paths
        """
        with torch.no_grad():
            from lightning_fabric import seed_everything

            seed_everything(100)
            x = torch.arange(self.config["orig_W"]).float() + 0.5
            y = torch.arange(self.config["orig_H"]).float() + 0.5
            points_xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
            points_3d, mask_to_align_depth, zbuf, best_depth = None, None, None, None
            points_3d_l = []
            colors_l = []
            radius_l = []
            normals_l = []
            focal_length_l = []
            is_sky_l = []
            now_scale_l = []
            all_depth_l = []
            depth_align_masks_l = []
            
            points = rearrange(points_xy, "h w c -> (h w) c")
            cameras = [self.get_camera_at_origin()]

            idxs = np.linspace(0, len(cameras)-1, num).astype(np.int32)
                
            roots_now = [roots[number] for number in range(total_num)]
            imgs = torch.stack([load_image_and_resize(root, height=self.config["orig_H"], width=self.config["orig_W"]) for root in roots_now], dim=0)
            
            for idx in idxs:
                camera = cameras[idx]
                root = roots[idx]
                img = imgs[idx]
                
                depth = self.moge.model.infer(img.to(self.device))['depth'].detach().cpu()
                depth = depth / depth.median() /100. + 1e-4
                            
                inf_mask = (depth == torch.inf) | self.generate_sky_mask(img).cpu()        
                is_sky = inf_mask.cpu()            
                            
                # depth[inf_mask] = 10.
                depth = target_depth.squeeze().detach().cpu() + 1e-4
                normals = self.get_normal( img[None])
                normals[:, 1:] *= -1
                    

                current_camera = convert_pytorch3d_kornia(camera,[self.config["orig_H"],self.config["orig_W"]])
                normals_world = current_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')    
                normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=self.config["orig_H"])
                new_normals = rearrange(normals, "b c h w -> (w h b) c")
                
                point_depth = rearrange(depth.squeeze()[None,None], "b c h w -> (w h b) c")
                points_3d = current_camera.unproject(points.cuda(), point_depth.cuda())
                colors = rearrange(img.squeeze()[None], "b c h w -> (w h b) c")
                is_sky = rearrange(is_sky.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                 
                depth_normalizer = background_hard_depth
                min_ratio = self.config['point_size_min_ratio']
                radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
                radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier']).to(self.device)
                grad_magnitude_mask = self.generate_grad_magnitude(1/depth.squeeze(), threshold=10)
                grad_magnitude_mask[:] = False
                depth_align_masks_l.append(~grad_magnitude_mask)
                grad_magnitude_mask = rearrange(grad_magnitude_mask.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()
                mask_no_filter = mask_no_filter & (depth > 1e-4)
                mask_no_filter = rearrange(mask_no_filter.squeeze()[None,None], "b c h w -> (w h b) c").detach().cpu().squeeze()

                # Compute now_scale for points corresponding to current camera
                filtered_points_3d = points_3d[~grad_magnitude_mask & mask_no_filter]
                if len(filtered_points_3d) > 0:
                    # Create default rotations and scales for the current batch of point cloud
                    num_points = len(filtered_points_3d)
                    default_rotations = torch.zeros(num_points, 4, device=self.device)
                    default_rotations[:, 0] = 1.0  # w=1, x=y=z=0
                    default_scales = torch.ones(num_points, 3, device=self.device) * 0.01
                    
                    # Use current camera to compute now_scale
                    xyz_scale = 1000  # Use the default xyz_scale
                    means3D_scaled = filtered_points_3d * xyz_scale
                    
                    q, s_target = compute_target_scale_per_frame(
                        camera,
                        means3D_scaled,
                        default_rotations,
                        default_scales,
                        rotation2normal_fn=rotation2normal,
                        min_cos=0.1,
                        config=self.config  
                    )
                    now_scale_batch = s_target.detach()
                else:
                    now_scale_batch = torch.empty(0, device=self.device)
                
                colors_l.append(colors[~grad_magnitude_mask & mask_no_filter])
                points_3d_l.append(filtered_points_3d)
                radius_l.append(radius[~grad_magnitude_mask[None]][None])
                normals_l.append(new_normals[~grad_magnitude_mask & mask_no_filter])
                focal_length_l.append(torch.tensor([camera.K[0,0,0]]*colors[~grad_magnitude_mask & mask_no_filter].shape[0], device=self.device))
                is_sky_l.append(is_sky[~grad_magnitude_mask & mask_no_filter])
                now_scale_l.append(now_scale_batch)
                all_depth_l.append(depth.squeeze())
                    
            self.points_3d = torch.cat(points_3d_l, dim=0)
            self.colors = torch.cat(colors_l, dim=0)
            self.radius = torch.cat(radius_l, dim=-1) 
            self.normals = torch.cat(normals_l, dim=0)
            self.focal_length = torch.cat(focal_length_l, dim=0)
            self.is_sky = torch.cat(is_sky_l, dim=0)
            self.now_scale = torch.cat(now_scale_l, dim=0) if now_scale_l else torch.empty(0, device=self.device)
                    
            self.points_3d_l = points_3d_l
            self.colors_l = colors_l
            self.radius_l = radius_l
            self.normals_l = normals_l
            self.focal_length_l = focal_length_l
            self.is_sky_l = is_sky_l    

            return self.points_3d, self.colors, self.radius, self.normals, imgs, cameras, self.focal_length, self.is_sky, all_depth_l, depth_align_masks_l, self.now_scale



def finetune_depth_model_step(model,target_depth, inpainted_image,a,b, mask_align=None, mask_cutoff=None, cutoff_depth=None):
    # next_depth, _ = get_depth_moge(model,zbuf,inpainted_image.detach().cuda())
    
    def infer(
        self, 
        image: torch.Tensor, 
        fov_x = None,
        resolution_level: int = 9,
        num_tokens: int = None,
        apply_mask: bool = True,
        force_projection: bool = True,
        use_fp16: bool = True,
    ) :
        """
        User-friendly inference function
        
        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)\        """
        
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=use_fp16):
            output = self.forward(image, num_tokens)
        points, mask = output['points'], output['mask']

        mask_binary = mask > self.mask_threshold

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        depth = points[..., 2] + shift[..., None, None]
        
        # If projection constraint is forced, recompute the point map using the actual depth map
        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

        # Apply mask if needed
        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        if omit_batch_dim:
            points = points.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
            depth = depth.squeeze(0)
            mask_binary = mask_binary.squeeze(0)
            mask = mask.squeeze(0)

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'mask_prob': torch.sigmoid(mask)
        }

        return return_dict

    
    device = model.device
    next_depth = infer( model,inpainted_image.detach().cuda() )['depth']
    inf_mask = (next_depth==torch.inf)
    next_depth[inf_mask] = 1e7
    # print(type(a),type(b),type(next_depth))
    next_depth = a * next_depth + b 
    assert mask_align is not None 
    mask_align = mask_align.cuda()
    # L1 loss for the mask_align region
    loss_align = F.l1_loss(target_depth.detach().cuda(), next_depth, reduction="none")
    # loss_align = F.mse_loss(target_depth.detach().cuda(), next_depth,reduction="none")
    # print(loss_align[(mask_align  * (~inf_mask)) > 0].max())
    if mask_align is not None and torch.any(mask_align):
        mask_align = mask_align.detach()
        loss_align = (loss_align * mask_align * (~inf_mask))[(mask_align  * (~inf_mask)) > 0].mean()
        # print(loss_align)
    else:
        loss_align = torch.zeros(1).to(device)

    # Hinge loss for the mask_cutoff region
    if mask_cutoff is not None and cutoff_depth is not None and torch.any(mask_cutoff):
        hinge_loss = (cutoff_depth - next_depth).clamp(min=0)
        hinge_loss = F.l1_loss(hinge_loss, torch.zeros_like(hinge_loss), reduction="none")
        mask_cutoff = mask_cutoff.detach()
        hinge_loss = (hinge_loss * mask_cutoff)[mask_cutoff > 0].mean()
    else:
        hinge_loss = torch.zeros(1).to(device)

    total_loss = loss_align + hinge_loss
    if torch.isnan(total_loss):
        raise ValueError("Depth FT loss is NaN")
    # print both losses and total loss
    # print(f"(1000x) loss_align: {loss_align.item()*1000:.4f}, hinge_loss: {hinge_loss.item()*1000:.4f}, total_loss: {total_loss.item()*1000:.4f}")

    return next_depth,total_loss


def load_image_and_resize(image_path, height, width, directly_resize=False):
    import torchvision.transforms.functional as TF

    img = Image.open(image_path)
        
    # Check if the image is already at the target dimensions
    if img.size == (width, height) or directly_resize:
        # If it's already the right size, just convert to RGB and tensor
        img = img.resize((width, height), Image.LANCZOS)
        img = img.convert('RGB')
        img = TF.to_tensor(img)
    else:
        # Calculate the scaling factor
        original_width, original_height = img.size
        width_scale = width / original_width
        height_scale = height / original_height
        scale = max(width_scale, height_scale)
        
        # Apply the scaling
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop the image
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        img = img.crop((left, top, right, bottom))
        
        # Convert to RGB and then to tensor
        img = img.convert('RGB')
        img = TF.to_tensor(img)
    return img

def compute_scale_and_shift_full(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.cpu().numpy()   
    if not isinstance(target, np.ndarray):  
        target = target.cpu().numpy()
    if not isinstance(mask, np.ndarray):    
        mask = mask.cpu().numpy()
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)
    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01
    # print(det,a_00,a_01,a_11,b_0,b_1,x_0,x_1)
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1

def compute_shift_only(prediction, target, mask, scale=1.0):
    
    """
    Compute the optimal shift so that scale * prediction + shift best approximates target

    Args:
        prediction: Predicted values
        target: Target values
        mask: Mask
        scale: Fixed scale factor (default 1.0)

    Returns:
        shift: Optimal shift value
    """
    # Convert to numpy arrays
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.cpu().numpy()   
    if not isinstance(target, np.ndarray):  
        target = target.cpu().numpy()
    if not isinstance(mask, np.ndarray):    
        mask = mask.cpu().numpy()
        
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)
    
    # Compute weighted average
    # Objective function: minimize sum(mask * (scale * prediction + shift - target)^2)
    # Differentiate and set to 0: sum(mask * (scale * prediction + shift - target)) = 0
    # Solution: shift = (sum(mask * target) - scale * sum(mask * prediction)) / sum(mask)
    
    sum_mask = np.sum(mask)
    if sum_mask == 0:
        return 0.0
    
    sum_target = np.sum(mask * target)
    sum_prediction = np.sum(mask * prediction)
    
    shift = (sum_target - scale * sum_prediction) / sum_mask
    
    return shift

def inpaint_nearest_bilateral_preserve(depth_np, mask_np, bilateral=True):
    """
    depth_np: (H, W) float32 depth map
    mask_np: (H, W) bool, True indicates holes to fill
    """
    from scipy.ndimage import distance_transform_edt
    import cv2
    
    # Nearest neighbor fill using distance transform
    dist, inds = distance_transform_edt(mask_np, return_indices=True)
    nearest_y, nearest_x = inds
    filled = depth_np[nearest_y, nearest_x]  # Complete image with filled holes
    
    if bilateral:
        # Bilateral filter the filled image
        smooth = cv2.bilateralFilter(filled.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=3)
        
        # Final result: original values + mask region replaced with smooth
        result = depth_np.copy()
        result[mask_np] = smooth[mask_np]
        return result
    else:
        # Only nearest neighbor fill
        result = depth_np.copy()
        result[mask_np] = filled[mask_np]
        return result
