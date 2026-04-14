#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os

import numpy as np
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from io import BytesIO
from simple_knn._C import distCUDA2
from utils.general import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system import mkdir_p
from utils.sh import RGB2SH
from utils.graphics import BasicPointCloud
from utils.general import strip_symmetric, build_scaling_rotation, normal2rotation, rotation2normal
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import copy
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
# global label management system - using externally defined global variables
from scene.cameras import Camera

def convert_pt3d_cam_to_3dgs_cam(pt3d_cam: PerspectiveCameras, image=None, xyz_scale=1, config=None):

    image = torch.zeros(3, config['orig_H'], config['orig_W'])

    transform_matrix_pt3d = pt3d_cam.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
    transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale
    transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
    opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=torch.device('cuda')))
    transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d
    transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
    c2w = np.array(transform_matrix)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    focal_length = pt3d_cam.K[0, 0, 0].item()
    half_img_size_x = pt3d_cam.K[0, 0, 2].item()
    fovx = 2*np.arctan(half_img_size_x / focal_length)
    
    half_img_size_y = pt3d_cam.K[0, 1, 2].item()
    fovy = 2*np.arctan(half_img_size_y / focal_length)
    tdgs_cam = Camera(image=image, R=R, T=T, FoVx=fovx, FoVy=fovy)

    return tdgs_cam
def get_global_label_id(label_name, global_label_names, global_label_map):
    """Get global ID for a label, create if not exists"""
    if label_name not in global_label_map:
        label_id = len(global_label_names)
        global_label_names.append(label_name)
        global_label_map[label_name] = label_id
        print(f"🏷️ Created new global label: '{label_name}' -> ID {label_id}")
    return global_label_map[label_name]

def get_global_label_name(label_id, global_label_names):
    """Get label name by ID"""
    if 0 <= label_id < len(global_label_names):
        return global_label_names[label_id]
    return "unknown"

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = lambda x: (torch.tanh(x) * 0.51).clamp(-0.5, 0.5) + 0.5
        self.inverse_opacity_activation = lambda y: torch.atanh((y - 0.5) / 0.51)

        self.rotation_activation = torch.nn.functional.normalize
        self.color_activation = lambda x: (torch.tanh(x) * 0.51).clamp(-0.5, 0.5) + 0.5
        self.inverse_color_activation = lambda y: torch.atanh((y - 0.5) / 0.51)

    
    def __init__(self, sh_degree : int, previous_gaussian=None, floater_dist2_threshold=0.0002, config = None):
        """
        args:
            previous_gaussian : GaussianModel; merge all its points directly into the current model, all set as trainable
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1.
        self.floater_dist2_threshold = floater_dist2_threshold
        self.setup_functions()
        assert config is not None, "config is required"
        self.config = config
        if previous_gaussian is not None:
            print("Merging previous gaussian into current model (all trainable)")
            
            # directly merge all points from previous into current trainable parameters
            # merge main model parameters
            all_xyz = torch.cat([previous_gaussian._xyz.detach(), previous_gaussian._xyz_prev], dim=0)
            if len(previous_gaussian._features_dc.shape) == 3 and previous_gaussian._features_dc.shape[1] == 3 and previous_gaussian._features_dc.shape[2] == 1:
                # if [N, 3, 1], convert to [N, 1, 3]
                previous_gaussian._features_dc = previous_gaussian._features_dc.permute(0, 2, 1)
            if len(previous_gaussian._features_dc_prev.shape) == 3 and previous_gaussian._features_dc_prev.shape[1] == 3 and previous_gaussian._features_dc_prev.shape[2] == 1:
                # if [N, 3, 1], convert to [N, 1, 3]
                previous_gaussian._features_dc_prev = previous_gaussian._features_dc_prev.permute(0, 2, 1)
            all_features_dc = torch.cat([previous_gaussian._features_dc.detach(), previous_gaussian._features_dc_prev], dim=0)
            all_scaling = torch.cat([previous_gaussian._scaling.detach(), previous_gaussian._scaling_prev], dim=0)
            all_rotation = torch.cat([previous_gaussian._rotation.detach(), previous_gaussian._rotation_prev], dim=0)
            all_opacity = torch.cat([previous_gaussian._opacity.detach(), previous_gaussian._opacity_prev], dim=0)
            all_focal_length = torch.cat([previous_gaussian._focal_length.detach(), previous_gaussian._focal_length_prev], dim=0)
            
            # set as trainable parameters
            self._xyz = nn.Parameter(all_xyz.requires_grad_(True))
            self._features_dc = nn.Parameter(all_features_dc.requires_grad_(True))
            self._scaling = nn.Parameter(all_scaling.requires_grad_(True))
            self._rotation = nn.Parameter(all_rotation.requires_grad_(True))
            self._opacity = nn.Parameter(all_opacity.requires_grad_(True))
            self._focal_length = all_focal_length  # this does not need gradients
            
                            # merge next_scale, prior_scale and now_scale
            all_next_scale = torch.cat([previous_gaussian.next_scale.detach(), previous_gaussian.next_scale_prev], dim=0)
            all_prior_scale = torch.cat([previous_gaussian.prior_scale.detach(), previous_gaussian.prior_scale_prev], dim=0)
            all_now_scale = torch.cat([previous_gaussian.now_scale.detach(), previous_gaussian.now_scale_prev], dim=0)
            self.next_scale = all_next_scale.detach()
            self.prior_scale = all_prior_scale.detach()
            self.now_scale = all_now_scale.detach()
            

            
            # merge training state (if exists)
            if hasattr(previous_gaussian, 'max_radii2D') and previous_gaussian.max_radii2D.numel() > 0:
                all_max_radii2D = torch.cat([
                    previous_gaussian.max_radii2D.detach(), 
                    getattr(previous_gaussian, 'max_radii2D_prev', torch.zeros(previous_gaussian._xyz_prev.shape[0], device='cuda'))
                ], dim=0)
                all_xyz_gradient_accum = torch.cat([
                    previous_gaussian.xyz_gradient_accum.detach(), 
                    getattr(previous_gaussian, 'xyz_gradient_accum_prev', torch.zeros(previous_gaussian._xyz_prev.shape[0], 1, device='cuda'))
                ], dim=0)
                all_denom = torch.cat([
                    previous_gaussian.denom.detach(), 
                    getattr(previous_gaussian, 'denom_prev', torch.zeros(previous_gaussian._xyz_prev.shape[0], 1, device='cuda'))
                ], dim=0)
            else:
                # if no training state, create new ones
                n_total = all_xyz.shape[0]
                all_max_radii2D = torch.zeros(n_total, device='cuda')
                all_xyz_gradient_accum = torch.zeros(n_total, 1, device='cuda')
                all_denom = torch.zeros(n_total, 1, device='cuda')
            
            self.max_radii2D = all_max_radii2D
            self.xyz_gradient_accum = all_xyz_gradient_accum
            self.denom = all_denom
            
            # merge filter_3D (if exists)
            if hasattr(previous_gaussian, 'filter_3D') and previous_gaussian.filter_3D.numel() > 0:
                all_filter_3D = torch.cat([
                    previous_gaussian.filter_3D.detach(), 
                    getattr(previous_gaussian, 'filter_3D_prev', torch.ones(previous_gaussian._xyz_prev.shape[0], 1, device='cuda') * 0.001)
                ], dim=0)
                self.filter_3D = all_filter_3D
            
            # transfer global masks (now all points are "current" points)
            self.visibility_filter_all = previous_gaussian.visibility_filter_all.clone()
            self.is_sky_filter = previous_gaussian.get_sky_filter_all.clone()
            self.delete_mask_all = previous_gaussian.delete_mask_all.clone()
            
            # transfer point labels - compatible with old and new modes
            if hasattr(previous_gaussian, 'get_point_labels_all'):
                # new mode: previous_gaussian has a separated label system
                all_labels = previous_gaussian.get_point_labels_all.clone()
                self.point_labels = all_labels  # all become trainable
                self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda')
            elif hasattr(previous_gaussian, 'point_labels') and previous_gaussian.point_labels.numel() > 0:
                # old mode: previous_gaussian's point_labels contains all points
                self.point_labels = previous_gaussian.point_labels.clone()
                self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda')
            else:
                # if previous has no labels, default to main label (0)
                self.point_labels = torch.zeros(all_xyz.shape[0], dtype=torch.long, device='cuda')
                self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda')
            
            print(f"Merged {all_xyz.shape[0]} total points into trainable model")
            print(f"  - xyz: {self._xyz.shape}")
            print(f"  - features_dc: {self._features_dc.shape}")
            print(f"  - focal_length: {self._focal_length.shape}")
            print(f"  - visibility_filter_all: {self.visibility_filter_all.shape}")
            
        else:
            # initialize as empty (brand new model)
            self._xyz = torch.empty(0).cuda()
            self._features_dc = torch.empty(0).cuda()
            self._scaling = torch.empty(0).cuda()
            self._rotation = torch.empty(0).cuda()
            self._opacity = torch.empty(0).cuda()
            self._focal_length = torch.empty(0).cuda()
            self.next_scale = torch.empty(0).cuda()
            self.prior_scale = torch.empty(0).cuda()
            self.now_scale = torch.empty(0).cuda()
            self.max_radii2D = torch.empty(0).cuda()
            self.xyz_gradient_accum = torch.empty(0).cuda()
            self.denom = torch.empty(0).cuda()
            self.visibility_filter_all = torch.empty(0, dtype=torch.bool).cuda()
            self.is_sky_filter = torch.empty(0, dtype=torch.bool).cuda()
            self.delete_mask_all = torch.empty(0, dtype=torch.bool).cuda()
            self.point_labels = torch.empty(0, dtype=torch.long, device='cuda')
        
        # all _prev attributes are always empty
        self._xyz_prev = torch.empty(0).cuda()
        self._features_dc_prev = torch.empty(0, 1, 3).cuda()
        self._scaling_prev = torch.empty(0).cuda()
        self._rotation_prev = torch.empty(0).cuda()
        self._opacity_prev = torch.empty(0).cuda()
        self.filter_3D_prev = torch.empty(0).cuda()
        self._focal_length_prev = torch.empty(0).cuda()
        self.next_scale_prev = torch.empty(0).cuda()
        self.prior_scale_prev = torch.empty(0).cuda()
        self.now_scale_prev = torch.empty(0).cuda()
        self.max_radii2D_prev = torch.empty(0).cuda()
        self.xyz_gradient_accum_prev = torch.empty(0, 1).cuda()
        self.denom_prev = torch.empty(0, 1).cuda()
        
        # new: separated sky filter attributes
        self.is_sky_filter_prev = torch.empty(0, dtype=torch.bool).cuda()
        self.visibility_filter_all_prev = torch.empty(0, dtype=torch.bool).cuda() 
        self.delete_mask_all_prev = torch.empty(0, dtype=torch.bool).cuda()
        
        # point label system - using global label mapping, separated by current/prev mode
        # self.point_labels = torch.empty(0, dtype=torch.long, device='cuda')      # labels of current trainable points
        self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda') # labels of non-trainable points
        
        print("Initialization complete - all points are trainable, no prev points")
    
    def get_label_mask(self, label_name):
        """Get mask of all points by label name (trainable + non-trainable)"""
        # dynamically import global variables
        global_label_map = None
        try:
            import sys
            for module_name, module in sys.modules.items():
                if hasattr(module, 'GLOBAL_LABEL_MAP'):
                    global_label_map = module.GLOBAL_LABEL_MAP
                    break
            if global_label_map is None:
                raise ImportError("Cannot find GLOBAL_LABEL_MAP")
        except ImportError:
            global_label_map = {"main": 0}
            
        if label_name in global_label_map:
            label_id = global_label_map[label_name]
            return self.get_point_labels_all == label_id
        return torch.zeros(self.get_point_labels_all.shape[0], dtype=torch.bool, device=self.get_point_labels_all.device)
    
    def get_current_label_mask(self, label_name):
        """Get mask of only current trainable points by label name"""
        # dynamically import global variables
        global_label_map = None
        try:
            import sys
            for module_name, module in sys.modules.items():
                if hasattr(module, 'GLOBAL_LABEL_MAP'):
                    global_label_map = module.GLOBAL_LABEL_MAP
                    break
            if global_label_map is None:
                raise ImportError("Cannot find GLOBAL_LABEL_MAP")
        except ImportError:
            global_label_map = {"main": 0}
            
        if label_name in global_label_map:
            label_id = global_label_map[label_name]
            # directly use current trainable points' labels
            if self.point_labels.numel() > 0:
                return self.point_labels == label_id
            else:
                n_current = self.get_xyz.shape[0]
                return torch.zeros(n_current, dtype=torch.bool, device='cuda')
        return torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device='cuda')
    
    def set_points_label(self, mask, label_name, global_label_names=None, global_label_map=None):
        """Set points specified by mask to a given label"""
        # if global variables not provided, try to import from run_inf module
        if global_label_names is None or global_label_map is None:
            try:
                import sys
                # find run_inf module
                run_inf_module = None
                for module_name, module in sys.modules.items():
                    if hasattr(module, 'GLOBAL_LABEL_NAMES') and hasattr(module, 'GLOBAL_LABEL_MAP'):
                        global_label_names = module.GLOBAL_LABEL_NAMES
                        global_label_map = module.GLOBAL_LABEL_MAP
                        break
                if global_label_names is None:
                    raise ImportError("Cannot find GLOBAL_LABEL_NAMES")
            except ImportError:
                # if not found, use default values
                global_label_names = ["main"]
                global_label_map = {"main": 0}
        
        label_id = get_global_label_id(label_name, global_label_names, global_label_map)
        
        # handle current and prev labels separately
        n_current = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0] if self._xyz_prev.numel() > 0 else 0
        n_total = n_current + n_prev
        
        if len(mask) != n_total:
            raise ValueError(f"Mask length ({len(mask)}) must equal total points ({n_total})")
        
        # split mask
        current_mask = mask[:n_current]
        prev_mask = mask[n_current:] if n_prev > 0 else torch.empty(0, dtype=torch.bool, device=mask.device)
        
        # set labels
        if current_mask.any():
            self.point_labels[current_mask] = label_id
        if n_prev > 0 and prev_mask.any():
            self.point_labels_prev[prev_mask] = label_id
    
    def get_points_by_label_id(self, label_id):
        """Get mask of all points by label ID"""
        return self.get_point_labels_all == label_id
    
    def list_point_labels(self):
        """List all labels and their point counts"""
        print("Point labels:")
        all_labels = self.get_point_labels_all
        unique_labels = torch.unique(all_labels)
        for label_id in unique_labels:
            label_name = get_global_label_name(label_id.item())
            count = (all_labels == label_id).sum().item()
            print(f"  {label_name} (ID {label_id}): {count} points")
    
    def add_gaussian_with_label(self, other_gaussian, label_name):
        """Merge another gaussian and assign a label to the new points"""
        n_original = self.get_xyz_all.shape[0]
        
        # normal merge
        self.merge_gaussian(other_gaussian)
        
        # set labels for new points
        n_new = self.get_xyz_all.shape[0]
        if n_new > n_original:
            new_points_mask = torch.zeros(n_new, dtype=torch.bool, device='cuda')
            new_points_mask[n_original:] = True
            self.set_points_label(new_points_mask, label_name)
            print(f"🔗 Added {n_new - n_original} points with label '{label_name}'")
    
    def get_points_by_label(self, label_name):
        """Get all point data for a specific label"""
        mask = self.get_label_mask(label_name)
        if mask.any():
            return {
                'xyz': self.get_xyz_all[mask],
                'features': self.get_features_all[mask],
                'scaling': self.get_scaling_all[mask],
                'rotation': self.get_rotation_all[mask],
                'opacity': self.get_opacity_all[mask],
                'count': mask.sum().item()
            }
        return None
    
    def remove_points_by_label(self, label_name):
        """Remove all points with a specific label"""
        # get mask based on point_labels (only includes labeled points in current+prev)
        base_mask = self.get_label_mask(label_name)
        
        if base_mask.any():
            # compute actual total point count
            n_curr = self.get_xyz.shape[0]
            n_prev = self._xyz_prev.shape[0] if self._xyz_prev.numel() > 0 else 0
            n_total = n_curr + n_prev
            
            # ensure mask covers all points
            if len(base_mask) != n_total:
                # if point_labels only covers some points, need to expand mask
                if len(base_mask) == n_curr:
                    # point_labels only contains current points, add False for prev points
                    prev_mask = torch.zeros(n_prev, dtype=torch.bool, device=base_mask.device)
                    full_mask = torch.cat([base_mask, prev_mask])
                elif len(base_mask) < n_total:
                    # point_labels is incomplete, expand to full size
                    missing = n_total - len(base_mask)
                    extra_mask = torch.zeros(missing, dtype=torch.bool, device=base_mask.device)
                    full_mask = torch.cat([base_mask, extra_mask])
                else:
                    # point_labels is larger than total points, this should not happen
                    print(f"⚠️ Warning: point_labels size ({len(base_mask)}) > total points ({n_total})")
                    full_mask = base_mask[:n_total]
            else:
                full_mask = base_mask
            
            print(f"🗑️ Removing {full_mask.sum().item()}/{n_total} points with label '{label_name}'")
            self.delete_all_points(full_mask)
        else:
            print(f"🗑️ No points found with label '{label_name}'")
    
    # label-based trainability control methods
    def set_trainable_by_labels(self, trainable_labels=None, non_trainable_labels=None):
        """
        Set trainability based on point labels

        Args:
            trainable_labels: list of trainable label names, None means no restriction
            non_trainable_labels: list of non-trainable label names, None means no restriction

        Note: if both arguments are specified, non_trainable_labels has higher priority
        """
        n_current = self.get_xyz.shape[0]
        if n_current == 0:
            print("⚠️  No trainable points to control")
            return
        
        # get labels of current trainable points
        current_labels = self.point_labels[:n_current]
        
        # initialize as all trainable
        trainable_mask = torch.ones(n_current, dtype=torch.bool, device='cuda')
        
        # apply non-trainable labels (higher priority)
        if non_trainable_labels is not None:
            for label_name in non_trainable_labels:
                label_mask = self.get_label_mask(label_name)[:n_current]  # only take current trainable points part
                trainable_mask[label_mask] = False
                count = label_mask.sum().item()
                print(f"🔒 Label '{label_name}': {count} points set to non-trainable")
        
        # apply trainable labels (if specified)
        if trainable_labels is not None:
            # if trainable labels are specified, only points with these labels are trainable
            specific_trainable_mask = torch.zeros(n_current, dtype=torch.bool, device='cuda')
            for label_name in trainable_labels:
                label_mask = self.get_label_mask(label_name)[:n_current]
                specific_trainable_mask[label_mask] = True
                count = label_mask.sum().item()
                print(f"✅ Label '{label_name}': {count} points set to trainable")
            
            # intersect with existing mask (considering non_trainable_labels effect)
            trainable_mask = trainable_mask & specific_trainable_mask
        
        n_trainable = trainable_mask.sum().item()
        n_non_trainable = (~trainable_mask).sum().item()
        
        print(f"📊 Trainability control result: {n_trainable} trainable, {n_non_trainable} non-trainable")
        
        # use existing set_trainable_mask function
        self.set_trainable_mask(trainable_mask)
    
    def freeze_labels(self, *label_names):
        """
        Freeze points with specified labels (set as non-trainable)

        Args:
            *label_names: label names to freeze
        """
        print(f"🧊 Freezing labels: {list(label_names)}")
        self.set_trainable_by_labels(non_trainable_labels=list(label_names))
    
    def train_only_labels(self, *label_names):
        """
        Train only points with specified labels, freeze all others

        Args:
            *label_names: label names allowed to train
        """
        print(f"🎯 Training only labels: {list(label_names)}")
        self.set_trainable_by_labels(trainable_labels=list(label_names))
    
    def unfreeze_all_labels(self):
        """
        Unfreeze all points (set all as trainable)
        """
        print("🔓 Unfreezing all points")
        n_current = self.get_xyz.shape[0]
        if n_current > 0:
            trainable_mask = torch.ones(n_current, dtype=torch.bool, device='cuda')
            self.set_trainable_mask(trainable_mask)
        
        # if there are prev points, merge them back to trainable
        if hasattr(self, '_xyz_prev') and self._xyz_prev.shape[0] > 0:
            print("🔄 Merging non-trainable points back to trainable")
            self.merge_all_to_trainable()
    
    def get_trainable_labels_info(self):
        """
        Get label info for current trainable and non-trainable points

        Returns:
            dict: statistics of labels for trainable and non-trainable points
        """
        # dynamically import global variables
        global_label_names = None
        global_label_map = None
        try:
            import sys
            for module_name, module in sys.modules.items():
                if hasattr(module, 'GLOBAL_LABEL_NAMES') and hasattr(module, 'GLOBAL_LABEL_MAP'):
                    global_label_names = module.GLOBAL_LABEL_NAMES
                    global_label_map = module.GLOBAL_LABEL_MAP
                    break
            if global_label_names is None:
                raise ImportError("Cannot find GLOBAL_LABEL_NAMES")
        except ImportError:
            global_label_names = ["main"]
            global_label_map = {"main": 0}
        
        n_current = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0] if hasattr(self, '_xyz_prev') else 0
        
        info = {
            'trainable': {},
            'non_trainable': {},
            'total_trainable_points': n_current,
            'total_non_trainable_points': n_prev
        }
        
        # count labels of trainable points
        if n_current > 0:
            current_labels = self.point_labels[:n_current]
            for label_name in global_label_names:
                if label_name in global_label_map:
                    label_id = global_label_map[label_name]
                    count = (current_labels == label_id).sum().item()
                    if count > 0:
                        info['trainable'][label_name] = count
        
        # count labels of non-trainable points
        if n_prev > 0:
            prev_labels = self.point_labels[n_current:n_current+n_prev]
            for label_name in global_label_names:
                if label_name in global_label_map:
                    label_id = global_label_map[label_name]
                    count = (prev_labels == label_id).sum().item()
                    if count > 0:
                        info['non_trainable'][label_name] = count
        
        return info
    
    def print_trainable_status(self):
        """
        Print current trainable status information
        """
        info = self.get_trainable_labels_info()
        
        print("🔍 Current Trainable Status:")
        print(f"  📈 Trainable points: {info['total_trainable_points']}")
        for label, count in info['trainable'].items():
            print(f"    - {label}: {count} points")
        
        print(f"  📉 Non-trainable points: {info['total_non_trainable_points']}")
        for label, count in info['non_trainable'].items():
            print(f"    - {label}: {count} points")
        
    def merge_gaussian(self, previous_gaussian):
        """
        Merge all points from another GaussianModel into the current model (all set as trainable)

        Args:
            previous_gaussian: GaussianModel; model to merge
        """
        print(f"Merging gaussian with {previous_gaussian.get_xyz_all.shape[0]} points into current model")
        
        # ensure global mask dimension consistency for both models before merge
        self._ensure_global_mask_consistency()
        previous_gaussian._ensure_global_mask_consistency()
        
        # record current model's point count before merge, this is critical!
        current_n = self.get_xyz.shape[0] if self._xyz.numel() > 0 else 0
        print(f"🔧 Current model has {current_n} points before merge")
        
        # ===== get all points from previous (current + prev) =====
        prev_all_xyz = previous_gaussian.get_xyz_all.detach()
        prev_all_features_dc = previous_gaussian.get_features_all.detach()
        prev_all_scaling = previous_gaussian.get_scaling_all.detach()
        prev_all_rotation = previous_gaussian.get_rotation_all.detach()
        prev_all_opacity = previous_gaussian.get_opacity_all.detach()
        prev_all_focal_length = previous_gaussian.get_focal_length_all.detach()
        prev_all_next_scale = previous_gaussian.get_next_scale_all.detach()
        prev_all_prior_scale = previous_gaussian.get_prior_scale_all.detach()
        prev_all_now_scale = previous_gaussian.get_now_scale_all.detach()
        
        # ===== merge with current model =====
        if self._xyz.numel() == 0:
            # current model is empty, set directly
            merged_xyz = prev_all_xyz
            merged_features_dc = prev_all_features_dc
            merged_scaling = self.scaling_inverse_activation(prev_all_scaling)
            merged_rotation = prev_all_rotation
            merged_opacity = self.inverse_opacity_activation(prev_all_opacity)
            merged_focal_length = prev_all_focal_length
            merged_next_scale = prev_all_next_scale
            merged_prior_scale = prev_all_prior_scale
            merged_now_scale = prev_all_now_scale
        else:
            # current model has data, concatenate
            merged_xyz = torch.cat([self._xyz.detach(), prev_all_xyz], dim=0)
            # ensure current feature dimensions are correct
            current_features = self._features_dc.detach()
            if len(current_features.shape) == 3 and current_features.shape[1] == 3 and current_features.shape[2] == 1:
                current_features = current_features.permute(0, 2, 1)  # [N, 3, 1] -> [N, 1, 3]
            merged_features_dc = torch.cat([current_features, prev_all_features_dc], dim=0)
            merged_scaling = torch.cat([self._scaling.detach(), self.scaling_inverse_activation(prev_all_scaling)], dim=0)
            merged_rotation = torch.cat([self._rotation.detach(), prev_all_rotation], dim=0)
            merged_opacity = torch.cat([self._opacity.detach(), self.inverse_opacity_activation(prev_all_opacity)], dim=0)
            merged_focal_length = torch.cat([self._focal_length.detach(), prev_all_focal_length], dim=0)
            merged_next_scale = torch.cat([self.next_scale.detach(), prev_all_next_scale], dim=0)
            merged_prior_scale = torch.cat([self.prior_scale.detach(), prev_all_prior_scale], dim=0)
            merged_now_scale = torch.cat([self.now_scale.detach(), prev_all_now_scale], dim=0)
        
        # set as trainable parameters
        self._xyz = nn.Parameter(merged_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(merged_features_dc.requires_grad_(True))
        self._scaling = nn.Parameter(merged_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(merged_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(merged_opacity.requires_grad_(True))
        self._focal_length = merged_focal_length  # no gradients needed
        self.next_scale = merged_next_scale.detach()
        self.prior_scale = merged_prior_scale.detach()
        self.now_scale = merged_now_scale.detach()
        
        # ===== reinitialize training state =====
        n_total = merged_xyz.shape[0]
        self.max_radii2D = torch.zeros(n_total, device="cuda")
        self.xyz_gradient_accum = torch.zeros(n_total, 1, device="cuda")
        self.denom = torch.zeros(n_total, 1, device="cuda")
        
        # # ===== merge filter_3D =====
        self.filter_3D = torch.ones(n_total, 1, device='cuda') * 0.001

        # prev_all_filter_3D = previous_gaussian.get_scaling_with_3D_filter_all.detach() if hasattr(previous_gaussian, 'filter_3D') else None
        # if prev_all_filter_3D is not None:
        #     if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
        #         # convert to filter values instead of scaling values
        #         prev_filter_3D = torch.sqrt(torch.square(prev_all_filter_3D) - torch.square(prev_all_scaling))
        #         self.filter_3D = torch.cat([self.filter_3D.detach(), prev_filter_3D], dim=0)
        #     else:
        #         # current has no filter_3D, create default values for all points
        #         self.filter_3D = torch.ones(n_total, 1, device='cuda') * 0.001
        # else:
        #     # previous has no filter_3D
        #     if not hasattr(self, 'filter_3D') or self.filter_3D.numel() == 0:
        #         self.filter_3D = torch.ones(n_total, 1, device='cuda') * 0.001
        #     else:
        #         # add default filter values for previous points
        #         prev_default_filter = torch.ones(prev_all_xyz.shape[0], 1, device='cuda') * 0.001
        #         self.filter_3D = torch.cat([self.filter_3D.detach(), prev_default_filter], dim=0)
        
        # ===== merge global masks =====
        # current_n was already correctly computed at the beginning of the function
        
        if current_n > 0:
            # current model has data
            current_visibility = self.visibility_filter_all[:current_n] if hasattr(self, 'visibility_filter_all') else torch.ones(current_n, dtype=torch.bool, device='cuda')
            current_delete = self.delete_mask_all[:current_n] if hasattr(self, 'delete_mask_all') else torch.zeros(current_n, dtype=torch.bool, device='cuda')
            # fix: is_sky_filter already corresponds to current points, ensure size matches
            if hasattr(self, 'is_sky_filter') and self.is_sky_filter.numel() > 0:
                if self.is_sky_filter.shape[0] == current_n:
                    current_sky = self.is_sky_filter
                elif self.is_sky_filter.shape[0] > current_n:
                    current_sky = self.is_sky_filter[:current_n]
                else:
                    # is_sky_filter size is smaller than current_n, need to expand
                    missing = current_n - self.is_sky_filter.shape[0]
                    extra = torch.zeros(missing, dtype=torch.bool, device='cuda')
                    current_sky = torch.cat([self.is_sky_filter, extra])
            else:
                current_sky = torch.zeros(current_n, dtype=torch.bool, device='cuda')
        else:
            # current model is empty
            current_visibility = torch.empty(0, dtype=torch.bool, device='cuda')
            current_delete = torch.empty(0, dtype=torch.bool, device='cuda')
            current_sky = torch.empty(0, dtype=torch.bool, device='cuda')
        
        # previous's global masks
        prev_visibility = previous_gaussian.visibility_filter_all.clone()
        prev_delete = previous_gaussian.delete_mask_all.clone()
        # fix: should get sky filter of all previous points, not just the trainable part
        prev_sky = previous_gaussian.get_sky_filter_all.clone()
        
        # merge global masks
        self.visibility_filter_all = torch.cat([current_visibility, prev_visibility])
        self.delete_mask_all = torch.cat([current_delete, prev_delete])
        # fix: after merge, all points become trainable, so is_sky_filter should include all merged points
        self.is_sky_filter = torch.cat([current_sky.to(prev_all_xyz.device), prev_sky.to(prev_all_xyz.device)])
        # reset is_sky_filter_prev to empty, because all points are trainable after merge
        self.is_sky_filter_prev = torch.empty(0, dtype=torch.bool, device='cuda')
        
        # merge point labels - using current/prev separation mode
        # process current model's labels
        if current_n > 0:
            current_labels = self.point_labels.clone() if self.point_labels.numel() > 0 else torch.zeros(current_n, dtype=torch.long, device='cuda')
        else:
            current_labels = torch.empty(0, dtype=torch.long, device='cuda')
        
        # process previous's labels - compatible with old and new modes
        # import pdb; pdb.set_trace()
        if hasattr(previous_gaussian, 'get_point_labels_all'):
            # new mode: previous_gaussian has a separated label system
            prev_all_labels = previous_gaussian.get_point_labels_all.clone()
        elif hasattr(previous_gaussian, 'point_labels') and previous_gaussian.point_labels.numel() > 0:
            # old mode: previous_gaussian's point_labels contains all points
            prev_all_labels = previous_gaussian.point_labels.clone()
        else:
            # no labels, default to main label (0)
            prev_all_labels = torch.zeros(prev_all_xyz.shape[0], dtype=torch.long, device='cuda')
        
        # new mode: current stored in point_labels, previous stored in point_labels_prev
        self.point_labels = torch.cat([current_labels, prev_all_labels], dim=0) 
        # self.point_labels_prev = prev_all_labels
        

        
        #         # ===== clear all _prev attributes (all trainable) =====
        self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda')
        self.next_scale_prev = torch.empty(0).cuda()
        self.prior_scale_prev = torch.empty(0).cuda()
        self._xyz_prev = torch.empty(0).cuda()
        self._features_dc_prev = torch.empty(0).cuda()
        self._scaling_prev = torch.empty(0).cuda()
        self._rotation_prev = torch.empty(0).cuda()
        self._opacity_prev = torch.empty(0).cuda()
        self.filter_3D_prev = torch.empty(0).cuda()
        self._focal_length_prev = torch.empty(0).cuda()
        self.max_radii2D_prev = torch.empty(0).cuda()
        self.xyz_gradient_accum_prev = torch.empty(0, 1).cuda()
        self.denom_prev = torch.empty(0, 1).cuda()
        
        print(f"Merge complete: {n_total} total trainable points")
        print(f"  - xyz: {self._xyz.shape}")
        print(f"  - features_dc: {self._features_dc.shape}")
        print(f"  - focal_length: {self._focal_length.shape}")
        print(f"  - visibility_filter_all: {self.visibility_filter_all.shape}")
        
        # verify global mask dimension correctness
        expected_total = current_n + prev_all_xyz.shape[0]
        print(f"🔧 Verification: current_n={current_n} + prev_points={prev_all_xyz.shape[0]} = expected_total={expected_total}")
        print(f"🔧 Actual total points: {n_total}")
        print(f"🔧 Global mask sizes: visibility={self.visibility_filter_all.shape[0]}, sky={self.is_sky_filter.shape[0]}, delete={self.delete_mask_all.shape[0]}")
        
        # ensure all dimensions match
        assert n_total == expected_total, f"Total points mismatch: {n_total} != {expected_total}"
        assert self.visibility_filter_all.shape[0] == n_total, f"visibility_filter_all size mismatch: {self.visibility_filter_all.shape[0]} != {n_total}"
        # fix: after merge all points are trainable, so is_sky_filter should equal n_total
        assert self.is_sky_filter.shape[0] == n_total, f"is_sky_filter size mismatch: {self.is_sky_filter.shape[0]} != {n_total}"
        assert self.delete_mask_all.shape[0] == n_total, f"delete_mask_all size mismatch: {self.delete_mask_all.shape[0]} != {n_total}"
        print("✅ All dimensions verified correctly!")
        
        # final check to ensure global mask dimension consistency
        self._ensure_global_mask_consistency()

    
    def merge_gaussian_with_trainability_control(self, previous_gaussian, auto_freeze_labels=None, auto_trainable_labels=None):
        """
        Smart merge: automatically control trainability based on labels

        Args:
            previous_gaussian: GaussianModel to merge
            auto_freeze_labels: list of labels to auto-freeze after merge, None means no auto-freeze
            auto_trainable_labels: list of labels allowed to train after merge, None means no restriction
        """
        print(f"🧠 Smart merging gaussian with trainability control...")
        
        # perform normal merge first
        self.merge_gaussian(previous_gaussian)
        
        # clear optimizer to avoid parameter mismatch issues, will be reset later
        if hasattr(self, 'optimizer'):
            print("🔧 Clearing optimizer after merge (will be reset later)...")
            self.optimizer = None
        
        # then apply trainability control
        if auto_freeze_labels is not None:
            print(f"🧊 Auto-freezing labels: {auto_freeze_labels}")
            self.freeze_labels(*auto_freeze_labels)
        elif auto_trainable_labels is not None:
            print(f"🎯 Auto-setting trainable labels: {auto_trainable_labels}")
            self.train_only_labels(*auto_trainable_labels)
        
        print(f"✅ Smart merge complete with trainability control applied")

    
    # return self
    # def __init__(self, sh_degree : int, previous_gaussian=None, floater_dist2_threshold=0.0002):
    #     """
    #     args:
    #         previous_gaussian : GaussianModel; We take all of its 3DGS particles, freeze them and use them for rendering only.
    #     """
    #     self.active_sh_degree = 0
    #     self.max_sh_degree = sh_degree  
    #     self._xyz = torch.empty(0).cuda()
    #     self._features_dc = torch.empty(0).cuda()
    #     self._scaling = torch.empty(0).cuda()
    #     self._rotation = torch.empty(0).cuda()
    #     self._opacity = torch.empty(0).cuda()
    #     self.max_radii2D = torch.empty(0).cuda()
    #     self.xyz_gradient_accum = torch.empty(0).cuda()
    #     self.denom = torch.empty(0).cuda()
    #     self.optimizer = None
    #     self.percent_dense = 0
    #     self.spatial_lr_scale = 1.
    #     self.floater_dist2_threshold = floater_dist2_threshold
    #     self._focal_length = torch.empty(0).cuda()  # new: record each point's focal_length
    #     self.setup_functions()

    #     if previous_gaussian is not None:
    #         # transfer model parameters
    #         self._xyz_prev = torch.cat([previous_gaussian._xyz.detach(), previous_gaussian._xyz_prev], dim=0)
    #         self._features_dc_prev = torch.cat([previous_gaussian._features_dc.detach(), previous_gaussian._features_dc_prev], dim=0)
    #         self._scaling_prev = torch.cat([previous_gaussian._scaling.detach(), previous_gaussian._scaling_prev], dim=0)
    #         self._rotation_prev = torch.cat([previous_gaussian._rotation.detach(), previous_gaussian._rotation_prev], dim=0)
    #         self._opacity_prev = torch.cat([previous_gaussian._opacity.detach(), previous_gaussian._opacity_prev], dim=0)
    #         if hasattr(previous_gaussian, 'filter_3D'):
    #             self.filter_3D_prev = torch.cat((previous_gaussian.filter_3D.detach(), previous_gaussian.filter_3D_prev), dim=0)
    #         else:
    #             self.filter_3D_prev = torch.empty(0).cuda()
    #         self._focal_length_prev = torch.cat([previous_gaussian._focal_length.detach(), previous_gaussian._focal_length_prev], dim=0)
    #         # imor
    #         # print("focal_length_prev",self._focal_length_prev.shape)
    #         # only transfer training variables after training has started
    #         if hasattr(previous_gaussian, 'max_radii2D') and previous_gaussian.max_radii2D.numel() > 0:
    #             self.max_radii2D_prev = torch.cat([previous_gaussian.max_radii2D.detach(), getattr(previous_gaussian, 'max_radii2D_prev', torch.empty(0).cuda())], dim=0)
    #             self.xyz_gradient_accum_prev = torch.cat([previous_gaussian.xyz_gradient_accum.detach(), getattr(previous_gaussian, 'xyz_gradient_accum_prev', torch.empty(0, 1).cuda())], dim=0)
    #             self.denom_prev = torch.cat([previous_gaussian.denom.detach(), getattr(previous_gaussian, 'denom_prev', torch.empty(0, 1).cuda())], dim=0)
    #         else:
    #             self.max_radii2D_prev = torch.empty(0).cuda()
    #             self.xyz_gradient_accum_prev = torch.empty(0, 1).cuda()
    #             self.denom_prev = torch.empty(0, 1).cuda()
            
    #         # transfer global masks
    #         self.visibility_filter_all = previous_gaussian.visibility_filter_all
    #         self.is_sky_filter = previous_gaussian.is_sky_filter
    #         self.delete_mask_all = previous_gaussian.delete_mask_all
    #     else:
    #         # initialize as empty
    #         self._xyz_prev = torch.empty(0).cuda()
    #         self._features_dc_prev = torch.empty(0).cuda()
    #         self._scaling_prev = torch.empty(0).cuda()
    #         self._rotation_prev = torch.empty(0).cuda()
    #         self._opacity_prev = torch.empty(0).cuda()
    #         self.filter_3D_prev = torch.empty(0).cuda()
    #         self._focal_length_prev = torch.empty(0).cuda()
    #         self.max_radii2D_prev = torch.empty(0).cuda()
    #         self.xyz_gradient_accum_prev = torch.empty(0, 1).cuda()
    #         self.denom_prev = torch.empty(0, 1).cuda()
    #         self.visibility_filter_all = torch.empty(0, dtype=torch.bool).cuda()
    #         self.is_sky_filter = torch.empty(0, dtype=torch.bool).cuda()
    #         self.delete_mask_all = torch.empty(0, dtype=torch.bool).cuda()

    
    def _check_consistency(self):
        """Check consistency of all tensor dimensions"""
        n_current = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0]
        n_total = n_current + n_prev
        
        # check basic model parameters
        assert self._focal_length.shape[0] == n_current, f"focal_length dimension error: {self._focal_length.shape[0]} vs {n_current}"

        # fix: only check training-related variables after training has started
        if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.numel() > 0:
            assert self.max_radii2D.shape[0] == n_current, f"max_radii2D dimension error: {self.max_radii2D.shape[0]} vs {n_current}"
            # assert self.xyz_gradient_accum.shape[0] == n_current, f"xyz_gradient_accum dimension error: {self.xyz_gradient_accum.shape[0]} vs {n_current}"
            # assert self.denom.shape[0] == n_current, f"denom dimension error: {self.denom.shape[0]} vs {n_current}"
        
        # check prev-related data
        if n_prev > 0:
            assert self._focal_length_prev.shape[0] == n_prev, f"focal_length_prev dimension error: {self._focal_length_prev.shape[0]} vs {n_prev}"
        
        # check global data
        assert self.visibility_filter_all.shape[0] == n_total, f"visibility_filter_all dimension error: {self.visibility_filter_all.shape[0]} vs {n_total}"
        assert self.delete_mask_all.shape[0] == n_total, f"delete_mask_all dimension error: {self.delete_mask_all.shape[0]} vs {n_total}"
        assert self.get_sky_filter_all.shape[0] == n_total, f"is_sky_filter dimension error: {self.get_sky_filter_all.shape[0]} vs {n_total}"

    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_sky_filter_all(self):
        return torch.cat([self.is_sky_filter.to("cuda"), self.is_sky_filter_prev.to("cuda")], dim=0)
    
    @property
    def get_point_labels_all(self):
        return torch.cat([self.point_labels, self.point_labels_prev], dim=0)
    

    
    @property
    def get_next_scale_all(self):
        return torch.cat([self.next_scale, self.next_scale_prev], dim=0)
    
    @property
    def get_prior_scale_all(self):
        return torch.cat([self.prior_scale, self.prior_scale_prev], dim=0)
    
    @property
    def get_now_scale_all(self):
        return torch.cat([self.now_scale, self.now_scale_prev], dim=0)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_focal_length_all(self):
        return torch.cat([self._focal_length, self._focal_length_prev], dim=0)
    
    @property
    def get_is_sky_filter_all(self):
        """Get sky filter of all points (trainable + non-trainable) - redirect to main function"""
        return self.get_sky_filter_all

    @property 
    def get_point_labels_all_v2(self):
        """Get labels of all points (trainable + non-trainable) - redirect to main function"""
        return self.get_point_labels_all

    @property
    def get_focal_length(self):
        return self._focal_length
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        return features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


    @property
    def get_scaling_all(self):
        # return self.scaling_activation(self._scaling)
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_prev], dim=0))
    
    # @property
    # def get_scaling_with_3D_filter_all(self):
    #     # scales = self.get_scaling
    #     scales = self.get_scaling_all
        
    #     # scales = torch.square(scales) + torch.square(self.filter_3D)
    #     scales = torch.square(scales) + torch.square(torch.cat([self.filter_3D, self.filter_3D_prev], dim=0))
    #     scales = torch.sqrt(scales)
    #     return scales
    @property
    def get_scaling_with_3D_filter_all(self):
        scales = self.get_scaling_all
        
        # add size check and protection
        try:
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                filter_all = torch.cat([self.filter_3D, self.filter_3D_prev], dim=0)
                
                # emergency handling when sizes don't match
                if scales.shape[0] != filter_all.shape[0]:
                    print(f"WARNING: Scale/Filter size mismatch! scales: {scales.shape[0]}, filter: {filter_all.shape[0]}")
                    # recreate matching filter
                    filter_all = torch.ones(scales.shape[0], 1, device=scales.device) * 0.001
                
                scales = torch.square(scales) + torch.square(filter_all)
                scales = torch.sqrt(scales)
            # if no filter_3D, return original scales directly
        except Exception as e:
            print(f"Error in get_scaling_with_3D_filter_all: {e}")
            # return original scales on error
            pass
        
        return scales
    
    @property
    def get_rotation_all(self):
        # return self.rotation_activation(self._rotation)
        return self.rotation_activation(torch.cat([self._rotation, self._rotation_prev], dim=0))
    
    @property
    def get_xyz_all(self):
        # return self._xyz
        return torch.cat([self._xyz, self._xyz_prev], dim=0)
    
    @property
    def get_features_all(self):
        # features_dc = self._features_dc
        # ensure dimension consistency: convert all to [N, 1, 3] format
        current_features = self._features_dc
        if len(current_features.shape) == 3 and current_features.shape[1] == 3 and current_features.shape[2] == 1:
            # if [N, 3, 1], convert to [N, 1, 3]
            current_features = current_features.permute(0, 2, 1)
        
        prev_features = self._features_dc_prev
        if len(prev_features.shape) == 3 and prev_features.shape[1] == 3 and prev_features.shape[2] == 1:
            # if [N, 3, 1], convert to [N, 1, 3]
            prev_features = prev_features.permute(0, 2, 1)
        
        features_dc = torch.cat([current_features, prev_features], dim=0)
        return features_dc
    
    @property
    def get_opacity_all(self):
        # return self.opacity_activation(self._opacity)
        return self.opacity_activation(torch.cat([self._opacity, self._opacity_prev], dim=0))
    
    @property
    def get_opacity_with_3D_filter_all(self):
        opacity = self.get_opacity_all
        
        try:
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                scales = self.get_scaling_all
                filter_all = torch.cat([self.filter_3D, self.filter_3D_prev], dim=0)
                
                # size check
                if scales.shape[0] != filter_all.shape[0]:
                    print(f"WARNING: Scale/Filter size mismatch in opacity calculation!")
                    return opacity  # return original opacity directly
                
                scales_square = torch.square(scales)
                det1 = scales_square.prod(dim=1)
                
                scales_after_square = scales_square + torch.square(filter_all)
                det2 = scales_after_square.prod(dim=1) 
                coef = torch.sqrt(det1 / det2)
                return opacity * coef[..., None]
        except Exception as e:
            print(f"Error in get_opacity_with_3D_filter_all: {e}")
            pass
        
        return opacity
    
    # @property
    # def get_opacity_with_3D_filter_all(self):
    #     # opacity = self.opacity_activation(self._opacity)
    #     opacity = self.get_opacity_all
    #     # apply 3D filter
    #     # scales = self.get_scaling
    #     scales = self.get_scaling_all
        
    #     scales_square = torch.square(scales)
    #     det1 = scales_square.prod(dim=1)
        
    #     # scales_after_square = scales_square + torch.square(self.filter_3D) 
    #     scales_after_square = scales_square + torch.square(torch.cat([self.filter_3D, self.filter_3D_prev], dim=0))
    #     det2 = scales_after_square.prod(dim=1) 
    #     coef = torch.sqrt(det1 / det2)
    #     return opacity * coef[..., None]

    def get_covariance_all(self, scaling_modifier = 1):
        # return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        return self.covariance_activation(self.get_scaling_all, scaling_modifier, torch.cat([self._rotation, self._rotation_prev], dim=0))


    @torch.no_grad()
    def compute_3D_filter(self, cameras, initialize_scaling=False):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        # focal_length = torch.empty()
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            # xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], other=z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x

            screen_normal = torch.tensor([[0, 0, -1]], device=xyz.device, dtype=torch.float32)
            point_normals_in_screen = rotation2normal(self.get_rotation) @ R
            point_normals_in_screen_xoz = F.normalize(point_normals_in_screen[:, [0, 2]], dim=1)
            screen_normal_xoz = F.normalize(screen_normal[:, [0, 2]], dim=1)
            cos_xz = torch.sum(point_normals_in_screen_xoz * screen_normal_xoz, dim=1)
            # assert torch.all(cos_xz >= 0), "All normals should be in the same direction of the screen normal. Current min value: {}".format(cos_xz.min())
            point_normals_in_screen_yoz = F.normalize(point_normals_in_screen[:, [1, 2]], dim=1)
            screen_normal_yoz = F.normalize(screen_normal[:, [1, 2]], dim=1)
            cos_yz = torch.sum(point_normals_in_screen_yoz * screen_normal_yoz, dim=1)
            # assert torch.all(cos_yz >= 0), "All normals should be in the same direction of the screen normal. Current min value: {}".format(cos_yz.min())
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length
        self.filter_3D = filter_3D[..., None]

        x_scale = distance / focal_length / cos_xz.clamp(min=1e-1)
        y_scale = distance / focal_length / cos_yz.clamp(min=1e-1)

        if initialize_scaling:
            print('Initializing scaling...')
            dist_scales = torch.exp(self._scaling)
            nyquist_scales = self.filter_3D.clone().repeat(1, 3)
            nyquist_scales[:, 0:1] = x_scale[..., None]
            nyquist_scales[:, 1:2] = y_scale[..., None]
            nyquist_scales *= 0.7
            scaling = torch.log(nyquist_scales)
            # scaling[:, 2] = torch.log(torch.tensor(0))
            # mixed_scales = (dist_scales * nyquist_scales).sqrt()
            # scaling = torch.log(mixed_scales)
            optimizable_tensors = self.replace_tensor_to_optimizer(scaling, 'scaling')
            self._scaling = optimizable_tensors['scaling']
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1



    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, focal_length: torch.Tensor, is_sky: torch.Tensor, new_now_scale: torch.Tensor):
        # ───────────────────── prepare point data ─────────────────────
        points = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        
        dist2 = torch.clamp_min(distCUDA2(points), 1e-7)
        floater_mask = dist2 > self.floater_dist2_threshold
        dist2 = dist2[~floater_mask]
        fused_point_cloud = points[~floater_mask]
        focal_length = focal_length[~floater_mask]

        colors = torch.from_numpy(np.asarray(pcd.colors)).float().cuda()
        fused_color = self.inverse_color_activation((colors * 1.01).clamp(0, 1))[~floater_mask]

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Floater ratio: {:.2f}%".format(floater_mask.float().mean().item() * 100))
        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        self.spatial_lr_scale = spatial_lr_scale
        normals = torch.from_numpy(pcd.normals).float().cuda()
        rots = normal2rotation(normals)[~floater_mask]

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales[:, 2] = torch.tensor(float('-inf'), device="cuda")  # special handling for z direction

        opacities = self.inverse_opacity_activation(
            0.15 * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
        )


        # ───────────────────── focal_length ─────────────────────
        focal_length_tensor = torch.tensor(focal_length, device="cuda")

        assert focal_length_tensor.shape[0] == fused_point_cloud.shape[0], "focal_length does not match point count"

        # ───────────────────── add point data ─────────────────────
        if self._xyz.numel() == 0:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self._focal_length = focal_length_tensor
        else:
            print(f"Adding these points to the existing model with {self.get_xyz.shape[0]} points")
            # record old current point count before updating _xyz
            n_old_current = self.get_xyz.shape[0]
            
            self._xyz = nn.Parameter(torch.cat([self._xyz, fused_point_cloud], dim=0).requires_grad_(True))
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, features[:, :, 0:1].transpose(1, 2)], dim=0).requires_grad_(True))
            self._scaling = nn.Parameter(torch.cat([self._scaling, scales], dim=0).requires_grad_(True))
            self._rotation = nn.Parameter(torch.cat([self._rotation, rots], dim=0).requires_grad_(True))
            self._opacity = nn.Parameter(torch.cat([self._opacity, opacities], dim=0).requires_grad_(True))
            self._focal_length = torch.cat([self._focal_length, focal_length_tensor], dim=0)

        n_new = fused_point_cloud.shape[0]
        
        # handle first creation case
        if self._xyz.shape[0] == n_new:  # first creation, no old points
            n_old_current = 0

        # ───────────────────── mask concatenation (curr+prev) ─────────────────────
        # fix: keep consistent with _xyz layout [old current, new points, old prev]
        if n_old_current > 0:
            # split old global masks
            old_current_vis = self.visibility_filter_all[:n_old_current]
            old_prev_vis = self.visibility_filter_all[n_old_current:]
            old_current_del = self.delete_mask_all[:n_old_current]  
            old_prev_del = self.delete_mask_all[n_old_current:]
            
            # reorganize: [old current, new points, old prev]
            self.visibility_filter_all = torch.cat([
                old_current_vis,                                           # old current
                torch.ones(n_new, dtype=torch.bool, device='cuda'),       # new points
                old_prev_vis                                               # old prev
            ])
            self.delete_mask_all = torch.cat([
                old_current_del,                                           # old current
                torch.zeros(n_new, dtype=torch.bool, device='cuda'),      # new points
                old_prev_del                                               # old prev
            ])
        else:
            # if no old current points, add new points directly to front
            self.visibility_filter_all = torch.cat([
                torch.ones(n_new, dtype=torch.bool, device='cuda'),  # new current points
                self.visibility_filter_all  # existing prev
            ])
            self.delete_mask_all = torch.cat([
                torch.zeros(n_new, dtype=torch.bool, device='cuda'),  # new current points
                self.delete_mask_all  # existing prev
            ])
        # fix: is_sky_filter also needs to be consistent with _xyz layout
        if n_old_current > 0:
            # split old is_sky_filter
            old_current_sky = self.is_sky_filter[:n_old_current]
            old_prev_sky = self.is_sky_filter[n_old_current:]
            
            # reorganize: [old current, new points, old prev]
            new_sky = is_sky.squeeze().to(old_current_sky.device)
            self.is_sky_filter = torch.cat([
                old_current_sky,        # old current
                new_sky,               # new points
                old_prev_sky           # old prev
            ])
        else:
            # if no old current points, add new points directly to front
            new_sky = is_sky.squeeze().to(self.is_sky_filter.device)
            if self.is_sky_filter.numel() > 0:
                self.is_sky_filter = torch.cat([new_sky, self.is_sky_filter])
            else:
                self.is_sky_filter = new_sky

        # ───────────────────── point label handling (current/prev separation mode) ─────────────────────
        # new points default to main label (ID=0), only added to current trainable points
        new_labels = torch.zeros(n_new, dtype=torch.long, device='cuda')
        if self.point_labels.numel() == 0:
            # first creation, set directly
            self.point_labels = new_labels
        else:
            # add to existing current trainable point labels
            self.point_labels = torch.cat([self.point_labels, new_labels])
        
        # ensure point_labels_prev exists (new points are all trainable, does not affect prev)
        if not hasattr(self, 'point_labels_prev'):
            self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda')


        
        # ───────────────────── next_scale, prior_scale and now_scale handling ─────────────────────
        # initialize next_scale and prior_scale for new points
        new_next_scale = torch.full((n_new,), float('inf'), device='cuda')
        new_prior_scale = torch.full((n_new,), float('-inf'), device='cuda')
        
        # initialize now_scale for new points (focal length/distance)
        # compute distance from points to camera, then use focal_length/distance as now_scale
        
        if self.next_scale.numel() == 0:
            # first creation, set directly
            self.next_scale = new_next_scale
            self.prior_scale = new_prior_scale
            self.now_scale = new_now_scale
        else:
            # add to existing scales
            self.next_scale = torch.cat([self.next_scale, new_next_scale]).detach()
            self.prior_scale = torch.cat([self.prior_scale, new_prior_scale]).detach()
            self.now_scale = torch.cat([self.now_scale, new_now_scale]).detach()

        # ───────────────────── max_radii initialization ─────────────────────
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")

        # ───────────────────── Sanity Check ─────────────────────
        
        self._check_consistency()


    @torch.no_grad()
    def get_inscreen_points(self, tdgs_cam):
        if isinstance(tdgs_cam, PerspectiveCameras):
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(tdgs_cam, config=self.config)
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here

        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen_x = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        in_screen_y = torch.logical_and(y >= 0, y < tdgs_cam.image_height)
        in_screen = torch.logical_and(in_screen_x, in_screen_y)

        return in_screen

    @torch.no_grad()
    def delete_points(self, tdgs_cam):
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here

        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen_x = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        in_screen_y = torch.logical_and(y >= 0, y < tdgs_cam.image_height)
        in_screen = torch.logical_and(in_screen_x, in_screen_y)
        
        # delete_mask = torch.logical_and(in_screen, ~self.is_sky_filter.to(xyz.device))
        delete_mask = in_screen
        self.delete_mask_all = self.delete_mask_all | delete_mask


    @torch.no_grad()
    def set_visible_and_restore_from_prev(self, camera, opt):
        """
        Under the current camera viewpoint:
        1. Set on-screen points among current points as visible;
        2. Restore visible historical points back to the training set;
        3. Remove these points from the historical pool to prevent duplication.
        """
        from gaussian_renderer import render

        render_pkg = render(camera, self, opt, bg_color=torch.tensor([0.7, 0.7, 0.7], device='cuda'), render_visible=True)
        visible_mask = render_pkg["visibility_filter"]

        n_total = self.get_xyz_all.shape[0]
        n_curr = self.get_xyz.shape[0]
        n_prev = n_total - n_curr

        visible_curr = visible_mask[:n_curr]
        visible_prev = visible_mask[n_curr:]

        # fix: set current visible points as trainable (based on [current, prev] order)
        self.visibility_filter_all[:n_curr] |= visible_curr
        self.visibility_filter = self.visibility_filter_all[:n_curr]

        if self._xyz_prev.shape[0] == 0:
            return

        indices_prev = torch.where(visible_prev)[0]
        
        if len(indices_prev) > 0:
            # === restore points from prev ===
            xyz_restore = self._xyz_prev[indices_prev]
            feat_dc_restore = self._features_dc_prev[indices_prev]
            scale_restore = self._scaling_prev[indices_prev]
            rot_restore = self._rotation_prev[indices_prev]
            opacity_restore = self._opacity_prev[indices_prev]
            focal_restore = self._focal_length_prev[indices_prev]
            
            # restore max_radii2D and filter_3D
            max_radii2D_restore = self.max_radii2D_prev[indices_prev]
            filter_3D_restore = self.filter_3D_prev[indices_prev]

            # === concatenate back to current trainable set ===
            self._xyz = nn.Parameter(torch.cat([self._xyz, xyz_restore], dim=0).requires_grad_(True))
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, feat_dc_restore], dim=0).requires_grad_(True))
            self._scaling = nn.Parameter(torch.cat([self._scaling, scale_restore], dim=0).requires_grad_(True))
            self._rotation = nn.Parameter(torch.cat([self._rotation, rot_restore], dim=0).requires_grad_(True))
            self._opacity = nn.Parameter(torch.cat([self._opacity, opacity_restore], dim=0).requires_grad_(True))
            self._focal_length = torch.cat([self._focal_length, focal_restore], dim=0)
            
            # restore max_radii2D and filter_3D
            self.max_radii2D = torch.cat([self.max_radii2D, max_radii2D_restore], dim=0)
            self.filter_3D = torch.cat([self.filter_3D, filter_3D_restore], dim=0)

            # FIX: restore scale tensors
            if hasattr(self, 'next_scale_prev') and self.next_scale_prev.numel() > 0:
                next_scale_restore = self.next_scale_prev[indices_prev]
                self.next_scale = torch.cat([self.next_scale, next_scale_restore]).detach()
            else:
                # if no prev next_scale, use default values
                next_scale_restore = torch.full((n_new,), float('inf'), device='cuda')
                self.next_scale = torch.cat([self.next_scale, next_scale_restore]).detach()
                
            if hasattr(self, 'prior_scale_prev') and self.prior_scale_prev.numel() > 0:
                prior_scale_restore = self.prior_scale_prev[indices_prev]
                self.prior_scale = torch.cat([self.prior_scale, prior_scale_restore]).detach()
            else:
                # if no prev prior_scale, use default values
                prior_scale_restore = torch.full((n_new,), float('-inf'), device='cuda')
                self.prior_scale = torch.cat([self.prior_scale, prior_scale_restore]).detach()
                
            if hasattr(self, 'now_scale_prev') and self.now_scale_prev.numel() > 0:
                now_scale_restore = self.now_scale_prev[indices_prev]
                self.now_scale = torch.cat([self.now_scale, now_scale_restore]).detach()
            else:
                # if no prev now_scale, use default values
                now_scale_restore = torch.full((n_new,), 0.01, device='cuda')
                self.now_scale = torch.cat([self.now_scale, now_scale_restore]).detach()

            n_new = len(indices_prev)

            new_vis = torch.ones(n_new, dtype=torch.bool, device='cuda')
            new_del = torch.zeros(n_new, dtype=torch.bool, device='cuda')
            new_sky = torch.zeros(n_new, dtype=torch.bool, device='cuda')

            # create correct boolean mask
            keep_mask_prev = torch.ones(n_prev, dtype=torch.bool, device='cuda')
            keep_mask_prev[indices_prev] = False

            # correct concat order [current, prev]
            self.visibility_filter_all = torch.cat([
                self.visibility_filter_all[:n_curr],                    # original current
                new_vis,                                                # restored points (new current)
                self.visibility_filter_all[n_curr:][keep_mask_prev]     # remaining prev
            ])
            self.delete_mask_all = torch.cat([
                self.delete_mask_all[:n_curr],                          # original current
                new_del,                                                # restored points (new current)
                self.delete_mask_all[n_curr:][keep_mask_prev]           # remaining prev
            ])
            self.is_sky_filter = torch.cat([
                self.is_sky_filter[:n_curr],                            # original current
                new_sky,                                                # restored points (new current)
                self.is_sky_filter[n_curr:][keep_mask_prev]             # remaining prev
            ])
            
            # restore point_labels - using current/prev separation mode
            if hasattr(self, 'point_labels_prev') and self.point_labels_prev.numel() > 0:
                # get labels of restored points from prev labels
                restored_labels = self.point_labels_prev[indices_prev]
                
                # add to current labels
                self.point_labels = torch.cat([self.point_labels, restored_labels])
                
                # remove restored points from prev labels
                self.point_labels_prev = self.point_labels_prev[keep_mask_prev]

            # remove restored points from prev
            keep_prev = torch.ones(self._xyz_prev.shape[0], dtype=torch.bool, device='cuda')
            keep_prev[indices_prev] = False
            self._xyz_prev = self._xyz_prev[keep_prev]
            self._features_dc_prev = self._features_dc_prev[keep_prev]
            self._scaling_prev = self._scaling_prev[keep_prev]
            self._rotation_prev = self._rotation_prev[keep_prev]
            self._opacity_prev = self._opacity_prev[keep_prev]
            self._focal_length_prev = self._focal_length_prev[keep_prev]
            
            # remove corresponding max_radii2D_prev and filter_3D_prev
            self.max_radii2D_prev = self.max_radii2D_prev[keep_prev]
            self.filter_3D_prev = self.filter_3D_prev[keep_prev]
            
            # FIX: remove corresponding scale tensors
            if hasattr(self, 'next_scale_prev') and self.next_scale_prev.numel() > 0:
                self.next_scale_prev = self.next_scale_prev[keep_prev]
            if hasattr(self, 'prior_scale_prev') and self.prior_scale_prev.numel() > 0:
                self.prior_scale_prev = self.prior_scale_prev[keep_prev]
            if hasattr(self, 'now_scale_prev') and self.now_scale_prev.numel() > 0:
                self.now_scale_prev = self.now_scale_prev[keep_prev]

                    # ---- final safety check ----
        self._check_consistency()
    
    def _ensure_global_mask_consistency(self):
        """Ensure all global mask dimensions match the total point count"""
        n_curr = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0] if self._xyz_prev.numel() > 0 else 0
        n_total = n_curr + n_prev
        
        # check and fix each global mask
        if self.visibility_filter_all.shape[0] != n_total:
            print(f"🔧 Fixing visibility_filter_all: {self.visibility_filter_all.shape[0]} -> {n_total}")
            if self.visibility_filter_all.shape[0] < n_total:
                # expand mask
                missing = n_total - self.visibility_filter_all.shape[0]
                extra = torch.ones(missing, dtype=torch.bool, device=self.visibility_filter_all.device)
                self.visibility_filter_all = torch.cat([self.visibility_filter_all, extra])
            else:
                # truncate mask
                self.visibility_filter_all = self.visibility_filter_all[:n_total]
        
        # important fix: is_sky_filter only corresponds to current points, should not be adjusted to n_total!
        if self.is_sky_filter.shape[0] != n_curr:
            print(f"🔧 Fixing is_sky_filter: {self.is_sky_filter.shape[0]} -> {n_curr} (current points only)")
            if self.is_sky_filter.shape[0] < n_curr:
                # expand mask to current points size
                missing = n_curr - self.is_sky_filter.shape[0]
                extra = torch.zeros(missing, dtype=torch.bool, device=self.is_sky_filter.device)
                self.is_sky_filter = torch.cat([self.is_sky_filter, extra])
            else:
                # truncate mask to current points size
                self.is_sky_filter = self.is_sky_filter[:n_curr]
        
        if self.delete_mask_all.shape[0] != n_total:
            print(f"🔧 Fixing delete_mask_all: {self.delete_mask_all.shape[0]} -> {n_total}")
            if self.delete_mask_all.shape[0] < n_total:
                # expand mask
                missing = n_total - self.delete_mask_all.shape[0]
                extra = torch.zeros(missing, dtype=torch.bool, device=self.delete_mask_all.device)
                self.delete_mask_all = torch.cat([self.delete_mask_all, extra])
            else:
                # truncate mask
                self.delete_mask_all = self.delete_mask_all[:n_total]
        
        # fix point_labels - using current/prev separation mode
        if hasattr(self, 'point_labels'):
            expected_current = n_curr
            expected_prev = n_prev
            
            # check current labels
            if self.point_labels.shape[0] != expected_current:
                print(f"🔧 Fixing point_labels (current): {self.point_labels.shape[0]} -> {expected_current}")
                if self.point_labels.shape[0] < expected_current:
                    # expand current labels
                    missing = expected_current - self.point_labels.shape[0]
                    extra = torch.zeros(missing, dtype=torch.long, device=self.point_labels.device)
                    self.point_labels = torch.cat([self.point_labels, extra])
                else:
                    # truncate current labels
                    self.point_labels = self.point_labels[:expected_current]
            
            # check prev labels
            if hasattr(self, 'point_labels_prev'):
                if self.point_labels_prev.shape[0] != expected_prev:
                    print(f"🔧 Fixing point_labels_prev: {self.point_labels_prev.shape[0]} -> {expected_prev}")
                    if self.point_labels_prev.shape[0] < expected_prev:
                        # expand prev labels
                        missing = expected_prev - self.point_labels_prev.shape[0]
                        extra = torch.zeros(missing, dtype=torch.long, device=self.point_labels_prev.device)
                        self.point_labels_prev = torch.cat([self.point_labels_prev, extra])
                    else:
                        # truncate prev labels
                        self.point_labels_prev = self.point_labels_prev[:expected_prev]
            else:
                # create empty prev labels
                self.point_labels_prev = torch.empty(0, dtype=torch.long, device=self.point_labels.device)

    def convert_to_pcd(self, include_invisible=False):
        """
        Convert Gaussian model back to BasicPointCloud format

        Note: this is an "approximate" inverse conversion, not perfectly exact, because:
        1. Point positions, colors, and normals change during training
        2. Points filtered by floater_mask cannot be recovered
        3. Clamp operations may cause color information loss
        4. Scaling and opacity information cannot be converted back to original distances

        Args:
            include_invisible (bool): whether to include invisible points

        Returns:
            BasicPointCloud: point cloud data containing points, colors, normals (post-training state)
        """
        from utils.graphics import BasicPointCloud
        from utils.general import rotation2normal
        
        # get visible points mask
        if include_invisible:
            valid_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device='cuda')
        else:
            # only include visible and non-deleted points
            valid_mask = self.visibility_filter_all & (~self.delete_mask_all)
        
        # extract 3D coordinate points
        points = self.get_xyz[valid_mask].detach().cpu().numpy()
        
        # recover colors - fully reverse create_from_pcd's color conversion
        features_dc = self.get_features[:, :, 0][valid_mask]  # [N, 3]
        colors_activated = self.color_activation(features_dc).detach().cpu().numpy()
        # reverse the (colors * 1.01).clamp(0, 1) operation
        colors = colors_activated / 1.01
        # note: due to clamp operations, some color information may have been lost
        
        # recover normals - convert back from quaternion rotation to normals
        rotations = self.get_rotation[valid_mask]  # [N, 4] quaternions
        normals = rotation2normal(rotations).detach().cpu().numpy()
        
        print(f"Converted {points.shape[0]} points to PCD")
        print(f"  Points shape: {points.shape}")
        print(f"  Colors shape: {colors.shape}, range: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"  Normals shape: {normals.shape}")
        
        return BasicPointCloud(
            points=points,
            colors=colors, 
            normals=normals
        )
    
    def save_as_ply(self, path, include_invisible=False):
        """
        Save Gaussian model as PLY format point cloud file

        Note: this is a post-training approximate point cloud, not the original input PCD

        Args:
            path (str): save path
            include_invisible (bool): whether to include invisible points
        """
        pcd = self.convert_to_pcd(include_invisible)
        
        # build PLY file content
        header = f"""ply
format ascii 1.0
element vertex {len(pcd.points)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
"""
        
        # convert colors from [0,1] to [0,255]
        colors_255 = (pcd.colors * 255).astype(np.uint8)
        
        with open(path, 'w') as f:
            f.write(header)
            for i in range(len(pcd.points)):
                x, y, z = pcd.points[i]
                nx, ny, nz = pcd.normals[i]
                r, g, b = colors_255[i]
                f.write(f"{x} {y} {z} {nx} {ny} {nz} {r} {g} {b}\n")
        
                print(f"Saved {len(pcd.points)} points to {path}")

    def gaussian2pytorch3d(self, xyz_scale=1000.0, include_invisible=True):
        """
        Convert Gaussian model to PyTorch3D Pointclouds format

        Args:
            xyz_scale (float): xyz coordinate scale factor for restoring to original scale
            include_invisible (bool): whether to include invisible points

        Returns:
            pytorch3d.structures.Pointclouds: PyTorch3D point cloud object
        """
        from pytorch3d.structures import Pointclouds
        from utils.general import rotation2normal
        
        # get valid points mask
        if include_invisible:
            valid_mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device='cuda')
        else:
            valid_mask = self.visibility_filter_all & (~self.delete_mask_all)
        
        # extract coordinates and restore scale
        points = self.get_xyz[valid_mask] / xyz_scale  # [N, 3] restore to original scale
        
        # recover colors (inverse conversion)
        features_dc = self.get_features[valid_mask, 0, :]  # [N, 3] RGB three channels
        colors = self.color_activation(features_dc) / 1.01  # reverse * 1.01 operation
        
        # recover normals
        rotations = self.get_rotation[valid_mask]  # [N, 4] quaternions
        # normals = rotation2normal(rotations)  # [N, 3]
        
        # create PyTorch3D point cloud
        pointcloud = Pointclouds(
            points=[points],
            features=[colors], 
            # normals=[normals]
        )
        
        return pointcloud
                         
    @torch.no_grad()
    def set_inscreen_points_to_visible(self, tdgs_cam):
        xyz = self.get_xyz_all
        R = torch.tensor(tdgs_cam.R, device=xyz.device, dtype=torch.float32)
        T = torch.tensor(tdgs_cam.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here

        xyz_cam = xyz @ R + T[None, :]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * tdgs_cam.focal_x + tdgs_cam.image_width / 2.0
        y = y / z * tdgs_cam.focal_y + tdgs_cam.image_height / 2.0
        
        in_screen = torch.logical_and(x >= 0, x < tdgs_cam.image_width)
        self.visibility_filter_all = self.visibility_filter_all | in_screen
        # return in_screen

    def my_load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(0., 0.99))
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        # save training_args for later optimizer re-setup
        self._last_training_args = training_args
        
        # apply any pending trainable_mask
        if hasattr(self, '_pending_trainable_mask'):
            print("🔧 Applying pending trainability control after training_setup...")
            pending_mask = self._pending_trainable_mask
            delattr(self, '_pending_trainable_mask')  # clear pending state
            self.set_trainable_mask(pending_mask)  # recursive call, optimizer exists this time

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False, use_higher_freq=True):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def delete_all_points(self, delete_mask, preserve_grad=True):
        """
        Simple function to delete all points (current + historical)

        Args:
            delete_mask: boolean mask of length n_current + n_prev, True means point to delete
            preserve_grad: whether to preserve gradient connections (set True when used during training)
        """
        # first ensure global mask dimension consistency
        self._ensure_global_mask_consistency()
        
        n_curr = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0] if self._xyz_prev.numel() > 0 else 0
        n_total = n_curr + n_prev
        
        assert len(delete_mask) == n_total, f"delete_mask length ({len(delete_mask)}) does not match total points ({n_total})"
        
        # decompose mask
        delete_curr = delete_mask[:n_curr]
        delete_prev = delete_mask[n_curr:] if n_prev > 0 else torch.empty(0, dtype=torch.bool, device=delete_mask.device)
        
        # ensure each mask is on the correct device
        if n_curr > 0:
            delete_curr = delete_curr.to(self._xyz.device)
        if n_prev > 0:
            delete_prev = delete_prev.to(self._xyz_prev.device)
        
        keep_curr = ~delete_curr
        keep_prev = ~delete_prev if n_prev > 0 else torch.empty(0, dtype=torch.bool, device=delete_mask.device)
        keep_all = ~delete_mask
        
        print(f"Deleting {delete_curr.sum().item()}/{n_curr} current points")
        if n_prev > 0:
            print(f"Deleting {delete_prev.sum().item()}/{n_prev} prev points")
        
        # === delete current points ===
        if delete_curr.any():
            # ensure keep_curr is on the same device as current point tensors
            keep_curr = keep_curr.to(self._xyz.device)
            
            if preserve_grad:
                # preserve gradient connections (used during training)
                self._xyz = nn.Parameter(self._xyz[keep_curr].clone().requires_grad_(True))
                self._features_dc = nn.Parameter(self._features_dc[keep_curr].clone().requires_grad_(True))
                self._opacity = nn.Parameter(self._opacity[keep_curr].clone().requires_grad_(True))
                self._scaling = nn.Parameter(self._scaling[keep_curr].clone().requires_grad_(True))
                self._rotation = nn.Parameter(self._rotation[keep_curr].clone().requires_grad_(True))
            else:
                # disconnect gradient connections (used outside training, saves memory)
                self._xyz = nn.Parameter(self._xyz[keep_curr].detach().requires_grad_(True))
                self._features_dc = nn.Parameter(self._features_dc[keep_curr].detach().requires_grad_(True))
                self._opacity = nn.Parameter(self._opacity[keep_curr].detach().requires_grad_(True))
                self._scaling = nn.Parameter(self._scaling[keep_curr].detach().requires_grad_(True))
                self._rotation = nn.Parameter(self._rotation[keep_curr].detach().requires_grad_(True))
            
            # current points' auxiliary data
            self.xyz_gradient_accum = self.xyz_gradient_accum[keep_curr]
            self.denom = self.denom[keep_curr]
            self.max_radii2D = self.max_radii2D[keep_curr]
            self._focal_length = self._focal_length[keep_curr]
            
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                self.filter_3D = self.filter_3D[keep_curr]
        
        # === delete historical points ===
        if n_prev > 0 and delete_prev.any():
            # ensure keep_prev is on the same device as historical point tensors
            keep_prev = keep_prev.to(self._xyz_prev.device)
            
            self._xyz_prev = self._xyz_prev[keep_prev]
            self._features_dc_prev = self._features_dc_prev[keep_prev]
            self._scaling_prev = self._scaling_prev[keep_prev]
            self._rotation_prev = self._rotation_prev[keep_prev]
            self._opacity_prev = self._opacity_prev[keep_prev]
            self._focal_length_prev = self._focal_length_prev[keep_prev]
            
            # historical points' auxiliary data
            if hasattr(self, 'max_radii2D_prev') and self.max_radii2D_prev.numel() > 0:
                self.max_radii2D_prev = self.max_radii2D_prev[keep_prev]
            if hasattr(self, 'xyz_gradient_accum_prev') and self.xyz_gradient_accum_prev.numel() > 0:
                self.xyz_gradient_accum_prev = self.xyz_gradient_accum_prev[keep_prev]
            if hasattr(self, 'denom_prev') and self.denom_prev.numel() > 0:
                self.denom_prev = self.denom_prev[keep_prev]
            if hasattr(self, 'filter_3D_prev') and self.filter_3D_prev.numel() > 0:
                self.filter_3D_prev = self.filter_3D_prev[keep_prev]
        
        # === delete global masks ===
        # ensure all global masks are on the same device as keep_all
        target_device = self.visibility_filter_all.device
        keep_all = keep_all.to(target_device)
        
        # ensure all global masks are on the same device
        self.visibility_filter_all = self.visibility_filter_all.to(target_device)
        self.delete_mask_all = self.delete_mask_all.to(target_device)
        
        # now safely perform indexing operations
        self.visibility_filter_all = self.visibility_filter_all[keep_all]
        self.delete_mask_all = self.delete_mask_all[keep_all]
        
        # === handle current and prev sky filters separately ===
        # import pdb; pdb.set_trace()
        
        if n_curr > 0 and delete_curr.any():
            keep_curr = keep_curr.to(self.is_sky_filter.device)
            self.is_sky_filter = self.is_sky_filter[keep_curr]
        
        if n_prev > 0 and delete_prev.any() and hasattr(self, 'is_sky_filter_prev'):
            keep_prev = keep_prev.to(self.is_sky_filter_prev.device)
            self.is_sky_filter_prev = self.is_sky_filter_prev[keep_prev]
        
        # update point labels - using current/prev separation mode
        if n_curr > 0 and delete_curr.any():
            keep_curr = keep_curr.to(self.point_labels.device)
            self.point_labels = self.point_labels[keep_curr]
        
        if n_prev > 0 and delete_prev.any() and hasattr(self, 'point_labels_prev'):
            keep_prev = keep_prev.to(self.point_labels_prev.device)
            self.point_labels_prev = self.point_labels_prev[keep_prev]


        
        # update next_scale, prior_scale and now_scale - using current/prev separation mode
        if n_curr > 0 and delete_curr.any():
            keep_curr = keep_curr.to(self.next_scale.device)
            self.next_scale = self.next_scale[keep_curr]
            self.prior_scale = self.prior_scale[keep_curr]
            if hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
                self.now_scale = self.now_scale[keep_curr]
        
        if n_prev > 0 and delete_prev.any() and hasattr(self, 'next_scale_prev'):
            keep_prev = keep_prev.to(self.next_scale_prev.device)
            self.next_scale_prev = self.next_scale_prev[keep_prev]
            self.prior_scale_prev = self.prior_scale_prev[keep_prev]
            if hasattr(self, 'now_scale_prev') and self.now_scale_prev.numel() > 0:
                self.now_scale_prev = self.now_scale_prev[keep_prev]
        
        # === re-setup optimizer (if it exists) ===
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            # save training_args for re-setup
            if hasattr(self, '_last_training_args'):
                print("Re-setting up optimizer after point deletion...")
                self.training_setup(self._last_training_args)
            else:
                print("Warning: optimizer exists but no training_args saved. You may need to manually call training_setup().")
        
        # consistency check
        self._check_consistency()


    def yield_splat_data_with_lod(self, path):
        """Generate splat file + companion LoD data file"""
        print('yielding splat data with LoD...')
        
        # 1. generate standard splat file (keep original format unchanged)
        splat_data = self.yield_splat_data(path)
        
        # 2. generate companion LoD data file
        self._yield_lod_companion_file(path)
        
        return splat_data

    def _yield_lod_companion_file(self, splat_path):
        """Generate LoD data corresponding to the splat file"""
        print('generating LoD companion file...')
        
        # use the same filter and sorting logic to ensure data correspondence
        filter_all = ~self.delete_mask_all
        filter_all = filter_all.cpu()
        
        # get LoD data
        prior_scales = self.get_prior_scale_all.cpu().numpy()[filter_all]
        now_scales = self.get_now_scale_all.cpu().numpy()[filter_all] 
        next_scales = self.get_next_scale_all.cpu().numpy()[filter_all]
        
        # key: use the same sorting logic as in yield_splat_data
        def apply_activation(x):
            return np.clip(np.tanh(x) * 0.51, -0.5, 0.5) + 0.5
        
        # rebuild the exact same sorting as the splat file
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()[filter_all]
        normals = np.zeros_like(xyz)
        dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        if self._features_dc_prev.numel() > 0:
            dc_prev = self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
            f_dc = torch.cat([dc, dc_prev], dim=0)
        else:
            f_dc = dc
        f_dc = f_dc.cpu().numpy()[filter_all]
        
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()[filter_all]
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()[filter_all]
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()[filter_all]
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()[filter_all]
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        # use the same sorting as splat
        sorted_indices = np.argsort(
            -np.exp(el["scale_0"] + el["scale_1"] + el["scale_2"]) * apply_activation(el["opacity"])
        )
        
        # generate LoD data (in sorted order)
        lod_path = splat_path.replace('.splat', '.lod')
        import struct
        with open(lod_path, 'wb') as f:
            # write header: 4 bytes for point count
            f.write(struct.pack('I', len(sorted_indices)))
            
            # write LoD data: 12 bytes per point (prior+now+next, 4 bytes float32 each)
            for idx in tqdm(sorted_indices, desc="Writing LoD data"):
                f.write(struct.pack('fff', 
                    prior_scales[idx],
                    now_scales[idx], 
                    next_scales[idx]
                ))
        
        print(f'✅ LoD companion file saved: {lod_path}')
        return lod_path

    def yield_splat_data_raw(self, path):
        print('yielding splat data raw...')
        def apply_activation(x):
            return np.clip(np.tanh(x) * 0.51, -0.5, 0.5) + 0.5
        
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(), self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()], dim=0).cpu().numpy()
        current_opacity_with_filter = self.get_opacity_with_3D_filter_all
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()        
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        vert = el
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) * apply_activation(vert["opacity"])
        )
        buffer = BytesIO()
        
        for idx in tqdm(sorted_indices):
            v = el[idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            color = np.array(
                [
                    apply_activation(v["f_dc_0"]),
                    apply_activation(v["f_dc_1"]),
                    apply_activation(v["f_dc_2"]),
                    apply_activation(v["opacity"]),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        splat_data = buffer.getvalue()
        with open(path, 'wb') as f:
            f.write(splat_data)
        print('splat data raw yielded')
        return splat_data
    
    def yield_splat_data(self, path):
        print('yielding splat data...')
        def apply_activation(x):
            return np.clip(np.tanh(x) * 0.51, -0.5, 0.5) + 0.5
        
        # filter_all = ~self.delete_mask_all & (~self.is_sky_filter)
        filter_all = ~self.delete_mask_all
        filter_all = filter_all.cpu()
        
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        xyz = xyz[filter_all]
        normals = np.zeros_like(xyz)
        dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()

        if self._features_dc_prev.numel() > 0:
            dc_prev = self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
            f_dc = torch.cat([dc, dc_prev], dim=0)
        else:
            f_dc = dc

        f_dc = f_dc.cpu().numpy()        
        f_dc = f_dc[filter_all]
        current_opacity_with_filter = self.get_opacity_with_3D_filter_all
        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy()        
        opacities = opacities[filter_all]
        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy()
        scale = scale[filter_all]
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()
        rotation = rotation[filter_all]
        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy()
        filters_3D = filters_3D[filter_all]
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        vert = el
        sorted_indices = range(len(el))
        # sorted_indices = np.argsort(
        #     -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"]) * apply_activation(vert["opacity"])
        # )
        buffer = BytesIO()
        
        for idx in tqdm(sorted_indices):
            v = el[idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            color = np.array(
                [
                    apply_activation(v["f_dc_0"]),
                    apply_activation(v["f_dc_1"]),
                    apply_activation(v["f_dc_2"]),
                    apply_activation(v["opacity"]),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        splat_data = buffer.getvalue()
        with open(path, 'wb') as f:
            f.write(splat_data)
        print('splat data yielded')
        # return splat_data


    def yield_spark_splat_data(self, path):
        gaus_copy = copy.deepcopy(self)
        # gaus_copy = copy.deepcopy(gaussians)
        gaus_copy.merge_all_to_trainable()
        mask = gaus_copy.point_labels == int(1e5)
        gaus_copy.set_trainable_mask(mask)
        # gaus_copy.delete_all_points(mask)
        gaus_copy.merge_all_to_trainable()
        gaus_copy.yield_splat_data(path)
        
    def save(self, path):
        """Save current GaussianModel state to file"""
        self.merge_all_to_trainable()
        keys = [
            "_xyz", "_features_dc", "_scaling", "_rotation", "_opacity",
            "_focal_length", "next_scale", "prior_scale", "now_scale",
            "max_radii2D", "xyz_gradient_accum", "denom", "filter_3D",
            "visibility_filter_all", "is_sky_filter", "delete_mask_all", "point_labels"
        ]
        
        # compatible saving of prev labels
        if hasattr(self, "point_labels_prev"):
            keys.append("point_labels_prev")

        state = {key: getattr(self, key).detach().cpu() for key in keys if hasattr(self, key)}

        # save model configuration
        state["max_sh_degree"] = self.max_sh_degree
        state["floater_dist2_threshold"] = self.floater_dist2_threshold

        # save global label mapping
        try:
            import sys as _sys
            names = None
            amap = None
            for _m in _sys.modules.values():
                if hasattr(_m, 'GLOBAL_LABEL_NAMES') and hasattr(_m, 'GLOBAL_LABEL_MAP'):
                    cand_names = getattr(_m, 'GLOBAL_LABEL_NAMES')
                    cand_map = getattr(_m, 'GLOBAL_LABEL_MAP')
                    if isinstance(cand_names, list) and isinstance(cand_map, dict):
                        names, amap = cand_names, cand_map
                        break
            if names is None:
                # fallback to default
                names = ["main"]
                amap = {"main": 0}
            state["global_label_names"] = list(names)
            state["global_label_map"] = dict(amap)
        except Exception:
            state["global_label_names"] = ["main"]
            state["global_label_map"] = {"main": 0}

        torch.save(state, path)
        print(f"GaussianModel state saved to: {path}")
    
    @classmethod
    def load(cls, path, config):
        """Create and restore GaussianModel from file"""
        state = torch.load(path, map_location="cuda")

        model = cls(
            sh_degree=state.get("max_sh_degree", 3),
            floater_dist2_threshold=state.get("floater_dist2_threshold", 0.0002),
            config=config
        )

        for key, val in state.items():
            if key in ["max_sh_degree", "floater_dist2_threshold", "global_label_names", "global_label_map"]:
                continue
            if hasattr(model, key):
                existing = getattr(model, key)
                if isinstance(existing, torch.nn.Parameter):
                    setattr(model, key, torch.nn.Parameter(val.cuda(), requires_grad=True))
                else:
                    setattr(model, key, val.cuda())
            else:
                setattr(model, key, val.cuda())  # optional: allow new attribute injection

        # restore global label mapping to module level and loaded modules
        loaded_names = state.get("global_label_names", None)
        loaded_map = state.get("global_label_map", None)
        if loaded_names is not None and loaded_map is not None:
            try:
                # sync global mapping to potentially existing runtime modules
                import sys as _sys
                for _m in _sys.modules.values():
                    if hasattr(_m, 'GLOBAL_LABEL_NAMES') and hasattr(_m, 'GLOBAL_LABEL_MAP'):
                        try:
                            names_obj = getattr(_m, 'GLOBAL_LABEL_NAMES')
                            map_obj = getattr(_m, 'GLOBAL_LABEL_MAP')
                            if isinstance(names_obj, list):
                                names_obj[:] = list(loaded_names)
                            else:
                                setattr(_m, 'GLOBAL_LABEL_NAMES', list(loaded_names))
                            if isinstance(map_obj, dict):
                                map_obj.clear(); map_obj.update(dict(loaded_map))
                            else:
                                setattr(_m, 'GLOBAL_LABEL_MAP', dict(loaded_map))
                        except Exception:
                            setattr(_m, 'GLOBAL_LABEL_NAMES', list(loaded_names))
                            setattr(_m, 'GLOBAL_LABEL_MAP', dict(loaded_map))
            except Exception as _e:
                print(f"⚠️ Failed to restore global labels: {_e}")

        print(f"GaussianModel loaded from {path}")
        return model
    
    def save_ply(self, path, use_higher_freq=True, use_splat=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat([self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous(), self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()], dim=0).cpu().numpy()
        current_opacity_with_filter = torch.cat([self.get_opacity_with_3D_filter, self.get_opacity_with_3D_filter_all], dim=0)
        opacities = self.inverse_opacity_activation(current_opacity_with_filter).detach().cpu().numpy()
        scale = torch.cat([self.scaling_inverse_activation(self.get_scaling_with_3D_filter), self.scaling_inverse_activation(self.get_scaling_with_3D_filter_all)], dim=0).cpu().numpy()
        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True, use_higher_freq=use_higher_freq)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        if use_splat:
            vert = el
            sorted_indices = np.argsort(
                -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
                / (1 + np.exp(-vert["opacity"]))
            )
            buffer = BytesIO()
            for idx in sorted_indices:
                v = el[idx]
                position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
                scales = np.exp(
                    np.array(
                        [v["scale_0"], v["scale_1"], v["scale_2"]],
                        dtype=np.float32,
                    )
                )
                rot = np.array(
                    [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                    dtype=np.float32,
                )
                SH_C0 = 0.28209479177387814
                color = np.array(
                    [
                        0.5 + SH_C0 * v["f_dc_0"],
                        0.5 + SH_C0 * v["f_dc_1"],
                        0.5 + SH_C0 * v["f_dc_2"],
                        1 / (1 + np.exp(-v["opacity"])),
                    ]
                )
                buffer.write(position.tobytes())
                buffer.write(scales.tobytes())
                buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
                buffer.write(
                    ((rot / np.linalg.norm(rot)) * 128 + 128)
                    .clip(0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )

            splat_data = buffer.getvalue()
            with open(path, "wb") as f:
                f.write(splat_data)
        else:
            PlyData([el]).write(path)

    def save_ply_with_filter(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filters_3D = self.filter_3D.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)

    
    def save_ply_all_with_filter(self, path):
        from plyfile import PlyElement, PlyData
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print('saving ply with filter...')

        # filter out deleted points (can add is_sky_filter logic etc.)
        filter_all = ~self.delete_mask_all
        filter_all = filter_all.cpu()

        # concatenate current and prev data (if prev exists)
        xyz = torch.cat([self._xyz.detach(), self._xyz_prev], dim=0).cpu().numpy() \
            if self._xyz_prev.numel() > 0 else self._xyz.detach().cpu().numpy()
        xyz = xyz[filter_all]
        normals = np.zeros_like(xyz)

        dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
        if self._features_dc_prev.numel() > 0:
            dc_prev = self._features_dc_prev.detach().transpose(1, 2).flatten(start_dim=1).contiguous()
            f_dc = torch.cat([dc, dc_prev], dim=0)
        else:
            f_dc = dc
        f_dc = f_dc.cpu().numpy()
        f_dc = f_dc[filter_all]

        opacities = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0).cpu().numpy() \
            if self._opacity_prev.numel() > 0 else self._opacity.detach().cpu().numpy()
        opacities = opacities[filter_all]

        scale = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0).cpu().numpy() \
            if self._scaling_prev.numel() > 0 else self._scaling.detach().cpu().numpy()
        scale = scale[filter_all]

        rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0).cpu().numpy() \
            if self._rotation_prev.numel() > 0 else self._rotation.detach().cpu().numpy()
        rotation = rotation[filter_all]

        filters_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0).cpu().numpy() \
            if self.filter_3D_prev.numel() > 0 else self.filter_3D.detach().cpu().numpy()
        filters_3D = filters_3D[filter_all]

        # define attribute field structure
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=False, use_higher_freq=False)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation, filters_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
        print('ply file saved')
    def load_ply_with_filter(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(np.asarray(plydata.elements[0]["filter_3D"]), dtype=torch.float, device="cuda")[:, None]

        self.active_sh_degree = self.max_sh_degree
        
    def save_ply_combined(self, gaussian, path, use_higher_freq=True, use_splat=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz_1 = self._xyz.detach().cpu().numpy()
        xyz_2 = gaussian._xyz.detach().cpu().numpy()
        xyz = np.concatenate((xyz_1, xyz_2), axis=0)
        normals = np.zeros_like(xyz)
        f_dc_1 = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_2 = gaussian._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = np.concatenate((f_dc_1, f_dc_2), axis=0)
    
        current_opacity_with_filter_1 = self.get_opacity_with_3D_filter
        opacities_1 = self.inverse_opacity_activation(current_opacity_with_filter_1).detach().cpu().numpy()
        current_opacity_with_filter_2 = gaussian.get_opacity_with_3D_filter
        opacities_2 = self.inverse_opacity_activation(current_opacity_with_filter_2).detach().cpu().numpy()
        opacities = np.concatenate((opacities_1, opacities_2), axis=0)
    
        scale_1 = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        scale_2 = gaussian.scaling_inverse_activation(gaussian.get_scaling_with_3D_filter).detach().cpu().numpy()
        scale = np.concatenate((scale_1, scale_2), axis=0)
    
        rotation_1 = self._rotation.detach().cpu().numpy()
        rotation_2 = gaussian._rotation.detach().cpu().numpy()
        rotation = np.concatenate((rotation_1, rotation_2), axis=0)
    
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True, use_higher_freq=use_higher_freq)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        if use_splat:
            vert = el
            sorted_indices = np.argsort(
                -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
                / (1 + np.exp(-vert["opacity"]))
            )
            buffer = BytesIO()
            for idx in sorted_indices:
                v = el[idx]
                position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
                scales = np.exp(
                    np.array(
                        [v["scale_0"], v["scale_1"], v["scale_2"]],
                        dtype=np.float32,
                    )
                )
                rot = np.array(
                    [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                    dtype=np.float32,
                )
                SH_C0 = 0.28209479177387814
                color = np.array(
                    [
                        0.5 + SH_C0 * v["f_dc_0"],
                        0.5 + SH_C0 * v["f_dc_1"],
                        0.5 + SH_C0 * v["f_dc_2"],
                        1 / (1 + np.exp(-v["opacity"])),
                    ]
                )
                buffer.write(position.tobytes())
                buffer.write(scales.tobytes())
                buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
                buffer.write(
                    ((rot / np.linalg.norm(rot)) * 128 + 128)
                    .clip(0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )

            splat_data = buffer.getvalue()
            with open(path, "wb") as f:
                f.write(splat_data)
        else:
            PlyData([el]).write(path)
            
    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = inverse_sigmoid(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    self.optimizer.state[group['params'][0]] = stored_state
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        Prune gaussians based on mask.
        
        Args:
            mask: Boolean mask for points to prune (True = prune, False = keep)
                  If mask length equals current trainable points, applies only to trainable points.
                  If mask length equals all points (current + prev), splits the mask appropriately.
        """
        n_current = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0] if self._xyz_prev.numel() > 0 else 0
        n_total = n_current + n_prev
        
        if len(mask) == n_current:
            # Mask is only for current trainable points
            valid_points_mask = ~mask
            
            # Apply optimizer pruning
            optimizable_tensors = self._prune_optimizer(valid_points_mask)
            
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            # Update training states for current points
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self._focal_length = self._focal_length[valid_points_mask]

            # Update filter_3D for current points
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                self.filter_3D = self.filter_3D[valid_points_mask]
            
            # Update next_scale, prior_scale and now_scale for current points
            if hasattr(self, 'next_scale') and self.next_scale.numel() > 0:
                self.next_scale = self.next_scale[valid_points_mask]
            if hasattr(self, 'prior_scale') and self.prior_scale.numel() > 0:
                self.prior_scale = self.prior_scale[valid_points_mask]
            if hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
                self.now_scale = self.now_scale[valid_points_mask]
            
            # Update global masks
            if n_prev > 0:
                # Keep prev points, only update current part
                current_vis = self.visibility_filter_all[:n_current][valid_points_mask]
                prev_vis = self.visibility_filter_all[n_current:]
                self.visibility_filter_all = torch.cat([current_vis, prev_vis])
                
                current_sky = self.is_sky_filter[:n_current][valid_points_mask] 
                prev_sky = self.is_sky_filter[n_current:]
                self.is_sky_filter = torch.cat([current_sky, prev_sky])
                
                current_del = self.delete_mask_all[:n_current][valid_points_mask]
                prev_del = self.delete_mask_all[n_current:]
                self.delete_mask_all = torch.cat([current_del, prev_del])
                
                # update point labels - using current/prev separation mode
                if hasattr(self, 'point_labels') and self.point_labels.numel() > 0:
                    self.point_labels = self.point_labels[valid_points_mask]
            else:
                # No prev points, update everything
                self.visibility_filter_all = self.visibility_filter_all[valid_points_mask]
                self.is_sky_filter = self.is_sky_filter[valid_points_mask]
                self.delete_mask_all = self.delete_mask_all[valid_points_mask]
                # update point labels - same as other attributes
                if hasattr(self, 'point_labels') and self.point_labels.numel() > 0:
                    self.point_labels = self.point_labels[valid_points_mask]

        elif len(mask) == n_total:
            # Mask is for all points (current + prev), use the old logic
            valid_points_mask = ~mask
            
            # Split mask for current and prev
            current_mask = valid_points_mask[:n_current]
            prev_mask = valid_points_mask[n_current:] if n_prev > 0 else torch.empty(0, dtype=torch.bool, device=mask.device)
            
            # Apply optimizer pruning only to current points
            optimizable_tensors = self._prune_optimizer(current_mask)
            
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            # Update training states for current points
            self.xyz_gradient_accum = self.xyz_gradient_accum[current_mask]
            self.denom = self.denom[current_mask]
            self.max_radii2D = self.max_radii2D[current_mask]
            self._focal_length = self._focal_length[current_mask]

            # Update filter_3D for current points
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                self.filter_3D = self.filter_3D[current_mask]
            
            # Update next_scale, prior_scale and now_scale for current points
            if hasattr(self, 'next_scale') and self.next_scale.numel() > 0:
                self.next_scale = self.next_scale[current_mask]
            if hasattr(self, 'prior_scale') and self.prior_scale.numel() > 0:
                self.prior_scale = self.prior_scale[current_mask]
            if hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
                self.now_scale = self.now_scale[current_mask]
            
            # Update prev points if needed
            if n_prev > 0 and prev_mask.numel() > 0:
                self._xyz_prev = self._xyz_prev[prev_mask]
                self._features_dc_prev = self._features_dc_prev[prev_mask]
                self._scaling_prev = self._scaling_prev[prev_mask] 
                self._rotation_prev = self._rotation_prev[prev_mask]
                self._opacity_prev = self._opacity_prev[prev_mask]
                self._focal_length_prev = self._focal_length_prev[prev_mask]
                
                # Update prev training states
                if hasattr(self, 'max_radii2D_prev') and self.max_radii2D_prev.numel() > 0:
                    self.max_radii2D_prev = self.max_radii2D_prev[prev_mask]
                if hasattr(self, 'filter_3D_prev') and self.filter_3D_prev.numel() > 0:
                    self.filter_3D_prev = self.filter_3D_prev[prev_mask]
                
                # Update next_scale, prior_scale and now_scale for prev points
                if hasattr(self, 'next_scale_prev') and self.next_scale_prev.numel() > 0:
                    self.next_scale_prev = self.next_scale_prev[prev_mask]
                if hasattr(self, 'prior_scale_prev') and self.prior_scale_prev.numel() > 0:
                    self.prior_scale_prev = self.prior_scale_prev[prev_mask]
                if hasattr(self, 'now_scale_prev') and self.now_scale_prev.numel() > 0:
                    self.now_scale_prev = self.now_scale_prev[prev_mask]
            
            # Update global masks - using data that includes all points
            self.visibility_filter_all = self.visibility_filter_all[valid_points_mask]
            
            # get is_sky_filter containing all points, then filter
            all_sky_filter = self.get_is_sky_filter_all
            filtered_sky = all_sky_filter[valid_points_mask]
            # redistribute: current points in front, prev points in back
            n_remaining = valid_points_mask.sum().item()
            n_remaining_current = (valid_points_mask[:n_current]).sum().item()
            n_remaining_prev = n_remaining - n_remaining_current
            
            self.is_sky_filter = filtered_sky[:n_remaining_current]
            if hasattr(self, 'is_sky_filter_prev') and n_remaining_prev > 0:
                self.is_sky_filter_prev = filtered_sky[n_remaining_current:]
            
            self.delete_mask_all = self.delete_mask_all[valid_points_mask]
            
            # update point labels - using current/prev separation mode
            if hasattr(self, 'point_labels') and self.point_labels.numel() > 0:
                self.point_labels = self.point_labels[current_mask]
            if hasattr(self, 'point_labels_prev') and self.point_labels_prev.numel() > 0 and n_prev > 0:
                self.point_labels_prev = self.point_labels_prev[prev_mask]
            
        else:
            raise ValueError(f"Mask length ({len(mask)}) must equal either current points ({n_current}) or total points ({n_total})")
        
        print(f"✅ Pruned gaussians: {mask.sum().item()} removed, {(~mask).sum().item()} remaining")


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_focal_length, source_mask=None, repeat_times=1):
        
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        n_added_points = new_xyz.shape[0] #- self.get_xyz.shape[0]
        print("Densify!",new_xyz.shape )
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._focal_length = torch.cat([self._focal_length, new_focal_length], dim=0)
        
        # add filter_3D update
        if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
            # create default filter_3D values for new points
            new_filter_3D = torch.ones(n_added_points, 1, device='cuda') * 0.001
            self.filter_3D = torch.cat([self.filter_3D, new_filter_3D], dim=0)
        else:
            # if no filter_3D existed before, create for all points
            total_points = self.get_xyz.shape[0]
            self.filter_3D = torch.ones(total_points, 1, device='cuda') * 0.001
        
        if n_added_points > 0:
            new_visibility = torch.ones(n_added_points, dtype=torch.bool, device='cuda')
            new_delete = torch.zeros(n_added_points, dtype=torch.bool, device='cuda') 
            new_sky = new_xyz[:, 2] > 50.0 # torch.zeros(n_added_points, dtype=torch.bool, device='cuda')
            
            self.visibility_filter_all = torch.cat([self.visibility_filter_all, new_visibility])
            self.delete_mask_all = torch.cat([self.delete_mask_all, new_delete])
            self.is_sky_filter = torch.cat([self.is_sky_filter.to(new_sky.device), new_sky])
            
            # update point labels - inherit labels from source points (only operates on current trainable points)
            if hasattr(self, 'point_labels') and self.point_labels.numel() > 0:
                if source_mask is not None:
                    # FIX: now point_labels only contains current trainable points, source_mask corresponds to current trainable points
                    source_labels = self.point_labels[source_mask]
                    new_labels = source_labels.repeat(repeat_times)
                else:
                    # no source point info, default to main
                    new_labels = torch.zeros(n_added_points, dtype=torch.long, device='cuda')
                # directly add to current trainable point labels
                self.point_labels = torch.cat([self.point_labels, new_labels])
            else:
                # important fix: do not reset existing labels!
                # if current point_labels is empty, check whether labels need to be restored from prev
                if hasattr(self, 'point_labels_prev') and self.point_labels_prev.numel() > 0:
                    print("⚠️ point_labels is empty but point_labels_prev exists. This might indicate a problem with set_trainable_mask.")
                    print("⚠️ Consider calling merge_all_to_trainable() to restore labels.")
                
                # only initialize labels for new points, do not reset all existing points
                total_current_points = self.get_xyz.shape[0]
                existing_points = total_current_points - n_added_points
                
                if existing_points > 0:
                    # if existing points have no labels, this is a problem, but do not force reset
                    print(f"⚠️ Warning: {existing_points} existing points have no labels. Consider investigating.")
                    # create default labels for existing points (main=0)
                    existing_labels = torch.zeros(existing_points, dtype=torch.long, device='cuda')
                    new_labels = torch.zeros(n_added_points, dtype=torch.long, device='cuda')
                    self.point_labels = torch.cat([existing_labels, new_labels])
                else:
                    # only new points, create default labels for them
                    self.point_labels = torch.zeros(n_added_points, dtype=torch.long, device='cuda')
            

            
            # update next_scale, prior_scale and now_scale - all new points use default values or inherit from parent points
            if hasattr(self, 'next_scale') and self.next_scale.numel() > 0:
                # assign default values for all new points
                new_next_scale = torch.full((n_added_points,), float('inf'), device='cuda')
                new_prior_scale = torch.full((n_added_points,), float('-inf'), device='cuda')
                
                # inherit source points' now_scale (if source_mask available)
                if source_mask is not None and hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
                    source_now_scale = self.now_scale[source_mask].detach()
                    new_now_scale = source_now_scale.repeat(repeat_times)
                else:
                    # no source points, use default values
                    new_now_scale = torch.full((n_added_points,), 0.01, device='cuda')
                
                # directly add to current trainable points' scales
                self.next_scale = torch.cat([self.next_scale, new_next_scale]).detach()
                self.prior_scale = torch.cat([self.prior_scale, new_prior_scale]).detach()
                self.now_scale = torch.cat([self.now_scale, new_now_scale]).detach()
            else:
                # initialize scale attributes
                total_current_points = self.get_xyz.shape[0]
                existing_points = total_current_points - n_added_points
                
                if existing_points > 0:
                    # create default values for existing points
                    existing_next_scale = torch.full((existing_points,), float('inf'), device='cuda')
                    existing_prior_scale = torch.full((existing_points,), float('-inf'), device='cuda')
                    existing_now_scale = torch.full((existing_points,), 0.01, device='cuda')
                    new_next_scale = torch.full((n_added_points,), float('inf'), device='cuda')
                    new_prior_scale = torch.full((n_added_points,), float('-inf'), device='cuda')
                    new_now_scale = torch.full((n_added_points,), 0.01, device='cuda')
                    self.next_scale = torch.cat([existing_next_scale, new_next_scale]).detach()
                    self.prior_scale = torch.cat([existing_prior_scale, new_prior_scale]).detach()
                    self.now_scale = torch.cat([existing_now_scale, new_now_scale]).detach()
                else:
                    # only new points
                    self.next_scale = torch.full((n_added_points,), float('inf'), device='cuda')
                    self.prior_scale = torch.full((n_added_points,), float('-inf'), device='cuda')
                    self.now_scale = torch.full((n_added_points,), 0.01, device='cuda')

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_focal_length = self._focal_length[selected_pts_mask].repeat(N)

        # pass source point mask to inherit labels
        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation, new_focal_length, selected_pts_mask, N)

        # after densification_postfix, current points count has increased, need to create prune_filter based on new count
        n_new_current = self.get_xyz.shape[0]  # new point count after densification_postfix
        prune_filter = torch.zeros(n_new_current, device="cuda", dtype=bool)
        prune_filter[:n_init_points] = selected_pts_mask  # mark original points to delete
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_focal_length = self._focal_length[selected_pts_mask]

        # pass source point mask to inherit labels
        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation, new_focal_length, selected_pts_mask, 1)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        print("pruning!",prune_mask.numel())
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def prune_large_scales(self, max_scale_threshold=None, scene_extent=None, scale_threshold_factor=99999):
        """
        Prune gaussians with scales that are too large
        
        Args:
            max_scale_threshold: Absolute maximum scale threshold. If None, uses relative threshold.
            scene_extent: Scene extent for relative threshold calculation
            scale_threshold_factor: Factor of scene extent to use as threshold
        """
        if max_scale_threshold is None and scene_extent is not None:
            max_scale_threshold = scale_threshold_factor * scene_extent
        elif max_scale_threshold is None:
            max_scale_threshold = 99999  # Default threshold
        
        # Get maximum scale per gaussian point
        max_scales = self.get_scaling.max(dim=1).values * (~self.is_sky_filter)
        
        # Create pruning mask for points with scales larger than threshold
        large_scale_mask = max_scales > max_scale_threshold
        
        if large_scale_mask.any():
            n_pruned = large_scale_mask.sum().item()
            print(f"🔧 Scale pruning: removing {n_pruned} points with max_scale > {max_scale_threshold:.6f}")
            self.prune_points(large_scale_mask)
            
            return n_pruned
        else:
            return 0
    
    def set_trainable_mask(self, trainable_mask):
        """
        Set which gaussians are trainable and which are non-trainable

        Args:
            trainable_mask: torch.Tensor (bool), length should equal the number of current trainable points
                        True means the point stays trainable, False means move to non-trainable
        """
        # check if optimizer exists, skip if not (wait for subsequent training_setup)
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            print("⚠️ Optimizer not available, skipping trainability control (will be applied after training_setup)")
            # save trainable_mask for later application
            self._pending_trainable_mask = trainable_mask.clone()
            return
            
        n_current = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0] if self._xyz_prev.numel() > 0 else 0
        n_total = n_current + n_prev
        
        if len(trainable_mask) != n_current:
            raise ValueError(f"trainable_mask length ({len(trainable_mask)}) must equal current trainable points ({n_current})")
        
        # separate trainable and non-trainable indices
        keep_trainable = trainable_mask  # True: stays trainable
        move_to_prev = ~trainable_mask   # False: move to non-trainable
        
        n_keep = keep_trainable.sum().item()
        n_move = move_to_prev.sum().item()
        
        print(f"🔄 Setting trainable mask: {n_keep} remain trainable, {n_move} moved to non-trainable")
        
        if n_move > 0:
            # === extract points to be moved to prev ===
            move_xyz = self._xyz[move_to_prev].detach()
            move_features_dc = self._features_dc[move_to_prev].detach()
            move_scaling = self._scaling[move_to_prev].detach()
            move_rotation = self._rotation[move_to_prev].detach()
            move_opacity = self._opacity[move_to_prev].detach()
            move_focal_length = self._focal_length[move_to_prev].detach()
            
            # training-related variables
            if hasattr(self, 'max_radii2D') and self.max_radii2D.numel() > 0:
                move_max_radii2D = self.max_radii2D[move_to_prev].detach()
            else:
                move_max_radii2D = torch.zeros(n_move, device='cuda')
                
            if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.numel() > 0:
                move_xyz_grad_accum = self.xyz_gradient_accum[move_to_prev].detach()
            else:
                move_xyz_grad_accum = torch.zeros(n_move, 1, device='cuda')
                
            if hasattr(self, 'denom') and self.denom.numel() > 0:
                move_denom = self.denom[move_to_prev].detach()
            else:
                move_denom = torch.zeros(n_move, 1, device='cuda')
                
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                move_filter_3D = self.filter_3D[move_to_prev].detach()
            else:
                move_filter_3D = torch.ones(n_move, 1, device='cuda') * 0.001
            
            # FIX: extract scale info for points to be moved
            if hasattr(self, 'next_scale') and self.next_scale.numel() > 0:
                move_next_scale = self.next_scale[move_to_prev].detach()
            else:
                move_next_scale = torch.full((n_move,), float('inf'), device='cuda')
                
            if hasattr(self, 'prior_scale') and self.prior_scale.numel() > 0:
                move_prior_scale = self.prior_scale[move_to_prev].detach()
            else:
                move_prior_scale = torch.full((n_move,), float('-inf'), device='cuda')
                
            if hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
                move_now_scale = self.now_scale[move_to_prev].detach()
            else:
                move_now_scale = torch.full((n_move,), 0.01, device='cuda')
            
            # FIX: extract global masks for points to be moved
            # here we need to extract the current trainable points' portion from global masks
            # ensure all tensors are on the same device
            move_to_prev = move_to_prev.to('cuda')
            move_visibility = self.visibility_filter_all[:n_current].to('cuda')[move_to_prev]
            move_delete = self.delete_mask_all[:n_current].to('cuda')[move_to_prev]
            move_sky = self.is_sky_filter[:n_current].to('cuda')[move_to_prev]
            
            # === add moved points to prev ===
            self._xyz_prev = torch.cat([self._xyz_prev, move_xyz], dim=0)
            self._features_dc_prev = torch.cat([self._features_dc_prev, move_features_dc], dim=0)
            self._scaling_prev = torch.cat([self._scaling_prev, move_scaling], dim=0)
            self._rotation_prev = torch.cat([self._rotation_prev, move_rotation], dim=0)
            self._opacity_prev = torch.cat([self._opacity_prev, move_opacity], dim=0)
            self._focal_length_prev = torch.cat([self._focal_length_prev, move_focal_length], dim=0)
            
            # training-related prev variables
            self.max_radii2D_prev = torch.cat([self.max_radii2D_prev, move_max_radii2D], dim=0)
            self.xyz_gradient_accum_prev = torch.cat([self.xyz_gradient_accum_prev, move_xyz_grad_accum], dim=0)
            self.denom_prev = torch.cat([self.denom_prev, move_denom], dim=0)
            self.filter_3D_prev = torch.cat([self.filter_3D_prev, move_filter_3D], dim=0)
            
            # FIX: add moved scale info to prev
            if hasattr(self, 'next_scale_prev'):
                self.next_scale_prev = torch.cat([self.next_scale_prev, move_next_scale], dim=0)
            else:
                self.next_scale_prev = move_next_scale.clone()
                
            if hasattr(self, 'prior_scale_prev'):
                self.prior_scale_prev = torch.cat([self.prior_scale_prev, move_prior_scale], dim=0)
            else:
                self.prior_scale_prev = move_prior_scale.clone()
                
            if hasattr(self, 'now_scale_prev'):
                self.now_scale_prev = torch.cat([self.now_scale_prev, move_now_scale], dim=0)
            else:
                self.now_scale_prev = move_now_scale.clone()
            
            # global mask prev variables
            self.is_sky_filter_prev = torch.cat([self.is_sky_filter_prev.to('cuda'), move_sky], dim=0)
            self.visibility_filter_all_prev = torch.cat([self.visibility_filter_all_prev.to('cuda'), move_visibility], dim=0)
            self.delete_mask_all_prev = torch.cat([self.delete_mask_all_prev.to('cuda'), move_delete], dim=0)
        else:
            # if no points to move, create empty move masks
            move_visibility = torch.empty(0, dtype=torch.bool, device='cuda')
            move_delete = torch.empty(0, dtype=torch.bool, device='cuda')
            move_sky = torch.empty(0, dtype=torch.bool, device='cuda')
            
            # ensure corresponding variables are on the correct device even when no points are moved
            if n_keep > 0:
                keep_trainable = keep_trainable.to('cuda')
                keep_visibility = self.visibility_filter_all[:n_current].to('cuda')[keep_trainable]
                keep_delete = self.delete_mask_all[:n_current].to('cuda')[keep_trainable]
                keep_sky = self.is_sky_filter[:n_current].to('cuda')[keep_trainable]
            else:
                keep_visibility = torch.empty(0, dtype=torch.bool, device='cuda')
                keep_delete = torch.empty(0, dtype=torch.bool, device='cuda')
                keep_sky = torch.empty(0, dtype=torch.bool, device='cuda')
        
        if n_keep > 0:
            # === keep trainable points ===
            # use optimizer's method to correctly update trainable parameters
            valid_points_mask = keep_trainable
            optimizable_tensors = self._prune_optimizer(valid_points_mask)
            
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            
            # update training state
            if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.numel() > 0:
                self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            else:
                self.xyz_gradient_accum = torch.zeros(n_keep, 1, device='cuda')
                
            if hasattr(self, 'denom') and self.denom.numel() > 0:
                self.denom = self.denom[valid_points_mask]
            else:
                self.denom = torch.zeros(n_keep, 1, device='cuda')
                
            if hasattr(self, 'max_radii2D') and self.max_radii2D.numel() > 0:
                self.max_radii2D = self.max_radii2D[valid_points_mask]
            else:
                self.max_radii2D = torch.zeros(n_keep, device='cuda')
                
            if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
                self.filter_3D = self.filter_3D[valid_points_mask]
            else:
                self.filter_3D = torch.ones(n_keep, 1, device='cuda') * 0.001
                
            # FIX: update current trainable points' scales (only keep unmoved ones)
            if hasattr(self, 'next_scale') and self.next_scale.numel() > 0:
                self.next_scale = self.next_scale[valid_points_mask]
            else:
                self.next_scale = torch.full((n_keep,), float('inf'), device='cuda')
                
            if hasattr(self, 'prior_scale') and self.prior_scale.numel() > 0:
                self.prior_scale = self.prior_scale[valid_points_mask]
            else:
                self.prior_scale = torch.full((n_keep,), float('-inf'), device='cuda')
                
            if hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
                self.now_scale = self.now_scale[valid_points_mask]
            else:
                self.now_scale = torch.full((n_keep,), 0.01, device='cuda')
                    
            self._focal_length = self._focal_length[valid_points_mask]
            
            # FIX: keep corresponding masks (only keep the trainable part)
            # ensure all tensors are on the same device
            keep_trainable = keep_trainable.to('cuda')
            keep_visibility = self.visibility_filter_all[:n_current].to('cuda')[keep_trainable]
            keep_delete = self.delete_mask_all[:n_current].to('cuda')[keep_trainable]
            keep_sky = self.is_sky_filter[:n_current].to('cuda')[keep_trainable]
            
            # update trainable portion's mask (only includes kept points)
            self.is_sky_filter = keep_sky
        else:
            # if no points remain trainable, create empty trainable parameters
            print("⚠️  No points remain trainable, creating empty trainable parameters")
            self._xyz = nn.Parameter(torch.empty(0, 3, device='cuda').requires_grad_(True))
            self._features_dc = nn.Parameter(torch.empty(0, 3, 1, device='cuda').requires_grad_(True))
            self._scaling = nn.Parameter(torch.empty(0, 3, device='cuda').requires_grad_(True))
            self._rotation = nn.Parameter(torch.empty(0, 4, device='cuda').requires_grad_(True))
            self._opacity = nn.Parameter(torch.empty(0, 1, device='cuda').requires_grad_(True))
            self._focal_length = torch.empty(0, device='cuda')
            
            # reset training state
            self.max_radii2D = torch.empty(0, device='cuda')
            self.xyz_gradient_accum = torch.empty(0, 1, device='cuda')
            self.denom = torch.empty(0, 1, device='cuda')
            if hasattr(self, 'filter_3D'):
                self.filter_3D = torch.empty(0, 1, device='cuda')
            
            # FIX: reset scale attributes to empty
            self.next_scale = torch.empty(0, device='cuda')
            self.prior_scale = torch.empty(0, device='cuda')
            self.now_scale = torch.empty(0, device='cuda')
            
            # create empty trainable mask
            self.is_sky_filter = torch.empty(0, dtype=torch.bool, device='cuda')
        
        # FIX: update point_labels - using current/prev separation mode
        if hasattr(self, 'point_labels') and self.point_labels.numel() > 0:
            # get labels of points to be moved (from current trainable points)
            if n_move > 0:
                move_labels = self.point_labels[move_to_prev]  # directly get from current labels
            else:
                move_labels = torch.empty(0, dtype=torch.long, device='cuda')
            
            # update current trainable points' labels (only keep unmoved ones)
            if n_keep > 0:
                self.point_labels = self.point_labels[keep_trainable]
            else:
                # important fix: do not completely clear point_labels!
                # when no points remain trainable, keep an empty tensor with correct device
                # this avoids triggering label resets in subsequent operations
                self.point_labels = torch.empty(0, dtype=torch.long, device=self.point_labels.device)
            
            # update prev labels (add moved points)
            if n_move > 0:
                if hasattr(self, 'point_labels_prev') and self.point_labels_prev.numel() > 0:
                    self.point_labels_prev = torch.cat([self.point_labels_prev, move_labels])
                else:
                    self.point_labels_prev = move_labels.clone()

        # FIX: reorganize global masks
        # new global mask layout: [current trainable points, non-trainable points (original prev + new move)]
        if n_prev > 0:
            # original prev portion
            prev_visibility = self.visibility_filter_all[n_current:].to('cuda')
            prev_delete = self.delete_mask_all[n_current:].to('cuda')
            
            # reorganize: [keep_current, move_to_prev, original_prev]
            self.visibility_filter_all = torch.cat([
                keep_visibility if n_keep > 0 else torch.empty(0, dtype=torch.bool, device='cuda'),
                move_visibility,        # points moved to prev
                prev_visibility         # original prev points
            ]).to('cuda')
            self.delete_mask_all = torch.cat([
                keep_delete if n_keep > 0 else torch.empty(0, dtype=torch.bool, device='cuda'),
                move_delete,           # points moved to prev
                prev_delete            # original prev points
            ]).to('cuda')
        else:
            # no original prev, only: [keep_current, move_to_prev]
            self.visibility_filter_all = torch.cat([
                keep_visibility if n_keep > 0 else torch.empty(0, dtype=torch.bool, device='cuda'),
                move_visibility
            ]).to('cuda')
            self.delete_mask_all = torch.cat([
                keep_delete if n_keep > 0 else torch.empty(0, dtype=torch.bool, device='cuda'),
                move_delete
            ]).to('cuda')
        
        print(f"✅ Trainable mask applied: {self.get_xyz.shape[0]} trainable, {self._xyz_prev.shape[0]} non-trainable")
        print(f"✅ Masks updated: is_sky_filter={self.is_sky_filter.shape[0]} (trainable), "
            f"is_sky_filter_prev={self.is_sky_filter_prev.shape[0]} (non-trainable)")
        print(f"✅ Global masks: visibility_filter_all={self.visibility_filter_all.shape[0]}, "
            f"delete_mask_all={self.delete_mask_all.shape[0]}")
        
        # final consistency check
        total_points = self.get_xyz.shape[0] + self._xyz_prev.shape[0]
        assert self.visibility_filter_all.shape[0] == total_points, f"visibility_filter_all size mismatch: {self.visibility_filter_all.shape[0]} vs {total_points}"
        assert self.delete_mask_all.shape[0] == total_points, f"delete_mask_all size mismatch: {self.delete_mask_all.shape[0]} vs {total_points}"
        assert self.is_sky_filter.shape[0] == self.get_xyz.shape[0], f"is_sky_filter size mismatch: {self.is_sky_filter.shape[0]} vs {self.get_xyz.shape[0]}"
        assert self.is_sky_filter_prev.shape[0] == self._xyz_prev.shape[0], f"is_sky_filter_prev size mismatch: {self.is_sky_filter_prev.shape[0]} vs {self._xyz_prev.shape[0]}"
        
        # FIX: add scale attribute consistency check
        if hasattr(self, 'next_scale'):
            assert self.next_scale.shape[0] == self.get_xyz.shape[0], f"next_scale size mismatch: {self.next_scale.shape[0]} vs {self.get_xyz.shape[0]}"
        if hasattr(self, 'prior_scale'):
            assert self.prior_scale.shape[0] == self.get_xyz.shape[0], f"prior_scale size mismatch: {self.prior_scale.shape[0]} vs {self.get_xyz.shape[0]}"
        if hasattr(self, 'now_scale'):
            assert self.now_scale.shape[0] == self.get_xyz.shape[0], f"now_scale size mismatch: {self.now_scale.shape[0]} vs {self.get_xyz.shape[0]}"
        if hasattr(self, 'next_scale_prev'):
            assert self.next_scale_prev.shape[0] == self._xyz_prev.shape[0], f"next_scale_prev size mismatch: {self.next_scale_prev.shape[0]} vs {self._xyz_prev.shape[0]}"
        if hasattr(self, 'prior_scale_prev'):
            assert self.prior_scale_prev.shape[0] == self._xyz_prev.shape[0], f"prior_scale_prev size mismatch: {self.prior_scale_prev.shape[0]} vs {self._xyz_prev.shape[0]}"
        if hasattr(self, 'now_scale_prev'):
            assert self.now_scale_prev.shape[0] == self._xyz_prev.shape[0], f"now_scale_prev size mismatch: {self.now_scale_prev.shape[0]} vs {self._xyz_prev.shape[0]}"    
    
    
    def restore_labels_from_backup(self):
        """
        When labels are accidentally lost, attempt to restore from point_labels_prev
        """
        if hasattr(self, 'point_labels_prev') and self.point_labels_prev.numel() > 0:
            print("🔄 Attempting to restore labels from point_labels_prev...")
            # call merge_all_to_trainable to restore labels
            self.merge_all_to_trainable()
            print("✅ Labels restored successfully!")
        else:
            print("❌ No labels found in point_labels_prev to restore from.")
            
    def merge_all_to_trainable(self):
        """
        After training, merge all gaussians (including non-trainable) back into main parameters
        """
        n_current = self.get_xyz.shape[0]
        n_prev = self._xyz_prev.shape[0]
        
        # if n_prev == 0:
        #     print("✅ No non-trainable points to merge")
        #     return
        
        print(f"🔄 Merging {n_prev} non-trainable points back to trainable parameters")
        
        # === merge all parameters ===
        # main model parameters
        merged_xyz = torch.cat([self._xyz.detach(), self._xyz_prev.detach()], dim=0)
        if len(self._features_dc_prev.shape) == 3 and self._features_dc_prev.shape[1] == 3 and self._features_dc_prev.shape[2] == 1:
            # if [N, 3, 1], convert to [N, 1, 3]
            self._features_dc_prev = self._features_dc_prev.permute(0, 2, 1)
        if len(self._features_dc.shape) == 3 and self._features_dc.shape[1] == 3 and self._features_dc.shape[2] == 1:
            # if [N, 3, 1], convert to [N, 1, 3]
            self._features_dc = self._features_dc.permute(0, 2, 1)
        merged_features_dc = torch.cat([self._features_dc.detach(), self._features_dc_prev.detach()], dim=0)
        merged_scaling = torch.cat([self._scaling.detach(), self._scaling_prev.detach()], dim=0)
        merged_rotation = torch.cat([self._rotation.detach(), self._rotation_prev.detach()], dim=0)
        merged_opacity = torch.cat([self._opacity.detach(), self._opacity_prev.detach()], dim=0)
        merged_focal_length = torch.cat([self._focal_length.detach(), self._focal_length_prev.detach()], dim=0)
        
        # set as trainable parameters
        self._xyz = nn.Parameter(merged_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(merged_features_dc.requires_grad_(True))
        self._scaling = nn.Parameter(merged_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(merged_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(merged_opacity.requires_grad_(True))
        self._focal_length = merged_focal_length  # no gradients needed
        
        # === merge training-related variables ===
        # max_radii2D
        if hasattr(self, 'max_radii2D') and self.max_radii2D.numel() > 0:
            if hasattr(self, 'max_radii2D_prev') and self.max_radii2D_prev.numel() > 0:
                merged_max_radii2D = torch.cat([self.max_radii2D.detach(), self.max_radii2D_prev.detach()], dim=0)
            else:
                merged_max_radii2D = torch.cat([self.max_radii2D.detach(), torch.zeros(n_prev, device='cuda')], dim=0)
        else:
            merged_max_radii2D = torch.zeros(n_current + n_prev, device='cuda')
        self.max_radii2D = merged_max_radii2D
        
        # xyz_gradient_accum
        if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.numel() > 0:
            if hasattr(self, 'xyz_gradient_accum_prev') and self.xyz_gradient_accum_prev.numel() > 0:
                merged_xyz_grad = torch.cat([self.xyz_gradient_accum.detach(), self.xyz_gradient_accum_prev.detach()], dim=0)
            else:
                merged_xyz_grad = torch.cat([self.xyz_gradient_accum.detach(), torch.zeros(n_prev, 1, device='cuda')], dim=0)
        else:
            merged_xyz_grad = torch.zeros(n_current + n_prev, 1, device='cuda')
        self.xyz_gradient_accum = merged_xyz_grad
        
        # denom
        if hasattr(self, 'denom') and self.denom.numel() > 0:
            if hasattr(self, 'denom_prev') and self.denom_prev.numel() > 0:
                merged_denom = torch.cat([self.denom.detach(), self.denom_prev.detach()], dim=0)
            else:
                merged_denom = torch.cat([self.denom.detach(), torch.zeros(n_prev, 1, device='cuda')], dim=0)
        else:
            merged_denom = torch.zeros(n_current + n_prev, 1, device='cuda')
        self.denom = merged_denom
        
        # filter_3D
        if hasattr(self, 'filter_3D') and self.filter_3D.numel() > 0:
            if hasattr(self, 'filter_3D_prev') and self.filter_3D_prev.numel() > 0:
                merged_filter_3D = torch.cat([self.filter_3D.detach(), self.filter_3D_prev.detach()], dim=0)
            else:
                merged_filter_3D = torch.cat([self.filter_3D.detach(), torch.ones(n_prev, 1, device='cuda') * 0.001], dim=0)
            self.filter_3D = merged_filter_3D

        # FIX: merge scale tensors
        # next_scale
        if hasattr(self, 'next_scale') and self.next_scale.numel() > 0:
            if hasattr(self, 'next_scale_prev') and self.next_scale_prev.numel() > 0:
                merged_next_scale = torch.cat([self.next_scale.detach(), self.next_scale_prev.detach()], dim=0)
            else:
                merged_next_scale = torch.cat([self.next_scale.detach(), torch.full((n_prev,), float('inf'), device='cuda')], dim=0)
        else:
            merged_next_scale = torch.full((n_current + n_prev,), float('inf'), device='cuda')
        self.next_scale = merged_next_scale.detach()
        
        # prior_scale  
        if hasattr(self, 'prior_scale') and self.prior_scale.numel() > 0:
            if hasattr(self, 'prior_scale_prev') and self.prior_scale_prev.numel() > 0:
                merged_prior_scale = torch.cat([self.prior_scale.detach(), self.prior_scale_prev.detach()], dim=0)
            else:
                merged_prior_scale = torch.cat([self.prior_scale.detach(), torch.full((n_prev,), float('-inf'), device='cuda')], dim=0)
        else:
            merged_prior_scale = torch.full((n_current + n_prev,), float('-inf'), device='cuda')
        self.prior_scale = merged_prior_scale.detach()
        
        # now_scale
        if hasattr(self, 'now_scale') and self.now_scale.numel() > 0:
            if hasattr(self, 'now_scale_prev') and self.now_scale_prev.numel() > 0:
                merged_now_scale = torch.cat([self.now_scale.detach(), self.now_scale_prev.detach()], dim=0)
            else:
                merged_now_scale = torch.cat([self.now_scale.detach(), torch.full((n_prev,), 0.01, device='cuda')], dim=0)
        else:
            merged_now_scale = torch.full((n_current + n_prev,), 0.01, device='cuda')
        self.now_scale = merged_now_scale.detach()
        
        # === merge mask attributes ===
        # is_sky_filter: merge trainable and non-trainable parts
        if hasattr(self, 'is_sky_filter') and self.is_sky_filter.numel() > 0:
            if hasattr(self, 'is_sky_filter_prev') and self.is_sky_filter_prev.numel() > 0:
                merged_is_sky_filter = torch.cat([self.is_sky_filter.detach(), self.is_sky_filter_prev.detach()], dim=0)
            else:
                merged_is_sky_filter = torch.cat([self.is_sky_filter.detach(), torch.zeros(n_prev, dtype=torch.bool, device='cuda')], dim=0)
        else:
            merged_is_sky_filter = torch.zeros(n_current + n_prev, dtype=torch.bool, device='cuda')
        self.is_sky_filter = merged_is_sky_filter
        print(self.is_sky_filter.shape)
        
        # visibility_filter_all: should already contain all points, just ensure correct size
        if hasattr(self, 'visibility_filter_all') and self.visibility_filter_all.numel() > 0:
            if self.visibility_filter_all.shape[0] != (n_current + n_prev):
                print(f"⚠️  visibility_filter_all size mismatch: {self.visibility_filter_all.shape[0]} vs {n_current + n_prev}")
                # if size is wrong, create new default values
                self.visibility_filter_all = torch.zeros(n_current + n_prev, dtype=torch.bool, device='cuda')
        else:
            self.visibility_filter_all = torch.zeros(n_current + n_prev, dtype=torch.bool, device='cuda')
        
        # delete_mask_all: should already contain all points, just ensure correct size
        if hasattr(self, 'delete_mask_all') and self.delete_mask_all.numel() > 0:
            if self.delete_mask_all.shape[0] != (n_current + n_prev):
                print(f"⚠️  delete_mask_all size mismatch: {self.delete_mask_all.shape[0]} vs {n_current + n_prev}")
                # if size is wrong, create new default values
                self.delete_mask_all = torch.zeros(n_current + n_prev, dtype=torch.bool, device='cuda')
        else:
            self.delete_mask_all = torch.zeros(n_current + n_prev, dtype=torch.bool, device='cuda')
        
        # === clear all prev parameters ===
        self._xyz_prev = torch.empty(0, 3, device='cuda')
        self._features_dc_prev = torch.empty(0, 1, 3, device='cuda')
        self._scaling_prev = torch.empty(0, 3, device='cuda')
        self._rotation_prev = torch.empty(0, 4, device='cuda')
        self._opacity_prev = torch.empty(0, 1, device='cuda')
        self._focal_length_prev = torch.empty(0, device='cuda')
        
        # clear prev training variables
        self.max_radii2D_prev = torch.empty(0, device='cuda')
        self.xyz_gradient_accum_prev = torch.empty(0, 1, device='cuda')
        self.denom_prev = torch.empty(0, 1, device='cuda')
        self.filter_3D_prev = torch.empty(0, 1, device='cuda')
        
        # FIX: clear prev scale tensors
        self.next_scale_prev = torch.empty(0, device='cuda')
        self.prior_scale_prev = torch.empty(0, device='cuda')
        self.now_scale_prev = torch.empty(0, device='cuda')
        
        # === merge point labels ===
        # merge current and prev labels into point_labels
        if hasattr(self, 'point_labels') and hasattr(self, 'point_labels_prev'):
            if self.point_labels.numel() > 0 and self.point_labels_prev.numel() > 0:
                merged_labels = torch.cat([self.point_labels.detach(), self.point_labels_prev.detach()], dim=0)
            elif self.point_labels.numel() > 0:
                merged_labels = self.point_labels.detach()
            elif self.point_labels_prev.numel() > 0:
                merged_labels = self.point_labels_prev.detach()
            else:
                merged_labels = torch.zeros(n_current + n_prev, dtype=torch.long, device='cuda')
            self.point_labels = merged_labels
        

        
        # clear prev point_labels
        self.point_labels_prev = torch.empty(0, dtype=torch.long, device='cuda')
        
        # clear prev mask variables
        self.is_sky_filter_prev = torch.empty(0, dtype=torch.bool, device='cuda')
        self.visibility_filter_all_prev = torch.empty(0, dtype=torch.bool, device='cuda')
        self.delete_mask_all_prev = torch.empty(0, dtype=torch.bool, device='cuda')
        
        # === re-setup optimizer (if needed) ===
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if hasattr(self, '_last_training_args'):
                print("🔄 Re-setting up optimizer with merged parameters...")
                self.training_setup(self._last_training_args)
            else:
                print("⚠️  Optimizer exists but no training_args saved. You may need to manually call training_setup().")
        
        print(f"✅ Merge complete: {self.get_xyz.shape[0]} total trainable points, 0 non-trainable points")