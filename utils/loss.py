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
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from simple_knn._C import distCUDA2

def scaling_regularization_loss(gaussians, camera, lambda_scale=10000, base_threshold=0.01, distance_factor=1.0):
    """
    Efficient scaling regularization loss
    Compute adaptive scale threshold based on camera distance, no need for nearest neighbor computation

    Args:
        gaussians: GaussianModel instance
        camera: current camera object
        lambda_scale: regularization strength (recommended 0.01-0.05)
        base_threshold: base scale threshold for nearby points (recommended 0.005-0.02)
        distance_factor: distance factor, farther points allow larger scale (recommended 1-3)

    Returns:
        loss: torch.Tensor, directly added to the main loss
    """
    if gaussians.get_xyz.shape[0] == 0:
        return torch.tensor(0.0, device='cuda')
    
    xyz = gaussians.get_xyz
    scales = gaussians.get_scaling
    max_scales = scales.max(dim=1).values
    
    # Compute point-to-camera distance - fast!
    R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
    T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
    xyz_cam = xyz @ R + T[None, :]
    distances = torch.norm(xyz_cam, dim=1)
    
    # Adaptive threshold based on distance: farther points allow larger scale
    adaptive_thresholds = base_threshold * (1.0 + distances * distance_factor)
    
    # Compute scale exceeding the threshold
    excess_scales = torch.clamp(max_scales - adaptive_thresholds, min=0.0)
    
    # Normalize penalty to avoid numerical explosion
    normalized_excess = excess_scales / (adaptive_thresholds + 1e-7)
    loss = lambda_scale * torch.mean(normalized_excess ** 2)
    
    return loss

def scaling_bidirectional_loss(gaussians,
                               camera,
                               rotation2normal_fn,
                               lambda_scale=1.0,
                               q_low=0.8,
                               q_high=1.2,
                               w_under=1.5,
                               w_over=1.0,
                               huber_delta=0.05,
                               center_push=0.0,
                               use_opacity_weight=False,
                               min_cos=0.05):
    """
    Bidirectional scale regularization: q = current/target
    - q < q_low  (too small) -> under penalty
    - q > q_high (too large) -> over penalty
    No penalty in between
    """
    xyz = gaussians.get_xyz
    if xyz.shape[0] == 0:
        return torch.tensor(0.0, device=xyz.device)

    scales = gaussians.get_scaling  # assumed to be linear scale [N,3]
    rotations = gaussians.get_rotation  # for normals
    # (if get_scaling returns log: scales = torch.exp(scales))

    with torch.no_grad():
        _, _, q = _target_scale(camera, xyz, rotations, scales, rotation2normal_fn, min_cos=min_cos)

    # under
    under_mask = q < q_low
    over_mask  = q > q_high

    # Compute deviation d
    d_under = (q_low - q).clamp(min=0.0)   # positive value
    d_over  = (q - q_high).clamp(min=0.0)

    # Huber penalty
    def huber(x, delta):
        # x >=0
        return torch.where(x <= delta,
                           0.5 * (x * x) / delta,
                           x - 0.5 * delta)

    under_loss = huber(d_under[under_mask], huber_delta)
    over_loss  = huber(d_over[over_mask], huber_delta)

    if use_opacity_weight:
        op = gaussians.get_opacity
        if op.shape[-1] != 1:
            opw = op.squeeze(-1)
        else:
            opw = op[:,0]
        # (only grab weights for those in the mask)
        under_loss = under_loss * opw[under_mask].detach()
        over_loss  = over_loss  * opw[over_mask].detach()

    loss_under = (under_loss.mean() if under_loss.numel() else torch.tensor(0.0, device=xyz.device))
    loss_over  = (over_loss.mean()  if over_loss.numel()  else torch.tensor(0.0, device=xyz.device))

    loss = lambda_scale * (w_under * loss_under + w_over * loss_over)

    if center_push > 0.0:
        # Mild centering: optional, only for q within the interval
        mid_mask = (~under_mask) & (~over_mask)
        if mid_mask.any():
            mid_q = q[mid_mask]
            loss_center = ((mid_q - 1.0)**2).mean()
            loss = loss + center_push * loss_center

    return loss

@torch.no_grad()
def _target_scale(camera, xyz, rotations, scales, rotation2normal_fn, min_cos=0.05):
    """
    xyz: [N,3] world coordinates
    rotations: corresponding to xyz
    scales: [N,3] linear scale (if log, please exp first)
    Returns s_target, s_cur, q
    """
    if xyz.numel() == 0:
        z = torch.empty(0, device=xyz.device)
        return z, z, z

    dev = xyz.device
    R = torch.tensor(camera.R, device=dev, dtype=torch.float32)
    T = torch.tensor(camera.T, device=dev, dtype=torch.float32)

    xyz_cam = xyz @ R + T[None,:]
    zc = xyz_cam[:,2].clamp_min(1e-4)

    normals_world = rotation2normal_fn(rotations)
    normals_cam = normals_world @ R
    nx, ny, nz = normals_cam[:,0], normals_cam[:,1], normals_cam[:,2]
    eps = 1e-8

    cos_xz = (nx.abs() / (nx*nx + nz*nz + eps).sqrt()).clamp(min=min_cos)
    cos_yz = (ny.abs() / (ny*ny + nz*nz + eps).sqrt()).clamp(min=min_cos)

    fx = camera.focal_x
    fy = getattr(camera, 'focal_y', fx)

    s_xt = zc / fx / cos_xz
    s_yt = zc / fy / cos_yz
    s_target = (s_xt * s_yt).sqrt()

    s_cur = scales.max(dim=1).values
    q = s_cur / (s_target + eps)
    return s_target, s_cur, q

# def scaling_regularization_loss(gaussians, lambda_scale=0.05, max_scale_factor=1.5):
#     """
#     Simple scaling regularization loss
#     Compute expected scale based on nearest neighbor distance, penalize overly large scale

#     Args:
#         gaussians: GaussianModel instance
#         lambda_scale: regularization strength (recommended 0.01-0.05)
#         max_scale_factor: maximum allowed scale multiplier relative to neighbor distance (recommended 5-15)

#     Returns:
#         loss: torch.Tensor, directly added to the main loss
#     """
#     if gaussians.get_xyz.shape[0] < 2:
#         return torch.tensor(0.0, device='cuda')
    
#     # Get current trainable points
#     xyz = gaussians.get_xyz
#     scales = gaussians.get_scaling
#     max_scales = scales.max(dim=1).values
    
#     # Compute nearest neighbor distance as expected scale reference
#     dist2 = torch.clamp_min(distCUDA2(xyz), 1e-7)
#     expected_scales = torch.sqrt(dist2)
    
#     # Compute scale multiplier relative to expected value
#     scale_ratios = max_scales / expected_scales
    
#     # Only penalize points exceeding the threshold: loss = (ratio - threshold)^2
#     excess_ratios = torch.clamp(scale_ratios - max_scale_factor, min=0.0)
#     loss = lambda_scale * torch.mean(excess_ratios ** 2)
    
#     return loss
        
    # except Exception as e:
    #     # If an error occurs, return 0 (e.g., CUDA out of memory, etc.)
    #     return torch.tensor(0.0, device='cuda')

def l1_loss(network_output, gt, no_loss_mask=None):
    if no_loss_mask is not None:
        no_loss_mask_expand = no_loss_mask.expand(gt.shape).bool()
        return (torch.abs((network_output - gt))[~no_loss_mask_expand]).mean()
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True, no_loss_mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, no_loss_mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, no_loss_mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if no_loss_mask is not None:
        no_loss_mask_expand = no_loss_mask.expand(ssim_map.shape).bool()
        ssim_map = ssim_map[~no_loss_mask_expand]

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


import numpy as np
import cv2
def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()

with torch.no_grad():
    kernelsize=3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize//2))
    kernel = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).reshape(1,1,kernelsize,kernelsize)
    conv.weight.data = kernel #torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None,None])
    cnt_map = conv((cnt_map * mask)[None,None])
    nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
    return nearMean_map


def anisotropy_regularizer(gaussian_model, r_threshold=4.0):
    """
    Anisotropy Regularizer for 3D Gaussian reconstruction.
    
    This regularizer constrains the ratio between the major axis length 
    and minor axis length of 3D Gaussians to prevent over-skinny kernels
    that may point outward from the object surface under large deformations.
    
    Formula: L_aniso = (1/|P|) * Σ_{p∈P} max{max(S_p) / min(S_p), r} - r
    
    Args:
        gaussian_model: GaussianModel instance containing the 3D Gaussian parameters
        r_threshold: float, threshold parameter r (default: 10.0)
        
    Returns:
        torch.Tensor: anisotropy regularization loss
    """
    # Get the scaling parameters of all 3D Gaussians
    scalings = gaussian_model.get_scaling  # Shape: [N, 3], where N is number of Gaussians
    
    # Compute max and min scaling values for each Gaussian
    max_scaling = torch.max(scalings, dim=1)[0]  # Shape: [N]
    min_scaling = torch.min(scalings, dim=1)[0]  # Shape: [N]
    
    # Compute the ratio max(S_p) / min(S_p) for each Gaussian
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    aspect_ratios = max_scaling / (min_scaling + epsilon)  # Shape: [N]
    
    # Apply the max operation with threshold r
    # max{max(S_p) / min(S_p), r} - r
    regularization_terms = torch.clamp(aspect_ratios, min=r_threshold) - r_threshold
    
    # Compute the mean over all Gaussians: (1/|P|) * Σ_{p∈P} 
    anisotropy_loss = torch.mean(regularization_terms)
    
    return anisotropy_loss