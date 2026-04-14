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

import torch
import math
from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh import eval_sh
import torch.nn.functional as F
from utils.general import build_rotation, rotation2normal
from pytorch3d.renderer import PerspectiveCameras 

# from util.utils import convert_pt3d_cam_to_3dgs_cam

# @torch.no_grad()
# def get_projection_render_mask(camera, gaussians, min_pixel_threshold=0.3):
    # """
    # Very simple projection size filtering function

    # Args:
    #     camera: 3DGS camera object
    #     gaussians: GaussianModel object
    #     min_pixel_threshold: Minimum pixel threshold, default 1.5
    #     xyz_scale: Coordinate scale factor, default 1000

    # Returns:
    #     mask: bool tensor [N], True means should be rendered
    # """
    # # Get gaussian data
    # xyz = gaussians.get_xyz_all  # [N, 3]
    # scales = gaussians.get_scaling_with_3D_filter_all.max(dim=1).values  # [N]
    
    # # Transform to camera space
    # R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
    # T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
    # xyz_cam = xyz @ R + T[None, :]
    # distances = xyz_cam[:, 2] 
    
    # # Compute projected size and filter
    # valid_mask = distances > 1e-5  # In front of camera
    # projected_sizes = scales * camera.focal_x / (distances + 1e-6)
    # render_mask = (projected_sizes > min_pixel_threshold) & valid_mask
    # # print(projected_sizes,projected_sizes.min(),projected_sizes.median())
    # # print(projected_sizes[render_mask],projected_sizes[render_mask].min(),projected_sizes[render_mask].median())
    # return render_mask



# @torch.no_grad()
# def compute_surface_lod_blend_weight(surface_ids, q, s3=1.4, s4=1.8):
#     """
#     Fade out (opacity attenuation) points with excessively large scale, based on whether there are better points with the same surface_id.
    
#     Args:
#         surface_ids: [N] LongTensor
#         q: [N] current point scale / target scale
#         s3, s4: fade-out start/end thresholds (s3 starts penalty, s4 fully removes)
        
#     Returns:
#         w: [N] blending weight (0~1)
#     """
#     N = q.shape[0]
#     w = torch.ones_like(q)

#     if N == 0:
#         return w

#     # group by surface_id
#     from torch_scatter import scatter_min
#     d = (q - 1.0).abs()
#     _, argmin_idx = scatter_min(d, surface_ids, dim=0)

#     is_winner = torch.zeros(N, dtype=torch.bool, device=q.device)
#     is_winner[argmin_idx] = True

#     # Whether to penalize (only for non-winners)
#     need_penalty = (~is_winner) & (q > s3)

#     # Linear fade-out weight
#     penalty_zone = (q > s3) & (q < s4)
#     penalty_weight = 1.0 - (q - s3) / (s4 - s3 + 1e-6)

#     w[need_penalty & penalty_zone] = penalty_weight[need_penalty & penalty_zone]
#     w[need_penalty & (q >= s4)] = 0.0  # Directly remove

#     return w.clamp(0.0, 1.0)


def compute_log_scale_weights(current_scale, prior_scale, next_scale):
    # Convert to log space
    log_current = torch.log(current_scale)
    log_prior = torch.log(prior_scale)
    log_next = torch.log(next_scale)

    # Construct mask to select elements that satisfy the condition
    valid_mask = (log_current >= log_prior) & (log_current <= log_next)

    # Initialize weights to 0 (for points that don't satisfy the condition)
    weight_prior = torch.zeros_like(log_current)
    weight_next = torch.zeros_like(log_current)

    # Compute t, only under the valid_mask condition
    t = torch.zeros_like(log_current)
    denom = log_next - log_prior
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)  # Prevent division by zero
    t[valid_mask] = (log_current[valid_mask] - log_prior[valid_mask]) / denom[valid_mask]
    t = torch.clamp(t, 0.0, 1.0)

    # Interpolation weights
    weight_prior[valid_mask] = 1.0 - t[valid_mask]
    weight_next[valid_mask] = t[valid_mask]

    prior_inf_mask = (prior_scale == (-torch.inf)) & (current_scale <= next_scale)
    next_inf_mask = (next_scale == (torch.inf)) & (current_scale >= prior_scale)
    weight_prior[next_inf_mask ] = 1.0
    weight_next[prior_inf_mask] = 1.0
    
    return weight_prior, weight_next

def render(viewpoint_camera, pc: GaussianModel, opt, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, render_visible=False, 
           exclude_sky=False,render_normals=False, render_dominant_ids=True, filter_scale = True, config=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    if pc.delete_mask_all.sum() > 0:
        pc.prune_points(pc.delete_mask_all)
        
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz_all.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means3D = pc.get_xyz_all
    means2D = screenspace_points
    # opacity = pc.get_opacity_with_3D_filter
    opacity = pc.get_opacity_all
    # opacity = pc.get_opacity_with_3D_filter_all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        # scales = pc.get_scaling_with_3D_filter
        # rotations = pc.get_rotation
        # scales = pc.get_scaling_with_3D_filter_all
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            # dir_pp = (pc.get_xyz_all - viewpoint_camera.camera_center.repeat(pc.get_features_all.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = pc.color_activation(shs_view)
        else:
            # shs = pc.get_features
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    # if render_visible:
    #     visibility_filter_all = pc.gaussians.max_radii2D #& ~pc.delete_mask_all  # Seen in screen
    # else:
    focals = pc.get_focal_length_all.unique()
    focal_0_mask = pc.get_focal_length_all>= focals[0]
# focal_0_mask = gaussians.get_focal_length_all == focals[0]
# focal_1_mask = gaussians.get_focal_length_all == focals[1]
# focal_2_mask = gaussians.get_focal_length_all == focals[2]
# focal_3_mask = gaussians.get_focal_length_all == focals[3]
# focal_4_mask = gaussians.get_focal_length_all == focals[4]
    visibility_filter_all = ~pc.delete_mask_all 

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter

    # New: Focal length filtering - only render points with focal_length < camera.focal_x
    # focal_length_filter = pc.get_focal_length_all.squeeze()/2 < viewpoint_camera.focal_x
    # filtered_visibility = visibility_filter_all & focal_length_filter
    filtered_visibility = visibility_filter_all #& get_projection_render_mask(viewpoint_camera, pc)
    # print(f"📏 Focal filtering: camera focal_x={viewpoint_camera.focal_x:.2f}, "
    #       f"filtered {focal_length_filter.sum()}/{len(focal_length_filter)} points")

    means3D = means3D[filtered_visibility]
    means2D = means2D[filtered_visibility]
    next_scale = pc.get_next_scale_all[filtered_visibility]
    prior_scale = pc.get_prior_scale_all[filtered_visibility]
    now_scale = pc.get_now_scale_all[filtered_visibility]
    # next_sqrt = torch.sqrt(next_scale*now_scale)
    # prior_sqrt = torch.sqrt(prior_scale*now_scale)
    # prior_sqrt[torch.isnan(prior_sqrt)] = - torch.inf
    
    shs = None if shs is None else shs[filtered_visibility]
    colors_precomp = None if colors_precomp is None else colors_precomp[filtered_visibility]
    opacity = opacity[filtered_visibility]
    scales = scales[filtered_visibility]
    rotations = rotations[filtered_visibility]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[filtered_visibility]
    
    if getattr(opt, 'lod_q_enable', True):
        # q, q_fd = compute_inv_target_scale_per_frame(
        #     viewpoint_camera,
        #     means3D,
        #     rotations,
        #     scales,
        #     rotation2normal_fn=rotation2normal,
        #     min_cos=getattr(opt, 'lod_q_min_cos', 0.1)
        # )
        _, q_fd = compute_inv_target_scale_per_frame(
            viewpoint_camera,
            means3D,
            min_z=1e-6,
            config=config
        )

        # Small scale attenuation (original logic)
        # s1 = getattr(opt, 'lod_q_s1', 0.5)
        # s2 = getattr(opt, 'lod_q_s2', 0.05)
        # if s2 >= s1: s2 = s1 - 1e-4
        # w_small = lod_weight_from_q(q, s1=s1, s2=s2, smooth=getattr(opt,'lod_q_smooth',True))

        # # Large scale attenuation (new surface_id winner-based)
        # s3 = getattr(opt, 'lod_q_s3', 1.0)
        # s4 = getattr(opt, 'lod_q_s4', 0.5)
        # # surface_ids = pc.get_surface_ids_all[filtered_visibility]
        # # w_large = compute_surface_lod_blend_weight(surface_ids, q, s3=s3, s4=s4)
        # w_large = lod_weight_from_q_reverse((q_fd/next_scale), s1=s3, s2=s4, smooth=getattr(opt,'lod_q_smooth',True))
        # w_prior = lod_weight_from_q((q_fd/prior_scale), s1=1.5, s2=1.0, smooth=getattr(opt,'lod_q_smooth',True)) + (prior_scale == (-torch.inf)).float()

        # # Blend the opacity attenuation from both directions
        # w = torch.min(w_small, w_large)
        # w = torch.min(w, w_prior)
        if filter_scale:
            w_prior_1, w_next_now = compute_log_scale_weights(q_fd, prior_scale, now_scale)
            w_prior_now, w_next_1 = compute_log_scale_weights(q_fd, now_scale, next_scale)
            # w = torch.max(w_next_now + (prior_scale == (-torch.inf)).float(), w_prior_now + (next_scale == (torch.inf)).float())
            w = torch.max(w_next_now , w_prior_now)
            

            # Apply to opacity
            opacity = opacity * w.unsqueeze(1)

            # (Optional) Filter out points with too small opacity
            scale_filter = w > 1e-2
            scale_filter = scale_filter & focal_0_mask
        else:
            scale_filter = torch.ones_like(filtered_visibility, dtype=torch.bool)
        if scale_filter.any():
            scales = scales[scale_filter]
            means3D = means3D[scale_filter]
            means2D = means2D[scale_filter]
            opacity = opacity[scale_filter]
            rotations = rotations[scale_filter]
            shs = None if shs is None else shs[scale_filter]
            colors_precomp = None if colors_precomp is None else colors_precomp[scale_filter]
            cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[scale_filter]
            # q = q[scale_filter]
            # surface_ids = surface_ids[scale_filter]
            filtered_visibility = filtered_visibility & scale_filter        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii_filtered, depth, median_depth, final_opacity, dominant_ids = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Map the filtered radii back to the original size
    radii = torch.zeros(len(visibility_filter_all), device=radii_filtered.device, dtype=radii_filtered.dtype)
    radii[filtered_visibility] = radii_filtered
    
    # Create the correctly sized visibility_filter
    visibility_filter = filtered_visibility & (radii > 0)

    render_normal = None
    if render_normals:
        # Render world-space normal map (H x W x 3) - directly compatible with create_from_pcd!
        point_normals_in_world = rotation2normal(rotations)

        render_normal, _, _, _, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=point_normals_in_world,  # Directly use world-space normals!
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        render_normal = F.normalize(render_normal, dim=0)        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # Process dominant_ids - map back to original gaussian indices
    dominant_ids_output = None
    if render_dominant_ids:
        if dominant_ids is not None:
            # Create mapping from filtered indices to original indices
            # Step 1: Get original indices after the first filtering
            original_indices = torch.arange(len(visibility_filter_all), device=filtered_visibility.device)[filtered_visibility].int()
            
            
            # Step 2: If there is a second filtering (scale_filter), map further
            # if 'scale_filter' in locals() and scale_filter is not None and scale_filter.any():
            #     original_indices = original_indices[scale_filter]
            
            # Step 3: Map filtered indices in dominant_ids back to original indices (efficient version)
            # Each value in dominant_ids is an index into the final means3D, needs mapping back to original point indices
            flat_dominant_ids = dominant_ids.flatten().int()
            
            # Create safe mapping: out-of-range indices remain unchanged
            valid_mask = flat_dominant_ids < len(original_indices)
            mapped_flat = torch.zeros_like(flat_dominant_ids)
            mapped_flat[valid_mask] = original_indices[flat_dominant_ids[valid_mask]]
            mapped_flat[~valid_mask] = flat_dominant_ids[~valid_mask]  # Keep original value
            
            dominant_ids_output = mapped_flat.reshape(dominant_ids.shape)
        else:
            dominant_ids_output = dominant_ids

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : visibility_filter,
            "radii": radii,
            "final_opacity": final_opacity,
            "depth": depth,
            "median_depth": median_depth,
            "render_normal": render_normal,
            "dominant_ids": dominant_ids_output,
            # "w": w[scale_filter],
            }
    

@torch.no_grad()
def compute_target_scale_per_frame(cam, means3D, rotations, scales, rotation2normal_fn, min_cos=0.05, config=None):
    """
    means3D: [M,3] world coordinates (the batch of points after filtering)
    rotations: same batch rotations (used to compute normals)
    scales: [M,3] linear scale (if log, apply exp first)
    return: q (current/target)
    """
    # global xyz_scale
    if isinstance(cam, PerspectiveCameras):
        from util.utils import convert_pt3d_cam_to_3dgs_cam
        cam = convert_pt3d_cam_to_3dgs_cam(cam, xyz_scale=1000, config=config)
        
    if means3D.numel() == 0:
        return torch.empty(0, device=means3D.device)

    dev = means3D.device
    R = torch.tensor(cam.R, device=dev, dtype=torch.float32)
    T = torch.tensor(cam.T, device=dev, dtype=torch.float32)
    xyz_cam = means3D @ R + T[None, :]
    z = xyz_cam[:, 2].clamp_min(1e-4)

    # Normal (world -> camera)
    # normals_world = rotation2normal_fn(rotations)          # [M,3]
    # normals_cam = normals_world @ R
    # nx, ny, nz = normals_cam[:,0], normals_cam[:,1], normals_cam[:,2]
    eps = 1e-8

    # cos_xz = (nx.abs() / (nx*nx + nz*nz + eps).sqrt()).clamp(min=min_cos)
    # cos_yz = (ny.abs() / (ny*ny + nz*nz + eps).sqrt()).clamp(min=min_cos)

    fx = cam.focal_x
    fy = getattr(cam, 'focal_y', fx)

    s_x_t = z / fx #/ cos_xz
    s_y_t = z / fy #/ cos_yz
    s_target = (s_x_t * s_y_t).sqrt()          # Equivalent target scale

    s_cur = scales.max(dim=1).values           # Current largest principal axis
    q = s_cur / (s_target + eps)
    return q, 1/s_target


@torch.no_grad()
def compute_inv_target_scale_per_frame(cam, means3D, *, xyz_scale=1000.0, min_z=1e-6, config=None):
    """
    Minimal/stable version: only computes inv_s_target = 1 / s_target = fx / z
    - cam: 3DGS-style camera (with R, T, focal_x), or PyTorch3D PerspectiveCameras
    - means3D: [M, 3] world coordinates (already filtered)
    - xyz_scale: scale used for conversion when cam is a PyTorch3D camera (consistent with the project)
    - min_z: lower bound for z, to avoid division by zero / negative values
    - config: configuration object containing orig_H and orig_W

    return:
        inv_s_target: [M], equals fx / z
        (if s_target is needed, use 1.0 / inv_s_target)
    """
    # Compatible with Pytorch3D cameras
    if isinstance(cam, PerspectiveCameras):
        from util.utils import convert_pt3d_cam_to_3dgs_cam
        cam = convert_pt3d_cam_to_3dgs_cam(cam, xyz_scale=1000., config=config)

    if means3D.numel() == 0:
        return torch.empty(0, device=means3D.device, dtype=means3D.dtype)

    dev, dtype = means3D.device, means3D.dtype

    # Use the same dtype as means3D to avoid casts from float32/float64 mixing
    R = torch.as_tensor(cam.R, device=dev, dtype=dtype)        # [3,3]
    T = torch.as_tensor(cam.T, device=dev, dtype=dtype)        # [3]
    fx = torch.as_tensor(getattr(cam, "focal_x"), device=dev, dtype=dtype)

    # World -> camera, then extract z; clip with lower bound to ensure numerical stability
    z = (means3D @ R + T).select(dim=1, index=2).clamp_min(min_z)

    inv_s_target = fx / z  # = 1 / (z / fx)
    return None, inv_s_target


# def lod_weight_from_q(q, s1=0.8, s2=0.5, smooth=True):
#     """
#     s2 < s1
#     q < s2 -> 0;  s2<=q<s1 -> 0..1;  q>=s1 ->1
#     """
#     if q.numel() == 0:
#         return q
#     w = torch.zeros_like(q)
#     mid = (q >= s2) & (q < s1)
#     if mid.any():
#         t = (q[mid] - s2) / (s1 - s2 + 1e-8)
#         if smooth:
#             t = t*t*(3 - 2*t)  # smoothstep
#         w[mid] = t
#     w[q >= s1] = 1.0
#     return w

# def lod_weight_from_q_reverse(q, s1=30.0, s2=20.0, smooth=True):
#     """
#     Reverse version:
#     q > s1 -> 0;  s2 <= q <= s1 -> 1..0;  q < s2 -> 1
#     """
#     if q.numel() == 0:
#         return q
#     w = torch.ones_like(q)
#     mid = (q > s2) & (q <= s1)
    
#     if mid.any():
#         t = (q[mid] - s2) / (s1 - s2 + 1e-8)
#         if smooth:
#             t = t*t*(3 - 2*t)  # smoothstep
#         w[mid] = 1.0 - t
#     w[q > s1] = 0.0
#     return w





