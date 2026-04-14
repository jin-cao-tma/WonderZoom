from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np
from gaussian_renderer import render

import scipy
import scipy.signal
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import torch
import io
import logging
from pathlib import Path
from tqdm import tqdm
from collections import deque
from torchvision.transforms import ToTensor
import os
import yaml
import shutil
# from .general_utils import save_video
from datetime import datetime
from pytorch3d.renderer import PerspectiveCameras
from datetime import datetime
from diffusers.configuration_utils import FrozenDict
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from scene.cameras import Camera

import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix


import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from torchvision.io import write_video
from pytorch3d.renderer.cameras import PerspectiveCameras

import torch
import torch.nn.functional as F
from pytorch3d.renderer.cameras import PerspectiveCameras



def compute_trajectory_distances(trajectory_cameras, use_focal=False):
    """
    Compute distances between all adjacent cameras in the trajectory, with separate normalization for rotation, translation, and focal length

    Args:
        trajectory_cameras: List of cameras
        use_focal: bool, whether to consider focal length changes

    Returns:
        list: Normalized combined distance list
    """
    import torch
    
    rotation_distances = []
    translation_distances = []
    focal_distances = []
    
    # 1. Compute raw distances between all adjacent poses
    for i in range(len(trajectory_cameras) - 1):
        cam1 = trajectory_cameras[i]
        cam2 = trajectory_cameras[i + 1]
        
        # Translation distance
        center1 = cam1.get_camera_center()  # [1, 3]
        center2 = cam2.get_camera_center()  # [1, 3]
        translation_dist = torch.norm(center2 - center1).item()
        translation_distances.append(translation_dist)
        
        # Rotation distance
        R1 = cam1.R[0]  # [3, 3]
        R2 = cam2.R[0]  # [3, 3]
        R_rel = torch.matmul(R2, R1.transpose(0, 1))
        trace = torch.trace(R_rel)
        cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        rotation_angle = torch.acos(cos_theta).item()  # radians
        rotation_distances.append(rotation_angle)
        
        # Focal length distance (if enabled)
        if use_focal:
            focal1 = cam1.K[0, 0, 0].item()  # fx
            focal2 = cam2.K[0, 0, 0].item()  # fx
            # Use log difference to compute focal length distance
            focal_dist = abs(torch.log(torch.tensor(focal2)) - torch.log(torch.tensor(focal1))).item()
            focal_distances.append(focal_dist)
    
    # 2. Normalize rotation, translation, and focal distances separately
    max_rotation = max(rotation_distances) if max(rotation_distances) > 0 else 1.0
    max_translation = max(translation_distances) if max(translation_distances) > 0 else 1.0
    
    normalized_rotation = [r / max_rotation for r in rotation_distances]
    normalized_translation = [t / max_translation for t in translation_distances]
    
    if use_focal and focal_distances:
        max_focal = max(focal_distances) if max(focal_distances) > 0 else 1.0
        normalized_focal = [f / max_focal for f in focal_distances]
    else:
        normalized_focal = []
    
    # 3. Combine normalized distances (weights can be adjusted)
    if use_focal and normalized_focal:
        rotation_weight = 1/3
        translation_weight = 1/3
        focal_weight = 1/3
    else:
        rotation_weight = 0.5
        translation_weight = 0.5
        focal_weight = 0.0
    
    combined_distances = []
    for i in range(len(normalized_rotation)):
        if use_focal and normalized_focal:
            combined_dist = (rotation_weight * normalized_rotation[i] + 
                            translation_weight * normalized_translation[i] + 
                            focal_weight * normalized_focal[i])
        else:
            combined_dist = (rotation_weight * normalized_rotation[i] + 
                            translation_weight * normalized_translation[i])
        combined_distances.append(combined_dist)
    
    print(f"Rotation distances: {rotation_distances}")
    print(f"Translation distances: {translation_distances}")
    if use_focal and focal_distances:
        print(f"Focal distances: {focal_distances}")
        print(f"Normalized focal: {normalized_focal}")
    print(f"Normalized rotation: {normalized_rotation}")
    print(f"Normalized translation: {normalized_translation}")
    print(f"Combined distances: {combined_distances}")
    
    return combined_distances
# --------- Basic: Look-At Rotation --------------------------------------------------

def look_at_rotation(eye: torch.Tensor,
                     at: torch.Tensor,
                     up: torch.Tensor) -> torch.Tensor:
    """
    Generate rotation matrix R that makes the camera look from eye to at point (batch version)
    Args:
        eye (N, 3) Camera position
        at  (N, 3) Target point
        up  (N, 3) World up direction
    Returns:
        R   (N, 3, 3) Rotation matrix (column vectors are right, up, forward respectively)
    """
    z_axis = F.normalize(at - eye, dim=-1, eps=1e-6)           # forward
    x_axis = F.normalize(torch.cross(up, z_axis, dim=-1),
                         dim=-1, eps=1e-6)                     # right
    y_axis = torch.cross(z_axis, x_axis, dim=-1)               # true up
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)          # (N,3,3)
    return R


# --------- Orbit: Around z-axis ------------------------------------------------------
def orbit_camera_about_z(cameras: PerspectiveCameras,
                         angles,
                         radius=1.0) -> PerspectiveCameras:
    """
    Make the camera orbit in the xy plane (around z-axis) while always looking at its original center point
    Args:
        cameras : Existing PerspectiveCameras (batch=N)
        angles  : float or (N,) Tensor, in radians
        radius  : float or (N,) Tensor, orbit radius
    Returns:
        new_cams: New cameras (same batch size)
    """
    device = cameras.device
    eye0 = cameras.get_camera_center()          # (N,3) Original camera position
    at   = eye0.clone()                         # Target = original position

    # Convert scalars to Tensor, auto broadcast
    if not torch.is_tensor(angles):
        angles = torch.tensor(angles, device=device)
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=device)
    angles = angles.view(-1).expand(eye0.shape[0])
    radius = radius.view(-1).expand(eye0.shape[0])

    # New eye: only change x, y, keep z unchanged
    eye = eye0.clone()
    eye[:, 0] = at[:, 0] + radius * torch.cos(angles)   # x
    eye[:, 1] = at[:, 1] + radius * torch.sin(angles)   # y

    # Generate R, T
    up = torch.tensor([0., 1., 0.], device=device).expand_as(eye)
    R = look_at_rotation(eye, at, up)                   # (N,3,3)
    T = -torch.bmm(R, eye.unsqueeze(-1)).squeeze(-1)    # (N,3)

    # Copy intrinsics & update extrinsics
    new_cams = cameras.clone()
    new_cams.R = R
    new_cams.T = T
    return new_cams

def compute_pose_distances(cur_camera, camera_list):
    """
    Compute rotation and translation distances between cur_camera and each camera in camera_list.

    Args:
        cur_camera: a PerspectiveCameras instance
        camera_list: list of PerspectiveCameras

    Returns:
        closest_idx: index of most similar camera
        rot_dists: list of rotation distances (in radians)
        trans_dists: list of translation distances
    """
    R1 = cur_camera.R[0]  # [3, 3]
    T1 = cur_camera.T[0]  # [3]

    rot_dists = []
    trans_dists = []

    for cam in camera_list:
        R2 = cam.R[0]
        T2 = cam.T[0]

        # Rotation distance
        R_rel = R1.T @ R2
        try:
            rotvec = R.from_matrix(R_rel.cpu().numpy()).as_rotvec()
            rot_dist = np.linalg.norm(rotvec)
        except ValueError:
            rot_dist = np.pi  # fallback: max distance if R is invalid

        # Translation distance
        trans_dist = torch.norm(T1 - T2).item()

        rot_dists.append(rot_dist)
        trans_dists.append(trans_dist)

    rot_dists = np.array(rot_dists)
    rot_dists /= rot_dists.max()
    trans_dists = np.array(trans_dists)
    trans_dists /= trans_dists.max()

    combined_dist = rot_dists + trans_dists  # or apply weights
    closest_idx = np.argmin(combined_dist)

    return closest_idx, rot_dists, trans_dists


def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions"""
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)  # Shortest path
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    sin_theta[sin_theta == 0] = 1e-6
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return F.normalize(s0 * q0 + s1 * q1, dim=-1)

import torch
from pytorch3d.renderer import PerspectiveCameras

def interpolate_cameras_K(camera1, camera_target, num_frames: int = 49, config=None):
    """
    Generate a list of cameras transitioning from camera1 to camera_target, using log interpolation
    for focal length combined with smoothstep for a smoother zoom in/out effect.
    """
    # ------- Internal helper functions ------- #
    def smoothstep(t: torch.Tensor) -> torch.Tensor:
        """3t^2 - 2t^3: Common ease-in-out curve"""
        return 3 * t**2 - 2 * t**3

    def interpolate_log_focal(K1: torch.Tensor, K2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate fx, fy in log space, keeping the principal point and other elements consistent with K1.
        """
        fx1, fy1 = K1[0, 0], K1[1, 1]
        fx2, fy2 = K2[0, 0], K2[1, 1]

        fx_t = torch.exp((1 - t) * torch.log(fx1) + t * torch.log(fx2))
        fy_t = torch.exp((1 - t) * torch.log(fy1) + t * torch.log(fy2))

        K_t = K1.clone()
        K_t[0, 0] = fx_t
        K_t[1, 1] = fy_t
        return K_t
    # --------------------------- #

    device = camera1.device
    K1, K2 = camera1.K[0], camera_target.K[0]
    R_t, T_t = camera_target.R[0], camera_target.T[0]

    # Get image dimensions
    image_size = (config['orig_H'], config['orig_W'])


    t_values = torch.linspace(0, 1, steps=num_frames, device=device)
    cameras = []

    for t in t_values:
        t_smooth = smoothstep(t)
        K_t = interpolate_log_focal(K1, K2, t_smooth)

        cam_t = PerspectiveCameras(
            K=K_t.unsqueeze(0),
            R=R_t.unsqueeze(0),
            T=T_t.unsqueeze(0),
            in_ndc=False,
            device=device,
            image_size=(image_size,)   # Use configured resolution
        )
        cameras.append(cam_t)

    return cameras


def interpolate_cameras_K_1(camera1, camera_target, num_frames=49, scale_factor=36, power=0.5, config=None):
    """
    Progressive growth-based K interpolation function, from camera1 to camera_target

    Args:
        camera1: Starting camera
        camera_target: Target camera
        num_frames: Total number of frames
        scale_factor: Scale factor (default 36)
        power: Power exponent (default 0.5, corresponding to i**0.5)
        config: Configuration object containing orig_H and orig_W

    Returns:
        list: List of cameras
    """
    device = camera1.device
    cameras = [camera1]
    
    # Get image dimensions

    image_size = (config['orig_H'], config['orig_W'])


    # Get initial and target K matrices
    K_initial = camera1.K[0].clone()
    K_target = camera_target.K[0].clone()
    R_target = camera_target.R[0].clone()
    T_target = camera_target.T[0].clone()
    
    # Compute K matrix difference
    K_diff = K_target - K_initial
    
    for i in range(1, num_frames):
        # Create new K matrix
        K_new = K_initial.clone()
        
        # Apply progressive interpolation
        progress = (i ** power) / ((num_frames - 1) ** power)  # Normalize to [0,1]
        K_new += K_diff * progress
        
        # Create new camera
        cam_new = PerspectiveCameras(
            K=K_new.unsqueeze(0),
            R=R_target.unsqueeze(0),
            T=T_target.unsqueeze(0),
            in_ndc=False,
            device=device,
            image_size=(image_size,)
        )
        cameras.append(cam_new)
    
    return cameras

def interpolate_cameras_RT(camera1, camera_target, num_frames=49, config=None):
    device = camera1.device
    K1 = camera1.K[0]
    R1 = camera1.R[0]
    R2 = camera_target.R[0]
    T1 = camera1.T[0]
    T2 = camera_target.T[0]

    # Get image dimensions
    image_size = (config['orig_H'], config['orig_W'])


    def orthogonalize(R):
        U, _, V = torch.linalg.svd(R)
        R_new = U @ V
        if torch.det(R_new) < 0:
            U[:, -1] *= -1
            R_new = U @ V
        return R_new

    def slerp(q0, q1, t):
        dot = torch.sum(q0 * q1)
        if dot > 0.9995:
            q_interp = (1 - t) * q0 + t * q1
            return F.normalize(q_interp, dim=0)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        s0 = torch.sin((1 - t) * theta_0) / sin_theta_0
        s1 = torch.sin(t * theta_0) / sin_theta_0
        return F.normalize(s0 * q0 + s1 * q1, dim=0)

    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    R1 = orthogonalize(R1)
    R2 = orthogonalize(R2)
    q1 = matrix_to_quaternion(R1.unsqueeze(0))[0]
    q2 = matrix_to_quaternion(R2.unsqueeze(0))[0]

    t_values = torch.linspace(0, 1, num_frames, device=device)
    cameras = []
    for t in t_values:
        # K_t = (1 - t) * K1 + t * K2
        K_t = K1
        T_t = (1 - t) * T1 + t * T2
        q_t = slerp(q1, q2, t)
        R_t = quaternion_to_matrix(q_t.unsqueeze(0))[0]
        cam_t = PerspectiveCameras(
            K=K_t.unsqueeze(0),
            R=R_t.unsqueeze(0),
            T=T_t.unsqueeze(0),
            in_ndc=False,
            device=device,
            image_size=(image_size,)
            
        )
        cameras.append(cam_t)

    return cameras


def convert_pt3d_cam_to_3dgs_cam(pt3d_cam: PerspectiveCameras, image=None, xyz_scale=1, config=None):
    # If no image is passed in, create with default dimensions

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
    focal_length_x = pt3d_cam.K[0, 0, 0].item()
    focal_length_y = pt3d_cam.K[0, 1, 1].item()
    half_img_size_x = pt3d_cam.K[0, 0, 2].item()
    fovx = 2*np.arctan(half_img_size_x / focal_length_x)
    
    half_img_size_y = pt3d_cam.K[0, 1, 2].item()
    fovy = 2*np.arctan(half_img_size_y / focal_length_y)
    tdgs_cam = Camera(image=image, R=R, T=T, FoVx=fovx, FoVy=fovy)

    return tdgs_cam


def save_rough_video(save_path,now_imgs):
    video_tensor = now_imgs#*(1-now_masks)  # Example data, replace with your data

    # Ensure tensor shape is [49, 3, 512, 512] before saving
    video_tensor = video_tensor.clip(0,1.)


    # Convert to uint8 type, since write_video expects uint8 data type
    video_tensor = (video_tensor * 255).to(torch.uint8)

    write_video(save_path, video_tensor, fps=16)


def rotate_pytorch3d_camera(camera:PerspectiveCameras, angle_rad:float, axis='x'):
    """
    Rotate a PyTorch3D camera object around the specified axis by the given angle.
    It should keep its own location in the world frame.
    This means that the following equation should hold:
    x_world @ P_w2c^new = x_world @ P_w2c^old @ P^(-1),
    where P^(-1) denotes the inverse of the desired transform matrix.
    
    Parameters:
        camera (PyTorch3D Camera): The camera object to rotate.
        angle_rad (float): The angle in radians by which to rotate the camera.
        axis (str): The axis around which to rotate the camera. Can be 'x', 'y', or 'z'.
    
    Returns:
        PyTorch3D Camera: The rotated camera object.
    """
    if axis == 'x':
        R = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
            [0, torch.sin(angle_rad), torch.cos(angle_rad)]
        ]).float()
    elif axis == 'y':
        R = torch.tensor([
            [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
            [0, 1, 0],
            [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
        ]).float()
    elif axis == 'z':
        R = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad), torch.cos(angle_rad), 0],
            [0, 0, 1]
        ]).float()
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Construct the pytorch3d-style P matrix from R and T. P=[[R', 0], [T, 1]]
    P = torch.eye(4)
    P[:3, :3] = R.transpose(0, 1)
    Pinv = torch.inverse(P).to(camera.device)

    P_old = camera.get_world_to_view_transform().get_matrix()
    P_new = P_old @ Pinv
    T_new = P_new[:, 3, :3]
    R_new = P_new[:, :3, :3]

    new_camera = camera.clone()
    new_camera.T = T_new
    new_camera.R = R_new
    
    return new_camera


def translate_pytorch3d_camera(camera:PerspectiveCameras, translation:torch.Tensor):
    """
    Translate a PyTorch3D camera object by the given translation vector.
    It should keep its own orientation in the world frame.
    This means that the following equation should hold:
    x_world @ P_w2c^new = x_world @ P_w2c^old @ P^(-1),
    where P^(-1) denotes the inverse of the desired transform matrix.
    
    Parameters:
        camera (PyTorch3D Camera): The camera object to translate.
        translation (torch.Tensor): The translation vector to apply to the camera.
    
    Returns:
        PyTorch3D Camera: The translated camera object.
    """
    # Construct the pytorch3d-style P matrix from R and T. P=[[R', 0], [T, 1]]
    P = torch.eye(4)
    P[3, :3] = translation
    Pinv = torch.inverse(P).to(camera.device)

    P_old = camera.get_world_to_view_transform().get_matrix()
    P_new = P_old @ Pinv
    T_new = P_new[:, 3, :3]
    R_new = P_new[:, :3, :3]

    new_camera = camera.clone()
    new_camera.T = T_new
    new_camera.R = R_new
    
    return new_camera


def find_biggest_connected_inpaint_region(mask):
    H, W = mask.shape
    visited = torch.zeros((H, W), dtype=torch.bool)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # up, right, down, left
    
    def bfs(i, j):
        queue = deque([(i, j)])
        region = []
        
        while queue:
            x, y = queue.popleft()
            if 0 <= x < H and 0 <= y < W and not visited[x, y] and mask[x, y] == 1:
                visited[x, y] = True
                region.append((x, y))
                for dx, dy in directions:
                    queue.append((x + dx, y + dy))
                    
        return region
    
    max_region = []
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                current_region = bfs(i, j)
                if len(current_region) > len(max_region):
                    max_region = current_region
    
    mask_connected = torch.zeros((H, W)).to(mask.device)
    for x, y in max_region:
        mask_connected[x, y] = 1
    return mask_connected


def edge_pad(img, mask, mode=1):
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, mask


def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask


def cv2_telea(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    return ret, mask


def mean_fill(img, mask):
    avg = img.mean(axis=0).mean(axis=0)
    img[mask < 1] = avg
    return img, mask

def estimate_scale_and_shift(x, y, init_method='identity', optimize_scale=True):
    assert len(x.shape) == 1 and len(y.shape) == 1, "Inputs should be 1D tensors"
    assert x.shape[0] == y.shape[0], "Input tensors should have the same length"

    n = x.shape[0]

    if init_method == 'identity':
        shift_init = 0.
        scale_init = 1.
    elif init_method == 'median':
        shift_init = (torch.median(y) - torch.median(x)).item()
        scale_init = (torch.sum(torch.abs(y - torch.median(y))) / n / (torch.sum(torch.abs(x - torch.median(x))) / n)).item()
    else:
        raise ValueError("init_method should be either 'identity' or 'median'")
    shift = torch.tensor(shift_init).cuda().requires_grad_()
    scale = torch.tensor(scale_init).cuda().requires_grad_()

    # Set optimizer and scheduler
    optimizer = torch.optim.Adam([shift, scale], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # Optimization loop
    for step in range(1000):  # Set the range to the number of steps you find appropriate
        optimizer.zero_grad()
        if optimize_scale:
            loss = torch.abs((x.detach() + shift) * scale - y.detach()).mean()
        else:
            loss = torch.abs(x.detach() + shift - y.detach()).mean()
        loss.backward()
        if step == 0:
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
        optimizer.step()
        scheduler.step(loss)

        # Early stopping condition if needed
        if step > 20 and scheduler._last_lr[0] < 1e-6:  # You might want to adjust these conditions
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
            break

    if optimize_scale:
        return scale.item(), shift.item()
    else:
        return 1., shift.item()


def save_depth_map(depth_map, file_name, vmin=None, vmax=None, save_clean=False):
    depth_map = np.squeeze(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map after squeezing must be 2D.")

    dpi = 100  # Adjust this value if necessary
    figsize = (depth_map.shape[1] / dpi, depth_map.shape[0] / dpi)  # Width, Height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)

    if not save_clean:
        # Standard save with labels and color bar
        cbar = fig.colorbar(cax)
        ax.set_title("Depth Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
    else:
        # Clean save without labels, color bar, or axis
        plt.axis('off')
        ax.set_aspect('equal', adjustable='box')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')  # Convert to RGB
    img = img.resize((depth_map.shape[1], depth_map.shape[0]), Image.Resampling.LANCZOS)  # Resize to original dimensions
    img.save(file_name, format='png')
    buf.close()
    plt.close()
    return img



"""
Apache-2.0 license
https://github.com/hafriedlander/stable-diffusion-grpcserver/blob/main/sdgrpcserver/services/generate.py
https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2
_handleImageAdjustment
"""

functbl = {
    "gaussian": gaussian_noise,
    "edge_pad": edge_pad,
    "cv2_ns": cv2_ns,
    "cv2_telea": cv2_telea,
}

def soft_stitching(source_img, target_img, mask, blur_size=11, sigma=2.5):
    # Apply Gaussian blur to the mask to create a soft transition area
    # The size of the kernel and the standard deviation can be adjusted
    # for more or less blending

    # blur_size  # Size of the Gaussian kernel, must be odd
    # sigma       # Standard deviation of the Gaussian kernel
    
    # Ensure the mask is float for blurring
    soft_mask = mask.float()

    # Adding padding to reduce edge effects during blurring
    padding = blur_size // 2
    soft_mask = F.pad(soft_mask, (padding, padding, padding, padding), mode='reflect')
    
    # Apply the Gaussian blur
    blurred_mask = gaussian_blur(soft_mask, kernel_size=(blur_size, blur_size), sigma=(sigma, sigma))
    
    # Remove the padding
    blurred_mask = blurred_mask[:, :, padding:-padding, padding:-padding]
    
    # Ensure the mask is within 0 and 1 after blurring
    blurred_mask = torch.clamp(blurred_mask, 0, 1)
    
    # Blend the images based on the blurred mask
    stitched_img = source_img * blurred_mask + target_img * (1 - blurred_mask)
    
    return stitched_img

def prepare_scheduler(scheduler):
    # if hasattr(scheduler.config, "steps_offset"):
    #     new_config = dict(scheduler.config)
    #     new_config["steps_offset"] = 0
    #     scheduler._internal_dict = FrozenDict(new_config)
    if hasattr(scheduler, "is_scale_input_called"):
        scheduler.is_scale_input_called = True  # to surpress the warning
    return scheduler


def load_example_yaml(example_name, yaml_path):
    with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
    yaml_data = None
    for d in data:
        if d['name'] == example_name:
            yaml_data = d
            break
    return yaml_data


def merge_frames(all_rundir, fps=10, save_dir=None, is_forward=False, save_depth=False, save_gif=True):
    """
    Merge frames from multiple run directories into a single directory with continuous naming.
    
    Parameters:
        all_rundir (list of pathlib.Path): Directories containing the run data.
        save_dir (pathlib.Path): Directory where all frames should be saved.
    """

    # Ensure save_dir/frames exists
    save_frames_dir = save_dir / 'frames'
    save_frames_dir.mkdir(parents=True, exist_ok=True)

    if save_depth:
        save_depth_dir = save_dir / 'depth'
        save_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize a counter for the new filenames
    global_counter = 0
    
    # Iterate through all provided run directories
    if is_forward:
        all_rundir = all_rundir[::-1]
    for rundir in all_rundir:
        # Ensure the rundir and the frames subdir exist
        if not rundir.exists():
            print(f"Warning: {rundir} does not exist. Skipping...")
            continue
        
        frames_dir = rundir / 'images' / 'frames'
        if not frames_dir.exists():
            print(f"Warning: {frames_dir} does not exist. Skipping...")
            continue

        if save_depth:
            depth_dir = rundir / 'images' / 'depth'
            if not depth_dir.exists():
                print(f"Warning: {depth_dir} does not exist. Skipping...")
                continue
        
        # Get all .png files in the frames directory, assuming no nested dirs
        frame_files = sorted(frames_dir.glob('*.png'), key=lambda x: int(x.stem))
        if save_depth:
            depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
        
        # Copy and rename each file
        for i, frame_file in enumerate(frame_files):
            # Form the new path and copy the file
            new_frame_path = save_frames_dir / f"{global_counter}.png"
            shutil.copy(str(frame_file), str(new_frame_path))

            if save_depth:
                # Form the new path and copy the file
                new_depth_path = save_depth_dir / f"{global_counter}.png"
                shutil.copy(str(depth_files[i]), str(new_depth_path))
            
            # Increment the global counter
            global_counter += 1
    
    last_keyframe_name = 'kf1.png' if is_forward else 'kf2.png'
    last_keyframe = all_rundir[-1] / 'images' / last_keyframe_name
    new_frame_path = save_frames_dir / f"{global_counter}.png"
    shutil.copy(str(last_keyframe), str(new_frame_path))

    if save_depth:
        last_depth_name = 'kf1_depth.png' if is_forward else 'kf2_depth.png'
        last_depth = all_rundir[-1] / 'images' / last_depth_name
        new_depth_path = save_depth_dir / f"{global_counter}.png"
        shutil.copy(str(last_depth), str(new_depth_path))

    frames = []
    for frame_file in sorted(save_frames_dir.glob('*.png'), key=lambda x: int(x.stem)):
        frame_image = Image.open(frame_file)
        frame = ToTensor()(frame_image).unsqueeze(0)
        frames.append(frame)

    if save_depth:
        depth = []
        for depth_file in sorted(save_depth_dir.glob('*.png'), key=lambda x: int(x.stem)):
            depth_image = Image.open(depth_file)
            depth_frame = ToTensor()(depth_image).unsqueeze(0)
            depth.append(depth_frame)

    video = (255 * torch.cat(frames, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (255 * torch.cat(frames[::-1], dim=0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "output.mp4", fps=fps, save_gif=save_gif)
    save_video(video_reverse, save_dir / "output_reverse.mp4", fps=fps, save_gif=save_gif)

    if save_depth:
        depth_video = (255 * torch.cat(depth, dim=0)).to(torch.uint8).detach().cpu()
        depth_video_reverse = (255 * torch.cat(depth[::-1], dim=0)).to(torch.uint8).detach().cpu()

        save_video(depth_video, save_dir / "output_depth.mp4", fps=fps, save_gif=save_gif)
        save_video(depth_video_reverse, save_dir / "output_depth_reverse.mp4", fps=fps, save_gif=save_gif)


def merge_keyframes(all_keyframes, save_dir, save_folder='keyframes', fps=1):
    """
    Save a list of PIL images sequentially into a directory.

    Parameters:
        all_keyframes (list): A list of PIL Image objects.
        save_dir (Path): A pathlib Path object indicating where to save the images.
    """
    # Ensure that the save_dir exists
    save_path = save_dir / save_folder
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save each keyframe with a sequential filename
    for i, frame in enumerate(all_keyframes):
        frame.save(save_path / f'{i}.png')

    all_keyframes = [ToTensor()(frame).unsqueeze(0) for frame in all_keyframes]
    all_keyframes = torch.cat(all_keyframes, dim=0)
    video = (255 * all_keyframes).to(torch.uint8).detach().cpu()
    video_reverse = (255 * all_keyframes.flip(0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "keyframes.mp4", fps=fps)
    save_video(video_reverse, save_dir / "keyframes_reverse.mp4", fps=fps)

class SimpleLogger:
    def __init__(self, log_path):
        # Ensure log_path is a Path object, whether provided as str or Path
        if not isinstance(log_path, Path):
            log_path = Path(log_path)
        
        # Ensure the file ends with '.log'
        if not log_path.name.endswith('.txt'):
            raise ValueError("Log file must end with '.txt' extension")

        # Create the directory if it does not exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(str(log_path))
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def print(self, message, attach_time=False):
        if attach_time:
            current_time = datetime.now().strftime("[%H:%M:%S]")
            self.logger.info(current_time)
        self.logger.info(message)

    # return torch.from_numpy(q)/255.
import subprocess
import json
import atexit

class OSEDiffService:
    def __init__(self, sr_env_path):
        self.process = None
        self.sr_env_path = sr_env_path
        self.start_service()

    def start_service(self):
        """Start the OSEDiff service process."""
        if self.process is None:
            cmd = [
                self.sr_env_path,
                "test_ose_diff_on.py",
                "--osediff_path", "preset/models/osediff.pkl",
                "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
                "--ram_ft_path", "DAPE.pth",
                "--ram_path", "ram_swin_large_14m.pth",
                "--upscale", "1"
            ]
            self.process = subprocess.Popen(
                cmd,
                cwd="SR/osediff",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for model loading to complete
            while True:
                line = self.process.stdout.readline()
                if not line:
                    error = self.process.stderr.read()
                    raise Exception(f"OSEDiff service failed to start. Error: {error}")
                
                print(f"[OSEDiff] {line.strip()}")  # Print all output
                
                try:
                    response = json.loads(line)
                    if response.get("status") == "ready":
                        print("[OSEDiff] Service is ready!")
                        break
                except json.JSONDecodeError:
                    continue

    def process_image(self, image_path, output_dir):
        """Send an image processing command to the service."""
        command = {
            "image_path": image_path,
            "output_dir": output_dir
        }
        
        # Send command
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        print(f"[OSEDiff] Sent: {command}")
        
        # Wait for result - simple version, wait for one response only
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise Exception("OSEDiff service process ended unexpectedly")
            
            print(f"[OSEDiff] Response: {line.strip()}")
            
            try:
                response = json.loads(line)
                if "status" in response:
                    if response["status"] == "error":
                        raise Exception(response.get("message", "Unknown error"))
                    elif response["status"] == "success":
                        return response
            except json.JSONDecodeError:
                # Non-JSON output, continue waiting
                continue

    def stop_service(self):
        """Stop the service process."""
        if self.process:
            self.process.terminate()
            self.process = None

def generate_llff_poses_around_z(
    n_frames=30,
    radius_x=1.5e-3/1/3/4,
    radius_y=1.51e-3/1/3/4,
    z=0.0,
    look_dir=[0, 0, 1.],
    device="cuda",
    image_size=None,
    focal_length=1024,
    config=None
):
    # If image_size is not passed in, get it from config

    image_size = (config['orig_H'], config['orig_W'])
    from pytorch3d.renderer import PerspectiveCameras
    import math

    cameras = []

    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        x = radius_x * math.cos(angle)
        y = radius_y * math.sin(angle)
        position = torch.tensor([x, y, z], device=device)

        # Fixed facing direction: default facing positive z-axis (forward-facing)
        forward = torch.tensor(look_dir, device=device)
        forward = forward / torch.norm(forward)

        # Up vector in world coordinates (try to keep y pointing up)
        up_world = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.cross(forward, up_world)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        up = up / torch.norm(up)

        # Construct camera rotation matrix (right, up, -forward)
        R = torch.stack([right.abs(), up.abs(), forward.abs()], dim=1).T.unsqueeze(0)

        # Camera translation (T = -R @ C)
        T = -torch.bmm(R, position.view(1, 3, 1)).squeeze(-1)

        # Intrinsic matrix
        K = torch.eye(4)[None].to(device)
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        K[0, 0, 2] = image_size[1] / 2
        K[0, 1, 2] = image_size[0] / 2
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1

        camera = PerspectiveCameras(
            K=K, R=R, T=T, in_ndc=False, image_size=[image_size], device=device
        )
        cameras.append(camera)

    return cameras

def interpolate_cameras_RT_and_K(camera1, camera_target, num_frames=49, config=None):
    """
    Simultaneously interpolate camera RT (position/rotation) and K (intrinsic matrix) to generate a smooth transition sequence

    Args:
        camera1: Starting camera
        camera_target: Target camera
        num_frames: Total number of frames
        config: Configuration object

    Returns:
        list: Interpolated camera list
    """
    import torch.nn.functional as F
    from pytorch3d.renderer import PerspectiveCameras
    
    device = camera1.device
    
    # Get RT parameters
    K1 = camera1.K[0]
    K2 = camera_target.K[0]
    R1 = camera1.R[0]
    R2 = camera_target.R[0]
    T1 = camera1.T[0]
    T2 = camera_target.T[0]

    # Get image dimensions
    image_size = (config['orig_H'], config['orig_W'])

    def orthogonalize(R):
        U, _, V = torch.linalg.svd(R)
        R_new = U @ V
        if torch.det(R_new) < 0:
            U[:, -1] *= -1
            R_new = U @ V
        return R_new

    def slerp(q0, q1, t):
        dot = torch.sum(q0 * q1)
        if dot > 0.9995:
            q_interp = (1 - t) * q0 + t * q1
            return F.normalize(q_interp, dim=0)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        s0 = torch.sin((1 - t) * theta_0) / sin_theta_0
        s1 = torch.sin(t * theta_0) / sin_theta_0
        return F.normalize(s0 * q0 + s1 * q1, dim=0)

    def matrix_to_quaternion(R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return torch.stack([qw, qx, qy, qz])

    def quaternion_to_matrix(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return torch.stack([
            torch.stack([1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw]),
            torch.stack([2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw]),
            torch.stack([2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2])
        ])

    def smoothstep(t):
        """3t^2 - 2t^3: Common ease-in-out curve"""
        return 3 * t**2 - 2 * t**3

    def interpolate_log_focal(K1, K2, t):
        """Interpolate fx, fy in log space"""
        fx1, fy1 = K1[0, 0], K1[1, 1]
        fx2, fy2 = K2[0, 0], K2[1, 1]

        fx_t = torch.exp((1 - t) * torch.log(fx1) + t * torch.log(fx2))
        fy_t = torch.exp((1 - t) * torch.log(fy1) + t * torch.log(fy2))

        K_t = K1.clone()
        K_t[0, 0] = fx_t
        K_t[1, 1] = fy_t
        return K_t

    # Convert to quaternions for interpolation
    q1 = matrix_to_quaternion(R1)
    q2 = matrix_to_quaternion(R2)

    cameras = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        # Use smoothstep for smooth interpolation
        t_smooth = smoothstep(torch.tensor(t, device=device))
        
        # Interpolate rotation (using SLERP)
        q_t = slerp(q1, q2, t_smooth)
        R_t = orthogonalize(quaternion_to_matrix(q_t))
        
        # Interpolate translation (linear interpolation)
        T_t = (1 - t_smooth) * T1 + t_smooth * T2
        
        # Interpolate K matrix (log space interpolation of focal length)
        K_t = interpolate_log_focal(K1, K2, t_smooth)

        # Create new camera
        camera = PerspectiveCameras(
            device=device,
            R=R_t.unsqueeze(0),
            T=T_t.unsqueeze(0),
            K=K_t.unsqueeze(0),
            image_size=[image_size]
        )
        cameras.append(camera)

    return cameras

