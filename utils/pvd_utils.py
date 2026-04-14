import trimesh
import torch
import numpy as np
import os
import math
# import torchvision
import scipy
from tqdm import tqdm
import cv2  # Assuming OpenCV is used for image saving
# from PIL import Image
import pytorch3d
import random
# from PIL import ImageGrab
# torchvision
# from torchvision.utils import save_image
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import imageio
import torch.nn.functional as F
# from torchvision.transforms import ToPILImage
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
# sys.path.append('./extern/dust3r')
# from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from torchvision.transforms import CenterCrop, Compose, Resize

def save_video(data,images_path,folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder]*len(data)
        images = [np.array(Image.open(os.path.join(folder_name,path))) for folder_name,path in zip(folder,data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})

def get_input_dict(img_tensor,idx,dtype = torch.float32):

    return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':img_tensor.to(dtype)}
    # return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':ToPILImage()((img_tensor.squeeze(0)+ 1) / 2)}


def rotate_theta(c2ws_input, theta, phi, r, device): 
    # theta: image tilt angle, the angle between the new y’ axis (in the yoz plane) and the y axis
    # Let the camera move on a sphere centered at [0,0,depth_avg]; first move on a sphere centered at [0,0,0] for easier rotation matrix computation, then translate
    c2ws = copy.deepcopy(c2ws_input)
    c2ws[:,2, 3] = c2ws[:,2, 3] + r  # Translate camera coordinate system along world -z direction by r
    # Compute rotation vector
    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    v = torch.tensor([0, torch.cos(theta), torch.sin(theta)])
    # Compute skew-symmetric matrix
    v_x = torch.zeros(3, 3).to(device)
    v_x[0, 1] = -v[2]
    v_x[0, 2] = v[1]
    v_x[1, 0] = v[2]
    v_x[1, 2] = -v[0]
    v_x[2, 0] = -v[1]
    v_x[2, 1] = v[0]

    # Compute square of skew-symmetric matrix
    v_x_square = torch.matmul(v_x, v_x)

    # Compute rotation matrix
    R = torch.eye(3).to(device) + torch.sin(phi) * v_x + (1 - torch.cos(phi)) * v_x_square

    # Convert to homogeneous representation
    R_h = torch.eye(4)
    R_h[:3, :3] = R
    Rot_mat = R_h.to(device)

    c2ws = torch.matmul(Rot_mat, c2ws)
    c2ws[:,2, 3]= c2ws[:,2, 3] - r # Finally subtract r, equivalent to rotating around z=|r| as center

    return c2ws

def sphere2pose(c2ws_input, theta, phi, r, device,x=None,y=None):
    c2ws = copy.deepcopy(c2ws_input)

    # First translate along world z-axis direction, then rotate
    c2ws[:,2,3] += r
    if x is not None:
        c2ws[:,1,3] += y
    if y is not None:
        c2ws[:,0,3] += x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([[1, 0, 0, 0],
                    [0, cos_value_x, -sin_value_x, 0],
                    [0, sin_value_x, cos_value_x, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1).to(device)
    
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([[cos_value_y, 0, sin_value_y, 0],
                    [0, 1, 0, 0],
                    [-sin_value_y, 0, cos_value_y, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1).to(device)
    
    c2ws = torch.matmul(rot_mat_x,c2ws)
    c2ws = torch.matmul(rot_mat_y,c2ws)

    return c2ws 

def generate_candidate_poses(c2ws_anchor,H,W,fs,c,theta, phi,num_candidates,device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    if num_candidates == 2:
        thetas = np.array([0,-theta])
        phis = np.array([phi,phi])
    elif num_candidates == 3:
        thetas = np.array([0,-theta,theta/2.]) #avoid too many downward
        phis = np.array([phi,phi,phi])
    else:
        raise ValueError("NBV mode only supports 2 or 3 candidates per iteration.")
    
    c2ws_list = []

    for th, ph in zip(thetas,phis):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), r=None, device= device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## Convert dust3r coordinate system to pytorch3d coordinate system
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,thetas,phis

def interpolate_poses_spline(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    def viewmatrix(lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    new_poses = points_to_poses(new_points) 
    poses_tensor = torch.from_numpy(new_poses)
    extra_row = torch.tensor(np.repeat([[0, 0, 0, 1]], n_interp, axis=0), dtype=torch.float32).unsqueeze(1)
    poses_final = torch.cat([poses_tensor, extra_row], dim=1)

    return poses_final

def interp_traj(c2ws: torch.Tensor, n_inserts: int = 49, device='cuda') -> torch.Tensor:
    
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        interpolated_path = interpolate_poses_spline(torch.stack([start_pose, end_pose])[:, :3, :].cpu().numpy(), n_inserts).to(device)
        interpolated_path = interpolated_path[:-1]
        interpolated_poses.append(interpolated_path)

    interpolated_poses.append(c2ws[-1:])
    full_path = torch.cat(interpolated_poses, dim=0)

    return full_path

def generate_traj(c2ws,H,W,fs,c,device):

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    
    return cameras, c2ws.shape[0]

def generate_traj_interp(c2ws,H,W,fs,c,ns,device):

    c2ws = interp_traj(c2ws,n_inserts = ns,device=device)
    num_views = c2ws.shape[0] 
    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)

    fs = interpolate_sequence(fs,ns-2,device=device)
    c = interpolate_sequence(c,ns-2,device=device)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    
    return cameras, num_views

def generate_traj_specified(c2ws_anchor,H,W,fs,c,theta, phi,d_r,d_x,d_y,frame,device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    thetas = np.linspace(0,theta,frame)
    phis = np.linspace(0,phi,frame)
    rs = np.linspace(0,d_r*c2ws_anchor[0,2,3].cpu(),frame)
    xs = np.linspace(0,d_x.cpu(),frame)
    ys = np.linspace(0,d_y.cpu(),frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas,phis,rs, xs, ys):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device, np.float32(x),np.float32(y))
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## Convert dust3r coordinate system to pytorch3d coordinate system
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,num_views

def generate_traj_txt(c2ws_anchor,H,W,fs,c,phi, theta, r,frame,device,viz_traj=False, save_dir = None):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi)>3:
        phis = txt_interpolation(phi,frame,mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi,frame,mode='linear')

    if len(theta)>3:
        thetas = txt_interpolation(theta,frame,mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta,frame,mode='linear')
    
    if len(r) >3:
        rs = txt_interpolation(r,frame,mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]        
    else:
        rs = txt_interpolation(r,frame,mode='linear')
    rs = rs*c2ws_anchor[0,2,3].cpu().numpy()
    print(c2ws_anchor.shape,c2ws_anchor[0,2,3])

    c2ws_list = []
    for th, ph, r in zip(thetas,phis,rs):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)

    if viz_traj:
        poses = c2ws.cpu().numpy()
        # visualizer(poses, os.path.join(save_dir,'viz_traj.png'))
        frames = [visualizer_frame(poses, i) for i in range(len(poses))]
        save_video(np.array(frames)/255.,os.path.join(save_dir,'viz_traj.mp4'))

    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## Convert dust3r coordinate system to pytorch3d coordinate system
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,num_views

def setup_renderer(cameras, image_size):
    # Define the settings for rasterization and shading.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius = 0.01,
        points_per_pixel = 10,
        bin_size = 0
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}

    return render_setup

def interpolate_sequence(sequence, k,device):

    N, M = sequence.size()
    weights = torch.linspace(0, 1, k+1).view(1, -1, 1).to(device)
    left_values = sequence[:-1].unsqueeze(1).repeat(1, k+1, 1)
    right_values = sequence[1:].unsqueeze(1).repeat(1, k+1, 1)
    new_sequence = torch.einsum("ijk,ijl->ijl", (1 - weights), left_values) + torch.einsum("ijk,ijl->ijl", weights, right_values)
    new_sequence = new_sequence.reshape(-1, M)
    new_sequence = torch.cat([new_sequence, sequence[-1].view(1, -1)], dim=0)
    return new_sequence

def focus_point_fn(c2ws: torch.Tensor) -> torch.Tensor:
    """Calculate nearest point to all focal axes in camera-to-world matrices."""
    # Extract camera directions and origins from c2ws
    directions, origins = c2ws[:, :3, 2:3], c2ws[:, :3, 3:4]
    m = torch.eye(3).to(c2ws.device) - directions * torch.transpose(directions, 1, 2)
    mt_m = torch.transpose(m, 1, 2) @ m
    focus_pt = torch.inverse(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_camera_path(c2ws: torch.Tensor, n_inserts: int = 15, device='cuda') -> torch.Tensor:
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        focus_point = focus_point_fn(torch.stack([start_pose,end_pose]))
        interpolated_path = interpolate_poses(start_pose, end_pose, focus_point, n_inserts, device)
        
        # Exclude the last pose (end_pose) for all pairs
        interpolated_path = interpolated_path[:-1]

        interpolated_poses.append(interpolated_path)
    # Concatenate all the interpolated paths
    interpolated_poses.append(c2ws[-1:])
    full_path = torch.cat(interpolated_poses, dim=0)
    return full_path

def interpolate_poses(start_pose: torch.Tensor, end_pose: torch.Tensor, focus_point: torch.Tensor, n_inserts: int = 15, device='cuda') -> torch.Tensor:
    dtype = start_pose.dtype
    start_distance = torch.sqrt((start_pose[0, 3] - focus_point[0])**2 + (start_pose[1, 3] - focus_point[1])**2 + (start_pose[2, 3] - focus_point[2])**2)
    end_distance = torch.sqrt((end_pose[0, 3] - focus_point[0])**2 + (end_pose[1, 3] - focus_point[1])**2 + (end_pose[2, 3] - focus_point[2])**2)
    start_rot = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    end_rot = R.from_matrix(end_pose[:3, :3].cpu().numpy())
    slerp_obj = Slerp([0, 1], R.from_quat([start_rot.as_quat(), end_rot.as_quat()]))

    inserted_c2ws = []

    for t in torch.linspace(0., 1., n_inserts + 2, dtype=dtype):  # Exclude the first and last point
        interpolated_rot = slerp_obj(t).as_matrix()
        interpolated_translation = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
        interpolated_distance = (1 - t) * start_distance + t * end_distance
        direction = (interpolated_translation - focus_point) / torch.norm(interpolated_translation - focus_point)
        interpolated_translation = focus_point + direction * interpolated_distance

        inserted_pose = torch.eye(4, dtype=dtype).to(device)
        inserted_pose[:3, :3] = torch.from_numpy(interpolated_rot).to(device)
        inserted_pose[:3, 3] = interpolated_translation
        inserted_c2ws.append(inserted_pose)

    path = torch.stack(inserted_c2ws)
    return path



def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def save_pointcloud_with_normals(imgs, pts3d, msk, save_path, mask_pc, reduce_pc):
    pc = get_pc(imgs, pts3d, msk,mask_pc,reduce_pc)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))


def get_pc(imgs, pts3d, mask, mask_pc=False, reduce_pc=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    if mask_pc:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    else:
        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])

    if reduce_pc:
        pts = pts.reshape(-1, 3)[::3]
        col = col.reshape(-1, 3)[::3]
    else:
        pts = pts.reshape(-1, 3)
        col = col.reshape(-1, 3)
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    # debug
    # pct.export('output.ply')
    # print('exporting output.ply')
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts

def world_to_kth(poses, k):
    # Transform world coordinate system to align with the k-th pose's camera coordinate system
    kth_pose = poses[k]
    inv_kth_pose = torch.inverse(kth_pose)
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    return new_poses

def world_point_to_kth(poses, points, k, device):
    # Transform world coordinate system to align with the k-th pose's camera coordinate system, also process point cloud
    kth_pose = poses[k]
    inv_kth_pose = torch.inverse(kth_pose)
    # Left-multiply all poses by kth_w2c to transform them into kth_pose's camera coordinate
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W*H, 1).to(device)], dim=-1)  
    new_points = inv_kth_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1)@ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[...,:3].view(N, W, H, _)

    return new_poses, new_points


def world_point_to_obj(poses, points, k, r, elevation, device):
    ## Purpose: transform world coordinate system to the center of the object

    ## First transform world coordinate system to the specified camera
    poses, points = world_point_to_kth(poses, points, k, device)
    
    ## Define target coordinate system pose, origin at object center (world coordinate [0,0,r]), Y-axis up, Z-axis pointing out of screen, X-axis right
    elevation_rad = torch.deg2rad(torch.tensor(180-elevation)).to(device)
    sin_value_x = torch.sin(elevation_rad)
    cos_value_x = torch.cos(elevation_rad)
    R = torch.tensor([[1, 0, 0,],
                    [0, cos_value_x, sin_value_x],
                    [0, -sin_value_x, cos_value_x]]).to(device)
    
    t = torch.tensor([0, 0, r]).to(device)
    pose_obj = torch.eye(4).to(device)
    pose_obj[:3, :3] = R
    pose_obj[:3, 3] = t

    ## Multiply all points and poses by the inverse of target coordinate system (w2c) to transform them into the target coordinate system
    inv_obj_pose = torch.inverse(pose_obj)
    new_poses = torch.bmm(inv_obj_pose.unsqueeze(0).expand_as(poses), poses)
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W*H, 1).to(device)], dim=-1)  
    new_points = inv_obj_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1)@ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[...,:3].view(N, W, H, _)
    
    return new_poses, new_points

def txt_interpolation(input_list,n,mode = 'smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew

def visualizer(camera_poses, save_path="out.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["blue" for _ in camera_poses]
    for pose, color in zip(camera_poses, colors):

        camera_positions = pose[:3, 3]
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera trajectory")
    # ax.view_init(90+30, -90)
    plt.savefig(save_path)
    plt.close()

def visualizer_frame(camera_poses, highlight_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Get the max and min values of camera_positions[2]
    z_values = [pose[:3, 3][2] for pose in camera_poses]
    z_min, z_max = min(z_values), max(z_values)

    # Create a colormap object
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#00008B", "#ADD8E6"])
    # cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, pose in enumerate(camera_poses):
        camera_positions = pose[:3, 3]
        color = "blue" if i == highlight_index else "blue"
        size = 100 if i == highlight_index else 25
        color = sm.to_rgba(camera_positions[2])  # Map color based on the value of camera_positions[2]
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
            s=size,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Camera trajectory")
    ax.view_init(90+30, -90)

    plt.ylim(-0.1,0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    # new_width = int(width * 0.6)
    # start_x = (width - new_width) // 2 + new_width // 5
    # end_x = start_x + new_width
    # img = img[:, start_x:end_x, :]
    
    
    plt.close()

    return img


def center_crop_image(input_image):

    height = 576
    width = 1024
    _,_,h,w = input_image.shape
    h_ratio = h / height
    w_ratio = w / width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < height:
            h = height
        input_image = Resize((h, width))(input_image)
        
    else:
        w = int(w / h_ratio)
        if w < width:
            w = width
        input_image = Resize((height, w))(input_image)

    transformer = Compose([
        # Resize(width),
        CenterCrop((height, width)),
    ])

    input_image = transformer(input_image)
    return input_image


def rotation_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular distance between two rotation matrices R1 and R2.
    The distance is measured as the angle between the two rotations.
    """
    # Compute the trace of the rotation matrix product
    trace = torch.trace(R1 @ R2.transpose(0, 1))  # Shape: scalar
    # Ensure the value is within valid range for arccos due to numerical errors
    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    return torch.acos(cos_theta)  # Returns the angular distance in radians

def sort_poses_by_proximity(poses: torch.Tensor, rotation_weight: float = 0.5, translation_weight: float = 0.5):
    """
    Sorts a set of poses to minimize the distance between consecutive poses considering both rotation and translation.

    Args:
        poses (torch.Tensor): A tensor of poses with shape (n, 3, 4), where each pose is a 3x4 matrix
                               with the top-left 3x3 part being the rotation matrix and the last column
                               being the translation vector.
        rotation_weight (float): Weight for the rotation distance in the combined metric.
        translation_weight (float): Weight for the translation distance in the combined metric.

    Returns:
        torch.Tensor: Indices of poses in the sorted order.
    """
    # Extract positions (last column of poses) and rotation matrices (top-left 3x3 of poses)
    rotations = poses[:, :3, :3]  # Shape: (n, 3, 3)
    translations = poses[:, :3, 3]  # Shape: (n, 3)

    # Compute pairwise rotation distances and translation distances
    n = poses.shape[0]
    rotation_matrix = torch.zeros((n, n))  # Shape: (n, n)
    translation_matrix = torch.zeros((n, n))  # Shape: (n, n)

    for i in range(n):
        for j in range(n):
            # Compute rotation distance between poses i and j
            rotation_matrix[i, j] = rotation_distance(rotations[i], rotations[j])
            # Compute translation distance (Euclidean distance between positions)
            translation_matrix[i, j] = torch.norm(translations[i] - translations[j])

    # Normalize rotation and translation distances to [0, 1]
    max_rotation = rotation_matrix.max()
    max_translation = translation_matrix.max()

    rotation_matrix = rotation_matrix / max_rotation if max_rotation > 0 else rotation_matrix
    translation_matrix = translation_matrix / max_translation if max_translation > 0 else translation_matrix

    # Combine the rotation and translation distances using weighted sum
    distance_matrix = rotation_weight * rotation_matrix + translation_weight * translation_matrix
    d_m_copy=distance_matrix.clone()

    # Initialize variables
    visited = torch.zeros(n, dtype=torch.bool)
    sorted_indices = []

    # Start from the first pose (or any arbitrary pose)
    sorted_indices.append(0)
    visited[0] = True

    # Greedily find the nearest neighbor
    for _ in range(n - 1):
        # Mask visited nodes in the distance matrix
        current_index_front = sorted_indices[0]
        current_index_tail = sorted_indices[-1]
        distance_matrix[current_index_front, visited] = float('inf')
        distance_matrix[current_index_tail, visited] = float('inf')

        # Find the nearest unvisited pose at the front or tail
        dis_front, dis_tail = distance_matrix[current_index_front].min(), distance_matrix[current_index_tail].min()
        
        if dis_front < dis_tail:
            next_index = torch.argmin(distance_matrix[current_index_front]).item()
            sorted_indices.insert(0, next_index)
            visited[next_index] = True
        else:
            next_index = torch.argmin(distance_matrix[current_index_tail]).item()
            sorted_indices.append(next_index)
            visited[next_index] = True
    
    sorted_distance = []
    for i in range(1, len(sorted_indices)):
        i1, i2 = sorted_indices[i - 1], sorted_indices[i]
        sorted_distance.append(d_m_copy[i1, i2])
    
    return torch.tensor(sorted_indices), torch.tensor(sorted_distance)

def allocate_weights_to_integers(weights: torch.Tensor, total: int):
    """
    Allocate integers based on weights, ensuring the sum equals total.

    Args:
        weights (torch.Tensor): A 1D tensor of floating-point weights.
        total (int): The total sum of the allocated integers.

    Returns:
        torch.Tensor: A 1D tensor of integers whose sum equals total.
    """
    # Normalize weights to sum to 1
    normalized_weights = weights / weights.sum()

    # Scale to the target total and take the floor for initial allocation
    scaled_values = normalized_weights * total
    integer_parts = torch.floor(scaled_values).to(torch.int32)

    # Compute the remainder to distribute
    remainder = total - integer_parts.sum()

    # Compute the fractional parts for distributing the remainder
    fractional_parts = scaled_values - integer_parts

    # Sort indices by the fractional parts in descending order
    sorted_indices = torch.argsort(fractional_parts, descending=True)

    # Distribute the remainder
    for i in range(remainder):
        integer_parts[sorted_indices[i]] += 1

    return integer_parts

def get_order_colmap(img_txt_path):
    from scipy.spatial.transform import Rotation as R
    from os.path import dirname as dn
    camera_poses = []

    # Read the images.txt file
    root=dn(dn(dn(img_txt_path)))
    with open(img_txt_path, 'r') as f:
        lines = f.readlines()

    name=[]
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#") or len(line.split()) < 10:
            continue  

        if i % 2 == 0: 
            parts = line.split()
            qw, qx, qy, qz = map(float, parts[1:5])  
            tx, ty, tz = map(float, parts[5:8])      

            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()  

            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz]

            camera_poses.append(transform_matrix)
            name.append(os.path.join(root,"images",parts[-1]))

    camera_poses = np.array(camera_poses)
    camera_poses[:,:3,3:]
    tten=sort_poses_by_proximity(torch.tensor(camera_poses))
    name_ordered=[]
    for item in tten:
        name_ordered.append(name[item.item()])
    return name_ordered 

def sort_poses_by_proximity(poses: torch.Tensor, rotation_weight: float = 0.5, translation_weight: float = 0.5):
    """
    Sorts a set of poses to minimize the distance between consecutive poses considering both rotation and translation.

    Args:
    Returns:
        torch.Tensor: Indices of poses in the sorted order.
    """
    # Extract positions (last column of poses) and rotation matrices (top-left 3x3 of poses)
    rotations = poses[:, :3, :3]  # Shape: (n, 3, 3)
    translations = poses[:, :3, 3]  # Shape: (n, 3)

    # Compute pairwise rotation distances and translation distances
    n = poses.shape[0]
    rotation_matrix = torch.zeros((n, n))  # Shape: (n, n)
    translation_matrix = torch.zeros((n, n))  # Shape: (n, n)

    for i in range(n):
        for j in range(n):
            # Compute rotation distance between poses i and j
            rotation_matrix[i, j] = rotation_distance(rotations[i], rotations[j])
            # Compute translation distance (Euclidean distance between positions)
            translation_matrix[i, j] = torch.norm(translations[i] - translations[j])

    # Normalize rotation and translation distances to [0, 1]
    max_rotation = rotation_matrix.max()
    max_translation = translation_matrix.max()

    rotation_matrix = rotation_matrix / max_rotation if max_rotation > 0 else rotation_matrix
    translation_matrix = translation_matrix / max_translation if max_translation > 0 else translation_matrix

    # Combine the rotation and translation distances using weighted sum
    distance_matrix = rotation_weight * rotation_matrix + translation_weight * translation_matrix


def generate_traj_multi(c2ws,H,W,fs,c,ns_total,device):
    
    def generate_traj_interp_2(c2ws,H,W,fs,c,ns,device):
        c2ws = interp_traj(c2ws, n_inserts = ns,device=device)
        # num_views = c2ws.shape[0] 
        R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
        R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
        new_c2w = torch.cat([R, T], 2)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
        R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
        # image_size = ((H, W),)  # (h, w)

        fs = interpolate_sequence(fs,ns-2,device=device)
        c = interpolate_sequence(c,ns-2,device=device)
        # return 
        # cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
        
        return R_new,T_new,fs,c
    
    image_size = ((H, W),)
    ns_total+=c2ws.shape[0]-1-(c2ws.shape[0]-1)*2
    d=torch.norm(c2ws[1:,:3,3]-c2ws[:-1,:3,3],dim=-1,)
    frames=allocate_weights_to_integers(d,ns_total)+2
    R_new,T_new,fs_new,c_new=[],[],[],[]
    for idx,frame in enumerate(frames):
        rr,tt,ff,cc = generate_traj_interp_2(c2ws[[idx,idx+1]],H,W,fs[[idx,idx+1]],
                                       c[[idx,idx+1]],frame.item(),device)
        R_new.append(rr[:-1]),T_new.append(tt[:-1])
        fs_new.append(ff[:-1]),c_new.append(cc[:-1])        
        
    R_new, T_new, fs_new, c_new = torch.concat(R_new,dim=0), torch.concat(T_new, dim=0), \
        torch.concat(fs_new,dim=0), torch.concat(c_new, dim=0)
    
    cameras = PerspectiveCameras(focal_length=fs_new, principal_point=c_new, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device) 
    
    return cameras, R_new.shape[0],d
    
    