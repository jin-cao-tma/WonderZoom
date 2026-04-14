import torch
import numpy as np
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Transform3d

def render_pointcloud(viewpoint_camera, pc, opt, bg_color: torch.Tensor, point_radius=0.005, points_per_pixel=10):
    """
    Render a real point cloud using PyTorch3D

    Args:
        viewpoint_camera: Camera parameters
        pc: GaussianModel (we only use position and color from it)
        opt: Rendering options
        bg_color: Background color
        point_radius: Radius of points (in world coordinate system)
        points_per_pixel: Maximum number of points per pixel
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get point positions and colors
    points_3d = pc.get_xyz_all  # [N, 3]

    # Get colors
    if hasattr(pc, 'get_features_all'):
        # Use features as colors
        colors = pc.color_activation(pc.get_features_all.squeeze())  # [N, 3]
    else:
        # If no color information, use white
        colors = torch.ones_like(points_3d)

    # Filter out deleted points
    if hasattr(pc, 'delete_mask_all'):
        valid_mask = ~pc.delete_mask_all
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]
    
    # Create PyTorch3D point cloud structure
    point_cloud = Pointclouds(points=[points_3d], features=[colors])
    
    # Set rasterization parameters
    raster_settings = PointsRasterizationSettings(
        image_size=(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)),
        radius=point_radius,
        points_per_pixel=points_per_pixel,
    )
    
    # Create rasterizer
    rasterizer = PointsRasterizer(
        cameras=None,  # We will set the camera manually
        raster_settings=raster_settings
    )
    
    # Create compositor
    compositor = AlphaCompositor(background_color=bg_color)
    
    # Create renderer
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=compositor
    )
    
    # Manually set camera matrices
    # Convert 3DGS camera parameters to PyTorch3D format
    R = viewpoint_camera.R.T  # PyTorch3D uses the transposed rotation matrix
    T = viewpoint_camera.T
    
    # Create transformation matrix
    transform = Transform3d(matrix=torch.eye(4, device=device).unsqueeze(0))
    transform = transform.rotate(torch.tensor(R, device=device).unsqueeze(0))
    transform = transform.translate(torch.tensor(T, device=device).unsqueeze(0))
    
    # Apply camera transformation
    point_cloud_transformed = point_cloud.transform(transform)
    
    # Render
    images = renderer(point_cloud_transformed)
    
    # Return format compatible with the original renderer
    return {
        "render": images[0],  # [H, W, 3]
        "viewspace_points": points_3d,  # For gradient computation
        "visibility_filter": torch.ones(len(points_3d), dtype=torch.bool, device=device),
        "radii": torch.ones(len(points_3d), device=device) * point_radius,
        "final_opacity": torch.ones(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device=device),
        "depth": torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device=device),
        "median_depth": torch.zeros(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device=device),
    }


def render_pointcloud_simple(viewpoint_camera, pc, opt, bg_color: torch.Tensor, point_size=2):
    """
    Simple point cloud renderer (no PyTorch3D dependency)
    Directly projects 3D points onto the 2D screen
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get point positions and colors
    points_3d = pc.get_xyz_all  # [N, 3]

    # Get colors
    if hasattr(pc, 'get_features_all'):
        colors = pc.color_activation(pc.get_features_all.squeeze())  # [N, 3]
    else:
        colors = torch.ones_like(points_3d)

    # Filter out deleted points
    if hasattr(pc, 'delete_mask_all'):
        valid_mask = ~pc.delete_mask_all
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]
    
    # Camera parameters
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    
    # Transform points from world coordinates to camera coordinates
    R = torch.tensor(viewpoint_camera.R, device=device, dtype=torch.float32)
    T = torch.tensor(viewpoint_camera.T, device=device, dtype=torch.float32)
    
    # Convert to camera coordinate system
    points_cam = points_3d @ R + T.unsqueeze(0)
    
    # Perspective projection to screen coordinates
    # Note: here we assume a simple pinhole camera model
    focal_x = viewpoint_camera.focal_x if hasattr(viewpoint_camera, 'focal_x') else width * 0.5
    focal_y = viewpoint_camera.focal_y if hasattr(viewpoint_camera, 'focal_y') else height * 0.5
    
    # Project
    x = points_cam[:, 0] / (points_cam[:, 2] + 1e-8) * focal_x + width * 0.5
    y = points_cam[:, 1] / (points_cam[:, 2] + 1e-8) * focal_y + height * 0.5
    z = points_cam[:, 2]
    
    # Filter points within screen bounds and with positive depth
    valid_depth = z > 0.1
    valid_x = (x >= 0) & (x < width)
    valid_y = (y >= 0) & (y < height)
    valid_mask = valid_depth & valid_x & valid_y
    
    # Initialize image
    image = bg_color.unsqueeze(0).unsqueeze(0).repeat(height, width, 1)
    depth_buffer = torch.full((height, width), float('inf'), device=device)
    
    if valid_mask.sum() > 0:
        valid_points = torch.stack([x[valid_mask], y[valid_mask]], dim=1).long()
        valid_colors = colors[valid_mask]
        valid_depths = z[valid_mask]
        
        # Simple depth test and rendering
        for i in range(len(valid_points)):
            px, py = valid_points[i]
            if 0 <= px < width and 0 <= py < height:
                if valid_depths[i] < depth_buffer[py, px]:
                    # Render a small square instead of a single pixel
                    for dx in range(-point_size//2, point_size//2 + 1):
                        for dy in range(-point_size//2, point_size//2 + 1):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                image[ny, nx] = valid_colors[i]
                                depth_buffer[ny, nx] = valid_depths[i]
    
    # Create dummy viewspace_points for gradient propagation
    screenspace_points = torch.zeros_like(points_3d, requires_grad=True)
    
    return {
        "render": image.permute(2, 0, 1),  # [3, H, W] format
        "viewspace_points": screenspace_points,
        "visibility_filter": valid_mask,
        "radii": torch.ones(len(points_3d), device=device),
        "final_opacity": torch.ones(height, width, device=device),
        "depth": depth_buffer,
        "median_depth": depth_buffer,
    } 