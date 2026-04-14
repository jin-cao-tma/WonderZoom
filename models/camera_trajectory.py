import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
import math

class CameraTrajectory:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.init_focal_length = config.get('init_focal_length', 1024)
        self.image_size = config.get('image_size', (480, 720))
        
    def get_init_camera(self):
        """Get initial camera at origin"""
        height, width = self.image_size
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.init_focal_length
        K[0, 1, 1] = self.init_focal_length
        K[0, 0, 2] = width // 2
        K[0, 1, 2] = height // 2
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1

        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)

        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=(self.image_size,), device=self.device)
        return camera
        
    def generate_circular_trajectory(self, n_frames=30, radius_x=1.5e-3, radius_y=1.51e-3, z=0.0, look_dir=[0, 0, 1.]):
        """Generate circular camera trajectory"""
        cameras = []
        for i in range(n_frames):
            angle = 2 * math.pi * i / n_frames
            x = radius_x * math.cos(angle)
            y = radius_y * math.sin(angle)
            position = torch.tensor([x, y, z], device=self.device)

            forward = torch.tensor(look_dir, device=self.device)
            forward = forward / torch.norm(forward)

            up_world = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            right = torch.cross(forward, up_world)
            right = right / torch.norm(right)
            up = torch.cross(right, forward)
            up = up / torch.norm(up)

            R = torch.stack([right.abs(), up.abs(), forward.abs()], dim=1).T.unsqueeze(0)
            T = -torch.bmm(R, position.view(1, 3, 1)).squeeze(-1)

            K = torch.eye(4)[None].to(self.device)
            K[0, 0, 0] = self.init_focal_length
            K[0, 1, 1] = self.init_focal_length
            K[0, 0, 2] = self.image_size[1] / 2
            K[0, 1, 2] = self.image_size[0] / 2
            K[0, 2, 3] = 1
            K[0, 3, 2] = 1

            camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=[self.image_size], device=self.device)
            cameras.append(camera)

        return cameras
        
    def interpolate_cameras(self, start_camera, end_camera, num_frames=49):
        """Interpolate between two camera poses"""
        cameras = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            
            # Interpolate translation
            T = (1 - t) * start_camera.T + t * end_camera.T
            
            # Interpolate rotation using SLERP
            R_start = start_camera.R
            R_end = end_camera.R
            R = self._slerp(R_start, R_end, t)
            
            # Use start camera's intrinsics
            K = start_camera.K
            
            camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=[self.image_size], device=self.device)
            cameras.append(camera)
            
        return cameras
        
    def _slerp(self, R1, R2, t):
        """Spherical linear interpolation between two rotation matrices"""
        # Convert to quaternions
        q1 = self._matrix_to_quaternion(R1)
        q2 = self._matrix_to_quaternion(R2)
        
        # SLERP
        dot = (q1 * q2).sum(dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        q = w1.unsqueeze(-1) * q1 + w2.unsqueeze(-1) * q2
        
        # Convert back to rotation matrix
        return self._quaternion_to_matrix(q)
        
    def _matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Implementation of matrix to quaternion conversion
        pass
        
    def _quaternion_to_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        # Implementation of quaternion to matrix conversion
        pass 