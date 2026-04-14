"""
WonderZoom Render-Only Server

Minimal entry point for loading a pre-trained .pth GaussianModel
and serving real-time 3DGS rendering via a SocketIO frontend.

Usage:
    python run_render_only.py \
        --pth_path /path/to/model.pth \
        --base-config ./config/base-config.yaml \
        --example_config ./config/more_examples/street.yaml \
        --port 7747
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np
import cv2
import time
import copy
import math
import threading
import warnings
from argparse import ArgumentParser
from omegaconf import OmegaConf
from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS
from pytorch3d.renderer import PerspectiveCameras

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments_in import GSParams
from util.utils import convert_pt3d_cam_to_3dgs_cam, interpolate_cameras_RT

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global label system (needed by GaussianModel)
# ---------------------------------------------------------------------------
GLOBAL_LABEL_NAMES = ["main"]
GLOBAL_LABEL_MAP = {"main": 0}

# ---------------------------------------------------------------------------
# Flask + SocketIO setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    compression=True,
    max_decode_packets=50,
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading',
)

# ---------------------------------------------------------------------------
# Global rendering state
# ---------------------------------------------------------------------------
xyz_scale = 1000
background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')

client_id = None
gaussians = None
opt = GSParams()
config = None
iter_number = None

# Camera state from frontend
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
fx_wonder = None
fy_wonder = None

# Render thread control
latest_frame = None
render_stop = False

# Image transmission settings
IMAGE_COMPRESSION_QUALITY = 80   # JPEG quality (1-100), lower = faster streaming but blurrier
MAX_IMAGE_SIZE = 1080            # max image dimension for transmission, lower = faster streaming
ENABLE_RESOLUTION_SCALING = True

# Orbit state (for Space-key NVS rotation)
orbit_state = {
    'is_orbiting': False,
    'cameras': None,
    'current_frame': 0,
    'mode': 'normal',
    'collected_frames': [],
    'collected_masks': [],
    'input_camera': None,
    'hq_ready': False,
}


# ===================================================================
# Camera helper (replaces the heavy VideoGaussianProcessor dependency)
# ===================================================================

def get_camera_at_origin(cfg):
    """Create a default PyTorch3D PerspectiveCameras at the origin."""
    device = torch.device("cuda")
    height, width = cfg["orig_H"], cfg["orig_W"]
    K = torch.zeros((1, 4, 4), device=device)
    K[0, 0, 0] = cfg["init_focal_length"]
    K[0, 1, 1] = cfg["init_focal_length"]
    K[0, 0, 2] = width // 2
    K[0, 1, 2] = height // 2
    K[0, 2, 3] = 1
    K[0, 3, 2] = 1
    R = torch.eye(3, device=device).unsqueeze(0)
    T = torch.zeros((1, 3), device=device)
    camera = PerspectiveCameras(
        K=K, R=R, T=T, in_ndc=False,
        image_size=((height, width),), device=device,
    )
    return camera


def get_camera_by_js_view_matrix(view_matrix, cfg, fx=None, fy=None):
    """Convert a 16-element frontend viewMatrix to a PyTorch3D camera."""
    device = torch.device("cuda")
    vm = torch.tensor(view_matrix, device=device, dtype=torch.float).reshape(4, 4)
    xy_negate = torch.tensor(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        device=device, dtype=torch.float,
    )
    vm_neg = vm @ xy_negate
    R = vm_neg[:3, :3].unsqueeze(0)
    T = vm_neg[3, :3].unsqueeze(0) / xyz_scale

    camera = get_camera_at_origin(cfg)
    camera.R = R
    camera.T = T
    if fx is not None:
        camera.K[0, 0, 0] = fx
    if fy is not None:
        camera.K[0, 1, 1] = fy
    return camera


# ===================================================================
# Load / Save helpers
# ===================================================================

def load_gaussian_with_global_labels(path, cfg):
    """Load a GaussianModel from a .pth file and restore global label mappings."""
    global GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP

    print(f"Loading GaussianModel from: {path}")
    state = torch.load(path, map_location="cuda")

    gm = GaussianModel(
        sh_degree=state.get("max_sh_degree", 3),
        floater_dist2_threshold=state.get("floater_dist2_threshold", 0.0002),
        config=cfg,
    )

    for key, val in state.items():
        if key in ("max_sh_degree", "floater_dist2_threshold",
                    "global_label_names", "global_label_map"):
            continue
        if hasattr(gm, key):
            existing = getattr(gm, key)
            if isinstance(existing, torch.nn.Parameter):
                setattr(gm, key, torch.nn.Parameter(val.cuda(), requires_grad=True))
            else:
                setattr(gm, key, val.cuda())
        else:
            setattr(gm, key, val.cuda())

    loaded_names = state.get("global_label_names", None)
    loaded_map = state.get("global_label_map", None)
    if loaded_names is not None and loaded_map is not None:
        GLOBAL_LABEL_NAMES.clear()
        GLOBAL_LABEL_NAMES.extend(loaded_names)
        GLOBAL_LABEL_MAP.clear()
        GLOBAL_LABEL_MAP.update(loaded_map)
        print(f"  Labels restored: {GLOBAL_LABEL_NAMES}")
    else:
        print("  No global labels in state, using defaults")

    n_points = gm.get_xyz_all.shape[0]
    print(f"  Loaded {n_points} points")
    return gm


def save_gaussian_with_global_labels(gm, path):
    """Save a GaussianModel to a .pth file including global label mappings."""
    gm.merge_all_to_trainable()

    keys = [
        "_xyz", "_features_dc", "_scaling", "_rotation", "_opacity",
        "_focal_length", "next_scale", "prior_scale", "now_scale",
        "max_radii2D", "xyz_gradient_accum", "denom", "filter_3D",
        "visibility_filter_all", "is_sky_filter", "delete_mask_all", "point_labels",
    ]
    if hasattr(gm, "point_labels_prev"):
        keys.append("point_labels_prev")

    state = {k: getattr(gm, k).detach().cpu()
             for k in keys if hasattr(gm, k)}

    state["max_sh_degree"] = gm.max_sh_degree
    state["floater_dist2_threshold"] = gm.floater_dist2_threshold
    state["global_label_names"] = list(GLOBAL_LABEL_NAMES)
    state["global_label_map"] = dict(GLOBAL_LABEL_MAP)

    torch.save(state, path)
    print(f"Saved GaussianModel to: {path}")


# ===================================================================
# Orbit camera generation (for Space-key NVS rotation)
# ===================================================================

def generate_orbit_cameras(center_camera, gaussians, opt, xyz_scale,
                           n_frames=49, radius_factor=8e-4,
                           height_variation=0e-4, n_spirals=1,
                           transition_frames=9):
    """Generate cameras in a spiral trajectory around the look-at point."""
    cameras = []
    device = center_camera.device

    orbit_frames = n_frames - transition_frames

    # Get current camera parameters
    transform_matrix_pt3d = center_camera.get_world_to_view_transform().get_matrix()[0].transpose(0, 1).inverse()
    center_R = transform_matrix_pt3d[:3, :3]
    center_T = transform_matrix_pt3d[:3, 3]
    center_K = center_camera.K

    # Compute radius proportional to focal length
    focal_length = center_K[0, 0, 0].item()
    reference_focal = 10240.0
    radius = radius_factor * min(focal_length / reference_focal, 1024.0 / 10240.) * 6
    radius *= 0.5 if center_camera.K[0, 0, 0] < 30538 else 1

    factor = (0.3 if center_camera.K[0, 0, 0] > 30538 else 1.0)
    radius *= 1

    # Step 1: Get scene depth via depth rendering
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(center_camera, xyz_scale=xyz_scale, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt,
                        torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda'),
                        render_visible=True, config=config)
    depth_map = render_pkg["median_depth"][0].squeeze().detach().cpu() / xyz_scale

    H, W = depth_map.shape
    center_depth = depth_map[H // 2, W // 2].item()

    if center_depth <= 0 or center_depth > 10:
        center_depth = 1e-2
    else:
        center_depth *= 1.00

    # Step 2: Compute look-at point
    camera_center = center_camera.get_camera_center().squeeze(0)
    R_transposed = center_R
    forward_world = R_transposed[:, 2]

    look_at_point = camera_center + center_depth * forward_world
    print(f"look_at_point: {look_at_point * 1000}")

    # Step 3: Compute coordinate system
    forward_unit = forward_world / torch.norm(forward_world)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.cross(forward_unit, world_up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward_unit)
    up = up / torch.norm(up)

    # Step 4: Generate orbit camera sequence
    orbit_cameras = []
    for i in range(1, orbit_frames + 1):
        t = i / orbit_frames
        angle = 2 * math.pi * n_spirals * t

        current_radius = radius
        offset_x = current_radius * math.cos(angle)
        offset_y = current_radius * math.sin(angle)
        offset_z = center_depth * 0.1 * t * 0

        height_offset = height_variation * math.sin(2 * math.pi * t * 1.5)

        new_camera_pos = (camera_center +
                          offset_x * right +
                          (offset_y + height_offset) * up +
                          offset_z * forward_unit + forward_unit * center_depth * (1 - factor))

        new_forward = look_at_point - new_camera_pos
        new_forward = new_forward / torch.norm(new_forward)

        new_right = torch.cross(new_forward, world_up)
        if torch.norm(new_right) > 1e-6:
            new_right = new_right / torch.norm(new_right)

        new_up = torch.cross(new_right, new_forward)

        R_cam_to_world = torch.stack([-new_right, new_up, new_forward], dim=1)

        T_new = new_camera_pos.unsqueeze(0)
        new_w2c = torch.zeros((4, 4), device=device)
        new_w2c[:3, :3] = R_cam_to_world
        new_w2c[:3, 3] = T_new
        new_w2c[3, 3] = 1
        new_w2c = new_w2c.inverse()

        camera = copy.deepcopy(center_camera)
        camera.R = new_w2c[:3, :3].transpose(0, 1).unsqueeze(0)
        camera.T = new_w2c[:3, 3].unsqueeze(0)
        camera.K = copy.deepcopy(center_K)
        camera.K[0, 0, 0] *= factor
        camera.K[0, 1, 1] *= factor
        camera.image_size = center_camera.image_size
        camera.device = device
        orbit_cameras.append(camera)

    # Step 5: Generate transition from current_camera to first orbit frame
    if transition_frames > 0 and len(orbit_cameras) > 0:
        first_orbit_camera = orbit_cameras[0]
        transition_cameras = interpolate_cameras_RT(
            center_camera, first_orbit_camera,
            num_frames=transition_frames + 1, config=config)
        transition_cameras = transition_cameras[:-1]
        cameras = transition_cameras + orbit_cameras
    else:
        cameras = orbit_cameras

    return cameras


# ===================================================================
# Render loop (runs in a background thread)
# ===================================================================

def render_current_scene():
    """Continuously render the current camera view and send JPEG frames to the client.
    Matches the render loop in run.py exactly (normal + orbit paths)."""
    global latest_frame, client_id, iter_number, gaussians, opt, background
    global view_matrix_wonder, orbit_state, render_stop

    while not render_stop:
        time.sleep(0.05)
        if gaussians is None:
            continue
        try:
            with torch.no_grad():
                if orbit_state['is_orbiting']:
                    # --- Orbit rendering path (Space key) ---
                    if orbit_state['cameras'] is None:
                        current_camera = get_camera_by_js_view_matrix(
                            view_matrix_wonder, config,
                            fx=fx_wonder, fy=fy_wonder,
                        )
                        if orbit_state['mode'] == 'high_quality':
                            orbit_state['cameras'] = generate_orbit_cameras(
                                current_camera, gaussians, opt, xyz_scale,
                                n_frames=121, transition_frames=21)
                        elif orbit_state['mode'] == 'crack_fix':
                            orbit_state['cameras'] = generate_orbit_cameras(
                                current_camera, gaussians, opt, xyz_scale,
                                n_frames=49, transition_frames=10)
                        else:
                            orbit_state['cameras'] = generate_orbit_cameras(
                                current_camera, gaussians, opt, xyz_scale,
                                n_frames=22, transition_frames=0)
                        orbit_state['current_frame'] = 0

                    current_camera = orbit_state['cameras'][orbit_state['current_frame']]

                    # 3DGS rendering
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
                    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)

                    # Collect frame and mask for HQ/crack_fix modes
                    image = render_pkg['render']
                    mask = (render_pkg["final_opacity"].detach().cpu()[0] < 0.8).float().detach().cpu()
                    image_cpu = image.squeeze().detach().cpu().permute(1, 2, 0)
                    if orbit_state['mode'] == 'high_quality' or orbit_state['mode'] == 'crack_fix':
                        orbit_state['collected_frames'].append(image_cpu)
                        orbit_state['collected_masks'].append(mask)

                    rendered_img = render_pkg['render']
                    rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
                    rendered_image = (rendered_image * 255).astype(np.uint8)
                    rendered_image = rendered_image[..., ::-1]
                    latest_frame = rendered_image

                    socketio.emit('server-state',
                                  f'Collecting frames: {len(orbit_state["collected_frames"])}/{len(orbit_state["cameras"])}',
                                  room=client_id)

                    orbit_state['current_frame'] = (orbit_state['current_frame'] + 1) % len(orbit_state['cameras'])

                    if orbit_state['current_frame'] == 0:
                        orbit_state['is_orbiting'] = False
                        if orbit_state['mode'] == 'high_quality':
                            print("Orbit complete (high_quality), setting hq_ready flag")
                            orbit_state['hq_ready'] = True
                        elif orbit_state['mode'] == 'crack_fix':
                            print("Orbit complete (crack_fix), setting hq_ready flag")
                            orbit_state['hq_ready'] = True
                        else:
                            orbit_state['cameras'] = None
                else:
                    # --- Normal rendering path ---
                    current_camera = get_camera_by_js_view_matrix(
                        view_matrix_wonder, config,
                        fx=fx_wonder, fy=fy_wonder,
                    )
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
                    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
                    rendered_img = render_pkg['render']
                    rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
                    rendered_image = (rendered_image * 255).astype(np.uint8)
                    rendered_image = rendered_image[..., ::-1]
                    latest_frame = rendered_image

        except Exception as e:
            print(f"Render error: {e}", flush=True)

        if latest_frame is not None and client_id is not None:
            global IMAGE_COMPRESSION_QUALITY, MAX_IMAGE_SIZE, ENABLE_RESOLUTION_SCALING

            processed_frame = latest_frame

            if ENABLE_RESOLUTION_SCALING:
                height, width = latest_frame.shape[:2]
                if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
                    scale = MAX_IMAGE_SIZE / max(height, width)
                    new_height, new_width = int(height * scale), int(width * scale)
                    processed_frame = cv2.resize(latest_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_COMPRESSION_QUALITY]
            success, encoded_img = cv2.imencode('.jpg', processed_frame, encode_params)

            if success:
                image_bytes = encoded_img.tobytes()
                socketio.emit('frame', image_bytes, room=client_id)
            else:
                print("Image encoding failed")

            socketio.emit('iter-number', f'Iter: {iter_number}', room=client_id)


# ===================================================================
# SocketIO event handlers
# ===================================================================

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    global client_id
    client_id = request.sid


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    global client_id
    client_id = None


@socketio.on('render-pose')
def handle_render_pose(data):
    global view_matrix_wonder, fx_wonder, fy_wonder

    if isinstance(data, dict) and 'viewMatrix' in data:
        view_matrix_wonder = data.get('viewMatrix')
        fx_wonder = data.get('fx') if data.get('fx') is not None else config["init_focal_length"]
        fy_wonder = data.get('fy') if data.get('fy') is not None else config["init_focal_length"]
    else:
        # fallback for old clients
        view_matrix_wonder = data
        fx_wonder = config["init_focal_length"]
        fy_wonder = config["init_focal_length"]


@socketio.on('generate-nvs')
def handle_generate_nvs():
    global orbit_state
    orbit_state['is_orbiting'] = True
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'normal'
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = None


@socketio.on('generate-nvs-hq')
def handle_generate_nvs_hq():
    global orbit_state
    print("Starting high-quality NVS...")
    current_camera = get_camera_by_js_view_matrix(
        view_matrix_wonder, config,
        fx=fx_wonder, fy=fy_wonder,
    )
    orbit_state['is_orbiting'] = True
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'high_quality'
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = current_camera
    socketio.emit('server-state', 'Starting high-quality NVS...', room=client_id)


@socketio.on('fix-small-cracks')
def handle_fix_small_cracks():
    global orbit_state
    print("Starting small cracks fixing...")
    current_camera = get_camera_by_js_view_matrix(
        view_matrix_wonder, config,
        fx=fx_wonder, fy=fy_wonder,
    )
    orbit_state['is_orbiting'] = True
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'crack_fix'
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = current_camera
    orbit_state['hq_ready'] = False
    socketio.emit('server-state', 'Starting small cracks fixing...', room=client_id)


# --- Stub handlers for generation events (no-op in render-only mode) ---

@socketio.on('start')
def handle_start(data):
    print('Received start signal (render-only mode, no-op)')

@socketio.on('gen')
def handle_gen(data):
    print('Received gen signal (render-only mode, no-op)')

@socketio.on('rewrite')
def handle_rewrite():
    print('Received rewrite signal (render-only mode, no-op)')

@socketio.on('toggle-use-cog')
def handle_toggle_use_cog(data):
    print('Received toggle-use-cog (render-only mode, no-op)')

@socketio.on('scene-prompt')
def handle_new_prompt(data):
    print(f'Received scene prompt: {data} (render-only mode, no-op)')

@socketio.on('undo')
def handle_undo():
    print('Received undo (render-only mode, no-op)')

@socketio.on('save')
def handle_save():
    print('Received save signal.')
    if gaussians is not None:
        save_path = './saved_model.pth'
        save_gaussian_with_global_labels(gaussians, save_path)
        socketio.emit('server-state', f'Model saved to {save_path}', room=client_id)

@socketio.on('delete')
def handle_delete(data):
    print('Received delete (render-only mode, no-op)')

@socketio.on('fill_hole')
def handle_fill_hole():
    print('Received fill_hole (render-only mode, no-op)')

@socketio.on('add-trajectory-point')
def handle_add_trajectory_point(data):
    print('Received add-trajectory-point (render-only mode, no-op)')

@socketio.on('clear-trajectory')
def handle_clear_trajectory():
    print('Received clear-trajectory (render-only mode, no-op)')

@socketio.on('goto-nearest-zoom')
def handle_goto_nearest_zoom():
    print('Received goto-nearest-zoom (render-only mode, no-op)')

@socketio.on('start-recording')
def handle_start_recording():
    print('Received start-recording (render-only mode, no-op)')


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="WonderZoom Render-Only Server")
    parser.add_argument("--pth_path", default=None, help="Path to the .pth GaussianModel file (overrides config pth_path)")
    parser.add_argument("--base-config", default="./config/base-config.yaml", help="Base config path")
    parser.add_argument("--example_config", default="./config/more_examples/street.yaml", help="Example config path")
    parser.add_argument("--port", default=7747, type=int, help="Server port")
    args = parser.parse_args()

    # ---- Load config ----
    base_cfg = OmegaConf.load(args.base_config)
    example_cfg = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_cfg, example_cfg)
    assert 'orig_H' in config and 'orig_W' in config, "orig_H and orig_W must be set in the example config"

    # ---- Load config-defined generate_orbit_cameras if present ----
    if 'generate_orbit_cameras_code' in config and config.generate_orbit_cameras_code:
        try:
            print("Loading custom generate_orbit_cameras from config...")
            local_ns = {}
            exec(config.generate_orbit_cameras_code, globals(), local_ns)
            generate_orbit_cameras = local_ns['generate_orbit_cameras']
            print("Successfully loaded custom generate_orbit_cameras")
        except Exception as e:
            print(f"Error loading custom generate_orbit_cameras: {e}, using default")

    # ---- Load gaussians from .pth ----
    pth_path = args.pth_path or config.get('pth_path', None)
    if pth_path is None:
        raise ValueError("No pth_path specified. Use --pth_path or set pth_path in the example config.")
    print(f"Loading model from: {pth_path}")
    gaussians = load_gaussian_with_global_labels(pth_path, config)

    # ---- Initialize camera state ----
    fx_wonder = config["init_focal_length"]
    fy_wonder = config["init_focal_length"]

    print(f"Ready! Starting server on port {args.port}")

    # ---- Start server thread ----
    server_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=args.port, allow_unsafe_werkzeug=True),
    )
    server_thread.start()

    # ---- Start render thread ----
    render_thread = threading.Thread(target=render_current_scene, daemon=True)
    render_thread.start()

    server_thread.join()
