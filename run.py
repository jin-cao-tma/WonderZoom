"""WonderZoom generation server.

Builds a multi-scale 3D Gaussian scene from one image and grows it interactively: camera moves
(Gen3C), zoom-ins (Chain-of-Zoom) and optional object insertion (Step1X-Edit). The browser UI
(splat-main/index_gen.html) talks to this process over Socket.IO; the three video/image models run
as worker processes in their own environments (services/, config/services.yaml).

    python run.py --example_config config/more_examples/street.yaml
    python run.py --image my_photo.jpg --name my_scene
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Bootstrap. This runs before torch is imported: CUDA_VISIBLE_DEVICES must be narrowed to the main
# GPU first, and --help must work without a GPU.
# ---------------------------------------------------------------------------------------------
WZ_ROOT = Path(__file__).resolve().parent
ORIG_CWD = os.getcwd()  # user-relative CLI paths (--image, --example_config, ...) resolve against it


def build_arg_parser():
    parser = ArgumentParser(description="WonderZoom generation server")
    parser.add_argument("--example_config", default="config/more_examples/street.yaml",
                        help="per-scene config merged over the base config")
    parser.add_argument("--base-config", "--base_config", dest="base_config", default="config/base-config.yaml",
                        help="base config")
    parser.add_argument("--image", default=None,
                        help="generate from your own image (uses config/custom_template.yaml unless "
                             "--example_config is given explicitly)")
    parser.add_argument("--name", default=None, help="scene name used with --image (default: the file name)")
    parser.add_argument("--services_config", default="config/services.yaml", help="model services config")
    parser.add_argument("--no_services", action="store_true",
                        help="do not start Gen3C / Chain-of-Zoom / Step1X-Edit (view and edit the initial scene only)")
    parser.add_argument("--gpu_policy", default=None, choices=["auto", "resident", "exclusive"],
                        help="override gpu.policy of the services config")
    parser.add_argument("--main_gpu", type=int, default=None,
                        help="logical GPU index of this process (default: gpu.main_device of the services config)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 to serve other machines)")
    parser.add_argument("--port", default=7747, type=int, help="port of the web UI / Socket.IO server")
    parser.add_argument("--stream_max_size", default=256, type=int,
                        help="longest edge (pixels) of the streamed preview frames")
    parser.add_argument("--stream_quality", default=20, type=int, help="JPEG quality (1-100) of the streamed frames")
    parser.add_argument("--debug", action="store_true",
                        help="open a post-mortem debugger when a request fails: ipdb if installed, else pdb "
                             "(default: log, roll back and continue)")
    parser.add_argument("--dry_run", action="store_true",
                        help="parse the configs, import every module and exit before loading any model")
    return parser


def _resolve_cli_path(path):
    """Resolve a user-given path against the caller's cwd first, then against the repository root."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    for base in (ORIG_CWD, str(WZ_ROOT)):
        candidate = os.path.abspath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(ORIG_CWD, path))


def _bootstrap_gpu(args):
    """Narrow CUDA_VISIBLE_DEVICES to the main GPU and remember the original list for the workers."""
    main_gpu = args.main_gpu
    if main_gpu is None:
        # gpu.main_device of the merged services config (services.yaml + services.local.yaml + env),
        # the same layers the service manager reads later. services.config needs only omegaconf.
        try:
            if str(WZ_ROOT) not in sys.path:
                sys.path.insert(0, str(WZ_ROOT))
            from services.config import load_services_config as _load_services_config
            services_cfg = _load_services_config(main_path=_resolve_cli_path(args.services_config),
                                                 local_path="config/services.local.yaml")
            main_gpu = int(services_cfg.gpu.get("main_device", 0) or 0)
        except Exception as e:
            print(f"warning: could not read gpu.main_device from the services config ({e}); using GPU 0",
                  file=sys.stderr)
            main_gpu = 0
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    # The workers map their logical device indices through this list ('' or unset = all GPUs).
    os.environ["WZ_PARENT_VISIBLE_DEVICES"] = original if original is not None else ""
    if original is not None and not original.strip():
        # Explicitly empty: the caller hides every GPU (e.g. a CPU-only --dry_run). Keep it that way.
        main_physical = ""
    elif original is None:
        main_physical = str(main_gpu)
    else:
        tokens = [t.strip() for t in original.split(",") if t.strip()]
        if main_gpu < 0 or main_gpu >= len(tokens):
            sys.exit(f"--main_gpu/gpu.main_device {main_gpu} does not exist: CUDA_VISIBLE_DEVICES={original}")
        main_physical = tokens[main_gpu]
    os.environ["CUDA_VISIBLE_DEVICES"] = main_physical
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return main_gpu


if __name__ == "__main__":
    ARGS = build_arg_parser().parse_args()
else:  # imported as a module (tools, tests): defaults only
    ARGS = build_arg_parser().parse_known_args([])[0]
MAIN_GPU = _bootstrap_gpu(ARGS) if __name__ == "__main__" else (ARGS.main_gpu or 0)

os.chdir(WZ_ROOT)
if sys.path[0] != str(WZ_ROOT):
    sys.path.insert(0, str(WZ_ROOT))
# The vendored MoGe and GeometryCrafter directories are put on sys.path by models/vdm_model.py
# (MoGe in front of site-packages, GeometryCrafter behind WonderZoom's own 'utils' package).

import torch
import gc
import random
import contextlib
import functools
import glob
import importlib
import importlib.util
import json
import logging
import signal
import traceback
from matplotlib import pyplot as plt

import shutil
from gaussian_renderer import compute_inv_target_scale_per_frame
from utils.general import rotation2normal
from utils.loss import anisotropy_regularizer

from PIL import Image
from datetime import datetime
import threading
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch.nn.functional as F
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline
from models.vdm_model import VideoGaussianProcessor, load_image_and_resize
from util.utils import compute_pose_distances, interpolate_cameras_RT, interpolate_cameras_K, convert_pt3d_cam_to_3dgs_cam, save_rough_video, compute_trajectory_distances
from util.segment_utils import create_mask_generator_repvit
from utils.zoom_utils import zoom_image_by_focal_change

from arguments_in import GSParams
from gaussian_renderer import render

from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim, scaling_regularization_loss
from random import randint
import time
import cv2
import warnings

import copy
from services import ServiceManager, ServiceError, load_services_config, MAIN_TENANT
# Optional object insertion: util.back_ground (GroundingDINO, SAM, SD2 inpainting), util.gpt4 and
# INR-Harmonization are imported lazily (see "Optional object insertion" below).


warnings.filterwarnings("ignore")

# global label management system - ensure all GaussianModel instances use a unified label mapping
GLOBAL_LABEL_NAMES = ["main"]  # global label name list, 0="main"
GLOBAL_LABEL_MAP = {"main": 0}  # name-to-ID fast mapping

app = Flask(__name__)
CORS(app)  # Enable CORS on the Flask app
# WebSocket optimization config - enable compression and optimized transport
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    compression=True,  # enable compressed transport
    max_decode_packets=50,  # increase decode packet count
    ping_timeout=60,  # increase ping timeout
    ping_interval=25,  # adjust ping interval
    async_mode='threading'  # use threading mode for better performance
)

xyz_scale = 1000
client_id = None
scene_name = None

# image transmission optimization config
IMAGE_COMPRESSION_QUALITY = ARGS.stream_quality  # JPEG quality (1-100) of the streamed frames
MAX_IMAGE_SIZE = ARGS.stream_max_size  # longest edge (pixels) of the streamed frames
ENABLE_RESOLUTION_SCALING = True  # whether to enable dynamic resolution scaling
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_delete = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
rewrite_background = True

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
latest_frame = None
keep_rendering = True

gaussians = None
opt = GSParams()
undo = False
save = False
delete = False
grounded_sam = None
fx_wonder = None
fy_wonder = None
# fx/fy of the accepted 'gen' request: the job uses them, since 'render-pose' keeps overwriting fx_wonder.
gen_fx = None
gen_fy = None
# Object prompt latched by handle_gen for the zoom-in it accepted (None for camera moves). A prompt
# sent while that zoom runs stays in scene_name for the next zoom-in.
job_scene_name = None
kf_gen = None
config = None
depth_align_masks = None
all_depths = None
gaussians_obj = None
gen_matrices = []   # always 4x4
cameras_train_memory = []
imgs_train_memory = []
first_time_move = True
orbit_state = {
    'is_orbiting': False,
    'cameras': None,
    'current_frame': 0 ,
    'mode': 'normal',  # 'normal' or 'high_quality'
    'collected_frames': [],  # For high quality mode
    'collected_masks': [],   # For high quality mode
    'input_camera': None,   # Store the initial camera for input.png
    'hq_ready': False      # Flag to indicate HQ processing should start
}
trajectory_points = [] 
full_pose_seq = None

# ---------------------------------------------------------------------------------------------
# Session directory, model services, GPU sharing and the status protocol
# ---------------------------------------------------------------------------------------------
SESSION_DIR = None  # <paths.runs_dir>/<example_name>/<YYYYmmdd-HHMMSS>; the cwd once start-up is done
SESSION_SUBDIRS = (
    "cache", "cache/zoom_frames", "cache/crack_fix", "cache/coz_with_object",
    "frames/saved_frames/input", "frames/saved_frames/frames", "frames/saved_frames/masks",
    "frames/saved_frames/output", "services", "logs", "scenes",
)
svc = None            # services.ServiceManager (Gen3C / Chain-of-Zoom / Step1X-Edit workers)
services_cfg = None   # merged config/services.yaml (+ services.local.yaml), absolute paths
FEATURES = {"gen3c": False, "coz": False, "objects": False, "gpt": False}
MAIN_DEVICE = "cuda"  # the main process only sees its own GPU (CUDA_VISIBLE_DEVICES is narrowed)
segment_model = None
segment_processor = None
normal_estimator = None
mask_generator = None
harmo_model = None
# Extra callables returning main-process models that must leave the GPU under the exclusive policy.
MAIN_MODEL_EXTRA = []

status_lock = threading.Lock()
server_status = {"state": "loading", "job": None, "message": "Starting...", "session_dir": None}
busy_job = None                  # job accepted from the client and not finished yet
complete_background_pose = None  # camera queued by 'complete-background'
coz_request_seed = None          # optional Chain-of-Zoom seed of the pending zoom ('gen' cozSeed, tests)


# ---------------------------------------------------------------------------------------------
# Scene lock and undo
# ---------------------------------------------------------------------------------------------
class SceneLock:
    """Re-entrant lock around the scene (gaussians and the global labels).

    The main loop holds it while a job changes the scene and the render thread while it renders, so a
    frame is never rendered from a half-updated model. suspended() releases it for the duration of a
    long model-service call (the scene is consistent then), so that the preview keeps streaming."""

    def __init__(self):
        self._lock = threading.RLock()
        self._owner = None
        self._depth = 0
        self.suspended_by = None  # thread that released the lock for a service call

    def acquire(self, blocking=True, timeout=-1):
        if not self._lock.acquire(blocking, timeout):
            return False
        self._owner = threading.get_ident()
        self._depth += 1
        return True

    def release(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("scene_lock released by a thread that does not hold it")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
        self._lock.release()

    def release_all(self):
        """Release every hold of the calling thread (error recovery)."""
        while self._owner == threading.get_ident():
            self.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    @contextlib.contextmanager
    def suspended(self):
        """Fully release the lock if this thread holds it; re-acquire it afterwards."""
        if self._owner != threading.get_ident():
            yield
            return
        depth = self._depth
        for _ in range(depth):
            self.release()
        self.suspended_by = threading.get_ident()
        try:
            yield
        finally:
            for _ in range(depth):
                self.acquire()
            self.suspended_by = None


scene_lock = SceneLock()
# Scene state before the last scene-changing job: restored when that job fails, otherwise kept for 'undo'.
scene_snapshot = None


def take_scene_snapshot():
    """Copy everything a job may change: the gaussians, the global labels, the generated camera poses
    and the training-camera memory."""
    return {
        "gaussians": copy.deepcopy(gaussians),
        "label_names": list(GLOBAL_LABEL_NAMES),
        "label_map": dict(GLOBAL_LABEL_MAP),
        "n_gen_matrices": len(gen_matrices),
        "n_cameras_memory": len(cameras_train_memory),
        "n_imgs_memory": len(imgs_train_memory),
        "opt": copy.deepcopy(opt),
    }


def restore_scene_snapshot(snapshot):
    """Go back to a snapshot of take_scene_snapshot (the snapshot is consumed). Call with scene_lock held."""
    global gaussians, opt
    gaussians = snapshot["gaussians"]
    opt = snapshot["opt"]  # jobs change opt temporarily; a failed job may not have restored it
    GLOBAL_LABEL_NAMES[:] = snapshot["label_names"]
    GLOBAL_LABEL_MAP.clear()
    GLOBAL_LABEL_MAP.update(snapshot["label_map"])
    del gen_matrices[snapshot["n_gen_matrices"]:]
    del cameras_train_memory[snapshot["n_cameras_memory"]:]
    del imgs_train_memory[snapshot["n_imgs_memory"]:]


# ---------------------------------------------------------------------------------------------
# Optional object insertion: GroundingDINO + SAM, SD2 inpainting, INR-Harmonization and GPT-4o
# prompts. Nothing of it is imported before it is used; features.object_insertion and
# features.harmonization ('auto') are resolved by detect_object_features() at start-up.
# ---------------------------------------------------------------------------------------------
OBJECTS_MISSING = []        # why object insertion is unavailable
GROUNDED_SAM_MISSING = []   # why GroundingDINO + SAM cannot be loaded
HARMONIZATION_MISSING = []  # why INR-Harmonization cannot be used
_harmonization_warned = False


def _gpt4():
    """util.gpt4 (GPT-4o prompts with config fallbacks), imported on first use."""
    return importlib.import_module("util.gpt4")


def _objects_cfg():
    return (services_cfg.get("objects") if services_cfg is not None else None) or {}


def _feature_flag(name):
    """features.<name> of the services config: True, False or 'auto'."""
    features = (services_cfg.get("features") if services_cfg is not None else None) or {}
    value = features.get(name, "auto")
    if isinstance(value, str):
        value = {"true": True, "false": False}.get(value.strip().lower(), "auto")
    return value if isinstance(value, bool) else "auto"


def grounded_sam_requirements():
    """What is missing for GroundingDINO + SAM (empty list: available)."""
    missing = []
    for module in ("groundingdino", "segment_anything"):
        if importlib.util.find_spec(module) is None:
            missing.append(f"python package '{module}' is not installed (scripts/install_objects_optional.sh)")
    objects = _objects_cfg()
    for key in ("groundingdino_checkpoint", "sam_checkpoint", "groundingdino_config"):
        path = objects.get(key)
        if path and not os.path.isfile(str(path)):
            missing.append(f"objects.{key} not found: {path} (scripts/download_checkpoints.sh --objects)")
    return missing


def harmonization_requirements():
    """What is missing for INR-Harmonization (empty list: available)."""
    missing = []
    objects = _objects_cfg()
    repo, ckpt = objects.get("inr_repo_dir"), objects.get("inr_checkpoint")
    if not repo or not os.path.isfile(os.path.join(str(repo), "inr_harmonization_model.py")):
        missing.append(f"INR-Harmonization is not set up in {repo} (scripts/setup_third_party.sh inr)")
    if not ckpt or not os.path.isfile(str(ckpt)):
        missing.append(f"objects.inr_checkpoint not found: {ckpt} (scripts/download_checkpoints.sh --objects)")
    for module in ("adamp", "albumentations"):
        if importlib.util.find_spec(module) is None:
            missing.append(f"python package '{module}' is not installed (requirements/main-objects.txt)")
    return missing


def detect_object_features():
    """Resolve features.object_insertion and features.harmonization ('auto' = available)."""
    GROUNDED_SAM_MISSING[:] = grounded_sam_requirements()
    flag = _feature_flag("object_insertion")
    if flag is False:
        OBJECTS_MISSING[:] = ["disabled in config/services.yaml (features.object_insertion: false)"]
    else:
        missing = []
        if svc is None or not svc.enabled("step1x"):
            missing.append("the Step1X-Edit service is not enabled (services.step1x, scripts/register_env.py)")
        OBJECTS_MISSING[:] = missing + GROUNDED_SAM_MISSING
        if OBJECTS_MISSING and flag is True:
            print("⚠️ features.object_insertion is true but object insertion is unavailable:")
    FEATURES["objects"] = not OBJECTS_MISSING
    if FEATURES["objects"]:
        print("✅ Object insertion available (GroundedSAM / inpainting load on first use)")
    else:
        print("ℹ️ Object insertion unavailable: " + "; ".join(OBJECTS_MISSING))
    if _feature_flag("harmonization") is False:
        HARMONIZATION_MISSING[:] = ["disabled in config/services.yaml (features.harmonization: false)"]
    else:
        HARMONIZATION_MISSING[:] = harmonization_requirements()
    if FEATURES["objects"] and HARMONIZATION_MISSING:
        print("ℹ️ INR-Harmonization unavailable (objects are inserted without it): " + "; ".join(HARMONIZATION_MISSING))


def get_grounded_sam():
    """GroundingDINO + SAM, built on first use from the objects.* checkpoints."""
    global grounded_sam
    if grounded_sam is None:
        if GROUNDED_SAM_MISSING:
            raise RuntimeError("GroundedSAM is unavailable: " + "; ".join(GROUNDED_SAM_MISSING))
        from util.back_ground import GroundedSAMSegmentationModel
        objects = _objects_cfg()
        grounded_sam = GroundedSAMSegmentationModel(
            grounding_config_path=objects.get("groundingdino_config"),
            grounding_checkpoint_path=objects.get("groundingdino_checkpoint"),
            sam_checkpoint_path=objects.get("sam_checkpoint"),
            device=MAIN_DEVICE)
    return grounded_sam


class LazyGroundedSAM:
    """Stand-in for GroundedSAM handed to the VideoGaussianProcessor (pull_foreground_depth_rewrite):
    the models are only loaded when the option is actually used."""

    def __getattr__(self, name):
        if name.startswith("__"):  # copy / pickle / introspection must not load the models
            raise AttributeError(name)
        return getattr(get_grounded_sam(), name)


def get_inpaint_background():
    """util.back_ground.inpaint_background, with the SD2 inpainting pipeline (objects.inpaint_model) loaded."""
    back_ground = importlib.import_module("util.back_ground")
    if back_ground._inpaint_pipeline is None:
        model_id = _objects_cfg().get("inpaint_model") or "sd2-community/stable-diffusion-2-inpainting"
        if not back_ground.load_inpaint_model(model_id=str(model_id), device=MAIN_DEVICE):
            raise RuntimeError(f"could not load the inpainting model {model_id} (objects.inpaint_model); see the log")
    return back_ground.inpaint_background


def get_harmo_model():
    """INR-Harmonization, built on first use; None when it is unavailable (the caller then inserts the
    edited object as it is)."""
    global harmo_model, _harmonization_warned
    if harmo_model is None and not HARMONIZATION_MISSING:
        objects = _objects_cfg()
        repo = str(objects.get("inr_repo_dir"))
        # Appended, not inserted: the INR repo has top-level 'datasets' and 'utils' packages that must
        # not shadow HF datasets or WonderZoom's utils.
        if repo not in sys.path:
            sys.path.append(repo)
        try:
            from inr_harmonization_model import INRHarmonizationModel
        except ImportError as e:
            HARMONIZATION_MISSING.append(f"cannot import INR-Harmonization from {repo}: {e}")
        else:
            harmo_model = INRHarmonizationModel(str(objects.get("inr_checkpoint")), device=MAIN_DEVICE)
    if harmo_model is None and not _harmonization_warned:
        print("⚠️ use_harmol: INR-Harmonization unavailable (" + "; ".join(HARMONIZATION_MISSING)
              + "); the edited object is inserted without harmonization")
        _harmonization_warned = True
    return harmo_model


def extract_fg_bg_for_rewrite(image_path):
    """Foreground words + background description for pull_foreground_depth_rewrite (GPT-4o)."""
    return _gpt4().extract_foreground_background(image_path, scene_name=config.get("scene_name"))


def emit_to_client(event, data):
    if client_id is not None:
        socketio.emit(event, data, room=client_id)


def set_status(state, job=None, message="", text=False):
    """Update and broadcast 'server-status'; text=True also sends the legacy 'server-state' line."""
    with status_lock:
        server_status.update(state=state, job=job, message=message, session_dir=SESSION_DIR)
        payload = dict(server_status)
    emit_to_client('server-status', payload)
    if text and message:
        emit_to_client('server-state', message)


def server_config_payload():
    cfg = config if config is not None else {}
    return {
        "init_focal_length": float(cfg.get("init_focal_length", 0) or 0),
        "gen_H": int(cfg.get("orig_H", 0) or 0),
        "gen_W": int(cfg.get("orig_W", 0) or 0),
        "features": dict(FEATURES),
    }


def begin_job(job, message):
    global busy_job
    busy_job = job
    set_status("busy", job, message, text=True)
    return time.time()


def scene_stats_payload(job=None, seconds=None):
    """'scene-stats': point count and labels of the scene (None before the initial scene exists)."""
    if gaussians is None:
        return None
    try:
        num_points = int(gaussians.get_xyz_all.shape[0])
    except Exception:
        num_points = None
    return {"num_points": num_points, "labels": list(GLOBAL_LABEL_NAMES), "last_job": job,
            "seconds": None if seconds is None else round(seconds, 1)}


def end_job(job, t0, message=None):
    """Report a finished job: 'scene-stats' plus the idle status."""
    global busy_job
    seconds = time.time() - t0
    stats = scene_stats_payload(job, seconds)
    if stats is not None:
        emit_to_client('scene-stats', stats)
    busy_job = None
    set_status("idle", None, message or f"{job} finished in {seconds:.0f} s")


def fail_job(job, exc):
    global busy_job
    busy_job = None
    message = f"{job or 'request'} failed: {exc}"
    set_status("error", job, message, text=True)


def debug_post_mortem(tb):
    """--debug: post-mortem debugger on a failed request (ipdb when installed, else pdb).

    ipdb is optional and not part of requirements/main.txt. Quitting the debugger ('q') returns
    here, so the caller's error handling (scene restore, lock release, status) still runs."""
    try:
        import ipdb as debugger
    except ImportError:
        import pdb as debugger
        print("--debug: ipdb is not installed (pip install ipdb); using pdb")
    try:
        debugger.post_mortem(tb)
    except Exception as e:  # bdb.BdbQuit and friends
        print(f"--debug: debugger exited ({type(e).__name__})")


def _require_service(name, label):
    """Wait for a service to finish loading; ServiceError when it is disabled or failed to start.

    A worker that was ready once and has since died (crash, request timeout, failed suspend/resume)
    is not waited for: the request itself restarts it (WorkerClient.ensure_started, inside the
    service's GPU lease)."""
    if svc is None or not svc.enabled(name):
        raise ServiceError(f"{label} service not enabled (see config/services.yaml and scripts/register_env.py)",
                           service=name)
    if not svc.wait_ready(name, timeout=0):
        if svc.client(name).state in ("dead", "ready"):  # 'ready' whose process has exited
            set_status("busy", busy_job, f"{label} worker is not running; restarting it...", text=True)
            return
        set_status("busy", busy_job, f"Waiting for {label} to finish loading...", text=True)
        if not svc.wait_ready(name):
            raise ServiceError(f"{label} service failed to start; see {os.path.join(svc.logs_dir, name + '.log')}",
                               service=name)


def call_gen3c_with_num_steps(condition_image_path, frames_dir, masks_dir, prompt, num_steps, out_dir=None, seed=None):
    """Gen3C novel-view video; returns dict(video_path, frames_dir, n_frames). Raises ServiceError."""
    if out_dir is None and svc is not None:
        out_dir = svc.new_request_dir("gen3c")
    with scene_lock.suspended():  # the scene is consistent while the worker runs: keep the preview live
        _require_service("gen3c", "Gen3C")
        result = svc.gen3c.generate(os.path.abspath(condition_image_path), os.path.abspath(frames_dir),
                                    os.path.abspath(masks_dir), prompt, num_steps, out_dir=os.path.abspath(out_dir),
                                    seed=seed)
    if not result or not result.get("frames_dir") or not os.path.isdir(result["frames_dir"]):
        raise ServiceError(f"Gen3C returned no frames ({result})", service="gen3c")
    return result


def call_coz_dual(prev_image_path, current_image_path, output_path=None, custom_prompt=None, seed=None):
    """One Chain-of-Zoom super-resolution step; returns the absolute output path. Raises ServiceError."""
    if seed is None:
        seed = coz_request_seed  # None: services.coz.seed, else drawn from `random` (paper behaviour)
    with scene_lock.suspended():
        _require_service("coz", "Chain-of-Zoom")
        if output_path is None:
            output_path = os.path.join(svc.new_request_dir("coz"), "coz_output.png")
        out = svc.coz.super_resolve_dual(os.path.abspath(prev_image_path), os.path.abspath(current_image_path),
                                         os.path.abspath(output_path), prompt=custom_prompt, seed=seed)
    if not out or not os.path.isfile(out):
        raise ServiceError(f"Chain-of-Zoom produced no image ({output_path})", service="coz")
    return out


def call_step1x_edit(image_path, prompt, output_path=None, seed=42, num_steps=28, cfg_guidance=6.0):
    """Step1X-Edit image edit; returns the absolute output path. Raises ServiceError."""
    with scene_lock.suspended():
        _require_service("step1x", "Step1X-Edit")
        if output_path is None:
            output_path = os.path.join(svc.new_request_dir("step1x"), "step1x_output.png")
        out = svc.step1x.edit(os.path.abspath(image_path), prompt, os.path.abspath(output_path), seed=seed,
                              num_steps=num_steps, cfg_guidance=cfg_guidance)
    if not out or not os.path.isfile(out):
        raise ServiceError(f"Step1X-Edit produced no image ({output_path})", service="step1x")
    return out


def main_models_lease():
    """GPU lease of the main-process models (a no-op unless the GPU policy is exclusive)."""
    return svc.lease(MAIN_TENANT) if svc is not None else contextlib.nullcontext()


def uses_main_models(fn):
    """Decorator: run fn while the main-process models are on the GPU."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with main_models_lease():
            return fn(*args, **kwargs)
    return wrapper


class MainModelsProxy:
    """Wraps the VideoGaussianProcessor (kf_gen) so that its heavy entry points hold the main-models
    GPU lease. Everything else (attributes, cheap camera helpers) is passed through unchanged."""

    HEAVY = frozenset((
        "process_single_img", "process_single_img_sky", "process_single_img_mask",
        "process_single_img_mask_wt_guide", "process_single_img_target", "process_zoomin_frames_rewrite",
        "process_zoomin_frames_overwrite", "process_zoomin_frames_overwrite_obj_mask", "process_video_frames",
        "process_frames_inpainting", "get_normal", "generate_sky_mask", "get_sam_masks", "finetune_depth_model",
    ))

    def __init__(self, target):
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name):
        value = getattr(object.__getattribute__(self, "_target"), name)
        if name in MainModelsProxy.HEAVY and callable(value):
            @functools.wraps(value)
            def leased(*args, **kwargs):
                with main_models_lease():
                    return value(*args, **kwargs)
            return leased
        return value

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_target"), name, value)


def _main_model_modules():
    """The heavy main-process models that are moved to host RAM under the exclusive policy."""
    target = getattr(kf_gen, "_target", kf_gen) if kf_gen is not None else None
    candidates = []
    if target is not None:
        candidates += [getattr(target, name, None) for name in ("moge", "pipe", "point_map_vae")]
    candidates += [segment_model, normal_estimator]
    if mask_generator is not None:
        candidates.append(getattr(getattr(mask_generator, "predictor", None), "model", None))
    if grounded_sam is not None:
        candidates.append(getattr(grounded_sam, "grounding_model", None))
        candidates.append(getattr(getattr(grounded_sam, "sam_predictor", None), "model", None))
    candidates.append(harmo_model)
    back_ground = sys.modules.get("util.back_ground")
    if back_ground is not None:
        candidates.append(getattr(back_ground, "_inpaint_pipeline", None))
    for getter in MAIN_MODEL_EXTRA:
        try:
            candidates.append(getter())
        except Exception:
            pass
    modules, seen = [], set()
    for module in candidates:
        if module is not None and hasattr(module, "to") and id(module) not in seen:
            seen.add(id(module))
            modules.append(module)
    return modules


def park_main_models():
    """Move the main-process models to host RAM (exclusive GPU policy)."""
    for module in _main_model_modules():
        module.to("cpu")
    empty_cache()
    return True


def unpark_main_models():
    """Move the main-process models back to the main GPU."""
    for module in _main_model_modules():
        module.to(MAIN_DEVICE)
    return True


def reset_main_models_after_failure():
    """Undo the in-place model changes a failed job can leave behind (per-request error handler).

    A zoom-in fine-tunes MoGe in place (train mode, gradients on, Adam steps) and restores the
    pristine weights and eval mode only when the fine-tune returns, so an exception inside it
    (e.g. CUDA out of memory) would leave the next jobs with partly fine-tuned weights. This puts
    MoGe back into the state it has after every successful zoom-in: pristine weights, eval mode,
    no gradients."""
    target = getattr(kf_gen, "_target", kf_gen) if kf_gen is not None else None
    model = getattr(getattr(target, "moge", None), "model", None)
    pristine = getattr(target, "_moge_pristine_state", None)
    if model is None or pristine is None:
        return
    try:
        with main_models_lease():  # never while a start-up thread is parking the models
            model.load_state_dict(pristine)
            model.requires_grad_(False)
            model.eval()
            for param in model.parameters():
                param.grad = None  # the fine-tune zeroes them before every step; this only frees memory
    except Exception as e:
        print(f"⚠️ could not restore the pristine MoGe weights after the failure: {e}")


def gen3c_prompt(hq=False):
    """Gen3C text prompt from the config. The paper-era code sent the literal string 'None', which is
    the default for exact parity; gen3c_prompt_hq (high-quality views) defaults to gen3c_prompt."""
    prompt = config.get('gen3c_prompt', 'None')
    if hq:
        prompt = config.get('gen3c_prompt_hq', prompt)
    return 'None' if prompt is None else str(prompt)


def reset_orbit_state():
    """Forget the orbit (preview / HQ-NVS / crack-fix) collected by the render thread."""
    orbit_state['is_orbiting'] = False
    orbit_state['hq_ready'] = False
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = None
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'normal'


def save_scene_snapshot(gaussians_to_save):
    """Save the scene to <session>/scenes/<example_name>_<NNN>.pth (loadable by run_render_only.py)."""
    scenes_dir = os.path.join(SESSION_DIR or os.getcwd(), "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    name = str(config.get('example_name', 'scene'))
    index = len(glob.glob(os.path.join(scenes_dir, f"{glob.escape(name)}_*.pth")))
    while True:
        path = os.path.join(scenes_dir, f"{name}_{index:03d}.pth")
        if not os.path.exists(path):
            break
        index += 1
    save_gaussian_with_global_labels(gaussians_to_save, path)
    return path


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def clear_frames_directories():
    """Clear frames/saved_frames related directories"""
    import glob


    directories_to_clear = [
        "frames/saved_frames/frames",
        "frames/saved_frames/output", 
        "frames/saved_frames/masks"
    ]

    total_cleared = 0
    print("🧹 Clearing frames directories...")

    for directory in directories_to_clear:
        # ensure directory exists
        os.makedirs(directory, exist_ok=True)

        # get all files
        files = glob.glob(os.path.join(directory, "*"))

        # delete all files
        for file in files:
            try:
                os.remove(file)
                total_cleared += 1
            except:
                pass

    print(f"✅ Cleared {total_cleared} files from frames directories")

@uses_main_models  # GroundedSAM / inpainting / harmonization run on the main GPU
def get_pure_background(image_path, foreground_word = None, mask = None):
    """Inpaint the object (foreground_word, or the given mask) out of image_path; returns the HxWx3 image."""
    foreground_list, background_description = _gpt4().extract_foreground_background(image_path, scene_name=foreground_word)

    if foreground_word is not None:
        foreground_list = [foreground_word]
    print(f"   Foreground objects: {foreground_list}")
    print(f"   Background description: {background_description}")
        # background_description = None

    # image_path = "./cache/object_frame_a small snail_detected.png"
    # foreground_list = ["a snail"]
    foreground_mask, combined_mask, masks = get_grounded_sam().get_combined_foreground_mask(image_path, foreground_list, kernel_size=9)
    if mask is not None:
        foreground_mask = mask.squeeze().bool()
        mask_np = mask.cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        dilated_mask_tensor = torch.from_numpy(dilated_mask).bool()
        foreground_mask = dilated_mask_tensor.squeeze().bool()
    else:
        # No mask given: use the GroundedSAM mask, already dilated with the same 9x9 ellipse.
        dilated_mask_tensor = foreground_mask.squeeze().bool()

    inpaint_background = get_inpaint_background()
    inpainted_image = inpaint_background(image_path, dilated_mask_tensor.squeeze().bool(), background_description, 50)
    print(f"   Inpainting complete")

    inpainted_image = torch.tensor(np.array(inpainted_image.convert("RGB")))/255.
    original_image = torch.tensor(np.array(Image.open(image_path).convert("RGB")))/255.

    inpainted_image = inpainted_image * foreground_mask[:,:,None] + original_image * ~foreground_mask[:,:,None]
    # Save inpainted image for debugging
    if hasattr(inpainted_image, 'save'):
        # If it's a PIL Image, save directly
        inpainted_image.save("./cache/inpainted_background.png")
    else:
        plt.imsave("./cache/inpainted_background.png", (inpainted_image.detach().cpu().numpy()*255.).astype(np.uint8),cmap = "gray" )

    return inpainted_image


@uses_main_models  # GroundedSAM / inpainting / harmonization run on the main GPU
def get_3d_background(current_camera):
    """
    Complete 3D background inpainting pipeline.
    Takes an image path and returns a merged Gaussian model with inpainted background.

    Args:
        current_camera: 

    Returns:
        GaussianModel: Merged Gaussian model with original foreground and inpainted background
    """
    global kf_gen, opt, xyz_scale, gaussians, config

    print("🎨 Starting 3D background inpainting pipeline...")

    # Ensure cache directory exists

    os.makedirs("./cache", exist_ok=True) 
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera,xyz_scale=xyz_scale, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
    median_depth = render_pkg['median_depth'][0]/xyz_scale
    image = render_pkg['render']
    plt.imsave("./cache/current_image.png", image.detach().permute(1,2,0).cpu().numpy(),cmap = "gray" )
    image_path = "./cache/current_image.png"
    # Step 1: Use GPT to extract foreground objects and background description
    print("🤖 Step 1: Extracting foreground and background with GPT...")
    foreground_list, background_description = _gpt4().extract_foreground_background(image_path)
    print(f"   Foreground objects: {foreground_list}")
    print(f"   Background description: {background_description}")
    if not foreground_list:
        raise RuntimeError("complete-background: no foreground objects to remove (GPT-4o needs OPENAI_API_KEY; "
                           "otherwise set foreground_words in the example config)")

    # Step 2: Use GroundedSAM to extract foreground mask
    print("🎯 Step 2: Extracting foreground mask with GroundedSAM...")
    foreground_mask, combined_mask, masks = get_grounded_sam().get_combined_foreground_mask(image_path, foreground_list, kernel_size=21)
    print(f"   Foreground mask shape: {foreground_mask.shape}")

    # Step 3: Inpaint the background
    print("🎨 Step 3: Inpainting background...")
    inpaint_background = get_inpaint_background()
    inpainted_image = inpaint_background(image_path, foreground_mask, background_description, 50)
    print(f"   Inpainting complete")

    # Save inpainted image for debugging
    if hasattr(inpainted_image, 'save'):
        # If it's a PIL Image, save directly
        inpainted_image.save("./cache/inpainted_background.png")
    else:
        # If it's a tensor, convert and save
        plt.imsave("./cache/inpainted_background.png", inpainted_image,cmap = "gray" )

    # Step 4: Process original image to get foreground Gaussian
    print("📊 Step 4: Processing original image for foreground Gaussian...")
    # points_3d_orig, colors_orig, _, normals_orig, imgs_orig, cameras_orig, focal_length_orig, is_sky_orig, all_depths_orig, depth_align_masks_orig = kf_gen.process_single_img([image_path])

    # # Create foreground Gaussian model
    # gaussians_foreground = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
    # traindata_foreground = kf_gen.convert_to_3dgs_traindata(points_3d_orig, colors_orig, normals_orig, imgs_orig, cameras_orig, xyz_scale=xyz_scale, use_no_loss_mask=False)
    # scene_foreground = Scene(traindata_foreground, gaussians_foreground, opt, focal_length_orig, is_sky_orig)

    # # Train foreground Gaussian
    # print("🏋️ Training foreground Gaussian...")
    # trainCameras_foreground = scene_foreground.getTrainCameras().copy()
    # compute_3D_filter(gaussians_foreground, cameras=trainCameras_foreground, initialize_scaling=True)
    t1 = opt.iterations
    opt.iterations = 200
    # train_gaussian(gaussians_foreground, scene_foreground, opt, initialize_scaling=True, xyz_scale=xyz_scale, newly_added_points=gaussians_foreground.get_xyz.shape[0])

    # Step 5: Process inpainted image to get background Gaussian
    print("📊 Step 5: Processing inpainted image for background Gaussian...")
    # Save inpainted image temporarily
    temp_inpainted_path = "./cache/temp_inpainted.png"
    if hasattr(inpainted_image, 'save'):
        # If it's a PIL Image, save directly
        inpainted_image.save(temp_inpainted_path)
    else:
        # If it's a tensor, convert and save
        plt.imsave(temp_inpainted_path, inpainted_image,cmap = "gray" )
    # plt.imsave("./cache/inpainted_background.png", combined_mask.squeeze().cpu().numpy(),cmap = "gray" )
    points_3d_bg, colors_bg, _, normals_bg, imgs_bg, cameras_bg, focal_length_bg, is_sky_bg, now_scale_bg = kf_gen.process_single_img_mask_wt_guide([temp_inpainted_path],combined_mask, [current_camera], median_depth, in_sam_masks=masks)

    # Create background Gaussian model
    gaussians_background = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9, config=config)
    traindata_background = kf_gen.convert_to_3dgs_traindata(points_3d_bg, colors_bg, normals_bg, imgs_bg, cameras_bg, xyz_scale=xyz_scale, use_no_loss_mask=False)
    scene_background = Scene(traindata_background, gaussians_background, opt, focal_length_bg, is_sky_bg, now_scale_bg)

    # Train background Gaussian
    print("🏋️ Training background Gaussian...")
    trainCameras_background = scene_background.getTrainCameras().copy()
    compute_3D_filter(gaussians_background, cameras=trainCameras_background, initialize_scaling=True)
    train_gaussian(gaussians_background, scene_background, opt, initialize_scaling=True,no_loss_masks = [~combined_mask.bool().squeeze()], xyz_scale=xyz_scale, newly_added_points=gaussians_background.get_xyz.shape[0])
    opt.iterations = t1
    # Step 6: set background label and freeze
    print("🔗 Step 6: Setting background label and freezing...")

    # set "background" label for all background points
    n_bg_points = gaussians_background.get_xyz_all.shape[0]
    all_points_mask = torch.ones(n_bg_points, dtype=torch.bool, device='cuda')
    gaussians_background.set_points_label(all_points_mask, "background")

    # freeze background points after training (set as non-trainable)
    gaussians_background.freeze_labels("background")

    print(f"✅ 3D background inpainting complete!")
    print(f"   Final Gaussian model has {gaussians_background.get_xyz_all.shape[0]} points (all labeled as 'background' and frozen)")

    # Clean up temporary file

    if os.path.exists(temp_inpainted_path):
        os.remove(temp_inpainted_path)

    return gaussians_background


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")

def run(config, continue_flag = True):
    global first_time_move, client_id, view_matrix, scene_name, latest_frame, keep_rendering, kf_gen, gaussians, opt, background, undo, save, delete, view_matrix_delete
    global trajectory_points, cameras_train_memory, imgs_train_memory, depth_align_masks, all_depths, full_pose_seq, gen_matrices, rewrite_background
    global gaussians_obj, GLOBAL_LABEL_MAP, GLOBAL_LABEL_NAMES, complete_background_pose, scene_snapshot, coz_request_seed
    global job_scene_name
    ###### ------------------ Load modules ------------------ ######
    keep_rendering = True
    if not continue_flag:
        gen_matrices = []
        cameras_train_memory = []
        imgs_train_memory = []
        trajectory_points = []
        GLOBAL_LABEL_MAP = {"main": 0}
        GLOBAL_LABEL_NAMES = ["main"]
        scene_snapshot = None
    if not continue_flag:
        t_init = begin_job('initial_scene', 'Building the initial scene...')
        seeding(config["seed"])


        # fx_wonder = config["init_focal_length"]
        # fy_wonder = config["init_focal_length"]


        start_keyframe = load_image_and_resize(config['image_filepath'], height=config['orig_H'], width=config['orig_W'])
        kf_gen.image_latest = start_keyframe


        socketio.emit('scene-prompt', scene_name, room=client_id)

        kf_gen.increment_kf_idx()
        ###### ------------------ Main loop ------------------ ######

        first_time_move = False
        scene_lock.acquire()  # the render thread waits until the initial scene is trained
        gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9, config=config)
            # gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')  # pure sky

        # if config['load_gen'] and os.path.exists(f'examples/sky_images/{example}/finished_3dgs.ply') and os.path.exists(f'examples/sky_images/{example}/visibility_filter_all.pth') and os.path.exists(f'examples/sky_images/{example}/is_sky_filter.pth') and os.path.exists(f'examples/sky_images/{example}/delete_mask_all.pth'):
        #     print("Loading existing 3DGS...")
        #     gaussians = GaussianModel(sh_degree=0, config=config)
        #     gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs.ply')
        #     gaussians.visibility_filter_all = torch.load(f'examples/sky_images/{example}/visibility_filter_all.pth').to('cuda')
        #     gaussians.is_sky_filter = torch.load(f'examples/sky_images/{example}/is_sky_filter.pth').to('cuda')
        #     gaussians.delete_mask_all = torch.load(f'examples/sky_images/{example}/delete_mask_all.pth').to('cuda')


        points_3d, colors, _, normals, imgs, cameras, focal_length, is_sky, all_depths, depth_align_masks, now_scale = kf_gen.process_single_img([config['image_filepath']])
        traindata = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs,  cameras, xyz_scale=xyz_scale, use_no_loss_mask=False)

        # gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians
        # )
        gen_matrices.append(kf_gen.get_camera_at_origin())
        scene = Scene(traindata, gaussians, opt, focal_length, is_sky, now_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
        # gaussians.set_visible_and_restore_from_prev(tdgs_cam, opt)
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
        opt = GSParams()
        t1= opt.iterations
        t2 = opt.densify_from_iter
        t3 = opt.densify_until_iter
        opt.iterations = config.get('initial_iterations', 300)
        opt.densify_from_iter = 300
        opt.densify_until_iter = 1000
        trainCameras = scene.getTrainCameras().copy()
        # if initialize_scaling:
        # import pdb; pdb.set_trace()
        compute_3D_filter(gaussians, cameras=trainCameras ,initialize_scaling=True)

        train_gaussian(gaussians, scene, opt, all_depths=all_depths, depth_align_masks=depth_align_masks, initialize_scaling=True, xyz_scale=xyz_scale, newly_added_points=gaussians.get_xyz.shape[0])
        opt.iterations = t1
        opt.densify_from_iter = t2
        opt.densify_until_iter = t3
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale, config=config)
        gaussians.set_inscreen_points_to_visible(tdgs_cam)
        gaussians.merge_all_to_trainable()
        gaussians.point_labels[:] = 0
        print(f"✅ Initial scene built in {time.time() - t_init:.0f} s")
        end_job('initial_scene', t_init, 'Initial scene ready')
        scene_lock.release()

    # gaussians.visibility_filter_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    # gaussians.delete_mask_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    # gaussians.is_sky_filter = torch.ones(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')

    i=0

    # Every scene-changing job holds scene_lock from its start to its end (the per-request error handler
    # releases it on failure) and starts from a snapshot of the scene: restored if the job fails, kept
    # as the 'undo' state otherwise. Model-service calls release the lock while they run.
    while True:
        job_snapshot = None
        try:
            i += 1

            socketio.emit('scene-prompt', scene_name, room=client_id)
            print('Waiting for scene gen signal...')
            socketio.emit('server-state', 'Waiting to generate new scenes...', room=client_id)

            while keep_rendering:
                time.sleep(0.05)

                # Check for high-quality NVS processing
                if orbit_state['hq_ready']:
                    orbit_state['hq_ready'] = False  # Reset flag
                    job_name = 'crack_fix' if orbit_state.get('mode') == 'crack_fix' else 'hq_nvs'
                    scene_lock.acquire()
                    scene_snapshot = None
                    job_snapshot = take_scene_snapshot()
                    t_job = begin_job(job_name, 'Fixing small cracks...' if job_name == 'crack_fix' else 'Generating high-quality views...')

                    if orbit_state.get('mode') == 'high_quality':
                        print("🎬 Processing high-quality NVS in main thread...")

                        # Use the existing process_orbit_frames logic but in main thread
                        if len(orbit_state['collected_frames']) > 0:
                            pc_imgs = orbit_state['collected_frames']
                            collected_masks = orbit_state['collected_masks']

                            # Expand masks by one pixel around True areas (1 values)
                            expanded_masks = []
                            for mask in collected_masks:
                                # Convert to numpy array if it's a tensor
                                if torch.is_tensor(mask):
                                    mask_np = mask.cpu().numpy()
                                else:
                                    mask_np = np.array(mask)

                                # Create kernel for dilation (3x3 kernel for 1-pixel expansion)
                                kernel = np.ones((3, 3), np.uint8)

                                # Dilate the mask (expand 1 values by one pixel)
                                expanded_mask = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1)

                                # Convert back to float (0,1 values)
                                expanded_mask = expanded_mask.astype(np.float32)

                                # Convert back to tensor if original was tensor
                                if torch.is_tensor(mask):
                                    expanded_mask = torch.from_numpy(expanded_mask).to(mask.device)

                                expanded_masks.append(expanded_mask)

                            collected_masks = expanded_masks
                            input_camera = orbit_state.get('input_camera', None)
                            cameras_train = orbit_state.get('cameras', None)

                            # Save frames using the stored input camera
                            save_rough_video_frames(pc_imgs, collected_masks, [input_camera], input_camera=input_camera)

                            print("🎬 Calling video diffusion for high-quality NVS...")

                            # Gen3C prompt of the high-quality views (paper runs sent the literal 'None')
                            prompt_3d = gen3c_prompt(hq=True)
                            print("="*80)
                            print("GENERATED HIGH-QUALITY NVS PROMPT:")
                            print("="*80)
                            print(prompt_3d)
                            print("="*80)

                            # Use Gen3C for high-quality NVS
                            condition_image_path = os.path.join(os.getcwd(), "frames/saved_frames/input.png")
                            frames_dir = os.path.join(os.getcwd(), "frames/saved_frames/frames")
                            masks_dir = os.path.join(os.getcwd(), "frames/saved_frames/masks")

                            print(f"📤 Calling Gen3C with:")
                            print(f"   Condition image: {condition_image_path}")
                            print(f"   Frames dir: {frames_dir}")
                            print(f"   Masks dir: {masks_dir}")

                            gen3c_result = call_gen3c_with_num_steps(
                                condition_image_path=condition_image_path,
                                frames_dir=frames_dir,
                                masks_dir=masks_dir,
                                prompt=prompt_3d,
                                num_steps=config.get('gen3c_steps_hq', 10)
                            )
                            output_video_path = gen3c_result['video_path']
                            print(f"✅ Gen3C generated video: {output_video_path}")
                            # frames written by this request (never a stale directory)
                            output_frames_path = gen3c_result['frames_dir']
                            concat_video_path = output_video_path
                            print(f"✅ Gen3C frames saved to: {output_frames_path}")
                            # output_frames_path, concat_video_path, output_video_path = "./outputs/park_3d/start000_frames49_strength0d75_frames", './outputs/park_3d/start000_frames49_strength0d75_concat.mp4', './outputs/park_3d/start000_frames49_strength0d75.mp4'
                            # Send videos to frontend
                            if concat_video_path and os.path.exists(concat_video_path):
                                with open(concat_video_path, 'rb') as f:
                                    concat_bytes = f.read()
                                socketio.emit('concat-video', concat_bytes, room=client_id)

                            if output_video_path and os.path.exists(output_video_path):
                                with open(output_video_path, 'rb') as f:
                                    out_bytes = f.read()
                                socketio.emit('out-video', out_bytes, room=client_id)

                            print("🎬 Processing output frames to update gaussians...")
                            socketio.emit('server-state', 'Processing frames to update scene...', room=client_id)

                            # Process output frames like in non-zoom case
                            if output_frames_path and os.path.exists(output_frames_path):
                                roots = sorted([os.path.join(output_frames_path, name) for name in os.listdir(output_frames_path)])

                                # Get the orbit cameras we used
                                # stored_cameras = orbit_state.get('cameras', [])
                                stored_cameras = cameras_train
                                if stored_cameras is None:
                                    stored_cameras = []

                                # Check if we have cameras to process
                                if len(stored_cameras) == 0:
                                    # Reported by the per-request error handler; the server keeps running.
                                    raise RuntimeError("High-quality NVS failed: no cameras stored")

                                print(f"📹 Processing {len(stored_cameras)} cameras for high-quality NVS")

                                # Process video frames to extract 3D points and update gaussians
                                points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, all_depths, depth_align_masks, now_scale = kf_gen.process_frames_inpainting(
                                    roots, stored_cameras, gaussians, xyz_scale, opt, num=9, total_num=len(stored_cameras), return_1=False
                                )
                                # Create new gaussian model
                                gaussians_new = GaussianModel(sh_degree=0, config=config)
                                gaussians_new.floater_dist2_threshold = torch.inf

                                # Convert to training data and create scene
                                print("🎨 Mixing Gen3C generated images with original collected images...")

                                # get original collected images and masks
                                collected_frames = pc_imgs # orbit_state.get('collected_frames', [])
                                collected_masks = collected_masks # orbit_state.get('collected_masks', [])

                                if len(collected_frames) > 0 and len(collected_masks) > 0:
                                    # ensure counts match
                                    min_frames = min(len(imgs_train), len(collected_frames), len(collected_masks))
                                    mixed_imgs = []

                                    for i in range(min_frames):
                                        gen3c_img = imgs_train[i]  # Gen3C generated image
                                        original_img = collected_frames[i].permute(2,0,1)  # original collected image
                                        mask = collected_masks[i]  # holes mask

                                        # ensure dimensions match


                                        # ensure mask dimensions are correct
                                        if mask.dim() == 2:
                                            mask = mask.unsqueeze(0).expand_as(gen3c_img)
                                        elif mask.dim() == 3 and mask.shape[0] == 1:
                                            mask = mask.expand_as(gen3c_img)

                                        # blend images: use Gen3C in mask regions, original image in non-mask regions
                                        mixed_img = torch.where(mask.bool(), gen3c_img, original_img)
                                        mixed_imgs.append(mixed_img)

                                        print(f"  Frame {i+1}: Mixed Gen3C holes with original image")

                                    # update imgs_train with blended images
                                    imgs_train = mixed_imgs[:min_frames]
                                    cameras_train = cameras_train[:min_frames]
                                    # plt.imshow(,imgs_train[50])
                                    # recreate scene with mixed images
                                    train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)
                                    scene = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)

                                    print(f"✅ Mixed {len(mixed_imgs)} images successfully")

                                train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)
                                scene = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)
                                trainCameras = scene.getTrainCameras().copy()

                                # Compute 3D filter and merge with previous gaussian
                                compute_3D_filter(gaussians_new, cameras=trainCameras, initialize_scaling=True)

                                # compute new point count (record before merging)
                                n_new_points = gaussians_new.get_xyz_all.shape[0]
                                print(f"🎬 HQ mode: Adding {n_new_points} new points to existing scene")

                                # set "background_tmp" label for all newly generated points (to avoid being affected by background freeze logic)
                                if n_new_points > 0:
                                    all_new_points_mask = torch.ones(n_new_points, dtype=torch.bool, device='cuda')
                                    gaussians_new.set_points_label(all_new_points_mask, "background_tmp")
                                    print(f"🏷️ Set 'background_tmp' label for {n_new_points} new HQ-NVS points")

                                gaussians_new.merge_gaussian(gaussians)

                                # Set up training and update visibility
                                for idx, pose in enumerate(cameras_train):
                                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)
                                    gaussians_new.set_inscreen_points_to_visible(tdgs_cam)
                                t2 = opt.position_lr_final
                                t3 = opt.position_lr_init
                                opt.position_lr_final = 0.000
                                opt.position_lr_init = 0.000
                                gaussians_new.training_setup(opt)

                                # Update global gaussians
                                gaussians = gaussians_new

                                # Train the gaussian with HQ mode
                                print("🎯 Training gaussians with high-quality NVS data (HQ mode)...")
                                socketio.emit('server-state', 'Training scene with new data...', room=client_id)
                                t1 = opt.iterations
                                # opt.iterations = 3000
                                opt.iterations = 1000


                                train_gaussian(gaussians, scene, opt, initialize_scaling=True, no_loss_masks=None, zoom_in=False, xyz_scale=xyz_scale, newly_added_points=n_new_points, hq_mode=True)
                                opt.iterations = t1
                                opt.position_lr_final = t2
                                opt.position_lr_init = t3
                                print("✅ High-quality NVS and gaussians update complete!")
                                socketio.emit('server-state', 'High-quality NVS complete!', room=client_id)

                                # after training, change all "background_tmp" labeled points to "background"
                                background_tmp_mask = gaussians.get_label_mask("background_tmp")
                                if background_tmp_mask.sum() > 0:
                                    gaussians.set_points_label(background_tmp_mask, "background")
                                    print(f"🏷️ Post-training: Changed {background_tmp_mask.sum()} points from 'background_tmp' to 'background'")
                            else:
                                raise RuntimeError("High-quality NVS failed: no output frames")

                            # Clear GPU memory
                            torch.cuda.empty_cache()

                        else:
                            raise RuntimeError("High-quality NVS failed: no frames collected")

                    elif orbit_state.get('mode') == 'crack_fix':
                        print("🔧 Processing small cracks fixing in main thread...")

                        if len(orbit_state['collected_frames']) > 0:
                            pc_imgs = orbit_state['collected_frames']
                            collected_masks = orbit_state['collected_masks']
                            input_camera = orbit_state.get('input_camera', None)
                            cameras_train = orbit_state.get('cameras', None)

                            print("🔧 Starting cv2.inpaint crack fixing...")
                            socketio.emit('server-state', 'Fixing small cracks with cv2.inpaint...', room=client_id)

                            # Apply cv2.inpaint directly to collected frames
                            fixed_frames = []

                            for i, (frame, mask) in enumerate(zip(pc_imgs, collected_masks)):
                                # Convert frame to numpy
                                frame_np = frame.cpu().numpy()
                                frame_uint8 = (frame_np * 255).astype(np.uint8)

                                # Convert mask to uint8
                                mask_np = mask.cpu().numpy().astype(np.uint8)
                                if mask_np.max() <= 1:
                                    mask_np = mask_np * 255

                                # Apply cv2.inpaint
                                if mask_np.sum() > 0:
                                    inpainted = cv2.inpaint(frame_uint8, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                                    # Convert back to tensor
                                    fixed_frame = torch.tensor(inpainted.astype(np.float32) / 255.0).permute(2, 0, 1)
                                    fixed_frames.append(fixed_frame)
                                    print(f"  Frame {i+1}: Fixed {mask_np.sum()} crack pixels")
                                else:
                                    # same CHW layout as the inpainted frames (the collected frames are HWC)
                                    fixed_frames.append(frame.permute(2, 0, 1))

                            print(f"✅ Fixed {len(fixed_frames)} frames with cv2.inpaint")

                            # Generate points from fixed frames (same as HQ-NVS but with fixed images)
                            if cameras_train and len(cameras_train) > 0:
                                print("🎬 Processing fixed frames to update gaussians...")
                                socketio.emit('server-state', 'Processing fixed frames to update scene...', room=client_id)

                                # Convert fixed frames back to image paths (save temporarily)

                                os.makedirs("./cache/crack_fix", exist_ok=True)
                                temp_roots = []
                                for i, frame in enumerate(fixed_frames):
                                    temp_path = f"./cache/crack_fix/frame_{i:04d}.png"
                                    plt.imsave(temp_path, frame.permute(1, 2, 0).cpu().numpy()[:,:,:3],cmap = "gray" )
                                    temp_roots.append(temp_path)

                                # Use process_frames_inpainting to generate points
                                points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, all_depths, depth_align_masks, now_scale = kf_gen.process_frames_inpainting(
                                    temp_roots, cameras_train, gaussians, xyz_scale, opt, num=9, total_num=len(cameras_train), return_1=False
                                )

                                if points_3d.numel() > 0:
                                    # Create new gaussian model
                                    gaussians_new = GaussianModel(sh_degree=0, config=config)
                                    gaussians_new.floater_dist2_threshold = torch.inf

                                    # Convert to training data and create scene
                                    train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)
                                    scene = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)
                                    trainCameras = scene.getTrainCameras().copy()

                                    # Compute 3D filter and merge with previous gaussian
                                    compute_3D_filter(gaussians_new, cameras=trainCameras, initialize_scaling=True)

                                    # compute new point count (record before merging)
                                    n_new_points = gaussians_new.get_xyz_all.shape[0]
                                    print(f"🔧 Crack fix mode: Adding {n_new_points} new points to existing scene")

                                    # set "background_tmp" label for all newly generated points (to avoid being affected by background freeze logic)
                                    if n_new_points > 0:
                                        all_new_points_mask = torch.ones(n_new_points, dtype=torch.bool, device='cuda')
                                        gaussians_new.set_points_label(all_new_points_mask, "background_tmp")
                                        print(f"🏷️ Set 'background_tmp' label for {n_new_points} new crack-fix points")

                                    gaussians_new.merge_gaussian(gaussians)

                                    # Set up training and update visibility
                                    for idx, pose in enumerate(cameras_train):
                                        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)
                                        gaussians_new.set_inscreen_points_to_visible(tdgs_cam)

                                    # Training
                                    print("🎯 Training gaussians with crack-fixed data...")
                                    socketio.emit('server-state', 'Training scene with crack-fixed data...', room=client_id)
                                    t1 = opt.iterations
                                    opt.iterations = 1000  # Shorter training for crack fixing
                                    gaussians_new.training_setup(opt)
                                    gaussians = gaussians_new
                                    train_gaussian(gaussians_new, scene, opt, initialize_scaling=True, no_loss_masks=None, zoom_in=False, xyz_scale=xyz_scale, newly_added_points=n_new_points, hq_mode=True)

                                    opt.iterations = t1
                                    print("✅ Small cracks fixing and gaussians update complete!")
                                    socketio.emit('server-state', 'Small cracks fixing complete!', room=client_id)

                                    # after training, change all "background_tmp" labeled points to "background"
                                    background_tmp_mask = gaussians.get_label_mask("background_tmp")
                                    if background_tmp_mask.sum() > 0:
                                        gaussians.set_points_label(background_tmp_mask, "background")
                                        print(f"🏷️ Post-training: Changed {background_tmp_mask.sum()} points from 'background_tmp' to 'background'")

                                    # Update global gaussians


                                # Clean up temporary files
                                for temp_path in temp_roots:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)

                            # Clear GPU memory
                            torch.cuda.empty_cache()

                        else:
                            raise RuntimeError("Crack fixing failed: no frames collected")

                    # Reset the orbit state after every HQ / crack-fix job, whatever path it took.
                    reset_orbit_state()
                    scene_snapshot, job_snapshot = job_snapshot, None
                    end_job(job_name, t_job)
                    scene_lock.release()
                if complete_background_pose is not None:
                    # 'complete-background' (P key): inpaint and add the background behind the current view.
                    background_pose = complete_background_pose
                    complete_background_pose = None
                    scene_lock.acquire()
                    scene_snapshot = None
                    job_snapshot = take_scene_snapshot()
                    t_job = begin_job('complete_background', 'Completing the background...')
                    gaussians_background = get_3d_background(background_pose)
                    gaussians.merge_gaussian(gaussians_background)
                    scene_snapshot, job_snapshot = job_snapshot, None
                    end_job('complete_background', t_job)
                    scene_lock.release()
                if delete:
                    delete = False
                    scene_lock.acquire()
                    scene_snapshot = None
                    job_snapshot = take_scene_snapshot()
                    t_job = begin_job('delete', 'Deleting...')
                    print("Deleting...")
                    current_pt3d_cam_delete = kf_gen.get_camera_by_js_view_matrix(view_matrix_delete,fx_wonder=fx_wonder, fy_wonder=fy_wonder, xyz_scale=xyz_scale)
                    tdgs_cam_delete = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam_delete, xyz_scale=xyz_scale, config=config)
                    inscreen_points = gaussians.get_inscreen_points(tdgs_cam_delete)
                    render_pkg = render(tdgs_cam_delete, gaussians, opt, bg_color=torch.tensor([0.7, 0.7, 0.7], device='cuda'), render_visible=True, config=config)
                    visibility_filter = render_pkg['visibility_filter']
                    inscreen_points = inscreen_points & visibility_filter
                    gaussians.delete_all_points(inscreen_points)
                    scene_snapshot, job_snapshot = job_snapshot, None
                    end_job('delete', t_job)
                    scene_lock.release()
                if save:
                    save = False
                    with scene_lock:  # saving merges the non-trainable points back (a scene mutation)
                        t_job = begin_job('save', 'Saving...')
                        print("Saving...")
                        save_path = save_scene_snapshot(gaussians)
                        socketio.emit('server-state', f'Saved: {save_path}', room=client_id)
                        end_job('save', t_job, f'Saved: {save_path}')
                if undo:
                    # Undo the last scene-changing job: scene, labels, generated poses and camera memory.
                    undo = False
                    with scene_lock:
                        t_job = begin_job('undo', 'Undoing...')
                        if scene_snapshot is None:
                            message = 'Nothing to undo'
                        else:
                            print("Undoing...")
                            restore_scene_snapshot(scene_snapshot)
                            scene_snapshot = None
                            message = 'Undo done'
                        socketio.emit('server-state', message, room=client_id)
                        end_job('undo', t_job, message)

            job_name = busy_job or 'generate'
            scene_lock.acquire()
            scene_snapshot = None
            job_snapshot = take_scene_snapshot()
            t_job = begin_job(job_name, 'Generating new scene...')

            # The fx/fy and the zoom-in / camera-move decision that handle_gen validated: 'render-pose'
            # keeps overwriting fx_wonder (60 Hz) while the job starts.
            job_fx = gen_fx if gen_fx is not None else fx_wonder
            job_fy = gen_fy if gen_fy is not None else fy_wonder
            cur_pose_np = kf_gen.get_camera_by_js_view_matrix(view_matrix, fx_wonder=job_fx, fy_wonder=job_fy, xyz_scale=xyz_scale)

                # (4,4) np.float32

            if job_name in ('zoom', 'move'):
                zoom_in = job_name == 'zoom'
            else:
                zoom_in = job_fx > config["init_focal_length"]
            if not zoom_in:
                pose_closestid, _, _ = compute_pose_distances(cur_pose_np, gen_matrices)
                pose_close = gen_matrices[pose_closestid]

            cameras_train, imgs_train = None, None
            def process_one_seq(pose_seq, first_time_move=False, zoom_in=False):
                global pc_imgs, masks, now_imgs, output_frames_path, concat_video_path, output_video_path
                global gaussians, imgs, cameras, points_3d, colors, normals, train_data, scene, scene_name
                global cameras_train_memory, imgs_train_memory
                global depth_align_masks, all_depths, gaussians_obj, update_gaussian_obj

                first_time_move = False  if zoom_in else first_time_move
                pc_imgs, masks = render_rough_video(pose_seq, gaussians, xyz_scale=xyz_scale)
                now_imgs = torch.stack(pc_imgs, dim=0)
                now_masks=torch.stack(masks,dim=0).float()[...,None].repeat(1,1,1,3)
                save_rough_video(f'rough_video.mp4', now_imgs)
                rough_video_path = 'rough_video.mp4'
                if os.path.exists(rough_video_path):
                    with open(rough_video_path, 'rb') as f:
                        rough_bytes = f.read()
                    socketio.emit('rough-video', rough_bytes, room=client_id)
                masks_path = 'mask_video.mp4'
                save_rough_video(masks_path, now_masks)
                if os.path.exists(masks_path):
                    with open(masks_path, 'rb') as f:
                        rough_bytes = f.read()
                    socketio.emit('out-video', rough_bytes, room=client_id)


                if zoom_in:
                    # Zoom in case: use Chain-of-Zoom super-resolution pipeline
                    video_path, _ = render_zoomin_rough_video3(pose_seq, gaussians, job_scene_name)
                    frames_dir = os.path.join(os.getcwd(), "frames/saved_frames/frames")
                    output_dir = os.path.join(os.getcwd(), "frames/saved_frames/output")

                    
                    output_frames_path, concat_video_path, output_video_path = output_dir, video_path, video_path

                else:
                    save_rough_video_frames(pc_imgs, masks, pose_seq)

                    # use Gen3C to generate the video (outputs go to a per-request directory)
                    if svc is not None and svc.enabled('gen3c'):
                        condition_image_path = os.path.join(os.getcwd(), "frames/saved_frames/input.png")
                        frames_dir = os.path.join(os.getcwd(), "frames/saved_frames/frames")
                        masks_dir = os.path.join(os.getcwd(), "frames/saved_frames/masks")
                        # paper runs sent the literal prompt 'None' (config gen3c_prompt)
                        prompt_3d = gen3c_prompt()
                        print(f"📤 Calling Gen3C with:")
                        print(f"   Condition image: {condition_image_path}")
                        print(f"   Frames dir: {frames_dir}")
                        print(f"   Masks dir: {masks_dir}")
                        print(f"   Prompt: {prompt_3d}")

                        # call Gen3C; raises ServiceError on failure (handled per request)
                        gen3c_result = call_gen3c_with_num_steps(
                            condition_image_path=condition_image_path,
                            frames_dir=frames_dir,
                            masks_dir=masks_dir,
                            prompt=prompt_3d,
                            num_steps=config.get('gen3c_steps_move', 18)
                        )
                        output_video_path = gen3c_result['video_path']
                        print(f"✅ Gen3C generated video: {output_video_path}")
                        # frames written by this request (never a stale directory)
                        output_frames_path = gen3c_result['frames_dir']
                        concat_video_path = output_video_path
                        print(f"✅ Gen3C frames saved to: {output_frames_path}")
                    else:
                        raise ServiceError("Gen3C service not enabled: camera moves need Gen3C", service="gen3c")

                # scene_name = None

                # clean up GPU memory
                torch.cuda.empty_cache()

                # send video to frontend
                if concat_video_path and os.path.exists(concat_video_path):
                    with open(concat_video_path, 'rb') as f:
                        concat_bytes = f.read()
                    socketio.emit('concat-video', concat_bytes, room=client_id)

                if output_video_path and os.path.exists(output_video_path):
                    with open(output_video_path, 'rb') as f:
                        out_bytes = f.read()
                    socketio.emit('out-video', out_bytes, room=client_id)

                # process output frames (frame sequence only exists in zoom_in case, Gen3C outputs video directly)
                if zoom_in and output_frames_path and os.path.exists(output_frames_path):
                    roots = sorted([os.path.join(output_frames_path, name) for name in os.listdir(output_frames_path) if name.endswith(('.png','.jpg'))])
                else:
                    # In Gen3C case, directly use the input frame sequence for subsequent processing
                    roots = sorted([os.path.join(output_frames_path, name) for name in os.listdir(output_frames_path) if name.endswith(('.png','.jpg'))])  # or set as needed


                cameras = pose_seq
                if not zoom_in:
                    if first_time_move:
                        points_3d, colors, _, normals, imgs_train,cameras_train, focal_length_train, is_sky, all_depths, depth_align_masks, now_scale = kf_gen.process_video_frames(roots, pose_seq, gaussians, xyz_scale, opt, num = 49,total_num = len(pose_seq),return_1=True)
                        for idx, pose in enumerate(cameras_train):
                            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)
                            gaussians.set_visible_and_restore_from_prev(tdgs_cam, opt)
                            gaussians.set_inscreen_points_to_visible(tdgs_cam)
                            render_pkg = render(tdgs_cam, gaussians, opt, bg_color=torch.tensor([0.7, 0.7, 0.7], device='cuda'), render_visible=True, config=config)
                    else:
                        points_3d, colors, _, normals, imgs_train,cameras_train, focal_length_train, is_sky, all_depths, depth_align_masks, now_scale = kf_gen.process_video_frames(roots, pose_seq, gaussians, xyz_scale, opt, num = 49,total_num = len(pose_seq),return_1=False)
                else:
                    idxs = np.linspace(0, len(cameras)-1, num=3).astype(np.int32)
                    cameras_train = [cameras[idx] for idx in idxs]
                    for idx, pose in enumerate(cameras_train):
                        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)
                        gaussians.set_inscreen_points_to_visible(tdgs_cam)

                    # detect existing objects
                    obj_to_refresh, detected_camera_idx = None, -1
                    skip_next_scale_setting = False  # default: do not skip next_scale setting

                    if zoom_in and len(cameras_train) >= 2:
                        print("🔄 Checking for existing objects to refresh (zoom in mode)...")

                        # check if pose[1] and pose[2] fully see an object (when pose_seq length is 3)
                        if len(cameras_train) == 3:
                            check_poses = [cameras_train[1], cameras_train[2]]
                        else:
                            # if not exactly 3 poses, check the middle poses
                            mid_start = len(cameras_train) // 3
                            mid_end = len(cameras_train) * 2 // 3
                            check_poses = cameras_train[mid_start:mid_end+1]

                        obj_to_refresh, detected_camera_idx = get_existing_objects_to_refresh(gaussians, check_poses)


                    if not rewrite_background:
                        # mark as overwrite mode, record original point count
                        overwrite_mode = True
                        n_original_points = gaussians.get_xyz_all.shape[0]
                        print(f"🎯 Overwrite mode: {n_original_points} original points")

                        saved_object_img_path = None  # for saving high-resolution images with the object

                        if obj_to_refresh is not None:
                            detected_camera = check_poses[detected_camera_idx]
                            print(f"🔄 Found object to refresh: '{obj_to_refresh}' detected by camera {detected_camera_idx+1}")

                            # Step 1: first render the zoomin video with the object, save detected frame
                            print(f"🎬 Step 1: Rendering zoomin video WITH object to save high-quality frame...")

                            # save coz output files with correct shading

                            if detected_camera_idx == 0:
                                tar_img_path = "./cache/coz_output.png"
                            else:
                                tar_img_path = "./cache/coz_output2.png"
                            detected_img = load_image_and_resize(tar_img_path, height=config['orig_H'], width=config['orig_W'])
                            # save the high-resolution object image corresponding to detected_camera_idx
                            saved_object_img_path = f"./cache/object_frame_{obj_to_refresh}_detected.png"
                            detected_img_pil = ToPILImage()(detected_img.cpu())
                            detected_img_pil.save(saved_object_img_path)
                            print(f"   ✅ Saved high-quality object frame to: {saved_object_img_path}")

                            # Step 2: remove object points from gaussian
                            print(f"🗑️ Step 2: Removing object '{obj_to_refresh}' from gaussian...")
                            # gaussians.remove_points_by_label(obj_to_refresh)

                            now_scene_name = obj_to_refresh  # save object name for later use
                            skip_next_scale_setting = False  # follow original logic, do not skip next_scale setting
                        else:
                            print("ℹ️ No existing objects found in cameras[1], cameras[2] (no object ≥99.9% visible)")

                            # use dominant_ids to detect if zooming in to an object
                            skip_next_scale_setting = False
                            print("🔄 Checking object dominance in cameras[1], cameras[2] using dominant_ids...")

                            # check the proportion of object points in dominant_ids
                            if hasattr(gaussians, 'point_labels') and gaussians.point_labels.numel() > 0:
                                global GLOBAL_LABEL_NAMES

                                object_dominance_ratios = []  # store the ratio of each object in each camera

                                for i, pose in enumerate(check_poses):
                                    print(f"  Analyzing camera {i+1} dominant_ids...")
                                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)

                                    with torch.no_grad():
                                        render_pkg = render(tdgs_cam, gaussians, opt, background, render_dominant_ids=True, config=config)
                                        dominant_ids = render_pkg["dominant_ids"]  # [H, W]

                                    # get unique dominant point indices and total count
                                    unique_dominant_ids = dominant_ids.unique()
                                    total_dominant_pixels = dominant_ids.numel()

                                    print(f"    Total dominant pixels: {total_dominant_pixels}")
                                    print(f"    Unique dominant points: {len(unique_dominant_ids)}")

                                    camera_ratios = {}

                                    # compute the proportion of each object in dominant_ids
                                    for label_name in GLOBAL_LABEL_NAMES:
                                        if label_name in ["main", "background"]:
                                            continue

                                        object_mask = gaussians.get_label_mask(label_name)
                                        if not object_mask.any():
                                            continue

                                        # get indices of object points
                                        object_indices = torch.nonzero(object_mask, as_tuple=False).squeeze(-1)

                                        if len(object_indices) == 0:
                                            continue

                                        # count the number of pixels in dominant_ids belonging to this object
                                        object_dominant_mask = torch.isin(dominant_ids, object_indices)
                                        object_dominant_pixels = object_dominant_mask.sum().item()
                                        object_dominance_ratio = object_dominant_pixels / total_dominant_pixels

                                        camera_ratios[label_name] = object_dominance_ratio
                                        print(f"    Object '{label_name}': {object_dominant_pixels}/{total_dominant_pixels} pixels ({object_dominance_ratio:.1%})")

                                    object_dominance_ratios.append(camera_ratios)

                                # check if any object's ratio is growing or exceeds the threshold
                                for label_name in GLOBAL_LABEL_NAMES:
                                    if label_name in ["main", "background"]:
                                        continue

                                    # get this object's ratio across cameras
                                    ratios = [camera_ratios.get(label_name, 0.0) for camera_ratios in object_dominance_ratios]

                                    if len(ratios) >= 2 and max(ratios) > 0:
                                        print(f"  Object '{label_name}' dominance trend: {' → '.join([f'{r:.1%}' for r in ratios])}")

                                        # check if it exceeds the 70% threshold
                                        max_ratio = max(ratios)
                                        if max_ratio >= 0.7:
                                            print(f"    ✅ Object '{label_name}' dominance {max_ratio:.1%} ≥ 70% threshold!")
                                            print(f"    🚫 Will skip next_scale setting due to high object dominance")
                                            skip_next_scale_setting = True
                                            break

                                        # check if there is a significant growth trend
                                        if len(ratios) >= 2:
                                            ratio_increasing = ratios[-1] > ratios[0] * 1.5  # last one grew more than 50% over the first
                                            if ratio_increasing and max_ratio > 0.3:  # and max ratio exceeds 30%
                                                print(f"    ✅ Object '{label_name}' dominance increasing significantly!")
                                                print(f"    🚫 Will skip next_scale setting due to dominance growth trend")
                                                skip_next_scale_setting = True
                                                break

                                if skip_next_scale_setting:
                                    print(f"✅ Detected object zoom-in via dominant_ids analysis")
                                else:
                                    print(f"ℹ️ No significant object dominance detected")

                            # only check cameras[0] when no object zoom-in is detected
                            if not skip_next_scale_setting:
                                print("ℹ️ No object zoom-in detected via dominant_ids analysis")
                                print("🔄 Checking cameras[0] for existing objects...")
                                obj_to_refresh_cam0, detected_camera_idx_cam0 = get_existing_objects_to_refresh(gaussians, [cameras_train[0]])

                                if obj_to_refresh_cam0 is not None:
                                    print(f"✅ Found object '{obj_to_refresh_cam0}' in cameras[0]")
                                    print(f"🚫 Will skip next_scale setting due to object in cameras[0]")
                                    skip_next_scale_setting = True
                                else:
                                    print("ℹ️ No existing objects found in cameras[0] either")
                                    print("✅ Will proceed with next_scale setting")
                                    skip_next_scale_setting = False

                        # Step 3: generate zoomin frames without object, train with normal logic
                        print(f"🎬 Step 3: Processing zoomin frames WITHOUT object (normal training)...")
                        print(f'std {get_current_std(cameras[0])} use overwrite (simplified)')
                        if  obj_to_refresh is not None or skip_next_scale_setting:
                            os.makedirs("./cache/coz_with_object", exist_ok=True)
                            if os.path.exists("./cache/img_0.png"):
                                shutil.copy2("./cache/img_0.png", "./cache/coz_with_object/img_0.png")
                            if os.path.exists("./cache/coz_output.png"):
                                shutil.copy2("./cache/coz_output.png", "./cache/coz_with_object/coz_output.png")
                            if os.path.exists("./cache/coz_output2.png"):
                                shutil.copy2("./cache/coz_output2.png", "./cache/coz_with_object/coz_output2.png")
                            print(f"   ✅ Saved coz output files with correct shading to ./cache/coz_with_object/")
                            gaussians.merge_all_to_trainable()
                            gaussians_obj = copy.deepcopy(gaussians)
                            if obj_to_refresh is not None:
                                mask = gaussians_obj.get_label_mask(obj_to_refresh)
                            else:
                                mask = gaussians_obj.get_label_mask(label_name)
                            gaussians_obj.delete_all_points(~mask)
                            gaussians.delete_all_points(mask)
                            ## without object
                            render_zoomin_rough_video3(pose_seq, gaussians, editing_prompt=None)
                        points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, now_scale = kf_gen.process_zoomin_frames_overwrite(
                            roots, pose_seq, gaussians, xyz_scale, opt
                        ) 
                    else:
                        # mark as rewrite mode
                        overwrite_mode = False

                        print(f'std {get_current_std(cameras[0])} use rewrite')
                        points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, now_scale = kf_gen.process_zoomin_frames_rewrite(roots, pose_seq, gaussians, xyz_scale, opt)

                    # if there is an object that needs refresh, apply linear blending to imgs_train
                    # if obj_to_refresh is not None and len(imgs_train) >= 3:
                    #     print(f"🎨 Applying linear blending for object '{obj_to_refresh}' in imgs_train...")

                    #     # load previously saved coz output with correct shading
                    #     coz_output_path = "./cache/coz_with_object/coz_output.png"
                    #     coz_output2_path = "./cache/coz_with_object/coz_output2.png"

                    #     if os.path.exists(coz_output_path) and os.path.exists(coz_output2_path):
                    #         # load coz output images
                    #         coz_output = plt.imread(coz_output_path)[:, :, :3]  # [H, W, 3]
                    #         coz_output2 = plt.imread(coz_output2_path)[:, :, :3]  # [H, W, 3]

                    #         # convert to tensor format [3, H, W]
                    #         coz_output_tensor = torch.from_numpy(coz_output).permute(2, 0, 1).float()
                    #         coz_output2_tensor = torch.from_numpy(coz_output2).permute(2, 0, 1).float()

                    #         # use SAM to segment object
                    #         print(f"   🔍 Using SAM to segment object '{obj_to_refresh}'...")

                    #         # blend imgs_train[1]
                    #         foreground_mask1, combined_mask1, masks1 = grounded_sam.get_combined_foreground_mask("./cache/coz_with_object/coz_output.png", [obj_to_refresh], kernel_size=7)
                    #         mask1_tensor = torch.tensor(foreground_mask1).float()  # [H, W]

                    #         # linear blending: mask=0 uses coz_output, mask=1 uses original image
                    #         imgs_train[1] = mask1_tensor.unsqueeze(0) * imgs_train[1] + (1 - mask1_tensor.unsqueeze(0)) * coz_output_tensor
                    #         print(f"   ✅ Applied linear blending to imgs_train[1]")

                    #         # blend imgs_train[2]
                    #         foreground_mask2, combined_mask2, masks2 = grounded_sam.get_combined_foreground_mask("./cache/coz_with_object/coz_output2.png", [obj_to_refresh], kernel_size=7)
                    #         mask2_tensor = torch.tensor(foreground_mask2).float()  # [H, W]

                    #         # linear blending: mask=0 uses coz_output2, mask=1 uses original image
                    #         imgs_train[2] = mask2_tensor.unsqueeze(0) * imgs_train[2] + (1 - mask2_tensor.unsqueeze(0)) * coz_output2_tensor
                    #         print(f"   ✅ Applied linear blending to imgs_train[2]")
                    #     else:
                    #         print(f"   ⚠️ Warning: Saved coz output files not found, skipping linear blending")

                    # # update memory data (zoom_in case)
                    print(f"📚 Updating memory with {len(cameras_train)} new cameras and {len(imgs_train)} new images")

                gaussians_new = GaussianModel(sh_degree=0, config=config)   

                gaussians_new.floater_dist2_threshold = torch.inf
                train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)
                scene = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)
                trainCameras = scene.getTrainCameras().copy()

                compute_3D_filter(gaussians_new, cameras=trainCameras ,initialize_scaling=True)

                # no longer need complex object package processing, package should always be None
                # compute new point count (before merge)
                n_new_points = gaussians_new.get_xyz_all.shape[0] if not first_time_move else 0


                if zoom_in:
                # assign next_scale: for old points visible in cameras_train[1], set next_scale to s_target if it is inf
                    gaussians_new = setup_gaussian_scales_and_merge(gaussians, gaussians_new, cameras_train)
                else:
                    gaussians_new.merge_gaussian(gaussians)


                for idx, pose in enumerate(cameras_train):
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)
                    # gaussians_new.set_visible_and_restore_from_prev(tdgs_cam, opt)
                    gaussians_new.set_inscreen_points_to_visible(tdgs_cam)
                    # if zoom_in:
                    #     if idx >0:
                    #         # gaussians_new.compute_3D_filter(scene.getTrainCameras().copy(), initialize_scaling=False)
                    #         # gaussians_new.set_visible_and_restore_from_prev(tdgs_cam, opt)
                    #         # gaussians_new.training_setup(opt)
                    #         render_pkg = render(tdgs_cam, gaussians_new, opt, bg_color=torch.tensor([0.7, 0.7, 0.7], device='cuda'), render_visible=True)
                    #         visible_mask = render_pkg["visibility_filter"]
                    #         gaussians_new.delete_mask_all|=visible_mask
                    #         gaussians_new.delete_all_points(gaussians_new.delete_mask_all)

                if zoom_in:
                    polr_init = opt.position_lr_init
                    polr_final = opt.position_lr_final
                    opt.position_lr_init = 0.00
                    opt.position_lr_final = 0.00
                gaussians_new.training_setup(opt)
                gaussians = gaussians_new


                if not zoom_in:
                    if len(cameras_train_memory) > 0:
                        # The memory replay of the paper-era code was disabled. The selection is kept
                        # because it re-partitions the trainable points (set_trainable_mask), which
                        # changes the point order seen by the training below.
                        print("🔍 Selecting memory cameras for training...")
                        select_memory_for_training(gaussians, cameras_train, cameras_train_memory, imgs_train_memory,
                                                   visibility_threshold=0.3)
                    print(f"📝 zoom_in:{zoom_in} training with current data only")
                    gaussians.merge_all_to_trainable()
                    n_total = gaussians.get_xyz_all.shape[0]
                    trainable_mask = torch.zeros(n_total, dtype=torch.bool, device='cuda')
                    trainable_mask[:n_new_points] = True  # new points are at the front

                    # overwrite mode: only train new points
                    train_gaussian(gaussians, scene, opt, initialize_scaling=True, zoom_in=zoom_in, xyz_scale=xyz_scale, newly_added_points=n_new_points, trainable_mask=trainable_mask, no_loss_masks = [~mask.squeeze().bool() for mask in masks])
                else:
                    t1 = opt.iterations
                    t2 = opt.densify_from_iter
                    opt.iterations = 400
                    opt.densify_from_iter = 1200

                    # overwrite mode: only train new points
                    gaussians.merge_all_to_trainable()
                    n_total = gaussians.get_xyz_all.shape[0]
                    trainable_mask = torch.zeros(n_total, dtype=torch.bool, device='cuda')
                    trainable_mask[:n_new_points] = True  # new points are at the front
                    print(f"🎯 Overwrite mode: Only {n_new_points}/{n_total} new points are trainable")

                    train_gaussian(gaussians, scene, opt, initialize_scaling=True, zoom_in=zoom_in, xyz_scale=xyz_scale, newly_added_points=n_new_points, trainable_mask=trainable_mask)

                    opt.iterations = t1
                    opt.position_lr_init = polr_init
                    opt.position_lr_final = polr_final
                    opt.densify_from_iter = t2


                # Step 4: if a high-resolution object image was saved earlier, use it to add the object back
                if 'saved_object_img_path' in locals() and saved_object_img_path is not None and 'now_scene_name' in locals():
                    print(f"🔗 Step 4: Adding high-quality object back using saved image...")
                    print(f"   Saved image: {saved_object_img_path}")
                    print(f"   Object name: {now_scene_name}")
                    print(f"   Camera: detected_camera")

                    gaussians = add_object_to_image_with_image(detected_camera, now_scene_name, saved_object_img_path, gaussians)
                    u = gaussians.get_label_mask(now_scene_name)
                    gaussians.merge_all_to_trainable()
                    gaussians.prior_scale[u] = gaussians.now_scale[u]/50
                    print(f"✅ Step 4 complete: Object '{now_scene_name}' added back successfully!")
                    # gaussians = add_object_to_image_with_image(current_camera, "a small bee", saved_object_img_path, gaussians)
                elif zoom_in and skip_next_scale_setting and gaussians_obj is not None:
                    # gaussians_obj
                    gaussians = update_gaussian_obj(cameras_train, label_name, ["./cache/coz_with_object/img_0.png","./cache/coz_with_object/coz_output.png","./cache/coz_with_object/coz_output2.png"], gaussians_obj, gaussians)
                    # gaussians.merge_gaussian(gaussians_obj)


                elif job_scene_name is not None and zoom_in:
                    # Object insertion at the end of the zoom: the prompt in effect when it was accepted
                    if FEATURES['objects']:
                        gaussians = add_object_to_image(cameras_train[-1], job_scene_name)
                    else:
                        print(f"⚠️ Object insertion unavailable ({'; '.join(OBJECTS_MISSING)}): '{job_scene_name}' not inserted")

                return cameras_train, imgs_train


            if zoom_in:
                if len(trajectory_points) != 2:
                    raise RuntimeError(f"Zoom-in needs exactly 2 trajectory points (H once, then R); got {len(trajectory_points)}")
                # zoom in + trajectory: use trajectory directly
                print(f"Zoom in with trajectory: {len(trajectory_points)} points")

                # interpolate between trajectory points
                full_pose_seq = []
                for i in range(len(trajectory_points) - 1):
                    start_cam = trajectory_points[i]
                    end_cam = trajectory_points[i + 1]
                    segment_seq = interpolate_cameras_K(start_cam, end_cam, num_frames = 49, config=config)  # zoom uses K interpolation
                    full_pose_seq.extend(segment_seq[:-1])
                full_pose_seq.append(trajectory_points[-1])

                cameras_train, imgs_train = process_one_seq(full_pose_seq, zoom_in=True)

                # in zoom_in case, do not add trajectory to gen_matrices
                trajectory_points = []  # clear
            else:  # not zoom in
                if len(trajectory_points) < 1:
                    raise RuntimeError("Camera move needs at least 1 trajectory point")
                print(f"Camera movement with trajectory: {len(trajectory_points)} points")

                # find the nearest already generated point
                pose_closestid, _, _ = compute_pose_distances(trajectory_points[0], gen_matrices)
                pose_close = gen_matrices[pose_closestid]

                # build full trajectory: closest -> trajectory[0] -> trajectory[1] -> ...
                full_trajectory = [pose_close] + trajectory_points

                # compute normalized distances
                distances = compute_trajectory_distances(full_trajectory, use_focal=False)

                total_distance = sum(distances)
                total_frames = 121 if (svc is not None and svc.enabled('gen3c')) else 49

                # allocate frames proportionally by distance
                frame_counts = []
                num_alloc= total_frames - (1+len(distances))
                for dist in distances:
                    frames = int(round(dist / total_distance * num_alloc))
                    frame_counts.append(max(frames, 0))  # at least 2 frames

                # adjust total frame count (rounding may cause deviation)
                current_total = sum(frame_counts)
                if current_total != num_alloc:
                    # adjust the frame count of the longest segment
                    max_idx = frame_counts.index(max(frame_counts))
                    frame_counts[max_idx] += (num_alloc - current_total)

                print(f"Frame allocation: {frame_counts} (total: {sum(frame_counts)})")

                # generate the complete sequence
                full_pose_seq = []
                for i in range(len(full_trajectory) - 1):
                    start_cam = full_trajectory[i]
                    end_cam = full_trajectory[i + 1]
                    segment_seq = interpolate_cameras_RT(start_cam, end_cam, num_frames=frame_counts[i]+2, config=config)
                    full_pose_seq.extend(segment_seq[:-1])  # remove last frame to avoid duplication

                full_pose_seq.append(full_trajectory[-1])  # add the last point

                print(f"Generated trajectory with {len(full_pose_seq)} total frames")

                cameras_train, imgs_train = process_one_seq(full_pose_seq, first_time_move, zoom_in=False)
                gen_matrices.extend(trajectory_points)

                trajectory_points = []  # clear
            keep_rendering = True
            cameras_train = cameras_train * 8 if zoom_in else cameras_train
            imgs_train = imgs_train * 8 if zoom_in else imgs_train
            cameras_train_memory.extend(cameras_train)
            imgs_train_memory.extend(imgs_train)
            if zoom_in and scene_name == job_scene_name:
                # This zoom-in used the prompt; camera moves keep it, and a prompt sent while the
                # zoom-in ran waits for the next one.
                scene_name = None
            job_scene_name = None
            coz_request_seed = None

            empty_cache()
            scene_snapshot, job_snapshot = job_snapshot, None
            end_job(job_name, t_job)
            scene_lock.release()
        except Exception as e:
            # Per-request error handling: report, restore the pre-job scene and keep serving.
            failed_job = busy_job or server_status.get('job')
            print(f"❌ {failed_job or 'request'} failed: {e}")
            traceback.print_exc()
            if ARGS.debug:
                debug_post_mortem(e.__traceback__)
            if job_snapshot is not None:
                with scene_lock:
                    restore_scene_snapshot(job_snapshot)
                job_snapshot = None
            scene_lock.release_all()  # the failed job may still hold it
            trajectory_points = []
            job_scene_name = None  # the object prompt stays in scene_name for the next zoom-in
            reset_orbit_state()
            complete_background_pose = None
            coz_request_seed = None
            delete = False
            save = False
            undo = False
            keep_rendering = True
            fail_job(failed_job, e)
            if failed_job == 'zoom':
                reset_main_models_after_failure()  # MoGe may be left half fine-tuned
            empty_cache()
            continue


def check_object_visibility_in_poses(gaussians, pose_list, object_label, visibility_threshold=0.999):
    """
    Check the visibility of a specified object in the given pose list, return the camera index where the object is detected.

    Args:
        gaussians: GaussianModel
        pose_list: list of camera poses
        object_label: the object label name to check
        visibility_threshold: visibility threshold, values above this are considered visible

    Returns:
        int: camera index where object is detected; if detected by multiple cameras, return the largest index; return -1 if not detected
    """
    global opt, background, xyz_scale

    if not hasattr(gaussians, 'point_labels') or gaussians.point_labels.numel() == 0:

        return -1

    # get the point mask for the object
    object_mask = gaussians.get_label_mask(object_label)
    if not object_mask.any():
        return -1

    total_object_points = object_mask.sum().item()
    detected_cameras = []

    print(f"🔍 Checking visibility of '{object_label}' ({total_object_points} points) in {len(pose_list)} poses...")

    for i, pose in enumerate(pose_list):
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(pose, xyz_scale=xyz_scale, config=config)

        with torch.no_grad():
            render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
            visibility_filter = render_pkg["visibility_filter"]

        # compute visibility of object points under the current pose
        visible_object_points = (object_mask & visibility_filter).sum().item()
        visibility_ratio = visible_object_points / total_object_points

        print(f"  Pose {i+1}/{len(pose_list)}: {visible_object_points}/{total_object_points} object points visible ({visibility_ratio:.2%})")

        if visibility_ratio >= visibility_threshold:
            print(f"  ✅ Object detected in pose {i+1} ({visibility_ratio:.2%} >= {visibility_threshold:.2%})")
            detected_cameras.append(i)

    if detected_cameras:
        # return the largest camera index (prefer later cameras)
        selected_camera = max(detected_cameras)
        print(f"  🎯 Selected camera {selected_camera+1} (from detected cameras: {[c+1 for c in detected_cameras]})")
        return selected_camera
    else:
        print(f"  ❌ Object '{object_label}' not detected in any pose")
        return -1

def get_existing_objects_to_refresh(gaussians, pose_list):
    """
    Get existing objects that need refreshing (non-main, non-background objects detected in pose_list).

    Args:
        gaussians: GaussianModel
        pose_list: list of camera poses

    Returns:
        tuple: (object_label, camera_index) if an object is detected, otherwise (None, -1)
               It is guaranteed that at most one object is detected at a time.
    """
    global GLOBAL_LABEL_NAMES

    if not hasattr(gaussians, 'point_labels') or gaussians.point_labels.numel() == 0:
        return (None, -1)

    for label_name in GLOBAL_LABEL_NAMES:
        # skip main and background
        if label_name in ["main", "background"]:
            continue

        # check if points with this label exist
        label_mask = gaussians.get_label_mask(label_name)
        if not label_mask.any():
            continue

        # check if detected in pose_list
        detected_camera_idx = check_object_visibility_in_poses(gaussians, pose_list, label_name)
        if detected_camera_idx >= 0:
            # found an object; since at most one object is detected at a time, return directly
            return (label_name, detected_camera_idx)

    return (None, -1)


@uses_main_models  # GroundedSAM / inpainting / harmonization run on the main GPU
def add_object_to_image( current_camera, scene_name, gaussians_input=None, re_run_step1x = True, use_harmol = None):
    """
    Complete pipeline function for adding an object to an image.

    Args:
        image_path (str): input image path
        scene_name (str): name of the object to add
        current_camera: current camera parameters; if None, infer from the image
        gaussians_input: input Gaussian model; if None, use the global gaussians

    Returns:
        GaussianModel: merged Gaussian model
    """
    global kf_gen, opt, xyz_scale, gaussians, gaussians_obj, config
    if use_harmol is None:
        use_harmol = config.get('use_harmol', True)

    print(f"🎯 Starting object addition pipeline for '{scene_name}'...")

    # use the input gaussians or global gaussians
    gaussians_to_use = gaussians_input if gaussians_input is not None else gaussians

    # ensure cache directory exists

    os.makedirs("./cache", exist_ok=True)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
    median_depth = render_pkg['median_depth'][0]/xyz_scale
    image = render_pkg['render']
    plt.imsave("./cache/current_image.png", image.detach().permute(1,2,0).cpu().numpy(),cmap = "gray" )
    image_path = "./cache/current_image.png"

    # Step 1: generate edit prompt
    print("🤖 Step 1: Generating edit prompt with GPT...")
    prompt_edit = _gpt4().generate_edit_prompt(image_path, None, scene_name=scene_name, short_length=2)
    print(f"   Generated prompt: {prompt_edit}")

    # Step 2: edit image with Step1X-Edit
    print("🎨 Step 2: Editing image with Step1X-Edit...")
    if re_run_step1x:
        img_edit_path = call_step1x_edit(image_path, prompt_edit, output_path="./cache/step1x_output1.png")
    else:
        img_edit_path = "./cache/step1x_output1.png"
    if img_edit_path is None:
        raise RuntimeError("Step1X-Edit failed to generate edited image")
    print(f"   Edited image saved to: {img_edit_path}")


    # Step 3: segment object with GroundedSAM
    print("🎯 Step 3: Segmenting object with GroundedSAM...")
    mask_result = get_grounded_sam().segment_and_visualize(img_edit_path, scene_name)
    if mask_result['masks'] is None or len(mask_result['masks']) == 0:
        raise RuntimeError(f"No mask found for object '{scene_name}'")

    mask = mask_result['masks'][0].squeeze()
    plt.imsave("./cache/mask.png", (mask.cpu().numpy()*255).astype(np.uint8),cmap = "gray" )
    print(f"   Object mask saved to: ./cache/mask.png")
    # INR-Harmonization of the edited image (config use_harmol); without it the edit is used as it is.
    harmonizer = get_harmo_model() if use_harmol else None
    if harmonizer is not None:
        harmonizer.inference('./cache/step1x_output1.png', './cache/mask.png', './cache/step1x_output.png')
    else:
        os.replace('./cache/step1x_output1.png', './cache/step1x_output.png')
    # Step 4: get camera parameters and depth information

    if current_camera is None:
        # if no camera is provided, use default camera or infer from image
        print("📷 Step 4: Using default camera parameters...")
        # camera inference from image can be implemented here as needed
        raise NotImplementedError("Camera inference from image not implemented yet")
    else:
        print("📷 Step 4: Using provided camera parameters...")
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
        render_pkg = render(tdgs_cam, gaussians_to_use, opt, background, render_visible=True, config=config)
        median_depth = render_pkg['median_depth'][0]/xyz_scale

    # Step 5: process edited image to generate 3D point cloud
    print("📊 Step 5: Processing edited image for 3D point cloud...")

    points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, now_scale = kf_gen.process_single_img_mask(
        ['./cache/step1x_output.png'], mask, [current_camera], median_depth, 1, 1
    )
    print(focal_length_train)
    # Step 6: create new Gaussian model and train
    print("🏗️ Step 6: Creating and training new Gaussian model...")

    # record original point count
    n_original = gaussians_to_use.get_xyz_all.shape[0]
    print(f"   Original gaussian points: {n_original}")

    # create new gaussian model
    t1 = opt.iterations
    t2 = opt.position_lr_init
    t3 = opt.position_lr_final
    opt.iterations = 200
    opt.position_lr_init = 0.00
    opt.position_lr_final = 0.000
    gaussians_new = GaussianModel(sh_degree=0, floater_dist2_threshold=torch.inf, config=config)

    train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)
    scene_new = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)
    gaussians_obj = copy.deepcopy(gaussians_new)
    n_new = gaussians_new.get_xyz_all.shape[0]
    print(f"   New gaussian points: {n_new}")

    trainCameras = scene_new.getTrainCameras().copy()
    compute_3D_filter(gaussians_new, cameras=trainCameras, initialize_scaling=True)
    background_image = get_pure_background("./cache/step1x_output.png", foreground_word = scene_name, mask =mask)

    # Step 6.5: first train gaussian with pure background image to adapt to background
    print("🎨 Step 6.5: First training phase - adapting gaussian to pure background...")
    print("   Training with background image using color_only mode")

    # save background image for training
    if hasattr(background_image, 'save'):
        # If it's a PIL Image, save directly
        background_image.save("./cache/pure_background.png")
    else:
        plt.imsave("./cache/pure_background.png", (background_image.detach().cpu().numpy()*255.).astype(np.uint8),cmap = "gray" )

    # first let gaussians_to_use adapt to the pure background
    # all points participate in training, with background image as target
    gaussians_to_use.merge_all_to_trainable()  # ensure all points are trainable

    # backup original training parameters
    original_iterations = opt.iterations
    original_feature_lr = opt.feature_lr

    # set background adaptation training parameters
    opt.iterations = 300  # moderate iteration count
    opt.feature_lr = 0.005  # set appropriate feature learning rate
    gaussians_to_use.training_setup(opt)
    print(f"   🎨 Set feature_lr: {original_feature_lr} -> {opt.feature_lr}")
    print(f"   🎨 Set iterations: {original_iterations} -> {opt.iterations}")

    # train gaussian to adapt to background
    print(f"   Training gaussian with pure background for {opt.iterations} iterations...")

    # convert background_image to the same format as training images
    if background_image.dim() == 3:  # [H, W, C]
        background_gt = background_image.permute(2, 0, 1)  # [C, H, W]
    else:  # already [C, H, W]
        background_gt = background_image

    # ensure on the correct device
    background_gt = background_gt.to('cuda')

    # simply replace images in scene_new
    original_images = []
    for cam in scene_new.train_cameras:
        print("cam.original_image.shape", cam.original_image.shape)
        original_images.append(cam.original_image.clone())
        cam.original_image = background_gt

    try:
        # train with color_only mode to let gaussian learn background colors
        train_gaussian(
            gaussians_to_use, scene_new, opt,
            initialize_scaling=False, zoom_in=True, no_loss_masks=None,  # no mask, train on full image
            xyz_scale=xyz_scale, newly_added_points=0,  # not new point training
            hq_mode=False,  # normal mode
            color_only=True  # only train colors, freeze geometry parameters
        )
    finally:
        # restore original images after training
        for i, cam in enumerate(scene_new.train_cameras):
            cam.original_image = original_images[i]

    # restore original training parameters
    opt.iterations = original_iterations
    opt.feature_lr = original_feature_lr
    print("✅ Background adaptation complete!")

    # Step 7: set labels for new points and update global label list
    print("🏷️ Step 7: Setting label for new points...")

    # automatically update GLOBAL_LABEL_NAMES
    global GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP
    if scene_name not in GLOBAL_LABEL_MAP:
        label_id = len(GLOBAL_LABEL_NAMES)
        GLOBAL_LABEL_NAMES.append(scene_name)
        GLOBAL_LABEL_MAP[scene_name] = label_id
        print(f"   ✅ Added new label '{scene_name}' to GLOBAL_LABEL_NAMES (ID: {label_id})")
    else:
        print(f"   ℹ️ Label '{scene_name}' already exists in GLOBAL_LABEL_NAMES")

    n_new_points = gaussians_new.get_xyz_all.shape[0]
    all_new_points_mask = torch.ones(n_new_points, dtype=torch.bool, device='cuda')
    gaussians_new.set_points_label(all_new_points_mask, scene_name, GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP)
    print(f"   Set {n_new_points} new points with label '{scene_name}'")
    gaussians_new.prior_scale = now_scale / 48.
    # Step 8: smart merge - only train newly added object
    print("🧠 Step 8: Smart merging with trainability control...")
    gaussians_new.merge_gaussian_with_trainability_control(
        gaussians_to_use, 
        auto_trainable_labels=[scene_name]  # only new object is trainable
    )
    gaussians_merged = gaussians_new
    print(f"   Merged gaussian points: {gaussians_merged.get_xyz_all.shape[0]}")

    # Step 9: reset training parameters
    print("⚙️ Step 9: Setting up training parameters...")
    gaussians_merged.training_setup(opt)

    # Step 10: train the merged model (only train newly added points)
    print("🚀 Step 10: Training merged model (new points only)...")

    print(f"   Training for {opt.iterations} iterations...")

    # trainability has already been set during smart merge

    # compute the number of newly added points
    n_total_trainable = gaussians_merged.get_xyz.shape[0]
    newly_added_points = n_total_trainable  # now only new points are trainable
    print(f"   Newly added trainable points: {newly_added_points}")

    # train the model
    gaussians = gaussians_merged
    train_gaussian(
        gaussians, scene_new, opt, 
        initialize_scaling=True, zoom_in=True, no_loss_masks=[~mask],
        xyz_scale=xyz_scale, newly_added_points=newly_added_points, 
        hq_mode=True
    )
    # import pdb; pdb.set_trace()
    opt.iterations = t1
    opt.position_lr_init = t2
    opt.position_lr_final = t3

    # Step 10.5: second training phase - let background adapt to new object (for shadow and interaction effects)


    # all points participate in training, but object region does not participate in loss computation
    gaussians.merge_all_to_trainable()  # ensure all points are trainable

    # use fewer iterations for fine-tuning


    # restore original settings
    opt.iterations = t1
    print("✅ Second training phase complete - background adapted to new object")

    # Step 11: freeze newly added points after training
    # print("🧊 Step 11: Freezing newly added points after training...")
    # gaussians.freeze_labels(scene_name)

    print(f"✅ Object addition complete!")
    print(f"   Final gaussian points: {gaussians.get_xyz_all.shape[0]}")
    print(f"   Object '{scene_name}' is now frozen (non-trainable)")
    print(f"   Use gaussians.train_only_labels('{scene_name}') to make it trainable again")

    return gaussians


@uses_main_models  # GroundedSAM / inpainting / harmonization run on the main GPU
def add_object_to_image_with_image(current_camera, scene_name, img_edit_path , gaussians_input=None, harmo = False,):
    """
    Complete pipeline function for adding an object to an image.

    Args:
        image_path (str): input image path
        scene_name (str): name of the object to add
        current_camera: current camera parameters; if None, infer from the image
        gaussians_input: input Gaussian model; if None, use the global gaussians

    Returns:
        GaussianModel: merged Gaussian model
    """
    global kf_gen, opt, xyz_scale, gaussians, config

    print(f"🎯 Starting object addition pipeline for '{scene_name}'...")

    # use the input gaussians or global gaussians
    gaussians_to_use = gaussians_input if gaussians_input is not None else gaussians


    # Step 3: segment object with GroundedSAM
    print("🎯 Step 3: Segmenting object with GroundedSAM...")
    mask_result = get_grounded_sam().segment_and_visualize(img_edit_path, scene_name)
    if mask_result['masks'] is None or len(mask_result['masks']) == 0:
        raise RuntimeError(f"No mask found for object '{scene_name}'")

    mask = mask_result['masks'][0].squeeze()
    plt.imsave("./cache/mask.png", (mask.cpu().numpy()*255).astype(np.uint8), cmap = "gray" )
    print(f"   Object mask saved to: ./cache/mask.png")

    # Step 4: get camera parameters and depth information
    if current_camera is None:
        # if no camera is provided, use default camera or infer from image
        print("📷 Step 4: Using default camera parameters...")
        # camera inference from image can be implemented here as needed
        raise NotImplementedError("Camera inference from image not implemented yet")
    else:
        print("📷 Step 4: Using provided camera parameters...")
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
        render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
        median_depth = render_pkg['median_depth'][0]/xyz_scale
        plt.imsave("./cache/median_depth.png", (median_depth.squeeze().detach().cpu().numpy()),cmap = "gray" )

    # Step 5: process edited image to generate 3D point cloud
    print("📊 Step 5: Processing edited image for 3D point cloud...")
    harmonizer = get_harmo_model() if harmo else None
    if harmonizer is not None:
        harmonizer.inference(img_edit_path, './cache/mask.png', img_edit_path)

    points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, now_scale = kf_gen.process_single_img_mask(
        [img_edit_path], mask, [current_camera], median_depth, 1, 1
    )
    # points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky,_,_, now_scale = kf_gen.process_single_img(
    #     [img_edit_path], 1, 1 )
    print(focal_length_train)
    # Step 6: create new Gaussian model and train
    print("🏗️ Step 6: Creating and training new Gaussian model...")

    # record original point count
    n_original = gaussians_to_use.get_xyz_all.shape[0]
    print(f"   Original gaussian points: {n_original}")

    # create new gaussian model
    gaussians_new = GaussianModel(sh_degree=0, floater_dist2_threshold=torch.inf, config=config)

    t1 = opt.iterations
    t2 = opt.position_lr_init
    t3 = opt.position_lr_final
    opt.iterations = 100
    opt.position_lr_init = 0.00
    opt.position_lr_final = 0.000   
    train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)

    scene_new = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)

    n_new = gaussians_new.get_xyz_all.shape[0]
    print(f"   New gaussian points: {n_new}")

    trainCameras = scene_new.getTrainCameras().copy()
    compute_3D_filter(gaussians_new, cameras=trainCameras, initialize_scaling=True)
    # Step 7: set labels for new points and update global label list
    print("🏷️ Step 7: Setting label for new points...")

    # automatically update GLOBAL_LABEL_NAMES
    global GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP
    if scene_name not in GLOBAL_LABEL_MAP:
        label_id = len(GLOBAL_LABEL_NAMES)
        GLOBAL_LABEL_NAMES.append(scene_name)
        GLOBAL_LABEL_MAP[scene_name] = label_id
        print(f"   ✅ Added new label '{scene_name}' to GLOBAL_LABEL_NAMES (ID: {label_id})")
    else:
        print(f"   ℹ️ Label '{scene_name}' already exists in GLOBAL_LABEL_NAMES")

    n_new_points = gaussians_new.get_xyz_all.shape[0]
    all_new_points_mask = torch.ones(n_new_points, dtype=torch.bool, device='cuda')
    gaussians_new.set_points_label(all_new_points_mask, scene_name, GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP)
    print(f"   Set {n_new_points} new points with label '{scene_name}'")

    # Step 8: smart merge - only train newly added object
    print("🧠 Step 8: Smart merging with trainability control...")
    # import pdb; pdb.set_trace()
    gaussians_new.merge_gaussian_with_trainability_control(
        gaussians_to_use, 
        auto_trainable_labels=[scene_name]  # only new object is trainable
    )
    gaussians = gaussians_new
    gaussians_merged = gaussians_new
    print(f"   Merged gaussian points: {gaussians_merged.get_xyz_all.shape[0]}")

    # Step 9: reset training parameters
    print("⚙️ Step 9: Setting up training parameters...")
    gaussians_merged.training_setup(opt)

    # Step 10: train the merged model (only train newly added points)
    print("🚀 Step 10: Training merged model (new points only)...")

    print(f"   Training for {opt.iterations} iterations...")

    # trainability has already been set during smart merge

    # compute the number of newly added points
    n_total_trainable = gaussians_merged.get_xyz.shape[0]
    newly_added_points = n_total_trainable  # now only new points are trainable
    print(f"   Newly added trainable points: {newly_added_points}")

    # train the model
    gaussians = gaussians_merged
    train_gaussian(
        gaussians, scene_new, opt, 
        initialize_scaling=True, zoom_in=True, no_loss_masks=[~mask],
        xyz_scale=xyz_scale, newly_added_points=newly_added_points, 
        hq_mode=True
    )
    opt.iterations = t1
    opt.position_lr_init = t2 
    opt.position_lr_final = t3

    # Step 10.5: second training phase - let background adapt to new object (for shadow and interaction effects)
    print("🎯 Step 10.5: Second training phase - adapting background to new object...")
    print("   Training all points with object mask as no_loss_mask to create shadows/reflections")

    # all points participate in training, but object region does not participate in loss computation
    gaussians.merge_all_to_trainable()  # ensure all points are trainable

    # use fewer iterations for fine-tuning
    # opt.iterations = 200  # use 1/4 of original iteration count
    # print(f"   Using {opt.iterations} iterations for background adaptation")

    # train_gaussian(
    #     gaussians, scene_new, opt,
    #     initialize_scaling=False, zoom_in=True, no_loss_masks=[mask],  # note: this is mask, not ~mask
    #     xyz_scale=xyz_scale, newly_added_points=0,  # not new point training
    #     hq_mode=False,  # normal mode
    #     color_only=True  # only train colors, freeze geometry parameters
    # )

    # restore original settings
    opt.iterations = t1
    print("✅ Second training phase complete - background adapted to new object")

    # Step 11: freeze newly added points after training
    # print("🧊 Step 11: Freezing newly added points after training...")
    # gaussians.freeze_labels(scene_name)

    print(f"✅ Object addition complete!")
    print(f"   Final gaussian points: {gaussians.get_xyz_all.shape[0]}")
    print(f"   Object '{scene_name}' is now frozen (non-trainable)")
    print(f"   Use gaussians.train_only_labels('{scene_name}') to make it trainable again")

    return gaussians


@uses_main_models  # GroundedSAM / inpainting / harmonization run on the main GPU
def update_gaussian_obj(pose_seq, scene_name, roots, gaussians_obj, gaussians_original):
    """
    Complete pipeline function for adding an object to an image.

    Args:
        image_path (str): input image path
        scene_name (str): name of the object to add
        current_camera: current camera parameters; if None, infer from the image
        gaussians_input: input Gaussian model; if None, use the global gaussians

    Returns:
        GaussianModel: merged Gaussian model
    """
    global kf_gen, opt, xyz_scale, background, config

    print(f"🎯 Starting object addition pipeline for '{scene_name}'...")

    # use the input gaussians or global gaussians
    gaussians_to_use = copy.deepcopy(gaussians_obj)
    cameras_train = pose_seq
    masks = []
    for idx, camera in enumerate(cameras_train):
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera, xyz_scale=xyz_scale, config=config)
        render_pkg = render(tdgs_cam, gaussians_to_use, opt, background, render_visible=True, config=config)
        mask = render_pkg["final_opacity"] > 0.6
        plt.imsave(f"./cache/loss_mask_{idx}.png", mask.detach().squeeze().cpu(),cmap = "gray" )
        masks.append(mask)

    points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky, now_scale = kf_gen.process_zoomin_frames_overwrite_obj_mask(
        roots, pose_seq, gaussians_to_use, xyz_scale, opt, masks
    )
    # points_3d, colors, _, normals, imgs_train, cameras_train, focal_length_train, is_sky,_,_, now_scale = kf_gen.process_single_img(
    #     [img_edit_path], 1, 1 )
    print(focal_length_train)
    # Step 6: create new Gaussian model and train
    print("🏗️ Step 6: Creating and training new Gaussian model...")

    # record original point count
    n_original = gaussians_to_use.get_xyz_all.shape[0]
    print(f"   Original gaussian points: {n_original}")

    # create new gaussian model
    gaussians_new = GaussianModel(sh_degree=0, floater_dist2_threshold=torch.inf, config=config)

    t1 = opt.iterations
    t2 = opt.position_lr_init
    t3 = opt.position_lr_final
    t4 = opt.densify_from_iter
    opt.iterations = 400
    opt.position_lr_init = 0.00
    opt.position_lr_final = 0.000   
    opt.densify_from_iter = 1000
    train_data = kf_gen.convert_to_3dgs_traindata(points_3d, colors, normals, imgs_train, cameras_train, xyz_scale)

    scene_new = Scene(train_data, gaussians_new, opt, focal_length_train, is_sky, now_scale)

    n_new = gaussians_new.get_xyz_all.shape[0]
    print(f"   New gaussian points: {n_new}")

    trainCameras = scene_new.getTrainCameras().copy()
    compute_3D_filter(gaussians_new, cameras=trainCameras, initialize_scaling=True)
    # Step 7: set labels for new points and update global label list
    print("🏷️ Step 7: Setting label for new points...")

    # automatically update GLOBAL_LABEL_NAMES
    global GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP
    if scene_name not in GLOBAL_LABEL_MAP:
        label_id = len(GLOBAL_LABEL_NAMES)
        GLOBAL_LABEL_NAMES.append(scene_name)
        GLOBAL_LABEL_MAP[scene_name] = label_id
        print(f"   ✅ Added new label '{scene_name}' to GLOBAL_LABEL_NAMES (ID: {label_id})")
    else:
        print(f"   ℹ️ Label '{scene_name}' already exists in GLOBAL_LABEL_NAMES")

    n_new_points = gaussians_new.get_xyz_all.shape[0]
    all_new_points_mask = torch.ones(n_new_points, dtype=torch.bool, device='cuda')
    gaussians_new.set_points_label(all_new_points_mask, scene_name, GLOBAL_LABEL_NAMES, GLOBAL_LABEL_MAP)
    print(f"   Set {n_new_points} new points with label '{scene_name}'")

    # Step 8: smart merge - only train newly added object
    print("🧠 Step 8: Smart merging with trainability control...")
    # import pdb; pdb.set_trace()
    # gaussians_new.merge_gaussian(gaussians_to_use)
    gaussians_new = setup_gaussian_scales_and_merge(gaussians_to_use, gaussians_new, cameras_train)

    gaussians = gaussians_new
    gaussians_merged = gaussians_new
    print(f"   Merged gaussian points: {gaussians_merged.get_xyz_all.shape[0]}")

    # Step 9: reset training parameters
    print("⚙️ Step 9: Setting up training parameters...")


    # Step 10: train the merged model (only train newly added points)
    print("🚀 Step 10: Training merged model (new points only)...")

    print(f"   Training for {opt.iterations} iterations...")

    # trainability has already been set during smart merge

    # compute the number of newly added points
    n_total_trainable = gaussians_merged.get_xyz.shape[0]
    newly_added_points = n_total_trainable  # now only new points are trainable
    print(f"   Newly added trainable points: {newly_added_points}")

    # train the model
    gaussians = gaussians_merged
    gaussians.merge_gaussian(gaussians_original)
    gaussians_merged.training_setup(opt)
    train_gaussian(
        gaussians, scene_new, opt, 
        initialize_scaling=True, zoom_in=True, no_loss_masks=None, #[~mask for mask in masks],
        xyz_scale=xyz_scale, newly_added_points=newly_added_points, 
        hq_mode=True )
    opt.iterations = t1
    opt.position_lr_init = t2 
    opt.position_lr_final = t3
    opt.densify_from_iter = t4
    # all points participate in training, but object region does not participate in loss computation
    gaussians.merge_all_to_trainable()  # ensure all points are trainable


    # restore original settings
    opt.iterations = t1
    print("✅ Second training phase complete - background adapted to new object")

    print(f"✅ Object addition complete!")
    print(f"   Final gaussian points: {gaussians.get_xyz_all.shape[0]}")
    print(f"   Object '{scene_name}' is now frozen (non-trainable)")
    print(f"   Use gaussians.train_only_labels('{scene_name}') to make it trainable again")

    return gaussians


def setup_gaussian_scales_and_merge(gaussians, gaussians_new, cameras_train,
                                    skip_next_scale_setting=False):
    """
    Set prior_scale and next_scale for Gaussian points, then merge old and new Gaussians.

    Args:
        gaussians: the existing Gaussian object
        gaussians_new: the new Gaussian object (contains points with two different focal lengths)
        cameras_train: training camera list, requires at least 2 cameras, preferably 3
        skip_next_scale_setting: whether to skip next_scale setting, default False

    Returns:
        gaussians: merged Gaussian object (with scales set)
    """
    global opt, xyz_scale, convert_pt3d_cam_to_3dgs_cam, compute_inv_target_scale_per_frame
    print("🔄 Updating next_scale for visible old points...")

    # 🔄 Setting prior_scale for gaussians_new points based on focal length
    print("🔄 Setting prior_scale for gaussians_new points...")
    focals = gaussians_new.get_focal_length_all.unique()
    focal_1_mask = gaussians_new.get_focal_length_all == focals[0]
    focal_2_mask = gaussians_new.get_focal_length_all == focals[1]

    # Set prior scale for focal_1 points using cameras_train[0]
    if focal_1_mask.any():
        ref_camera = cameras_train[0]
        ref_camera_3dgs = convert_pt3d_cam_to_3dgs_cam(ref_camera, config=config)

        # Get focal_1 points data
        new_xyz = gaussians_new.get_xyz_all[focal_1_mask]
        new_rotations = gaussians_new.get_rotation_all[focal_1_mask]
        new_scales = gaussians_new.get_scaling_all[focal_1_mask]

        # Get visibility for focal_1 points
        with torch.no_grad():
            visible_filter = gaussians_new.get_inscreen_points(ref_camera_3dgs)[focal_1_mask]

        print(f"   Focal_1: Visible points: {visible_filter.sum()}/{len(visible_filter)}")

        # Compute s_target for visible focal_1 points
        if visible_filter.sum() > 0:
            q, s_target = compute_inv_target_scale_per_frame(
                ref_camera,
                new_xyz[visible_filter],
                config=config
            )

            # find points that need updating: visible and prior_scale is still -inf
            old_prior_scale = gaussians_new.get_prior_scale_all[focal_1_mask]
            visible_inf_mask = torch.isinf(old_prior_scale[visible_filter]) & (old_prior_scale[visible_filter] < 0)
            if visible_inf_mask.any():
                # Update prior_scale for focal_1 points
                scale_visible = gaussians_new.prior_scale[focal_1_mask]
                scale_visible[visible_filter] = torch.where(visible_inf_mask, s_target, scale_visible[visible_filter])
                gaussians_new.prior_scale[focal_1_mask] = scale_visible
                print(f"✅ Updated prior_scale for {visible_inf_mask.sum()} focal_1 points (from -inf to s_target)")
                print(f"   s_target range: {s_target[visible_inf_mask].min():.6f} - {s_target[visible_inf_mask].max():.6f}")
            else:
                print("ℹ️ No visible focal_1 points need prior_scale update (no visible -inf points)")

        # 🆕 Setting next_scale for focal_1 points using cameras_train[2]
        if len(cameras_train) > 2:
            ref_camera = cameras_train[2]
            ref_camera_3dgs = convert_pt3d_cam_to_3dgs_cam(ref_camera, config=config)

            # Get visibility for focal_1 points with cameras_train[2]
            with torch.no_grad():
                visible_filter = gaussians_new.get_inscreen_points(ref_camera_3dgs)[focal_1_mask]

            print(f"   Focal_1 with cameras_train[2]: Visible points: {visible_filter.sum()}/{len(visible_filter)}")

            # Compute s_target for visible focal_1 points using cameras_train[2]
            if visible_filter.sum() > 0:
                q, s_target = compute_inv_target_scale_per_frame(
                    ref_camera,
                    new_xyz[visible_filter],
                    config=config
                )

                # Set next_scale for focal_1 points
                scale_visible = gaussians_new.next_scale[focal_1_mask]
                scale_visible[visible_filter] = s_target
                point_labels = gaussians_new.point_labels[focal_1_mask]
                point_labels[visible_filter & (point_labels == 0)] = int(1e5)
                gaussians_new.point_labels[focal_1_mask] = point_labels
                gaussians_new.next_scale[focal_1_mask] = scale_visible
                print(f"✅ Set next_scale for {visible_filter.sum()} focal_1 points using cameras_train[2]")
                print(f"   s_target range: {s_target.min():.6f} - {s_target.max():.6f}")
        else:
            print("⚠️ Not enough cameras for focal_1 next_scale setting (need cameras_train[2])")

    # Set prior scale for focal_2 points using cameras_train[1]
    if focal_2_mask.any():
        ref_camera = cameras_train[1]
        ref_camera_3dgs = convert_pt3d_cam_to_3dgs_cam(ref_camera, config=config)

        # Get focal_2 points data
        new_xyz = gaussians_new.get_xyz_all[focal_2_mask]
        new_rotations = gaussians_new.get_rotation_all[focal_2_mask]
        new_scales = gaussians_new.get_scaling_all[focal_2_mask]

        # Get visibility for focal_2 points
        with torch.no_grad():
            visible_filter = gaussians_new.get_inscreen_points(ref_camera_3dgs)[focal_2_mask]

        print(f"   Focal_2: Visible points: {visible_filter.sum()}/{len(visible_filter)}")

        # Compute s_target for visible focal_2 points
        if visible_filter.sum() > 0:
            q, s_target = compute_inv_target_scale_per_frame(
                ref_camera,
                new_xyz[visible_filter],
                config=config
            )

            # find points that need updating: visible and prior_scale is still -inf
            old_prior_scale = gaussians_new.get_prior_scale_all[focal_2_mask]
            visible_inf_mask = torch.isinf(old_prior_scale[visible_filter]) & (old_prior_scale[visible_filter] < 0)
            if visible_inf_mask.any():
                # Update prior_scale for focal_2 points
                scale_visible = gaussians_new.prior_scale[focal_2_mask]
                scale_visible[visible_filter] = torch.where(visible_inf_mask, s_target, scale_visible[visible_filter])
                gaussians_new.prior_scale[focal_2_mask] = scale_visible
                print(f"✅ Updated prior_scale for {visible_inf_mask.sum()} focal_2 points (from -inf to s_target)")
                print(f"   s_target range: {s_target[visible_inf_mask].min():.6f} - {s_target[visible_inf_mask].max():.6f}")
            else:
                print("ℹ️ No visible focal_2 points need prior_scale update (no visible -inf points)")

    # use cameras_train[1] as reference camera - only execute when skip_next_scale_setting is False
    if not skip_next_scale_setting and len(cameras_train) > 1:
        print("🔄 Proceeding with next_scale setting")
        ref_camera = cameras_train[1]
        ref_camera_3dgs = convert_pt3d_cam_to_3dgs_cam(ref_camera, config=config)

        # get data for old points
        old_xyz = gaussians.get_xyz_all
        old_rotations = gaussians.get_rotation_all
        old_scales = gaussians.get_scaling_all
        old_next_scale = gaussians.get_next_scale_all

        # render to get visibility
        with torch.no_grad():
            visible_filter = gaussians.get_inscreen_points(ref_camera_3dgs)

        print(f"   Visible points: {visible_filter.sum()}/{len(visible_filter)}")

        # compute s_target only for visible points
        if visible_filter.sum() > 0:
            q, s_target = compute_inv_target_scale_per_frame(
                ref_camera,
                old_xyz[visible_filter],
                config=config
            )

            # find points that need updating: visible and next_scale is still inf
            visible_inf_mask = torch.isinf(old_next_scale[visible_filter])
            if visible_inf_mask.any():
                # update next_scale
                scale_visible = gaussians.next_scale[visible_filter]
                scale_visible[visible_inf_mask] = s_target[visible_inf_mask]
                gaussians.next_scale[visible_filter] = scale_visible
                print(f"✅ Updated next_scale for {visible_inf_mask.sum()} points (from inf to s_target)")
                print(f"   s_target range: {s_target[visible_inf_mask].min():.6f} - {s_target[visible_inf_mask].max():.6f}")
            else:
                print("ℹ️ No visible points need next_scale update (no visible inf points)")
        else:
            print("ℹ️ No visible points found")
    elif skip_next_scale_setting:
        print("⚠️ Skipping next_scale setting as requested")
    else:
        print("⚠️ Not enough cameras for next_scale setting (need cameras_train[1])")

    # merge Gaussians
    print("🔄 Merging gaussians...")
    gaussians_new.merge_gaussian(gaussians)

    return gaussians_new


def select_memory_for_training(gaussians, current_cameras, memory_cameras, memory_imgs, visibility_threshold=0.3):
    """
    Select cameras and images from memory for training.

    Args:
        gaussians: current GaussianModel
        current_cameras: list of current training cameras
        memory_cameras: list of cameras from memory
        memory_imgs: list of images from memory
        visibility_threshold: trainable point ratio threshold

    Returns:
        selected_cameras: list of selected memory cameras
        selected_imgs: list of selected memory images
    """
    global opt, background, xyz_scale

    print(f"📊 Selecting memory for training from {len(memory_cameras)} memory cameras...")

    # Step 1: render with current cameras to get visibility filter, determine trainable gaussians
    print("🔍 Step 1: Determining trainable gaussians from current cameras...")

    all_visible_gaussians = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device='cuda')

    for cam in current_cameras:
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cam, xyz_scale=xyz_scale, config=config)
        with torch.no_grad():
            render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
            visibility_filter = render_pkg["visibility_filter"]
            all_visible_gaussians[visibility_filter] = True

    n_trainable = all_visible_gaussians.sum().item()
    print(f"✅ Found {n_trainable} trainable gaussians from current cameras")

    if n_trainable == 0:
        print("⚠️  No trainable gaussians found, skipping memory selection")
        return [], []

    # Step 2: mark non-visible gaussians as non-trainable
    print("🔄 Step 2: Marking non-visible gaussians as non-trainable...")

    # set trainable mask: only gaussians visible to current cameras are trainable
    gaussians.set_trainable_mask(all_visible_gaussians)

    n_non_trainable = (~all_visible_gaussians).sum().item()
    print(f"✅ Set {n_trainable} gaussians as trainable, {n_non_trainable} as non-trainable")

    # Step 3: render with each memory camera, compute trainable point ratio
    print("🎯 Step 3: Evaluating memory cameras...")

    selected_cameras = []
    selected_imgs = []

    for i, mem_cam in enumerate(memory_cameras):
        try:
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(mem_cam, xyz_scale=xyz_scale, config=config)

            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
                visibility_filter = render_pkg["visibility_filter"]

                # get the number of trainable points
                n_trainable = gaussians.get_xyz.shape[0]

                # ensure only processing trainable points' visibility_filter
                if len(visibility_filter) > n_trainable:
                    # visibility_filter contains all points, take only the first n_trainable (corresponding to trainable points)
                    trainable_visibility_filter = visibility_filter[:n_trainable]
                else:
                    # if visibility_filter length equals trainable point count, use directly
                    trainable_visibility_filter = visibility_filter

                # compute the ratio of trainable points in the current viewpoint
                # all_visible_gaussians length should equal n_trainable
                # if len(all_visible_gaussians) != n_trainable:
                #     print(f"    ⚠️  Size mismatch: all_visible_gaussians={len(all_visible_gaussians)}, n_trainable={n_trainable}")
                #     continue

                # visible_trainable = all_visible_gaussians[trainable_visibility_filter]
                # trainable_ratio = visible_trainable.sum().item() / len(trainable_visibility_filter) if len(trainable_visibility_filter) > 0 else 0
                trainable_ratio = (visibility_filter&all_visible_gaussians).sum().item() / min(all_visible_gaussians.sum().item(),visibility_filter.sum().item())
                print(f"  📷 Memory camera {i+1}/{len(memory_cameras)}: {(visibility_filter&all_visible_gaussians).sum().item()}/{all_visible_gaussians.sum().item()} trainable points visible, ratio: {trainable_ratio:.3f}")

                if trainable_ratio > visibility_threshold:
                    selected_cameras.append(mem_cam)
                    selected_imgs.append(memory_imgs[i])
                    print(f"    ✅ Selected (ratio {trainable_ratio:.3f} > {visibility_threshold})")
                else:
                    print(f"    ❌ Skipped (ratio {trainable_ratio:.3f} <= {visibility_threshold})")

        except Exception as e:
            print(f"    ⚠️  Error processing memory camera {i+1}: {e}")
            continue

    print(f"🎉 Memory selection complete: {len(selected_cameras)}/{len(memory_cameras)} cameras selected")
    return selected_cameras, selected_imgs




def get_current_std(in_camera = None):
    global view_matrix_wonder, fx_wonder, fy_wonder, xyz_scale, gaussians, opt, background
    current_camera = kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, fx_wonder=fx_wonder, fy_wonder=fy_wonder, xyz_scale=xyz_scale)
    if in_camera is not None:
        current_camera = in_camera
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera,xyz_scale=xyz_scale, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt, background, config=config)
    depth = render_pkg['median_depth'][0].flatten()
    # return 99999999999999999
    if (depth > 50.).float().mean() > 0.1:
        return 99999999

    valid_mask = (depth > 1e-9) & (depth < 50.)
    return depth[valid_mask].std()

def train_gaussian(gaussians: GaussianModel, scene: Scene, opt: GSParams, all_depths=None, depth_align_masks=None, xyz_scale=xyz_scale, no_loss_masks=None, newly_added_points=0, hq_mode=False, trainable_mask=None, color_only=False, **kwargs):
    import math

    # force freeze points with background label
    global GLOBAL_LABEL_NAMES
    in_trainable_mask = trainable_mask

    # color-only mode variable initialization
    original_lr_dict = {}
    original_feature_lr = None

    # control sky layer trainability: newly added points (including sky) are trainable, previously trained sky points are not
    if trainable_mask is None and hasattr(gaussians, 'is_sky_filter') and gaussians.is_sky_filter.numel() > 0:
        n_current = gaussians.get_xyz.shape[0]  # current trainable point count
        n_total = gaussians.get_xyz_all.shape[0]  # total point count

        if n_current > 0:
            # check if there is a marker for newly added points

            if newly_added_points > 0:
                # case where new points have been added
                n_existing = n_current - newly_added_points

                if hq_mode:
                    # HQ mode: only new points are trainable, all existing points are frozen
                    print(f"🎬 HQ mode training control:")
                    print(f"  📊 Total points: {n_current} (existing: {n_existing}, new: {newly_added_points})")

                    # newly added points: all trainable (including sky points)
                    new_trainable_mask = torch.ones(newly_added_points, dtype=torch.bool, device='cuda')

                    # all existing points are non-trainable
                    existing_trainable_mask = torch.zeros(n_existing, dtype=torch.bool, device='cuda') if n_existing > 0 else torch.empty(0, dtype=torch.bool, device='cuda')

                    # merge mask - note: order in merge_gaussian is [new points, existing points]
                    trainable_mask = torch.cat([new_trainable_mask, existing_trainable_mask]).to('cuda')

                    print(f"  🆕 New points: {newly_added_points} trainable (all allowed for HQ mode)")
                    print(f"  🔒 Existing points: 0/{n_existing} trainable (all frozen for HQ mode)")
                    print(f"  📝 Result: {trainable_mask.sum().item()}/{n_current} points will be trained")

                else:
                    # normal mode: existing sky points are non-trainable, existing non-sky points are trainable, all new points are trainable
                    print(f"🔄 Sky layer training control:")
                    print(f"  📊 Total points: {n_current} (existing: {n_existing}, new: {newly_added_points})")

                    # get sky filter corresponding to trainable points
                    # note: is_sky_filter contains all points, we need to find the portion corresponding to current trainable points
                    # this needs to be determined based on actual merge logic

                    # assume new points are added at the end and trainable points are in order
                    if n_total == n_current:
                        # all points are trainable
                        current_sky_filter = gaussians.is_sky_filter.to('cuda')
                    else:
                        # some points are trainable, need to find the corresponding sky filter
                        # here we assume the first n_current points are the trainable ones
                        current_sky_filter = gaussians.is_sky_filter[:n_current].to('cuda')

                    # process existing points and newly added points separately
                    if n_existing > 0:
                        # existing points: sky points are non-trainable, non-sky points are trainable
                        existing_sky_filter = current_sky_filter[:n_existing]
                        existing_trainable_mask = (~existing_sky_filter).to('cuda')
                        n_existing_trainable = existing_trainable_mask.sum().item()
                        n_existing_sky = existing_sky_filter.sum().item()
                        print(f"  🔒 Existing points: {n_existing_trainable}/{n_existing} trainable ({n_existing_sky} sky points frozen)")
                    else:
                        existing_trainable_mask = torch.empty(0, dtype=torch.bool, device='cuda')
                        n_existing_trainable = 0
                        n_existing_sky = 0

                    # newly added points: all trainable (including sky points)
                    new_trainable_mask = torch.ones(newly_added_points, dtype=torch.bool, device='cuda')
                    n_new_sky = current_sky_filter[n_existing:].sum().item() if newly_added_points > 0 else 0
                    print(f"  🆕 New points: {newly_added_points} trainable ({n_new_sky} sky points allowed)")

                    # merge mask - note: order in merge_gaussian is [new points, existing points]
                    trainable_mask = torch.cat([new_trainable_mask, existing_trainable_mask]).to('cuda')

                    print(f"  📝 Result: {trainable_mask.sum().item()}/{n_current} points will be trained")

            else:
                # no new points added: use default strategy (all non-sky points are trainable)
                if n_total == n_current:
                    current_sky_filter = gaussians.is_sky_filter.to('cuda')
                else:
                    current_sky_filter = gaussians.is_sky_filter[:n_current].to('cuda')

                trainable_mask = (~current_sky_filter).to('cuda')
                n_trainable = trainable_mask.sum().item()
                n_sky = current_sky_filter.sum().item()
                print(f"🌍 Normal training: {n_trainable}/{n_current} points trainable ({n_sky} sky points frozen)")

            # set trainable mask
            trainable_mask = trainable_mask.to('cuda')
            gaussians.set_trainable_mask(trainable_mask)

        else:
            print("⚠️  No trainable points found, skipping sky layer control")
    else:
        print("⚠️  No sky filter found, all points remain trainable")

    # handle merging of background and external trainable_mask
    if "background" in GLOBAL_LABEL_NAMES:
        background_mask = gaussians.get_label_mask("background")
        if background_mask.any():
            n_background = background_mask.sum().item()
            print(f"🧊 Found {n_background} background points")

            if in_trainable_mask is not None:
                # has external mask, merge: exclude background points
                print(f"🔧 Merging external trainable_mask with background exclusion")
                final_trainable_mask = in_trainable_mask & (~background_mask)
                n_final = final_trainable_mask.sum().item()
                print(f"🔧 Final trainable points: {n_final}/{len(final_trainable_mask)} (external mask + background exclusion)")
                gaussians.set_trainable_mask(final_trainable_mask)
            else:
                # no external mask, freeze background using original logic
                print(f"🧊 Force freezing {n_background} background points")
                gaussians.freeze_labels("background")
        else:
            # no background points
            if in_trainable_mask is not None:
                print(f"🔧 Using provided trainable_mask: {in_trainable_mask.sum().item()}/{len(in_trainable_mask)} points trainable")
                gaussians.set_trainable_mask(in_trainable_mask)
    else:
        # no background label
        if in_trainable_mask is not None:
            print(f"🔧 Using provided trainable_mask: {in_trainable_mask.sum().item()}/{len(in_trainable_mask)} points trainable")
            gaussians.set_trainable_mask(in_trainable_mask)

    # Color-only training mode: only train colors, freeze geometry parameters
    if color_only:
        print("🎨 Color-only training mode: Freezing geometry parameters, training colors only")
        # save original learning rates
        for group in gaussians.optimizer.param_groups:
            original_lr_dict[group['name']] = group['lr']

        # save original feature_lr
        original_feature_lr = opt.feature_lr

        # set appropriate feature_lr for color training
        opt.feature_lr = 0.0025
        gaussians.training_setup(opt)
        print(f"   🎨 Set feature_lr: {original_feature_lr} -> {opt.feature_lr}")

        # update feature learning rate in optimizer
        gaussians.update_learning_rate(1)  # trigger learning rate update

        # freeze geometry parameters: set learning rate to 0
        geometry_params = ['xyz', 'scaling', 'rotation', 'opacity']
        for group in gaussians.optimizer.param_groups:
            if group['name'] in geometry_params:
                group['lr'] = 0.0
                print(f"   🧊 Frozen {group['name']} (lr: {original_lr_dict[group['name']]} -> 0.0)")
            else:
                print(f"   🎨 Training {group['name']} (lr: {group['lr']})")

        print("   🚫 Densification disabled in color-only mode")

    iterable_gauss = range(1, opt.iterations + 1)
    pbar = tqdm(iterable_gauss, desc="Training Gaussians")

    for iteration in pbar:
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:       
            gaussians.oneupSHdegree()

        viewpoint_stack = scene.getTrainCameras().copy()
        train_idx = randint(0, len(viewpoint_stack)-1)
        no_loss_mask = no_loss_masks[train_idx] if no_loss_masks is not None else None
        viewpoint_cam = viewpoint_stack.pop(train_idx)

        render_pkg = render(viewpoint_cam, gaussians, opt, background, config=config)
        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # your original depth rendering function
        def simple_differentiable_depth_render(gaussians, viewpoint_cam):
            # ... keep original implementation ...
            device = gaussians.get_xyz_all.device
            H, W = config['orig_H'], config['orig_W']

            xyz = gaussians.get_xyz_all
            R = torch.tensor(viewpoint_cam.R, device=device, dtype=torch.float32)
            T = torch.tensor(viewpoint_cam.T, device=device, dtype=torch.float32)

            xyz_cam = xyz @ R + T[None, :]
            depths = xyz_cam[:, 2] / xyz_scale

            valid_mask = depths > 1e-9
            if not valid_mask.any():
                return torch.zeros(H, W, device=device, requires_grad=True), torch.zeros(H, W, device=device, dtype=torch.bool)

            xyz_cam = xyz_cam[valid_mask]
            depths = depths[valid_mask]

            focal_x = W / (2 * math.tan(viewpoint_cam.FoVx / 2))
            focal_y = H / (2 * math.tan(viewpoint_cam.FoVy / 2))

            x_screen = (xyz_cam[:, 0] / xyz_cam[:, 2]) * focal_x + W / 2
            y_screen = (xyz_cam[:, 1] / xyz_cam[:, 2]) * focal_y + H / 2

            x_pixel = torch.clamp(torch.round(x_screen).long(), 0, W-1)
            y_pixel = torch.clamp(torch.round(y_screen).long(), 0, H-1)

            pixel_indices = y_pixel * W + x_pixel
            depth_map_flat = torch.full((H * W,), float('inf'), device=device, dtype=torch.float32)

            try:
                depth_map_flat = depth_map_flat.scatter_reduce(0, pixel_indices, depths, reduce='amin')
            except:
                for i in range(len(depths)):
                    idx = pixel_indices[i].item()
                    if depth_map_flat[idx] > depths[i]:
                        depth_map_flat[idx] = depths[i]

            depth_map = depth_map_flat.view(H, W)

            finite_mask = depth_map != float('inf')
            valid_mask = depth_map > 1e-9
            valid_mask = valid_mask & finite_mask & (depth_map < 200)
            if finite_mask.any():
                depth_map = torch.where(finite_mask, depth_map, -1)

            return depth_map, valid_mask.squeeze()

        # compute losses
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image, no_loss_mask=no_loss_mask)
        scaling_loss = scaling_regularization_loss(gaussians, viewpoint_cam)


        if all_depths is not None and train_idx < len(all_depths):
            target_depth = all_depths[train_idx].to('cuda')  # [H, W]
            depth_mask = depth_align_masks[train_idx].to('cuda')  # [H, W]

            # first render current depth map
            depth_now, valid_mask = simple_differentiable_depth_render(gaussians, viewpoint_cam)
            valid_mask = valid_mask.to(depth_now.device) & depth_mask.to(depth_now.device)

            # compute depth loss (for fine-tuning)
            depth_loss = (depth_now - target_depth)**2 * valid_mask.to(depth_now.device)
            depth_loss = depth_loss.sum() / (valid_mask.sum() + 1e-9)

            if iteration % 10 == 0:
                mean_error = torch.abs(depth_now - target_depth)[valid_mask].mean()
                print(f"Pixel-wise depth error: {mean_error.item():.4f}")

        else:
            depth_loss = torch.tensor(0.0, device='cuda')

        # combine losses
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, no_loss_mask=no_loss_mask)) + depth_loss + scaling_loss

        # ... subsequent code unchanged ...

        aniso_loss = torch.tensor(0.0, device='cuda')
        if hasattr(opt, 'lambda_aniso') and opt.lambda_aniso > 0:
            aniso_loss = anisotropy_regularizer(gaussians, r_threshold=2.0)
            loss += opt.lambda_aniso * aniso_loss

        loss.backward()

        # update tqdm progress bar to display loss
        postfix_dict = {
            'loss': f'{loss.item():.6f}',
            'l1_loss': f'{Ll1.item():.6f}',
            'iter': f'{iteration}/{opt.iterations}',
            'aniso_loss': f'{aniso_loss.item()*opt.lambda_aniso:.6f}' if hasattr(opt, 'lambda_aniso') and opt.lambda_aniso > 0 else '0.000000',
            'scaling_loss': f'{scaling_loss.item():.6f}',
            'depth_loss': f'{depth_loss.item():.6f}' if all_depths is not None else 'None'
        }

        # Add anisotropy loss to progress bar if being used
        if hasattr(opt, 'lambda_aniso') and opt.lambda_aniso > 0:
            postfix_dict['aniso_loss'] = f'{aniso_loss.item():.6f}'

        pbar.set_postfix(postfix_dict)


        # ... densification etc. code unchanged ...
        with torch.no_grad():
            n_trainable = gaussians.get_xyz.shape[0]

            if len(visibility_filter) > n_trainable:
                trainable_visibility_filter = visibility_filter[:n_trainable]
                trainable_radii = radii[:n_trainable]
                trainable_viewspace_grad = viewspace_point_tensor.grad[:n_trainable] if viewspace_point_tensor.grad is not None else None
            else:
                trainable_visibility_filter = visibility_filter
                trainable_radii = radii
                trainable_viewspace_grad = viewspace_point_tensor.grad

            # in color-only mode, skip densification and only adjust colors
            if iteration < opt.densify_until_iter and not color_only:
                gaussians.max_radii2D[trainable_visibility_filter] = torch.max(
                    gaussians.max_radii2D[trainable_visibility_filter], trainable_radii[trainable_visibility_filter])

                if trainable_viewspace_grad is not None:
                    gaussians.add_densification_stats(trainable_viewspace_grad, trainable_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0 
                    or (opt.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    pbar.close()

    # restore original learning rates (if in color-only mode)
    if color_only:
        print("🎨 Restoring original learning rates after color-only training")

        # restore original feature_lr
        opt.feature_lr = original_feature_lr
        print(f"   🎨 Restored feature_lr: 0.0025 -> {original_feature_lr}")

        # update optimizer learning rate
        gaussians.update_learning_rate(1)  # trigger learning rate update

        for group in gaussians.optimizer.param_groups:
            if group['name'] in original_lr_dict:
                original_lr = original_lr_dict[group['name']]
                group['lr'] = original_lr
                print(f"   ✅ Restored {group['name']} lr: 0.0 -> {original_lr}")

    # after training, set points with opacity > 0.5 to 1.0 (fully opaque)
    # with torch.no_grad():
    #     current_opacity = gaussians.get_opacity_all
    #     gaussians.merge_all_to_trainable()# get opacity for all points
    #     high_opacity_mask = current_opacity.squeeze() > 0.5  # mask of shape [N]

    #     if high_opacity_mask.any():
    #         n_high_opacity = high_opacity_mask.sum().item()
    #         # use model's inverse_opacity_activation to correctly set opacity to 1.0
    #         target_opacity = 1.0
    #         target_raw = gaussians.inverse_opacity_activation(torch.tensor(target_opacity, device='cuda'))
    #         gaussians._opacity.data[high_opacity_mask] = target_raw
    #         print(f"🔆 Post-training: Set {n_high_opacity} points with opacity > 0.5 to opacity = {target_opacity}")
    #     else:
    #         print("🔆 Post-training: No points with opacity > 0.5 found")

    print("🔄 Merging all gaussians back to trainable parameters...")
    gaussians.merge_all_to_trainable()


def save_gaussian_with_global_labels(gaussians, path):
    """
    Save GaussianModel in the run_inf module, ensuring global label mapping is correctly saved
    """
    print(f"🏷️  Saving GaussianModel with global labels to: {path}")

    # first merge all points to trainable
    gaussians.merge_all_to_trainable()

    # collect all attributes to save
    keys = [
        "_xyz", "_features_dc", "_scaling", "_rotation", "_opacity",
        "_focal_length", "next_scale", "prior_scale", "now_scale",
        "max_radii2D", "xyz_gradient_accum", "denom", "filter_3D",
        "visibility_filter_all", "is_sky_filter", "delete_mask_all", "point_labels"
    ]

    # backward-compatible saving of prev labels
    if hasattr(gaussians, "point_labels_prev"):
        keys.append("point_labels_prev")

    state = {key: getattr(gaussians, key).detach().cpu() for key in keys if hasattr(gaussians, key)}

    # save model configuration
    state["max_sh_degree"] = gaussians.max_sh_degree
    state["floater_dist2_threshold"] = gaussians.floater_dist2_threshold

    # directly access global variables of this module
    state["global_label_names"] = list(GLOBAL_LABEL_NAMES)
    state["global_label_map"] = dict(GLOBAL_LABEL_MAP)

    print(f"🏷️  Saving global labels: {state['global_label_names']}")
    print(f"🏷️  Total labels: {len(state['global_label_names'])}")

    torch.save(state, path)
    print(f"✅ GaussianModel with global labels saved to: {path}")

def load_gaussian_with_global_labels(path, config):
    """
    Load GaussianModel in the run_inf module, ensuring global label mapping is correctly restored
    """
    print(f"🏷️  Loading GaussianModel with global labels from: {path}")

    state = torch.load(path, map_location="cuda")

    # create new GaussianModel instance
    gaussians = GaussianModel(
        sh_degree=state.get("max_sh_degree", 3),
        floater_dist2_threshold=state.get("floater_dist2_threshold", 0.0002),
        config=config
    )

    # restore all attributes
    for key, val in state.items():
        if key in ["max_sh_degree", "floater_dist2_threshold", "global_label_names", "global_label_map"]:
            continue
        if hasattr(gaussians, key):
            existing = getattr(gaussians, key)
            if isinstance(existing, torch.nn.Parameter):
                setattr(gaussians, key, torch.nn.Parameter(val.cuda(), requires_grad=True))
            else:
                setattr(gaussians, key, val.cuda())
        else:
            setattr(gaussians, key, val.cuda())

    # directly update global variables of this module
    loaded_names = state.get("global_label_names", None)
    loaded_map = state.get("global_label_map", None)

    if loaded_names is not None and loaded_map is not None:
        print(f"🏷️  Loading global labels: {loaded_names}")

        # clear and update global variables
        GLOBAL_LABEL_NAMES.clear()
        GLOBAL_LABEL_NAMES.extend(loaded_names)

        GLOBAL_LABEL_MAP.clear()
        GLOBAL_LABEL_MAP.update(loaded_map)

        print(f"✅ Updated global labels: {GLOBAL_LABEL_NAMES}")
        print(f"✅ Updated global map: {GLOBAL_LABEL_MAP}")
    else:
        print("⚠️  No global labels found in saved state, using defaults")

    print(f"✅ GaussianModel with global labels loaded from: {path}")
    return gaussians


class _WebSocketCloseNoise(logging.Filter):
    """Drop Werkzeug's report of a closed Socket.IO WebSocket.

    Engine.IO takes the socket over for the WebSocket and never starts an HTTP response, so when the
    browser disconnects Werkzeug logs a '500' request line plus an 'AssertionError: write() before
    start_response' traceback. Both are harmless; any other request error is still logged."""

    def filter(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, AssertionError) and "write() before start_response" in str(exc):
            return False
        message = record.getMessage()
        if "write() before start_response" in message:
            return False
        if "/socket.io/" in message and "transport=websocket" in message and '" 500 ' in message:
            return False
        return True


def start_server(host, port):
    logging.getLogger("werkzeug").addFilter(_WebSocketCloseNoise())
    # allow_unsafe_werkzeug: flask_socketio refuses the Werkzeug server when stdin is not a TTY
    # (nohup, slurm, tests). The default bind address is 127.0.0.1; use --host 0.0.0.0 to expose it.
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)


SPLAT_DIR = str(WZ_ROOT / "splat-main")


@app.route('/')
def serve_index():
    return send_from_directory(SPLAT_DIR, 'index_gen.html')


@app.route('/splat-main/<path:filename>')
def serve_splat_static(filename):
    return send_from_directory(SPLAT_DIR, filename)


def _busy_reason(allow_preview=True):
    """Why a new request cannot start now, or None."""
    if gaussians is None or kf_gen is None or server_status.get('state') == 'loading':
        return 'the scene is still loading'
    if busy_job is not None:
        return f'busy with {busy_job}'
    if not keep_rendering:
        return 'a generation request is pending'
    if orbit_state.get('hq_ready') or (orbit_state.get('is_orbiting') and
                                        (not allow_preview or orbit_state.get('mode') in ('high_quality', 'crack_fix'))):
        return 'an orbit is in progress'
    return None


def _reject(action, reason):
    message = f'{action} ignored: {reason}'
    print(f"⚠️ {message}")
    socketio.emit('server-state', message, room=request.sid)


@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    global client_id
    client_id = request.sid
    emit('server-config', server_config_payload())
    with status_lock:
        payload = dict(server_status)
    emit('server-status', payload)
    emit('scene-prompt', scene_name)
    if busy_job is None and scene_lock.acquire(timeout=0.5):  # a page opened after the last job
        try:
            stats = scene_stats_payload()
        finally:
            scene_lock.release()
        if stats is not None:
            emit('scene-stats', stats)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)
    global client_id
    if client_id == request.sid:
        client_id = None

@socketio.on('rewrite')
def handle_rewrite():
    global rewrite_background
    rewrite_background = not rewrite_background
    socketio.emit('server-state', f"🔄 Rewrite background: {rewrite_background}", room=client_id)
    print(f"🔄 Rewrite background: {rewrite_background}")


@socketio.on("gen")
def handle_gen(data):
    global view_matrix, keep_rendering, trajectory_points, fx_wonder, fy_wonder, busy_job, coz_request_seed
    global gen_fx, gen_fy, job_scene_name

    if not isinstance(data, dict) or not isinstance(data.get('viewMatrix'), (list, tuple)) or len(data['viewMatrix']) != 16:
        _reject('gen', 'invalid request (expected {viewMatrix: 16 floats, fx, fy, addToTrajectory})')
        return
    reason = _busy_reason(allow_preview=True)
    if reason:
        _reject('gen', reason)
        return
    fx = data.get('fx')
    fy = data.get('fy')
    fx = float(fx if fx is not None else (fx_wonder if fx_wonder is not None else config["init_focal_length"]))
    fy = float(fy if fy is not None else (fy_wonder if fy_wonder is not None else fx))
    add_to_trajectory = bool(data.get('addToTrajectory', False))
    coz_seed = data.get('cozSeed')  # optional fixed Chain-of-Zoom seed (tests/e2e_headless.py --coz_seed)
    if coz_seed is not None and (isinstance(coz_seed, bool) or not isinstance(coz_seed, int)):
        _reject('gen', 'cozSeed must be an integer')
        return
    zoom_in = fx > config["init_focal_length"]
    n_points = len(trajectory_points) + (1 if add_to_trajectory else 0)
    if zoom_in:
        if not FEATURES['coz']:
            _reject('zoom-in', 'Chain-of-Zoom service not enabled')
            return
        if n_points != 2:
            _reject('zoom-in', f'set exactly 1 H point (the start view) before R; {len(trajectory_points)} point(s) set'
                    + ('' if add_to_trajectory else ' (without R, exactly 2 are needed)'))
            return
    else:
        if not FEATURES['gen3c']:
            _reject('camera move', 'Gen3C service not enabled')
            return
        if n_points < 1:
            _reject('camera move', 'no trajectory point: add one with H or press R')
            return

    view_matrix = data['viewMatrix']
    fx_wonder = fx
    fy_wonder = fy
    gen_fx, gen_fy = fx, fy
    if add_to_trajectory:
        # R key: first add current pose to trajectory
        current_camera = kf_gen.get_camera_by_js_view_matrix(
            view_matrix, fx_wonder=fx_wonder, fy_wonder=fy_wonder, xyz_scale=xyz_scale
        )
        trajectory_points.append(current_camera)
        print(f"Added current camera to trajectory. Total points: {len(trajectory_points)}")

    coz_request_seed = coz_seed if zoom_in else None
    # The object prompt in effect now is inserted at the end of this zoom-in; camera moves keep it.
    job_scene_name = scene_name if zoom_in else None
    busy_job = 'zoom' if zoom_in else 'move'
    set_status('busy', busy_job, f'{busy_job} request accepted')
    keep_rendering = False


@socketio.on('generate-nvs')
def handle_generate_nvs():
    global orbit_state
    reason = _busy_reason(allow_preview=False)
    if reason:
        _reject('orbit preview', reason)
        return
    # Start orbiting
    orbit_state['is_orbiting'] = True
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'normal'
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = None

@socketio.on('generate-nvs-hq')
def handle_generate_nvs_hq():
    global orbit_state, view_matrix_wonder, fx_wonder, fy_wonder, xyz_scale, kf_gen, busy_job
    reason = _busy_reason(allow_preview=False)
    if reason:
        _reject('high-quality NVS', reason)
        return
    if not FEATURES['gen3c']:
        _reject('high-quality NVS', 'Gen3C service not enabled')
        return
    print("🎬 Starting high-quality NVS (ctrl++shift+space)...")

    # Store the current camera for input.png
    current_camera = kf_gen.get_camera_by_js_view_matrix(
        view_matrix_wonder,
        xyz_scale=xyz_scale,
        fx_wonder=fx_wonder,
        fy_wonder=fy_wonder
    )

    # Start high-quality orbiting
    busy_job = 'hq_nvs'
    orbit_state['is_orbiting'] = True
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'high_quality'
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = current_camera  # Store current camera for input.png
    set_status('busy', busy_job, 'Collecting orbit frames for high-quality NVS...')
    socketio.emit('server-state', 'Starting high-quality NVS...', room=client_id)

@socketio.on('fix-small-cracks')
def handle_fix_small_cracks():
    global orbit_state, view_matrix_wonder, fx_wonder, fy_wonder, xyz_scale, kf_gen, busy_job
    reason = _busy_reason(allow_preview=False)
    if reason:
        _reject('crack fixing', reason)
        return
    print("🔧 Starting small cracks fixing (cv2.inpaint mode)...")

    # Store the current camera for input.png
    current_camera = kf_gen.get_camera_by_js_view_matrix(
        view_matrix_wonder,
        xyz_scale=xyz_scale,
        fx_wonder=fx_wonder,
        fy_wonder=fy_wonder
    )

    # Start crack fixing orbiting - same as HQ mode but different processing
    busy_job = 'crack_fix'
    orbit_state['is_orbiting'] = True
    orbit_state['cameras'] = None
    orbit_state['current_frame'] = 0
    orbit_state['mode'] = 'crack_fix'  # New mode for crack fixing
    orbit_state['collected_frames'] = []
    orbit_state['collected_masks'] = []
    orbit_state['input_camera'] = current_camera
    orbit_state['hq_ready'] = False
    set_status('busy', busy_job, 'Collecting orbit frames for crack fixing...')
    socketio.emit('server-state', 'Starting small cracks fixing...', room=client_id)


@socketio.on('render-pose')
def handle_render_pose(data):
    global view_matrix_wonder, fx_wonder, fy_wonder, keep_rendering

    if config is None:
        return
    if isinstance(data, dict) and 'viewMatrix' in data:
        view_matrix_wonder = data.get('viewMatrix')
        fx_wonder = data.get('fx') if data.get('fx') is not None else config["init_focal_length"]
        fy_wonder = data.get('fy') if data.get('fy') is not None else config["init_focal_length"]
    else:
        # fallback for old clients
        view_matrix_wonder = data
        fx_wonder = config["init_focal_length"]
        fy_wonder = config["init_focal_length"]

@socketio.on('scene-prompt')
def handle_new_prompt(data):
    """Object to insert at the end of the next zoom-in ('' clears it).

    The prompt is kept across camera moves until a zoom-in uses it. A zoom-in uses the prompt that
    was set when it was accepted (handle_gen); a prompt sent while it runs waits for the next one."""
    global scene_name
    if not isinstance(data, str):
        _reject('scene-prompt', 'expected a string')
        return
    zoom_running = busy_job == 'zoom'
    if data.strip() == "" or data == "None" or data == "none":
        scene_name = None
        print('Received None scene prompt ')
        if zoom_running and job_scene_name is not None:
            socketio.emit('server-state', f"Object prompt cleared; the zoom-in in progress still inserts "
                                          f"'{job_scene_name}'", room=request.sid)
        return
    if not FEATURES['objects']:
        # The zoom-in proceeds without insertion.
        scene_name = None
        message = 'object insertion unavailable: ' + '; '.join(OBJECTS_MISSING or ['see the server log'])
        print(f"⚠️ {message}")
        socketio.emit('server-state', message, room=request.sid)
        socketio.emit('scene-prompt', '', room=request.sid)
        return
    scene_name = data
    print('Received new scene prompt: ' + data)
    if zoom_running:
        socketio.emit('server-state', f"Object '{data}' will be inserted at the end of the next zoom-in "
                                      "(not the one in progress)", room=request.sid)

@socketio.on('undo')
def handle_undo():
    """Queue an undo of the last scene-changing job (performed by the main loop)."""
    global undo, busy_job
    print('Received undo signal.')
    reason = _busy_reason(allow_preview=True)
    if reason:
        _reject('undo', reason)
        return
    if scene_snapshot is None:
        _reject('undo', 'nothing to undo')
        return
    busy_job = 'undo'
    set_status('busy', busy_job, 'undo request accepted')
    undo = True

@socketio.on('save')
def handle_save():
    print('Received save signal.')
    reason = _busy_reason(allow_preview=True)
    if reason:
        _reject('save', reason)
        return
    global save, busy_job
    busy_job = 'save'
    set_status('busy', busy_job, 'save request accepted')
    save = True

@socketio.on('delete')
def handle_delete(data):
    print('Received delete signal.')
    global delete, view_matrix_delete, busy_job
    if not isinstance(data, (list, tuple)) or len(data) != 16:
        _reject('delete', 'expected a view matrix (16 floats)')
        return
    reason = _busy_reason(allow_preview=True)
    if reason:
        _reject('delete', reason)
        return
    view_matrix_delete = data
    busy_job = 'delete'
    set_status('busy', busy_job, 'delete request accepted')
    delete = True

@socketio.on('add-trajectory-point')
def handle_add_trajectory_point(data):
    global trajectory_points, fx_wonder, fy_wonder

    if not isinstance(data, dict) or not isinstance(data.get('viewMatrix'), (list, tuple)):
        _reject('add-trajectory-point', 'expected {viewMatrix, fx, fy}')
        return
    if kf_gen is None:
        _reject('add-trajectory-point', 'the scene is still loading')
        return
    if busy_job in ('zoom', 'move'):
        # The running job reads and then clears trajectory_points: a point added now would be
        # dropped, or recorded in gen_matrices as generated although it never was.
        _reject('add-trajectory-point', f'busy with {busy_job}; add it when the job has finished')
        return
    view_matrix_data = data.get('viewMatrix')
    fx = data.get('fx', fx_wonder)
    fy = data.get('fy', fy_wonder)

    # convert frontend viewMatrix to backend camera object
    camera_pose = kf_gen.get_camera_by_js_view_matrix(
        view_matrix_data,
        fx_wonder=fx,
        fy_wonder=fy,
        xyz_scale=xyz_scale
    )

    trajectory_points.append(camera_pose)

    print(f"Added trajectory point {len(trajectory_points)}. Total points: {len(trajectory_points)}")
    socketio.emit('server-state', f'Trajectory point {len(trajectory_points)} added', room=client_id)


@socketio.on('clear-trajectory')
def handle_clear_trajectory():
    global trajectory_points
    if busy_job in ('zoom', 'move'):
        _reject('clear-trajectory', f'busy with {busy_job}')
        return
    trajectory_points = []
    print("Trajectory cleared")
    socketio.emit('server-state', 'Trajectory cleared', room=client_id)

@socketio.on('complete-background')
def handle_complete_background():
    """P key: inpaint the background behind the objects of the current view (object-insertion stack)."""
    global complete_background_pose, busy_job
    if not FEATURES.get('objects'):
        _reject('complete-background', 'object insertion is not available')
        return
    reason = _busy_reason(allow_preview=False)
    if reason:
        _reject('complete-background', reason)
        return
    complete_background_pose = kf_gen.get_camera_by_js_view_matrix(
        view_matrix_wonder,
        fx_wonder=fx_wonder,
        fy_wonder=fy_wonder,
        xyz_scale=xyz_scale
    )
    busy_job = 'complete_background'
    set_status('busy', busy_job, 'complete-background request accepted')


def generate_orbit_cameras(center_camera, gaussians, opt, xyz_scale, n_frames=49, radius_factor=8e-4, height_variation=0e-4, n_spirals=1, transition_frames=9):
    """Generate cameras in a spiral trajectory for better 3D perception

    Args:
        center_camera: The current camera (will be cameras[0])
        gaussians: Gaussian model
        opt: Options
        xyz_scale: Scale factor
        n_frames: Total number of frames (default 49)
        radius_factor: Base radius factor, will be scaled by focal length (default 6e-4)
        height_variation: Height variation during orbit
        n_spirals: Number of spirals
        transition_frames: Number of frames to transition from center_camera to orbit start (default 9)
    """
    import math
    import torch

    cameras = []
    device = center_camera.device

    # Calculate orbit frames count
    orbit_frames = n_frames - transition_frames

    # get current camera parameters
    transform_matrix_pt3d = center_camera.get_world_to_view_transform().get_matrix()[0].transpose(0,1).inverse() # c2w
    center_R = transform_matrix_pt3d[:3, :3]
    center_T = transform_matrix_pt3d[:3, 3]
    center_K = center_camera.K

    # compute radius proportional to focal length
    focal_length = center_K[0, 0, 0].item()  # fx
    reference_focal = 10240.0  # reference focal length
    radius = radius_factor * min(focal_length / reference_focal, 1024.0/10240.)*6
    radius *= 0.5 if center_camera.K[0,0,0]<30538 else 1

    # print(f"Focal length: {focal_length:.2f}, Calculated radius: {radius:.6f}")
    factor = (0.3 if center_camera.K[0,0,0]>30538 else 1.0)
    # factor = (0.4 if center_camera.K[0,0,0]>140000 else 1.0)
    radius *=  1 #(factor*1 if center_camera.K[0,0,0]>6538 else 1)
    # Step 1: get scene depth via depth rendering
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(center_camera, xyz_scale=xyz_scale, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt, torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda'), render_visible=True, config=config)
    depth_map = render_pkg["median_depth"][0].squeeze().detach().cpu() / xyz_scale

    H, W = depth_map.shape
    center_depth = depth_map[H//2, W//2].item()

    if center_depth <= 0 or center_depth > 10:
        center_depth = 1e-2
    else:
        center_depth *= 1.00

    # Step 2: compute look-at point
    camera_center = center_camera.get_camera_center().squeeze(0)  # [3]

    R_transposed = center_R #.transpose(-2, -1)  # [1, 3, 3]
    forward_world = R_transposed[ :, 2]  # [3] forward direction

    # look-at point = camera center + depth * forward direction
    look_at_point = camera_center + center_depth * forward_world
    print(f"look_at_point: {look_at_point*1000}")
    # print(f"Camera center: {camera_center}")
    # print(f"Forward direction: {forward_world}")
    # print(f"Look-at point: {look_at_point}")

    # Step 3: compute coordinate system
    forward_unit = forward_world / torch.norm(forward_world)

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    # if torch.abs(torch.dot(forward_unit, world_up)) > 0.9:
    #     world_up = torch.tensor([1.0, 0.0, 0.0], device=device)

    # world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    # forward_unit = torch.tensor([0.0, 0.0, 1.0], device=device)
    right = torch.cross(forward_unit, world_up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward_unit)
    up = up / torch.norm(up)

    # Step 4: generate orbit camera sequence 
    orbit_cameras = []
    for i in range(1,orbit_frames+1):
        t = i / orbit_frames
        angle = 2 * math.pi * n_spirals * t

        current_radius = radius 

        offset_x = current_radius * math.cos(angle)
        offset_y = current_radius * math.sin(angle)
        offset_z = center_depth * 0.1 * t *0

        height_offset = height_variation * math.sin(2 * math.pi * t * 1.5) 

        new_camera_pos = (camera_center + 
                         offset_x * right + 
                         (offset_y + height_offset) * up + 
                         offset_z * forward_unit + forward_unit*center_depth*(1-factor) )

        # compute new forward direction (pointing to look-at point)
        new_forward = look_at_point - new_camera_pos
        new_forward = new_forward / torch.norm(new_forward)

        # recompute coordinate system
        new_right = torch.cross(new_forward, world_up)
        if torch.norm(new_right) > 1e-6:
            new_right = new_right / torch.norm(new_right)

        new_up = torch.cross(new_right, new_forward)

        # build rotation matrix
        R_cam_to_world = torch.stack([-new_right, new_up, new_forward], dim=1)  # [3, 3] column vectors

        T_new = new_camera_pos.unsqueeze(0)
        new_w2c = torch.zeros((4,4), device=device)
        new_w2c[:3, :3] = R_cam_to_world
        new_w2c[:3, 3] = T_new
        new_w2c[3, 3] = 1
        new_w2c = new_w2c.inverse()

        # camera1 = PerspectiveCameras(
        #     R=new_w2c[:3, :3].transpose(0,1).unsqueeze(0), 
        #     T=new_w2c[:3, 3].unsqueeze(0), 
        #     # R=center_R,
        #     # T=center_T,
        #     K=center_K,
        #     image_size=center_camera.image_size, 
        #     device=device
        # )

        camera = copy.deepcopy(center_camera)

        camera.R = new_w2c[:3, :3].transpose(0,1).unsqueeze(0)
        camera.T = new_w2c[:3, 3].unsqueeze(0)
        camera.K = copy.deepcopy(center_K)
        camera.K[0, 0, 0] *= factor
        camera.K[ 0,1,1] *= factor
        camera.image_size = center_camera.image_size
        camera.device = device
        # comprehensive_camera_comparison(camera1, camera)
        orbit_cameras.append(camera)

    # Step 5: generate transition sequence from current_camera to the first orbit frame
    if transition_frames > 0 and len(orbit_cameras) > 0:
        first_orbit_camera = orbit_cameras[0]

        # use existing interpolation function to generate transition frames
        from util.utils import interpolate_cameras_RT
        transition_cameras = interpolate_cameras_RT(center_camera, first_orbit_camera, num_frames=transition_frames + 1, config=config)

        # remove the last frame (because it duplicates orbit_cameras[0])
        transition_cameras = transition_cameras[:-1]

        print(f"🎬 Generated {len(transition_cameras)} transition frames + {len(orbit_cameras)} orbit frames = {len(transition_cameras) + len(orbit_cameras)} total")

        # combine: transition frames + orbit frames
        cameras = transition_cameras + orbit_cameras
    else:
        # if no transition frames, use orbit cameras directly
        # print(f"🎬 Generated {len(orbit_cameras)} pure orbit frames (no transition)")
        cameras = orbit_cameras

    # verify camera setup
    if len(cameras) > 0:
        if transition_frames > 0:
            # with transition frames, the first frame should be current_camera
            center_pos_original = center_camera.get_camera_center()
            center_pos_first = cameras[0].get_camera_center()
            is_same = torch.allclose(center_pos_original, center_pos_first, atol=1e-4)
            # print(f"✅ cameras[0] is current_camera: {is_same}")
            # if not is_same:
            #     print(f"   Original center: {center_pos_original.squeeze()}")
            #     print(f"   First camera center: {center_pos_first.squeeze()}")
        else:
            # without transition frames, the first frame is the first orbit frame (not current_camera)
            # print(f"✅ Pure orbit mode: cameras[0] is orbit start (not current_camera)")
            pass

    return cameras
    # return [center_camera] * len(cameras)


print_error = ARGS.debug  # print render-thread exceptions (skipped silently otherwise)
render_stop = False       # set on exit to stop the render thread
_render_errors_reported = set()


def _render_gate():
    """Context manager around one preview frame, yielding whether it may be drawn.

    With model services this is svc.render_frame(): False while a worker uses the main GPU under the
    exclusive policy, and a worker lease waits for the frame being drawn before it parks the main
    models. Without a service manager every frame is allowed."""
    render_frame = getattr(svc, "render_frame", None) if svc is not None else None
    if render_frame is None:
        return contextlib.nullcontext(True)
    return render_frame()


def _report_render_error(e):
    """Render-thread exceptions skip the frame; print each distinct one once (always with --debug)."""
    key = f"{type(e).__name__}: {e}"
    if print_error or key not in _render_errors_reported:
        _render_errors_reported.add(key)
        print(f"⚠️ render thread: {key}")
        if print_error:
            traceback.print_exc()


ORBIT_MAX_FAILED_FRAMES = 20  # orbit frames in a row that fail without progress before the orbit is abandoned
_orbit_failed_frames = 0


def _note_orbit_frame(failed, frame_before, error=None):
    """Count orbit frames that failed without advancing; after ORBIT_MAX_FAILED_FRAMES in a row,
    abandon the orbit and report the job as failed.

    An HQ-NVS / crack-fix orbit holds busy_job until the main loop has processed its frames, and
    the frames are handed over only when the orbit completes (hq_ready). A frame that fails every
    time (e.g. a generate_orbit_cameras from the config that raises, or a persistent CUDA error)
    would otherwise keep every later request rejected until the server is restarted."""
    global _orbit_failed_frames
    if not failed or not orbit_state.get('is_orbiting') or orbit_state.get('current_frame') != frame_before:
        _orbit_failed_frames = 0
        return
    _orbit_failed_frames += 1
    if _orbit_failed_frames < ORBIT_MAX_FAILED_FRAMES:
        return
    _orbit_failed_frames = 0
    job = {'high_quality': 'hq_nvs', 'crack_fix': 'crack_fix'}.get(orbit_state.get('mode'))
    reason = RuntimeError(f"orbit rendering failed {ORBIT_MAX_FAILED_FRAMES} times in a row "
                          f"({type(error).__name__}: {error})")
    print(f"❌ {reason}; orbit abandoned")
    reset_orbit_state()
    try:
        if job is not None and busy_job == job:
            fail_job(job, reason)
        else:
            emit_to_client('server-state', f"orbit preview stopped: {reason}")
    except Exception as e:  # never let the render thread die over a status message
        print(f"⚠️ could not report the abandoned orbit: {e}")


def render_current_scene():
    """Render thread: streams the current view (or the orbit preview / HQ-NVS / crack-fix orbit, whose
    frames it collects for the main loop) as JPEG 'frame' events."""
    global latest_frame, orbit_state

    while not render_stop:
        time.sleep(0.05)
        if gaussians is None:
            continue
        # Each frame is drawn inside the render gate: under the exclusive GPU policy a model worker
        # that needs the main GPU waits for it, and no frame is drawn while the worker runs.
        with _render_gate() as allowed:
            if not allowed:
                continue
            # The main loop holds scene_lock while it changes the scene: skip the frame instead of
            # rendering a half-updated model.
            if not scene_lock.acquire(timeout=0.05):
                continue
            frame_before = orbit_state.get('current_frame')
            try:
                _draw_frame()
                _note_orbit_frame(False, frame_before)
            except Exception as e:
                _report_render_error(e)
                _note_orbit_frame(True, frame_before, e)
            finally:
                scene_lock.release()

        if latest_frame is not None and client_id is not None:
            try:
                _send_frame(latest_frame)
            except Exception as e:  # keep the render thread alive (e.g. the client just left)
                _report_render_error(e)


def _draw_frame():
    """Render the current view, or the next orbit camera, into latest_frame (scene_lock held)."""
    global latest_frame, orbit_state
    if scene_lock.suspended_by is not None and gaussians.delete_mask_all.any():
        # A job is waiting for a model service: render() would prune the points it marked for
        # deletion, so leave the model alone until the job continues.
        return
    with torch.no_grad():
        if orbit_state['is_orbiting']:
            # Get orbit cameras
            if orbit_state['cameras'] is None:
                # Use the current camera position and orientation
                current_camera = kf_gen.get_camera_by_js_view_matrix(
                    view_matrix_wonder, 
                    xyz_scale=xyz_scale, 
                    fx_wonder=fx_wonder, 
                    fy_wonder=fy_wonder
                )
                # Different parameters based on mode
                if orbit_state['mode'] == 'high_quality':
                    # High-quality NVS: with transition frames for smooth video (Gen3C: 121 frames)
                    orbit_state['cameras'] = generate_orbit_cameras(
                        current_camera, gaussians, opt, xyz_scale, 
                        n_frames=121, transition_frames=21
                    )
                elif orbit_state['mode'] == 'crack_fix':
                    # Crack fixing mode: same as HQ mode but shorter
                    orbit_state['cameras'] = generate_orbit_cameras(
                        current_camera, gaussians, opt, xyz_scale, 
                        n_frames=49, transition_frames=10
                    )
                else:
                    # Normal orbit: no transition frames
                    orbit_state['cameras'] = generate_orbit_cameras(
                        current_camera, gaussians, opt, xyz_scale, 
                        n_frames=22, transition_frames=0
                    )
                orbit_state['current_frame'] = 0

            # Get current orbit camera
            current_camera = orbit_state['cameras'][orbit_state['current_frame']]

            # 3DGS rendering; in high-quality / crack-fix mode the frames and hole masks are collected
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
            render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)

            # Collect frame and mask
            image = render_pkg['render']
            mask = (render_pkg["final_opacity"].detach().cpu()[0] < 0.8).float().detach().cpu()
            image = image.squeeze().detach().cpu().permute(1, 2, 0)
            if orbit_state['mode'] == 'high_quality' or orbit_state['mode'] == 'crack_fix':
                orbit_state['collected_frames'].append(image)
                orbit_state['collected_masks'].append(mask)

            rendered_img = render_pkg['render']
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image
            # Update progress
            if orbit_state['mode'] in ('high_quality', 'crack_fix'):
                socketio.emit('server-state', f'Collecting frames: {len(orbit_state["collected_frames"])}/{len(orbit_state["cameras"])}', room=client_id)
            else:
                socketio.emit('server-state', f'Orbit preview: {orbit_state["current_frame"] + 1}/{len(orbit_state["cameras"])}', room=client_id)

            orbit_state['current_frame'] = (orbit_state['current_frame'] + 1) % len(orbit_state['cameras'])

            # If we've completed a full orbit, stop orbiting
            if orbit_state['current_frame'] == 0:
                orbit_state['is_orbiting'] = False

                # Set flag for high quality processing in main thread
                if orbit_state['mode'] == 'high_quality':
                    print("🎬 Orbit complete, setting flag for main thread processing...")
                    orbit_state['hq_ready'] = True
                    # Don't clear cameras yet - will be cleared after HQ processing
                elif orbit_state['mode'] == 'crack_fix':
                    print("🔧 Orbit complete, setting flag for crack fixing processing...")
                    orbit_state['hq_ready'] = True
                    # Don't clear cameras yet - will be cleared after crack fix processing
                else:
                    # Normal mode - clear cameras immediately
                    orbit_state['cameras'] = None
                    socketio.emit('server-state', 'Orbit preview finished', room=client_id)
        else:
            # Normal rendering of the current view
            current_camera = kf_gen.get_camera_by_js_view_matrix(
                view_matrix_wonder, 
                xyz_scale=xyz_scale, 
                fx_wonder=fx_wonder, 
                fy_wonder=fy_wonder
            )
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_camera, xyz_scale=xyz_scale, config=config)
            render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
            rendered_img = render_pkg['render']
            rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image


def _send_frame(frame):
    """JPEG-encode a rendered BGR frame (longest edge <= MAX_IMAGE_SIZE) and send it as 'frame'."""
    processed_frame = frame
    if ENABLE_RESOLUTION_SCALING:
        height, width = frame.shape[:2]
        if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
            # scale to MAX_IMAGE_SIZE pixels on the longest edge while maintaining aspect ratio
            scale = MAX_IMAGE_SIZE / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            processed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_COMPRESSION_QUALITY]
    success, encoded_img = cv2.imencode('.jpg', processed_frame, encode_params)
    if success:
        emit_to_client('frame', encoded_img.tobytes())
    else:
        print("⚠️ Image encoding failed")


# simple thread restart function


@torch.no_grad()
def compute_3D_filter(self, cameras, initialize_scaling=False):
    print("Computing 3D filter")
    #TODO consider focal length and image width, zoom in
    # self = gaussians
    xyz = self.get_xyz
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    # print("dis shape", distance.shape)
    valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        # print(all_same)  # True

    for idx,camera in enumerate(cameras):

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
        # print(valid.max(),valid_depth.max(),in_screen.max())
        # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
        distance[valid] = torch.min(distance[valid], other=z[valid])

        valid_points = torch.logical_or(valid_points, valid)
            # print(focal_length.shape)
        screen_normal = torch.tensor([[0, 0, -1]], device=xyz.device, dtype=torch.float32)
        point_normals_in_screen = rotation2normal(self.get_rotation) @ R
        point_normals_in_screen_xoz = F.normalize(point_normals_in_screen[:, [0, 2]], dim=1)
        screen_normal_xoz = F.normalize(screen_normal[:, [0, 2]], dim=1)
        cos_xz = torch.sum(point_normals_in_screen_xoz * screen_normal_xoz, dim=1)
        # print(cos_xz.shape)
        # assert torch.all(cos_xz >= 0), "All normals should be in the same direction of the screen normal. Current min value: {}".format(cos_xz.min())
        point_normals_in_screen_yoz = F.normalize(point_normals_in_screen[:, [1, 2]], dim=1)
        screen_normal_yoz = F.normalize(screen_normal[:, [1, 2]], dim=1)
        cos_yz = torch.sum(point_normals_in_screen_yoz * screen_normal_yoz, dim=1)

    try:
        if (~valid_points).max() == True:
            print((~valid_points).max(),valid_points.max())
            distance[~valid_points] = distance[valid_points].max()
    except:
        pass

    #TODO remove hard coded value
    #TODO box to gaussian transform
    # print(distance.shape,xyz.shape)
    filter_3D = distance / self._focal_length
    self.filter_3D = filter_3D[..., None]

    x_scale = distance / self._focal_length / cos_xz.clamp(min=1e-1)
    y_scale = distance / self._focal_length / cos_yz.clamp(min=1e-1)
    # import pdb ; pdb.set_trace()
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


def render_zoomin_rough_video3(cameras, gaussians, editing_prompt = None):
    # clear frame directories
    clear_frames_directories()

    # select 3 keyframes for high-quality processing
    idxs = np.linspace(0, len(cameras)-1, 3).astype(np.int32)

    # generate the first frame (smallest focal length)
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cameras[idxs[0]], xyz_scale=xyz_scale, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
    img_0 = render_pkg['render']
    plt.imsave(f"./cache/img_0.png", (img_0.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),cmap = "gray" )
    plt.imsave(f"./frames/saved_frames/input.png", (img_0.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),cmap = "gray" )

    # generate the second frame (medium focal length) with CoZ super-resolution
    img_1 = zoom_image_by_focal_change(img_0, focal_1=cameras[idxs[0]].K[0,0,0], focal_2=cameras[idxs[1]].K[0,0,0])
    plt.imsave(f"./cache/img_1.png", (img_1.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),cmap = "gray" )    
    call_coz_dual("./cache/img_0.png", "./cache/img_1.png", output_path=f"./cache/coz_output.png")
    img_1_high = plt.imread(f"./cache/coz_output.png")[:,:,:3]
    # img_0 = plt.imread(f"./cache/img_0.png")[:,:,:3]
    # generate the third frame (largest focal length) with CoZ super-resolution + Step1X edit
    img_2 = zoom_image_by_focal_change(img_1_high, focal_1=cameras[idxs[1]].K[0,0,0], focal_2=cameras[idxs[2]].K[0,0,0])
    plt.imsave(f"./cache/img_2.png", (img_2*255).astype(np.uint8),cmap = "gray" )  
    # img_2 = zoom_image_by_focal_change(img_0, focal_1=cameras[idxs[0]].K[0,0,0], focal_2=cameras[idxs[2]].K[0,0,0])
    # plt.imsave(f"./cache/img_2.png", (img_2*255).astype(np.uint8))      
    call_coz_dual("./cache/coz_output.png", "./cache/img_2.png", output_path=f"./cache/coz_output2.png")
    img_2_high = plt.imread("./cache/coz_output2.png")
    img_2_edit = None
    # if editing_prompt is not None:
    #     img_2_edit_path = call_step1x_edit(f"./example_images/street.jpg", "remove streetlight", output_path="./cache/step1x_output.png")
    #     img_2_edit = plt.imread(img_2_edit_path)

    # get key focal length values
    focal_0 = cameras[idxs[0]].K[0,0,0].item()
    focal_1 = cameras[idxs[1]].K[0,0,0].item()  
    focal_2 = cameras[idxs[2]].K[0,0,0].item()

    print(f"🎬 Rendering zoom video with {len(cameras)} frames")
    print(f"📏 Focal lengths: {focal_0:.1f} → {focal_1:.1f} → {focal_2:.1f}")
    print(f"🧅 Onion-style layering: img_0 (base) + img_1_high (middle) + img_2_high (center)")

    # generate corresponding frames for each camera
    video_frames = []
    os.makedirs("./cache/zoom_frames", exist_ok=True)

    for i, camera in tqdm(enumerate(cameras)):
        current_focal = camera.K[0,0,0].item()

        # base layer: img_0 - scale by focal length
        if isinstance(img_0, torch.Tensor):
            img_0_np = (img_0.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        else:
            img_0_np = (img_0 * 255).astype(np.uint8)

        h, w = img_0_np.shape[:2]

        # compute display size for img_0: focal_0/current_focal
        size_ratio = focal_0 / current_focal
        frame_np = zoom_image_by_focal_change(img_0_np, focal_0, current_focal)

        h, w = frame_np.shape[:2]
        frame_np = frame_np*1. / frame_np.max()
        # middle layer: img_1_high - always displayed, high priority
        if img_1_high is not None:
            # compute display size for img_1_high: focal_1/current_focal
            size_ratio =  current_focal / focal_1

            if size_ratio <= 1.0:
                # shrink and display at center
                resized_h = int(h * size_ratio)
                resized_w = int(w * size_ratio)
                # resize img_1_high to computed size
                img_1_resized = cv2.resize(img_1_high, (resized_w, resized_h))

                # compute placement position (centered)
                start_y = (h - resized_h) // 2
                start_x = (w - resized_w) // 2

                # overlay onto frame
                frame_np[start_y:start_y+resized_h, start_x:start_x+resized_w] = img_1_resized
            else:
                # crop center portion to fill the frame
                frame_np = zoom_image_by_focal_change(img_1_high, focal_1, current_focal)
                frame_np = frame_np*1. / frame_np.max()

        # inner layer: img_2_high - always displayed, highest priority
        if img_2_high is not None:
            # compute display size for img_2_high: focal_2/current_focal  
            size_ratio =  current_focal / focal_2

            if size_ratio <= 1.0:
                # shrink and display at center
                resized_h = int(h * size_ratio)
                resized_w = int(w * size_ratio)

                # resize img_2_high to computed size
                img_2_resized = cv2.resize(img_2_high, (resized_w, resized_h))

                # compute placement position (centered)
                start_y = (h - resized_h) // 2
                start_x = (w - resized_w) // 2

                # overlay onto frame
                frame_np[start_y:start_y+resized_h, start_x:start_x+resized_w] = img_2_resized
            else:
                # crop center portion to fill the frame
                raise NotImplementedError("Not implemented")
                # frame_np = zoom_image_by_focal_change(img_2_high, focal_2, current_focal)

        frame_num = str(i).zfill(8)
        save_dir = "./frames"
        cv2.imwrite(os.path.join(save_dir, f"saved_frames/output/test_{frame_num}.png"), cv2.cvtColor((frame_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"saved_frames/frames/test_{frame_num}.png"), cv2.cvtColor((frame_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        video_frames.append(frame_np)


    # generate video
    video_frames = torch.from_numpy(np.stack(video_frames, axis=0))

    print("🎬 Generating zoom video...")
    video_path = "./cache/zoom_video.mp4"
    save_rough_video(video_path, video_frames)
    print(f"✅ Zoom video saved: {video_path}")

    return video_path, img_2_edit


@torch.no_grad()
def render_rough_video(cameras, gaussians, xyz_scale=xyz_scale, opacity_threshold=0.6):
    """
    Render video frame sequence (3DGS) and the hole masks (final opacity < opacity_threshold).

    Args:
        cameras: list of cameras
        gaussians: Gaussian model
        xyz_scale: coordinate scaling factor
        opacity_threshold: opacity threshold
    """
    import warnings


    # suppress all types of warnings
    warnings.filterwarnings("ignore")
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # suppress PyTorch-specific warnings  
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # suppress all warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pc_imgs = []
        masks = []

        for i in tqdm(range(len(cameras)), desc="Rendering frames (3DGS)"):
            next_camera = cameras[i]
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(next_camera, xyz_scale=xyz_scale, config=config)
            render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
            image = render_pkg['render']
            masks.append((render_pkg["final_opacity"].detach().cpu()[0]<opacity_threshold).float().detach().cpu())
            pc_imgs.append(image.squeeze().detach().cpu().permute(1,2,0))

        return pc_imgs, masks


def save_rough_video_frames(pc_imgs, masks, cameras, save_dir="./frames", input_camera=None, inpaint=False):
    """
    Process and save video frames with inpainting.

    Args:
        pc_imgs (list): List of rendered images
        masks (list): List of masks for each frame
        cameras (list): List of cameras (for orbit frames)
        save_dir (str): Directory to save the processed frames
        input_camera: Camera to use for input.png (if None, uses cameras[0])

    """
    global gaussians, opt, background, xyz_scale

    import numpy as np
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # Create save directories
    # os.makedirs(os.path.join(save_dir, "saved_frames/input"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "saved_frames/output"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "saved_frames/frames"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "saved_frames/masks"), exist_ok=True)

    # clear existing frames and masks
    clear_frames_directories()

    # Save input frame using the specified input_camera or cameras[0]
    camera_for_input = input_camera if input_camera is not None else cameras[0]
    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(camera_for_input, xyz_scale=xyz_scale, config=config)
    render_pkg = render(tdgs_cam, gaussians, opt, background, render_visible=True, config=config)
    image_s = render_pkg['render']
    plt.imsave(
        os.path.join(save_dir, "saved_frames/input.png"),
        (image_s.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),cmap = "gray" 
    )

    # Log which camera was used
    camera_source = "input_camera (user's view)" if input_camera is not None else "cameras[0] (transition start)"
    print(f"💾 Saved input.png using {camera_source}")

    # Verify cameras[0] is the expected starting camera
    if len(pc_imgs) > 0:
        if input_camera is not None:
            # In high-quality mode, cameras[0] should be close to input_camera
            center_input = input_camera.get_camera_center()
            center_first = cameras[0].get_camera_center()
            distance = torch.norm(center_input - center_first).item()
            print(f"📏 Distance between input_camera and cameras[0]: {distance:.6f}")
        else:
            print(f"📹 Using cameras[0] as input (normal mode)");

    # Process frames
    now_imgs = torch.stack(pc_imgs, dim=0)
    now_masks = torch.stack(masks, dim=0).float()[...,None].repeat(1,1,1,3)

    for i in tqdm(range(len(now_imgs))):
        # Process current frame
        img_now = (now_imgs[i]*(1-now_masks[i]))
        mask_now = now_masks[i][...,0].clone()

        # q = (img_now.numpy()*255).clip(0,255.).astype(np.uint8)
        # Apply inpainting
        if inpaint :
            q = cv2.inpaint(
                (img_now.numpy()*255).clip(0,255.).astype(np.uint8),
                (mask_now.float().numpy()*255).astype(np.uint8),
                3,
                cv2.INPAINT_TELEA
            )
        else :
            q = (img_now.numpy()*255).clip(0,255.).astype(np.uint8)

        # Save processed frame and mask
        u = (mask_now*255)[...,None].repeat(1,1,3).int().numpy().astype(np.uint8)

        # Save with padded zeros in filename
        frame_num = str(i).zfill(8)
        cv2.imwrite(os.path.join(save_dir, f"saved_frames/output/test_{frame_num}.png"), cv2.cvtColor(q, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"saved_frames/frames/test_{frame_num}.png"), cv2.cvtColor(q, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"saved_frames/masks/mask_{frame_num}.png"), u)    


if __name__ == "__main__":
    print("🚀 Starting the WonderZoom generation server...")
    args = ARGS

    def reload_config(example_config=None, base_config=None, image=None, name=None):
        """Merge the base and the example config (the example wins, as in the paper-era code) and
        resolve the generation settings. Configs may execute code (generate_orbit_cameras_code)."""
        global config, kf_gen, generate_orbit_cameras
        base_file = _resolve_cli_path(base_config or args.base_config)
        base = OmegaConf.load(base_file)
        if example_config is not None:
            example = OmegaConf.load(_resolve_cli_path(example_config))
        else:
            example = OmegaConf.create({})
        config = OmegaConf.merge(base, example)

        if image is not None:
            config['image_filepath'] = _resolve_cli_path(image)
            config['example_name'] = name or Path(image).stem
        elif name:
            config['example_name'] = name
        if not config.get('example_name'):
            config['example_name'] = Path(str(config.get('image_filepath', 'scene'))).stem
        # Paths inside the configs are relative to the repository root.
        image_filepath = config.get('image_filepath')
        if not image_filepath:
            raise ValueError("no input image: set image_filepath in the example config or pass --image")
        if not os.path.isabs(str(image_filepath)):
            image_filepath = str(WZ_ROOT / str(image_filepath))
        if not os.path.isfile(image_filepath):
            raise FileNotFoundError(f"input image not found: {image_filepath}")
        config['image_filepath'] = os.path.abspath(image_filepath)

        # Generation resolution (paper: 720x1088). Chain-of-Zoom's SD3 backbone needs multiples of 16.
        config['orig_H'] = int(config.get('gen_H', 720))
        config['orig_W'] = int(config.get('gen_W', 1088))
        if config['orig_H'] % 16 or config['orig_W'] % 16:
            raise ValueError(f"gen_H/gen_W must be multiples of 16, got {config['orig_H']}x{config['orig_W']}")

        if kf_gen is not None:
            kf_gen.config = config

        # dynamically load generate_orbit_cameras function (if defined in config)
        if 'generate_orbit_cameras_code' in config and config.generate_orbit_cameras_code:
            try:
                print(f"Loading custom generate_orbit_cameras function from config...")
                # create a local namespace to execute the function code
                local_namespace = {}
                exec(config.generate_orbit_cameras_code, globals(), local_namespace)
                # replace the global generate_orbit_cameras function
                generate_orbit_cameras = local_namespace['generate_orbit_cameras']
                print(f"Successfully loaded custom generate_orbit_cameras function")
            except Exception as e:
                print(f"Error loading custom generate_orbit_cameras function: {e}")
                print(f"Using default generate_orbit_cameras function")

    # --image uses config/custom_template.yaml unless --example_config was given explicitly.
    example_config_path = args.example_config
    if args.image is not None and args.example_config == build_arg_parser().get_default("example_config"):
        template = WZ_ROOT / "config" / "custom_template.yaml"
        example_config_path = str(template) if template.is_file() else None
    reload_config(example_config=example_config_path, base_config=args.base_config, image=args.image, name=args.name)

    services_cfg = load_services_config(
        main_path=_resolve_cli_path(args.services_config),
        local_path='config/services.local.yaml',
        overrides=dict(policy=args.gpu_policy, main_gpu=args.main_gpu, no_services=args.no_services or None))

    # GPT-4o prompts (features.gpt: auto = OPENAI_API_KEY is set) and their config fallbacks.
    _gpt4().configure(enabled=(services_cfg.get('features') or {}).get('gpt', 'auto'),
                      foreground_words=config.get('foreground_words'),
                      background_prompt=config.get('background_prompt'),
                      object_edit_prompt_template=config.get('object_edit_prompt_template'))
    FEATURES['gpt'] = _gpt4().gpt_available()

    if args.dry_run:
        print(json.dumps({
            "image_filepath": config['image_filepath'],
            "example_name": str(config['example_name']),
            "gen_H": config['orig_H'], "gen_W": config['orig_W'],
            "seed": config.get('seed'),
            "num_finetune_depth_model_steps": config.get('num_finetune_depth_model_steps'),
            "gen3c_prompt": gen3c_prompt(), "gen3c_prompt_hq": gen3c_prompt(hq=True),
            "gpu_policy": services_cfg.gpu.policy, "main_gpu": MAIN_GPU,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "WZ_PARENT_VISIBLE_DEVICES": os.environ.get("WZ_PARENT_VISIBLE_DEVICES"),
            "runs_dir": services_cfg.paths.runs_dir,
            "services": {name: {"enabled": bool(services_cfg.services[name].get("enabled")),
                                "python": services_cfg.services[name].get("python")}
                         for name in services_cfg.services},
            "gpt": FEATURES['gpt'],
            "objects": {"object_insertion": str(_feature_flag("object_insertion")),
                        "grounded_sam_missing": grounded_sam_requirements(),
                        "harmonization_missing": harmonization_requirements()},
        }, indent=1))
        print("dry run: configs and imports OK; exiting before loading any model")
        sys.exit(0)

    # Per-session work directory: runs/<example_name>/<YYYYmmdd-HHMMSS>/. The process chdirs into
    # it once every path has been resolved, so the relative './cache' and './frames' paths used
    # throughout this file land there.
    SESSION_DIR = os.path.join(services_cfg.paths.runs_dir, str(config['example_name']),
                               datetime.now().strftime("%Y%m%d-%H%M%S"))
    for sub in SESSION_SUBDIRS:
        os.makedirs(os.path.join(SESSION_DIR, sub), exist_ok=True)
    config['runs_dir'] = SESSION_DIR
    config['session_dir'] = SESSION_DIR
    OmegaConf.save(config, os.path.join(SESSION_DIR, "config.yaml"))
    print(f"📁 Session directory: {SESSION_DIR}")

    # Model services (worker processes). They start after the main models are loaded.
    svc = ServiceManager(services_cfg, SESSION_DIR, log=print)
    FEATURES['gen3c'] = svc.enabled('gen3c')
    FEATURES['coz'] = svc.enabled('coz')
    # Object insertion needs the step1x service plus GroundingDINO / SAM and their checkpoints.
    detect_object_features()
    for name in ('gen3c', 'coz', 'step1x'):
        if not svc.enabled(name):
            print(f"⚠️ {name} service disabled (no interpreter registered or services.{name}.enabled is false)")

    # Exclusive GPU policy: the main models are parked in host RAM while a worker runs.
    svc.register_main_tenant(park_main_models, unpark_main_models, device=MAIN_GPU,
                             release_cache_fn=torch.cuda.empty_cache)
    main_exclusive = svc.arbiter.is_exclusive(MAIN_TENANT)
    if main_exclusive:
        # Same results up to floating-point differences; keeps GeometryCrafter's intermediates in host RAM.
        if OmegaConf.select(config, 'geometrycrafter.low_memory_usage', default=None) in (None, 'auto'):
            OmegaConf.update(config, 'geometrycrafter.low_memory_usage', True, force_add=True)
    print(f"🖥️ Main GPU: logical {MAIN_GPU} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}), "
          f"policy {'exclusive' if main_exclusive else 'resident'}")

    # Start the web server first so that the UI can show the loading status.
    set_status('loading', None, 'Loading models...')
    server_thread = threading.Thread(target=start_server, args=(args.host, args.port), daemon=True)
    server_thread.start()
    print(f"🌐 Open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/")

    # start rendering thread (it idles until the initial scene exists)
    render_thread = threading.Thread(target=render_current_scene)
    render_thread.daemon = True
    render_thread.start()
    # === model loaders ===
    def load_segment():
        global segment_processor, segment_model
        segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to("cuda")

    def load_normal_estimator():
        global normal_estimator
        normal_estimator = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16
        ).to(config["device"])

    def load_mask_generator():
        global mask_generator
        mask_generator = create_mask_generator_repvit(services_cfg.main_models.repvit_sam_checkpoint)

    def _on_sigterm(signum, frame):
        # `kill` / a scheduler: unwind like Ctrl+C so that the model workers are shut down.
        print("Terminated (SIGTERM); shutting down")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_sigterm)

    exit_code = 0
    try:
        t_models = time.time()
        load_mask_generator()
        load_segment()
        load_normal_estimator()
        # GroundedSAM for pull_foreground_depth_rewrite (configs such as sunflower.yaml) is loaded on
        # first use; without the GroundedSAM stack the option is skipped with a warning.
        kf_gen = MainModelsProxy(VideoGaussianProcessor(
            config=config, segment_model=segment_model, segment_processor=segment_processor,
            normal_estimator=normal_estimator, mask_generator=mask_generator, moge=None,
            grounded_sam=None if GROUNDED_SAM_MISSING else LazyGroundedSAM()))
        # Foreground words for that option: config foreground_words, else GPT-4o (None = option skipped).
        kf_gen.extract_fg_bg = extract_fg_bg_for_rewrite if FEATURES['gpt'] else None
        print(f"✅ All components loaded in {time.time() - t_models:.0f} s")

        # Resident GPUs load the workers in parallel with the initial scene; exclusive GPUs load them
        # one after the other, each parked right after it reports ready.
        svc.start_async()

        # Every path is absolute from here on: move into the session directory.
        os.chdir(SESSION_DIR)
        run(config, continue_flag=False)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        # Errors of individual requests are handled inside run(); this is start-up or the initial scene.
        traceback.print_exc()
        fail_job(server_status.get('job') or 'startup', e)
        if args.debug:
            debug_post_mortem(e.__traceback__)
        exit_code = 1
    finally:
        render_stop = True
        svc.shutdown()
    sys.exit(exit_code)
