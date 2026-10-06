#!/usr/bin/env python3
"""Check the WonderZoom environments: versions, CUDA, compiled kernels, checkpoints, HF access.

Each environment is checked with its own registered interpreter (config/services.local.yaml,
written by the install scripts; WZ_<ENV>_PYTHON overrides it). The driver needs only the Python
standard library and re-runs this file inside every environment with --inner.

Usage:
    python scripts/check_install.py                          # every registered environment
    python scripts/check_install.py --env main --objects
    python scripts/check_install.py --env gen3c --python /path/to/envs/wz-gen3c/bin/python
    HF_HUB_OFFLINE=1 python scripts/check_install.py --env main --load-models core
    python scripts/check_install.py --env main --no-checkpoints --no-gpu   # before downloading, on a login node

Per environment it checks: package versions against the pinned requirement files, CUDA
availability, the GPU architectures compiled into the CUDA extensions, a tiny kernel run for every
compiled extension (pytorch3d, rasterizer, simple-knn, GroundingDINO, apex, transformer-engine,
flash-attn, liger), the imports the workers need, the pinned third-party clones, checkpoint
presence (checkpoints dir and HF cache, no network) and Hugging Face access (gated SD3).
Exit status: 0 when nothing failed (warnings are allowed), 1 otherwise.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
ENVS = ("main", "gen3c", "coz", "step1x")
MARK = "@@WZCHECK@@ "


def external_dir():
    """$WZ_EXTERNAL_DIR (relative to the repository root, like paths.external_dir) or <repo>/external."""
    return os.path.join(ROOT, os.environ.get("WZ_EXTERNAL_DIR") or "external")


# Fallbacks when config/services.yaml cannot be loaded (no omegaconf in the driver's python).
DEFAULT_REPO_DIRS = {
    "main": "",
    "gen3c": os.path.join(external_dir(), "GEN3C"),
    "coz": os.path.join(external_dir(), "Chain-of-Zoom"),
    "step1x": os.path.join(external_dir(), "Step1X-Edit"),
}
SERVICE_PIN = {"gen3c": "GEN3C_COMMIT", "coz": "COZ_COMMIT", "step1x": "STEP1X_COMMIT"}
SERVICE_FILES = {
    "coz": ("third_party/chain_of_zoom/wonderzoom_coz.py", "wonderzoom_coz.py"),
    "step1x": ("third_party/step1x_edit/simple_step1x.py", "simple_step1x.py"),
}
HF_GROUPS = {"main": ("core",), "gen3c": ("gen3c",), "coz": ("coz",), "step1x": ("step1x",)}
CRITICAL = {"torch", "torchvision", "transformers", "diffusers", "accelerate", "huggingface-hub",
            "tokenizers", "numpy", "transformer-engine", "flash-attn", "megatron-core"}


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


# =======================================================================================
# Inner mode: runs inside one environment and reports results as marker lines on stdout.
# =======================================================================================
class Reporter(object):
    def __init__(self):
        self.counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}

    def emit(self, status, name, detail=""):
        self.counts[status] += 1
        sys.stdout.write(MARK + json.dumps({"status": status, "name": name, "detail": str(detail)}) + "\n")
        sys.stdout.flush()

    def ok(self, name, detail=""):
        self.emit("ok", name, detail)

    def warn(self, name, detail=""):
        self.emit("warn", name, detail)

    def fail(self, name, detail=""):
        self.emit("fail", name, detail)

    def skip(self, name, detail=""):
        self.emit("skip", name, detail)

    def run(self, name, fn, *args, **kwargs):
        """Run fn(); report its return string as OK, or the exception as FAIL."""
        try:
            detail = fn(*args, **kwargs)
        except Exception as e:  # every check reports instead of aborting the run
            msg = str(e).strip().splitlines()
            self.fail(name, "%s: %s" % (type(e).__name__, msg[0] if msg else ""))
            return False
        if detail is not False:
            self.ok(name, detail or "")
        return True


def _import(name):
    import importlib

    return importlib.import_module(name)


def _version(name):
    import importlib.metadata as md

    return md.version(name)


def load_pins():
    sys.path.insert(0, SCRIPTS)
    try:
        import prefetch_hf
    finally:
        sys.path.remove(SCRIPTS)
    return prefetch_hf, prefetch_hf.load_pins()


def check_pins_file(R, path, label, extra=None, strip_local=True):
    """Compare installed versions against 'name==version' lines of a requirements file."""
    import importlib.metadata as md

    pins = {}
    if path and os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;#]+)", line)
            if m:
                pins[m.group(1)] = m.group(2)
    elif path:
        R.fail("versions (%s)" % label, "%s not found" % path)
    pins.update(extra or {})
    mismatched, missing = [], []
    for name, want in sorted(pins.items()):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            missing.append("%s==%s" % (name, want))
            continue
        cmp_have = have.split("+")[0] if strip_local and "+" not in want else have
        if cmp_have != want:
            mismatched.append((name, have, want))
    if missing:
        R.fail("versions (%s)" % label, "not installed: " + ", ".join(missing))
    bad_critical = [m for m in mismatched if _norm(m[0]) in CRITICAL]
    other = [m for m in mismatched if _norm(m[0]) not in CRITICAL]
    if bad_critical:
        R.fail("versions (%s)" % label, "; ".join("%s %s (pinned %s)" % m for m in bad_critical))
    if other:
        R.warn("versions (%s)" % label, "; ".join("%s %s (pinned %s)" % m for m in other))
    if not missing and not mismatched:
        R.ok("versions (%s)" % label, "%d pinned packages match" % len(pins))


def check_vcs_commit(R, dist, want, label):
    import importlib.metadata as md

    try:
        info = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
    except md.PackageNotFoundError:
        R.fail(label, "%s is not installed" % dist)
        return
    commit = info.get("vcs_info", {}).get("commit_id")
    if commit == want:
        R.ok(label, "%s @%s" % (dist, want[:12]))
    elif commit:
        R.warn(label, "%s installed from %s, pinned %s" % (dist, commit[:12], want[:12]))
    else:
        R.warn(label, "%s was not installed from git; cannot confirm the pinned commit %s" % (dist, want[:12]))


def find_cuobjdump():
    import shutil

    for cand in (os.path.join(sys.prefix, "bin", "cuobjdump"),
                 os.path.join(os.environ.get("CUDA_HOME", "/nonexistent"), "bin", "cuobjdump")):
        if os.path.isfile(cand):
            return cand
    return shutil.which("cuobjdump")


def module_file(name):
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise ImportError("cannot find module %s" % name)
    return spec.origin


def check_archs(R, label, so_path, capability):
    """Report the GPU code compiled into SO_PATH and whether it can run on this GPU."""
    tool = find_cuobjdump()
    if not tool:
        R.skip("archs %s" % label, "cuobjdump not found")
        return
    archs = {"elf": set(), "ptx": set()}
    for kind in ("elf", "ptx"):
        out = subprocess.run([tool, "--list-" + kind, so_path], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True).stdout
        archs[kind].update(int(x) for x in re.findall(r"sm_(\d+)", out))
    if not archs["elf"] and not archs["ptx"]:
        R.warn("archs %s" % label, "no CUDA code found by cuobjdump in %s" % os.path.basename(so_path))
        return
    desc = "cubin %s%s" % (",".join("sm_%d" % a for a in sorted(archs["elf"])) or "-",
                           (" ptx " + ",".join("sm_%d" % a for a in sorted(archs["ptx"]))) if archs["ptx"] else "")
    if capability is None:
        R.ok("archs %s" % label, desc)
        return
    major, minor = capability
    dev = major * 10 + minor
    # A cubin runs on GPUs of the same major version and an equal or higher minor version; PTX
    # can be JIT-compiled for any newer GPU.
    cubin_ok = any(a // 10 == major and a % 10 <= minor for a in archs["elf"])
    ptx_ok = any(a <= dev for a in archs["ptx"])
    if dev in archs["elf"]:
        R.ok("archs %s" % label, desc)
    elif cubin_ok or ptx_ok:
        R.ok("archs %s" % label, "%s (runs on sm_%d via %s)" % (desc, dev, "compatible cubin" if cubin_ok else "PTX JIT"))
    else:
        R.fail("archs %s" % label, "%s: no code for this GPU (sm_%d); rebuild with TORCH_CUDA_ARCH_LIST "
               "including %d.%d" % (desc, dev, major, minor))


def check_torch_cuda(R, args):
    import torch

    R.ok("torch", "%s (built for CUDA %s, cuDNN %s)" % (torch.__version__, torch.version.cuda,
                                                       torch.backends.cudnn.version()))
    if args.no_gpu:
        R.skip("cuda", "--no-gpu")
        return None
    if not torch.cuda.is_available():
        R.fail("cuda", "torch.cuda.is_available() is False (no GPU visible, or driver too old for CUDA %s)"
               % torch.version.cuda)
        return None
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    R.ok("cuda", "%s, sm_%d%d, %.1f of %.1f GB free" % (torch.cuda.get_device_name(0), cap[0], cap[1],
                                                         free / 1e9, total / 1e9))
    return cap


def check_files(R, label, base, names, sizes=None):
    missing, wrong = [], []
    for n in names:
        p = os.path.join(base, n)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            missing.append(n)
        elif sizes and n in sizes and os.path.getsize(p) != sizes[n]:
            wrong.append("%s (%d bytes, expected %d)" % (n, os.path.getsize(p), sizes[n]))
    if missing:
        R.fail(label, "missing in %s: %s" % (base, ", ".join(missing)))
    elif wrong:
        R.fail(label, "wrong size: " + ", ".join(wrong))
    else:
        R.ok(label, "%d files in %s" % (len(names), base))


def check_checkpoints(R, args, groups):
    """Checkpoint presence without network: local dirs, the HF cache and checksums.sha256 files."""
    if getattr(args, "no_checkpoints", False):
        R.skip("checkpoints", "--no-checkpoints (%s)" % ", ".join(groups))
        return
    ph, pins = load_pins()
    try:
        from huggingface_hub import constants
        cache = constants.HF_HUB_CACHE
    except Exception:
        cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    if os.environ.get("TRANSFORMERS_CACHE"):
        R.warn("hf cache", "TRANSFORMERS_CACHE=%s is set; transformers looks there, not in %s"
               % (os.environ["TRANSFORMERS_CACHE"], cache))
    for item in ph.HF_ITEMS:
        if item["group"] not in groups:
            continue
        repo, rev = pins["HF_REPO_" + item["key"]], pins["HF_REV_" + item["key"]]
        label = "ckpt %s" % item["name"]
        dest = ph.item_destination(item, args.ckpt_dir)
        if dest is None:
            rtype = item.get("repo_type", "model")
            base = os.path.join(cache, "%ss--%s" % (rtype, repo.replace("/", "--")), "snapshots", rev)
        else:
            base = dest
        if not os.path.isdir(base):
            R.fail(label, "%s@%s not downloaded (%s); run scripts/download_checkpoints.sh --%s"
                   % (repo, rev[:10], base, item["group"]))
            continue
        present = []
        for dirpath, _, files in os.walk(base):
            for f in files:
                p = os.path.join(dirpath, f)
                if os.path.isfile(p) and os.path.getsize(p) > 0:  # follows cache symlinks to blobs
                    present.append(os.path.relpath(p, base).replace(os.sep, "/"))
        # 'require' lists the files the worker's start-up check needs when the download takes
        # the whole repository (allow=None).
        allow = item.get("require") or item.get("allow")
        if allow is None:
            ok = bool(present)
            missing = [] if ok else ["(any file)"]
        else:
            missing = [pat for pat in allow if not any(fnmatch.fnmatchcase(f, pat) for f in present)]
        if missing:
            R.fail(label, "%s@%s incomplete, missing %s; run scripts/download_checkpoints.sh --%s"
                   % (repo, rev[:10], ", ".join(missing), item["group"]))
        else:
            R.ok(label, "%s@%s (%d files)" % (repo, rev[:10], len(present)))
    for e in ph.load_checksums():
        if e["group"] not in groups:
            continue
        p = os.path.join(args.ckpt_dir, e["path"])
        label = "ckpt %s" % os.path.basename(e["path"])
        if not os.path.isfile(p):
            if e["url"].startswith("gdrive:"):
                R.warn(label, "%s missing (optional; Google Drive download, see download_checkpoints.sh)" % p)
            else:
                R.fail(label, "%s missing; run scripts/download_checkpoints.sh --%s" % (p, e["group"]))
        elif os.path.getsize(p) != e["size"]:
            R.fail(label, "%s has %d bytes, expected %d" % (p, os.path.getsize(p), e["size"]))
        else:
            R.ok(label, "%s (%d bytes)" % (p, e["size"]))


def hf_offline():
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes", "on")


def check_hf_access(R, args, gated_items=()):
    if args.offline or hf_offline():
        R.skip("hf access", "offline")
        return
    ph, pins = load_pins()
    try:
        import huggingface_hub as hh
    except ImportError:
        R.skip("hf access", "huggingface_hub not installed in this env")
        return
    try:
        hh.HfApi().repo_info(pins["HF_REPO_ONEFORMER"], revision=pins["HF_REV_ONEFORMER"], timeout=20)
        token = None
        try:
            token = hh.get_token()
        except Exception:
            pass
        R.ok("hf access", "%s reachable, token %s" % (hh.constants.ENDPOINT, "configured" if token else "not configured"))
    except Exception as e:
        R.warn("hf access", "cannot reach the Hub (%s); fine if every checkpoint is cached" % type(e).__name__)
        return
    for item in ph.HF_ITEMS:
        if item["name"] not in gated_items:
            continue
        repo, rev = pins["HF_REPO_" + item["key"]], pins["HF_REV_" + item["key"]]
        ok, why = ph.check_access(hh, repo, rev, item.get("repo_type", "model"), "model_index.json")
        if ok:
            R.ok("gated %s" % item["name"], "%s: %s" % (repo, why))
        else:
            R.warn("gated %s" % item["name"], "%s: %s. Needed only for downloading; accept the license on "
                   "https://huggingface.co/%s and log in" % (repo, why, repo))


# --- main env ------------------------------------------------------------------------------
def kernel_pytorch3d():
    import torch
    from pytorch3d.renderer import PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer
    from pytorch3d.structures import Pointclouds

    torch.manual_seed(0)
    dev = torch.device("cuda")
    pts = torch.rand(1, 100, 3, device=dev) - 0.5
    pts[..., 2] += 3.0
    raster = PointsRasterizer(cameras=PerspectiveCameras(device=dev),
                              raster_settings=PointsRasterizationSettings(image_size=32, radius=0.05,
                                                                          points_per_pixel=4))
    frags = raster(Pointclouds(points=pts))
    torch.cuda.synchronize()
    hit = frags.idx[..., 0] >= 0
    if frags.idx.shape != (1, 32, 32, 4) or not bool(hit.any()):
        raise RuntimeError("no point was rasterized; the CUDA kernel did not run correctly")
    # Every hit must report the depth of the point it hit.
    z = pts[0, frags.idx[..., 0][hit], 2]
    if not torch.allclose(frags.zbuf[..., 0][hit], z, atol=1e-4):
        raise RuntimeError("z-buffer does not match the rasterized points")
    return "point rasterization OK (%d pixels hit, depths match)" % int(hit.sum())


def kernel_rasterizer():
    import math

    import torch
    from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer

    torch.manual_seed(0)
    dev = torch.device("cuda")
    n, h, w, fov = 10, 64, 64, math.radians(60)
    znear, zfar, t = 0.01, 100.0, math.tan(fov / 2)
    proj = torch.zeros(4, 4)
    proj[0, 0] = proj[1, 1] = 1.0 / t
    proj[3, 2] = 1.0
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    view = torch.eye(4, device=dev)
    settings = GaussianRasterizationSettings(
        image_height=h, image_width=w, tanfovx=t, tanfovy=t, bg=torch.zeros(3, device=dev),
        scale_modifier=1.0, viewmatrix=view, projmatrix=(view @ proj.t().to(dev)),
        sh_degree=0, campos=torch.zeros(3, device=dev), prefiltered=False, debug=False)
    means = torch.rand(n, 3, device=dev) - 0.5
    means[:, 2] += 3.0
    out = GaussianRasterizer(settings)(
        means3D=means, means2D=torch.zeros(n, 3, device=dev), opacities=torch.full((n, 1), 0.8, device=dev),
        colors_precomp=torch.rand(n, 3, device=dev), scales=torch.full((n, 3), 0.1, device=dev),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(n, 1))
    torch.cuda.synchronize()
    color, radii = out[0], out[1]
    if tuple(color.shape) != (3, h, w) or not bool(torch.isfinite(color).all()):
        raise RuntimeError("bad output %s" % (tuple(color.shape),))
    visible = int((radii > 0).sum())
    # All 10 Gaussians lie inside the frustum; an empty image means the kernels did not run.
    if visible == 0 or float(color.max()) <= 0.0:
        raise RuntimeError("nothing was rendered; the CUDA kernels did not run correctly")
    return "rendered %d Gaussians (%d visible, max %.2f)" % (n, visible, float(color.max()))


def kernel_simple_knn():
    import torch
    from simple_knn._C import distCUDA2

    torch.manual_seed(0)
    pts = torch.rand(100, 3, device="cuda")
    d = distCUDA2(pts)
    torch.cuda.synchronize()
    # Reference: mean squared distance to the 3 nearest neighbours.
    dist2 = torch.cdist(pts, pts).pow(2)
    dist2.fill_diagonal_(float("inf"))
    ref = dist2.topk(3, largest=False).values.mean(1)
    if d.shape[0] != 100 or not bool(torch.isfinite(d).all()):
        raise RuntimeError("bad output")
    if not torch.allclose(d, ref, rtol=1e-3, atol=1e-6):
        raise RuntimeError("differs from the brute-force reference (max abs err %.3g)" % float((d - ref).abs().max()))
    return "distCUDA2 on 100 points matches brute force"


def kernel_groundingdino():
    import torch
    from groundingdino import _C
    from groundingdino.models.GroundingDINO.ms_deform_attn import multi_scale_deformable_attn_pytorch

    torch.manual_seed(0)
    dev = "cuda"
    shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long, device=dev)
    starts = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    value = torch.rand(1, int(shapes.prod(1).sum()), 2, 8, device=dev)
    loc = torch.rand(1, 3, 2, 2, 2, 2, device=dev)
    attn = torch.rand(1, 3, 2, 2, 2, device=dev)
    attn = attn / attn.sum((-1, -2), keepdim=True)
    out = _C.ms_deform_attn_forward(value, shapes, starts, loc, attn, 64)
    ref = multi_scale_deformable_attn_pytorch(value, shapes, loc, attn)
    torch.cuda.synchronize()
    # The extension only prints kernel launch errors, so compare with the PyTorch implementation.
    err = float((out - ref).abs().max())
    if tuple(out.shape) != (1, 3, 16) or not err < 1e-4:
        raise RuntimeError("ms_deform_attn differs from the PyTorch reference (max abs err %.3g): the kernel "
                           "did not run on this GPU" % err)
    return "ms_deform_attn forward matches the PyTorch reference"


def inner_main(R, args):
    ph, pins = load_pins()
    check_pins_file(R, os.path.join(ROOT, "requirements", "main.txt"), "requirements/main.txt")
    if args.objects:
        check_pins_file(R, os.path.join(ROOT, "requirements", "main-objects.txt"), "requirements/main-objects.txt")
    check_vcs_commit(R, "pytorch3d", pins["PYTORCH3D_COMMIT"], "pytorch3d commit")
    check_vcs_commit(R, "utils3d", pins["UTILS3D_COMMIT"], "utils3d commit")
    cap = None
    try:
        cap = check_torch_cuda(R, args)
    except Exception as e:
        R.fail("torch", "%s: %s" % (type(e).__name__, e))

    for p in (ROOT, os.path.join(ROOT, "GeometryCrafter"), os.path.join(ROOT, "MoGe")):
        sys.path.insert(0, p)
    modules = ["pytorch3d.renderer", "depth_diff_gaussian_rasterization_min", "simple_knn._C", "repvit_sam",
               "transformers", "diffusers", "accelerate", "huggingface_hub", "timm", "utils3d", "kornia", "cv2",
               "skimage.measure", "imageio", "imageio_ffmpeg", "av", "plyfile", "omegaconf", "einops",
               "flask", "flask_cors", "flask_socketio", "socketio",
               # vendored WonderZoom dependencies
               "marigold_lcm.marigold_pipeline", "geometrycrafter", "moge.model.v1"]
    for name in modules:
        R.run("import %s" % name, lambda n=name: (_import(n), "")[1])
    R.run("import OneFormer", lambda: (_import("transformers").OneFormerForUniversalSegmentation, "")[1])
    try:
        import repvit_sam
        if not os.path.realpath(repvit_sam.__file__).startswith(os.path.realpath(os.path.join(ROOT, "RepViT", "sam"))):
            R.warn("repvit_sam location", "imported from %s, not this checkout's RepViT/sam" % repvit_sam.__file__)
    except Exception:
        pass

    extensions = [("pytorch3d", "pytorch3d._C", kernel_pytorch3d),
                  ("rasterizer", "depth_diff_gaussian_rasterization_min._C", kernel_rasterizer),
                  ("simple_knn", "simple_knn._C", kernel_simple_knn)]
    if args.objects:
        extensions.append(("groundingdino", "groundingdino._C", kernel_groundingdino))
    for label, mod, kernel in extensions:
        try:
            so = module_file(mod)
        except Exception as e:
            R.fail("archs %s" % label, str(e))
            continue
        check_archs(R, label, so, cap)
        if cap is None:
            R.skip("kernel %s" % label, "no GPU")
        else:
            R.run("kernel %s" % label, kernel)

    if args.objects:
        inner_objects(R, args)
    glm = os.path.join(ROOT, "submodules", "depth-diff-gaussian-rasterization-min", "third_party", "glm", "glm", "glm.hpp")
    if os.path.isfile(glm):
        R.ok("glm headers", os.path.dirname(os.path.dirname(glm)))
    else:
        R.warn("glm headers", "missing (only needed to rebuild the rasterizer): bash scripts/setup_third_party.sh glm")
    groups = ("core", "objects") if args.objects else ("core",)
    check_checkpoints(R, args, groups)
    check_hf_access(R, args)
    if args.load_models == "core":
        load_core_models(R, args, ph, pins)


def inner_objects(R, args):
    for name in ("groundingdino", "groundingdino.models", "groundingdino.util.slconfig", "segment_anything",
                 "supervision", "pycocotools", "albumentations", "adamp", "openai", "gdown"):
        R.run("import %s" % name, lambda n=name: (_import(n), "")[1])
    try:
        import groundingdino
        cfg = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")
        if os.path.isfile(cfg):
            R.ok("groundingdino config", cfg)
        else:
            R.fail("groundingdino config", "%s missing; install GroundingDINO editable "
                   "(scripts/install_objects_optional.sh)" % cfg)
    except Exception:
        pass
    try:
        import segment_anything
        if os.path.isfile(os.path.join(os.path.dirname(segment_anything.__file__), "build_sam_hq.py")):
            R.ok("segment_anything copy", "Grounded-Segment-Anything version")
        else:
            R.warn("segment_anything copy", "not the Grounded-Segment-Anything copy (PyPI build?)")
    except Exception:
        pass
    inr = os.path.join(external_dir(), "INR-Harmonization")
    wrapper = os.path.join(inr, "inr_harmonization_model.py")
    if os.path.isfile(wrapper):
        sys.path.insert(0, inr)
        R.run("import inr_harmonization_model", lambda: (_import("inr_harmonization_model"), "")[1])
        sys.path.remove(inr)
    else:
        R.warn("INR-Harmonization", "%s missing; harmonization disabled (scripts/setup_third_party.sh inr)" % wrapper)


def load_core_models(R, args, ph, pins):
    """Instantiate every main-process model on the CPU with HF_HUB_OFFLINE=1 (proves no downloads)."""
    if not hf_offline():
        R.fail("load-models", "run with HF_HUB_OFFLINE=1 so that missing files are reported, not downloaded")
        return
    import gc

    import torch

    workspace = os.path.join(ROOT, "workspace")
    had_workspace = os.path.isdir(workspace)

    def rev(key):
        return pins["HF_REPO_" + key], pins["HF_REV_" + key]

    def oneformer():
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
        repo, r = rev("ONEFORMER")
        OneFormerProcessor.from_pretrained(repo, revision=r)
        OneFormerForUniversalSegmentation.from_pretrained(repo, revision=r)
        return repo

    def marigold():
        from marigold_lcm.marigold_pipeline import MarigoldNormalsPipeline
        repo, r = rev("MARIGOLD_NORMALS")
        MarigoldNormalsPipeline.from_pretrained(repo, revision=r, torch_dtype=torch.bfloat16)
        return repo

    def geometrycrafter():
        from geometrycrafter import (GeometryCrafterDiffPipeline, PMapAutoencoderKLTemporalDecoder,
                                     UNetSpatioTemporalConditionModelVid2vid)
        repo, r = rev("GEOMETRYCRAFTER")
        unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
            repo, subfolder="unet_diff", revision=r, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        PMapAutoencoderKLTemporalDecoder.from_pretrained(
            repo, subfolder="point_map_vae", revision=r, low_cpu_mem_usage=True, torch_dtype=torch.float32)
        svd, svd_rev = rev("SVD_XT")
        GeometryCrafterDiffPipeline.from_pretrained(svd, unet=unet, torch_dtype=torch.float16, variant="fp16",
                                                    revision=svd_rev)
        return "%s + %s" % (repo, svd)

    def moge():
        from moge.model.v1 import MoGeModel
        repo, r = rev("MOGE_VITL")
        MoGeModel.from_pretrained(repo, revision=r)
        return repo

    def repvit():
        from repvit_sam import sam_model_registry
        path = os.path.join(args.ckpt_dir, "repvit_sam.pt")
        sam_model_registry["repvit"](checkpoint=path)
        return path

    for name, fn in (("OneFormer", oneformer), ("Marigold normals", marigold), ("GeometryCrafter", geometrycrafter),
                     ("MoGe", moge), ("RepViT-SAM", repvit)):
        t0 = time.time()
        R.run("load %s" % name, lambda f=fn, t=t0: "%s (%.0f s, CPU, offline)" % (f(), time.time() - t))
        gc.collect()
    if not had_workspace and os.path.isdir(workspace):
        R.fail("workspace/cache", "loading created %s; a loader still uses cache_dir='workspace/cache'" % workspace)
    else:
        R.ok("workspace/cache", "not created")


# --- gen3c env -----------------------------------------------------------------------------
def kernel_apex():
    import torch
    from apex.normalization import FusedRMSNorm

    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    y = FusedRMSNorm(64, eps=1e-5).cuda()(x)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    torch.cuda.synchronize()
    if not torch.allclose(y, ref, atol=1e-4):
        raise RuntimeError("FusedRMSNorm differs from the reference (max abs err %.3g)" % float((y - ref).abs().max()))
    return "FusedRMSNorm forward matches the reference"


def kernel_te():
    import torch
    import torch.nn.functional as F
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    attn = te.DotProductAttention(num_attention_heads=2, kv_channels=64, attn_mask_type="no_mask", qkv_format="bshd")
    out = attn(q, k, v).float()
    ref = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                         v.transpose(1, 2).float()).transpose(1, 2).reshape(1, 128, 128)
    lin = te.Linear(64, 64, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    y = lin(x).float()
    y_ref = F.linear(x.float(), lin.weight.float(), lin.bias.float())
    torch.cuda.synchronize()
    if not torch.allclose(out, ref, atol=3e-2, rtol=3e-2):
        raise RuntimeError("DotProductAttention differs from SDPA (max abs err %.3g)" % float((out - ref).abs().max()))
    if not torch.allclose(y, y_ref, atol=5e-2, rtol=5e-2):
        raise RuntimeError("te.Linear differs from F.linear (max abs err %.3g)" % float((y - y_ref).abs().max()))
    return "DotProductAttention and Linear (bf16) match PyTorch"


def inner_gen3c(R, args):
    check_pins_file(R, os.path.join(args.repo_dir, "requirements.txt"), "GEN3C requirements.txt",
                    extra={"transformer_engine": "1.12.0"})
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.isdir(cuda_home):
        import glob
        if glob.glob(os.path.join(cuda_home, "lib*", "libnvrtc.so*")):
            R.ok("CUDA_HOME", "%s (has libnvrtc for transformer-engine)" % cuda_home)
        else:
            R.warn("CUDA_HOME", "%s has no lib*/libnvrtc.so*; transformer-engine may fail to JIT" % cuda_home)
    else:
        R.fail("CUDA_HOME", "not set; the Gen3C worker sets CUDA_HOME=<env prefix>")
    cap = None
    try:
        cap = check_torch_cuda(R, args)
    except Exception as e:
        R.fail("torch", "%s: %s" % (type(e).__name__, e))
    for name in ("megatron.core", "transformer_engine.pytorch", "amp_C", "apex.normalization"):
        R.run("import %s" % name, lambda n=name: (_import(n), "")[1])
    R.run("import Gen3cPipeline",
          lambda: (_import("cosmos_predict1.diffusion.inference.gen3c_pipeline").Gen3cPipeline, "")[1])
    try:
        import importlib.util
        if importlib.util.find_spec("moge") is not None:
            R.ok("moge", "installed but not needed by the worker")
    except Exception:
        pass
    for label, mod in (("apex amp_C", "amp_C"), ("apex fused_layer_norm", "fused_layer_norm_cuda"),
                       ("transformer_engine_torch", "transformer_engine_torch")):
        try:
            check_archs(R, label, module_file(mod), cap)
        except Exception as e:
            R.fail("archs %s" % label, str(e))
    if cap is None:
        R.skip("kernel apex", "no GPU")
        R.skip("kernel transformer-engine", "no GPU")
    else:
        R.run("kernel apex", kernel_apex)
        R.run("kernel transformer-engine", kernel_te)
    check_checkpoints(R, args, HF_GROUPS["gen3c"])


# --- coz env -------------------------------------------------------------------------------
def kernel_basic():
    import torch

    x = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(8, 4, 3, 3, device="cuda", dtype=torch.float16) * 0.1
    y = torch.nn.functional.conv2d(x, w)
    m = torch.randn(64, 64, device="cuda", dtype=torch.float16) * 0.1
    z = m @ m
    torch.cuda.synchronize()
    if not (torch.isfinite(y).all() and torch.isfinite(z).all()):
        raise RuntimeError("non-finite output")
    return "fp16 conv2d (cuDNN) + matmul OK"


def inner_coz(R, args):
    check_pins_file(R, os.path.join(args.repo_dir, "requirements.txt"), "Chain-of-Zoom requirements.txt")
    cap = None
    try:
        cap = check_torch_cuda(R, args)
    except Exception as e:
        R.fail("torch", "%s: %s" % (type(e).__name__, e))
    for name in ("diffusers", "transformers", "peft", "qwen_vl_utils", "osediff_sd3", "wonderzoom_coz"):
        R.run("import %s" % name, lambda n=name: (_import(n), "")[1])
    R.run("import Qwen2.5-VL", lambda: (_import("transformers").Qwen2_5_VLForConditionalGeneration,
                                        _import("qwen_vl_utils").process_vision_info, "")[2])
    if cap is None:
        R.skip("kernel torch", "no GPU")
    else:
        R.run("kernel torch", kernel_basic)
    check_files(R, "ckpt CoZ SR LoRA/VAE", args.repo_dir,
                ["ckpt/SR_LoRA/model_20001.pkl", "ckpt/SR_VAE/vae_encoder_20001.pt"],
                sizes={"ckpt/SR_LoRA/model_20001.pkl": 8111108, "ckpt/SR_VAE/vae_encoder_20001.pt": 69346330})
    check_checkpoints(R, args, HF_GROUPS["coz"])
    check_hf_access(R, args, gated_items=("sd3_medium",))


# --- step1x env ----------------------------------------------------------------------------
def kernel_flash_attn():
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_func

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = flash_attn_func(q, k, v).float()
    ref = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                         v.transpose(1, 2).float()).transpose(1, 2)
    torch.cuda.synchronize()
    if not torch.allclose(out, ref, atol=3e-2, rtol=3e-2):
        raise RuntimeError("flash_attn_func differs from SDPA (max abs err %.3g)" % float((out - ref).abs().max()))
    return "flash_attn_func (bf16) matches SDPA"


def kernel_liger():
    import torch
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    torch.manual_seed(0)
    x = torch.randn(2, 64, device="cuda")
    y = LigerRMSNorm(64, eps=1e-6).cuda()(x)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.cuda.synchronize()
    if not torch.allclose(y, ref, atol=1e-4):
        raise RuntimeError("LigerRMSNorm differs from the reference (max abs err %.3g)" % float((y - ref).abs().max()))
    return "LigerRMSNorm (Triton) matches the reference"


def inner_step1x(R, args):
    check_pins_file(R, os.path.join(ROOT, "requirements", "step1x.txt"), "requirements/step1x.txt",
                    extra={"torch": "2.7.1", "torchvision": "0.22.1", "flash_attn": "2.7.4.post1"})
    cap = None
    try:
        cap = check_torch_cuda(R, args)
    except Exception as e:
        R.fail("torch", "%s: %s" % (type(e).__name__, e))
    for name in ("flash_attn", "flash_attn_2_cuda", "xfuser", "transformers", "qwen_vl_utils"):
        R.run("import %s" % name, lambda n=name: (_import(n), "")[1])

    def fa2():
        from transformers.utils import is_flash_attn_2_available
        if not is_flash_attn_2_available():
            raise RuntimeError("transformers does not detect flash-attn 2")
        return ""
    R.run("flash-attn in transformers", fa2)
    if cap is None:
        R.skip("import simple_step1x", "liger_kernel needs a visible GPU at import time")
    else:
        R.run("import liger_kernel", lambda: (_import("liger_kernel"), "")[1])
        R.run("import simple_step1x", lambda: (_import("simple_step1x"), "")[1])
    try:
        check_archs(R, "flash_attn", module_file("flash_attn_2_cuda"), cap)
    except Exception as e:
        R.fail("archs flash_attn", str(e))
    if cap is None:
        R.skip("kernel flash-attn", "no GPU")
        R.skip("kernel liger", "no GPU")
    else:
        R.run("kernel flash-attn", kernel_flash_attn)
        R.run("kernel liger", kernel_liger)
    check_checkpoints(R, args, HF_GROUPS["step1x"])


def inner(args):
    import platform
    import warnings

    warnings.filterwarnings("ignore")
    R = Reporter()
    v = sys.version_info
    (R.ok if v[:2] == (3, 10) else R.warn)("python", "%d.%d.%d at %s (%s)" % (v[0], v[1], v[2], sys.executable,
                                                                           platform.machine()))
    fn = {"main": inner_main, "gen3c": inner_gen3c, "coz": inner_coz, "step1x": inner_step1x}[args.inner]
    try:
        fn(R, args)
    except Exception as e:
        R.fail("checker", "%s: %s" % (type(e).__name__, e))
    sys.stdout.write(MARK + json.dumps({"done": True, "counts": R.counts}) + "\n")
    sys.stdout.flush()
    return 0


# =======================================================================================
# Driver mode (standard library only)
# =======================================================================================
def load_register_env():
    sys.path.insert(0, SCRIPTS)
    try:
        import register_env
    finally:
        sys.path.remove(SCRIPTS)
    return register_env


def read_pins_stdlib():
    pins = {}
    with open(os.path.join(ROOT, "third_party", "pins.env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                pins[k] = v
    return pins


def services_config():
    """config/services.yaml merged with services.local.yaml via services.load_services_config()
    (absolute paths, interpolations resolved) as a plain dict; None when omegaconf is missing."""
    sys.path.insert(0, ROOT)
    try:
        from omegaconf import OmegaConf
        from services.config import load_services_config

        return OmegaConf.to_container(load_services_config(), resolve=True)
    except Exception:
        return None
    finally:
        sys.path.remove(ROOT)


def configured(cfg, *keys):
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]
    return node


def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, universal_newlines=True, check=True).stdout.strip()
    except Exception:
        return None


def driver_checks(name, repo_dir, results):
    """Checks that need no environment: pinned clones and WonderZoom's copied files."""
    def add(status, check, detail):
        results.append({"status": status, "name": check, "detail": detail})

    pins = read_pins_stdlib()
    if name in SERVICE_PIN:
        want = pins.get(SERVICE_PIN[name])
        head = git_head(repo_dir) if os.path.isdir(repo_dir) else None
        if not os.path.isdir(repo_dir):
            add("fail", "source %s" % os.path.basename(repo_dir),
                "%s missing; run scripts/setup_third_party.sh %s" % (repo_dir, name))
        elif head == want:
            add("ok", "source %s" % os.path.basename(repo_dir), "%s @%s (pinned)" % (repo_dir, want[:12]))
        else:
            add("fail", "source %s" % os.path.basename(repo_dir),
                "%s is at %s, pinned %s; run scripts/setup_third_party.sh %s" % (repo_dir, head, want, name))
    if name in SERVICE_FILES:
        src_rel, dst_name = SERVICE_FILES[name]
        src, dst = os.path.join(ROOT, src_rel), os.path.join(repo_dir, dst_name)
        if not os.path.isfile(dst):
            add("fail", dst_name, "missing in %s; run scripts/setup_third_party.sh %s" % (repo_dir, name))
        elif os.path.isfile(src) and open(src, "rb").read() != open(dst, "rb").read():
            add("warn", dst_name, "differs from %s; re-run scripts/setup_third_party.sh %s" % (src_rel, name))
        else:
            add("ok", dst_name, dst)


STATUS_TAG = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}


def print_result(r):
    print("  %s %-28s %s" % (STATUS_TAG[r["status"]], r["name"], r["detail"]))
    sys.stdout.flush()


def load_services_base():
    """services/base.py loaded as a standalone module (standard library only), so the driver does
    not import the services package (which needs omegaconf)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_wz_services_base", os.path.join(ROOT, "services", "base.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker_dry_import(name, python, repo_dir, args):
    """Run '<env python> services/workers/<name>_worker.py --dry-import' in exactly the environment
    the ServiceManager gives the worker (services.base.build_worker_env). Returns one result."""
    check = "worker %s --dry-import" % name
    script = os.path.join(ROOT, "services", "workers", "%s_worker.py" % name)
    if not os.path.isfile(script):
        return {"status": "fail", "name": check, "detail": "%s is missing" % script}
    if not os.path.isdir(repo_dir):
        return {"status": "skip", "name": check, "detail": "%s missing; run scripts/setup_third_party.sh %s"
                % (repo_dir, name)}
    if args.no_gpu and name == "step1x":
        return {"status": "skip", "name": check, "detail": "--no-gpu (liger_kernel needs a visible GPU at import)"}
    try:
        base = load_services_base()
        cuda_home = base.auto_cuda_home(python) if name == "gen3c" else None
        if name == "gen3c" and cuda_home is None:
            cuda_home = base.env_prefix(python)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
        if args.no_gpu:
            visible = ""
        env = base.build_worker_env(python, repo_dir, visible, cuda_home=cuda_home)
    except Exception as e:
        return {"status": "fail", "name": check, "detail": "cannot build the worker environment: %s" % e}
    try:
        p = subprocess.run([python, script, "--dry-import"], cwd=repo_dir, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, errors="replace",
                           timeout=min(args.timeout, 900))
    except subprocess.TimeoutExpired:
        return {"status": "fail", "name": check, "detail": "timed out"}
    msg = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                msg = json.loads(line)
            except ValueError:
                continue
    if p.returncode == 0 and msg and msg.get("event") == "dry-import" and msg.get("ok"):
        info = msg.get("info") or {}
        return {"status": "ok", "name": check, "detail": "imports fine (%ss)" % info.get("seconds", "?")}
    if msg and msg.get("event") == "fatal":
        detail = msg.get("error", "fatal")
    else:
        detail = "exit code %s; stderr: %s" % (p.returncode, " | ".join(p.stderr.strip().splitlines()[-3:]))
    return {"status": "fail", "name": check, "detail": detail}


def run_env(name, python, repo_dir, args):
    """Run the inner checks for one environment; return the list of results."""
    results = []
    driver_checks(name, repo_dir, results)
    for r in results:
        print_result(r)
    env = dict(os.environ)
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(python)))
    env["PATH"] = os.path.join(prefix, "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("PYTHONPATH", None)
    # Same as the service workers and scripts/run_server.sh: torch must load the CUDA/cuDNN
    # libraries of its own wheels, not those of a system toolkit on LD_LIBRARY_PATH.
    if name != "main" or os.environ.get("WZ_KEEP_LD_LIBRARY_PATH", "0") != "1":
        env.pop("LD_LIBRARY_PATH", None)
    if name != "main":
        env["PYTHONPATH"] = repo_dir
    if name == "gen3c":
        env["CUDA_HOME"] = prefix
    if args.load_models:
        env["HF_HUB_OFFLINE"] = "1"
    cmd = [python, "-u", os.path.abspath(__file__), "--inner", name, "--repo-dir", repo_dir,
           "--ckpt-dir", args.ckpt_dir]
    for flag in ("objects", "no_gpu", "offline", "no_checkpoints"):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    if args.load_models:
        cmd += ["--load-models", args.load_models]
    cwd = repo_dir if os.path.isdir(repo_dir) else ROOT
    tail, done = [], [None]
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, errors="replace")
    except OSError as e:
        results.append({"status": "fail", "name": "interpreter", "detail": "cannot run %s: %s" % (python, e)})
        print_result(results[-1])
        return results
    timer = threading.Timer(args.timeout, proc.kill)
    timer.start()
    try:
        for line in proc.stdout:
            if line.startswith(MARK):
                msg = json.loads(line[len(MARK):])
                if msg.get("done"):
                    done[0] = msg
                    continue
                results.append(msg)
                print_result(msg)
            else:
                tail.append(line.rstrip("\n"))
                del tail[:-40]
                if args.verbose:
                    print("      | " + line.rstrip("\n"))
        rc = proc.wait()
    finally:
        timer.cancel()
    if done[0] is None:
        detail = "checker exited with code %s before finishing%s" % (
            rc, " (timeout after %d s)" % args.timeout if rc in (-9, 137) else "")
        results.append({"status": "fail", "name": "checker", "detail": detail})
        print_result(results[-1])
        for line in tail[-15:]:
            print("      | " + line)
    if name != "main":
        results.append(worker_dry_import(name, python, repo_dir, args))
        print_result(results[-1])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", action="append", choices=ENVS, help="environment to check (repeatable)")
    parser.add_argument("--all", action="store_true", help="check all four environments")
    parser.add_argument("--objects", action="store_true", help="also check the optional object-insertion stack")
    parser.add_argument("--python", help="interpreter to use for the single --env given")
    parser.add_argument("--repo-dir", help="third-party source dir for the single --env given")
    parser.add_argument("--ckpt-dir", default=None, help="checkpoints dir (default: $WZ_CKPT_DIR or checkpoints/)")
    parser.add_argument("--no-gpu", action="store_true", help="skip CUDA and kernel checks")
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="skip the checkpoint checks (before scripts/download_checkpoints.sh has run)")
    parser.add_argument("--load-models", choices=("core",), help="main env: load the main-process models offline")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per environment (default: 1800)")
    parser.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    parser.add_argument("--verbose", action="store_true", help="show the checked libraries' own output")
    parser.add_argument("--inner", choices=ENVS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.inner:
        args.repo_dir = args.repo_dir or ROOT
        return inner(args)

    cfg = services_config()
    if configured(cfg, "paths", "external_dir"):
        # The inner checks (INR-Harmonization) look for the clones where the runtime does.
        os.environ["WZ_EXTERNAL_DIR"] = configured(cfg, "paths", "external_dir")
    ckpt = args.ckpt_dir or os.environ.get("WZ_CKPT_DIR") or configured(cfg, "paths", "checkpoints_dir") or "checkpoints"
    args.ckpt_dir = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    reg = load_register_env()
    if args.all:
        envs = list(ENVS)
    elif args.env:
        envs = [e for e in ENVS if e in args.env]
    else:
        envs = [e for e in ENVS if reg.registered_python(e)]
        if not envs:
            print("No environment is registered in config/services.local.yaml; run the install scripts "
                  "(scripts/install_all.sh) or pass --env NAME --python PATH.")
            return 1
    if (args.python or args.repo_dir) and len(envs) != 1:
        parser.error("--python and --repo-dir need exactly one --env")
    if args.load_models and "main" not in envs:
        parser.error("--load-models applies to the main env")

    report, failed = {}, False
    for name in envs:
        section = ("main",) if name == "main" else ("services", name)
        python = args.python or reg.registered_python(name) or configured(cfg, *(section + ("python",)))
        repo_dir = (args.repo_dir or (configured(cfg, "services", name, "repo_dir") if name != "main" else None)
                    or os.path.join(ROOT, DEFAULT_REPO_DIRS[name]))
        repo_dir = os.path.abspath(repo_dir)
        print("\n== %s  (%s)" % (name, python or "not registered"))
        if not python:
            res = [{"status": "fail", "name": "interpreter",
                    "detail": "not registered; run scripts/install_env_%s.sh or "
                              "'python scripts/register_env.py %s /path/to/bin/python'" % (name, name)}]
            print_result(res[0])
        elif not os.path.isfile(python):
            res = [{"status": "fail", "name": "interpreter", "detail": "%s does not exist" % python}]
            print_result(res[0])
        else:
            res = run_env(name, python, repo_dir, args)
        report[name] = {"python": python, "repo_dir": repo_dir, "results": res}
        failed = failed or any(r["status"] == "fail" for r in res)

    print("\nSummary:")
    for name, data in report.items():
        counts = {k: sum(1 for r in data["results"] if r["status"] == k) for k in ("ok", "warn", "fail", "skip")}
        print("  %-7s %3d ok  %3d warn  %3d fail  %3d skipped" % (name, counts["ok"], counts["warn"],
                                                                   counts["fail"], counts["skip"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    print("\n%s" % ("Some checks FAILED (see [FAIL] lines above)." if failed else "All checks passed."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
