#!/usr/bin/env bash
# Build the main WonderZoom conda environment (generation server, 3DGS CUDA extensions and the
# in-process models). Default env name: wz-main.
#
# Usage: bash scripts/install_env_main.sh [--name NAME | --prefix DIR] [--objects] [--force-rebuild] [--dry-run]
#   --name NAME       conda environment name (default: wz-main)
#   --prefix DIR      conda environment prefix (overrides --name)
#   --objects         also install the optional object-insertion stack
#                     (scripts/install_objects_optional.sh: GroundedSAM, INR, openai, gdown)
#   --force-rebuild   rebuild pytorch3d and the CUDA extensions even if they already import
#   --dry-run         show what would be done; only read-only checks are run
#
# Environment variables:
#   MAX_JOBS              parallel compile jobs (default: number of CPUs, at most 8; each nvcc job can use 4-6 GB RAM)
#   TORCH_CUDA_ARCH_LIST  GPU architectures to compile for (default '8.0;8.6;8.9;9.0': A100,
#                         RTX 30xx/A6000, RTX 40xx/L40S/RTX 6000 Ada, H100). A single entry such as
#                         '8.9' builds several times faster, but the result only runs on that GPU family.
#   PIP_CONFIG_FILE       set to /dev/null to ignore a pip.conf that adds unreachable extra indexes
#   WZ_TRACE=1            print every command (bash xtrace)
#
# Steps, each skipped when already done:
#   conda toolchain (envs/wz-main.yml: GCC 12.4, CUDA 12.4 nvcc/toolkit) -> torch 2.4.0 (cu124)
#   -> requirements/main.txt -> pytorch3d (PYTORCH3D_COMMIT) -> GLM headers -> rasterizer and
#   simple-knn -> RepViT-SAM (editable) -> [--objects] -> strict import check -> register the
#   interpreter in config/services.local.yaml.
# Expect 1-2 hours on a 2-CPU machine; pytorch3d alone takes 45-90 min with MAX_JOBS=2.
set -Eeuo pipefail

SCRIPT=install_env_main
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../third_party/pins.env
source "$ROOT/third_party/pins.env"

NAME=wz-main
PREFIX=
OBJECTS=0
FORCE_REBUILD=0
DRY_RUN=0
TORCH_PINS=(torch==2.4.0 torchvision==0.19.0)
TORCH_INDEX=https://download.pytorch.org/whl/cu124

log() { printf '[%s %s] %s\n' "$SCRIPT" "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$SCRIPT" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$SCRIPT" "$*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
# run CMD...: execute, or only print it with --dry-run.
run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf '[%s] (dry run) would run: %s\n' "$SCRIPT" "$*"
  else
    "$@"
  fi
}
trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

while [ $# -gt 0 ]; do
  case "$1" in
    --name) [ $# -ge 2 ] || die "--name needs a value"; NAME=$2; shift ;;
    --name=*) NAME=${1#*=} ;;
    --prefix) [ $# -ge 2 ] || die "--prefix needs a value"; PREFIX=$2; shift ;;
    --prefix=*) PREFIX=${1#*=} ;;
    --objects) OBJECTS=1 ;;
    --force-rebuild) FORCE_REBUILD=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

NPROC=$(nproc 2>/dev/null || echo 2)
export MAX_JOBS=${MAX_JOBS:-$((NPROC < 8 ? NPROC : 8))}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}
export FORCE_CUDA=1   # build CUDA kernels even when no GPU is visible (e.g. on a login node)
if [ "${WZ_TRACE:-0}" = 1 ]; then set -x; fi

# --- conda helpers -----------------------------------------------------------------------
command -v conda >/dev/null 2>&1 || die "conda not found on PATH (install Miniconda/Miniforge first)"
CONDA_BASE=$(conda info --base)
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

prefix_of_name() { conda env list | awk -v n="$1" '$1 == n { print $NF; exit }'; }

activate() {
  # conda's activation scripts (e.g. the conda-forge gcc toolchain) read unset variables such as
  # SYS_SYSROOT, so activation must run without nounset and xtrace.
  local trace=0
  case $- in *x*) trace=1; set +x ;; esac
  set +u
  conda activate "$1"
  set -u
  if [ "$trace" = 1 ]; then set -x; fi
  [ "${CONDA_PREFIX:-}" = "$1" ] || die "could not activate $1"
}

if [ -n "$PREFIX" ]; then
  PREFIX=$(realpath -m "$PREFIX")
else
  PREFIX=$(prefix_of_name "$NAME")
fi

# --- 1. conda toolchain ------------------------------------------------------------------
if [ -n "$PREFIX" ] && [ -d "$PREFIX/conda-meta" ]; then
  log "conda env $PREFIX exists"
elif [ "$DRY_RUN" = 1 ]; then
  log "(dry run) would create the conda env ${PREFIX:-$NAME} from envs/wz-main.yml, then run every step"
  exit 0
else
  log "creating conda env from envs/wz-main.yml (conda-forge toolchain)"
  if [ -n "$PREFIX" ]; then
    conda env create -p "$PREFIX" -f "$ROOT/envs/wz-main.yml"
  else
    conda env create -n "$NAME" -f "$ROOT/envs/wz-main.yml"
    PREFIX=$(prefix_of_name "$NAME")
    [ -n "$PREFIX" ] || die "conda env '$NAME' was not found after creation"
  fi
fi
activate "$PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
PY="$CONDA_PREFIX/bin/python"
pip_install() { run "$PY" -m pip install --disable-pip-version-check "$@"; }
log "env: $CONDA_PREFIX  MAX_JOBS=$MAX_JOBS  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
cd "$ROOT"

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
# Keep torch fixed while installing everything else.
printf '%s\n' "${TORCH_PINS[@]}" > "$TMPD/torch-constraints.txt"

# --- 2. torch ----------------------------------------------------------------------------
if "$PY" -I -c "import sys, torch, torchvision; sys.exit(0 if (torch.__version__, torchvision.__version__) == ('2.4.0+cu124', '0.19.0+cu124') else 1)" 2>/dev/null; then
  log "torch 2.4.0+cu124 already installed"
else
  log "installing ${TORCH_PINS[*]} from $TORCH_INDEX"
  pip_install "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
fi

# --- 3. Python requirements (pip skips what is already satisfied) -------------------------
log "installing requirements/main.txt"
pip_install -r "$ROOT/requirements/main.txt" -c "$TMPD/torch-constraints.txt"

# --- 4. pytorch3d ------------------------------------------------------------------------
rebuild_args=()
if [ "$FORCE_REBUILD" = 1 ]; then
  rebuild_args=(--force-reinstall --no-deps --no-cache-dir)
fi
if [ "$FORCE_REBUILD" = 0 ] && "$PY" -I - "$PYTORCH3D_COMMIT" <<'EOF' 2>/dev/null
import importlib.metadata as md, json, sys
import torch  # noqa: F401  (loads libc10/libtorch, which the extension links against)
import pytorch3d._C  # noqa: F401  (compiled extension present)
info = json.loads(md.distribution("pytorch3d").read_text("direct_url.json") or "{}")
sys.exit(0 if info.get("vcs_info", {}).get("commit_id") == sys.argv[1] else 1)
EOF
then
  log "pytorch3d @${PYTORCH3D_COMMIT:0:12} already installed"
else
  log "building pytorch3d @${PYTORCH3D_COMMIT:0:12} (45-90 min with MAX_JOBS=2)"
  pip_install --no-build-isolation "${rebuild_args[@]}" -c "$TMPD/torch-constraints.txt" \
    "git+${PYTORCH3D_URL}@${PYTORCH3D_COMMIT}"
fi

# --- 5. GLM headers + 3DGS CUDA extensions -----------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  bash "$ROOT/scripts/setup_third_party.sh" --check glm || log "(dry run) would fetch the GLM headers"
else
  bash "$ROOT/scripts/setup_third_party.sh" glm
fi
SUBMODULES=("$ROOT/submodules/depth-diff-gaussian-rasterization-min" "$ROOT/submodules/simple-knn")
if [ "$FORCE_REBUILD" = 0 ] && "$PY" -I -c "import torch, depth_diff_gaussian_rasterization_min._C, simple_knn._C" 2>/dev/null; then
  log "rasterizer and simple-knn already installed"
else
  if [ "$FORCE_REBUILD" = 1 ]; then
    for d in "${SUBMODULES[@]}"; do run rm -rf "$d/build"; done
  fi
  log "building depth-diff-gaussian-rasterization-min and simple-knn"
  pip_install --no-build-isolation --no-cache-dir "${rebuild_args[@]}" "${SUBMODULES[@]}"
fi

# --- 6. RepViT-SAM (editable, vendored in RepViT/sam) -------------------------------------
if "$PY" -I -c "import os, sys, repvit_sam; sys.exit(0 if os.path.realpath(repvit_sam.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep) else 1)" "$ROOT/RepViT/sam" 2>/dev/null; then
  log "repvit_sam already installed from RepViT/sam"
else
  log "installing RepViT/sam (editable)"
  pip_install --no-deps -e "$ROOT/RepViT/sam"
fi

# --- 7. optional object-insertion stack --------------------------------------------------
if [ "$OBJECTS" = 1 ]; then
  objects_args=(--prefix "$CONDA_PREFIX")
  if [ "$DRY_RUN" = 1 ]; then objects_args+=(--dry-run); fi
  bash "$ROOT/scripts/install_objects_optional.sh" "${objects_args[@]}"
fi

# --- 8. strict import check ---------------------------------------------------------------
log "checking imports"
check_status=0
"$PY" -I - "$ROOT/requirements/main.txt" <<'EOF' || check_status=$?
import importlib, importlib.metadata as md, re, sys, warnings

warnings.filterwarnings("ignore")
errors = []
for line in open(sys.argv[1]):
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
    if m:
        name, want = m.groups()
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            errors.append("%s is not installed (want %s)" % (name, want))
            continue
        if have != want:
            errors.append("%s %s installed, requirements/main.txt pins %s" % (name, have, want))
modules = [
    "torch", "torchvision", "pytorch3d", "pytorch3d._C", "pytorch3d.renderer",
    "depth_diff_gaussian_rasterization_min", "simple_knn._C", "repvit_sam",
    "transformers", "diffusers", "accelerate", "huggingface_hub", "timm", "utils3d",
    "kornia", "cv2", "skimage.measure", "scipy", "matplotlib", "imageio", "imageio_ffmpeg",
    "av", "plyfile", "decord", "omegaconf", "einops", "iopath",
    "flask", "flask_cors", "flask_socketio", "socketio", "engineio",
]
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as e:  # report every failure, not just the first
        errors.append("import %s: %s: %s" % (name, type(e).__name__, e))
import torch
if torch.__version__ != "2.4.0+cu124":
    errors.append("torch is %s, expected 2.4.0+cu124" % torch.__version__)
if torch.cuda.is_available():
    x = torch.ones(8, device="cuda") * 2
    torch.cuda.synchronize()
    print("CUDA OK: %s (sm_%d%d), torch %s, CUDA %s" % (
        (torch.cuda.get_device_name(0),) + torch.cuda.get_device_capability(0)
        + (torch.__version__, torch.version.cuda)))
else:
    print("WARNING: no GPU visible; GPU kernels were not tested (run scripts/check_install.py on a GPU node)")
if errors:
    print("IMPORT CHECK FAILED:")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("MAIN_ENV_OK")
EOF
if [ "$check_status" != 0 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    log "(dry run) the env is not complete yet; the steps above would complete it"
  else
    die "import check failed (see above)"
  fi
fi

# --- 9. register --------------------------------------------------------------------------
run "$PY" "$ROOT/scripts/register_env.py" main "$CONDA_PREFIX/bin/python"
next="$PY scripts/check_install.py --env main"
if [ "$OBJECTS" = 1 ]; then next="$next --objects"; fi
log "done. Next: $next"
