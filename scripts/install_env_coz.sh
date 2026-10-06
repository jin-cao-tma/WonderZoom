#!/usr/bin/env bash
# Build the Chain-of-Zoom worker environment (default name: wz-coz) from the upstream
# requirements.txt at the pinned commit. No CUDA compilation is needed.
#
# Usage: bash scripts/install_env_coz.sh [--name NAME | --prefix DIR] [--dry-run]
#   --name NAME    conda environment name (default: wz-coz)
#   --prefix DIR   conda environment prefix (overrides --name)
#   --dry-run      show what would be done; only read-only checks are run
#
# Environment variables:
#   PIP_CONFIG_FILE  set to /dev/null to ignore a pip.conf that adds unreachable extra indexes
#   WZ_TRACE=1       print every command (bash xtrace)
#   (MAX_JOBS and TORCH_CUDA_ARCH_LIST are accepted for symmetry; nothing is compiled here.)
#
# Steps, each skipped when already done:
#   external/Chain-of-Zoom at its pin + wonderzoom_coz.py -> conda env (Python 3.10, conda-forge)
#   -> external/Chain-of-Zoom/requirements.txt (torch 2.4.1 cu121, diffusers 0.32.1,
#   transformers 4.49.0, peft 0.15.2, qwen-vl-utils 0.0.8) -> strict import check -> register.
# The model weights (SD3-medium, gated, and Qwen2.5-VL-3B) come from
# scripts/download_checkpoints.sh --coz; the SR LoRA/VAE checkpoints arrive with the clone.
set -Eeuo pipefail

SCRIPT=install_env_coz
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../third_party/pins.env
source "$ROOT/third_party/pins.env"

NAME=wz-coz
PREFIX=
DRY_RUN=0
EXTERNAL_DIR=${WZ_EXTERNAL_DIR:-$ROOT/external}

log() { printf '[%s %s] %s\n' "$SCRIPT" "$(date +%H:%M:%S)" "$*"; }
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
    --dry-run) DRY_RUN=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

NPROC=$(nproc 2>/dev/null || echo 2)
export MAX_JOBS=${MAX_JOBS:-$((NPROC < 8 ? NPROC : 8))}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}
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
  # Activation scripts may read unset variables, so run conda activate without nounset/xtrace.
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

# --- 1. pinned source + wonderzoom_coz.py ------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  bash "$ROOT/scripts/setup_third_party.sh" --check --external-dir "$EXTERNAL_DIR" coz ||
    { log "(dry run) would clone Chain-of-Zoom at its pin first; nothing else can be checked yet"; exit 0; }
else
  bash "$ROOT/scripts/setup_third_party.sh" --external-dir "$EXTERNAL_DIR" coz
fi
COZ_DIR="$EXTERNAL_DIR/Chain-of-Zoom"

# --- 2. conda env ------------------------------------------------------------------------
if [ -n "$PREFIX" ] && [ -d "$PREFIX/conda-meta" ]; then
  log "conda env $PREFIX exists"
elif [ "$DRY_RUN" = 1 ]; then
  log "(dry run) would create the conda env ${PREFIX:-$NAME} (python 3.10), then run every step"
  exit 0
else
  log "creating conda env (python 3.10)"
  if [ -n "$PREFIX" ]; then
    conda create -y -p "$PREFIX" -c conda-forge --override-channels python=3.10 pip
  else
    conda create -y -n "$NAME" -c conda-forge --override-channels python=3.10 pip
    PREFIX=$(prefix_of_name "$NAME")
    [ -n "$PREFIX" ] || die "conda env '$NAME' was not found after creation"
  fi
fi
activate "$PREFIX"
PY="$CONDA_PREFIX/bin/python"
log "env: $CONDA_PREFIX"

# --- 3. upstream requirements (exact pins of the working environment) --------------------
log "installing external/Chain-of-Zoom/requirements.txt"
run "$PY" -m pip install --disable-pip-version-check -r "$COZ_DIR/requirements.txt"

# --- 4. strict import check ---------------------------------------------------------------
log "checking imports"
check_status=0
(
  cd "$COZ_DIR"
  env -u LD_LIBRARY_PATH PYTHONPATH="$COZ_DIR" "$PY" - "$COZ_DIR/requirements.txt" <<'EOF'
import importlib, importlib.metadata as md, re, sys, warnings

warnings.filterwarnings("ignore")
errors = []
for line in open(sys.argv[1]):
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;#]+)", line)
    if m:
        name, want = m.groups()
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            errors.append("%s is not installed (want %s)" % (name, want))
            continue
        if have.split("+")[0] != want:
            errors.append("%s %s installed, Chain-of-Zoom requirements.txt pins %s" % (name, have, want))
for name in ("torch", "diffusers", "transformers", "peft", "qwen_vl_utils", "osediff_sd3", "wonderzoom_coz"):
    try:
        importlib.import_module(name)
    except Exception as e:
        errors.append("import %s: %s: %s" % (name, type(e).__name__, e))
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    from qwen_vl_utils import process_vision_info  # noqa: F401
except Exception as e:
    errors.append("Qwen2.5-VL classes: %s: %s" % (type(e).__name__, e))
import torch
if torch.cuda.is_available():
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    (x @ x).sum().item()
    print("CUDA OK: %s, torch %s" % (torch.cuda.get_device_name(0), torch.__version__))
else:
    print("WARNING: no GPU visible; run scripts/check_install.py on a GPU node")
if errors:
    print("IMPORT CHECK FAILED:")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("COZ_ENV_OK")
EOF
) || check_status=$?
if [ "$check_status" != 0 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    log "(dry run) the env is not complete yet; the steps above would complete it"
  else
    die "import check failed (see above)"
  fi
fi

# --- 5. register --------------------------------------------------------------------------
run "$PY" "$ROOT/scripts/register_env.py" coz "$CONDA_PREFIX/bin/python"
log "done. Next: python scripts/check_install.py --env coz"
