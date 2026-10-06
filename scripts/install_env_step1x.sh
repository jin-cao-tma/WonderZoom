#!/usr/bin/env bash
# Build the Step1X-Edit worker environment (default name: wz-step1x). Only needed for object
# insertion. No CUDA compilation: flash-attn comes as a prebuilt wheel.
#
# Usage: bash scripts/install_env_step1x.sh [--name NAME | --prefix DIR] [--dry-run]
#   --name NAME    conda environment name (default: wz-step1x)
#   --prefix DIR   conda environment prefix (overrides --name)
#   --dry-run      show what would be done; only read-only checks are run
#
# Environment variables:
#   PIP_CONFIG_FILE  set to /dev/null to ignore a pip.conf that adds unreachable extra indexes
#   WZ_TRACE=1       print every command (bash xtrace)
#   (MAX_JOBS and TORCH_CUDA_ARCH_LIST are accepted for symmetry; nothing is compiled here.)
#
# Requirements: Linux x86_64, an NVIDIA driver for CUDA 12.6 (PyPI torch 2.7.1 is the cu126
# build), and a host C compiler at run time (Triton JIT-compiles the liger RMSNorm kernel).
#
# Steps, each skipped when already done:
#   external/Step1X-Edit at its pin + simple_step1x.py -> conda env (Python 3.10, conda-forge)
#   -> torch 2.7.1 + torchvision 0.22.1 -> requirements/step1x.txt -> flash-attn wheel
#   (FLASH_ATTN_WHEEL_URL in third_party/pins.env) -> strict import check -> register.
set -Eeuo pipefail

SCRIPT=install_env_step1x
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../third_party/pins.env
source "$ROOT/third_party/pins.env"

NAME=wz-step1x
PREFIX=
DRY_RUN=0
EXTERNAL_DIR=${WZ_EXTERNAL_DIR:-$ROOT/external}
TORCH_PINS=(torch==2.7.1 torchvision==0.22.1)
FLASH_ATTN_VERSION=2.7.4.post1

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

# --- 1. pinned source + simple_step1x.py -------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  bash "$ROOT/scripts/setup_third_party.sh" --check --external-dir "$EXTERNAL_DIR" step1x ||
    { log "(dry run) would clone Step1X-Edit at its pin first; nothing else can be checked yet"; exit 0; }
else
  bash "$ROOT/scripts/setup_third_party.sh" --external-dir "$EXTERNAL_DIR" step1x
fi
STEP1X_DIR="$EXTERNAL_DIR/Step1X-Edit"

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
pip_install() { run "$PY" -m pip install --disable-pip-version-check "$@"; }
log "env: $CONDA_PREFIX"

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
printf '%s\n' "${TORCH_PINS[@]}" > "$TMPD/torch-constraints.txt"

# --- 3. torch (PyPI default wheels = CUDA 12.6 build) ------------------------------------
if "$PY" -I -c "import sys, importlib.metadata as m; sys.exit(0 if (m.version('torch'), m.version('torchvision')) == ('2.7.1', '0.22.1') else 1)" 2>/dev/null; then
  log "torch 2.7.1 already installed"
else
  log "installing ${TORCH_PINS[*]}"
  pip_install "${TORCH_PINS[@]}"
fi

# --- 4. requirements ---------------------------------------------------------------------
log "installing requirements/step1x.txt"
pip_install -r "$ROOT/requirements/step1x.txt" -c "$TMPD/torch-constraints.txt"

# --- 5. flash-attn wheel -----------------------------------------------------------------
if "$PY" -I -c "import sys, importlib.metadata as m; sys.exit(0 if m.version('flash_attn') == '$FLASH_ATTN_VERSION' else 1)" 2>/dev/null; then
  log "flash-attn $FLASH_ATTN_VERSION already installed"
else
  log "installing flash-attn $FLASH_ATTN_VERSION (prebuilt wheel)"
  pip_install --no-deps "$FLASH_ATTN_WHEEL_URL"
fi

# --- 6. strict import check ---------------------------------------------------------------
log "checking imports"
check_status=0
(
  cd "$STEP1X_DIR"
  env -u LD_LIBRARY_PATH PYTHONPATH="$STEP1X_DIR" "$PY" - "$ROOT/requirements/step1x.txt" "$FLASH_ATTN_VERSION" <<'EOF'
import importlib, importlib.metadata as md, re, sys, warnings

warnings.filterwarnings("ignore")
errors = []
pins = {"torch": "2.7.1", "torchvision": "0.22.1", "flash_attn": sys.argv[2]}
for line in open(sys.argv[1]):
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
    if m:
        pins[m.group(1)] = m.group(2)
for name, want in pins.items():
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        errors.append("%s is not installed (want %s)" % (name, want))
        continue
    if have.split("+")[0] != want:
        errors.append("%s %s installed, expected %s" % (name, have, want))
import torch
modules = ["flash_attn", "flash_attn_2_cuda", "xfuser", "transformers", "diffusers", "qwen_vl_utils"]
if torch.cuda.is_available():
    # liger_kernel (imported by simple_step1x) needs a visible GPU at import time.
    modules += ["liger_kernel", "simple_step1x"]
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as e:
        errors.append("import %s: %s: %s" % (name, type(e).__name__, e))
try:
    from transformers.utils import is_flash_attn_2_available
    if not is_flash_attn_2_available():
        errors.append("transformers does not see flash-attn 2")
except Exception as e:
    errors.append("is_flash_attn_2_available: %s: %s" % (type(e).__name__, e))
if torch.cuda.is_available():
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn_func(q, q, q).float().sum().item()
    print("CUDA OK: %s, torch %s, flash-attn kernel ran" % (torch.cuda.get_device_name(0), torch.__version__))
else:
    print("WARNING: no GPU visible; liger_kernel/simple_step1x imports and GPU kernels were skipped")
if errors:
    print("IMPORT CHECK FAILED:")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("STEP1X_ENV_OK")
EOF
) || check_status=$?
if [ "$check_status" != 0 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    log "(dry run) the env is not complete yet; the steps above would complete it"
  else
    die "import check failed (see above)"
  fi
fi

# --- 7. register --------------------------------------------------------------------------
run "$PY" "$ROOT/scripts/register_env.py" step1x "$CONDA_PREFIX/bin/python"
log "done. Next: python scripts/check_install.py --env step1x"
