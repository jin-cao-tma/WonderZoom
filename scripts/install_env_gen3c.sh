#!/usr/bin/env bash
# Build the Gen3C worker environment (default name: wz-gen3c) following the upstream GEN3C
# INSTALL.md recipe at the pinned commit, minus MoGe (the WonderZoom worker does not use
# gen3c_persistent.py or gen3c_single_image.py, the only modules that import it).
#
# Usage: bash scripts/install_env_gen3c.sh [--name NAME | --prefix DIR] [--force-rebuild] [--dry-run]
#   --name NAME       conda environment name (default: wz-gen3c)
#   --prefix DIR      conda environment prefix (overrides --name)
#   --force-rebuild   rebuild transformer-engine and apex even if they already import
#   --dry-run         show what would be done; only read-only checks are run
#
# Environment variables:
#   MAX_JOBS              parallel compile jobs (default: number of CPUs, at most 8; each nvcc job can use 4-6 GB RAM)
#   TORCH_CUDA_ARCH_LIST  GPU architectures for apex (default '8.0;8.6;8.9;9.0'). A single entry
#                         such as '8.9' builds faster but only runs on that GPU family.
#   NVTE_CUDA_ARCHS       architectures for transformer-engine (TE default: 70;80;89;90)
#   PIP_CONFIG_FILE       set to /dev/null to ignore a pip.conf that adds unreachable extra indexes
#   WZ_TRACE=1            print every command (bash xtrace)
#
# Steps, each skipped when already done:
#   external/GEN3C + external/apex at their pins -> conda env from external/GEN3C/cosmos-predict1.yaml
#   -> external/GEN3C/requirements.txt -> header symlinks for the TE build -> pybind11, ninja
#   -> transformer-engine[pytorch]==1.12.0 (source build) -> apex (--cpp_ext --cuda_ext)
#   -> strict import check (Gen3cPipeline) -> register the interpreter.
# The transformer-engine and apex builds take several hours on a 2-CPU machine.
# apex with CUDA extensions is required even for inference: Gen3C imports amp_C at import time.
set -Eeuo pipefail

SCRIPT=install_env_gen3c
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../third_party/pins.env
source "$ROOT/third_party/pins.env"

NAME=wz-gen3c
PREFIX=
FORCE_REBUILD=0
DRY_RUN=0
EXTERNAL_DIR=${WZ_EXTERNAL_DIR:-$ROOT/external}

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
export NVTE_FRAMEWORK=pytorch
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

# --- 1. pinned sources -------------------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  bash "$ROOT/scripts/setup_third_party.sh" --check --external-dir "$EXTERNAL_DIR" gen3c apex ||
    { log "(dry run) would clone GEN3C and apex at their pins first; nothing else can be checked yet"; exit 0; }
else
  bash "$ROOT/scripts/setup_third_party.sh" --external-dir "$EXTERNAL_DIR" gen3c apex
fi
GEN3C_DIR="$EXTERNAL_DIR/GEN3C"
APEX_DIR="$EXTERNAL_DIR/apex"

# --- 2. conda env (upstream cosmos-predict1.yaml: Python 3.10, GCC 12.4, CUDA 12.4 toolkit) --
if [ -n "$PREFIX" ] && [ -d "$PREFIX/conda-meta" ]; then
  log "conda env $PREFIX exists"
elif [ "$DRY_RUN" = 1 ]; then
  log "(dry run) would create the conda env ${PREFIX:-$NAME} from cosmos-predict1.yaml, then run every step"
  exit 0
else
  log "creating conda env from external/GEN3C/cosmos-predict1.yaml"
  if [ -n "$PREFIX" ]; then
    conda env create -p "$PREFIX" -f "$GEN3C_DIR/cosmos-predict1.yaml"
  else
    conda env create -n "$NAME" -f "$GEN3C_DIR/cosmos-predict1.yaml"
    PREFIX=$(prefix_of_name "$NAME")
    [ -n "$PREFIX" ] || die "conda env '$NAME' was not found after creation"
  fi
fi
activate "$PREFIX"
# transformer-engine finds NVRTC through CUDA_HOME, both when building and at run time.
export CUDA_HOME="$CONDA_PREFIX"
PY="$CONDA_PREFIX/bin/python"
pip_install() { run "$PY" -m pip install --disable-pip-version-check "$@"; }
log "env: $CONDA_PREFIX  MAX_JOBS=$MAX_JOBS  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
# Upstream pins as a constraints file, so that later steps cannot move them. pip constraints
# cannot carry extras, so 'imageio[pyav,ffmpeg]==2.37.0' becomes 'imageio==2.37.0'.
sed -E -e 's/\[[^]]*\]//' -e '/^[[:space:]]*(#|$)/d' "$GEN3C_DIR/requirements.txt" > "$TMPD/gen3c-constraints.txt"

# --- 3. upstream requirements ------------------------------------------------------------
log "installing external/GEN3C/requirements.txt"
pip_install -r "$GEN3C_DIR/requirements.txt"

# --- 4. expose the pip CUDA headers to the TE/apex builds (upstream INSTALL.md) -----------
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
shopt -s nullglob
headers=("$SITE"/nvidia/*/include/*)
shopt -u nullglob
if [ ${#headers[@]} -gt 0 ]; then
  log "symlinking ${#headers[@]} CUDA header entries from site-packages/nvidia into $CONDA_PREFIX/include"
  run mkdir -p "$CONDA_PREFIX/include/python3.10"
  run ln -sf "${headers[@]}" "$CONDA_PREFIX/include/"
  run ln -sf "${headers[@]}" "$CONDA_PREFIX/include/python3.10"
else
  warn "no headers found under $SITE/nvidia/*/include"
fi

# --- 5. build helpers + transformer-engine -----------------------------------------------
pip_install pybind11==2.13.6 ninja==1.11.1.4 -c "$TMPD/gen3c-constraints.txt"
if [ "$FORCE_REBUILD" = 0 ] && "$PY" -I -c "import sys, importlib.metadata as m; sys.exit(0 if m.version('transformer_engine') == '1.12.0' else 1); " 2>/dev/null \
  && "$PY" -I -c "import transformer_engine.pytorch" >/dev/null 2>&1; then
  log "transformer-engine 1.12.0 already installed"
else
  log "building transformer-engine[pytorch]==1.12.0 (long: compiles transformer_engine_torch)"
  te_args=()
  if [ "$FORCE_REBUILD" = 1 ]; then te_args=(--force-reinstall --no-cache-dir); fi
  pip_install "${te_args[@]}" "transformer-engine[pytorch]==1.12.0" -c "$TMPD/gen3c-constraints.txt"
fi

# --- 6. apex with C++/CUDA extensions ----------------------------------------------------
if [ "$FORCE_REBUILD" = 0 ] && "$PY" -I -c "import torch, apex, amp_C, fused_layer_norm_cuda" 2>/dev/null; then
  log "apex (with CUDA extensions) already installed"
else
  if [ "$FORCE_REBUILD" = 1 ]; then run rm -rf "$APEX_DIR/build"; fi
  log "building apex @${APEX_COMMIT:0:12} with --cpp_ext --cuda_ext"
  pip_install -v --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    -c "$TMPD/gen3c-constraints.txt" "$APEX_DIR"
fi

# --- 7. strict import check (GEN3C's scripts/test_environment.py exits 0 even on failure) --
log "checking imports (about a minute)"
check_status=0
(
  cd "$GEN3C_DIR"
  env -u LD_LIBRARY_PATH PYTHONPATH="$GEN3C_DIR" "$PY" - "$GEN3C_DIR/requirements.txt" <<'EOF'
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
            errors.append("%s %s installed, GEN3C requirements.txt pins %s" % (name, have, want))
for name, want in (("transformer_engine", "1.12.0"),):
    try:
        if md.version(name) != want:
            errors.append("%s %s installed, expected %s" % (name, md.version(name), want))
    except md.PackageNotFoundError:
        errors.append("%s is not installed" % name)
for name in ("torch", "transformers", "megatron.core", "transformer_engine.pytorch", "amp_C",
             "apex.normalization", "cosmos_predict1.diffusion.inference.gen3c_pipeline"):
    try:
        importlib.import_module(name)
    except Exception as e:
        errors.append("import %s: %s: %s" % (name, type(e).__name__, e))
try:
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline  # noqa: F401
except Exception as e:
    errors.append("Gen3cPipeline: %s: %s" % (type(e).__name__, e))
import torch
if torch.cuda.is_available():
    from apex.normalization import FusedRMSNorm
    y = FusedRMSNorm(64).cuda()(torch.randn(4, 64, device="cuda"))
    torch.cuda.synchronize()
    print("CUDA OK: %s, torch %s, apex fused kernel ran" % (torch.cuda.get_device_name(0), torch.__version__))
else:
    print("WARNING: no GPU visible; GPU kernels were not tested (run scripts/check_install.py on a GPU node)")
if errors:
    print("IMPORT CHECK FAILED:")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("GEN3C_ENV_OK")
EOF
) || check_status=$?
if [ "$check_status" != 0 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    log "(dry run) the env is not complete yet; the steps above would complete it"
  else
    die "import check failed (see above)"
  fi
fi

# --- 8. register --------------------------------------------------------------------------
run "$PY" "$ROOT/scripts/register_env.py" gen3c "$CONDA_PREFIX/bin/python"
log "done. Next: python scripts/check_install.py --env gen3c"
