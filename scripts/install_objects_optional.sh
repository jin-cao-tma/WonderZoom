#!/usr/bin/env bash
# Add the optional object-insertion stack to an existing main environment (default: wz-main):
# GroundedSAM (GroundingDINO + segment_anything), INR-Harmonization, the OpenAI client for GPT
# prompts, and gdown for the INR checkpoint. Object insertion also needs the Step1X-Edit worker
# (scripts/install_env_step1x.sh) and `scripts/download_checkpoints.sh --objects --step1x`.
#
# Usage: bash scripts/install_objects_optional.sh [--name NAME | --prefix DIR] [--force-rebuild] [--dry-run]
#   --name NAME       main conda environment name (default: wz-main)
#   --prefix DIR      main conda environment prefix (overrides --name)
#   --force-rebuild   rebuild GroundingDINO's CUDA extension even if it already imports
#   --dry-run         show what would be done; only read-only checks are run
#
# Environment variables:
#   MAX_JOBS              parallel compile jobs (default: number of CPUs, at most 8; each nvcc job can use 4-6 GB RAM)
#   TORCH_CUDA_ARCH_LIST  GPU architectures for groundingdino._C (default '8.0;8.6;8.9;9.0').
#                         A single entry such as '8.9' builds faster but only runs on that GPU family.
#   PIP_CONFIG_FILE       set to /dev/null to ignore a pip.conf that adds unreachable extra indexes
#   WZ_TRACE=1            print every command (bash xtrace)
#
# Steps, each skipped when already done:
#   requirements/main-objects.txt (constrained by requirements/main.txt)
#   -> external/Grounded-Segment-Anything at GSAM_COMMIT: GroundingDINO (editable, --no-build-isolation,
#      compiles groundingdino._C) and segment_anything
#   -> external/INR-Harmonization at INR_COMMIT + WonderZoom patch and wrapper
#   -> strict import check -> register the main interpreter.
set -Eeuo pipefail

SCRIPT=install_objects_optional
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../third_party/pins.env
source "$ROOT/third_party/pins.env"

NAME=wz-main
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
if [ -z "$PREFIX" ] || [ ! -d "$PREFIX/conda-meta" ]; then
  die "main env ${PREFIX:-$NAME} not found; run scripts/install_env_main.sh first (or pass --prefix/--name)"
fi
activate "$PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
PY="$CONDA_PREFIX/bin/python"
pip_install() { run "$PY" -m pip install --disable-pip-version-check "$@"; }
log "env: $CONDA_PREFIX  MAX_JOBS=$MAX_JOBS  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
"$PY" -I -c "import torch" 2>/dev/null || die "torch is missing in $CONDA_PREFIX; run scripts/install_env_main.sh first"

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
# requirements/main.txt as constraints (no direct URLs or extras allowed in constraints files),
# plus torch, so nothing installed here can move the main environment's pins.
{
  sed -E -e '/^[[:space:]]*(#|$)/d' -e '/@/d' -e 's/\[[^]]*\]//' "$ROOT/requirements/main.txt"
  printf 'torch==2.4.0\ntorchvision==0.19.0\n'
} > "$TMPD/main-constraints.txt"

# --- 1. Python packages ------------------------------------------------------------------
log "installing requirements/main-objects.txt"
# Lines marked '# --no-deps' are installed separately without dependency resolution (see the file).
grep -v -E '^[^#]+#[[:space:]]*--no-deps' "$ROOT/requirements/main-objects.txt" > "$TMPD/objects.txt"
mapfile -t nodeps < <(grep -E '^[^#]+#[[:space:]]*--no-deps' "$ROOT/requirements/main-objects.txt" | sed -E 's/[[:space:]]*#.*$//')
pip_install -r "$TMPD/objects.txt" -c "$TMPD/main-constraints.txt"
if [ ${#nodeps[@]} -gt 0 ]; then
  pip_install --no-deps "${nodeps[@]}"
fi

# --- 2. GroundingDINO + segment_anything from the pinned Grounded-Segment-Anything -------
if [ "$DRY_RUN" = 1 ]; then
  bash "$ROOT/scripts/setup_third_party.sh" --check --external-dir "$EXTERNAL_DIR" gsam ||
    log "(dry run) would clone Grounded-Segment-Anything at its pin"
else
  bash "$ROOT/scripts/setup_third_party.sh" --external-dir "$EXTERNAL_DIR" gsam
fi
GSAM_DIR="$EXTERNAL_DIR/Grounded-Segment-Anything"
if [ "$FORCE_REBUILD" = 0 ] && "$PY" -I -c "import os, sys, torch, groundingdino, groundingdino._C; sys.exit(0 if os.path.realpath(groundingdino.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep) else 1)" "$GSAM_DIR/GroundingDINO" 2>/dev/null; then
  log "groundingdino (with _C) already installed from $GSAM_DIR/GroundingDINO"
else
  if [ "$FORCE_REBUILD" = 1 ]; then
    run rm -rf "$GSAM_DIR/GroundingDINO/build"
    run find "$GSAM_DIR/GroundingDINO/groundingdino" -maxdepth 1 -name '_C*.so' -delete
  fi
  log "building GroundingDINO (editable; compiles groundingdino._C)"
  # Editable, as in the reference setup: groundingdino/config/*.py is not a package and is only
  # reachable from the source tree. BUILD_WITH_CUDA/AM_I_DOCKER force the CUDA build even when
  # no GPU is visible during the build.
  export BUILD_WITH_CUDA=1 AM_I_DOCKER=1
  pip_install --no-build-isolation --no-deps -e "$GSAM_DIR/GroundingDINO"
  unset BUILD_WITH_CUDA AM_I_DOCKER
fi
if "$PY" -I -c "import os, sys, segment_anything; sys.exit(0 if os.path.isfile(os.path.join(os.path.dirname(segment_anything.__file__), 'build_sam_hq.py')) else 1)" 2>/dev/null; then
  log "segment_anything (Grounded-SAM copy) already installed"
else
  log "installing segment_anything from $GSAM_DIR/segment_anything"
  pip_install --no-deps --force-reinstall "$GSAM_DIR/segment_anything"
fi

# --- 3. INR-Harmonization (clone + WonderZoom patch + wrapper) ---------------------------
inr_ok=1
inr_args=(--external-dir "$EXTERNAL_DIR")
if [ "$DRY_RUN" = 1 ]; then inr_args+=(--check); fi
if ! bash "$ROOT/scripts/setup_third_party.sh" "${inr_args[@]}" inr; then
  inr_ok=0
  warn "INR-Harmonization setup failed; harmonization stays unavailable until it succeeds"
fi

# --- 4. strict import check ---------------------------------------------------------------
log "checking imports"
check_status=0
"$PY" -I - "$ROOT/requirements/main.txt" "$ROOT/requirements/main-objects.txt" "$EXTERNAL_DIR/INR-Harmonization" "$inr_ok" <<'EOF' || check_status=$?
import importlib, importlib.metadata as md, os, re, sys, warnings

warnings.filterwarnings("ignore")
errors = []
for path in sys.argv[1:3]:
    for line in open(path):
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if m:
            name, want = m.groups()
            try:
                have = md.version(name)
            except md.PackageNotFoundError:
                errors.append("%s is not installed (want %s)" % (name, want))
                continue
            if have != want:
                errors.append("%s %s installed, %s pins %s" % (name, have, os.path.basename(path), want))
import torch  # load libtorch/libc10 first: groundingdino._C links against them
for name in ("groundingdino", "groundingdino._C", "groundingdino.models", "groundingdino.util.slconfig",
             "segment_anything", "supervision", "pycocotools", "albumentations", "adamp", "openai", "gdown"):
    try:
        importlib.import_module(name)
    except Exception as e:
        errors.append("import %s: %s: %s" % (name, type(e).__name__, e))
try:
    import groundingdino
    cfg = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")
    if not os.path.isfile(cfg):
        errors.append("GroundingDINO config not found at %s" % cfg)
except Exception:
    pass
inr_dir, inr_ok = sys.argv[3], sys.argv[4] == "1"
if inr_ok and os.path.isfile(os.path.join(inr_dir, "inr_harmonization_model.py")):
    sys.path.insert(0, inr_dir)
    try:
        importlib.import_module("inr_harmonization_model")
    except Exception as e:
        errors.append("import inr_harmonization_model: %s: %s" % (type(e).__name__, e))
    finally:
        sys.path.remove(inr_dir)
else:
    print("WARNING: INR-Harmonization is not set up; harmonization will be disabled")
import torch
if errors:
    print("skipping the groundingdino._C kernel test (imports failed)")
elif torch.cuda.is_available():
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
    if err < 1e-4:
        print("CUDA OK: groundingdino._C matches the PyTorch reference on %s" % torch.cuda.get_device_name(0))
    else:
        errors.append("groundingdino._C differs from the PyTorch reference (max abs err %.3g): it was not "
                      "built for this GPU; rebuild with --force-rebuild and a TORCH_CUDA_ARCH_LIST that "
                      "includes it" % err)
else:
    print("WARNING: no GPU visible; groundingdino._C kernel was not tested")
if errors:
    print("IMPORT CHECK FAILED:")
    for e in errors:
        print("  - " + e)
    sys.exit(1)
print("OBJECTS_OK")
EOF
if [ "$check_status" != 0 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    log "(dry run) the object stack is not complete yet; the steps above would complete it"
  else
    die "import check failed (see above)"
  fi
fi

# --- 5. register --------------------------------------------------------------------------
run "$PY" "$ROOT/scripts/register_env.py" main "$CONDA_PREFIX/bin/python"
log "done. Next: python scripts/check_install.py --env main --objects"
