#!/usr/bin/env bash
# Build every WonderZoom environment in sequence, then run scripts/check_install.py on them
# (--no-checkpoints, since the checkpoints come in the next step; --no-gpu when no GPU is visible).
#
# Usage: bash scripts/install_all.sh [--objects] [--prefix-root DIR] [--only LIST] [--skip LIST] [--keep-going] [--dry-run]
#   --objects           also install the object-insertion stack into wz-main and build wz-step1x
#   --prefix-root DIR   create the envs as DIR/wz-main, DIR/wz-gen3c, DIR/wz-coz, DIR/wz-step1x
#                       (default: named conda envs wz-main, wz-gen3c, wz-coz, wz-step1x)
#   --only LIST         comma-separated subset of: main,gen3c,coz,step1x
#   --skip LIST         comma-separated environments to leave out
#   --keep-going        continue with the next environment after a failure
#   --dry-run           show what each install script would do (read-only checks only)
#
# Environment variables are passed through: MAX_JOBS, TORCH_CUDA_ARCH_LIST (default
# '8.0;8.6;8.9;9.0'; a single arch such as '8.9' builds much faster), PIP_CONFIG_FILE, WZ_TRACE.
# Logs go to logs/install/<env>.log. On a 2-CPU machine the full build takes several hours;
# transformer-engine, apex and pytorch3d dominate. Each step is idempotent, so a failed or
# interrupted run can simply be restarted.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OBJECTS=0
PREFIX_ROOT=
ONLY=
SKIP=
KEEP_GOING=0
DRY_RUN=0

log() { printf '[install_all %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[install_all] ERROR: %s\n' "$*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
# True when a GPU is visible to this shell (nvidia-smi lists one and CUDA_VISIBLE_DEVICES does not hide them all).
gpu_visible() {
  case "${CUDA_VISIBLE_DEVICES-unset}" in "" | -1) return 1 ;; esac
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '
}

while [ $# -gt 0 ]; do
  case "$1" in
    --objects) OBJECTS=1 ;;
    --prefix-root) [ $# -ge 2 ] || die "--prefix-root needs a value"; PREFIX_ROOT=$2; shift ;;
    --prefix-root=*) PREFIX_ROOT=${1#*=} ;;
    --only) [ $# -ge 2 ] || die "--only needs a value"; ONLY=$2; shift ;;
    --only=*) ONLY=${1#*=} ;;
    --skip) [ $# -ge 2 ] || die "--skip needs a value"; SKIP=$2; shift ;;
    --skip=*) SKIP=${1#*=} ;;
    --keep-going) KEEP_GOING=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

envs=(main gen3c coz)
if [ "$OBJECTS" = 1 ]; then envs+=(step1x); fi
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a envs <<< "$ONLY"
fi
selected=()
for e in "${envs[@]}"; do
  case "$e" in main | gen3c | coz | step1x) ;; *) die "unknown environment '$e'" ;; esac
  case ",$SKIP," in *",$e,"*) continue ;; esac
  selected+=("$e")
done
[ ${#selected[@]} -gt 0 ] || die "nothing to install"
if [ -n "$PREFIX_ROOT" ]; then
  PREFIX_ROOT=$(realpath -m "$PREFIX_ROOT")
fi

LOG_DIR="$ROOT/logs/install"
mkdir -p "$LOG_DIR"
log "environments: ${selected[*]}  (logs: ${LOG_DIR#"$ROOT"/})"

failed=()
built=()
for e in "${selected[@]}"; do
  args=()
  if [ -n "$PREFIX_ROOT" ]; then args+=(--prefix "$PREFIX_ROOT/wz-$e"); fi
  if [ "$e" = main ] && [ "$OBJECTS" = 1 ]; then args+=(--objects); fi
  if [ "$DRY_RUN" = 1 ]; then args+=(--dry-run); fi
  log "==> scripts/install_env_$e.sh ${args[*]}"
  set +e
  bash "$ROOT/scripts/install_env_$e.sh" "${args[@]}" 2>&1 | tee "$LOG_DIR/$e.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    failed+=("$e")
    log "!! $e failed (exit $rc); see ${LOG_DIR#"$ROOT"/}/$e.log"
    [ "$KEEP_GOING" = 1 ] || break
  else
    built+=("$e")
  fi
done

if [ ${#built[@]} -gt 0 ] && [ "$DRY_RUN" = 0 ]; then
  check_args=()
  for e in "${built[@]}"; do check_args+=(--env "$e"); done
  if [ "$OBJECTS" = 1 ]; then check_args+=(--objects); fi
  # Checkpoints are downloaded after this script (scripts/download_checkpoints.sh), so skip their
  # checks here; the full check is 'python scripts/check_install.py' once they are in place.
  check_args+=(--no-checkpoints)
  if ! gpu_visible; then
    log "no GPU visible: skipping the CUDA and kernel checks (re-run check_install.py on a GPU node)"
    check_args+=(--no-gpu)
  fi
  # check_install.py only needs the standard library; run it with the registered main python.
  PY=$(python3 "$ROOT/scripts/register_env.py" --get main 2>/dev/null || command -v python3 || true)
  if [ -n "$PY" ]; then
    log "==> scripts/check_install.py ${check_args[*]}"
    set +e
    "$PY" "$ROOT/scripts/check_install.py" "${check_args[@]}" 2>&1 | tee "$LOG_DIR/check_install.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then failed+=("check_install (see ${LOG_DIR#"$ROOT"/}/check_install.log)"); fi
  fi
fi

if [ ${#failed[@]} -gt 0 ]; then
  die "failed: ${failed[*]}"
fi
if [ "$DRY_RUN" = 1 ]; then
  log "dry run finished: ${built[*]}"
  exit 0
fi
groups="--core --gen3c --coz"
check="python scripts/check_install.py"
if [ "$OBJECTS" = 1 ]; then groups="$groups --step1x --objects"; check="$check --objects"; fi
log "all done: ${built[*]}. Next: bash scripts/download_checkpoints.sh $groups"
log "then, on a GPU node: $check"
