#!/usr/bin/env bash
# Clone the third-party code WonderZoom needs, at the exact commits in third_party/pins.env,
# and add WonderZoom's files to the clones.
#
# Usage: bash scripts/setup_third_party.sh [--check] [--external-dir DIR] (COMPONENT ... | --all)
#   gen3c    external/GEN3C                       Gen3C worker code (pristine, no patch)
#   apex     external/apex                        built into wz-gen3c by install_env_gen3c.sh
#   coz      external/Chain-of-Zoom               + third_party/chain_of_zoom/wonderzoom_coz.py
#   step1x   external/Step1X-Edit                 + third_party/step1x_edit/simple_step1x.py
#   inr      external/INR-Harmonization           + third_party/inr_harmonization/inr_harmonization.patch
#                                                 + third_party/inr_harmonization/inr_harmonization_model.py
#   gsam     external/Grounded-Segment-Anything   GroundingDINO + segment_anything (optional objects)
#   glm      submodules/depth-diff-gaussian-rasterization-min/third_party/glm (GLM headers)
#   --all    all of the above
# Options:
#   --external-dir DIR   clone into DIR instead of <repo>/external (or set WZ_EXTERNAL_DIR)
#   --check              verify existing clones only; clone and change nothing. Without a
#                        COMPONENT, checks every component that is already set up.
#
# Each clone is checked out at its pinned commit and HEAD is verified afterwards. Re-running is
# safe: finished steps are skipped. A clone at another commit is moved to the pin only when it
# has no local changes to tracked files; otherwise the script stops instead of touching it.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PINS="$ROOT/third_party/pins.env"
# shellcheck source=../third_party/pins.env
source "$PINS"

EXTERNAL_DIR=${WZ_EXTERNAL_DIR:-$ROOT/external}
CHECK_ONLY=0
ALL_COMPONENTS=(gen3c apex coz step1x inr gsam glm)
export GIT_TERMINAL_PROMPT=0   # fail instead of prompting for credentials on a bad URL

log() { printf '[setup_third_party] %s\n' "$*"; }
warn() { printf '[setup_third_party] WARNING: %s\n' "$*" >&2; }
err() { printf '[setup_third_party] ERROR: %s\n' "$*" >&2; }

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

# True when DIR is the top level of its own git work tree (a clone or a submodule checkout),
# not merely a directory inside some other repository.
is_git_toplevel() {
  local dir=$1 top
  [ -d "$dir" ] || return 1
  top=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null) || return 1
  [ "$(cd "$top" && pwd -P)" = "$(cd "$dir" && pwd -P)" ]
}

# ensure_clone NAME URL COMMIT DIR: clone/check out DIR at COMMIT and verify HEAD.
ensure_clone() {
  local name=$1 url=$2 commit=$3 dir=$4 head tmp
  if [ -e "$dir" ] && ! is_git_toplevel "$dir"; then
    err "$name: $dir exists but is not a git clone; move it away and re-run."
    return 1
  fi
  if [ ! -e "$dir" ]; then
    if [ "$CHECK_ONLY" = 1 ]; then
      err "$name: $dir is missing (run: bash scripts/setup_third_party.sh $name)"
      return 1
    fi
    log "$name: cloning $url"
    mkdir -p "$(dirname "$dir")"
    tmp="$dir.partial.$$"
    rm -rf "$tmp"
    # Clone into a temporary name so that an interrupted clone is never mistaken for a finished one.
    git clone --quiet --no-checkout "$url" "$tmp"
    if ! git -C "$tmp" cat-file -e "$commit^{commit}" 2>/dev/null; then
      git -C "$tmp" fetch --quiet origin "$commit"
    fi
    git -C "$tmp" -c advice.detachedHead=false checkout --quiet --detach "$commit"
    mv "$tmp" "$dir"
  fi
  head=$(git -C "$dir" rev-parse HEAD)
  if [ "$head" != "$commit" ]; then
    if [ "$CHECK_ONLY" = 1 ]; then
      err "$name: $dir is at $head, expected the pin $commit"
      return 1
    fi
    if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
      err "$name: $dir is at $head with local changes, expected $commit."
      err "  Inspect it with 'git -C $dir status', then reset or remove it and re-run."
      return 1
    fi
    log "$name: moving $dir from ${head:0:12} to the pinned ${commit:0:12}"
    if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
      git -C "$dir" fetch --quiet origin "$commit"
    fi
    git -C "$dir" -c advice.detachedHead=false checkout --quiet --detach "$commit"
    head=$(git -C "$dir" rev-parse HEAD)
  fi
  if [ "$head" != "$commit" ]; then
    err "$name: HEAD of $dir is $head, expected $commit"
    return 1
  fi
  log "$name: $dir @ ${commit:0:12} (pinned)"
}

# install_file SRC DST: copy one of WonderZoom's files into a clone (skipped when identical).
install_file() {
  local src=$1 dst=$2 rel=${1#"$ROOT"/}
  if [ ! -f "$src" ]; then
    err "missing $rel; it ships with the WonderZoom source tree."
    return 1
  fi
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
    log "  $(basename "$dst") is up to date"
    return 0
  fi
  if [ "$CHECK_ONLY" = 1 ]; then
    err "  $dst is missing or differs from $rel"
    return 1
  fi
  cp "$src" "$dst.tmp.$$"
  mv "$dst.tmp.$$" "$dst"
  log "  installed $rel -> ${dst#"$ROOT"/}"
}

# apply_patch DIR PATCH: apply PATCH to the clone in DIR once ('git apply --check' first).
apply_patch() {
  local dir=$1 patch=$2 rel=${2#"$ROOT"/}
  if [ ! -f "$patch" ]; then
    err "missing $rel; it ships with the WonderZoom source tree."
    return 1
  fi
  if git -C "$dir" apply --reverse --check "$patch" >/dev/null 2>&1; then
    log "  $(basename "$patch") is already applied"
    return 0
  fi
  if [ "$CHECK_ONLY" = 1 ]; then
    err "  $(basename "$patch") is not applied to $dir"
    return 1
  fi
  if ! git -C "$dir" apply --check "$patch"; then
    err "  $rel does not apply cleanly to $dir."
    err "  Reset the clone with: git -C $dir checkout -- . && git -C $dir clean -fd"
    return 1
  fi
  git -C "$dir" apply "$patch"
  log "  applied $rel"
}

# report_status DIR EXPECTED...: list changes in the clone; warn about anything unexpected.
report_status() {
  local dir=$1 line path unexpected=0
  shift
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    path=${line:3}
    case " $* " in
      *" $path "*) continue ;;
    esac
    case "$path" in
      __pycache__/ | */__pycache__/ | build/ | *.egg-info/) continue ;;
    esac
    if [ "$unexpected" = 0 ]; then
      warn "unexpected changes in $dir (not from WonderZoom):"
    fi
    unexpected=1
    printf '    %s\n' "$line" >&2
  done < <(git -C "$dir" status --porcelain)
  return 0
}

setup_gen3c() {
  ensure_clone gen3c "$GEN3C_URL" "$GEN3C_COMMIT" "$EXTERNAL_DIR/GEN3C"
  report_status "$EXTERNAL_DIR/GEN3C"
}

setup_apex() {
  ensure_clone apex "$APEX_URL" "$APEX_COMMIT" "$EXTERNAL_DIR/apex"
}

setup_coz() {
  local dir="$EXTERNAL_DIR/Chain-of-Zoom"
  ensure_clone coz "$COZ_URL" "$COZ_COMMIT" "$dir"
  install_file "$ROOT/third_party/chain_of_zoom/wonderzoom_coz.py" "$dir/wonderzoom_coz.py"
  local ckpt
  for ckpt in ckpt/SR_LoRA/model_20001.pkl ckpt/SR_VAE/vae_encoder_20001.pt; do
    if [ ! -s "$dir/$ckpt" ]; then
      err "  $dir/$ckpt is missing; it should come with the clone"
      return 1
    fi
  done
  report_status "$dir" wonderzoom_coz.py
}

setup_step1x() {
  local dir="$EXTERNAL_DIR/Step1X-Edit"
  ensure_clone step1x "$STEP1X_URL" "$STEP1X_COMMIT" "$dir"
  install_file "$ROOT/third_party/step1x_edit/simple_step1x.py" "$dir/simple_step1x.py"
  report_status "$dir" simple_step1x.py
}

setup_inr() {
  local dir="$EXTERNAL_DIR/INR-Harmonization" patch="$ROOT/third_party/inr_harmonization/inr_harmonization.patch"
  local expected=()
  ensure_clone inr "$INR_URL" "$INR_COMMIT" "$dir"
  apply_patch "$dir" "$patch"
  install_file "$ROOT/third_party/inr_harmonization/inr_harmonization_model.py" "$dir/inr_harmonization_model.py"
  # Every path the patch touches (old and new names of renamed files) is an expected change.
  mapfile -t expected < <(sed -n 's#^diff --git a/\(.*\) b/\(.*\)$#\1\n\2#p' "$patch" | sort -u)
  report_status "$dir" inr_harmonization_model.py "${expected[@]}"
}

setup_gsam() {
  ensure_clone gsam "$GSAM_URL" "$GSAM_COMMIT" "$EXTERNAL_DIR/Grounded-Segment-Anything"
  report_status "$EXTERNAL_DIR/Grounded-Segment-Anything"
}

GLM_DIR="$ROOT/submodules/depth-diff-gaussian-rasterization-min/third_party/glm"

setup_glm() {
  local dir="$GLM_DIR"
  if [ -f "$dir/glm/glm.hpp" ] && ! is_git_toplevel "$dir"; then
    log "glm: headers already present in ${dir#"$ROOT"/} (not a git clone; commit not verified)"
    return 0
  fi
  if [ -d "$dir" ] && ! is_git_toplevel "$dir" && [ -z "$(ls -A "$dir")" ]; then
    # An uninitialised submodule directory: replace it with a clone.
    [ "$CHECK_ONLY" = 1 ] || rmdir "$dir"
  fi
  ensure_clone glm "$GLM_URL" "$GLM_COMMIT" "$dir"
  [ -f "$dir/glm/glm.hpp" ] || { err "glm: $dir/glm/glm.hpp not found after checkout"; return 1; }
}

# ---------------------------------------------------------------------------------------
components=()
while [ $# -gt 0 ]; do
  case "$1" in
    --all) components+=("${ALL_COMPONENTS[@]}") ;;
    --external-dir)
      [ $# -ge 2 ] || { err "--external-dir needs a value"; exit 2; }
      EXTERNAL_DIR=$2
      shift
      ;;
    --external-dir=*) EXTERNAL_DIR=${1#*=} ;;
    --check) CHECK_ONLY=1 ;;
    -h | --help) usage; exit 0 ;;
    gen3c | apex | coz | step1x | inr | gsam | glm) components+=("$1") ;;
    *) err "unknown argument: $1"; usage >&2; exit 2 ;;
  esac
  shift
done
if [ ${#components[@]} -eq 0 ] && [ "$CHECK_ONLY" = 0 ]; then
  usage >&2
  exit 2
fi
command -v git >/dev/null 2>&1 || { err "git is required"; exit 1; }
if [ "$CHECK_ONLY" = 0 ]; then
  mkdir -p "$EXTERNAL_DIR"
fi
if [ -d "$EXTERNAL_DIR" ]; then
  EXTERNAL_DIR=$(cd "$EXTERNAL_DIR" && pwd)
fi

# --check without components: verify whatever has been set up so far.
if [ ${#components[@]} -eq 0 ]; then
  for c in "${ALL_COMPONENTS[@]}"; do
    case "$c" in
      gen3c) d="$EXTERNAL_DIR/GEN3C" ;;
      apex) d="$EXTERNAL_DIR/apex" ;;
      coz) d="$EXTERNAL_DIR/Chain-of-Zoom" ;;
      step1x) d="$EXTERNAL_DIR/Step1X-Edit" ;;
      inr) d="$EXTERNAL_DIR/INR-Harmonization" ;;
      gsam) d="$EXTERNAL_DIR/Grounded-Segment-Anything" ;;
      glm) d="$GLM_DIR" ;;
    esac
    if [ -e "$d" ]; then
      components+=("$c")
    else
      log "$c: not set up (${d#"$ROOT"/}), skipped"
    fi
  done
  if [ ${#components[@]} -eq 0 ]; then
    log "nothing is set up yet; nothing to check"
    exit 0
  fi
fi

# Run each component in its own subshell so that one failure does not hide the others.
declare -A seen=()
ordered=()
failed=()
for c in "${components[@]}"; do
  [ -z "${seen[$c]:-}" ] || continue
  seen[$c]=1
  ordered+=("$c")
  set +e
  ( set -Eeuo pipefail; "setup_$c" )
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    failed+=("$c")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  err "failed: ${failed[*]} (succeeded: $(( ${#ordered[@]} - ${#failed[@]} ))/${#ordered[@]})"
  exit 1
fi
log "done: ${ordered[*]}"
