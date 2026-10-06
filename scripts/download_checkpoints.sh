#!/usr/bin/env bash
# Download the WonderZoom checkpoints: pinned Hugging Face revisions with exact file lists
# (scripts/prefetch_hf.py) plus the files listed in third_party/checksums.sha256, which are
# fetched with curl (or gdown for Google Drive) and verified by sha256.
#
# Usage: bash scripts/download_checkpoints.sh GROUP... [options]
# Groups:
#   --core      main-process models: OneFormer, Marigold normals, GeometryCrafter + SVD-xt parts,
#               MoGe ViT-L, RepViT-SAM (required)                                       ~12 GB
#   --gen3c     Gen3C-Cosmos-7B, Cosmos tokenizer, T5-11B -> checkpoints/gen3c/          ~76 GB
#   --coz       Stable Diffusion 3 Medium (gated, see below) + Qwen2.5-VL-3B             ~23 GB
#   --step1x    Step1X-Edit v1.0 -> checkpoints/step1x/ + Qwen2.5-VL-7B (optional)       ~42 GB
#   --objects   GroundingDINO, SAM ViT-H, BERT, SD2 inpainting, INR (optional)          ~6.8 GB
#   --scenes    released render-only scenes -> gaussian/ (optional)                     ~7.8 GB
#   --all       all of the above                                                       ~168 GB
# Options:
#   --dry-run       list files, sizes and what is missing per group; download nothing
#   --verify        also re-hash files that are already present (default: size check only)
#   --skip-gated    download everything else when the account cannot access a gated model
#   --python PATH   Python with huggingface_hub >= 0.23 (default: the registered wz-main python)
#   --ckpt-dir DIR  checkpoints directory (default: $WZ_CKPT_DIR, else checkpoints/)
#   --only NAMES    comma-separated items to (re)fetch, e.g. --only sd3_medium,sam_vit_h_4b8939.pth
#                   (names are listed by --dry-run; selects every group unless groups are given)
#
# Hub files go to the Hugging Face cache ($HF_HOME/hub, default ~/.cache/huggingface/hub), except
# Gen3C and Step1X-Edit, which go to the checkpoints directory. Put both on a fast local disk.
# Free space is checked before anything is downloaded. Re-running is safe: files that are
# already present are skipped, so a second run downloads nothing.
#
# SD3-medium (--coz) is gated: accept its license on https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers,
# then run `huggingface-cli login` or export HF_TOKEN. The access check runs first and the
# script stops with instructions when access is missing (exit code 3).
# HF_ENDPOINT (mirrors) and HF_HUB_ENABLE_HF_TRANSFER=1 (with hf_transfer installed) are honoured.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKSUMS="$ROOT/third_party/checksums.sha256"
ALL_GROUPS=(core gen3c coz step1x objects scenes)

groups=()
ONLY=
DRY_RUN=0
VERIFY=0
SKIP_GATED=0
PY=${WZ_PYTHON:-}
CKPT_DIR=${WZ_CKPT_DIR:-checkpoints}

log() { printf '[download %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[download] WARNING: %s\n' "$*" >&2; }
err() { printf '[download] ERROR: %s\n' "$*" >&2; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --core | --gen3c | --coz | --step1x | --objects | --scenes) groups+=("${1#--}") ;;
    --all) groups+=("${ALL_GROUPS[@]}") ;;
    --dry-run) DRY_RUN=1 ;;
    --verify) VERIFY=1 ;;
    --skip-gated) SKIP_GATED=1 ;;
    --python) [ $# -ge 2 ] || { err "--python needs a value"; exit 2; }; PY=$2; shift ;;
    --python=*) PY=${1#*=} ;;
    --ckpt-dir) [ $# -ge 2 ] || { err "--ckpt-dir needs a value"; exit 2; }; CKPT_DIR=$2; shift ;;
    --ckpt-dir=*) CKPT_DIR=${1#*=} ;;
    --only) [ $# -ge 2 ] || { err "--only needs a value"; exit 2; }; ONLY=$2; shift ;;
    --only=*) ONLY=${1#*=} ;;
    -h | --help) usage; exit 0 ;;
    *) err "unknown argument: $1"; usage >&2; exit 2 ;;
  esac
  shift
done
if [ ${#groups[@]} -eq 0 ] && [ -n "$ONLY" ]; then
  groups=("${ALL_GROUPS[@]}")
fi
if [ ${#groups[@]} -eq 0 ]; then
  usage >&2
  exit 2
fi
# Keep the canonical order and drop duplicates.
selected=()
for g in "${ALL_GROUPS[@]}"; do
  case " ${groups[*]} " in *" $g "*) selected+=("$g") ;; esac
done
case "$CKPT_DIR" in /*) ;; *) CKPT_DIR="$ROOT/$CKPT_DIR" ;; esac

# --- a Python with huggingface_hub --------------------------------------------------------
any_python() {
  local c
  for c in python3 "${CONDA_PYTHON_EXE:-}"; do
    if [ -n "$c" ] && command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
  done
  return 1
}
has_hub() {
  [ -n "$1" ] && [ -x "$1" ] && "$1" -c "import sys, huggingface_hub as h; sys.exit(0 if tuple(int(x) for x in h.__version__.split('.')[:2]) >= (0, 23) else 1)" >/dev/null 2>&1
}
if [ -n "$PY" ]; then
  has_hub "$PY" || { err "$PY cannot import huggingface_hub >= 0.23"; exit 1; }
else
  BOOT=$(any_python || true)
  candidates=()
  if [ -n "$BOOT" ]; then
    for env in main step1x coz gen3c; do
      candidates+=("$("$BOOT" "$ROOT/scripts/register_env.py" --get "$env" 2>/dev/null || true)")
    done
  fi
  candidates+=("$BOOT")
  for c in "${candidates[@]}"; do
    if has_hub "$c"; then PY=$c; break; fi
  done
  if [ -z "$PY" ]; then
    err "no Python with huggingface_hub >= 0.23 found. Build the main env first"
    err "(bash scripts/install_env_main.sh), or run: pip install 'huggingface_hub>=0.23' and pass --python."
    exit 1
  fi
fi
if [ -n "$ONLY" ]; then
  log "items: $ONLY   python: $PY"
else
  log "groups: ${selected[*]}   python: $PY"
fi

# --- 1. Hugging Face files (plan, gated-access check, space check, download) -------------
hf_args=(--groups "$(IFS=,; echo "${selected[*]}")" --ckpt-dir "$CKPT_DIR")
if [ "$DRY_RUN" = 1 ]; then hf_args+=(--dry-run); fi
if [ "$VERIFY" = 1 ]; then hf_args+=(--verify); fi
if [ "$SKIP_GATED" = 1 ]; then hf_args+=(--skip-gated); fi
if [ -n "$ONLY" ]; then hf_args+=(--only "$ONLY"); fi
set +e
"$PY" "$ROOT/scripts/prefetch_hf.py" "${hf_args[@]}"
hf_rc=$?
set -e
case "$hf_rc" in
  0) ;;
  3) err "stopping: no access to a gated model (see above); nothing was downloaded."; exit 3 ;;
  *) err "Hugging Face step failed (exit $hf_rc)"
     if [ "$DRY_RUN" = 1 ]; then exit "$hf_rc"; fi ;;
esac
if [ "$DRY_RUN" = 1 ]; then
  log "dry run: files outside the Hub would be fetched with curl/gdown and checked against ${CHECKSUMS#"$ROOT"/}"
  exit 0
fi

# --- 2. files outside the Hub (third_party/checksums.sha256) -----------------------------
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

fetch_url() {  # fetch_url URL OUT
  local url=$1 out=$2
  local progress=(-sS)
  command -v curl >/dev/null 2>&1 || { err "curl is required to download $url"; return 1; }
  if [ -t 2 ]; then progress=(--progress-bar); fi
  # Resume a partial download when possible; fall back to a fresh one.
  if curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 "${progress[@]}" -C - -o "$out" "$url"; then
    return 0
  fi
  rm -f "$out"
  curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 "${progress[@]}" -o "$out" "$url"
}

fetch_gdrive() {  # fetch_gdrive FILE_ID OUT
  local id=$1 out=$2
  if "$PY" -c "import gdown" >/dev/null 2>&1; then
    "$PY" -m gdown --fuzzy "https://drive.google.com/uc?id=$id" -O "$out" && return 0
  else
    warn "gdown is not installed in $PY (pip install gdown==5.2.0; included in the --objects env install)"
  fi
  return 1
}

manual=()
failures=()
fetch_entry() {  # fetch_entry SHA SIZE URL REL
  local sha=$1 size=$2 url=$3 rel=$4 dest part have
  dest="$CKPT_DIR/$rel"
  part="$dest.part"
  if [ -f "$dest" ] && [ "$(file_size "$dest")" = "$size" ]; then
    if [ "$VERIFY" = 0 ]; then
      log "ok        $rel (present)"
      return 0
    fi
    have=$(sha256_of "$dest")
    if [ "$have" = "$sha" ]; then
      log "verified  $rel"
      return 0
    fi
    warn "$rel has the right size but the wrong sha256; deleting it and downloading again"
    rm -f "$dest"
  elif [ -f "$dest" ]; then
    warn "$rel has size $(file_size "$dest"), expected $size; deleting it and downloading again"
    rm -f "$dest"
  fi
  mkdir -p "$(dirname "$dest")"
  if [ -f "$part" ] && [ "$(file_size "$part")" = "$size" ]; then
    log "resume    $rel (partial download is complete)"
  else
    log "download  $rel ($size bytes) from ${url%%\?*}"
    case "$url" in
      gdrive:*)
        rm -f "$part"
        if ! fetch_gdrive "${url#gdrive:}" "$part"; then
          rm -f "$part"
          manual+=("$rel|$url|$sha|$size")
          return 0
        fi
        ;;
      *)
        if ! fetch_url "$url" "$part"; then
          failures+=("$rel: download failed from $url")
          return 0
        fi
        ;;
    esac
  fi
  have=$(sha256_of "$part")
  if [ "$have" != "$sha" ]; then
    mv -f "$part" "$dest.bad"
    failures+=("$rel: sha256 $have does not match the expected $sha (kept as $rel.bad)")
    return 0
  fi
  mv -f "$part" "$dest"
  log "verified  $rel"
}

meta=
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    "# size="*) meta=$line; continue ;;
    "#"* | "") continue ;;
  esac
  sha=${line%% *}
  rel=${line#*  }
  size=$(sed -n 's/.*size=\([0-9][0-9]*\).*/\1/p' <<< "$meta")
  group=$(sed -n 's/.*group=\([^ ]*\).*/\1/p' <<< "$meta")
  url=$(sed -n 's/.*url=\([^ ]*\).*/\1/p' <<< "$meta")
  meta=
  if [ -z "$size" ] || [ -z "$url" ]; then
    err "malformed entry for $rel in ${CHECKSUMS#"$ROOT"/}"
    failures+=("$rel: malformed checksums entry")
    continue
  fi
  case " ${selected[*]} " in *" ${group:-core} "*) ;; *) continue ;; esac
  if [ -n "$ONLY" ]; then
    case ",$ONLY," in *",$(basename "$rel"),"*) ;; *) continue ;; esac
  fi
  fetch_entry "$sha" "$size" "$url" "$rel"
done < "$CHECKSUMS"

# --- summary ------------------------------------------------------------------------------
for m in "${manual[@]}"; do
  IFS='|' read -r rel url sha size <<< "$m"
  warn "$rel needs a manual download (it is only hosted on Google Drive):"
  warn "  1. open https://drive.google.com/file/d/${url#gdrive:}/view in a browser and download it"
  warn "  2. save it as $CKPT_DIR/$rel ($size bytes, sha256 $sha)"
  warn "  3. re-run this script with the same options to verify it"
  warn "  Only harmonization needs it; everything else works without it."
done
if [ ${#failures[@]} -gt 0 ] || [ "$hf_rc" -ne 0 ]; then
  for f in "${failures[@]}"; do err "$f"; done
  err "some downloads failed; re-run the same command to retry (finished files are kept)."
  exit 1
fi
if [ -n "$ONLY" ]; then log "done: $ONLY"; else log "done: ${selected[*]}"; fi
