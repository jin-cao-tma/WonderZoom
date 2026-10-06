#!/usr/bin/env bash
# Start the WonderZoom server (run.py: generation, or --view) with the registered wz-main interpreter.
#
# Usage: bash scripts/run_server.sh [run.py arguments...]     (--wrapper-help: show this text)
#   bash scripts/run_server.sh --example_config config/more_examples/street.yaml
#   bash scripts/run_server.sh --image my_photo.jpg --port 7747
#   bash scripts/run_server.sh --view --example_config config/more_examples/street.yaml
#
# - The interpreter is WZ_MAIN_PYTHON, else the one registered in config/services.local.yaml by
#   scripts/install_env_main.sh (see `python scripts/register_env.py --show`).
# - Keeps the caller's working directory, so relative --image / --example_config / --services_config
#   paths work from anywhere (run.py resolves them against that directory first, then against the
#   repository root, and then changes into the repository root itself).
# - Exports PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and PYTHONUNBUFFERED=1 (unless already
#   set) and passes --services_config <repo>/config/services.yaml unless given.
# - Clears LD_LIBRARY_PATH, as the service workers do, so torch loads the CUDA/cuDNN libraries of
#   its own wheels and not a system toolkit's; set WZ_KEEP_LD_LIBRARY_PATH=1 to keep it.
# The browser UI is served by run.py itself at http://localhost:<port>/ (default port 7747).
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

err() { printf '[run_server] ERROR: %s\n' "$*" >&2; }

case "${1:-}" in
  --wrapper-help)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
    exit 0
    ;;
esac

# --- interpreter ---------------------------------------------------------------------------
registered_main() {
  local boot
  for boot in python3 "${CONDA_PYTHON_EXE:-}"; do
    if [ -n "$boot" ] && command -v "$boot" >/dev/null 2>&1; then
      "$boot" "$ROOT/scripts/register_env.py" --get main 2>/dev/null && return 0
    fi
  done
  # No Python 3 on PATH: read the file written by register_env.py directly.
  [ -f "$ROOT/config/services.local.yaml" ] || return 1
  awk '/^main:/ { m = 1; next } /^[^ #]/ { m = 0 } m && /^[ ]+python:/ { sub(/^[ ]+python:[ ]*/, ""); gsub(/"/, ""); print; exit }' \
    "$ROOT/config/services.local.yaml"
}
PY=${WZ_MAIN_PYTHON:-$(registered_main || true)}
if [ -z "$PY" ] || [ "$PY" = null ]; then
  err "no main interpreter registered. Build the env with: bash scripts/install_env_main.sh"
  err "or register an existing one: python scripts/register_env.py main /path/to/envs/wz-main/bin/python"
  exit 1
fi
if [ ! -x "$PY" ]; then
  err "registered main interpreter $PY does not exist; re-run scripts/install_env_main.sh or register_env.py"
  exit 1
fi

# --- arguments ---------------------------------------------------------------------------
PORT=7747
HOST=127.0.0.1
has_services_config=0
prev=
for a in "$@"; do
  case "$prev" in
    --port) PORT=$a ;;
    --host) HOST=$a ;;
  esac
  case "$a" in
    --port=*) PORT=${a#*=} ;;
    --host=*) HOST=${a#*=} ;;
    --services_config | --services_config=*) has_services_config=1 ;;
  esac
  prev=$a
done
extra=()
if [ "$has_services_config" = 0 ]; then
  extra=(--services_config "$ROOT/config/services.yaml")
fi

# --- environment -------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Unbuffered output, so that a log file (run_server.sh ... > server.log) shows progress as it happens.
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
if [ "${WZ_KEEP_LD_LIBRARY_PATH:-0}" != 1 ]; then
  unset LD_LIBRARY_PATH
fi
# No `cd "$ROOT"`: run.py records the caller's cwd for the relative CLI paths and chdirs itself.

HOSTNAME_F=$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo this-server)
cat <<EOF
[run_server] python: $PY
[run_server] open http://localhost:$PORT/ in a browser once the server reports it is ready.
[run_server] On a remote machine, forward the port from your laptop first:
[run_server]     ssh -N -L $PORT:localhost:$PORT ${USER:-user}@$HOSTNAME_F
EOF
if [ "$HOST" = 0.0.0.0 ]; then
  echo "[run_server] (listening on all interfaces: http://$HOSTNAME_F:$PORT/ also works if the port is reachable)"
fi
exec "$PY" "$ROOT/run.py" "${extra[@]}" "$@"
