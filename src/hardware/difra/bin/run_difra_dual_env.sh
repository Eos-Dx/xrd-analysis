#!/bin/bash

set -euo pipefail

# Determine repository root (four levels up: bin -> difra -> hardware -> src -> root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="$REPO_ROOT/src/hardware/difra/resources/config/global.json"
MAIN_CONFIG_PATH="$REPO_ROOT/src/hardware/difra/resources/config/main.json"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] 'conda' was not found on PATH. Please ensure conda is initialized in your shell."
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

GUI_ENV="${DIFRA_GUI_ENV:-}"
if [ -z "$GUI_ENV" ]; then
  GUI_ENV=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['conda'])" 2>/dev/null || true)
fi
if [ -z "$GUI_ENV" ]; then
  GUI_ENV="eosdx13"
fi

SIDECAR_ENV="${DIFRA_SIDECAR_ENV:-ulster37}"
if ! conda run --live-stream --no-capture-output -n "$SIDECAR_ENV" python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
  echo "[ERROR] Sidecar env '$SIDECAR_ENV' is not available."
  echo "[ERROR] Install/create the legacy env (expected: ulster37) or set DIFRA_SIDECAR_ENV."
  exit 1
fi
SIDECAR_HOST="${PIXET_SIDECAR_HOST:-127.0.0.1}"
SIDECAR_PORT="${PIXET_SIDECAR_PORT:-51001}"
GRPC_ENV="${DIFRA_GRPC_ENV:-$GUI_ENV}"
GRPC_HOST="${DIFRA_GRPC_HOST:-127.0.0.1}"
GRPC_PORT="${DIFRA_GRPC_PORT:-50061}"
GRPC_CONFIG="${DIFRA_GRPC_CONFIG:-}"
if [ -n "${HARDWARE_CLIENT_MODE:-}" ] && [ "${HARDWARE_CLIENT_MODE}" != "grpc" ]; then
  echo "[WARN] HARDWARE_CLIENT_MODE=${HARDWARE_CLIENT_MODE} overridden to grpc"
fi
CLIENT_MODE="grpc"

if [ -z "$GRPC_CONFIG" ]; then
  GRPC_CONFIG=$(
    python3 - "$CONFIG_PATH" "$MAIN_CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

global_cfg = Path(sys.argv[1])
main_cfg = Path(sys.argv[2])
chosen = main_cfg

try:
    if global_cfg.exists():
        data = json.loads(global_cfg.read_text())
        setup = str(data.get("default_setup") or "").strip()
        if setup:
            setup_path = global_cfg.parent / "setups" / f"{setup}.json"
            if setup_path.exists():
                chosen = setup_path
except Exception:
    pass

print(chosen)
PY
  )
fi

wait_for_port() {
  python3 - "$1" "$2" "$3" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
label = sys.argv[3]
last_error = None
for _ in range(100):
    try:
        with socket.create_connection((host, port), timeout=0.25):
            print(f"[INFO] {label} ready at {host}:{port}")
            sys.exit(0)
    except OSError as exc:
        last_error = exc
        time.sleep(0.1)
print(f"[ERROR] {label} did not become ready at {host}:{port}: {last_error}")
sys.exit(1)
PY
}

kill_local_listener_on_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  case "$host" in
    127.0.0.1|localhost|::1|0.0.0.0)
      ;;
    *)
      echo "[WARN] ${label} host is non-local (${host}); skipping forced restart."
      return 0
      ;;
  esac

  if ! command -v lsof >/dev/null 2>&1; then
    echo "[WARN] lsof is unavailable; cannot force-restart ${label} on port ${port}."
    return 0
  fi

  local pids
  pids="$( (lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true) | tr '\n' ' ' | sed 's/[[:space:]]*$//' )"
  if [ -z "$pids" ]; then
    return 0
  fi

  echo "[INFO] Restarting ${label}: killing existing listener(s) on ${host}:${port} -> ${pids}"
  kill $pids >/dev/null 2>&1 || true
  sleep 0.2
  pids="$( (lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true) | tr '\n' ' ' | sed 's/[[:space:]]*$//' )"
  if [ -n "$pids" ]; then
    kill -9 $pids >/dev/null 2>&1 || true
  fi
}

SIDECAR_CMD=(
  conda run --live-stream --no-capture-output -n "$SIDECAR_ENV" \
  python -u "$REPO_ROOT/src/hardware/difra/scripts/pixet_sidecar_server.py" \
  --host "$SIDECAR_HOST" --port "$SIDECAR_PORT" --owner-pid "$$"
)
GRPC_CMD=(
  conda run --live-stream --no-capture-output -n "$GRPC_ENV" \
  python -u "$REPO_ROOT/src/hardware/difra/grpc_server/server.py" \
  --host "$GRPC_HOST" --port "$GRPC_PORT" --config "$GRPC_CONFIG"
)

SIDECAR_PID=""
GRPC_PID=""

kill_local_listener_on_port "$SIDECAR_HOST" "$SIDECAR_PORT" "Detector sidecar"
kill_local_listener_on_port "$GRPC_HOST" "$GRPC_PORT" "DiFRA gRPC"

echo "[INFO] Starting sidecar env=$SIDECAR_ENV endpoint=${SIDECAR_HOST}:${SIDECAR_PORT}"
"${SIDECAR_CMD[@]}" &
SIDECAR_PID=$!

echo "[INFO] Starting gRPC env=$GRPC_ENV endpoint=${GRPC_HOST}:${GRPC_PORT} config=${GRPC_CONFIG}"
"${GRPC_CMD[@]}" &
GRPC_PID=$!

cleanup() {
  if [ -n "${SIDECAR_PID:-}" ] && kill -0 "$SIDECAR_PID" >/dev/null 2>&1; then
    kill "$SIDECAR_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${GRPC_PID:-}" ] && kill -0 "$GRPC_PID" >/dev/null 2>&1; then
    kill "$GRPC_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_port "$SIDECAR_HOST" "$SIDECAR_PORT" "Detector sidecar"
wait_for_port "$GRPC_HOST" "$GRPC_PORT" "DiFRA gRPC server"

export PIXET_BACKEND=sidecar
export DETECTOR_BACKEND=sidecar
export PIXET_SIDECAR_HOST="$SIDECAR_HOST"
export PIXET_SIDECAR_PORT="$SIDECAR_PORT"
export HARDWARE_CLIENT_MODE="$CLIENT_MODE"
export DIFRA_GRPC_HOST="$GRPC_HOST"
export DIFRA_GRPC_PORT="$GRPC_PORT"

echo "[INFO] Starting DiFRA GUI env=$GUI_ENV mode=$HARDWARE_CLIENT_MODE grpc=${DIFRA_GRPC_HOST}:${DIFRA_GRPC_PORT} detector_backend=sidecar"
conda run --live-stream --no-capture-output -n "$GUI_ENV" python -u "$REPO_ROOT/src/hardware/difra/gui/main_app.py" "$@"
