#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

GLOBAL_CONFIG="$REPO_ROOT/src/hardware/difra/resources/config/global.json"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] 'conda' not found on PATH."
  exit 1
fi

GUI_ENV="${DIFRA_GUI_ENV:-}"
if [ -z "$GUI_ENV" ]; then
  GUI_ENV=$(python3 - "$GLOBAL_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
if cfg.exists():
    try:
        data = json.loads(cfg.read_text())
        print(str(data.get("conda", "")).strip())
    except Exception:
        print("")
else:
    print("")
PY
  )
fi
if [ -z "$GUI_ENV" ]; then
  GUI_ENV="eosdx13"
fi

if [ -z "${DIFRA_LEGACY_PYTHON:-}" ]; then
  if [ -z "${DIFRA_LEGACY_ENV:-}" ]; then
    CONDA_ENVS_JSON="$(conda env list --json)"
    DIFRA_LEGACY_ENV=$(
      python3 - "$CONDA_ENVS_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
names = {Path(p).name for p in payload.get("envs", [])}
if "ulster37" in names:
    print("ulster37")
elif "ulster38" in names:
    print("ulster38")
else:
    print("")
PY
    )
    if [ -n "$DIFRA_LEGACY_ENV" ]; then
      export DIFRA_LEGACY_ENV
      echo "[INFO] Using legacy env: $DIFRA_LEGACY_ENV"
    else
      echo "[WARN] ulster38/ulster37 not found; tests will use current Python unless DIFRA_LEGACY_PYTHON is set."
    fi
  else
    echo "[INFO] Using requested legacy env: $DIFRA_LEGACY_ENV"
  fi
else
  echo "[INFO] Using explicit legacy python: $DIFRA_LEGACY_PYTHON"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[INFO] Running hardware stack tests in GUI env: $GUI_ENV"
conda run --live-stream --no-capture-output -n "$GUI_ENV" \
  python -m pytest -q -s \
  "$REPO_ROOT/src/hardware/difra/tests/test_detector_integration_timing_e2e.py" \
  "$REPO_ROOT/src/hardware/difra/tests/manual_hardware_real_legacy_e2e.py"
