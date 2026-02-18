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
else:
    print("")
PY
    )
    if [ -n "$DIFRA_LEGACY_ENV" ]; then
      export DIFRA_LEGACY_ENV
      echo "[INFO] Using legacy env: $DIFRA_LEGACY_ENV"
    else
      echo "[ERROR] Legacy sidecar env 'ulster37' not found."
      echo "[ERROR] Set DIFRA_LEGACY_ENV=ulster37 or DIFRA_LEGACY_PYTHON to Python 3.7."
      exit 1
    fi
  else
    echo "[INFO] Using requested legacy env: $DIFRA_LEGACY_ENV"
  fi
  LEGACY_PY_VERSION="$(
    conda run --no-capture-output -n "$DIFRA_LEGACY_ENV" \
      python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" \
      2>/dev/null | tail -n 1
  )"
  if [ "$LEGACY_PY_VERSION" != "3.7" ]; then
    echo "[ERROR] Legacy env '$DIFRA_LEGACY_ENV' must be Python 3.7, found '$LEGACY_PY_VERSION'."
    exit 1
  fi
else
  LEGACY_PY_VERSION="$("$DIFRA_LEGACY_PYTHON" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null || true)"
  if [ "$LEGACY_PY_VERSION" != "3.7" ]; then
    echo "[ERROR] DIFRA_LEGACY_PYTHON must be Python 3.7, found '$LEGACY_PY_VERSION'."
    exit 1
  fi
  echo "[INFO] Using explicit legacy python: $DIFRA_LEGACY_PYTHON"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFRA_EXPECT_STAGE_TYPE="${DIFRA_EXPECT_STAGE_TYPE:-Kinesis}"
export DIFRA_EXPECT_STAGE_CLASS="${DIFRA_EXPECT_STAGE_CLASS:-XYStageLibController}"
export DIFRA_EXPECT_DETECTOR_CLASS="${DIFRA_EXPECT_DETECTOR_CLASS:-PixetSidecarDetectorController}"

echo "[INFO] Running hardware stack tests in GUI env: $GUI_ENV"
echo "[INFO] Expected route: stage_type=$DIFRA_EXPECT_STAGE_TYPE stage_class=$DIFRA_EXPECT_STAGE_CLASS detector_class=$DIFRA_EXPECT_DETECTOR_CLASS"
conda run --live-stream --no-capture-output -n "$GUI_ENV" \
  python -m pytest -q -s \
  "$REPO_ROOT/src/hardware/difra/tests/test_detector_integration_timing_e2e.py" \
  "$REPO_ROOT/src/hardware/difra/tests/manual_hardware_real_legacy_e2e.py"
