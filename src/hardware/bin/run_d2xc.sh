#!/bin/bash

# Determine repository root from this script directory (3 levels up: bin -> hardware -> src -> root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG_PATH="$REPO_ROOT/src/hardware/difra/resources/config/global.json"

# Read conda env name from JSON using Python
CONDA_ENV=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['conda'])" 2>/dev/null)

if [ -z "$CONDA_ENV" ]; then
  echo "[ERROR] Could not read 'conda' from $CONFIG_PATH"
  exit 1
fi

if ! command -v conda &> /dev/null; then
  echo "[ERROR] 'conda' was not found on PATH. Please ensure conda is initialized in your shell."
  exit 1
fi

echo "Starting D2XC software..."
echo "Using conda environment: $CONDA_ENV"
echo "Repository root: $REPO_ROOT"

# Launch the D2XC GUI using the specified conda environment
conda run -n "$CONDA_ENV" python "$REPO_ROOT/src/hardware/difra/gui/main_app.py" "$@"
