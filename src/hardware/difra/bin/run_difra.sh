#!/bin/bash

# Keep this entry-point stable; delegate to dual-env launcher.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_difra_dual_env.sh" "$@"
