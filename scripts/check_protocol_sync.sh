#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANONICAL="$ROOT_DIR/src/hardware/protocol/hub/v1/hub.proto"
HW_SERVER="$ROOT_DIR/src/hardware/Omniscan/omniscan-hw-server/proto/hub/v1/hub.proto"
ORCHESTRATOR="$ROOT_DIR/src/hardware/Omniscan/omniscan-orchestrator/proto/hub/v1/hub.proto"

for f in "$CANONICAL" "$HW_SERVER" "$ORCHESTRATOR"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing protocol file: $f" >&2
    exit 2
  fi
done

cmp -s "$CANONICAL" "$HW_SERVER" || {
  echo "Protocol drift: hw-server proto differs from canonical" >&2
  exit 1
}

cmp -s "$CANONICAL" "$ORCHESTRATOR" || {
  echo "Protocol drift: orchestrator proto differs from canonical" >&2
  exit 1
}

echo "Protocol sync check passed"
