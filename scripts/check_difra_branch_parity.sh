#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! git rev-parse --verify difra_h5 >/dev/null 2>&1; then
  echo "Branch not found: difra_h5" >&2
  exit 2
fi
if ! git rev-parse --verify omniscan_sd >/dev/null 2>&1; then
  echo "Branch not found: omniscan_sd" >&2
  exit 2
fi

if git diff --quiet difra_h5..omniscan_sd -- src/hardware/difra; then
  echo "DiFRA parity check passed (difra_h5 == omniscan_sd for src/hardware/difra)"
else
  echo "DiFRA parity check failed: src/hardware/difra differs between difra_h5 and omniscan_sd" >&2
  git diff --name-status difra_h5..omniscan_sd -- src/hardware/difra
  exit 1
fi
