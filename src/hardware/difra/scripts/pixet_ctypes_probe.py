#!/usr/bin/env python3
"""
Minimal PIXet C API probe for Python 3.11+ via ctypes (without pypixet).

Usage (Windows):
  python src/hardware/difra/scripts/pixet_ctypes_probe.py ^
    --sdk-path "C:\\Program Files\\PIXet Pro" ^
    --device-index 0 ^
    --exposure-ms 100 ^
    --out frame.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow running the script directly from repository root.
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.difra.hardware.pixet_ctypes_api import PxcoreError, PixetCtypesAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe PIXet C API via ctypes.")
    parser.add_argument(
        "--sdk-path",
        default=os.environ.get("PIXET_SDK_PATH", r"C:\Program Files\PIXet Pro"),
        help="Path containing pxcore.dll and related PIXet SDK files.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only list devices and exit.")
    parser.add_argument("--device-index", type=int, default=0, help="Device index to measure.")
    parser.add_argument("--exposure-ms", type=float, default=100.0, help="Exposure in milliseconds.")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output .txt file (saved using numpy.savetxt).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = PixetCtypesAPI(Path(args.sdk_path))
    try:
        api.initialize()
        print(f"PIXet version: {api.get_version()}")
        devices = api.list_devices()
        print(f"Devices found: {len(devices)}")
        for dev in devices:
            print(f"  [{dev.index}] {dev.name} ({dev.width}x{dev.height})")

        if args.list_only:
            return 0
        if not devices:
            raise PxcoreError("No devices detected.")

        frame = api.measure_single_frame(args.device_index, args.exposure_ms / 1000.0)
        print(
            f"Measured frame: shape={frame.shape}, dtype={frame.dtype}, "
            f"min={int(frame.min())}, max={int(frame.max())}, sum={int(frame.sum())}"
        )

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, frame, fmt="%d")
            print(f"Saved frame to: {out_path}")
        return 0
    finally:
        api.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
