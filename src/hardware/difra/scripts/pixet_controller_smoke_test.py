#!/usr/bin/env python3
"""Smoke test for DiFRA Advacam/Pixet detector controller.

This script exercises the real DiFRA `PixetDetectorController` class using the
ctypes backend and attempts:
1) Detector initialization
2) Single capture
3) Basic output validation/statistics
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow running directly from repository root or arbitrary working directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.difra.hardware.detectors import (
    PixetDetectorController,
    PixetLegacyDetectorController,
    PixetSidecarDetectorController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hardware smoke test via DiFRA PixetDetectorController."
    )
    parser.add_argument(
        "--sdk-path",
        default=r"C:\Program Files\PIXet Pro",
        help="Path to PIXet SDK directory containing pxcore.dll.",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Optional device-id substring to match (e.g. MiniPIX G08-W0299).",
    )
    parser.add_argument(
        "--alias",
        default="PRIMARY",
        help="Logical detector alias used by DiFRA logs/output metadata.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Configured detector width for controller initialization.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Configured detector height for controller initialization.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames to integrate for capture.",
    )
    parser.add_argument(
        "--exposure-s",
        type=float,
        default=0.1,
        help="Exposure time in seconds.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "src" / "hardware" / "difra" / "logs"),
        help="Directory where capture artifacts (.txt/.dsc) are written.",
    )
    parser.add_argument(
        "--backend",
        choices=["ctypes", "pypixet", "sidecar"],
        default="ctypes",
        help="PIXet backend to test.",
    )
    parser.add_argument(
        "--detector-type",
        choices=["Pixet", "DummyDetector"],
        default="Pixet",
        help="Detector type passed to controller config (use DummyDetector for sidecar demo tests).",
    )
    parser.add_argument(
        "--sidecar-host",
        default="127.0.0.1",
        help="Sidecar host (for --backend sidecar).",
    )
    parser.add_argument(
        "--sidecar-port",
        type=int,
        default=51001,
        help="Sidecar port (for --backend sidecar).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_config: dict[str, str] = {
        "pixet_sdk_path": args.sdk_path,
        "type": args.detector_type,
    }
    if args.device_id.strip():
        detector_config["id"] = args.device_id.strip()

    if args.backend == "pypixet":
        controller_cls = PixetLegacyDetectorController
    elif args.backend == "sidecar":
        controller_cls = PixetSidecarDetectorController
        detector_config["pixet_sidecar"] = {
            "host": args.sidecar_host,
            "port": int(args.sidecar_port),
            "timeout_s": 15.0,
        }
    else:
        controller_cls = PixetDetectorController
    controller = controller_cls(
        alias=args.alias,
        size=(args.width, args.height),
        config=detector_config,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = out_dir / f"pixet_controller_smoke_{args.alias.lower()}_{timestamp}"
    txt_path = base_path.with_suffix(".txt")
    dsc_path = base_path.with_suffix(".dsc")

    try:
        print(
            f"[INFO] Initializing detector via DiFRA class: {args.alias} "
            f"(backend={args.backend})"
        )
        initialized = controller.init_detector()
        if not initialized:
            print("[FAIL] init_detector() returned False (detector unavailable or setup issue).")
            return 2

        print(
            f"[INFO] Capturing with frames={args.frames}, exposure_s={args.exposure_s:.3f}"
        )
        captured = controller.capture_point(
            Nframes=max(args.frames, 1),
            Nseconds=max(float(args.exposure_s), 0.0),
            filename_base=str(base_path),
        )
        if not captured:
            print("[FAIL] capture_point() returned False.")
            return 3

        if not txt_path.exists():
            print(f"[FAIL] Capture reported success but file not found: {txt_path}")
            return 4

        data = np.loadtxt(txt_path)
        print("[PASS] Capture completed via PixetDetectorController")
        print(f"[INFO] txt={txt_path}")
        print(f"[INFO] dsc_exists={dsc_path.exists()} path={dsc_path}")
        print(
            "[INFO] stats "
            f"shape={data.shape} min={float(data.min()):.3f} "
            f"max={float(data.max()):.3f} mean={float(data.mean()):.3f} "
            f"sum={float(data.sum()):.3f}"
        )
        return 0
    finally:
        controller.deinit_detector()


if __name__ == "__main__":
    raise SystemExit(main())
