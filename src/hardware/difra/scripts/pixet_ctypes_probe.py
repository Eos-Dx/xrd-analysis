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
import ctypes
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

PXC_TRG_NO = 0


class PxcoreError(RuntimeError):
    pass


class PxcoreApi:
    def __init__(self, sdk_path: Path) -> None:
        if os.name != "nt":
            raise PxcoreError("This script supports Windows only.")

        self.sdk_path = sdk_path
        self.dll_path = sdk_path / "pxcore.dll"
        if not self.dll_path.exists():
            raise PxcoreError(f"pxcore.dll not found: {self.dll_path}")

        # Make dependent DLLs (AdvsSdk.dll, hw libs, runtime DLLs) visible.
        self._dll_dir_ctx = os.add_dll_directory(str(self.sdk_path))
        self.lib = ctypes.CDLL(str(self.dll_path))
        self._bind()

    def _bind(self) -> None:
        self.lib.pxcSetDirectories.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.pxcSetDirectories.restype = ctypes.c_int

        self.lib.pxcInitialize.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.pxcInitialize.restype = ctypes.c_int

        self.lib.pxcExit.argtypes = []
        self.lib.pxcExit.restype = ctypes.c_int

        self.lib.pxcGetVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        self.lib.pxcGetVersion.restype = ctypes.c_int

        self.lib.pxcRefreshDevices.argtypes = []
        self.lib.pxcRefreshDevices.restype = ctypes.c_int

        self.lib.pxcGetDevicesCount.argtypes = []
        self.lib.pxcGetDevicesCount.restype = ctypes.c_int

        self.lib.pxcGetDeviceName.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_uint]
        self.lib.pxcGetDeviceName.restype = ctypes.c_int

        self.lib.pxcGetDeviceDimensions.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.pxcGetDeviceDimensions.restype = ctypes.c_int

        self.lib.pxcMeasureSingleFrame.argtypes = [
            ctypes.c_uint,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_uint,
        ]
        self.lib.pxcMeasureSingleFrame.restype = ctypes.c_int

        self.lib.pxcGetLastError.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        self.lib.pxcGetLastError.restype = ctypes.c_int

    def _last_error(self) -> str:
        buf = ctypes.create_string_buffer(2048)
        try:
            self.lib.pxcGetLastError(buf, ctypes.c_uint(len(buf)))
            return buf.value.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _check_rc(self, rc: int, fn: str) -> int:
        if rc < 0:
            err = self._last_error()
            suffix = f" | last_error={err}" if err else ""
            raise PxcoreError(f"{fn} failed rc={rc}{suffix}")
        return rc

    def initialize(self) -> None:
        logs_dir = self.sdk_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self._check_rc(
            self.lib.pxcSetDirectories(
                str(self.sdk_path).encode("utf-8"),
                str(logs_dir).encode("utf-8"),
            ),
            "pxcSetDirectories",
        )
        self._check_rc(self.lib.pxcInitialize(0, None), "pxcInitialize")

    def shutdown(self) -> None:
        try:
            self.lib.pxcExit()
        finally:
            self._dll_dir_ctx.close()

    def get_version(self) -> str:
        buf = ctypes.create_string_buffer(128)
        self._check_rc(self.lib.pxcGetVersion(buf, ctypes.c_uint(len(buf))), "pxcGetVersion")
        return buf.value.decode("utf-8", errors="replace").strip()

    def list_devices(self) -> List[Tuple[int, str, int, int]]:
        self._check_rc(self.lib.pxcRefreshDevices(), "pxcRefreshDevices")
        count = self._check_rc(self.lib.pxcGetDevicesCount(), "pxcGetDevicesCount")
        devices: List[Tuple[int, str, int, int]] = []
        for idx in range(count):
            name_buf = ctypes.create_string_buffer(256)
            self._check_rc(
                self.lib.pxcGetDeviceName(ctypes.c_uint(idx), name_buf, ctypes.c_uint(len(name_buf))),
                "pxcGetDeviceName",
            )
            width = ctypes.c_uint(0)
            height = ctypes.c_uint(0)
            self._check_rc(
                self.lib.pxcGetDeviceDimensions(
                    ctypes.c_uint(idx), ctypes.byref(width), ctypes.byref(height)
                ),
                "pxcGetDeviceDimensions",
            )
            devices.append(
                (
                    idx,
                    name_buf.value.decode("utf-8", errors="replace").strip(),
                    int(width.value),
                    int(height.value),
                )
            )
        return devices

    def measure_single_frame(self, device_index: int, exposure_ms: float) -> np.ndarray:
        width = ctypes.c_uint(0)
        height = ctypes.c_uint(0)
        self._check_rc(
            self.lib.pxcGetDeviceDimensions(
                ctypes.c_uint(device_index), ctypes.byref(width), ctypes.byref(height)
            ),
            "pxcGetDeviceDimensions",
        )
        n_pixels = int(width.value) * int(height.value)
        if n_pixels <= 0:
            raise PxcoreError(f"Invalid detector dimensions: {width.value}x{height.value}")

        frame = (ctypes.c_ushort * n_pixels)()
        size = ctypes.c_uint(n_pixels)
        exposure_s = float(exposure_ms) / 1000.0

        self._check_rc(
            self.lib.pxcMeasureSingleFrame(
                ctypes.c_uint(device_index),
                ctypes.c_double(exposure_s),
                frame,
                ctypes.byref(size),
                ctypes.c_uint(PXC_TRG_NO),
            ),
            "pxcMeasureSingleFrame",
        )

        used = int(size.value)
        data = np.ctypeslib.as_array(frame)[:used].astype(np.uint16, copy=True)
        return data.reshape((int(height.value), int(width.value)))


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
    api = PxcoreApi(Path(args.sdk_path))
    try:
        api.initialize()
        print(f"PIXet version: {api.get_version()}")
        devices = api.list_devices()
        print(f"Devices found: {len(devices)}")
        for idx, name, w, h in devices:
            print(f"  [{idx}] {name} ({w}x{h})")

        if args.list_only:
            return 0
        if not devices:
            raise PxcoreError("No devices detected.")

        frame = api.measure_single_frame(args.device_index, args.exposure_ms)
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
