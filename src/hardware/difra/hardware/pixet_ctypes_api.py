"""PIXet C API bindings for Windows via ctypes.

This module provides a minimal, Python-version-independent bridge to PIXet
through pxcore.dll, avoiding pypixet.pyd runtime constraints.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

PXC_TRG_NO = 0


class PxcoreError(RuntimeError):
    """Raised when PIXet C API operations fail."""


@dataclass(frozen=True)
class PixetDeviceInfo:
    index: int
    name: str
    width: int
    height: int


class PixetCtypesAPI:
    def __init__(self, sdk_path: Path) -> None:
        if os.name != "nt":
            raise PxcoreError("PIXet ctypes backend is supported on Windows only.")

        self.sdk_path = Path(sdk_path)
        self.dll_path = self.sdk_path / "pxcore.dll"
        if not self.dll_path.exists():
            raise PxcoreError(f"pxcore.dll not found: {self.dll_path}")

        self._dll_dir_ctx = None
        self.lib = None
        self.initialized = False

    def _bind(self) -> None:
        assert self.lib is not None

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
        if self.lib is None:
            return ""
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

    def _resolve_logs_dir(self) -> Path:
        env_logs_dir = os.environ.get("PIXET_LOGS_DIR")
        candidates: list[Path] = []
        if env_logs_dir:
            candidates.append(Path(env_logs_dir))
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "PixetPro" / "logs")
            candidates.append(Path(local_app_data) / "PIXet Pro" / "logs")
        candidates.append(self.sdk_path / "logs")

        candidates.append(Path.home() / ".pixet" / "logs")

        attempted: list[str] = []
        for candidate in candidates:
            attempted.append(str(candidate))
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue

        attempted_paths = ", ".join(attempted)
        raise PxcoreError(
            f"Unable to create writable PIXet log directory. Tried: {attempted_paths}"
        )
    def _resolve_configs_dir(self) -> Path:
        env_configs_dir = os.environ.get("PIXET_CONFIGS_DIR")
        candidates: list[Path] = []
        if env_configs_dir:
            candidates.append(Path(env_configs_dir))

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "PixetPro" / "configs")
            candidates.append(Path(local_app_data) / "PIXet Pro" / "configs")
        candidates.append(self.sdk_path / "configs")

        attempted: list[str] = []
        for candidate in candidates:
            attempted.append(str(candidate))
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue

        attempted_paths = ", ".join(attempted)
        raise PxcoreError(
            f"Unable to create/read PIXet configs directory. Tried: {attempted_paths}"
        )

    def initialize(self) -> None:
        if self.initialized:
            return

        self._dll_dir_ctx = os.add_dll_directory(str(self.sdk_path))
        self.lib = ctypes.CDLL(str(self.dll_path))
        self._bind()
        configs_dir = self._resolve_configs_dir()
        logs_dir = self._resolve_logs_dir()
        self._check_rc(
            self.lib.pxcSetDirectories(
                str(configs_dir).encode("utf-8"),
                str(logs_dir).encode("utf-8"),
            ),
            "pxcSetDirectories",
        )
        self._check_rc(self.lib.pxcInitialize(0, None), "pxcInitialize")
        self.initialized = True

    def shutdown(self) -> None:
        if self.initialized and self.lib is not None:
            try:
                self.lib.pxcExit()
            finally:
                self.initialized = False
        if self._dll_dir_ctx is not None:
            self._dll_dir_ctx.close()
            self._dll_dir_ctx = None

    def get_version(self) -> str:
        if self.lib is None:
            raise PxcoreError("pxcore is not initialized")
        buf = ctypes.create_string_buffer(128)
        self._check_rc(self.lib.pxcGetVersion(buf, ctypes.c_uint(len(buf))), "pxcGetVersion")
        return buf.value.decode("utf-8", errors="replace").strip()

    def list_devices(self) -> List[PixetDeviceInfo]:
        if self.lib is None:
            raise PxcoreError("pxcore is not initialized")

        self._check_rc(self.lib.pxcRefreshDevices(), "pxcRefreshDevices")
        count = self._check_rc(self.lib.pxcGetDevicesCount(), "pxcGetDevicesCount")
        result: List[PixetDeviceInfo] = []
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
            result.append(
                PixetDeviceInfo(
                    index=idx,
                    name=name_buf.value.decode("utf-8", errors="replace").strip(),
                    width=int(width.value),
                    height=int(height.value),
                )
            )
        return result

    def find_device(self, id_substring: str) -> Optional[PixetDeviceInfo]:
        needle = (id_substring or "").strip()
        if not needle:
            return None
        for device in self.list_devices():
            if needle in device.name:
                return device
        return None

    def measure_single_frame(self, device_index: int, exposure_seconds: float) -> np.ndarray:
        if self.lib is None:
            raise PxcoreError("pxcore is not initialized")

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
        self._check_rc(
            self.lib.pxcMeasureSingleFrame(
                ctypes.c_uint(device_index),
                ctypes.c_double(float(exposure_seconds)),
                frame,
                ctypes.byref(size),
                ctypes.c_uint(PXC_TRG_NO),
            ),
            "pxcMeasureSingleFrame",
        )

        used = int(size.value)
        data = np.ctypeslib.as_array(frame)[:used].astype(np.uint16, copy=True)
        return data.reshape((int(height.value), int(width.value)))
