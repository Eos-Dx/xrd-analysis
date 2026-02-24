"""PIXet detector backend using ctypes API wrapper."""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from hardware.difra.hardware.detector_controller_base import DetectorController
from hardware.difra.hardware.pixet_ctypes_api import PxcoreError, PixetCtypesAPI
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class PixetDetectorController(DetectorController):
    def __init__(self, alias, size=(256, 256), config=None):
        self.alias = alias
        self.size = tuple(size)
        self.config = config or {}
        self.dev_id = self.config.get("id")
        self._api = None
        self.device_index = None
        self._stream_thread = None
        self._streaming = threading.Event()

    def init_detector(self):
        pixet_sdk_path = os.environ.get("PIXET_SDK_PATH") or self.config.get(
            "pixet_sdk_path"
        )
        logger.info(
            "Initializing Pixet detector",
            detector=self.alias,
            device_id=self.dev_id,
            pixet_sdk_path=pixet_sdk_path,
        )

        if pixet_sdk_path:
            if not os.path.isdir(pixet_sdk_path):
                logger.error(
                    "Configured PIXET SDK path does not exist",
                    sdk_path=pixet_sdk_path,
                    detector=self.alias,
                    path_exists=False,
                    hint="Check the 'pixet_sdk_path' in your setup configuration file or set PIXET_SDK_PATH environment variable",
                )
                return False
        else:
            logger.warning(
                "No PIXET SDK path configured",
                detector=self.alias,
                hint="Set 'pixet_sdk_path' in detector config or PIXET_SDK_PATH environment variable",
            )
        if not pixet_sdk_path:
            return False

        try:
            self._api = PixetCtypesAPI(Path(pixet_sdk_path))
            self._api.initialize()
            pixet_version = self._api.get_version()
            devices = self._api.list_devices()
        except Exception as e:
            logger.error(
                "Failed to initialize PIXet C API backend",
                detector=self.alias,
                sdk_path=pixet_sdk_path,
                error=str(e),
                exc_info=True,
            )
            self._safe_shutdown()
            return False

        if not devices:
            logger.error("No Pixet devices connected", detector=self.alias)
            self._safe_shutdown()
            return False

        selected = None
        if self.dev_id:
            for dev in devices:
                if self.dev_id in dev.name:
                    selected = dev
                    break
        if selected is None:
            selected = devices[0]
            logger.warning(
                "Configured Pixet device ID not found; using first detected device",
                detector=self.alias,
                requested_device_id=self.dev_id,
                selected_device=selected.name,
            )

        self.device_index = selected.index
        detected_size = (selected.width, selected.height)
        if detected_size != self.size:
            logger.warning(
                "Detector size from hardware differs from config; using hardware size",
                detector=self.alias,
                configured_size=self.size,
                detected_size=detected_size,
            )
            self.size = detected_size

        logger.info(
            "Initialized Pixet detector through C API",
            detector=self.alias,
            pixet_version=pixet_version,
            device_name=selected.name,
            device_index=self.device_index,
            size=self.size,
        )
        return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        filename = f"{filename_base}.txt"
        dsc_filename = f"{filename_base}.dsc"
        if self._api is None or self.device_index is None:
            logger.error("Pixet detector is not initialized", detector=self.alias)
            return False
        try:
            n_frames = max(int(Nframes), 1)
            exposure_seconds = float(Nseconds)
            integrated = None
            for _ in range(n_frames):
                frame = self._api.measure_single_frame(
                    self.device_index, exposure_seconds
                )
                if integrated is None:
                    integrated = frame.astype(np.float64)
                else:
                    integrated += frame
            assert integrated is not None
            np.savetxt(filename, integrated, fmt="%.6f")
            self._write_descriptor_file(
                dsc_filename=dsc_filename,
                exposure_seconds=exposure_seconds,
                n_frames=n_frames,
                data=integrated,
            )
        except Exception as e:
            logger.error(
                "Exception during Pixet acquisition", detector=self.alias, error=str(e)
            )
            return False
        logger.info(
            "Pixet capture successful",
            detector=self.alias,
            frames=Nframes,
            integration_time=Nseconds,
        )
        return True

    def deinit_detector(self):
        try:
            if self._api is not None:
                logger.info("Deinitializing Pixet detector", detector=self.alias)
                self._api.shutdown()
                logger.info("Pixet detector safely deinitialized", detector=self.alias)
        except Exception as e:
            logger.error(
                "Error during Pixet detector deinitialization",
                detector=self.alias,
                error=str(e),
            )
        finally:
            self._api = None
            self.device_index = None

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        self.stop_stream()
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval, frames),
            daemon=True,
        )
        self._stream_thread.start()
        logger.info("Pixet streaming started", detector=self.alias, exposure=exposure)

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            logger.info("Pixet streaming stopped", detector=self.alias)
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        while self._streaming.is_set():
            try:
                if self._api is None or self.device_index is None:
                    frame = None
                else:
                    frame = self._api.measure_single_frame(
                        self.device_index, float(exposure) * max(int(frames), 1)
                    )
                    if frame is not None:
                        frame = frame[: self.size[1], : self.size[0]]
                callback({self.alias: frame})
            except Exception as e:
                logger.warning(
                    "Pixet frame capture error during streaming",
                    detector=self.alias,
                    error=str(e),
                )
                callback({self.alias: None})
            if interval:
                time.sleep(interval)

    def _write_descriptor_file(
        self,
        dsc_filename: str,
        exposure_seconds: float,
        n_frames: int,
        data: np.ndarray,
    ) -> None:
        width, height = int(self.size[0]), int(self.size[1])
        total_time = max(float(exposure_seconds), 0.0) * max(int(n_frames), 1)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dsc_content = (
            "[F0]\n"
            f"Type=i16 [X,Y,C] width={width} height={height}\n"
            "Acq mode=INTEGRATING\n"
            f"Acq time={total_time:.6f}\n"
            f"Frames={n_frames}\n"
            f"Start time={timestamp}\n"
            f"ChipboardID={self.dev_id or self.alias}\n"
            "Interface=USB\n"
            "Pixel size=0.055 mm\n"
            "Layout=1x1\n"
            "# Generated by DiFRA PIXet ctypes backend\n"
            f"# Detector: {self.alias}\n"
            f"# Total counts: {float(data.sum()):.0f}\n"
            f"# Max value: {float(data.max()):.0f}\n"
            f"# Mean value: {float(data.mean()):.2f}\n"
        )
        try:
            with open(dsc_filename, "w", encoding="utf-8") as f:
                f.write(dsc_content)
        except Exception as e:
            logger.warning(
                "Failed to generate Pixet .dsc file",
                detector=self.alias,
                file=dsc_filename,
                error=str(e),
            )

    def _safe_shutdown(self) -> None:
        if self._api is not None:
            try:
                self._api.shutdown()
            except PxcoreError:
                pass
        self._api = None
        self.device_index = None

    def convert_to_container_format(
        self,
        raw_file_path: str,
        container_version: str = "0.2",
    ) -> str:
        raw_path = Path(raw_file_path)
        if container_version == "0.2":
            npy_path = raw_path.with_suffix(".npy")
            if not npy_path.exists():
                try:
                    data = np.loadtxt(raw_path)
                    np.save(npy_path, data)
                    logger.info(
                        "Converted for container v%s",
                        container_version,
                        detector=self.alias,
                        input_file=raw_path.name,
                        output_file=npy_path.name,
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to convert {raw_path} to .npy: {e}")
            else:
                logger.debug(
                    "Container format file already exists",
                    detector=self.alias,
                    file=npy_path.name,
                )
            return str(npy_path)
        raise ValueError(
            f"Detector {self.alias} does not support container version {container_version}"
        )

    def get_raw_file_patterns(self):
        return ["*.txt", "*.dsc"]
