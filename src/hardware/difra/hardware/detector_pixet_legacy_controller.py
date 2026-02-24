"""Legacy PIXet controller using vendor pypixet bindings."""

import os
import threading
import time
from pathlib import Path

import numpy as np

from hardware.difra.hardware.detector_controller_base import DetectorController
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class PixetLegacyDetectorController(DetectorController):
    """Legacy PIXet controller using vendor `pypixet` bindings."""

    def __init__(self, alias, size=(256, 256), config=None):
        self.alias = alias
        self.size = tuple(size)
        self.config = config or {}
        self.dev_id = self.config.get("id")
        self.detector = None
        self.pixet = None
        self._stream_thread = None
        self._streaming = threading.Event()

    def init_detector(self):
        import sys

        pixet_sdk_path = os.environ.get("PIXET_SDK_PATH") or self.config.get(
            "pixet_sdk_path"
        )
        logger.info(
            "Initializing legacy Pixet detector (pypixet)",
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
                )
                return False
            os.environ["PATH"] = pixet_sdk_path + os.pathsep + os.environ.get("PATH", "")
            if pixet_sdk_path not in sys.path:
                sys.path.insert(0, pixet_sdk_path)
        else:
            logger.warning(
                "No PIXET SDK path configured",
                detector=self.alias,
                hint="Set 'pixet_sdk_path' in detector config or PIXET_SDK_PATH",
            )
            return False

        try:
            import pypixet
        except ImportError as e:
            logger.error(
                "Failed to import pypixet",
                detector=self.alias,
                error=str(e),
                hint="Use Python 3.7 runtime with PIXet SDK python bindings installed",
            )
            return False

        original_cwd = os.getcwd()
        try:
            os.chdir(pixet_sdk_path)
            pypixet.start()
        finally:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)

        pixet = pypixet.pixet
        devices = pixet.devices()
        if not devices or devices[0].fullName() == "FileDevice 0":
            logger.error("No Pixet devices connected", detector=self.alias)
            try:
                pixet.exitPixet()
                pypixet.exit()
            except Exception:
                pass
            return False

        selected = None
        for dev in devices:
            name = dev.fullName()
            if self.dev_id and self.dev_id in name:
                selected = dev
                break
        if selected is None:
            selected = devices[0]
            logger.warning(
                "Configured Pixet device ID not found; using first detected device",
                detector=self.alias,
                requested_device_id=self.dev_id,
                selected_device=selected.fullName(),
            )

        self.detector = selected
        self.pixet = pixet
        logger.info(
            "Initialized legacy Pixet detector",
            detector=self.alias,
            device_name=selected.fullName(),
        )
        return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        filename = f"{filename_base}.txt"
        if self.detector is None or self.pixet is None:
            logger.error("Legacy Pixet detector is not initialized", detector=self.alias)
            return False
        try:
            rc = self.detector.doSimpleIntegralAcquisition(
                max(int(Nframes), 1),
                float(Nseconds),
                self.pixet.PX_FTYPE_AUTODETECT,
                filename,
            )
        except Exception as e:
            logger.error(
                "Exception during legacy Pixet acquisition",
                detector=self.alias,
                error=str(e),
            )
            return False
        if rc != 0:
            err = ""
            try:
                err = self.detector.lastError()
            except Exception:
                pass
            logger.error(
                "Legacy Pixet capture error",
                detector=self.alias,
                return_code=rc,
                error=err,
            )
            return False
        logger.info(
            "Legacy Pixet capture successful",
            detector=self.alias,
            frames=Nframes,
            integration_time=Nseconds,
        )
        return True

    def deinit_detector(self):
        if self.pixet:
            try:
                self.pixet.exitPixet()
                import pypixet

                pypixet.exit()
            except Exception as e:
                logger.error(
                    "Error during legacy Pixet detector deinitialization",
                    detector=self.alias,
                    error=str(e),
                )
            finally:
                self.pixet, self.detector = None, None

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        self.stop_stream()
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval, frames),
            daemon=True,
        )
        self._stream_thread.start()
        logger.info(
            "Legacy Pixet streaming started",
            detector=self.alias,
            exposure=exposure,
        )

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            logger.info("Legacy Pixet streaming stopped", detector=self.alias)
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        import tempfile

        while self._streaming.is_set():
            tmpdir = tempfile.mkdtemp()
            tmpfile = os.path.join(tmpdir, f"stream_{self.alias}.txt")
            try:
                rc = self.detector.doSimpleIntegralAcquisition(
                    max(int(frames), 1),
                    float(exposure),
                    self.pixet.PX_FTYPE_AUTODETECT,
                    tmpfile,
                )
                if rc != 0:
                    callback({self.alias: None})
                else:
                    frame = np.loadtxt(tmpfile)
                    frame = frame[: self.size[1], : self.size[0]]
                    callback({self.alias: frame})
            except Exception as e:
                logger.warning(
                    "Legacy Pixet frame capture error during streaming",
                    detector=self.alias,
                    error=str(e),
                )
                callback({self.alias: None})
            finally:
                try:
                    os.remove(tmpfile)
                    os.rmdir(tmpdir)
                except Exception:
                    pass
            if interval:
                time.sleep(interval)

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
                except Exception as e:
                    raise RuntimeError(f"Failed to convert {raw_path} to .npy: {e}")
            return str(npy_path)
        raise ValueError(
            f"Detector {self.alias} does not support container version {container_version}"
        )

    def get_raw_file_patterns(self):
        return ["*.txt", "*.dsc"]
