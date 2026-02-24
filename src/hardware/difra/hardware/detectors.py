# detectors.py
import json
import os
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from hardware.difra.hardware.pixet_ctypes_api import PxcoreError, PixetCtypesAPI
from hardware.difra.utils.logger import (
    get_module_logger,
    log_hardware_state,
    with_logging,
)

logger = get_module_logger(__name__)


class DetectorController(ABC):
    """Abstract base class for all detector controllers."""

    @abstractmethod
    def init_detector(self):
        pass

    @abstractmethod
    def capture_point(self, Nframes, Nseconds, filename_base):
        pass

    @abstractmethod
    def deinit_detector(self):
        pass

    @abstractmethod
    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        pass

    @abstractmethod
    def stop_stream(self):
        pass
    
    @abstractmethod
    def convert_to_container_format(self, raw_file_path: str, container_version: str = "0.2") -> str:
        """Convert detector raw output to container format.
        
        Detectors are responsible for converting their raw output format to the format
        required by the active container version. This enables version-independent and
        detector-independent container workflows.
        
        Args:
            raw_file_path: Path to detector raw output file (e.g., .txt for Advacam)
            container_version: Container schema version (default: "0.2")
        
        Returns:
            Path to converted file in container format (e.g., .npy for v0.2)
        
        Raises:
            ValueError: If container version is not supported
            RuntimeError: If conversion fails
        
        Example:
            For container v0.2 (Advacam detector):
            - Input: "measurement_001.txt" (ASCII data)
            - Output: "measurement_001.npy" (numpy binary)
        """
        pass
    
    def get_raw_file_patterns(self):
        """Return list of glob patterns for detector raw output files.
        
        Used for archiving raw detector files after container generation.
        
        Returns:
            List of glob patterns (e.g., ['*.txt', '*.dsc'] for Advacam)
        
        Example:
            Advacam detectors: ['*.txt', '*.dsc']
            Bruker detectors: ['*.raw', '*.brml']
        """
        return []
class DummyDetectorController:
    def __init__(self, alias="DUMMY", size=(256, 256)):
        self.alias = alias  # Unique name from config, e.g. "DUMMY_DETECTOR_1"
        self.size = size  # (width, height), from config
        self._stream_thread = None
        self._streaming = threading.Event()
        logger.debug(f"Initialized DummyDetectorController", detector=alias, size=size)

    @log_hardware_state("dev")
    @with_logging("detector_init")
    def init_detector(self):
        logger.detector_event(self.alias, "initialization started")
        logger.detector_event(self.alias, "initialization completed")
        return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        # Simulate Pixet-style integrated acquisition across Nframes.
        filename = f"{filename_base}.txt"
        self._dummy_acquire(filename, Nseconds, Nframes)
        return True

    def _dummy_acquire(self, filename, Nseconds, Nframes=1):
        target_duration_s = max(float(Nseconds), 0.0) * max(int(Nframes), 1)
        started_at = time.perf_counter()
        width, height = self.size

        # Sum frames to emulate an integrated image (caller may divide by Nframes to average)
        integrated = np.zeros((height, width), dtype=float)
        for _ in range(max(int(Nframes), 1)):
            x, y = np.arange(width), np.arange(height)
            X, Y = np.meshgrid(x, y)
            x0, y0 = np.random.uniform(0, width), np.random.uniform(0, height)
            sigma = np.random.uniform(5, min(width, height) / 4)
            amp = np.random.uniform(1e5, 2e6)
            frame = amp * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)))
            frame += np.random.normal(scale=amp * 0.1, size=frame.shape)
            integrated += frame

        # Save .txt file (ASCII data)
        np.savetxt(filename, integrated, fmt="%.6f")
        logger.file_operation(
            "save",
            filename,
            detector=self.alias,
            integration_time=Nseconds,
            frames=Nframes,
        )
        
        # Generate fake .dsc file (descriptor) to mimic real Advacam/Pixet detector
        dsc_filename = filename.replace(".txt", ".dsc")
        self._generate_fake_dsc(dsc_filename, Nseconds, Nframes, integrated)

        # Keep total wall-clock close to requested integration duration.
        elapsed = time.perf_counter() - started_at
        remaining = target_duration_s - elapsed
        if remaining > 0:
            # Use a short active tail to reduce sleep overshoot jitter on 1s captures.
            if remaining > 0.003:
                time.sleep(remaining - 0.0015)
            while (time.perf_counter() - started_at) < target_duration_s:
                pass
    
    def _generate_fake_dsc(self, dsc_filename, Nseconds, Nframes, data):
        """Generate a fake .dsc descriptor file to mimic real Advacam detector output.
        
        Args:
            dsc_filename: Path to .dsc file
            Nseconds: Integration time per frame
            Nframes: Number of frames
            data: The integrated data array
        """
        import datetime
        
        width, height = self.size
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time = float(Nseconds) * max(int(Nframes), 1)
        
        # Fake descriptor content mimicking Pixet format
        dsc_content = f"""[F0]
Type=i16 [X,Y,C] width={width} height={height}
Acq mode=INTEGRATING
Acq time={total_time:.6f}
Frames={Nframes}
Start time={timestamp}
ChipboardID=DEMO-{self.alias}
Interface=USB
Pixel size=0.055 mm
Layout=1x1
# DEMO MODE - Fake descriptor file
# Generated by DummyDetectorController
# Detector: {self.alias}
# Total counts: {data.sum():.0f}
# Max value: {data.max():.0f}
# Mean value: {data.mean():.2f}
"""
        
        try:
            with open(dsc_filename, 'w') as f:
                f.write(dsc_content)
            logger.file_operation(
                "save",
                dsc_filename,
                detector=self.alias,
                integration_time=Nseconds,
                frames=Nframes,
                file_type="descriptor"
            )
        except Exception as e:
            logger.warning(f"Failed to generate .dsc file: {e}", detector=self.alias)

    @with_logging("detector_deinit")
    def deinit_detector(self):
        logger.detector_event(self.alias, "deinitialization completed")

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        self.stop_stream()
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval),
            daemon=True,
        )
        self._stream_thread.start()
        logger.detector_event(self.alias, "streaming started", exposure=exposure)

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            logger.detector_event(self.alias, "streaming stopped")
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval):
        width, height = self.size
        while self._streaming.is_set():
            # Generate a random 2D Gaussian frame for real-time visualization
            x, y = np.arange(width), np.arange(height)
            X, Y = np.meshgrid(x, y)
            x0, y0 = np.random.uniform(0, width), np.random.uniform(0, height)
            sigma = np.random.uniform(5, min(width, height) / 4)
            amp = np.random.uniform(1e5, 2e6)
            frame = amp * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)))
            frame += np.random.normal(scale=amp * 0.1, size=frame.shape)
            # Callback with correct alias
            callback({self.alias: frame})
            time.sleep(exposure)
            if interval:
                time.sleep(interval)
    
    def convert_to_container_format(self, raw_file_path: str, container_version: str = "0.2") -> str:
        """Convert Advacam .txt format to container format.
        
        For container v0.2: converts ASCII .txt to binary .npy
        
        Args:
            raw_file_path: Path to .txt file (ASCII detector output)
            container_version: Container schema version
        
        Returns:
            Path to .npy file
        """
        from pathlib import Path
        
        raw_path = Path(raw_file_path)
        
        if container_version == "0.2":
            # Container v0.2 expects .npy files for processed detector data input
            npy_path = raw_path.with_suffix('.npy')
            
            if not npy_path.exists():
                try:
                    # Load ASCII data and save as numpy binary
                    data = np.loadtxt(raw_path)
                    np.save(npy_path, data)
                    logger.info(
                        f"Converted for container v{container_version}",
                        detector=self.alias,
                        input_file=raw_path.name,
                        output_file=npy_path.name
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to convert {raw_path} to .npy: {e}"
                    )
            else:
                logger.debug(
                    f"Container format file already exists",
                    detector=self.alias,
                    file=npy_path.name
                )
            
            return str(npy_path)
        else:
            raise ValueError(
                f"Detector {self.alias} does not support container version {container_version}"
            )
    
    def get_raw_file_patterns(self):
        """Return Advacam raw file patterns for archiving.
        
        Returns:
            List of patterns for .txt (ASCII data) and .dsc (descriptor)
        """
        return ['*.txt', '*.dsc']


class PixetDetectorController(DetectorController):
    def __init__(self, alias, size=(256, 256), config=None):
        self.alias = alias
        self.size = tuple(size)  # (width, height)
        self.config = config or {}  # detector config from main.json
        self.dev_id = self.config.get(
            "id"
        )  # physical device id string to match (from config)
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
            logger.error(
                "Pixet detector is not initialized",
                detector=self.alias,
            )
            return False
        try:
            n_frames = max(int(Nframes), 1)
            exposure_seconds = float(Nseconds)
            integrated = None
            for _ in range(n_frames):
                frame = self._api.measure_single_frame(self.device_index, exposure_seconds)
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
                        # Crop to configured size if needed.
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
    
    def convert_to_container_format(self, raw_file_path: str, container_version: str = "0.2") -> str:
        """Convert Pixet/Advacam .txt format to container format.
        
        For container v0.2: converts ASCII .txt to binary .npy
        Pixet detectors use the same Advacam format (.txt ASCII + .dsc descriptor)
        
        Args:
            raw_file_path: Path to .txt file (ASCII detector output)
            container_version: Container schema version
        
        Returns:
            Path to .npy file
        """
        from pathlib import Path
        
        raw_path = Path(raw_file_path)
        
        if container_version == "0.2":
            # Container v0.2 expects .npy files for processed detector data input
            npy_path = raw_path.with_suffix('.npy')
            
            if not npy_path.exists():
                try:
                    # Load ASCII data and save as numpy binary
                    data = np.loadtxt(raw_path)
                    np.save(npy_path, data)
                    logger.info(
                        f"Converted for container v{container_version}",
                        detector=self.alias,
                        input_file=raw_path.name,
                        output_file=npy_path.name
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to convert {raw_path} to .npy: {e}"
                    )
            else:
                logger.debug(
                    f"Container format file already exists",
                    detector=self.alias,
                    file=npy_path.name
                )
            
            return str(npy_path)
        else:
            raise ValueError(
                f"Detector {self.alias} does not support container version {container_version}"
            )
    
    def get_raw_file_patterns(self):
        """Return Pixet/Advacam raw file patterns for archiving.
        
        Returns:
            List of patterns for .txt (ASCII data) and .dsc (descriptor)
        """
        return ['*.txt', '*.dsc']


class PixetLegacyDetectorController(DetectorController):
    """Legacy PIXet controller using vendor `pypixet` bindings."""

    def __init__(self, alias, size=(256, 256), config=None):
        self.alias = alias
        self.size = tuple(size)  # (width, height)
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
            # Ensure vendor DLL/module path is discoverable.
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
        self, raw_file_path: str, container_version: str = "0.2"
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


class PixetSidecarError(RuntimeError):
    """Raised when sidecar communication or command execution fails."""


class PixetSidecarDetectorController(DetectorController):
    """PIXet controller proxying hardware calls to external socket sidecar."""

    def __init__(self, alias, size=(256, 256), config=None):
        self.alias = alias
        self.size = tuple(size)
        self.config = config or {}
        sidecar_cfg = self.config.get("pixet_sidecar", {}) or {}
        self.sidecar_host = str(
            sidecar_cfg.get(
                "host",
                self.config.get(
                    "sidecar_host",
                    os.environ.get("PIXET_SIDECAR_HOST", "127.0.0.1"),
                ),
            )
        )
        self.sidecar_port = int(
            sidecar_cfg.get(
                "port",
                self.config.get(
                    "sidecar_port",
                    os.environ.get("PIXET_SIDECAR_PORT", "51001"),
                ),
            )
        )
        self.timeout_s = float(
            sidecar_cfg.get(
                "timeout_s",
                self.config.get(
                    "sidecar_timeout_s",
                    os.environ.get("PIXET_SIDECAR_TIMEOUT_S", "10.0"),
                ),
            )
        )
        self.capture_timeout_pad_s = float(
            sidecar_cfg.get(
                "capture_timeout_pad_s",
                self.config.get(
                    "sidecar_capture_timeout_pad_s",
                    os.environ.get("PIXET_SIDECAR_CAPTURE_TIMEOUT_PAD_S", "30.0"),
                ),
            )
        )
        self._stream_thread = None
        self._streaming = threading.Event()

    def _rpc(self, cmd: str, args: dict, timeout_s: Optional[float] = None):
        req_id = str(uuid.uuid4())
        payload = (
            json.dumps(
                {
                    "id": req_id,
                    "cmd": cmd,
                    "args": args,
                },
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        rpc_timeout_s = float(timeout_s) if timeout_s is not None else float(self.timeout_s)
        rpc_timeout_s = max(rpc_timeout_s, 0.1)
        try:
            with socket.create_connection(
                (self.sidecar_host, self.sidecar_port),
                timeout=rpc_timeout_s,
            ) as sock:
                sock.settimeout(rpc_timeout_s)
                sock.sendall(payload)

                response_bytes = b""
                while b"\n" not in response_bytes:
                    chunk = sock.recv(65536)
                    if not chunk:
                        raise PixetSidecarError("Sidecar closed connection without response")
                    response_bytes += chunk
        except Exception as e:
            raise PixetSidecarError(
                f"Sidecar connection failed ({self.sidecar_host}:{self.sidecar_port}): {e}"
            )

        line = response_bytes.split(b"\n", 1)[0]
        try:
            response = json.loads(line.decode("utf-8"))
        except Exception as e:
            raise PixetSidecarError(f"Invalid sidecar response JSON: {e}")

        if not response.get("ok"):
            raise PixetSidecarError(response.get("error", "Unknown sidecar error"))
        return response.get("result")

    def init_detector(self):
        logger.info(
            "Initializing Pixet detector via sidecar",
            detector=self.alias,
            sidecar_host=self.sidecar_host,
            sidecar_port=self.sidecar_port,
        )
        result = self._rpc(
            "init_detector",
            {
                "alias": self.alias,
                "size": [int(self.size[0]), int(self.size[1])],
                "config": dict(self.config),
            },
        )
        initialized = bool((result or {}).get("initialized", False))
        detected_size = (result or {}).get("size")
        if (
            isinstance(detected_size, list)
            and len(detected_size) == 2
            and all(isinstance(v, int) for v in detected_size)
        ):
            self.size = (int(detected_size[0]), int(detected_size[1]))
        return initialized

    def capture_point(self, Nframes, Nseconds, filename_base):
        nframes = max(int(Nframes), 1)
        nseconds = float(Nseconds)
        expected_capture_s = max(nseconds, 0.0) * nframes
        capture_timeout_s = max(
            self.timeout_s,
            expected_capture_s + self.capture_timeout_pad_s,
        )
        result = self._rpc(
            "capture_point",
            {
                "alias": self.alias,
                "Nframes": nframes,
                "Nseconds": nseconds,
                "filename_base": str(filename_base),
            },
            timeout_s=capture_timeout_s,
        )
        return bool((result or {}).get("captured", False))

    def deinit_detector(self):
        try:
            self._rpc("deinit_detector", {"alias": self.alias})
        except PixetSidecarError as e:
            logger.warning(
                "Sidecar deinit failed",
                detector=self.alias,
                error=str(e),
            )

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        self.stop_stream()
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval, frames),
            daemon=True,
        )
        self._stream_thread.start()
        logger.info("Sidecar streaming started", detector=self.alias, exposure=exposure)

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            logger.info("Sidecar streaming stopped", detector=self.alias)
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        while self._streaming.is_set():
            try:
                result = self._rpc(
                    "capture_frame",
                    {
                        "alias": self.alias,
                        "exposure_s": float(exposure),
                        "frames": max(int(frames), 1),
                    },
                )
                frame_payload = (result or {}).get("frame")
                if frame_payload is None:
                    frame = None
                else:
                    frame = np.asarray(frame_payload, dtype=np.float64)
                    if frame.ndim == 2:
                        frame = frame[: self.size[1], : self.size[0]]
                    else:
                        frame = None
                callback({self.alias: frame})
            except Exception as e:
                logger.warning(
                    "Sidecar frame capture error during streaming",
                    detector=self.alias,
                    error=str(e),
                )
                callback({self.alias: None})
            if interval:
                time.sleep(interval)

    def convert_to_container_format(
        self, raw_file_path: str, container_version: str = "0.2"
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
