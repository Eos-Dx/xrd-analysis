# detectors.py
import os
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

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


import threading
import time

import numpy as np


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
        t = threading.Thread(target=self._dummy_acquire, args=(filename, Nseconds, Nframes))
        t.start()
        t.join()
        return True

    def _dummy_acquire(self, filename, Nseconds, Nframes=1):
        # Simulate total acquisition duration
        time.sleep(float(Nseconds) * max(int(Nframes), 1))
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
        self.detector = None  # will be set after init
        self.pixet = None
        self._stream_thread = None
        self._streaming = threading.Event()

    def init_detector(self):
        # Note: PIXET SDK path should be added to Windows PATH at application startup
        # (before PyQt5 import) to prevent Qt DLL conflicts. See main_app.py.
        # This method only validates the path is accessible.
        
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
        try:
            logger.debug("Attempting to import pypixet module", detector=self.alias)
            import pypixet
            logger.info("pypixet module imported successfully", detector=self.alias)
        except ImportError as e:
            logger.error(
                "Error importing pypixet module",
                error=str(e),
                detector=self.alias,
                sys_path=sys.path[:5],  # Show first 5 paths
                hint="Set PIXET_SDK_PATH env var or add 'pixet_sdk_path' to detector config",
                exc_info=True,
            )
            return False

        # IMPORTANT: pixet.ini uses relative paths (e.g., hwlibs\\minipix.dll).
        # Ensure those resolve by temporarily changing CWD to the PIXET SDK folder
        original_cwd = os.getcwd()
        try:
            if pixet_sdk_path and os.path.isdir(pixet_sdk_path):
                logger.debug(
                    "Temporarily changing working directory for pypixet.start()",
                    from_cwd=original_cwd,
                    to_cwd=pixet_sdk_path,
                    detector=self.alias,
                )
                os.chdir(pixet_sdk_path)
            pypixet.start()
        finally:
            if os.getcwd() != original_cwd:
                os.chdir(original_cwd)
                logger.debug(
                    "Restored working directory after pypixet.start()",
                    cwd=original_cwd,
                    detector=self.alias,
                )

        pixet = pypixet.pixet
        devices = pixet.devices()
        if not devices or devices[0].fullName() == "FileDevice 0":
            logger.error("No Pixet devices connected")
            pixet.exitPixet()
            pypixet.exit()
            return False
        for dev in devices:
            name = dev.fullName()
            if self.dev_id in name:
                self.detector = dev
                break
        if not self.detector:
            logger.error(
                "Could not find Pixet device",
                detector=self.alias,
                device_id=self.dev_id,
            )
            return False
        self.pixet = pixet
        logger.info(
            "Assigned Pixet device to detector",
            device_id=self.dev_id,
            detector=self.alias,
        )
        return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        filename = f"{filename_base}.txt"
        try:
            rc = self.detector.doSimpleIntegralAcquisition(
                Nframes, Nseconds, self.pixet.PX_FTYPE_AUTODETECT, filename
            )
        except Exception as e:
            logger.error(
                "Exception during Pixet acquisition", detector=self.alias, error=str(e)
            )
            return False
        if rc != 0:
            logger.error(
                "Pixet capture error",
                detector=self.alias,
                return_code=rc,
                error=self.detector.lastError(),
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
        if self.pixet:
            try:
                logger.info("Deinitializing Pixet detector", detector=self.alias)
                self.pixet.exitPixet()
                import pypixet

                pypixet.exit()
                logger.info("Pixet detector safely deinitialized", detector=self.alias)
            except Exception as e:
                logger.error(
                    "Error during Pixet detector deinitialization",
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
        logger.info("Pixet streaming started", detector=self.alias, exposure=exposure)

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            logger.info("Pixet streaming stopped", detector=self.alias)
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        import tempfile

        while self._streaming.is_set():
            tmpdir = tempfile.mkdtemp()
            tmpfile = os.path.join(tmpdir, f"stream_{self.alias}.txt")
            try:
                rc = self.detector.doSimpleIntegralAcquisition(
                    frames, exposure, self.pixet.PX_FTYPE_AUTODETECT, tmpfile
                )
                if rc != 0:
                    logger.warning(
                        "Pixet frame capture error during streaming",
                        detector=self.alias,
                        return_code=rc,
                        error=self.detector.lastError(),
                    )
                    frame = None
                else:
                    try:
                        frame = np.loadtxt(tmpfile)
                        if frame is not None:
                            # Crop to the expected size if the loaded frame is larger
                            frame = frame[: self.size[0], : self.size[1]]
                            # Optionally: if you want, assert the shape now
                            assert (
                                frame.shape == self.size
                            ), f"Frame shape {frame.shape} does not match expected {self.size}"
                    except Exception as e:
                        logger.warning(
                            "Pixet frame loading error during streaming",
                            detector=self.alias,
                            error=str(e),
                        )
                        frame = None
                callback({self.alias: frame})
            finally:
                try:
                    os.remove(tmpfile)
                    os.rmdir(tmpdir)
                except Exception:
                    pass
            if interval:
                time.sleep(interval)
    
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
