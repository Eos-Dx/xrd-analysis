# detectors.py
import os
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod

import numpy as np


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


import threading
import time

import numpy as np


class DummyDetectorController:
    def __init__(self, alias="DUMMY", size=(256, 256)):
        self.alias = alias  # Unique name from config, e.g. "DUMMY_DETECTOR_1"
        self.size = size  # (width, height), from config
        self._stream_thread = None
        self._streaming = threading.Event()

    def init_detector(self):
        print(f"DEV mode: Dummy init_detector called for {self.alias}.")
        return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        # Simulate acquisition delay for this dummy detector
        filename = f"{filename_base}.txt"
        t = threading.Thread(target=self._dummy_acquire, args=(filename, Nseconds))
        t.start()
        t.join()
        return True

    def _dummy_acquire(self, filename, Nseconds, Nframes=1):
        time.sleep(Nseconds * Nframes)  # Simulate integration time
        width, height = self.size
        # Generate a random 2D Gaussian blob
        x, y = np.arange(width), np.arange(height)
        X, Y = np.meshgrid(x, y)
        x0, y0 = np.random.uniform(0, width), np.random.uniform(0, height)
        sigma = np.random.uniform(5, min(width, height) / 4)
        amp = np.random.uniform(1e5, 2e6)
        frame = amp * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)))
        frame += np.random.normal(scale=amp * 0.1, size=frame.shape)
        np.savetxt(filename, frame, fmt="%.6f")
        print(
            f"DEV mode: Dummy acquisition complete for {self.alias}, saved to {filename}."
        )

    def deinit_detector(self):
        print(f"DEV mode: Dummy deinit_detector called for {self.alias}.")

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        self.stop_stream()
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval),
            daemon=True,
        )
        self._stream_thread.start()
        print(f"DEV mode: Streaming started for {self.alias}")

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            print(f"Stream stopped for {self.alias}.")
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
        sys.path.insert(0, "D:\\API_PIXet_Pro_1.8.3_Windows_x86_64")
        try:
            import pypixet
        except ImportError as e:
            print("Error importing pypixet:", e)
            return False
        print(f"Initializing detector for {self.alias} (ID: {self.dev_id})...")
        pypixet.start()
        pixet = pypixet.pixet
        devices = pixet.devices()
        if not devices or devices[0].fullName() == "FileDevice 0":
            print("No devices connected")
            pixet.exitPixet()
            pypixet.exit()
            return False
        for dev in devices:
            name = dev.fullName()
            if self.dev_id in name:
                self.detector = dev
                break
        if not self.detector:
            print(f"Could not find device for {self.alias} with id {self.dev_id}")
            return False
        self.pixet = pixet
        print(f"Assigned device {self.dev_id} to alias {self.alias}")
        return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        filename = f"{filename_base}.txt"
        try:
            rc = self.detector.doSimpleIntegralAcquisition(
                Nframes, Nseconds, self.pixet.PX_FTYPE_AUTODETECT, filename
            )
        except Exception as e:
            print(f"Exception during acquisition for {self.alias}: {e}")
            return False
        if rc != 0:
            print(f"Capture error for {self.alias}: {rc}, {self.detector.lastError()}")
            return False
        print(f"Capture successful for {self.alias}")
        return True

    def deinit_detector(self):
        if self.pixet:
            try:
                print(f"Deinitializing detector hardware for {self.alias}...")
                self.pixet.exitPixet()
                import pypixet

                pypixet.exit()
                print(f"Detector {self.alias} safely deinitialized.")
            except Exception as e:
                print(f"Error during deinit_detector ({self.alias}): {e}")
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
        print(f"REAL mode: Streaming started for {self.alias}")

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            print(f"Stream stopped for {self.alias}.")
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
                    print(
                        f"Frame error for {self.alias}: {rc}, {self.detector.lastError()}"
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
                        print(f"Loading error for {self.alias}: {e}")
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
