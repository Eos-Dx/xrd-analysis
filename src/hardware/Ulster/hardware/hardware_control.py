import os
import sys
import tempfile
import threading
import time
from ctypes import CDLL, c_char_p, c_int, c_short

import numpy as np


class DetectorController:
    SAXS_ID = "MiniPIX G08-W0299"
    WAXS_ID = "MiniPIX G05-W0339"

    def __init__(self, capture_enabled=True, dev=True):
        self.capture_enabled = capture_enabled
        self.dev = dev
        self.pixet = None
        self.detectors = {}  # {'WAXS': device, 'SAXS': device}
        self._stream_thread = None
        self._streaming = threading.Event()

    def init_detector(self):
        if self.dev:
            print("DEV mode: Dummy init_detector called.")
            self.pixet = True
            self.detectors = {"WAXS": True, "SAXS": True}
            return True
        else:
            sys.path.insert(0, "D:\\API_PIXet_Pro_1.8.3_Windows_x86_64")
            try:
                import pypixet
            except ImportError as e:
                print("Error importing pypixet:", e)
                return False
            print("Initializing detectors...")
            pypixet.start()
            pixet = pypixet.pixet
            devices = pixet.devices()
            if not devices or devices[0].fullName() == "FileDevice 0":
                print("No devices connected")
                pixet.exitPixet()
                pypixet.exit()
                self.pixet, self.detectors = None, {}
                return False

            # Assign detectors by ID
            for dev in devices:
                name = dev.fullName()
                if self.WAXS_ID in name:
                    self.detectors["WAXS"] = dev
                elif self.SAXS_ID in name:
                    self.detectors["SAXS"] = dev
            if len(self.detectors) < 2:
                print(
                    f"Could not find both detectors: Found {self.detectors.keys()}"
                )
                return False
            self.pixet = pixet
            print(f"Detector assignment: {self.detectors.keys()}")
            return True

    def capture_point(self, Nframes, Nseconds, filename_base):
        """Parallel acquisition for both detectors."""
        if self.dev:
            print("DEV mode: Dummy parallel capture_point for both detectors.")
            threads = []
            for name in self.detectors.keys():
                filename = f"{filename_base}_{name}.txt"
                t = threading.Thread(
                    target=self._dummy_acquire, args=(filename, Nseconds)
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            return True
        else:
            threads = []
            results = {}

            def acquire(det, fname, name):
                try:
                    rc = det.doSimpleIntegralAcquisition(
                        Nframes,
                        Nseconds,
                        self.pixet.PX_FTYPE_AUTODETECT,
                        fname,
                    )
                    results[name] = rc
                except Exception as e:
                    results[name] = str(e)

            for name, det in self.detectors.items():
                filename = f"{filename_base}_{name}.txt"
                t = threading.Thread(
                    target=acquire, args=(det, filename, name)
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            # Check for errors
            for name, rc in results.items():
                if rc != 0:
                    print(f"Capture error for {name}:", rc)
                    return False
            print("Capture successful for both detectors (parallel).")
            return True

    def _dummy_acquire(self, filename, Nseconds):
        x = np.arange(256)
        y = np.arange(256)
        X, Y = np.meshgrid(x, y)
        gaussian = np.random.normal(loc=1e6, scale=1e5, size=(256, 256))
        time.sleep(Nseconds)
        np.savetxt(filename, gaussian, fmt="%.6f")

    def deinit_detector(self):
        if self.dev:
            print("DEV mode: Dummy deinit_detector called.")
            self.pixet, self.detectors = None, {}
            return
        if self.pixet:
            try:
                print("Deinitializing detector hardware...")
                self.pixet.exitPixet()
                import pypixet

                pypixet.exit()
                print("Detector safely deinitialized.")
            except Exception as e:
                print(f"Error during deinit_detector: {e}")
            finally:
                self.pixet, self.detectors = None, {}

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        """
        Begin continuous frame acquisition from both detectors in parallel.
        callback(frames: dict) -> None, e.g. {"WAXS": np.ndarray, "SAXS": np.ndarray}
        """
        if not self.capture_enabled:
            raise RuntimeError("Capture not enabled")

        # Stop existing stream if any
        self.stop_stream()

        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval, frames),
            daemon=True,
        )
        self._stream_thread.start()
        mode = "DEV" if self.dev else "REAL"
        print(
            f"{mode} mode: Parallel stream started (exp={exposure}s, interval={interval}s)"
        )

    def stop_stream(self):
        """Stop any running stream."""
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            print("Stream stopped.")
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        """Internal: parallel loop that grabs frames from both detectors back-to-back."""
        if self.dev:
            while self._streaming.is_set():
                results = {}
                for name in self.detectors.keys():
                    x = np.arange(256)
                    y = np.arange(256)
                    X, Y = np.meshgrid(x, y)
                    x0, y0 = np.random.uniform(0, 256), np.random.uniform(
                        0, 256
                    )
                    sigma = np.random.uniform(5, 20)
                    amp = np.random.uniform(1e5, 2e6)
                    frame = amp * np.exp(
                        -(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
                    )
                    results[name] = frame
                time.sleep(exposure)
                callback(results)
                if interval:
                    time.sleep(interval)
        else:
            while self._streaming.is_set():
                threads = []
                results = {}
                tmpdirs = {}
                tmpfiles = {}

                def acquire_frame(det, name):
                    tmpdir = tempfile.mkdtemp()
                    tmpfile = os.path.join(tmpdir, f"stream_{name}.txt")
                    tmpdirs[name] = tmpdir
                    tmpfiles[name] = tmpfile
                    rc = det.doSimpleIntegralAcquisition(
                        frames,
                        exposure,
                        self.pixet.PX_FTYPE_AUTODETECT,
                        tmpfile,
                    )
                    if rc != 0:
                        print(f"Frame error for {name}:", rc, det.lastError())
                        results[name] = None
                    else:
                        try:
                            frame = np.loadtxt(tmpfile)
                            results[name] = frame
                        except Exception as e:
                            print(f"Loading error for {name}: {e}")
                            results[name] = None

                # Start one thread per detector
                for name, det in self.detectors.items():
                    t = threading.Thread(
                        target=acquire_frame, args=(det, name)
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                callback(results)

                # Cleanup
                for name in tmpdirs:
                    try:
                        os.remove(tmpfiles[name])
                        os.rmdir(tmpdirs[name])
                    except Exception:
                        pass

                if interval:
                    time.sleep(interval)


class XYStageLibController:
    """
    Controller for Thorlabs M30XY stage using Kinesis DLL, with optional DEV dummy mode.
    Supports position scaling between mm and device units.
    """

    def __init__(
        self,
        serial_num: str = "101370874",
        x_chan: int = 2,
        y_chan: int = 1,
        scaling_factor: int = 10000,
        dev: bool = False,
        sim: bool = False,
        poll_interval_ms: int = 250,
    ):
        self.serial = serial_num.encode()
        self.x_chan = x_chan
        self.y_chan = y_chan
        self.scaling_factor = scaling_factor  # device units per 1 mm
        self.dev = dev
        self.sim = sim
        self.poll_interval_ms = poll_interval_ms
        self.lib = None
        self.stage = None

    def init_stage(self):
        if self.dev:
            print("DEV mode: initializing dummy stage")

            class DummyStage:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: print(
                        f"DEV mode: Called {name} with args {args} {kwargs}"
                    )

            self.stage = DummyStage()
            self._x, self._y = 0, 0
            return True
        else:
            try:
                if sys.version_info < (3, 8):
                    os.chdir(r"C:\Program Files\Thorlabs\Kinesis")
                else:
                    os.add_dll_directory(r"C:\Program Files\Thorlabs\Kinesis")
                self.lib = CDLL("Thorlabs.MotionControl.Benchtop.DCServo.dll")

                if self.sim:
                    self.lib.TLI_InitializeSimulations()

                if self.lib.TLI_BuildDeviceList() != 0:
                    return False

                self.lib.BDC_Open(c_char_p(self.serial))
                self.lib.BDC_StartPolling(
                    c_char_p(self.serial),
                    c_short(self.x_chan),
                    c_int(self.poll_interval_ms),
                )
                self.lib.BDC_StartPolling(
                    c_char_p(self.serial),
                    c_short(self.y_chan),
                    c_int(self.poll_interval_ms),
                )
                self.lib.BDC_EnableChannel(
                    c_char_p(self.serial), c_short(self.x_chan)
                )
                self.lib.BDC_EnableChannel(
                    c_char_p(self.serial), c_short(self.y_chan)
                )
                time.sleep(0.5)
                print("Stage initialized and polling started.")
                return True
            except Exception as e:
                print(f"Error during init_stage: {e}")
                return False

    def home_stage(self, timeout_s: int = 45):
        if self.dev:
            print("DEV mode: homing dummy stage")
            time.sleep(1)
            return 0.0, 0.0
        if not self.lib:
            raise RuntimeError("Stage not initialized")

        self.lib.BDC_Home(c_char_p(self.serial), c_short(self.x_chan))
        self.lib.BDC_Home(c_char_p(self.serial), c_short(self.y_chan))
        start = time.time()
        while time.time() - start < timeout_s:
            self.lib.BDC_RequestPosition(
                c_char_p(self.serial), c_short(self.x_chan)
            )
            self.lib.BDC_RequestPosition(
                c_char_p(self.serial), c_short(self.y_chan)
            )
            time.sleep(0.5)
            x_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.x_chan)
            )
            y_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.y_chan)
            )
            if abs(x_dev) + abs(y_dev) <= 3:
                x_mm = x_dev / self.scaling_factor
                y_mm = y_dev / self.scaling_factor
                print(
                    f"Homed successfully at X={x_mm:.3f} mm, Y={y_mm:.3f} mm"
                )
                return x_mm, y_mm
        raise TimeoutError("Homing timed out")

    def move_stage(self, x_mm: float, y_mm: float, move_timeout: int = 20):
        if self.dev:
            print(f"DEV mode: moving dummy stage to X={x_mm} mm, Y={y_mm} mm")
            time.sleep(0.5)
            self._x, self._y = x_mm, y_mm
            return x_mm, y_mm
        if not self.lib:
            raise RuntimeError("Stage not initialized")

        x_dev = int(x_mm * self.scaling_factor)
        y_dev = int(y_mm * self.scaling_factor)
        self.lib.BDC_SetMoveAbsolutePosition(
            c_char_p(self.serial), c_short(self.x_chan), c_int(x_dev)
        )
        self.lib.BDC_SetMoveAbsolutePosition(
            c_char_p(self.serial), c_short(self.y_chan), c_int(y_dev)
        )
        time.sleep(0.25)
        self.lib.BDC_MoveAbsolute(c_char_p(self.serial), c_short(self.x_chan))
        self.lib.BDC_MoveAbsolute(c_char_p(self.serial), c_short(self.y_chan))

        start = time.time()
        while time.time() - start < move_timeout:
            self.lib.BDC_RequestPosition(
                c_char_p(self.serial), c_short(self.x_chan)
            )
            self.lib.BDC_RequestPosition(
                c_char_p(self.serial), c_short(self.y_chan)
            )
            time.sleep(0.5)
            curr_x_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.x_chan)
            )
            curr_y_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.y_chan)
            )
            if abs(curr_x_dev - x_dev) + abs(curr_y_dev - y_dev) <= 4:
                curr_x_mm = curr_x_dev / self.scaling_factor
                curr_y_mm = curr_y_dev / self.scaling_factor
                print(f"Moved to X={curr_x_mm:.3f} mm, Y={curr_y_mm:.3f} mm")
                return curr_x_mm, curr_y_mm
        raise TimeoutError("Move timed out")

    def close_stage(self):
        """
        Safely close the hardware stage connection via DLL and cleanup resources.
        """
        if self.dev:
            print("DEV mode: closing dummy stage")
            self.lib = None
            return
        if self.lib:
            try:
                print("Closing hardware stage connection via DLL...")
                self.lib.BDC_Close(c_char_p(self.serial))
                if self.sim:
                    self.lib.TLI_UninitializeSimulations()
                print("Stage closed.")
            except Exception as e:
                print(f"Error during close_stage: {e}")
            finally:
                self.lib = None

    def deinit(self):
        """
        Alias for close_stage to deinitialize hardware safely.
        """
        self.close_stage()

    def get_xy_position(self):
        if self.dev:
            return self._x, self._y
        else:
            curr_x_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.x_chan)
            )
            curr_y_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.y_chan)
            )
            curr_x_mm = curr_x_dev / self.scaling_factor
            curr_y_mm = curr_y_dev / self.scaling_factor
            return curr_x_mm, curr_y_mm


class HardwareController:
    def __init__(self, config):
        self.config = config
        self.stage_controller = None
        self.detector_controller = None
        self.hardware_initialized = False

    def initialize(self):
        dev_mode = self.config.get("DEV", True)
        serial_str = self.config.get("serial_number_XY", "default_serial")
        self.stage_controller = XYStageLibController(
            serial_num=serial_str, x_chan=2, y_chan=1, dev=dev_mode
        )
        res_xystage = self.stage_controller.init_stage()
        self.detector_controller = DetectorController(
            capture_enabled=True, dev=dev_mode
        )
        res_det = self.detector_controller.init_detector()
        self.hardware_initialized = res_xystage and res_det
        return res_xystage, res_det

    def deinitialize(self):
        if self.stage_controller:
            try:
                self.stage_controller.deinit()
            except Exception:
                pass
        if self.detector_controller:
            try:
                self.detector_controller.deinit_detector()
            except Exception:
                pass
        self.hardware_initialized = False

    def get_xy_position(self):
        if self.stage_controller:
            return self.stage_controller.get_xy_position()
        return 0, 0

    def move_stage(self, x, y, timeout=10):
        if self.stage_controller:
            return self.stage_controller.move_stage(x, y, move_timeout=timeout)
        return x, y

    def home_stage(self, timeout=10):
        if self.stage_controller:
            return self.stage_controller.home_stage(home_timeout=timeout)
        return 0, 0
