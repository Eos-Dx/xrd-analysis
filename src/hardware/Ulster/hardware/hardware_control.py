from ctypes import CDLL, c_short, c_int, c_char_p
import sys
import threading
import time
import numpy as np
import os
import tempfile


class DetectorController:

    def __init__(self, capture_enabled=True, dev=True):
        self.capture_enabled = capture_enabled
        self.dev = dev
        self.pixet = None
        self.detector = None
        self._stream_thread = None
        self._streaming = threading.Event()

    def init_detector(self):
        if self.dev:
            print("DEV mode: Dummy init_detector called.")
            # In DEV mode, we simply set dummy (non-None) values.
            self.pixet = True
            self.detector = True
            return True
        else:
            # Real detector initialization.
            sys.path.insert(0, 'D:\\API_PIXet_Pro_1.8.3_Windows_x86_64')
            #sys.path.insert(0, r'D:\OneDrive\OneDrive - Matur\General - Ulster\Equipment\Xena\M30XY Stage\Code\XYscan')
            try:
                import pypixet
            except ImportError as e:
                print("Error importing pypixet:", e)
                return False
            print("Initializing detector...")
            pypixet.start()
            pixet = pypixet.pixet
            devices = pixet.devices()
            if devices[0].fullName() == 'FileDevice 0':
                print("No devices connected")
                pixet.exitPixet()
                pypixet.exit()
                self.pixet, self.detector = None, None
                return False
            else:
                self.pixet = pixet
                self.detector = devices[0]
                print("Detector initialized.")
                return True

    def capture_point(self, Nframes, Nseconds, filename):
        if self.dev:
            # --- Dummy capture (DEV mode): Create two Gaussians and combine them. ---
            print(f"DEV mode: Dummy capture_point called. Saving to {filename}")
            # Create a 256x256 grid.
            x = np.arange(256)
            y = np.arange(256)
            X, Y = np.meshgrid(x, y)
            # First Gaussian: peak > 1e6.
            x0_1 = np.random.uniform(0, 256)
            y0_1 = np.random.uniform(0, 256)
            sigma_x1 = np.random.uniform(5, 15)
            sigma_y1 = np.random.uniform(5, 15)
            amplitude1 = np.random.uniform(1e6 + 1, 2e6)
            gaussian1 = amplitude1 * np.exp(-(((X - x0_1) ** 2) / (2 * sigma_x1 ** 2) +
                                              ((Y - y0_1) ** 2) / (2 * sigma_y1 ** 2)))
            # Second Gaussian: peak < 1e6.
            x0_2 = np.random.uniform(0, 256)
            y0_2 = np.random.uniform(0, 256)
            sigma_x2 = np.random.uniform(5, 15)
            sigma_y2 = np.random.uniform(5, 15)
            amplitude2 = np.random.uniform(1e5, 1e6 - 1)
            gaussian2 = amplitude2 * np.exp(-(((X - x0_2) ** 2) / (2 * sigma_x2 ** 2) +
                                              ((Y - y0_2) ** 2) / (2 * sigma_y2 ** 2)))
            combined = gaussian1 + gaussian2
            import time
            time.sleep(Nseconds)
            np.savetxt(filename, combined, fmt='%.6f')
            return True
        else:
            print(f"Capturing at {filename} ...")
            try:
                rc = self.detector.doSimpleIntegralAcquisition(Nframes, Nseconds, self.pixet.PX_FTYPE_AUTODETECT, filename)
                if rc == 0:
                    print("Capture successful.")
                    return True
                else:
                    print("Capture error:", rc)
                    return False
            except Exception as e:
                print(f'During capture: {e}')
                return False

    def deinit_detector(self):
        """
        Safely deinitialize the detector hardware and cleanup resources.
        """
        if self.dev:
            print("DEV mode: Dummy deinit_detector called.")
            self.pixet, self.detector = None, None
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
                self.pixet, self.detector = None, None

    # ──────── Real-Time (Looped) Streaming ──────────────────────────

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        """
        Begin continuous frame acquisition.
        callback(frame: np.ndarray) -> None
        exposure: seconds per frame
        interval: extra idle time between frames
        """
        if not self.capture_enabled:
            raise RuntimeError("Capture not enabled")

        # Stop existing stream if any
        self.stop_stream()

        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval, frames),
            daemon=True
        )
        self._stream_thread.start()
        mode = "DEV" if self.dev else "REAL"
        print(f"{mode} mode: Stream started (exp={exposure}s, interval={interval}s)")

    def stop_stream(self):
        """Stop any running stream."""
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            print("Stream stopped.")
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        """Internal: loop that grabs frames back‐to‐back."""
        tmpdir = None
        tmpfile = None

        if not self.dev:
            # prepare a temp file for real captures
            tmpdir = tempfile.mkdtemp()
            tmpfile = os.path.join(tmpdir, "stream_frame.txt")

        try:
            while self._streaming.is_set():
                if self.dev:
                    # Dummy Gaussian blob
                    x = np.arange(256); y = np.arange(256)
                    X, Y = np.meshgrid(x, y)
                    x0, y0 = np.random.uniform(0,256), np.random.uniform(0,256)
                    sigma = np.random.uniform(5,20)
                    amp  = np.random.uniform(1e5,2e6)
                    frame = amp * np.exp(-(((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)))
                    time.sleep(exposure)
                    callback(frame)

                else:
                    # Real acquisition: 1 frame
                    rc = self.detector.doSimpleIntegralAcquisition(
                        frames,
                        exposure,
                        self.pixet.PX_FTYPE_AUTODETECT,
                        tmpfile
                    )
                    if rc != 0:
                        print("Frame error:", rc, self.detector.lastError())
                    else:
                        frame = np.loadtxt(tmpfile)
                        callback(frame)
                    if interval:
                        time.sleep(interval)
        finally:
            # Cleanup
            if tmpdir:
                try:
                    os.remove(tmpfile)
                    os.rmdir(tmpdir)
                except:
                    pass


class XYStageLibController:
    """
    Controller for Thorlabs M30XY stage using Kinesis DLL, with optional DEV dummy mode.
    Supports position scaling between mm and device units.
    """
    def __init__(self, serial_num: str = '101370874', x_chan: int = 2, y_chan: int = 1,
                 scaling_factor: int = 10000, dev: bool = False, sim: bool = False,
                 poll_interval_ms: int = 250):
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
            print('DEV mode: initializing dummy stage')
            class DummyStage:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: print(f"DEV mode: Called {name} with args {args} {kwargs}")
            self.stage = DummyStage()
            self._x, self._y = 0, 0
            return True
        else:
            try:
                if sys.version_info < (3, 8):
                    os.chdir(r'C:\Program Files\Thorlabs\Kinesis')
                else:
                    os.add_dll_directory(r'C:\Program Files\Thorlabs\Kinesis')
                self.lib = CDLL('Thorlabs.MotionControl.Benchtop.DCServo.dll')

                if self.sim:
                    self.lib.TLI_InitializeSimulations()

                if self.lib.TLI_BuildDeviceList() != 0:
                    return False

                self.lib.BDC_Open(c_char_p(self.serial))
                self.lib.BDC_StartPolling(c_char_p(self.serial), c_short(self.x_chan), c_int(self.poll_interval_ms))
                self.lib.BDC_StartPolling(c_char_p(self.serial), c_short(self.y_chan), c_int(self.poll_interval_ms))
                self.lib.BDC_EnableChannel(c_char_p(self.serial), c_short(self.x_chan))
                self.lib.BDC_EnableChannel(c_char_p(self.serial), c_short(self.y_chan))
                time.sleep(0.5)
                print('Stage initialized and polling started.')
                return True
            except Exception as e:
                print(f'Error during init_stage: {e}')
                return False

    def home_stage(self, timeout_s: int = 45):
        if self.dev:
            print('DEV mode: homing dummy stage')
            time.sleep(1)
            return 0.0, 0.0
        if not self.lib:
            raise RuntimeError('Stage not initialized')

        self.lib.BDC_Home(c_char_p(self.serial), c_short(self.x_chan))
        self.lib.BDC_Home(c_char_p(self.serial), c_short(self.y_chan))
        start = time.time()
        while time.time() - start < timeout_s:
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.y_chan))
            time.sleep(0.5)
            x_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.x_chan))
            y_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.y_chan))
            if abs(x_dev) + abs(y_dev) <= 3:
                x_mm = x_dev / self.scaling_factor
                y_mm = y_dev / self.scaling_factor
                print(f'Homed successfully at X={x_mm:.3f} mm, Y={y_mm:.3f} mm')
                return x_mm, y_mm
        raise TimeoutError('Homing timed out')

    def move_stage(self, x_mm: float, y_mm: float, move_timeout: int = 20):
        if self.dev:
            print(f'DEV mode: moving dummy stage to X={x_mm} mm, Y={y_mm} mm')
            time.sleep(0.5)
            self._x, self._y = x_mm, y_mm
            return x_mm, y_mm
        if not self.lib:
            raise RuntimeError('Stage not initialized')

        x_dev = int(x_mm * self.scaling_factor)
        y_dev = int(y_mm * self.scaling_factor)
        self.lib.BDC_SetMoveAbsolutePosition(c_char_p(self.serial), c_short(self.x_chan), c_int(x_dev))
        self.lib.BDC_SetMoveAbsolutePosition(c_char_p(self.serial), c_short(self.y_chan), c_int(y_dev))
        time.sleep(0.25)
        self.lib.BDC_MoveAbsolute(c_char_p(self.serial), c_short(self.x_chan))
        self.lib.BDC_MoveAbsolute(c_char_p(self.serial), c_short(self.y_chan))

        start = time.time()
        while time.time() - start < move_timeout:
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.y_chan))
            time.sleep(0.5)
            curr_x_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.x_chan))
            curr_y_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.y_chan))
            if abs(curr_x_dev - x_dev) + abs(curr_y_dev - y_dev) <= 4:
                curr_x_mm = curr_x_dev / self.scaling_factor
                curr_y_mm = curr_y_dev / self.scaling_factor
                print(f'Moved to X={curr_x_mm:.3f} mm, Y={curr_y_mm:.3f} mm')
                return curr_x_mm, curr_y_mm
        raise TimeoutError('Move timed out')

    def close_stage(self):
        """
        Safely close the hardware stage connection via DLL and cleanup resources.
        """
        if self.dev:
            print('DEV mode: closing dummy stage')
            self.lib = None
            return
        if self.lib:
            try:
                print('Closing hardware stage connection via DLL...')
                self.lib.BDC_Close(c_char_p(self.serial))
                if self.sim:
                    self.lib.TLI_UninitializeSimulations()
                print('Stage closed.')
            except Exception as e:
                print(f'Error during close_stage: {e}')
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
             curr_x_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.x_chan))
             curr_y_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.y_chan))
             curr_x_mm = curr_x_dev / self.scaling_factor
             curr_y_mm = curr_y_dev / self.scaling_factor
             return curr_x_mm, curr_y_mm

# class XYStageController:
#     def __init__(self, serial_num="default_serial", x_chan=2, y_chan=1, dev=True, scaling_factor=10000):
#         """
#         Initializes the XY stage controller.
#         Parameters:
#           serial_num (str): Serial number of the stage.
#           x_chan (int): X channel number.
#           y_chan (int): Y channel number.
#           dev (bool): If True, uses dummy functions.
#         """
#         self.serial_num = serial_num
#         self.x_chan = x_chan
#         self.y_chan = y_chan
#         self.dev = dev
#         self.stage = None  # In DEV mode, we use a dummy stage object.
#         self.scaling_factor = scaling_factor
#
#     def init_stage(self):
#         if self.dev:
#             print("DEV mode: Dummy init_stage called.")
#             # Create a dummy stage with dummy function calls.
#             class DummyStage:
#                 def __getattr__(self, name):
#                     return lambda *args, **kwargs: print(f"DEV mode: Called {name} with args {args} and kwargs {kwargs}")
#             self.stage = DummyStage()
#             return True
#         else:
#             from pylablib.devices import Thorlabs
#             devices = Thorlabs.list_kinesis_devices()
#             if not devices:
#                 print("No Thorlabs devices found!")
#                 self.stage = None
#                 return False
#             else:
#                 print("Detected Thorlabs devices:")
#                 for dev in devices:
#                     print(dev)
#
#                 self.stage = Thorlabs.KinesisMotor(str(self.serial_num))
#                 self.stage.open()  # Open the device connection.
#                 self.stage.set_supported_channels(2)
#                 print("Enabled channels:", self.stage.get_all_channels())
#
#                 return True
#
#     def home_stage(self, home_timeout=10):
#         if self.stage is None:
#             print("Stage not initialized.")
#             return None, None
#         if self.dev:
#             time.sleep(2)
#             print("DEV mode: Dummy home_stage called.")
#             return 0, 0
#         else:
#             print("Homing stage on X and Y axes...")
#             self.stage.home(channel=1)
#             self.stage.home(channel=2)
#             self.stage.wait_for_home(channel=1, timeout=home_timeout)
#             self.stage.wait_for_home(channel=2, timeout=home_timeout)
#             x_final = self.stage.get_position(channel=self.x_chan, scale=True) / self.scaling_factor
#             y_final = self.stage.get_position(channel=self.y_chan, scale=True) / self.scaling_factor
#             print(f"Final homed positions: X = {x_final} mm, Y = {y_final} mm")
#             return x_final, y_final
#
#     def move_stage(self, x_new, y_new, move_timeout=10):
#         import time
#         if self.stage is None:
#             print("Stage not initialized.")
#             return None, None
#         if self.dev:
#             time.sleep(1)
#             print(f"DEV mode: Dummy move_stage called. Pretending to move to ({x_new}, {y_new}).")
#             return x_new, y_new
#         else:
#
#             # Move X-axis
#             print(f"Moving X-axis to {x_new} mm: channel {self.x_chan}")
#             self.stage.move_to(x_new * self.scaling_factor, channel=self.x_chan, scale=True)
#             self.stage.wait_move(channel=self.x_chan, timeout=move_timeout)
#             x = self.stage.get_position(channel=self.x_chan, scale=True) / self.scaling_factor
#             print(f"X-axis moved to: {x}")
#
#             # Move Y-axis
#             print(f"Moving Y-axis to {y_new} mm: channel {self.y_chan}")
#             self.stage.move_to(y_new * self.scaling_factor, channel=self.y_chan, scale=True)
#             self.stage.wait_move(channel=self.y_chan, timeout=move_timeout)
#             y = self.stage.get_position(channel=self.y_chan, scale=True) / self.scaling_factor
#             print(f"Y-axis moved to: {y}")
#
#             print(f"Final positions: X = {x} mm, Y = {y} mm")
#             return x, y
#
#     def get_xy_position(self):
#         if self.stage is None:
#             return None, None
#         if self.dev:
#             return -1, -1
#         else:
#             x_final = self.stage.get_position(channel=self.x_chan, scale=True) / self.scaling_factor
#             y_final = self.stage.get_position(channel=self.y_chan, scale=True) / self.scaling_factor
#             return x_final, y_final


