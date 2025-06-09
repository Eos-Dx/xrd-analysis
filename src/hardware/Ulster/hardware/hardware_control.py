import time
import sys
import numpy as np
from ctypes import CDLL, c_short, c_int, c_char_p
import os


class DetectorController:
    def __init__(self, capture_enabled=True, dev=True):
        """
        Initializes the detector controller.
        Parameters:
          capture_enabled (bool): Whether to enable capture.
          dev (bool): If True, uses dummy functions.
        """
        self.capture_enabled = capture_enabled
        self.dev = dev
        self.pixet = None
        self.detector = None

    def init_detector(self):
        if self.dev:
            print("DEV mode: Dummy init_detector called.")
            self.pixet = True
            self.detector = True
            return True
        else:
            sys.path.insert(0, 'D:\\API_PIXet_Pro_1.8.3_Windows_x86_64')
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
            print(f"DEV mode: Dummy capture_point called. Saving to {filename}")
            x = np.arange(256)
            y = np.arange(256)
            X, Y = np.meshgrid(x, y)
            # First Gaussian
            x0_1, y0_1 = np.random.uniform(0, 256), np.random.uniform(0, 256)
            sigma_x1, sigma_y1 = np.random.uniform(5, 15), np.random.uniform(5, 15)
            amplitude1 = np.random.uniform(1e6 + 1, 2e6)
            gaussian1 = amplitude1 * np.exp(-(((X - x0_1) ** 2) / (2 * sigma_x1 ** 2) + ((Y - y0_1) ** 2) / (2 * sigma_y1 ** 2)))
            # Second Gaussian
            x0_2, y0_2 = np.random.uniform(0, 256), np.random.uniform(0, 256)
            sigma_x2, sigma_y2 = np.random.uniform(5, 15), np.random.uniform(5, 15)
            amplitude2 = np.random.uniform(1e5, 1e6 - 1)
            gaussian2 = amplitude2 * np.exp(-(((X - x0_2) ** 2) / (2 * sigma_x2 ** 2) + ((Y - y0_2) ** 2) / (2 * sigma_y2 ** 2)))
            combined = gaussian1 + gaussian2
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


class XYStageController:
    def __init__(self, serial_num="default_serial", x_chan=2, y_chan=1, dev=True, scaling_factor=10000):
        self.serial_num = serial_num
        self.x_chan = x_chan
        self.y_chan = y_chan
        self.dev = dev
        self.stage = None
        self.scaling_factor = scaling_factor

    def init_stage(self):
        if self.dev:
            print("DEV mode: Dummy init_stage called.")
            class DummyStage:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: print(f"DEV mode: Called {name} with args {args} and kwargs {kwargs}")
            self.stage = DummyStage()
            return True
        else:
            from pylablib.devices import Thorlabs
            devices = Thorlabs.list_kinesis_devices()
            if not devices:
                print("No Thorlabs devices found!")
                return False
            self.stage = Thorlabs.KinesisMotor(str(self.serial_num))
            self.stage.open()
            self.stage.set_supported_channels(2)
            return True

    def home_stage(self, home_timeout=10):
        if self.stage is None:
            print("Stage not initialized.")
            return None, None
        if self.dev:
            time.sleep(2)
            print("DEV mode: Dummy home_stage called.")
            return 0, 0

    def move_stage(self, x_new, y_new, move_timeout=10):
        if self.stage is None:
            print("Stage not initialized.")
            return None, None
        if self.dev:
            time.sleep(1)
            print(f"DEV mode: Dummy move_stage called. Pretending to move to ({x_new}, {y_new}).")
            return x_new, y_new

    def get_xy_position(self):
        if self.stage is None:
            return None, None
        if self.dev:
            return -1, -1

    def deinit_stage(self):
        """
        Safely close the stage connection and cleanup resources.
        """
        if self.stage is None:
            print("Stage not initialized or already deinitialized.")
            return
        if self.dev:
            print("DEV mode: Dummy deinit_stage called.")
            self.stage = None
            return
        try:
            print("Closing hardware stage connection...")
            self.stage.close()
            print("Stage connection safely closed.")
        except Exception as e:
            print(f"Error during deinit_stage: {e}")
        finally:
            self.stage = None


class XYStageLibController:
    def __init__(self, serial_num: str = '101370874', x_chan: int = 2, y_chan: int = 1,
                 scaling_factor: int = 10000, dev: bool = False, sim: bool = False,
                 poll_interval_ms: int = 250):
        self.serial = serial_num.encode()
        self.x_chan = x_chan
        self.y_chan = y_chan
        self.scaling_factor = scaling_factor
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
            return True

    def home_stage(self, timeout_s: int = 45):
        # Implementation omitted for brevity
        pass

    def move_stage(self, x_mm: float, y_mm: float, move_timeout: int = 20):
        # Implementation omitted for brevity
        pass

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
