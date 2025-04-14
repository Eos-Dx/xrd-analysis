import time
import sys
import numpy as np

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
            # In DEV mode, we simply set dummy (non-None) values.
            self.pixet = True
            self.detector = True
        else:
            # Real detector initialization.
            sys.path.insert(0, 'C:\\Program Files\\PIXet Pro')
            import pypixet
            print("Initializing detector...")
            pypixet.start()
            pixet = pypixet.pixet
            devices = pixet.devices()
            if devices[0].fullName() == 'FileDevice 0':
                print("No devices connected")
                pixet.exitPixet()
                pypixet.exit()
                self.pixet, self.detector = None, None
            else:
                self.pixet = pixet
                self.detector = devices[0]
                print("Detector initialized.")

    def capture_point(self, Nframes, Nseconds, filename):
        if self.dev:
            # --- Dummy capture (DEV mode): Create two Gaussians and combine them. ---
            print(f"DEV mode: Dummy capture_point called. Saving to {filename}")
            # Create a 100x100 grid.
            x = np.arange(100)
            y = np.arange(100)
            X, Y = np.meshgrid(x, y)
            # First Gaussian: peak > 1e6.
            x0_1 = np.random.uniform(0, 100)
            y0_1 = np.random.uniform(0, 100)
            sigma_x1 = np.random.uniform(5, 15)
            sigma_y1 = np.random.uniform(5, 15)
            amplitude1 = np.random.uniform(1e6 + 1, 2e6)
            gaussian1 = amplitude1 * np.exp(-(((X - x0_1) ** 2) / (2 * sigma_x1 ** 2) +
                                              ((Y - y0_1) ** 2) / (2 * sigma_y1 ** 2)))
            # Second Gaussian: peak < 1e6.
            x0_2 = np.random.uniform(0, 100)
            y0_2 = np.random.uniform(0, 100)
            sigma_x2 = np.random.uniform(5, 15)
            sigma_y2 = np.random.uniform(5, 15)
            amplitude2 = np.random.uniform(1e5, 1e6 - 1)
            gaussian2 = amplitude2 * np.exp(-(((X - x0_2) ** 2) / (2 * sigma_x2 ** 2) +
                                              ((Y - y0_2) ** 2) / (2 * sigma_y2 ** 2)))
            combined = gaussian1 + gaussian2
            np.savetxt(filename, combined, fmt='%.6f')
        else:
            print(f"Capturing at {filename} ...")
            try:
                rc = self.detector.doSimpleIntegralAcquisition(Nframes, Nseconds, self.pixet.PX_FTYPE_AUTODETECT, filename)
                if rc == 0:
                    print("Capture successful.")
                else:
                    print("Capture error:", rc)
            except Exception as e:
                print(f'During capture: {e}')


class XYStageController:
    def __init__(self, serial_num="default_serial", x_chan=2, y_chan=1, dev=True, scaling_factor=10000):
        """
        Initializes the XY stage controller.
        Parameters:
          serial_num (str): Serial number of the stage.
          x_chan (int): X channel number.
          y_chan (int): Y channel number.
          dev (bool): If True, uses dummy functions.
        """
        self.serial_num = serial_num
        self.x_chan = x_chan
        self.y_chan = y_chan
        self.dev = dev
        self.stage = None  # In DEV mode, we use a dummy stage object.
        self.scaling_factor = scaling_factor

    def init_stage(self):
        if self.dev:
            print("DEV mode: Dummy init_stage called.")
            # Create a dummy stage with dummy function calls.
            class DummyStage:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: print(f"DEV mode: Called {name} with args {args} and kwargs {kwargs}")
            self.stage = DummyStage()
            return self.stage
        else:
            from pylablib.devices import Thorlabs
            devices = Thorlabs.list_kinesis_devices()
            if not devices:
                print("No Thorlabs devices found!")
                self.stage = None
                return None
            else:
                print("Detected Thorlabs devices:")
                for dev in devices:
                    print(dev)
            self.stage = Thorlabs.KinesisMotor(str(self.serial_num))
            self.stage.open()  # Open the device connection.
            time.sleep(1)  # Allow time for initialization.
            return self.stage

    def home_stage(self, home_timeout=10):
        if self.stage is None:
            print("Stage not initialized.")
            return None, None
        if self.dev:
            time.sleep(2)
            print("DEV mode: Dummy home_stage called.")
            return 0, 0
        else:
            print("Homing stage on X and Y axes...")
            self.stage.home(channel=1)
            self.stage.home(channel=2)
            self.stage.wait_for_home(channel=1, timeout=home_timeout)
            self.stage.wait_for_home(channel=2, timeout=home_timeout)
            x_final = self.stage.get_position(channel=self.x_chan, scale=True) / self.scaling_factor
            y_final = self.stage.get_position(channel=self.y_chan, scale=True) / self.scaling_factor
            print(f"Final homed positions: X = {x_final} mm, Y = {y_final} mm")
            return x_final, y_final

    def move_stage(self, x_new, y_new, move_timeout=10):
        if self.stage is None:
            print("Stage not initialized.")
            return None, None
        if self.dev:
            time.sleep(1)
            print(f"DEV mode: Dummy move_stage called. Pretending to move to ({x_new}, {y_new}).")
            return x_new, y_new
        else:
            print(f"Moving stage: Target X = {x_new} mm, Target Y = {y_new} mm")
            self.stage.move_to(y_new * self.scaling_factor, channel=self.y_chan, scale=True)
            self.stage.move_to(x_new * self.scaling_factor, channel=self.x_chan, scale=True)
            self.stage.wait_move(channel=self.x_chan, timeout=move_timeout)
            self.stage.wait_move(channel=self.y_chan, timeout=move_timeout)
            x_final = self.stage.get_position(channel=self.x_chan, scale=True) / self.scaling_factor
            y_final = self.stage.get_position(channel=self.y_chan, scale=True) / self.scaling_factor
            print(f"Final positions: X = {x_final} mm, Y = {y_final} mm")
            return x_final, y_final

    def get_xy_position(self):
        if self.stage is None:
            return None, None
        if self.dev:
            return -1, -1
        else:
            x_final = self.stage.get_position(channel=self.x_chan, scale=True) / self.scaling_factor
            y_final = self.stage.get_position(channel=self.y_chan, scale=True) / self.scaling_factor
            return x_final, y_final
