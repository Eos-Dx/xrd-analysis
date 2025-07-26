# xystages.py
from ctypes import CDLL, c_char_p, c_int, c_short
from abc import ABC, abstractmethod
import time
import sys
import os


class BaseStageController(ABC):
    """Abstract base for all translation stages."""

    @abstractmethod
    def init_stage(self): pass

    @abstractmethod
    def home_stage(self, timeout_s=45): pass

    @abstractmethod
    def move_stage(self, x_mm, y_mm, move_timeout=20): pass

    @abstractmethod
    def get_xy_position(self): pass

    @abstractmethod
    def deinit(self): pass


class DummyStageController(BaseStageController):
    """DEV mode dummy stage controller."""

    def __init__(self, config):
        self._x = 0.0
        self._y = 0.0
        self.alias = config.get("alias", "DUMMY")
        self.id = config.get("id", "DUMMY-000")

    def init_stage(self):
        print(f"Dummy stage '{self.alias}' initialized.")
        return True

    def home_stage(self, timeout_s=45):
        print(f"Dummy stage '{self.alias}' homing.")
        time.sleep(1)
        self._x, self._y = 0.0, 0.0
        return self._x, self._y

    def move_stage(self, x_mm, y_mm, move_timeout=20):
        print(f"Dummy stage moving to X={x_mm}, Y={y_mm}")
        time.sleep(0.25)
        self._x, self._y = x_mm, y_mm
        return self._x, self._y

    def get_xy_position(self):
        return self._x, self._y

    def deinit(self):
        print(f"Dummy stage '{self.alias}' deinitialized.")


class XYStageLibController(BaseStageController):
    """Real stage controller using Thorlabs Kinesis DLL."""

    def __init__(
        self,
        config,
        x_chan=2,
        y_chan=1,
        scaling_factor=10000,
        sim=False,
        poll_interval_ms=250,
    ):
        self.serial = config["id"].encode()
        self.alias = config.get("alias", "XY_STAGE")
        self.x_chan = x_chan
        self.y_chan = y_chan
        self.scaling_factor = scaling_factor
        self.sim = sim
        self.poll_interval_ms = poll_interval_ms
        self.lib = None

    def init_stage(self):
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
            self.lib.BDC_StartPolling(c_char_p(self.serial), c_short(self.x_chan), c_int(self.poll_interval_ms))
            self.lib.BDC_StartPolling(c_char_p(self.serial), c_short(self.y_chan), c_int(self.poll_interval_ms))
            self.lib.BDC_EnableChannel(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_EnableChannel(c_char_p(self.serial), c_short(self.y_chan))
            time.sleep(0.5)
            print(f"Stage '{self.alias}' initialized.")
            return True
        except Exception as e:
            print(f"Error during stage init: {e}")
            return False

    def home_stage(self, timeout_s=45):
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
                print(f"Stage homed to X={x_mm:.3f}, Y={y_mm:.3f}")
                return x_mm, y_mm
        raise TimeoutError("Stage homing timed out")

    def move_stage(self, x_mm, y_mm, move_timeout=20):
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
                x_mm = curr_x_dev / self.scaling_factor
                y_mm = curr_y_dev / self.scaling_factor
                print(f"Stage moved to X={x_mm:.3f}, Y={y_mm:.3f}")
                return x_mm, y_mm
        raise TimeoutError("Stage move timed out")

    def get_xy_position(self):
        x_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.x_chan))
        y_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.y_chan))
        return x_dev / self.scaling_factor, y_dev / self.scaling_factor

    def deinit(self):
        try:
            self.lib.BDC_Close(c_char_p(self.serial))
            if self.sim:
                self.lib.TLI_UninitializeSimulations()
            print(f"Stage '{self.alias}' deinitialized.")
        except Exception as e:
            print(f"Stage deinit error: {e}")
        finally:
            self.lib = None
