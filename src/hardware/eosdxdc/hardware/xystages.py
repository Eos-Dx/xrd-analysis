# xystages.py
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from ctypes import CDLL, c_char_p, c_int, c_short


class StageAxisLimitError(Exception):
    """Exception raised when stage move exceeds axis limits."""

    def __init__(self, axis: str, value: float, min_limit: float, max_limit: float):
        super().__init__(
            f"Stage {axis} position {value:.3f} mm exceeds limits [{min_limit:.1f}, {max_limit:.1f}] mm"
        )
        self.axis = axis
        self.value = value
        self.min_limit = min_limit
        self.max_limit = max_limit


class BaseStageController(ABC):
    """Abstract base for all translation stages."""

    DEFAULT_LIMIT = (-14.0, 14.0)
    DEFAULT_HOME = (9.25, 6.0)
    DEFAULT_LOAD = (-13.9, -6.0)

    def _parse_limits(self, config):
        """Parse per-axis limits from the config's settings.limits_mm.
        Accepts either arrays [min,max] or objects {min:.., max:..} per axis.
        """
        limits = {"x": self.DEFAULT_LIMIT, "y": self.DEFAULT_LIMIT}
        try:
            settings = (config or {}).get("settings", {})
            cfg = settings.get("limits_mm", {})

            def _pair(v):
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return float(v[0]), float(v[1])
                if isinstance(v, dict):
                    mn = v.get("min")
                    mx = v.get("max")
                    if mn is not None and mx is not None:
                        return float(mn), float(mx)
                return None

            x_pair = _pair(cfg.get("x")) if isinstance(cfg, dict) else None
            y_pair = _pair(cfg.get("y")) if isinstance(cfg, dict) else None
            if x_pair:
                limits["x"] = x_pair
            if y_pair:
                limits["y"] = y_pair
        except Exception:
            pass
        return limits

    def _parse_home_load(self, config):
        """Parse home and load positions from config settings.
        Returns a dict with 'home' and 'load' keys containing (x, y) tuples.
        """
        positions = {"home": self.DEFAULT_HOME, "load": self.DEFAULT_LOAD}
        try:
            settings = (config or {}).get("settings", {})

            def _parse_position(pos_data):
                if isinstance(pos_data, (list, tuple)) and len(pos_data) == 2:
                    return (float(pos_data[0]), float(pos_data[1]))
                return None

            home_pos = _parse_position(settings.get("home"))
            load_pos = _parse_position(settings.get("load"))

            if home_pos:
                positions["home"] = home_pos
            if load_pos:
                positions["load"] = load_pos
        except Exception:
            pass
        return positions

    def get_limits(self):
        """Return dict with per-axis (min,max) tuple: { 'x': (min,max), 'y': (min,max) }"""
        return getattr(
            self, "_limits", {"x": self.DEFAULT_LIMIT, "y": self.DEFAULT_LIMIT}
        )

    def get_home_load_positions(self):
        """Return dict with 'home' and 'load' keys containing (x, y) tuples."""
        return getattr(
            self, "_positions", {"home": self.DEFAULT_HOME, "load": self.DEFAULT_LOAD}
        )

    def _check_axis_limits(self, x_mm, y_mm):
        """Check if the requested position is within axis limits.
        Raises StageAxisLimitError if exceeded."""
        limits = self.get_limits()
        x_min, x_max = limits["x"]
        y_min, y_max = limits["y"]
        if x_mm < x_min or x_mm > x_max:
            raise StageAxisLimitError("X", x_mm, x_min, x_max)
        if y_mm < y_min or y_mm > y_max:
            raise StageAxisLimitError("Y", y_mm, y_min, y_max)

    @abstractmethod
    def init_stage(self):
        pass

    @abstractmethod
    def home_stage(self, timeout_s=45):
        pass

    @abstractmethod
    def move_stage(self, x_mm, y_mm, move_timeout=20):
        pass

    @abstractmethod
    def get_xy_position(self):
        pass

    @abstractmethod
    def deinit(self):
        pass


class DummyStageController(BaseStageController):
    """DEV mode dummy stage controller."""

    def __init__(self, config):
        self._x = 0.0
        self._y = 0.0
        self.alias = config.get("alias", "DUMMY")
        self.id = config.get("id", "DUMMY-000")
        self._limits = self._parse_limits(config)
        self._positions = self._parse_home_load(config)

    def init_stage(self):
        print(f"Dummy stage '{self.alias}' initialized.")
        return True

    def home_stage(self, timeout_s=45):
        logging.info(f"Dummy stage '{self.alias}' homing operation started")
        print(f"Dummy stage '{self.alias}' homing.")
        time.sleep(1)
        self._x, self._y = 0.0, 0.0
        return self._x, self._y

    def move_stage(self, x_mm, y_mm, move_timeout=20):
        # Check axis limits before moving
        self._check_axis_limits(x_mm, y_mm)

        logging.info(
            f"Dummy stage '{self.alias}' move operation started: target ({x_mm:.3f}, {y_mm:.3f})"
        )
        print(f"Dummy stage moving to X={x_mm}, Y={y_mm}")
        time.sleep(0.25)
        self._x, self._y = x_mm, y_mm
        logging.info(
            f"Dummy stage '{self.alias}' move completed successfully to ({x_mm:.3f}, {y_mm:.3f})"
        )
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
        self._limits = self._parse_limits(config)
        self._positions = self._parse_home_load(config)

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
            self.lib.BDC_EnableChannel(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_EnableChannel(c_char_p(self.serial), c_short(self.y_chan))
            time.sleep(0.5)
            print(f"Stage '{self.alias}' initialized.")
            return True
        except Exception as e:
            print(f"Error during stage init: {e}")
            return False

    def home_stage(self, timeout_s=45):
        logging.info(f"Real stage '{self.alias}' homing operation started")
        self.lib.BDC_Home(c_char_p(self.serial), c_short(self.x_chan))
        self.lib.BDC_Home(c_char_p(self.serial), c_short(self.y_chan))
        start = time.time()
        while time.time() - start < timeout_s:
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.y_chan))
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
                print(f"Stage homed to X={x_mm:.3f}, Y={y_mm:.3f}")
                return x_mm, y_mm
        raise TimeoutError("Stage homing timed out")

    def move_stage(self, x_mm, y_mm, move_timeout=20):
        # Check axis limits before moving
        self._check_axis_limits(x_mm, y_mm)

        logging.info(
            f"Real stage '{self.alias}' move operation started: target ({x_mm:.3f}, {y_mm:.3f})"
        )
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
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.y_chan))
            time.sleep(0.5)
            curr_x_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.x_chan)
            )
            curr_y_dev = self.lib.BDC_GetPosition(
                c_char_p(self.serial), c_short(self.y_chan)
            )
            if abs(curr_x_dev - x_dev) + abs(curr_y_dev - y_dev) <= 4:
                x_mm = curr_x_dev / self.scaling_factor
                y_mm = curr_y_dev / self.scaling_factor
                print(f"Stage moved to X={x_mm:.3f}, Y={y_mm:.3f}")
                return x_mm, y_mm
        # Log timeout error with details
        logging.error(
            f"Real stage '{self.alias}' move operation timed out after {move_timeout}s. "
            f"Target: ({x_mm:.3f}, {y_mm:.3f}). Please check hardware and try again."
        )
        raise TimeoutError(f"Stage move timed out after {move_timeout} seconds")

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
