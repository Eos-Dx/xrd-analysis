# xystages.py
import logging
import os
import sys
import time
import queue
import threading
from abc import ABC, abstractmethod
from ctypes import CDLL, c_char_p, c_int, c_short

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


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
        self._io_lock = threading.RLock()
        self.alias = config.get("alias", "DUMMY")
        self.id = config.get("id", "DUMMY-000")
        self._limits = self._parse_limits(config)
        self._positions = self._parse_home_load(config)

    def init_stage(self):
        print(f"Dummy stage '{self.alias}' initialized.")
        return True

    def home_stage(self, timeout_s=45):
        with self._io_lock:
            logging.info(f"Dummy stage '{self.alias}' homing operation started")
            print(f"Dummy stage '{self.alias}' homing.")
            time.sleep(1)
            self._x, self._y = 0.0, 0.0
            return self._x, self._y

    def move_stage(self, x_mm, y_mm, move_timeout=20):
        with self._io_lock:
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
        with self._io_lock:
            return self._x, self._y

    def deinit(self):
        print(f"Dummy stage '{self.alias}' deinitialized.")


class MarlinStageController(BaseStageController):
    """Stage controller for Marlin/GRBL firmware via serial communication."""

    DEFAULT_LIMIT = (0.0, 90.0)  # Marlin stage default limits
    DEFAULT_HOME = (0.0, 0.0)  # Marlin home position
    DEFAULT_LOAD = (90.0, 0.0)  # Sample out position

    def __init__(
        self,
        config,
        port=None,
        baudrate=115200,
        timeout=1.0,
        feedrate=3000,
        homing_timeout=15,
    ):
        """
        Initialize Marlin stage controller.

        Args:
            config: Configuration dict with 'id' (port), 'alias', and optional 'settings'
            port: Serial port (e.g., 'COM4'). If None, uses config['id']
            baudrate: Serial baud rate (default: 115200)
            timeout: Serial read timeout in seconds
            feedrate: Movement speed in mm/min (default: 3000)
            homing_timeout: Timeout for homing operation in seconds
        """
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial is required for MarlinStageController")

        self.alias = config.get("alias", "MARLIN_STAGE")
        self.port = port or config.get("id")
        self.baudrate = baudrate
        self.timeout = timeout
        self.feedrate = feedrate
        self.homing_timeout = homing_timeout
        self._io_lock = threading.RLock()

        self.ser = None
        self.ser_thread = None
        self.out_q = queue.Queue()
        self._running = False

        # Current position tracking
        self._x = 0.0
        self._y = 0.0

        # Parse limits and positions from config
        self._limits = self._parse_limits(config)
        self._positions = self._parse_home_load(config)

        # Override Y limits if not specified (Marlin stage typically has different Y range)
        if "y" not in config.get("settings", {}).get("limits_mm", {}):
            self._limits["y"] = (0.0, 100.0)

    def _serial_reader_thread(self):
        """Background thread to continuously read from serial port."""
        while self._running:
            try:
                if self.ser and self.ser.in_waiting:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        self.out_q.put(line)
                else:
                    time.sleep(0.01)
            except Exception as e:
                logging.error(f"Serial read error on '{self.alias}': {e}")
                self._running = False

    def _send_gcode(self, command, wait_for_ok=True, timeout=5.0):
        """
        Send G-code command to the controller.

        Args:
            command: G-code command string (or multi-line string)
            wait_for_ok: Wait for 'ok' response from controller
            timeout: Maximum time to wait for response

        Returns:
            True if successful (or if not waiting for ok), False on timeout
        """
        with self._io_lock:
            if not self.ser or not self.ser.is_open:
                raise RuntimeError(f"Serial port not open on '{self.alias}'")

            lines = command.strip().splitlines()
            for line in lines:
                cmd = line.strip()
                if not cmd:
                    continue
                try:
                    self.ser.write((cmd + "\n").encode())
                    time.sleep(0.02)  # Small delay between commands
                    logging.debug(f"Sent to '{self.alias}': {cmd}")
                except Exception as e:
                    logging.error(f"Error sending command to '{self.alias}': {e}")
                    raise

            if wait_for_ok:
                start = time.time()
                while time.time() - start < timeout:
                    if not self.out_q.empty():
                        response = self.out_q.get()
                        if "ok" in response.lower():
                            return True
                    time.sleep(0.05)
                logging.warning(f"Timeout waiting for 'ok' from '{self.alias}'")
                return False

            return True

    def _request_position(self):
        """Request current position via M114 and parse response."""
        with self._io_lock:
            # Clear queue before requesting
            while not self.out_q.empty():
                self.out_q.get()

            self._send_gcode("M114", wait_for_ok=False)
            time.sleep(0.1)

            # Parse position from response
            start = time.time()
            while time.time() - start < 2.0:
                if not self.out_q.empty():
                    line = self.out_q.get()
                    if line.startswith("X:"):
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.startswith("X:") and (
                                    i == 0 or parts[i - 1] != "Count"
                                ):
                                    self._x = float(part[2:])
                                elif part.startswith("Y:") and (
                                    i == 0 or parts[i - 1] != "Count"
                                ):
                                    self._y = float(part[2:])
                                if part == "Count":
                                    break
                            return self._x, self._y
                        except (ValueError, IndexError) as e:
                            logging.warning(
                                f"Error parsing position from '{self.alias}': {e}"
                            )
                time.sleep(0.05)

            return self._x, self._y

    def init_stage(self):
        """
        Initialize the Marlin stage controller by opening serial connection.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Wait for controller to reset

            # Start serial reader thread
            self._running = True
            self.ser_thread = threading.Thread(
                target=self._serial_reader_thread, daemon=True
            )
            self.ser_thread.start()

            # Clear buffers
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            # Set absolute positioning mode
            self._send_gcode("G90", wait_for_ok=True)

            # Request initial position
            self._request_position()

            print(f"Marlin stage '{self.alias}' initialized on {self.port}")
            logging.info(
                f"Marlin stage '{self.alias}' initialized on {self.port} at {self.baudrate} baud"
            )
            return True

        except Exception as e:
            logging.error(f"Error initializing Marlin stage '{self.alias}': {e}")
            print(f"Error initializing stage '{self.alias}': {e}")
            self.ser = None
            return False

    def home_stage(self, timeout_s=45):
        """
        Home the stage using G28 command.

        Args:
            timeout_s: Maximum time to wait for homing to complete

        Returns:
            tuple: (x_mm, y_mm) position after homing
        """
        with self._io_lock:
            logging.info(f"Marlin stage '{self.alias}' homing operation started")
            print(f"Marlin stage '{self.alias}' homing...")

            # Send homing command
            self._send_gcode("G28 X Y", wait_for_ok=False)

            # Wait for homing to complete
            time.sleep(min(timeout_s, self.homing_timeout))

            # Update position to home
            self._x, self._y = 0.0, 0.0

            # Request actual position from controller
            pos = self._request_position()

            print(
                f"Marlin stage '{self.alias}' homed to X={self._x:.3f}, Y={self._y:.3f}"
            )
            logging.info(
                f"Marlin stage '{self.alias}' homing completed at ({self._x:.3f}, {self._y:.3f})"
            )
            return pos

    def move_stage(self, x_mm, y_mm, move_timeout=20):
        """
        Move stage to absolute position.

        Args:
            x_mm: Target X position in mm
            y_mm: Target Y position in mm
            move_timeout: Maximum time to wait for move to complete

        Returns:
            tuple: (x_mm, y_mm) final position
        """
        with self._io_lock:
            # Check axis limits before moving
            self._check_axis_limits(x_mm, y_mm)

            logging.info(
                f"Marlin stage '{self.alias}' move operation started: target ({x_mm:.3f}, {y_mm:.3f})"
            )

            # Calculate movement time based on distance and feedrate
            start_x, start_y = self._x, self._y
            dx = abs(x_mm - start_x)
            dy = abs(y_mm - start_y)
            max_distance = max(dx, dy)
            estimated_time = (max_distance / (self.feedrate / 60.0)) + 0.5  # Add buffer

            # Send move command
            cmd = f"G90\nG0 X{x_mm:.3f} Y{y_mm:.3f} F{self.feedrate}"
            self._send_gcode(cmd, wait_for_ok=False)

            # Wait for movement to complete
            time.sleep(min(estimated_time, move_timeout))

            # Update internal position
            self._x = x_mm
            self._y = y_mm

            # Request actual position
            final_pos = self._request_position()

            print(
                f"Marlin stage '{self.alias}' moved to X={self._x:.3f}, Y={self._y:.3f}"
            )
            logging.info(
                f"Marlin stage '{self.alias}' move completed successfully to ({self._x:.3f}, {self._y:.3f})"
            )
            return final_pos

    def get_xy_position(self):
        """
        Get current XY position.

        Returns:
            tuple: (x_mm, y_mm) current position
        """
        with self._io_lock:
            return self._request_position()

    def emergency_stop(self):
        """Send emergency stop command (M112)."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"M112\n")
                logging.warning(f"Emergency stop sent to '{self.alias}'")
                print(f"Emergency stop activated on '{self.alias}'")
            except Exception as e:
                logging.error(f"Error sending emergency stop to '{self.alias}': {e}")

    def deinit(self):
        """Close serial connection and cleanup."""
        try:
            self._running = False
            if self.ser_thread:
                self.ser_thread.join(timeout=0.5)
                self.ser_thread = None

            if self.ser and self.ser.is_open:
                self.ser.close()

            print(f"Marlin stage '{self.alias}' deinitialized.")
            logging.info(f"Marlin stage '{self.alias}' deinitialized")
        except Exception as e:
            logging.error(f"Error during Marlin stage '{self.alias}' deinit: {e}")
        finally:
            self.ser = None


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
        self.config = config or {}
        self.serial = config["id"].encode()
        self.alias = config.get("alias", "XY_STAGE")
        self.x_chan = x_chan
        self.y_chan = y_chan
        self.scaling_factor = scaling_factor
        self.sim = sim
        self.poll_interval_ms = poll_interval_ms
        self.lib = None
        self._kinesis_sdk_path = self._resolve_kinesis_sdk_path(self.config)
        self._io_lock = threading.RLock()
        self._pos_cache_lock = threading.RLock()
        self._last_x_mm = 0.0
        self._last_y_mm = 0.0
        self._limits = self._parse_limits(config)
        self._positions = self._parse_home_load(config)

    def _resolve_kinesis_sdk_path(self, config):
        candidates = []

        env_path = str(os.environ.get("KINESIS_SDK_PATH", "")).strip()
        if env_path:
            candidates.append(env_path)

        cfg_path = str((config or {}).get("kinesis_sdk_path", "")).strip()
        if cfg_path:
            candidates.append(cfg_path)

        settings_path = str(
            ((config or {}).get("settings", {}) or {}).get("kinesis_sdk_path", "")
        ).strip()
        if settings_path:
            candidates.append(settings_path)

        candidates.extend(
            [
                r"C:\Program Files\Thorlabs\Kinesis",
                r"C:\Program Files (x86)\Thorlabs\Kinesis",
            ]
        )

        program_files = str(os.environ.get("ProgramFiles", "")).strip()
        if program_files:
            candidates.append(os.path.join(program_files, "Thorlabs", "Kinesis"))

        program_files_x86 = str(os.environ.get("ProgramFiles(x86)", "")).strip()
        if program_files_x86:
            candidates.append(os.path.join(program_files_x86, "Thorlabs", "Kinesis"))

        seen = set()
        for path in candidates:
            normalized = os.path.normpath(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            dll_path = os.path.join(
                normalized, "Thorlabs.MotionControl.Benchtop.DCServo.dll"
            )
            if os.path.isdir(normalized) and os.path.isfile(dll_path):
                return normalized
        return ""

    def init_stage(self):
        try:
            if not self._kinesis_sdk_path:
                print(
                    "Error during stage init: unable to locate Thorlabs Kinesis SDK "
                    "(set KINESIS_SDK_PATH to folder containing Thorlabs.MotionControl.Benchtop.DCServo.dll)"
                )
                return False

            dll_name = "Thorlabs.MotionControl.Benchtop.DCServo.dll"
            dll_path = os.path.join(self._kinesis_sdk_path, dll_name)
            if sys.version_info < (3, 8):
                os.chdir(self._kinesis_sdk_path)
                self.lib = CDLL(dll_path if os.path.isfile(dll_path) else dll_name)
            else:
                os.add_dll_directory(self._kinesis_sdk_path)
                self.lib = CDLL(dll_path if os.path.isfile(dll_path) else dll_name)

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
            try:
                x_mm, y_mm = self.get_xy_position()
                with self._pos_cache_lock:
                    self._last_x_mm = float(x_mm)
                    self._last_y_mm = float(y_mm)
            except Exception:
                pass
            print(f"Stage '{self.alias}' initialized.")
            return True
        except Exception as e:
            print(f"Error during stage init: {e}")
            return False

    def home_stage(self, timeout_s=45):
        with self._io_lock:
            logging.info(
                "Real stage homing operation started",
                extra={
                    "stage_alias": self.alias,
                    "thread_name": threading.current_thread().name,
                },
            )
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
        with self._io_lock:
            # Check axis limits before moving
            self._check_axis_limits(x_mm, y_mm)

            logging.info(
                "Real stage move operation started",
                extra={
                    "stage_alias": self.alias,
                    "target_x_mm": x_mm,
                    "target_y_mm": y_mm,
                    "thread_name": threading.current_thread().name,
                },
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
                with self._pos_cache_lock:
                    self._last_x_mm = curr_x_dev / self.scaling_factor
                    self._last_y_mm = curr_y_dev / self.scaling_factor
                if abs(curr_x_dev - x_dev) + abs(curr_y_dev - y_dev) <= 1000:
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
        acquired = self._io_lock.acquire(timeout=0.05)
        if not acquired:
            with self._pos_cache_lock:
                return float(self._last_x_mm), float(self._last_y_mm)
        try:
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.x_chan))
            self.lib.BDC_RequestPosition(c_char_p(self.serial), c_short(self.y_chan))
            time.sleep(0.05)
            x_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.x_chan))
            y_dev = self.lib.BDC_GetPosition(c_char_p(self.serial), c_short(self.y_chan))
            x_mm = x_dev / self.scaling_factor
            y_mm = y_dev / self.scaling_factor
            with self._pos_cache_lock:
                self._last_x_mm = x_mm
                self._last_y_mm = y_mm
            return x_mm, y_mm
        finally:
            self._io_lock.release()

    def deinit(self):
        lib = getattr(self, "lib", None)
        if lib is None:
            print(f"Stage '{self.alias}' already deinitialized.")
            return

        try:
            try:
                lib.BDC_StopPolling(c_char_p(self.serial), c_short(self.x_chan))
                lib.BDC_StopPolling(c_char_p(self.serial), c_short(self.y_chan))
            except Exception:
                pass
            lib.BDC_Close(c_char_p(self.serial))
            if self.sim:
                lib.TLI_UninitializeSimulations()
            print(f"Stage '{self.alias}' deinitialized.")
        except Exception as e:
            print(f"Stage deinit error: {e}")
        finally:
            self.lib = None
