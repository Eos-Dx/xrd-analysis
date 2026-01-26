import math
import threading
import time
from typing import Callable, Optional, Tuple

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class ContinuousMovementController(QObject):
    """
    Controller for continuous circular movement during AgBH measurements.

    Implements a specific movement pattern to smooth out sample inconsistencies:
    - Moves in circular pattern with clock positions (12, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6)
    - Gradually decreases radius during measurement
    - Returns to original position when finished
    """

    # Movement pattern in clock positions (hours)
    MOVEMENT_PATTERN = [12, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6]

    # Signals
    movement_started = pyqtSignal()
    movement_stopped = pyqtSignal()
    movement_error = pyqtSignal(str)  # Error message
    position_changed = pyqtSignal(float, float)  # x, y coordinates

    def __init__(self, stage_controller, parent=None):
        """
        Initialize the continuous movement controller.

        Args:
            stage_controller: The hardware stage controller (BaseStageController)
            parent: Qt parent object
        """
        super().__init__(parent)
        self.stage_controller = stage_controller
        self.original_position: Optional[Tuple[float, float]] = None
        self.is_active = False
        self.movement_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Movement parameters
        self.max_radius = 2.0  # mm
        self.measurement_duration = 60.0  # seconds
        self.movement_interval = 0.5  # seconds between movements (fixed)

        # Radius decrease strategy (stepwise per full cycle)
        self.radius_step = 0.2  # mm per full pattern cycle
        self.min_radius = 0.2  # Minimum radius (mm)
        self._current_radius = self.max_radius

    def configure(self, max_radius: float, measurement_duration: float):
        """
        Configure movement parameters.

        Args:
            max_radius: Maximum radius in mm
            measurement_duration: Total measurement duration in seconds
        """
        self.max_radius = max_radius
        self.measurement_duration = measurement_duration
        # Note: movement_interval remains fixed (no adaptive timing) to keep motion predictable

    def start_movement(self, center_x: float, center_y: float) -> bool:
        """
        Start continuous movement around the specified center position.

        Args:
            center_x: Center X coordinate in mm
            center_y: Center Y coordinate in mm

        Returns:
            bool: True if movement started successfully, False otherwise
        """
        if self.is_active:
            print("Movement already active")
            return False

        if not self.stage_controller:
            self.movement_error.emit("No stage controller available")
            return False

        # Safety check: verify that the movement pattern won't exceed stage limits
        if not self._validate_movement_pattern(center_x, center_y):
            self.movement_error.emit(
                f"Movement pattern would exceed stage limits. "
                f"Center: ({center_x:.3f}, {center_y:.3f}), Max radius: {self.max_radius:.3f}mm"
            )
            return False

        try:
            # Store original position
            self.original_position = (center_x, center_y)

            # Reset stop event
            self.stop_event.clear()

            # Record start time for radius tracking helpers
            self._start_time = time.time()

            # Start movement in separate thread
            self.movement_thread = threading.Thread(
                target=self._movement_loop, args=(center_x, center_y), daemon=True
            )
            self.is_active = True
            self.movement_thread.start()

            self.movement_started.emit()
            print(
                f"Started continuous movement around ({center_x:.3f}, {center_y:.3f})"
            )
            return True

        except Exception as e:
            self.movement_error.emit(f"Failed to start movement: {str(e)}")
            self.is_active = False
            return False

    def stop_movement(self, return_to_origin: bool = True) -> bool:
        """
        Stop continuous movement and optionally return to original position.

        Args:
            return_to_origin: Whether to return to original position

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_active:
            return True

        print("Stopping continuous movement...")

        # Signal the movement thread to stop
        self.stop_event.set()

        # Wait for thread to finish
        if self.movement_thread and self.movement_thread.is_alive():
            self.movement_thread.join(timeout=5.0)  # Wait up to 5 seconds

        self.is_active = False

        # Return to original position if requested and available
        if return_to_origin and self.original_position and self.stage_controller:
            try:
                x, y = self.original_position
                self.stage_controller.move_stage(x, y)
                print(f"Returned to original position ({x:.3f}, {y:.3f})")
                self.position_changed.emit(x, y)
            except Exception as e:
                self.movement_error.emit(
                    f"Failed to return to original position: {str(e)}"
                )

        self.movement_stopped.emit()
        print("Continuous movement stopped")
        return True

    def _movement_loop(self, center_x: float, center_y: float):
        """
        Main movement loop running in separate thread.

        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate
        """
        start_time = time.time()
        pattern_index = 0
        current_radius = self.max_radius
        self._current_radius = current_radius

        try:
            while not self.stop_event.is_set():
                elapsed_time = time.time() - start_time

                # Check if measurement duration exceeded
                if elapsed_time >= self.measurement_duration:
                    break

                # Get next position in pattern using current (stepwise) radius
                clock_position = self.MOVEMENT_PATTERN[pattern_index]
                target_x, target_y = self._clock_to_coordinates(
                    center_x, center_y, clock_position, current_radius
                )

                # Check stage limits before moving
                if self._check_stage_limits(target_x, target_y):
                    try:
                        # Move to target position
                        self.stage_controller.move_stage(target_x, target_y)
                        self.position_changed.emit(target_x, target_y)

                    except Exception as e:
                        error_msg = f"Stage movement failed: {str(e)}"
                        print(error_msg)
                        self.movement_error.emit(error_msg)
                        break
                else:
                    print(
                        f"Skipping position ({target_x:.3f}, {target_y:.3f}) - outside stage limits"
                    )

                # Compute next index and check for full cycle completion
                next_index = (pattern_index + 1) % len(self.MOVEMENT_PATTERN)
                if next_index == 0:
                    # Completed one full cycle → step radius down
                    current_radius = max(
                        self.min_radius, current_radius - self.radius_step
                    )
                    self._current_radius = current_radius

                # Advance pattern index
                pattern_index = next_index

                # Wait for next movement (fixed interval)
                if not self.stop_event.wait(self.movement_interval):
                    continue  # Continue if not stopping
                else:
                    break  # Stop was requested

        except Exception as e:
            error_msg = f"Movement loop error: {str(e)}"
            print(error_msg)
            self.movement_error.emit(error_msg)

        finally:
            self.is_active = False

    # Removed adaptive continuous radius to avoid implicit timing coupling.
    # Radius now decreases stepwise after each full pattern cycle.

    def _clock_to_coordinates(
        self, center_x: float, center_y: float, clock_position: int, radius: float
    ) -> Tuple[float, float]:
        """
        Convert clock position to X,Y coordinates.

        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate
            clock_position: Clock position (1-12)
            radius: Radius in mm

        Returns:
            Tuple[float, float]: Target X,Y coordinates
        """
        # Convert clock position to angle (12 o'clock = 0°, clockwise)
        # Clock position 12 = 90°, 3 = 0°, 6 = 270°, 9 = 180°
        angle_degrees = 90 - (clock_position * 30)  # 30° per hour
        angle_radians = math.radians(angle_degrees)

        # Calculate offset from center
        offset_x = radius * math.cos(angle_radians)
        offset_y = radius * math.sin(angle_radians)

        return center_x + offset_x, center_y + offset_y

    def _validate_movement_pattern(self, center_x: float, center_y: float) -> bool:
        """
        Validate that the entire movement pattern is within stage limits.

        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate

        Returns:
            bool: True if all positions are within limits, False otherwise
        """
        if not self.stage_controller:
            return False

        try:
            # Check all positions in the movement pattern with maximum radius
            for clock_position in self.MOVEMENT_PATTERN:
                target_x, target_y = self._clock_to_coordinates(
                    center_x, center_y, clock_position, self.max_radius
                )
                if not self._check_stage_limits(target_x, target_y):
                    print(
                        f"Position ({target_x:.3f}, {target_y:.3f}) at clock {clock_position} would exceed stage limits"
                    )
                    return False
            return True
        except Exception as e:
            print(f"Error validating movement pattern: {e}")
            return False

    def _check_stage_limits(self, x: float, y: float) -> bool:
        """
        Check if position is within stage limits.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            bool: True if within limits, False otherwise
        """
        if not self.stage_controller:
            return False

        try:
            limits = self.stage_controller.get_limits()
            x_min, x_max = limits.get("x", (-14.0, 14.0))
            y_min, y_max = limits.get("y", (-14.0, 14.0))

            return x_min <= x <= x_max and y_min <= y <= y_max

        except Exception as e:
            print(f"Error checking stage limits: {e}")
            return False

    def is_moving(self) -> bool:
        """
        Check if movement is currently active.

        Returns:
            bool: True if movement is active
        """
        return self.is_active

    def get_current_radius(self) -> float:
        """
        Get the current movement radius.

        Returns:
            float: Current radius in mm
        """
        return getattr(self, "_current_radius", self.max_radius)
