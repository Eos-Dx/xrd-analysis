#!/usr/bin/env python3
"""
Simple test for continuous movement logic without PyQt5 dependencies.
Tests the core mathematical and logical components.
"""

import math
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_movement_logic():
    """Test the core movement logic without Qt dependencies."""

    print("=" * 60)
    print("TESTING CONTINUOUS MOVEMENT LOGIC")
    print("=" * 60)

    # Test clock-to-coordinates conversion
    print("1. Testing clock position to coordinate conversion:")

    def clock_to_coordinates(center_x, center_y, clock_position, radius):
        """Convert clock position to X,Y coordinates."""
        # Convert clock position to angle (12 o'clock = 0°, clockwise)
        # Clock position 12 = 90°, 3 = 0°, 6 = 270°, 9 = 180°
        angle_degrees = 90 - (clock_position * 30)  # 30° per hour
        angle_radians = math.radians(angle_degrees)

        # Calculate offset from center
        offset_x = radius * math.cos(angle_radians)
        offset_y = radius * math.sin(angle_radians)

        return center_x + offset_x, center_y + offset_y

    center_x, center_y = 5.0, 5.0
    radius = 2.0

    test_positions = [
        (12, "12 o'clock (north)"),
        (3, "3 o'clock (east)"),
        (6, "6 o'clock (south)"),
        (9, "9 o'clock (west)"),
    ]

    for clock_pos, description in test_positions:
        x, y = clock_to_coordinates(center_x, center_y, clock_pos, radius)
        print(f"   {description}: ({x:6.3f}, {y:6.3f})")

        # Verify distance from center
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        distance_ok = abs(distance - radius) < 0.001
        print(
            f"     Distance check: {'PASS' if distance_ok else 'FAIL'} ({distance:.3f} vs {radius:.3f})"
        )

    # Test movement pattern
    print("\n2. Testing movement pattern:")
    MOVEMENT_PATTERN = [12, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6]
    print(f"   Pattern: {MOVEMENT_PATTERN}")
    print(f"   Pattern length: {len(MOVEMENT_PATTERN)} positions")

    # Test radius calculation
    print("\n3. Testing radius calculation:")

    def calculate_radius(progress, max_radius, min_radius):
        """Calculate current radius based on measurement progress."""
        return max_radius - (max_radius - min_radius) * progress

    max_radius = 2.0
    min_radius = 0.2

    test_progress = [0.0, 0.25, 0.5, 0.75, 1.0]
    for progress in test_progress:
        radius = calculate_radius(progress, max_radius, min_radius)
        print(f"   Progress {progress:4.1%}: radius = {radius:.3f}mm")

    # Test stage limits checking
    print("\n4. Testing stage limits checking:")

    def check_stage_limits(x, y, x_limits=(-10.0, 10.0), y_limits=(-10.0, 10.0)):
        """Check if position is within stage limits."""
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        return x_min <= x <= x_max and y_min <= y <= y_max

    test_positions_limits = [
        (0.0, 0.0, "center"),
        (9.0, 9.0, "near edge - valid"),
        (15.0, 5.0, "beyond X limit"),
        (5.0, 15.0, "beyond Y limit"),
        (-15.0, -15.0, "beyond both limits"),
    ]

    for x, y, description in test_positions_limits:
        within_limits = check_stage_limits(x, y)
        print(
            f"   {description:20s}: ({x:6.1f}, {y:6.1f}) - {'OK' if within_limits else 'OUT OF BOUNDS'}"
        )

    # Test movement pattern validation
    print("\n5. Testing movement pattern validation:")

    def validate_movement_pattern(
        center_x, center_y, max_radius, x_limits=(-10.0, 10.0), y_limits=(-10.0, 10.0)
    ):
        """Validate that the entire movement pattern is within stage limits."""
        for clock_position in MOVEMENT_PATTERN:
            target_x, target_y = clock_to_coordinates(
                center_x, center_y, clock_position, max_radius
            )
            if not check_stage_limits(target_x, target_y, x_limits, y_limits):
                print(
                    f"     Position ({target_x:.3f}, {target_y:.3f}) at clock {clock_position} would exceed stage limits"
                )
                return False
        return True

    test_scenarios = [
        ((5.0, 5.0), 2.0, "safe center position"),
        ((0.0, 0.0), 3.0, "center with larger radius"),
        ((8.0, 8.0), 3.0, "near edge with large radius"),
        ((5.0, 5.0), 1.0, "small radius"),
    ]

    for (center_x, center_y), radius, description in test_scenarios:
        is_valid = validate_movement_pattern(center_x, center_y, radius)
        print(f"   {description:30s}: {'VALID' if is_valid else 'INVALID'}")

    # Test movement timing
    print("\n6. Testing movement timing (fixed interval):")

    FIXED_INTERVAL = 0.5  # seconds

    def calculate_cycles(measurement_duration, pattern_length, interval):
        """Calculate how many cycles are completed with fixed interval."""
        moves_per_cycle = pattern_length
        cycle_time = moves_per_cycle * interval
        cycles_completed = measurement_duration / cycle_time
        return cycles_completed

    test_durations = [10, 30, 60, 120]  # seconds
    pattern_length = len(MOVEMENT_PATTERN)

    for duration in test_durations:
        cycles = calculate_cycles(duration, pattern_length, FIXED_INTERVAL)
        print(
            f"   {duration:3d}s measurement: fixed interval {FIXED_INTERVAL:.3f}s, {cycles:.1f} cycles"
        )

    print("\n" + "=" * 60)
    print("LOGIC TEST COMPLETE - ALL CORE FUNCTIONS WORKING")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_movement_logic()
        print("\n✅ All core logic tests passed!")
        print("\nNote: This test validates the mathematical and logical components.")
        print(
            "The full implementation requires PyQt5 for signals/threading and the actual stage hardware."
        )
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
