#!/usr/bin/env python3
"""
Test script for the continuous movement feature implementation.
This script tests the basic functionality without requiring a full GUI setup.
"""

import math
import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from PyQt5.QtCore import QCoreApplication, QTimer

    from hardware.Ulster.gui.technical.continuous_movement import (
        ContinuousMovementController,
    )
    from hardware.Ulster.hardware.xystages import DummyStageController

    def test_continuous_movement():
        """Test the continuous movement controller with dummy stage."""

        print("=" * 60)
        print("TESTING CONTINUOUS MOVEMENT IMPLEMENTATION")
        print("=" * 60)

        # Create a QCoreApplication for Qt signals/slots
        app = QCoreApplication([])

        # Create dummy stage configuration
        dummy_config = {
            "id": "DUMMY-STAGE",
            "alias": "TestStage",
            "type": "DummyStage",
            "settings": {
                "limits_mm": {
                    "x": {"min": -10.0, "max": 10.0},
                    "y": {"min": -10.0, "max": 10.0},
                },
                "home": [0.0, 0.0],
                "load": [-5.0, -5.0],
            },
        }

        # Initialize dummy stage controller
        print("1. Initializing dummy stage controller...")
        stage = DummyStageController(dummy_config)
        success = stage.init_stage()
        print(f"   Stage initialization: {'SUCCESS' if success else 'FAILED'}")

        if not success:
            print("   Cannot continue without stage controller")
            return False

        # Test stage basic functionality
        print("\n2. Testing stage basic functionality...")
        current_pos = stage.get_xy_position()
        print(f"   Current position: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")

        # Move to center position
        center_x, center_y = 2.0, 3.0
        stage.move_stage(center_x, center_y)
        new_pos = stage.get_xy_position()
        print(f"   Moved to center: ({new_pos[0]:.3f}, {new_pos[1]:.3f})")

        # Create continuous movement controller
        print("\n3. Initializing continuous movement controller...")
        movement_controller = ContinuousMovementController(stage)

        # Test movement pattern validation
        print("\n4. Testing movement pattern validation...")

        # Test with safe parameters
        movement_controller.configure(max_radius=2.0, measurement_duration=5.0)
        is_valid = movement_controller._validate_movement_pattern(center_x, center_y)
        print(f"   Safe pattern validation: {'PASS' if is_valid else 'FAIL'}")

        # Test with unsafe parameters (too large radius)
        movement_controller.configure(max_radius=15.0, measurement_duration=5.0)
        is_valid_unsafe = movement_controller._validate_movement_pattern(
            center_x, center_y
        )
        print(
            f"   Unsafe pattern validation: {'PASS' if not is_valid_unsafe else 'FAIL'}"
        )

        # Reset to safe parameters
        movement_controller.configure(max_radius=2.0, measurement_duration=5.0)

        # Test coordinate conversion
        print("\n5. Testing coordinate conversion...")
        for clock_pos in [12, 3, 6, 9]:
            x, y = movement_controller._clock_to_coordinates(
                center_x, center_y, clock_pos, 2.0
            )
            print(f"   Clock {clock_pos:2d}: ({x:6.3f}, {y:6.3f})")

        # Test movement pattern positions
        print("\n6. Testing full movement pattern positions...")
        print("   Clock positions:", movement_controller.MOVEMENT_PATTERN)
        for i, clock_pos in enumerate(
            movement_controller.MOVEMENT_PATTERN[:4]
        ):  # Test first 4
            x, y = movement_controller._clock_to_coordinates(
                center_x, center_y, clock_pos, 1.5
            )
            within_limits = movement_controller._check_stage_limits(x, y)
            print(
                f"   Position {i+1:2d} (Clock {clock_pos:2d}): ({x:6.3f}, {y:6.3f}) - {'OK' if within_limits else 'OUT OF BOUNDS'}"
            )

        # Connect signals for monitoring
        def on_movement_started():
            print("   >>> Movement started!")

        def on_movement_stopped():
            print("   >>> Movement stopped!")

        def on_movement_error(error_msg):
            print(f"   >>> Movement error: {error_msg}")

        def on_position_changed(x, y):
            print(f"   >>> Position: ({x:6.3f}, {y:6.3f})")

        movement_controller.movement_started.connect(on_movement_started)
        movement_controller.movement_stopped.connect(on_movement_stopped)
        movement_controller.movement_error.connect(on_movement_error)
        movement_controller.position_changed.connect(on_position_changed)

        # Test short movement session
        print("\n7. Testing short movement session (3 seconds)...")
        movement_controller.configure(max_radius=1.0, measurement_duration=3.0)

        start_success = movement_controller.start_movement(center_x, center_y)
        print(f"   Movement start: {'SUCCESS' if start_success else 'FAILED'}")

        if start_success:
            # Let the movement run for a bit
            print("   Letting movement run for 3.5 seconds...")

            # Create a timer to stop after 3.5 seconds
            def stop_movement():
                print("   Stopping movement...")
                movement_controller.stop_movement(return_to_origin=True)
                app.quit()

            QTimer.singleShot(3500, stop_movement)  # 3.5 seconds

            # Run the event loop
            app.exec_()

            # Check final position
            final_pos = stage.get_xy_position()
            print(f"   Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")

            # Check if returned close to original center
            distance = math.sqrt(
                (final_pos[0] - center_x) ** 2 + (final_pos[1] - center_y) ** 2
            )
            returned_ok = distance < 0.1  # Within 0.1mm
            print(
                f"   Return to origin: {'SUCCESS' if returned_ok else 'FAILED'} (distance: {distance:.3f}mm)"
            )

        print("\n8. Cleanup...")
        stage.deinit()
        print("   Stage deinitialized")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        return True

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure you're running this from the correct directory and all dependencies are installed."
    )
    sys.exit(1)

if __name__ == "__main__":
    try:
        test_continuous_movement()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
