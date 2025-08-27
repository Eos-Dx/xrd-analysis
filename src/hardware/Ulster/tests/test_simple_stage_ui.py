#!/usr/bin/env python
"""
Simple test for stage UI error handling without complex imports.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.xystages import DummyStageController, StageAxisLimitError


class SimpleUIController:
    """Simplified UI controller for testing stage limits."""

    def __init__(self):
        self.x_value = 0.0
        self.y_value = 0.0
        self.last_error_message = None
        self.hardware_initialized = True
        self.stage_controller = None

    def show_error(self, title, message):
        """Mock error message display."""
        self.last_error_message = {"title": title, "message": message}

    def goto_stage_position(self):
        """Simplified version of the GoTo functionality."""
        if not (
            hasattr(self, "stage_controller")
            and getattr(self, "hardware_initialized", False)
            and self.stage_controller is not None
        ):
            self.show_error("Stage Not Ready", "Stage not initialized; cannot GoTo.")
            return

        x = self.x_value
        y = self.y_value
        try:
            new_x, new_y = self.stage_controller.move_stage(x, y)
            # Success - no error message
            self.last_error_message = None
        except StageAxisLimitError as e:
            # Get limits for the error message
            try:
                limits = self.stage_controller.get_limits()
                x_min, x_max = limits.get("x", (None, None))
                y_min, y_max = limits.get("y", (None, None))
                message = (
                    f"Requested position ({x:.3f}, {y:.3f}) is outside limits:\n"
                    f"X[{x_min:.1f}, {x_max:.1f}] mm, Y[{y_min:.1f}, {y_max:.1f}] mm"
                )
            except Exception:
                message = str(e)
            self.show_error("Stage Move Error", message)
        except Exception as e:
            self.show_error("Stage Move Error", str(e))


class TestSimpleStageUI(unittest.TestCase):
    """Test stage UI functionality with limits."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = SimpleUIController()
        stage_config = {"alias": "test_stage", "id": "test-123"}
        self.ui.stage_controller = DummyStageController(stage_config)

    def test_valid_move_no_error(self):
        """Test that valid moves don't show error."""
        self.ui.x_value = 5.0
        self.ui.y_value = -5.0
        self.ui.goto_stage_position()

        # Should not show error
        self.assertIsNone(self.ui.last_error_message)

        # Should have moved to requested position
        x, y = self.ui.stage_controller.get_xy_position()
        self.assertEqual(x, 5.0)
        self.assertEqual(y, -5.0)

    def test_x_limit_exceeded_shows_error(self):
        """Test that exceeding X limit shows specific error."""
        self.ui.x_value = 15.0  # Beyond +14.0 limit
        self.ui.y_value = 0.0
        self.ui.goto_stage_position()

        # Should show error
        self.assertIsNotNone(self.ui.last_error_message)
        error = self.ui.last_error_message
        self.assertEqual(error["title"], "Stage Move Error")
        self.assertIn("15.000", error["message"])
        self.assertIn("outside limits", error["message"])
        self.assertIn("-14.0", error["message"])
        self.assertIn("14.0", error["message"])

    def test_y_limit_exceeded_shows_error(self):
        """Test that exceeding Y limit shows specific error."""
        self.ui.x_value = 0.0
        self.ui.y_value = -15.0  # Beyond -14.0 limit
        self.ui.goto_stage_position()

        # Should show error
        self.assertIsNotNone(self.ui.last_error_message)
        error = self.ui.last_error_message
        self.assertEqual(error["title"], "Stage Move Error")
        self.assertIn("-15.000", error["message"])
        self.assertIn("outside limits", error["message"])

    def test_exact_limits_work(self):
        """Test that exact limit values are accepted."""
        # Test positive limit
        self.ui.x_value = 14.0
        self.ui.y_value = 14.0
        self.ui.goto_stage_position()
        self.assertIsNone(self.ui.last_error_message)

        # Test negative limit
        self.ui.x_value = -14.0
        self.ui.y_value = -14.0
        self.ui.goto_stage_position()
        self.assertIsNone(self.ui.last_error_message)

    def test_hardware_not_initialized_shows_error(self):
        """Test that uninitialized hardware shows error."""
        self.ui.hardware_initialized = False
        self.ui.x_value = 5.0
        self.ui.y_value = 5.0
        self.ui.goto_stage_position()

        # Should show error
        self.assertIsNotNone(self.ui.last_error_message)
        error = self.ui.last_error_message
        self.assertEqual(error["title"], "Stage Not Ready")
        self.assertIn("Stage not initialized", error["message"])


if __name__ == "__main__":
    unittest.main()
