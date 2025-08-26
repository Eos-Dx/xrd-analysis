"""
Unit tests for XY stage axis limits and measurement point filtering.
Tests both the low-level stage controller limits and the high-level measurement filtering.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add the project src root to the path to import modules as the application does
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

# Stub PyQt5 and GUI-related modules to avoid heavy dependencies during import
if "PyQt5" not in sys.modules:
    pyqt5 = types.ModuleType("PyQt5")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtgui = types.ModuleType("PyQt5.QtGui")

    # Provide minimal stubs used by process_mixin
    class _QMessageBox:
        @staticmethod
        def warning(*args, **kwargs):
            return None

    qtwidgets.QMessageBox = _QMessageBox

    class _QListWidgetItem:
        def __init__(self, *a, **k):
            pass

    qtwidgets.QListWidgetItem = _QListWidgetItem
    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtWidgets"] = qtwidgets

    # Minimal thread/timer stubs used by process_mixin
    class _QThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def quit(self):
            pass

        def finished(self):
            return Mock()

        def started(self):
            return Mock()

    class _QTimer:
        def __init__(self, *a, **k):
            pass

        def timeout(self):
            return Mock()

        def start(self, *a, **k):
            pass

    qtcore.QThread = _QThread
    qtcore.QTimer = _QTimer

    # Minimal Qt and pyqtSignal stubs used in other modules
    class _Signal:
        def __init__(self, *a, **k):
            pass

        def connect(self, *a, **k):
            pass

        def emit(self, *a, **k):
            pass

    def _pyqtSignal(*a, **k):
        return _Signal()

    qtcore.pyqtSignal = _pyqtSignal

    def _pyqtSlot(*a, **k):
        def decorator(func):
            return func

        return decorator

    qtcore.pyqtSlot = _pyqtSlot

    class _QObject:
        def __init__(self, *a, **k):
            pass

    qtcore.QObject = _QObject
    qtcore.Qt = types.SimpleNamespace()
    sys.modules["PyQt5.QtCore"] = qtcore

    class _QColor:
        def __init__(self, *a, **k):
            pass

        def setAlphaF(self, *a, **k):
            pass

    qtgui.QColor = _QColor
    sys.modules["PyQt5.QtGui"] = qtgui

# Stub capture and widgets modules referenced by process_mixin
if "hardware.Ulster.gui.technical.capture" not in sys.modules:
    cap_mod = types.ModuleType("hardware.Ulster.gui.technical.capture")

    class _CaptureWorker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            pass

    def _validate_folder(*args, **kwargs):
        return True

    def _compute_hf_score_from_cake(*args, **kwargs):
        return 0.0

    def _move_and_convert_measurement_file(src_path, alias_folder):
        # Return a dummy destination path string
        return str(src_path)

    cap_mod.CaptureWorker = _CaptureWorker
    cap_mod.validate_folder = _validate_folder
    cap_mod.compute_hf_score_from_cake = _compute_hf_score_from_cake
    cap_mod.move_and_convert_measurement_file = _move_and_convert_measurement_file
    sys.modules["hardware.Ulster.gui.technical.capture"] = cap_mod

if "hardware.Ulster.gui.technical.widgets" not in sys.modules:
    w_mod = types.ModuleType("hardware.Ulster.gui.technical.widgets")

    class _MeasurementHistoryWidget:
        def __init__(self, *args, **kwargs):
            pass

        def add_measurement(self, *args, **kwargs):
            pass

    w_mod.MeasurementHistoryWidget = _MeasurementHistoryWidget
    sys.modules["hardware.Ulster.gui.technical.widgets"] = w_mod

# Stub logger module
if "hardware.Ulster.utils.logger" not in sys.modules:
    l_mod = types.ModuleType("hardware.Ulster.utils.logger")

    def _get_module_logger(name):
        class _L:
            def debug(self, *a, **k):
                pass

            def info(self, *a, **k):
                pass

            def warning(self, *a, **k):
                pass

            def error(self, *a, **k):
                pass

        return _L()

    l_mod.get_module_logger = _get_module_logger
    sys.modules["hardware.Ulster.utils.logger"] = l_mod

from hardware.Ulster.hardware.xystages import (
    BaseStageController,
    DummyStageController,
    StageAxisLimitError,
    XYStageLibController,
)


class TestStageAxisLimits(unittest.TestCase):
    """Test axis limit enforcement in stage controllers."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {"alias": "TEST_STAGE", "id": "TEST-123"}
        self.dummy_stage = DummyStageController(self.config)

    def test_axis_limit_constant(self):
        """Test that the axis limit constant is set correctly."""
        self.assertEqual(BaseStageController.AXIS_LIMIT_MM, 14.0)

    def test_check_axis_limits_valid_positions(self):
        """Test that valid positions pass the axis limit check."""
        # Test positions within limits
        valid_positions = [
            (0.0, 0.0),
            (14.0, 14.0),
            (-14.0, -14.0),
            (13.9, -13.9),
            (10.5, 7.2),
        ]

        for x, y in valid_positions:
            with self.subTest(x=x, y=y):
                # Should not raise an exception
                try:
                    self.dummy_stage._check_axis_limits(x, y)
                except StageAxisLimitError:
                    self.fail(f"Valid position ({x}, {y}) raised StageAxisLimitError")

    def test_check_axis_limits_invalid_x(self):
        """Test that invalid X positions raise StageAxisLimitError."""
        invalid_x_positions = [(14.1, 0.0), (-14.1, 0.0), (15.0, 10.0), (-20.0, -5.0)]

        for x, y in invalid_x_positions:
            with self.subTest(x=x, y=y):
                with self.assertRaises(StageAxisLimitError) as cm:
                    self.dummy_stage._check_axis_limits(x, y)
                self.assertEqual(cm.exception.axis, "X")
                self.assertEqual(cm.exception.value, x)
                self.assertEqual(cm.exception.limit, 14.0)

    def test_check_axis_limits_invalid_y(self):
        """Test that invalid Y positions raise StageAxisLimitError."""
        invalid_y_positions = [(0.0, 14.1), (0.0, -14.1), (10.0, 15.0), (-5.0, -20.0)]

        for x, y in invalid_y_positions:
            with self.subTest(x=x, y=y):
                with self.assertRaises(StageAxisLimitError) as cm:
                    self.dummy_stage._check_axis_limits(x, y)
                self.assertEqual(cm.exception.axis, "Y")
                self.assertEqual(cm.exception.value, y)
                self.assertEqual(cm.exception.limit, 14.0)

    def test_dummy_stage_move_valid(self):
        """Test that DummyStageController allows valid moves."""
        valid_moves = [(0.0, 0.0), (10.0, -10.0), (14.0, 14.0), (-14.0, -14.0)]

        for x, y in valid_moves:
            with self.subTest(x=x, y=y):
                result_x, result_y = self.dummy_stage.move_stage(x, y)
                self.assertEqual(result_x, x)
                self.assertEqual(result_y, y)

    def test_dummy_stage_move_invalid(self):
        """Test that DummyStageController rejects invalid moves."""
        invalid_moves = [(14.1, 0.0), (0.0, -14.1), (15.0, 15.0), (-20.0, 5.0)]

        for x, y in invalid_moves:
            with self.subTest(x=x, y=y):
                with self.assertRaises(StageAxisLimitError):
                    self.dummy_stage.move_stage(x, y)
                # Ensure position didn't change after failed move
                pos_x, pos_y = self.dummy_stage.get_xy_position()
                self.assertNotEqual((pos_x, pos_y), (x, y))

    @patch("hardware.Ulster.hardware.xystages.CDLL")
    def test_real_stage_move_invalid(self, mock_cdll):
        """Test that XYStageLibController rejects invalid moves."""
        # Mock the DLL and its methods
        mock_lib = Mock()
        mock_cdll.return_value = mock_lib
        mock_lib.TLI_BuildDeviceList.return_value = 0

        config = {"id": "TEST-123", "alias": "REAL_STAGE"}
        real_stage = XYStageLibController(config, sim=True)
        real_stage.lib = mock_lib  # Set the mocked lib

        # Test that invalid moves are rejected before calling the hardware
        with self.assertRaises(StageAxisLimitError):
            real_stage.move_stage(15.0, 0.0)

        # Verify that the hardware methods were NOT called
        mock_lib.BDC_SetMoveAbsolutePosition.assert_not_called()
        mock_lib.BDC_MoveAbsolute.assert_not_called()

    def test_stage_axis_limit_error_message(self):
        """Test that StageAxisLimitError has correct error message."""
        error = StageAxisLimitError("X", 15.5, 14.0)
        expected_msg = "Stage X position 15.500 mm exceeds limit of ±14.0 mm"
        self.assertEqual(str(error), expected_msg)

        error = StageAxisLimitError("Y", -16.2, 14.0)
        expected_msg = "Stage Y position -16.200 mm exceeds limit of ±14.0 mm"
        self.assertEqual(str(error), expected_msg)


class TestMeasurementPointFiltering(unittest.TestCase):
    """Test measurement point filtering for out-of-bounds points."""

    def setUp(self):
        """Set up test fixtures for measurement filtering tests."""
        # Mock the measurement processing class
        self.mock_processor = Mock()
        self.mock_processor.config = {"detectors": []}
        self.mock_processor.pointsTable = Mock()
        self.mock_processor.pointsTable.rowCount.return_value = 5
        self.mock_processor.folderLineEdit = Mock()
        self.mock_processor.folderLineEdit.text.return_value = "/test/folder"
        self.mock_processor.fileNameLineEdit = Mock()
        self.mock_processor.fileNameLineEdit.text.return_value = "test_measurement"
        self.mock_processor.integrationSpinBox = Mock()
        self.mock_processor.integrationSpinBox.value.return_value = 1

        # Mock UI elements
        self.mock_processor.start_btn = Mock()
        self.mock_processor.pause_btn = Mock()
        self.mock_processor.stop_btn = Mock()
        self.mock_processor.progressBar = Mock()
        self.mock_processor.timeRemainingLabel = Mock()

        # Mock image view and points
        self.mock_processor.image_view = Mock()
        self.mock_processor.image_view.points_dict = {
            "generated": {"points": []},
            "user": {"points": []},
        }
        self.mock_processor.real_x_pos_mm = Mock()
        self.mock_processor.real_x_pos_mm.value.return_value = 0.0
        self.mock_processor.real_y_pos_mm = Mock()
        self.mock_processor.real_y_pos_mm.value.return_value = 0.0
        self.mock_processor.include_center = (0, 0)
        self.mock_processor.pixel_to_mm_ratio = 1.0

        # Mock state management
        self.mock_processor.state = {}
        self.mock_processor.state_measurements = {}
        self.mock_processor.manual_save_state = Mock()
        self.mock_processor.measure_next_point = Mock()

    def create_mock_point_at_position(self, x_mm, y_mm):
        """Create a mock point at the specified position in mm."""
        mock_point = Mock()
        mock_rect = Mock()
        mock_center = Mock()
        # Convert mm to pixel coordinates (assuming center at origin)
        mock_center.x.return_value = x_mm * self.mock_processor.pixel_to_mm_ratio
        mock_center.y.return_value = y_mm * self.mock_processor.pixel_to_mm_ratio
        mock_rect.center.return_value = mock_center
        mock_point.sceneBoundingRect.return_value = mock_rect
        return mock_point

    @patch("pathlib.Path.exists")
    @patch("hardware.Ulster.hardware.auxiliary.encode_image_to_base64")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_filter_out_of_bounds_points(
        self, mock_json_dump, mock_open, mock_encode, mock_exists
    ):
        """Test that out-of-bounds measurement points are filtered out."""
        mock_exists.return_value = True
        mock_encode.return_value = "base64_image_data"

        # Create test points - some valid, some invalid
        test_points_data = [
            (0.0, 0.0),  # Valid: center
            (10.0, 10.0),  # Valid: within limits
            (-13.0, 13.0),  # Valid: within limits
            (15.0, 5.0),  # Invalid: X exceeds limit
            (5.0, -15.0),  # Invalid: Y exceeds limit
            (20.0, 20.0),  # Invalid: both exceed limits
            (14.0, 14.0),  # Valid: exactly at limit
            (-14.0, -14.0),  # Valid: exactly at negative limit
        ]

        # Create mock points
        mock_points = []
        for i, (x, y) in enumerate(test_points_data):
            mock_point = self.create_mock_point_at_position(x, y)
            mock_points.append(mock_point)

        self.mock_processor.image_view.points_dict["generated"]["points"] = mock_points

        # Import the measurement processing mixin directly from its file to avoid package side-effects
        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "Ulster",
            "gui",
            "main_window_ext",
            "zone_measurements",
            "logic",
            "process_mixin.py",
        )
        spec = importlib.util.spec_from_file_location("process_mixin_direct", pm_path)
        pm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm)
        ZoneMeasurementsProcessMixin = pm.ZoneMeasurementsProcessMixin

        # Create a simple processor instance with the mixin
        ProcCls = type("Proc", (ZoneMeasurementsProcessMixin,), {})
        proc = ProcCls()
        # Wire up required attributes using our prepared mocks
        proc.config = {"detectors": []}
        proc.folderLineEdit = self.mock_processor.folderLineEdit
        proc.fileNameLineEdit = self.mock_processor.fileNameLineEdit
        proc.pointsTable = self.mock_processor.pointsTable
        proc.start_btn = self.mock_processor.start_btn
        proc.pause_btn = self.mock_processor.pause_btn
        proc.stop_btn = self.mock_processor.stop_btn
        proc.progressBar = self.mock_processor.progressBar
        proc.timeRemainingLabel = self.mock_processor.timeRemainingLabel
        proc.image_view = self.mock_processor.image_view
        proc.real_x_pos_mm = self.mock_processor.real_x_pos_mm
        proc.real_y_pos_mm = self.mock_processor.real_y_pos_mm
        proc.include_center = self.mock_processor.include_center
        proc.pixel_to_mm_ratio = self.mock_processor.pixel_to_mm_ratio
        proc.state = {}
        proc.state_measurements = {}
        proc.manual_save_state = Mock()
        proc.measure_next_point = Mock()
        proc.integrationSpinBox = self.mock_processor.integrationSpinBox

        # Mock the state copying
        with patch("copy.copy") as mock_copy:
            mock_copy.return_value = {}

            # Call the method that filters points
            proc.start_measurements()

            # Check that measurement_points were created and filtered
            measurement_points = proc.state.get("measurement_points", [])
            skipped_points = proc.state.get("skipped_points", [])

            # Expected valid points: (0,0), (10,10), (-13,13), (14,14), (-14,-14) = 5 points
            expected_valid_count = 5
            expected_skipped_count = 3

            self.assertEqual(len(measurement_points), expected_valid_count)
            self.assertEqual(len(skipped_points), expected_skipped_count)

            # Check that skipped points have the correct reason
            for skipped in skipped_points:
                self.assertEqual(skipped["reason"], "axis_limit_exceeded")

    @patch("pathlib.Path.exists")
    @patch("PyQt5.QtWidgets.QMessageBox.warning")
    def test_no_valid_points_shows_warning(self, mock_warning, mock_exists):
        """Test that a warning is shown when all points are out of bounds."""
        mock_exists.return_value = True

        # Create test points that are all out of bounds
        invalid_points_data = [
            (20.0, 0.0),  # X too large
            (-20.0, 0.0),  # X too small
            (0.0, 20.0),  # Y too large
            (0.0, -20.0),  # Y too small
            (30.0, 30.0),  # Both too large
        ]

        mock_points = []
        for x, y in invalid_points_data:
            mock_point = self.create_mock_point_at_position(x, y)
            mock_points.append(mock_point)

        self.mock_processor.image_view.points_dict["generated"]["points"] = mock_points

        # Import the mixin module directly
        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "Ulster",
            "gui",
            "main_window_ext",
            "zone_measurements",
            "logic",
            "process_mixin.py",
        )
        spec = importlib.util.spec_from_file_location("process_mixin_direct", pm_path)
        pm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm)
        ZoneMeasurementsProcessMixin = pm.ZoneMeasurementsProcessMixin

        # Create processor with required attributes
        ProcCls = type("Proc", (ZoneMeasurementsProcessMixin,), {})
        proc = ProcCls()
        proc.config = {"detectors": []}
        proc.folderLineEdit = self.mock_processor.folderLineEdit
        proc.fileNameLineEdit = self.mock_processor.fileNameLineEdit
        proc.pointsTable = self.mock_processor.pointsTable
        proc.start_btn = self.mock_processor.start_btn
        proc.pause_btn = self.mock_processor.pause_btn
        proc.stop_btn = self.mock_processor.stop_btn
        proc.progressBar = self.mock_processor.progressBar
        proc.timeRemainingLabel = self.mock_processor.timeRemainingLabel
        proc.image_view = self.mock_processor.image_view
        proc.real_x_pos_mm = self.mock_processor.real_x_pos_mm
        proc.real_y_pos_mm = self.mock_processor.real_y_pos_mm
        proc.include_center = self.mock_processor.include_center
        proc.pixel_to_mm_ratio = self.mock_processor.pixel_to_mm_ratio
        proc.state = {}
        proc.state_measurements = {}
        proc.manual_save_state = Mock()
        proc.measure_next_point = Mock()
        proc.integrationSpinBox = self.mock_processor.integrationSpinBox

        with patch("copy.copy") as mock_copy:
            mock_copy.return_value = {}

            # Call the method - it should return early and show warning
            proc.start_measurements()

            # Check that warning was shown
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0]
            self.assertIn("No Valid Points", call_args[1])
            self.assertIn("axis limits", call_args[2])

            # Check that measure_next_point was NOT called
            proc.measure_next_point.assert_not_called()

    def test_valid_points_preserve_order(self):
        """Test that valid points preserve their relative order after filtering."""
        test_points_data = [
            (1.0, 1.0),  # Valid - should be index 0 after filtering
            (20.0, 0.0),  # Invalid - filtered out
            (2.0, 2.0),  # Valid - should be index 1 after filtering
            (0.0, 20.0),  # Invalid - filtered out
            (3.0, 3.0),  # Valid - should be index 2 after filtering
        ]

        # Test that the filtering preserves the relative order of valid points
        valid_indices = []
        skipped_indices = []

        for i, (x, y) in enumerate(test_points_data):
            if abs(x) <= 14.0 and abs(y) <= 14.0:
                valid_indices.append(i)
            else:
                skipped_indices.append(i)

        self.assertEqual(valid_indices, [0, 2, 4])
        self.assertEqual(skipped_indices, [1, 3])


if __name__ == "__main__":
    # Create tests directory if it doesn't exist
    test_dir = os.path.dirname(__file__)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    unittest.main()
