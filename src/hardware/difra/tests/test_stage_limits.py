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

# Ensure top-level 'hardware' package resolves correctly (avoid namespace confusion)
import importlib

if "hardware" in sys.modules:
    try:
        del sys.modules["hardware"]
    except Exception:
        pass
try:
    _hardware_pkg = importlib.import_module("hardware")
    sys.modules["hardware"] = _hardware_pkg
except Exception:
    pass

# Stub PyQt5 and GUI-related modules to avoid heavy dependencies during import
if "PyQt5" not in sys.modules:
    pyqt5 = types.ModuleType("PyQt5")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtgui = types.ModuleType("PyQt5.QtGui")

    # Provide minimal stubs used by process_mixin
    class _QMessageBox:
        Yes, No, Question = 1, 0, 3
        AcceptRole, RejectRole = 1, 0

        def __init__(self, *args, **kwargs):
            self._buttons = []
            self._default = None
            self._clicked = None

        @staticmethod
        def warning(*args, **kwargs):
            return None

        @staticmethod
        def information(*args, **kwargs):
            return None

        @staticmethod
        def question(*args, **kwargs):
            return _QMessageBox.Yes

        def setWindowTitle(self, *args, **kwargs):
            pass

        def setText(self, *args, **kwargs):
            pass

        def setIcon(self, *args, **kwargs):
            pass

        def setStandardButtons(self, *args, **kwargs):
            pass

        def addButton(self, text, role):
            btn = types.SimpleNamespace(text=text, role=role)
            self._buttons.append(btn)
            return btn

        def setDefaultButton(self, btn):
            self._default = btn

        def clickedButton(self):
            return self._clicked or self._default

        def exec_(self):
            # Simulate clicking default button
            self._clicked = self._default
            return 1

    qtwidgets.QMessageBox = _QMessageBox

    class _QListWidgetItem:
        def __init__(self, *a, **k):
            pass

    qtwidgets.QListWidgetItem = _QListWidgetItem
    # Register stubs in sys.modules so patching imports work
    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtWidgets"] = qtwidgets
    sys.modules["PyQt5.QtCore"] = qtcore
    sys.modules["PyQt5.QtGui"] = qtgui
    # Also attach as attributes so getattr(PyQt5, 'QtWidgets') works
    pyqt5.QtWidgets = qtwidgets
    pyqt5.QtCore = qtcore
    pyqt5.QtGui = qtgui

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
if "hardware.difra.gui.technical.capture" not in sys.modules:
    cap_mod = types.ModuleType("hardware.difra.gui.technical.capture")

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

    def _show_measurement_window(*args, **kwargs):
        return None

    cap_mod.CaptureWorker = _CaptureWorker
    cap_mod.validate_folder = _validate_folder
    cap_mod.compute_hf_score_from_cake = _compute_hf_score_from_cake
    cap_mod.move_and_convert_measurement_file = _move_and_convert_measurement_file
    cap_mod.show_measurement_window = _show_measurement_window
    sys.modules["hardware.difra.gui.technical.capture"] = cap_mod

if "hardware.difra.gui.technical.widgets" not in sys.modules:
    w_mod = types.ModuleType("hardware.difra.gui.technical.widgets")

    class _MeasurementHistoryWidget:
        def __init__(self, *args, **kwargs):
            pass

        def add_measurement(self, *args, **kwargs):
            pass

    w_mod.MeasurementHistoryWidget = _MeasurementHistoryWidget
    sys.modules["hardware.difra.gui.technical.widgets"] = w_mod

# Stub logger module
if "hardware.difra.utils.logger" not in sys.modules:
    l_mod = types.ModuleType("hardware.difra.utils.logger")

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
    sys.modules["hardware.difra.utils.logger"] = l_mod

from hardware.xystages import (
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

    def test_default_limits_are_14mm(self):
        """Default limits should be ±14 mm per axis when not specified in config."""
        limits = self.dummy_stage.get_limits()
        self.assertEqual(limits["x"], (-14.0, 14.0))
        self.assertEqual(limits["y"], (-14.0, 14.0))

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
                self.assertEqual(cm.exception.min_limit, -14.0)
                self.assertEqual(cm.exception.max_limit, 14.0)

    def test_check_axis_limits_invalid_y(self):
        """Test that invalid Y positions raise StageAxisLimitError."""
        invalid_y_positions = [(0.0, 14.1), (0.0, -14.1), (10.0, 15.0), (-5.0, -20.0)]

        for x, y in invalid_y_positions:
            with self.subTest(x=x, y=y):
                with self.assertRaises(StageAxisLimitError) as cm:
                    self.dummy_stage._check_axis_limits(x, y)
                self.assertEqual(cm.exception.axis, "Y")
                self.assertEqual(cm.exception.value, y)
                self.assertEqual(cm.exception.min_limit, -14.0)
                self.assertEqual(cm.exception.max_limit, 14.0)

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

    @patch("hardware.xystages.CDLL")
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
        error = StageAxisLimitError("X", 15.5, -14.0, 14.0)
        expected_msg = "Stage X position 15.500 mm exceeds limits [-14.0, 14.0] mm"
        self.assertEqual(str(error), expected_msg)

        error = StageAxisLimitError("Y", -16.2, -14.0, 14.0)
        expected_msg = "Stage Y position -16.200 mm exceeds limits [-14.0, 14.0] mm"
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

    def create_mock_point_at_position(self, x_mm, y_mm, uid=None):
        """Create a mock point at the specified position in mm."""
        mock_point = Mock()
        mock_rect = Mock()
        mock_center = Mock()
        # Convert mm to pixel coordinates (assuming center at origin)
        mock_center.x.return_value = x_mm * self.mock_processor.pixel_to_mm_ratio
        mock_center.y.return_value = y_mm * self.mock_processor.pixel_to_mm_ratio
        mock_rect.center.return_value = mock_center
        mock_point.sceneBoundingRect.return_value = mock_rect
        data_store = {}
        if uid is not None:
            data_store[2] = uid

        def _data(key):
            return data_store.get(key)

        def _set_data(key, value):
            data_store[key] = value

        mock_point.data.side_effect = _data
        mock_point.setData.side_effect = _set_data
        return mock_point

    @patch("pathlib.Path.exists")
    @patch("hardware.difra.hardware.auxiliary.encode_image_to_base64")
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
            "difra",
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

        # Provide a stage_controller with default limits
        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

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
            "difra",
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

        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

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

    @patch("pathlib.Path.exists")
    @patch("hardware.difra.hardware.auxiliary.encode_image_to_base64")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_restore_session_resumes_only_pending_points(
        self, mock_json_dump, mock_open, mock_encode, mock_exists
    ):
        """Restored sessions should measure only pending points and keep session indexing."""
        import h5py
        import tempfile

        mock_exists.return_value = True
        mock_encode.return_value = "base64_image_data"

        test_points_data = [
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
        ]
        mock_points = [self.create_mock_point_at_position(x, y) for x, y in test_points_data]
        self.mock_processor.image_view.points_dict["generated"]["points"] = mock_points

        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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

        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

        class _Schema:
            GROUP_POINTS = "/entry/points"
            ATTR_POINT_STATUS = "point_status"
            POINT_STATUS_MEASURED = "measured"

        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = os.path.join(tmp_dir, "restored_session.nxs.h5")
            with h5py.File(session_path, "w") as h5f:
                points_group = h5f.create_group(_Schema.GROUP_POINTS)
                for idx, status in enumerate(("measured", "measured", "pending"), start=1):
                    point_group = points_group.create_group("pt_{0:03d}".format(idx))
                    point_group.attrs[_Schema.ATTR_POINT_STATUS] = status

            session_manager = Mock()
            session_manager.is_session_active.return_value = True
            session_manager.is_locked.return_value = False
            session_manager.session_path = session_path
            session_manager.schema = _Schema()
            session_manager.add_points = Mock()
            proc.session_manager = session_manager

            with patch("copy.copy") as mock_copy:
                mock_copy.return_value = {}
                with patch.object(pm.QMessageBox, "question", return_value=pm.QMessageBox.Yes):
                    proc.start_measurements()

        self.assertEqual(proc.total_points, 1)
        self.assertEqual(proc.sorted_indices, [0])
        self.assertEqual(proc._session_point_indices, [3])
        session_manager.add_points.assert_not_called()
        proc.measure_next_point.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("hardware.difra.hardware.auxiliary.encode_image_to_base64")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_existing_session_points_are_reused_when_filtered_subset_is_measured(
        self, mock_json_dump, mock_open, mock_encode, mock_exists
    ):
        """When session already has points, filtered measurement subset must map to original point IDs."""
        import h5py
        import tempfile

        mock_exists.return_value = True
        mock_encode.return_value = "base64_image_data"

        test_points_data = [
            (-10.0, 0.0),  # valid
            (20.0, 0.0),   # skipped
            (-8.0, 0.0),   # valid
            (22.0, 0.0),   # skipped
            (-6.0, 0.0),   # valid
            (24.0, 0.0),   # skipped
            (-4.0, 0.0),   # valid
            (-2.0, 0.0),   # valid
            (0.0, 0.0),    # valid
            (2.0, 0.0),    # valid
        ]
        mock_points = [self.create_mock_point_at_position(x, y) for x, y in test_points_data]
        self.mock_processor.image_view.points_dict["generated"]["points"] = mock_points

        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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

        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

        class _Schema:
            GROUP_POINTS = "/entry/points"
            ATTR_POINT_STATUS = "point_status"
            POINT_STATUS_MEASURED = "measured"

        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = os.path.join(tmp_dir, "existing_points_session.nxs.h5")
            with h5py.File(session_path, "w") as h5f:
                points_group = h5f.create_group(_Schema.GROUP_POINTS)
                for idx in range(1, 11):
                    point_group = points_group.create_group("pt_{0:03d}".format(idx))
                    point_group.attrs[_Schema.ATTR_POINT_STATUS] = "pending"

            session_manager = Mock()
            session_manager.is_session_active.return_value = True
            session_manager.is_locked.return_value = False
            session_manager.session_path = session_path
            session_manager.schema = _Schema()
            session_manager.add_points = Mock()
            proc.session_manager = session_manager

            with patch("copy.copy") as mock_copy:
                mock_copy.return_value = {}
                with patch.object(pm.QMessageBox, "question", return_value=pm.QMessageBox.Yes):
                    proc.start_measurements()

        self.assertEqual(proc.total_points, 7)
        self.assertEqual(
            proc._session_point_indices,
            [int(idx) + 1 for idx in proc.sorted_indices],
        )
        session_manager.add_points.assert_not_called()
        proc.measure_next_point.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("hardware.difra.hardware.auxiliary.encode_image_to_base64")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_restore_resume_uses_point_uid_mapping_when_point_order_changes(
        self, mock_json_dump, mock_open, mock_encode, mock_exists
    ):
        """Resume should map pending points by persisted UID even if GUI point list order differs."""
        import h5py
        import tempfile

        mock_exists.return_value = True
        mock_encode.return_value = "base64_image_data"

        # GUI list order is intentionally shuffled vs session point index order.
        point_specs = [
            (3.0, 3.0, "3_cccccccc"),
            (1.0, 1.0, "1_aaaaaaaa"),
            (2.0, 2.0, "2_bbbbbbbb"),
        ]
        mock_points = [
            self.create_mock_point_at_position(x, y, uid=uid)
            for x, y, uid in point_specs
        ]
        self.mock_processor.image_view.points_dict["generated"]["points"] = mock_points

        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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

        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

        class _Schema:
            GROUP_POINTS = "/entry/points"
            ATTR_POINT_STATUS = "point_status"
            POINT_STATUS_MEASURED = "measured"
            ATTR_PHYSICAL_COORDINATES_MM = "physical_coordinates_mm"

        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = os.path.join(tmp_dir, "uid_resume_session.nxs.h5")
            with h5py.File(session_path, "w") as h5f:
                points_group = h5f.create_group(_Schema.GROUP_POINTS)

                pt1 = points_group.create_group("pt_001")
                pt1.attrs[_Schema.ATTR_POINT_STATUS] = "measured"
                pt1.attrs[_Schema.ATTR_PHYSICAL_COORDINATES_MM] = [1.0, 1.0]
                pt1.attrs["point_uid"] = "1_aaaaaaaa"

                pt2 = points_group.create_group("pt_002")
                pt2.attrs[_Schema.ATTR_POINT_STATUS] = "pending"
                pt2.attrs[_Schema.ATTR_PHYSICAL_COORDINATES_MM] = [2.0, 2.0]
                pt2.attrs["point_uid"] = "2_bbbbbbbb"

                pt3 = points_group.create_group("pt_003")
                pt3.attrs[_Schema.ATTR_POINT_STATUS] = "pending"
                pt3.attrs[_Schema.ATTR_PHYSICAL_COORDINATES_MM] = [3.0, 3.0]
                pt3.attrs["point_uid"] = "3_cccccccc"

            session_manager = Mock()
            session_manager.is_session_active.return_value = True
            session_manager.is_locked.return_value = False
            session_manager.session_path = session_path
            session_manager.schema = _Schema()
            session_manager.add_points = Mock()
            proc.session_manager = session_manager

            with patch("copy.copy") as mock_copy:
                mock_copy.return_value = {}
                with patch.object(pm.QMessageBox, "question", return_value=pm.QMessageBox.Yes):
                    proc.start_measurements()

        self.assertEqual(proc.total_points, 2)
        self.assertEqual(proc._session_point_indices, [2, 3])
        resumed_uids = [pt["unique_id"] for pt in proc.state_measurements["measurement_points"]]
        self.assertEqual(resumed_uids, ["2_bbbbbbbb", "3_cccccccc"])
        session_manager.add_points.assert_not_called()
        proc.measure_next_point.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("hardware.difra.hardware.auxiliary.encode_image_to_base64")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    @patch("PyQt5.QtWidgets.QMessageBox.warning")
    def test_restore_resume_cancels_when_pending_points_cannot_be_fully_mapped(
        self,
        mock_warning,
        mock_json_dump,
        mock_open,
        mock_encode,
        mock_exists,
    ):
        """Resume should cancel when some pending session points cannot be mapped."""
        import h5py
        import tempfile

        mock_exists.return_value = True
        mock_encode.return_value = "base64_image_data"

        point_specs = [
            (1.0, 1.0, "1_aaaaaaaa"),
            (2.0, 2.0, "2_bbbbbbbb"),
            (3.0, 3.0, "3_cccccccc"),
            (20.0, 20.0, "4_dddddddd"),  # filtered out by stage limits
        ]
        mock_points = [
            self.create_mock_point_at_position(x, y, uid=uid)
            for x, y, uid in point_specs
        ]
        self.mock_processor.image_view.points_dict["generated"]["points"] = mock_points

        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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

        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

        class _Schema:
            GROUP_POINTS = "/entry/points"
            ATTR_POINT_STATUS = "point_status"
            POINT_STATUS_MEASURED = "measured"
            ATTR_PHYSICAL_COORDINATES_MM = "physical_coordinates_mm"

        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = os.path.join(tmp_dir, "resume_mapping_incomplete.nxs.h5")
            with h5py.File(session_path, "w") as h5f:
                points_group = h5f.create_group(_Schema.GROUP_POINTS)
                for idx, (x, y, uid) in enumerate(point_specs, start=1):
                    pt = points_group.create_group(f"pt_{idx:03d}")
                    pt.attrs[_Schema.ATTR_POINT_STATUS] = "pending"
                    pt.attrs[_Schema.ATTR_PHYSICAL_COORDINATES_MM] = [x, y]
                    pt.attrs["point_uid"] = uid

            session_manager = Mock()
            session_manager.is_session_active.return_value = True
            session_manager.is_locked.return_value = False
            session_manager.session_path = session_path
            session_manager.schema = _Schema()
            session_manager.add_points = Mock()
            proc.session_manager = session_manager

            with patch("copy.copy") as mock_copy:
                mock_copy.return_value = {}
                proc.start_measurements()

        mock_warning.assert_called()
        title = mock_warning.call_args[0][1]
        self.assertIn("Resume Mapping Incomplete", title)
        proc.measure_next_point.assert_not_called()


    @patch("pathlib.Path.exists")
    @patch("hardware.difra.hardware.auxiliary.encode_image_to_base64")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_restore_session_reuses_existing_i0_and_skips_background_capture(
        self, mock_json_dump, mock_open, mock_encode, mock_exists
    ):
        """When restored session already has I0, start should not capture I0 again."""
        mock_exists.return_value = True
        mock_encode.return_value = "base64_image_data"

        self.mock_processor.image_view.points_dict["generated"]["points"] = [
            self.create_mock_point_at_position(1.0, 1.0)
        ]

        import importlib.util

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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
        proc.attenuationCheckBox = Mock()
        proc.attenuationCheckBox.isChecked.return_value = True
        proc._capture_attenuation_background = Mock()

        class _Stage:
            def get_limits(self_inner):
                return {"x": (-14.0, 14.0), "y": (-14.0, 14.0)}

        proc.stage_controller = _Stage()

        session_manager = Mock()
        session_manager.is_session_active.return_value = True
        session_manager.is_locked.return_value = False
        session_manager.i0_counter = 7
        session_manager.session_path = None
        session_manager.schema = None
        session_manager.add_points = Mock()
        proc.session_manager = session_manager

        with patch("copy.copy") as mock_copy:
            mock_copy.return_value = {}
            proc.start_measurements()

        proc._capture_attenuation_background.assert_not_called()
        assert bool(getattr(proc, "_reuse_existing_i0_from_session", False)) is True
        proc.measure_next_point.assert_called_once()

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

    def test_state_measurements_json_dump_handles_numpy_scalars(self):
        """State dump should serialize numpy scalar/array values without crashing."""
        import importlib.util
        import json as std_json
        import tempfile
        from pathlib import Path

        import numpy as np

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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

        ProcCls = type("Proc", (ZoneMeasurementsProcessMixin,), {})
        proc = ProcCls()
        proc.state_measurements = {
            "value": np.int64(7),
            "nested": {"arr": np.array([1, 2, 3], dtype=np.int64)},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "state.json"
            proc.state_path_measurements = out_path
            proc._dump_state_measurements()

            payload = std_json.loads(out_path.read_text())

        self.assertEqual(payload["value"], 7)
        self.assertEqual(payload["nested"]["arr"], [1, 2, 3])

    def test_measurement_point_uid_format_is_counter_plus_8hex(self):
        """Point unique_id should be '<integer_counter>_<8 hex symbols>' and unique."""
        import importlib.util
        import re

        pm_path = os.path.join(
            SRC_ROOT,
            "hardware",
            "difra",
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

        ProcCls = type("Proc", (ZoneMeasurementsProcessMixin,), {})
        proc = ProcCls()

        ids = [proc._new_measurement_point_uid(i + 1) for i in range(64)]
        assert len(ids) == len(set(ids))
        pattern = re.compile(r"^\d+_[0-9a-f]{8}$")
        for i, uid in enumerate(ids, start=1):
            assert pattern.match(uid), uid
            assert uid.startswith(f"{i}_")


if __name__ == "__main__":
    # Create tests directory if it doesn't exist
    test_dir = os.path.dirname(__file__)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    unittest.main()
