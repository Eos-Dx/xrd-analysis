"""
Tests for technical meta generation, including PONI_LAB, PONI_LAB_PATH, and PONI_LAB_VALUES.
This test stubs PyQt5 to avoid heavy GUI dependencies and simulates user selections.
"""

import json
import os
import sys
import tempfile
import types
import unittest

# ---- Stub minimal PyQt5 symbols before importing the module under test ----
if "PyQt5" not in sys.modules:
    pyqt5 = types.ModuleType("PyQt5")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    qtcore = types.ModuleType("PyQt5.QtCore")

    # QtCore stubs
    class _QThread:
        def __init__(self, *a, **k):
            pass

    class _QTimer:
        def __init__(self, *a, **k):
            pass

    class _Qt:
        UserRole = 32

    qtcore.QThread = _QThread
    qtcore.QTimer = _QTimer
    qtcore.Qt = _Qt()

    # Add pyqtSignal/pyqtSlot/QObject stubs used by other modules
    class _Signal:
        def __init__(self, *a, **k):
            pass

        def connect(self, *a, **k):
            pass

        def emit(self, *a, **k):
            pass

    def _pyqtSignal(*a, **k):
        return _Signal()

    def _pyqtSlot(*a, **k):
        def decorator(func):
            return func

        return decorator

    class _QObject:
        def __init__(self, *a, **k):
            pass

    qtcore.pyqtSignal = _pyqtSignal
    qtcore.pyqtSlot = _pyqtSlot
    qtcore.QObject = _QObject

    # QtWidgets stubs (only what's needed for import and QMessageBox use)
    class _QDialog:
        Accepted = 1
        Rejected = 0

        def exec_(self):
            return _QDialog.Accepted

    class _QMessageBox:
        Yes = 1
        No = 0

        @staticmethod
        def warning(*args, **kwargs):
            return None

        @staticmethod
        def question(*args, **kwargs):
            # Default to Yes for overwrite prompts
            return _QMessageBox.Yes

        @staticmethod
        def critical(*args, **kwargs):
            return None

        @staticmethod
        def information(*args, **kwargs):
            return None

    # Define placeholder widget classes used in imports
    class _Placeholder:
        def __init__(self, *a, **k):
            pass

    qtwidgets.QDialog = _QDialog
    qtwidgets.QMessageBox = _QMessageBox
    qtwidgets.QComboBox = _Placeholder
    qtwidgets.QDockWidget = _Placeholder
    qtwidgets.QTabWidget = _Placeholder
    qtwidgets.QDoubleSpinBox = _Placeholder
    qtwidgets.QFileDialog = _Placeholder
    qtwidgets.QFormLayout = _Placeholder
    qtwidgets.QHBoxLayout = _Placeholder
    qtwidgets.QLabel = _Placeholder
    qtwidgets.QLineEdit = _Placeholder
    qtwidgets.QListWidget = _Placeholder
    qtwidgets.QListWidgetItem = _Placeholder
    qtwidgets.QTableWidget = _Placeholder
    qtwidgets.QTableWidgetItem = _Placeholder
    qtwidgets.QPushButton = _Placeholder
    qtwidgets.QScrollArea = _Placeholder
    qtwidgets.QSpinBox = _Placeholder
    qtwidgets.QVBoxLayout = _Placeholder
    qtwidgets.QWidget = _Placeholder

    # Fallback for any other widget types imported elsewhere
    def _widgets_getattr(name):
        return _Placeholder

    qtwidgets.__getattr__ = _widgets_getattr

    # QtGui stubs
    qtg = types.ModuleType("PyQt5.QtGui")

    class _QColor:
        def __init__(self, *a, **k):
            pass

        def setAlphaF(self, *a, **k):
            pass

    qtg.QColor = _QColor

    # Fallback for any other QtGui symbols
    def _qtg_getattr(name):
        return _Placeholder

    qtg.__getattr__ = _qtg_getattr

    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtCore"] = qtcore
    sys.modules["PyQt5.QtWidgets"] = qtwidgets
    sys.modules["PyQt5.QtGui"] = qtg


# Add src root to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Stub capture and widgets modules referenced by process_mixin to avoid Qt/matplotlib imports
if "hardware.Ulster.gui.technical.capture" not in sys.modules:
    cap_mod = types.ModuleType("hardware.Ulster.gui.technical.capture")

    class _CaptureWorker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            pass

    def _validate_folder(path):
        return path

    def _compute_hf_score_from_cake(*args, **kwargs):
        return 0.0

    def _move_and_convert_measurement_file(src_path, alias_folder):
        return str(src_path)

    def _show_measurement_window(*args, **kwargs):
        return None

    cap_mod.CaptureWorker = _CaptureWorker
    cap_mod.validate_folder = _validate_folder
    cap_mod.show_measurement_window = _show_measurement_window
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


class TestTechnicalMetaGeneration(unittest.TestCase):
    def setUp(self):
        # Import module under test after stubbing PyQt5
        from hardware.Ulster.gui.main_window_ext import technical_measurements as tm

        self.tm = tm

        # Monkeypatch validate_folder to be identity
        self.tm.validate_folder = lambda p: p

        # Stub dialog to auto-return provided PONI files
        class _DummyPoniDialog:
            def __init__(self, aliases, current_poni_files=None, parent=None):
                self.aliases = aliases
                self._ponis = {}

            def exec_(self):
                return 1  # Accepted

            def get_poni_files(self):
                # Provide a mapping prepared by the test case
                return self._ponis

        self.DummyPoniDialog = _DummyPoniDialog
        self.tm.PoniFileSelectionDialog = self.DummyPoniDialog

    def _make_stub_aux_table(self, rows_data):
        """Create a stub auxTable with:
        rows_data: list of dicts with keys: {file_path, type_text, alias_text}
        """
        tm = self.tm

        class _StubIndex:
            def __init__(self, row):
                self._row = row

            def row(self):
                return self._row

        class _StubSelectionModel:
            def __init__(self, count):
                self._count = count

            def selectedRows(self):
                return [_StubIndex(i) for i in range(self._count)]

        class _StubItem:
            def __init__(self, path):
                self._path = path

            def data(self, _role):
                return self._path

        def _make_cb(text):
            cb = tm.QComboBox()
            # Attach a currentText method dynamically
            cb.currentText = lambda: text
            return cb

        class _StubAuxTable:
            def __init__(self, rows):
                self._rows = rows
                self._sel = _StubSelectionModel(len(rows))

            def selectionModel(self):
                return self._sel

            def item(self, row, col):
                # Only (row,0) is used
                return _StubItem(self._rows[row]["file_path"]) if col == 0 else None

            def cellWidget(self, row, col):
                if col == 1:
                    return _make_cb(self._rows[row]["type_text"])
                if col == 2:
                    return _make_cb(self._rows[row]["alias_text"])
                return None

        return _StubAuxTable(rows_data)

    def test_generate_technical_meta_with_poni_sections(self):
        tm = self.tm

        # Create temp directory and files
        with tempfile.TemporaryDirectory() as tmpdir:
            meas_path = os.path.join(tmpdir, "SAXS_meas.npy")
            with open(meas_path, "wb") as f:
                f.write(b"dummy")

            # Prepare PONI files
            poni_path = os.path.join(tmpdir, "saxs.poni")
            poni_text = "poni_version: 2.1\nDetector: SAXS\n"
            with open(poni_path, "w", encoding="utf-8") as f:
                f.write(poni_text)

            # Prepare the mixin instance without full Qt setup
            obj = object.__new__(tm.TechnicalMeasurementsMixin)

            # Required UI-like attributes
            class _LE:
                def __init__(self, txt):
                    self._txt = txt

                def text(self):
                    return self._txt

            obj.auxNameLE = _LE("techmeta")
            obj.folderLE = _LE(tmpdir)
            obj.NO_SELECTION_LABEL = "— Select —"

            # Prepare auxTable with one selected row, type AgBH for alias SAXS
            obj.auxTable = self._make_stub_aux_table(
                [{"file_path": meas_path, "type_text": "AgBH", "alias_text": "SAXS"}]
            )

            # Configure dummy dialog return values
            dummy_dialog = self.DummyPoniDialog(
                aliases=["SAXS"], current_poni_files=None, parent=obj
            )
            dummy_dialog._ponis = {"SAXS": poni_path}
            tm.PoniFileSelectionDialog = (
                lambda aliases, current_poni_files=None, parent=None: dummy_dialog
            )

            # Run
            obj.generate_technical_meta()

            # Verify output JSON
            out_path = os.path.join(tmpdir, "technical_meta_techmeta.json")
            self.assertTrue(os.path.exists(out_path), "Meta file was not created")
            with open(out_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Check measurement mapping
            self.assertIn("AgBH", meta)
            self.assertIn("SAXS", meta["AgBH"])
            self.assertEqual(meta["AgBH"]["SAXS"], os.path.basename(meas_path))

            # Check PONI sections
            self.assertIn("PONI_LAB", meta)
            self.assertEqual(meta["PONI_LAB"].get("SAXS"), os.path.basename(poni_path))

            self.assertIn("PONI_LAB_PATH", meta)
            self.assertEqual(meta["PONI_LAB_PATH"].get("SAXS"), poni_path)

            self.assertIn("PONI_LAB_VALUES", meta)
            self.assertEqual(meta["PONI_LAB_VALUES"].get("SAXS"), poni_text)
