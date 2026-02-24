"""GUI integration test for technical workflow button interactions.

This test exercises a full user path:
1. Start GUI harness
2. Login as "sad"
3. Initialize hardware
4. Configure detector distances (17 cm for both detectors)
5. Click AUX measurement 4 times (8 files total for 2 detectors)
6. Assign technical types and set all rows as primary
7. Click Generate H5
8. Confirm lock/bundle flow and validate resulting container schema content
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from hardware.container.v0_2 import schema
from hardware.difra.gui.main_window_ext import technical_measurements
from hardware.difra.gui.main_window_ext.technical import (
    h5_generation_mixin,
    h5_management_mixin,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeOperatorManager:
    def __init__(self, *args, **kwargs):
        pass

    def get_current_operator_id(self):
        return "sad"


class _FakeDistanceDialog:
    def __init__(self, detector_configs, current_distances=None, parent=None):
        self.detector_configs = detector_configs

    def exec_(self):
        return QDialog.Accepted

    def get_distances(self):
        return {
            detector_config.get("id"): 17.0
            for detector_config in self.detector_configs
            if detector_config.get("id")
        }


class _FakePoniSelectionDialog:
    def __init__(self, aliases, current_poni_files=None, parent=None):
        self.aliases = aliases

    def exec_(self):
        return QDialog.Accepted

    def get_poni_files(self):
        return {}


class _TechnicalWorkflowHarness(QMainWindow, technical_measurements.TechnicalMeasurementsMixin):
    def __init__(self, config, work_dir: Path):
        super().__init__()
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.detector_controller = {}
        self.hardware_controller = None
        self.stage_controller = None
        self.masks = {}
        self.ponis = {}
        self.poni_files = {}
        self._detector_distances = {}

        self._capture_iteration = 0
        self.logged_user = None
        self.hardware_initialized = False
        self.measurement_logs = []

        self._build_ui()
        self.enable_measurement_controls(False)

    def _build_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Login + hardware init controls
        login_row = QHBoxLayout()
        self.login_btn = QPushButton("Login sad")
        self.login_btn.clicked.connect(self._login_sad)
        login_row.addWidget(self.login_btn)

        self.initializeBtn = QPushButton("Initialize Hardware")
        self.initializeBtn.clicked.connect(self._initialize_hardware)
        login_row.addWidget(self.initializeBtn)

        layout.addLayout(login_row)

        # Controls required by TechnicalMeasurementsMixin
        self.integrationTimeSpin = QDoubleSpinBox()
        self.integrationTimeSpin.setValue(1.0)
        layout.addWidget(self.integrationTimeSpin)

        self.captureFramesSpin = QSpinBox()
        self.captureFramesSpin.setValue(1)
        layout.addWidget(self.captureFramesSpin)

        self.moveContinuousCheck = QCheckBox("Move Continuous")
        layout.addWidget(self.moveContinuousCheck)

        self.movementRadiusSpin = QDoubleSpinBox()
        self.movementRadiusSpin.setValue(2.0)
        layout.addWidget(self.movementRadiusSpin)

        self.folderLE = QLineEdit(str(self.work_dir))
        layout.addWidget(self.folderLE)

        self.auxNameLE = QLineEdit("aux")
        layout.addWidget(self.auxNameLE)

        self._aux_status = QLabel("")
        layout.addWidget(self._aux_status)

        self._aux_timer = QTimer(self)
        self._aux_timer.setInterval(100)
        self._aux_timer.timeout.connect(lambda: None)

        self.auxTable = QTableWidget()
        self.auxTable.setColumnCount(4)
        self.auxTable.setHorizontalHeaderLabels(["Primary", "File", "Type", "Alias"])
        layout.addWidget(self.auxTable)

        self.framesSpin = QSpinBox()
        self.framesSpin.setValue(1)
        layout.addWidget(self.framesSpin)

        self.rtBtn = QPushButton("Real-time")
        self.rtBtn.setCheckable(True)
        layout.addWidget(self.rtBtn)

        button_row = QHBoxLayout()
        self.dist_btn = QPushButton("Distances...")
        self.dist_btn.clicked.connect(self.configure_detector_distances)
        button_row.addWidget(self.dist_btn)

        self.auxBtn = QPushButton("Measure AUX")
        self.auxBtn.clicked.connect(self.measure_aux)
        button_row.addWidget(self.auxBtn)

        self.gen_h5_btn = QPushButton("Gen H5")
        self.gen_h5_btn.clicked.connect(self.generate_technical_h5)
        button_row.addWidget(self.gen_h5_btn)

        self.pyfai_btn = QPushButton("PyFAI")
        button_row.addWidget(self.pyfai_btn)

        self.validate_btn = QPushButton("Validate")
        button_row.addWidget(self.validate_btn)
        layout.addLayout(button_row)

    def _append_measurement_log(self, message: str):
        self.measurement_logs.append(message)

    def _login_sad(self):
        self.logged_user = "sad"

    def _initialize_hardware(self):
        self.hardware_initialized = True
        self.enable_measurement_controls(True)

    def _start_capture(self, typ: str):
        """Deterministic capture stub: one click -> 2 detector files."""
        technical_types = ["AGBH", "EMPTY", "BACKGROUND", "DARK"]
        current_type = technical_types[self._capture_iteration % len(technical_types)]
        capture_index = self._capture_iteration + 1
        timestamp = f"20260213_1200{capture_index:02d}"

        for alias in ("PRIMARY", "SECONDARY"):
            file_path = (
                self.work_dir
                / f"{current_type.lower()}_{capture_index:03d}_{timestamp}_{alias}.npy"
            )
            np.save(file_path, np.full((8, 8), capture_index, dtype=np.float32))
            self._add_aux_item_to_list(alias, str(file_path))

        self._capture_iteration += 1
        self._aux_timer.stop()
        self._aux_status.setText("Done")


def _patch_message_boxes(monkeypatch):
    dialogs = {"info": [], "warning": [], "critical": []}

    def _question(*args, **kwargs):
        return QMessageBox.Yes

    def _information(*args, **kwargs):
        dialogs["info"].append(args[2] if len(args) > 2 else "")
        return QMessageBox.Yes

    def _warning(*args, **kwargs):
        dialogs["warning"].append(args[2] if len(args) > 2 else "")
        return QMessageBox.Yes

    def _critical(*args, **kwargs):
        dialogs["critical"].append(args[2] if len(args) > 2 else "")
        return QMessageBox.Yes

    for module in (technical_measurements, h5_generation_mixin, h5_management_mixin):
        monkeypatch.setattr(module.QMessageBox, "question", staticmethod(_question))
        monkeypatch.setattr(module.QMessageBox, "information", staticmethod(_information))
        monkeypatch.setattr(module.QMessageBox, "warning", staticmethod(_warning))
        monkeypatch.setattr(module.QMessageBox, "critical", staticmethod(_critical))
    monkeypatch.setattr(
        h5_generation_mixin.QInputDialog,
        "getDouble",
        staticmethod(lambda *args, **kwargs: (0.0, True)),
    )

    return dialogs


def _set_primary_checked(table: QTableWidget, row: int):
    container = table.cellWidget(row, 0)
    checkbox = container.findChild(QCheckBox) if container else None
    assert checkbox is not None, f"Missing primary checkbox for row {row}"
    if not checkbox.isChecked():
        QTest.mouseClick(checkbox, Qt.LeftButton)
    assert checkbox.isChecked()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def test_gui_button_driven_technical_workflow(qapp, tmp_path, monkeypatch):
    # Keep technical workflow deterministic and local
    monkeypatch.setattr(technical_measurements, "_get_technical_imports", lambda: True)
    monkeypatch.setattr(
        "hardware.difra.gui.detector_distance_config_dialog.DetectorDistanceConfigDialog",
        _FakeDistanceDialog,
    )
    monkeypatch.setattr(technical_measurements, "PoniFileSelectionDialog", _FakePoniSelectionDialog)
    monkeypatch.setattr("hardware.difra.gui.operator_manager.OperatorManager", _FakeOperatorManager)
    dialogs = _patch_message_boxes(monkeypatch)

    work_dir = tmp_path / "technical_workflow"
    archive_dir = tmp_path / "archive" / "technical"
    temp_dir = tmp_path / "temp"

    config = {
        "DEV": True,
        "validate_containers_before_locking": True,
        "detectors": [
            {
                "id": "det_primary",
                "alias": "PRIMARY",
                "type": "Advacam",
                "size": {"width": 8, "height": 8},
                "pixel_size_um": [55.0, 55.0],
            },
            {
                "id": "det_secondary",
                "alias": "SECONDARY",
                "type": "Advacam",
                "size": {"width": 8, "height": 8},
                "pixel_size_um": [55.0, 55.0],
            },
        ],
        "dev_active_detectors": ["det_primary", "det_secondary"],
        "active_detectors": ["det_primary", "det_secondary"],
        "technical_temp_folder": str(temp_dir),
        "technical_archive_folder": str(archive_dir),
        "technical_archive_patterns": ["*.npy"],
    }

    window = _TechnicalWorkflowHarness(config=config, work_dir=work_dir)
    window.show()
    qapp.processEvents()

    # 1) Login as sad
    QTest.mouseClick(window.login_btn, Qt.LeftButton)
    assert window.logged_user == "sad"

    # 2) Init hardware
    QTest.mouseClick(window.initializeBtn, Qt.LeftButton)
    assert window.hardware_initialized is True

    # 3) Click Distances and set 17 cm for both detectors
    QTest.mouseClick(window.dist_btn, Qt.LeftButton)
    assert window._detector_distances == {
        "det_primary": 17.0,
        "det_secondary": 17.0,
    }

    # 4) Measure AUX four times -> 8 files (2 detectors x 4 runs)
    for _ in range(4):
        QTest.mouseClick(window.auxBtn, Qt.LeftButton)
        qapp.processEvents()

    assert window.auxTable.rowCount() == 8

    # 5) Assign types and set all as primary
    expected_types = [
        "AGBH", "AGBH",
        "EMPTY", "EMPTY",
        "BACKGROUND", "BACKGROUND",
        "DARK", "DARK",
    ]
    for row, expected_type in enumerate(expected_types):
        type_cb = window.auxTable.cellWidget(row, 2)
        assert isinstance(type_cb, QComboBox)
        type_cb.setCurrentText(expected_type)
        assert type_cb.currentText() == expected_type
        _set_primary_checked(window.auxTable, row)

    # 6) Generate H5, accept fake PONI fallback and lock flow
    QTest.mouseClick(window.gen_h5_btn, Qt.LeftButton)
    qapp.processEvents()

    # 7) Find resulting technical container and validate schema-level content
    generated = sorted(work_dir.glob("technical_*.nxs.h5"))
    assert len(generated) == 1, f"Expected exactly one technical container, found: {generated}"
    container_path = generated[0]

    with h5py.File(container_path, "r") as h5f:
        assert _decode_attr(h5f.attrs.get(schema.ATTR_CONTAINER_TYPE)) == schema.CONTAINER_TYPE_TECHNICAL
        assert _decode_attr(h5f.attrs.get(schema.ATTR_SCHEMA_VERSION)) == schema.SCHEMA_VERSION
        assert bool(h5f.attrs.get("locked", False)) is True
        assert schema.GROUP_TECHNICAL in h5f
        assert schema.GROUP_TECHNICAL_PONI in h5f

        event_ids = sorted(name for name in h5f[schema.GROUP_TECHNICAL].keys() if name.startswith("tech_evt_"))
        assert len(event_ids) == 4

        type_counts = {"AGBH": 0, "EMPTY": 0, "BACKGROUND": 0, "DARK": 0}
        for event_id in event_ids:
            event_group = h5f[f"{schema.GROUP_TECHNICAL}/{event_id}"]
            event_type = _decode_attr(event_group.attrs.get("type", ""))
            assert event_type in type_counts
            type_counts[event_type] += 1

            detector_groups = sorted(name for name in event_group.keys() if name.startswith("det_"))
            assert detector_groups == ["det_primary", "det_secondary"]
            for detector_group_name in detector_groups:
                detector_group = event_group[detector_group_name]
                assert float(detector_group.attrs.get(schema.ATTR_INTEGRATION_TIME_MS)) == 1000.0
                assert int(detector_group.attrs.get(schema.ATTR_N_FRAMES)) == 1

        assert type_counts == {"AGBH": 1, "EMPTY": 1, "BACKGROUND": 1, "DARK": 1}

        poni_group = h5f[schema.GROUP_TECHNICAL_PONI]
        assert "poni_primary" in poni_group
        assert "poni_secondary" in poni_group

    # Lock workflow archives raw files without ZIP bundling.
    archive_zips = sorted(archive_dir.glob("*.zip"))
    assert not archive_zips, f"Unexpected ZIP bundle(s): {archive_zips}"
    archive_subdirs = [p for p in archive_dir.glob("*") if p.is_dir()]
    assert archive_subdirs, f"No archive subfolder created in {archive_dir}"
    archived_npy = sorted(archive_subdirs[-1].glob("*.npy"))
    assert archived_npy, "Expected archived raw .npy files in technical archive folder"

    # No critical dialog should have been shown
    assert not dialogs["critical"], f"Unexpected critical dialogs: {dialogs['critical']}"
