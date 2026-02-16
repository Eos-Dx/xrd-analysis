"""GUI-level tests for loading technical and session containers."""

import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGraphicsScene,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from hardware.container.v0_2 import schema, technical_container, writer as session_writer
from hardware.container.v0_2.container_manager import lock_container
from hardware.difra.gui.main_window_ext import session_mixin, technical_measurements
from hardware.difra.gui.main_window_ext.technical import h5_management_mixin
from hardware.difra.gui.main_window_ext import state_saver_extension
from hardware.difra.gui.session_manager import SessionManager


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _patch_non_blocking_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.Ok))


def _make_technical_container(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)

    detector_config = [
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
    ]

    aux_measurements = {}
    for technical_type in ("DARK", "EMPTY", "BACKGROUND", "AGBH"):
        aux_measurements[technical_type] = {}
        for alias in ("PRIMARY", "SECONDARY"):
            npy_path = folder / f"{technical_type.lower()}_001_20260213_120000_{alias}.npy"
            np.save(npy_path, np.full((8, 8), len(technical_type), dtype=np.float32))
            aux_measurements[technical_type][alias] = str(npy_path)

    poni_content = (
        "# synthetic poni\n"
        "Distance: 0.17\n"
        "PixelSize1: 5.5e-05\n"
        "PixelSize2: 5.5e-05\n"
        "Poni1: 0.01\n"
        "Poni2: 0.02\n"
        "Rot1: 0\n"
        "Rot2: 0\n"
        "Rot3: 0\n"
        "Wavelength: 1.5406e-10\n"
    )
    poni_data = {
        "PRIMARY": (poni_content, "primary.poni"),
        "SECONDARY": (poni_content, "secondary.poni"),
    }

    _container_id, file_path = technical_container.generate_from_aux_table(
        folder=folder,
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=detector_config,
        active_detector_ids=["det_primary", "det_secondary"],
        distances_cm={"PRIMARY": 17.0, "SECONDARY": 17.0},
    )
    return Path(file_path)


class _TechnicalLoadHarness(QMainWindow, technical_measurements.TechnicalMeasurementsMixin):
    def __init__(self, config: dict, work_dir: Path):
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
        self.measurement_logs = []

        self._build_ui()

    def _append_measurement_log(self, message: str):
        self.measurement_logs.append(message)

    def _build_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

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

        self.auxTable = QTableWidget()
        self.auxTable.setColumnCount(4)
        self.auxTable.setHorizontalHeaderLabels(["Primary", "File", "Type", "Alias"])
        layout.addWidget(self.auxTable)


class _SessionLoadHarness(QMainWindow, session_mixin.SessionMixin):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.session_manager = SessionManager(config=config)
        self.status_updates = 0

    def update_session_status(self):
        self.status_updates += 1


class _FakeImageView:
    def __init__(self):
        self.scene = QGraphicsScene()
        self.shapes = []
        self.points_dict = {
            "generated": {"points": [], "zones": []},
            "user": {"points": [], "zones": []},
        }
        self.current_image_path = None
        self.image_item = None
        self.rotation_angle = 0
        self.crop_rect = None

    def set_image(self, pixmap, image_path=None):
        self.scene.clear()
        self.image_item = self.scene.addPixmap(pixmap)
        self.current_image_path = image_path


class _SessionRestoreHarness(
    QMainWindow, session_mixin.SessionMixin, state_saver_extension.StateSaverMixin
):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.session_manager = SessionManager(config=config)
        self.status_updates = 0
        self.image_view = _FakeImageView()
        self.state = {}
        self.next_point_id = 1
        self.pixel_to_mm_ratio = 2.0

    def update_session_status(self):
        self.status_updates += 1

    def update_points_table(self):
        pass

    def update_shape_table(self):
        pass

    def update_coordinates(self):
        pass


class _SessionAwareTechnicalLoadHarness(
    _TechnicalLoadHarness, session_mixin.SessionMixin
):
    def __init__(self, config: dict, work_dir: Path):
        _TechnicalLoadHarness.__init__(self, config=config, work_dir=work_dir)
        self.session_manager = SessionManager(config=config)
        self.status_updates = 0

    def update_session_status(self):
        self.status_updates += 1


def _get_primary_checkbox(table: QTableWidget, row: int) -> QCheckBox:
    container = table.cellWidget(row, 0)
    assert container is not None
    checkbox = container.findChild(QCheckBox)
    assert checkbox is not None
    return checkbox


def test_load_technical_h5_sets_primary_and_types(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    technical_path = _make_technical_container(tmp_path / "technical_h5_source")
    config = {
        "DEV": True,
        "detectors": [
            {"id": "det_primary", "alias": "PRIMARY"},
            {"id": "det_secondary", "alias": "SECONDARY"},
        ],
        "dev_active_detectors": ["det_primary", "det_secondary"],
        "active_detectors": ["det_primary", "det_secondary"],
    }
    harness = _TechnicalLoadHarness(config=config, work_dir=tmp_path / "ui_h5")
    harness.show()
    qapp.processEvents()

    monkeypatch.setattr(
        h5_management_mixin.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(technical_path), "NeXus HDF5 Files (*.nxs.h5)")),
    )

    harness.load_technical_h5()
    qapp.processEvents()

    assert harness.auxTable.rowCount() == 8

    type_counts = Counter()
    for row in range(harness.auxTable.rowCount()):
        checkbox = _get_primary_checkbox(harness.auxTable, row)
        assert checkbox.isChecked() is True

        type_cb = harness.auxTable.cellWidget(row, 2)
        technical_type = type_cb.currentText()
        assert technical_type in {"DARK", "EMPTY", "BACKGROUND", "AGBH"}
        type_counts[technical_type] += 1

        file_item = harness.auxTable.item(row, 1)
        assert file_item is not None
        source_value = file_item.data(Qt.UserRole)
        assert isinstance(source_value, str)
        assert source_value.endswith(".npy")

    assert type_counts == {"DARK": 2, "EMPTY": 2, "BACKGROUND": 2, "AGBH": 2}


def test_load_technical_files_without_container(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    raw_dir = tmp_path / "technical_raw_only"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_files = []
    for technical_type in ("DARK", "EMPTY", "BACKGROUND", "AGBH"):
        npy_path = raw_dir / f"{technical_type.lower()}_001_20260213_120100_PRIMARY.npy"
        np.save(npy_path, np.ones((8, 8), dtype=np.float32))
        raw_files.append(str(npy_path))

    config = {
        "DEV": True,
        "detectors": [{"id": "det_primary", "alias": "PRIMARY"}],
        "dev_active_detectors": ["det_primary"],
        "active_detectors": ["det_primary"],
    }
    harness = _TechnicalLoadHarness(config=config, work_dir=tmp_path / "ui_raw")
    harness.show()
    qapp.processEvents()

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: (raw_files, "NumPy Arrays (*.npy)")),
    )

    harness.load_technical_files()
    qapp.processEvents()

    assert harness.auxTable.rowCount() == len(raw_files)

    seen_types = set()
    seen_files = set()
    for row in range(harness.auxTable.rowCount()):
        type_cb = harness.auxTable.cellWidget(row, 2)
        seen_types.add(type_cb.currentText())

        file_item = harness.auxTable.item(row, 1)
        source_value = file_item.data(Qt.UserRole)
        seen_files.add(source_value)
        assert source_value in raw_files

    assert seen_types == {"DARK", "EMPTY", "BACKGROUND", "AGBH"}
    assert seen_files == set(raw_files)


def test_open_existing_session_container_updates_state(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    session_dir = tmp_path / "session_source"
    session_id, session_path = session_writer.create_session_container(
        folder=session_dir,
        sample_id="SAMPLE_LOAD_001",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-13",
    )
    lock_container(Path(session_path), user_id="sad")

    harness = _SessionLoadHarness(config={"operator_id": "sad"})
    harness.show()
    qapp.processEvents()

    monkeypatch.setattr(
        session_mixin.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(session_path), "NeXus HDF5 Files (*.nxs.h5)")),
    )

    harness.on_restore_session()
    qapp.processEvents()

    assert harness.session_manager.session_path == Path(session_path)
    assert harness.session_manager.sample_id == "SAMPLE_LOAD_001"
    assert harness.session_manager.session_id == session_id
    assert harness.status_updates == 1


def test_aux_state_roundtrip_uses_current_columns(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    raw_dir = tmp_path / "aux_state_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / "dark_001_20260213_121000_PRIMARY.npy"
    np.save(file_path, np.ones((8, 8), dtype=np.float32))

    config = {
        "DEV": True,
        "detectors": [{"id": "det_primary", "alias": "PRIMARY"}],
        "dev_active_detectors": ["det_primary"],
        "active_detectors": ["det_primary"],
    }
    harness = _TechnicalLoadHarness(config=config, work_dir=tmp_path / "ui_state")
    harness.show()
    qapp.processEvents()

    harness._add_aux_item_to_list("PRIMARY", str(file_path))
    row = 0
    type_cb = harness.auxTable.cellWidget(row, 2)
    type_cb.setCurrentText("DARK")
    primary_cb = _get_primary_checkbox(harness.auxTable, row)
    primary_cb.setChecked(True)

    state = harness.build_aux_state()
    assert len(state) == 1
    assert state[0]["file_path"] == str(file_path)
    assert state[0]["type"] == "DARK"
    assert state[0]["alias"] == "PRIMARY"
    assert state[0]["is_primary"] is True

    harness.restore_technical_aux_rows(state)
    qapp.processEvents()
    assert harness.auxTable.rowCount() == 1
    restored_type_cb = harness.auxTable.cellWidget(0, 2)
    assert restored_type_cb.currentText() == "DARK"
    restored_primary_cb = _get_primary_checkbox(harness.auxTable, 0)
    assert restored_primary_cb.isChecked() is True


def test_restore_session_recovers_image_zones_and_points(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    technical_folder = tmp_path / "technical_restore"
    technical_path = _make_technical_container(technical_folder)
    lock_container(technical_path, user_id="sad")

    config = {
        "technical_folder": str(technical_folder),
        "operator_id": "sad",
        "site_id": "ULSTER",
        "machine_name": "DIFRA_TEST",
        "beam_energy_kev": 17.5,
    }
    manager = SessionManager(config=config)
    _session_id, session_path = manager.create_session(
        folder=tmp_path / "sessions_restore",
        distance_cm=17.0,
        sample_id="RESTORE_SAMPLE",
        study_name="RESTORE_STUDY",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-13",
    )
    manager.add_sample_image(
        image_data=np.full((32, 32), 7, dtype=np.uint8),
        image_index=1,
        image_type="sample",
    )
    manager.add_zone(
        zone_index=1,
        zone_role="sample_holder",
        geometry_px=[8, 8, 16, 16],
        shape="circle",
        holder_diameter_mm=8.0,
    )
    manager.add_zone(
        zone_index=2,
        zone_role="exclude",
        geometry_px=[12, 12, 4, 4],
        shape="circle",
    )
    manager.add_points(
        [
            {
                "pixel_coordinates": [10 + i, 11 + i],
                "physical_coordinates_mm": [0.0, 0.0],
                "point_status": "pending",
            }
            for i in range(5)
        ]
    )
    manager.close_session()

    harness = _SessionRestoreHarness(config=config)
    harness.show()
    qapp.processEvents()

    monkeypatch.setattr(
        session_mixin.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(session_path), "NeXus HDF5 Files (*.nxs.h5)")),
    )

    harness.on_restore_session()
    qapp.processEvents()

    assert harness.session_manager.session_path == Path(session_path)
    assert harness.session_manager.sample_id == "RESTORE_SAMPLE"
    assert harness.session_manager.study_name == "RESTORE_STUDY"
    assert harness.status_updates == 1

    assert harness.image_view.image_item is not None
    assert len(harness.image_view.shapes) == 2
    assert len(harness.image_view.points_dict["generated"]["points"]) == 5
    assert len(harness.state.get("shapes", [])) == 2
    assert len(harness.state.get("zone_points", [])) == 5


def test_sync_workspace_snapshot_to_unlocked_session(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    technical_folder = tmp_path / "technical_sync"
    technical_path = _make_technical_container(technical_folder)
    lock_container(technical_path, user_id="sad")

    config = {
        "technical_folder": str(technical_folder),
        "operator_id": "sad",
        "site_id": "ULSTER",
        "machine_name": "DIFRA_TEST",
        "beam_energy_kev": 17.5,
    }
    harness = _SessionRestoreHarness(config=config)
    harness.show()
    qapp.processEvents()

    _session_id, session_path = harness.session_manager.create_session(
        folder=tmp_path / "sessions_sync",
        distance_cm=17.0,
        sample_id="SYNC_SAMPLE",
        study_name="SYNC_STUDY",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-13",
    )

    monkeypatch.setattr(
        harness,
        "_extract_current_image_array",
        lambda: np.full((24, 24), 123, dtype=np.uint8),
    )
    harness.state = {
        "shapes": [
            {
                "id": 1,
                "type": "circle",
                "role": "include",
                "geometry": {"x": 4, "y": 4, "width": 12, "height": 12},
            },
            {
                "id": 2,
                "type": "circle",
                "role": "exclude",
                "geometry": {"x": 8, "y": 8, "width": 4, "height": 4},
            },
        ],
        "zone_points": [
            {"id": i + 1, "x": 5 + i, "y": 6 + i, "type": "generated", "radius": 5}
            for i in range(5)
        ],
    }

    harness.sync_workspace_to_session_container(state=harness.state)

    with h5py.File(session_path, "r") as h5f:
        assert "/entry/images/img_001" in h5f
        assert len(h5f["/entry/images/zones"].keys()) == 2
        assert len(h5f["/entry/points"].keys()) == 5
        mapping = h5f["/entry/images/mapping/mapping"][()]
        if isinstance(mapping, bytes):
            mapping = mapping.decode("utf-8")
        assert "ratio" in mapping


def test_loading_technical_updates_active_unlocked_session(qapp, tmp_path, monkeypatch):
    _patch_non_blocking_dialogs(monkeypatch)

    technical_a = _make_technical_container(tmp_path / "technical_a")
    technical_b = _make_technical_container(tmp_path / "technical_b")
    lock_container(technical_a, user_id="sad")

    config = {
        "DEV": True,
        "detectors": [
            {"id": "det_primary", "alias": "PRIMARY"},
            {"id": "det_secondary", "alias": "SECONDARY"},
        ],
        "dev_active_detectors": ["det_primary", "det_secondary"],
        "active_detectors": ["det_primary", "det_secondary"],
        "technical_folder": str(tmp_path / "technical_a"),
        "operator_id": "sad",
    }
    harness = _SessionAwareTechnicalLoadHarness(
        config=config,
        work_dir=tmp_path / "ui_session_tech",
    )
    harness.show()
    qapp.processEvents()

    _session_id, session_path = harness.session_manager.create_session(
        folder=tmp_path / "sessions_update",
        distance_cm=17.0,
        sample_id="TECH_SWAP",
        study_name="TECH_SWAP_STUDY",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-13",
    )

    monkeypatch.setattr(
        h5_management_mixin.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(technical_b), "NeXus HDF5 Files (*.nxs.h5)")),
    )

    harness.load_technical_h5()
    qapp.processEvents()

    with h5py.File(session_path, "r") as h5f:
        source_file = h5f[schema.GROUP_CALIBRATION_SNAPSHOT].attrs.get("source_file", "")
        if isinstance(source_file, bytes):
            source_file = source_file.decode("utf-8")
        assert source_file == str(technical_b)

    assert harness.session_manager.technical_container_path == Path(technical_b)
