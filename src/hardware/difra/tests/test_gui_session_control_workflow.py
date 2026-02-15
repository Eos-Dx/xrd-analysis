"""GUI integration test for session-control workflow interactions.

Workflow covered:
1. Login as user ``sad``
2. Upload sample image
3. Create three circle zones (sample_holder/include/exclude)
4. Generate 5 measurement points
5. Start measurements
6. Close/lock session container
7. Validate resulting session container against v0.2 schema expectations
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hardware.container.v0_2 import schema, technical_container
from hardware.container.v0_2.container_manager import lock_container
from hardware.difra.gui.session_manager import SessionManager


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _create_locked_technical_container(technical_folder: Path) -> Path:
    technical_folder.mkdir(parents=True, exist_ok=True)

    detector_config = [
        {
            "id": "det_primary",
            "alias": "PRIMARY",
            "type": "Advacam",
            "size": {"width": 16, "height": 16},
            "pixel_size_um": [55.0, 55.0],
        },
        {
            "id": "det_secondary",
            "alias": "SECONDARY",
            "type": "Advacam",
            "size": {"width": 16, "height": 16},
            "pixel_size_um": [55.0, 55.0],
        },
    ]
    active_detector_ids = ["det_primary", "det_secondary"]

    aux_measurements = {}
    for technical_type in ("DARK", "EMPTY", "BACKGROUND", "AGBH"):
        aux_measurements[technical_type] = {}
        for alias in ("PRIMARY", "SECONDARY"):
            file_path = technical_folder / f"{technical_type.lower()}_{alias.lower()}.npy"
            np.save(file_path, np.full((16, 16), len(technical_type), dtype=np.float32))
            aux_measurements[technical_type][alias] = str(file_path)

    poni_template = (
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
        "PRIMARY": (poni_template, "primary.poni"),
        "SECONDARY": (poni_template, "secondary.poni"),
    }

    _container_id, file_path = technical_container.generate_from_aux_table(
        folder=technical_folder,
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=detector_config,
        active_detector_ids=active_detector_ids,
        distances_cm={"PRIMARY": 17.0, "SECONDARY": 17.0},
    )

    lock_container(Path(file_path), user_id="sad")
    return Path(file_path)


class _SessionWorkflowHarness(QMainWindow):
    """Small GUI harness exposing workflow actions as buttons."""

    def __init__(self, base_dir: Path):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.technical_folder = self.base_dir / "technical"
        self.session_folder = self.base_dir / "sessions"
        self.technical_container = _create_locked_technical_container(self.technical_folder)

        self.logged_user = None
        self.session_manager = SessionManager(
            config={
                "technical_folder": str(self.technical_folder),
                "operator_id": "sad",
                "site_id": "ULSTER",
                "machine_name": "DIFRA_TEST",
                "beam_energy_kev": 17.5,
            }
        )
        self.generated_points = []
        self.created_zones = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.sample_id_le = QLineEdit("SAMPLE_SESSION_001")
        layout.addWidget(self.sample_id_le)

        self.status_label = QLabel("idle")
        layout.addWidget(self.status_label)

        row = QHBoxLayout()
        self.login_btn = QPushButton("Login sad")
        self.login_btn.clicked.connect(self._login_sad)
        row.addWidget(self.login_btn)

        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_image_btn.clicked.connect(self._upload_image_and_create_session)
        row.addWidget(self.upload_image_btn)

        layout.addLayout(row)

        row2 = QHBoxLayout()
        self.zones_btn = QPushButton("Set Zones")
        self.zones_btn.clicked.connect(self._set_zones)
        row2.addWidget(self.zones_btn)

        self.points_btn = QPushButton("Generate 5 Points")
        self.points_btn.clicked.connect(self._generate_points)
        row2.addWidget(self.points_btn)

        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.attenuation_btn = QPushButton("Measure Attenuation")
        self.attenuation_btn.clicked.connect(self._measure_attenuation)
        row3.addWidget(self.attenuation_btn)

        self.measure_btn = QPushButton("Start Measurement")
        self.measure_btn.clicked.connect(self._start_measurement)
        row3.addWidget(self.measure_btn)

        self.close_btn = QPushButton("Close Container")
        self.close_btn.clicked.connect(self._close_container)
        row3.addWidget(self.close_btn)

        layout.addLayout(row3)

    def _login_sad(self):
        self.logged_user = "sad"
        self.status_label.setText("logged")

    def _upload_image_and_create_session(self):
        sample_id = self.sample_id_le.text().strip()
        if not sample_id:
            raise ValueError("sample_id required")

        self.session_manager.operator_id = self.logged_user or "sad"
        self.session_manager.create_session(
            folder=self.session_folder,
            distance_cm=17.0,
            sample_id=sample_id,
            study_name="STUDY_GUI",
            operator_id=self.logged_user or "sad",
            site_id="ULSTER",
            machine_name="DIFRA_TEST",
            beam_energy_keV=17.5,
            acquisition_date="2026-02-13",
        )

        image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        self.session_manager.add_sample_image(image_data=image, image_index=1, image_type="sample")
        self.status_label.setText("session_created")

    def _set_zones(self):
        zones = [
            (1, "sample_holder", {"center": [32, 32], "radius": 20}, 25.0),
            (2, "include", {"center": [32, 32], "radius": 12}, None),
            (3, "exclude", {"center": [38, 32], "radius": 5}, None),
        ]
        self.created_zones = []
        for zone_index, role, geometry, holder_diameter in zones:
            path = self.session_manager.add_zone(
                zone_index=zone_index,
                geometry_px=json.dumps(geometry),
                shape="circle",
                zone_role=role,
                holder_diameter_mm=holder_diameter,
            )
            self.created_zones.append(path)
        self.status_label.setText("zones_set")

    def _generate_points(self):
        points = []
        for idx in range(5):
            points.append(
                {
                    "pixel_coordinates": [20 + idx * 3, 20 + idx * 2],
                    "physical_coordinates_mm": [1.5 * idx, 1.0 * idx],
                    "point_status": schema.POINT_STATUS_PENDING,
                }
            )
        self.generated_points = self.session_manager.add_points(points)
        self.status_label.setText("points_generated")

    def _start_measurement(self):
        measurement_data = {
            "det_primary": np.full((16, 16), 10.0, dtype=np.float32),
            "det_secondary": np.full((16, 16), 20.0, dtype=np.float32),
        }
        detector_metadata = {
            "det_primary": {"integration_time_ms": 1000.0, "beam_energy_keV": 17.5},
            "det_secondary": {"integration_time_ms": 1000.0, "beam_energy_keV": 17.5},
        }
        poni_alias_map = {"PRIMARY": "det_primary", "SECONDARY": "det_secondary"}

        for point_index in range(1, 6):
            self.session_manager.add_measurement(
                point_index=point_index,
                measurement_data=measurement_data,
                detector_metadata=detector_metadata,
                poni_alias_map=poni_alias_map,
            )
        self.status_label.setText("measured")

    def _measure_attenuation(self):
        measurement_data = {
            "det_primary": np.full((16, 16), 30.0, dtype=np.float32),
            "det_secondary": np.full((16, 16), 40.0, dtype=np.float32),
        }
        detector_metadata = {
            "det_primary": {"integration_time_ms": 800.0, "beam_energy_keV": 17.5},
            "det_secondary": {"integration_time_ms": 800.0, "beam_energy_keV": 17.5},
        }
        poni_alias_map = {"PRIMARY": "det_primary", "SECONDARY": "det_secondary"}

        self.session_manager.add_attenuation_measurement(
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            mode="without",
        )
        self.session_manager.add_attenuation_measurement(
            measurement_data=measurement_data,
            detector_metadata=detector_metadata,
            poni_alias_map=poni_alias_map,
            mode="with",
        )
        self.session_manager.link_attenuation_to_points(num_points=5)
        self.status_label.setText("attenuation_measured")

    def _close_container(self):
        if not self.session_manager.session_path:
            raise RuntimeError("No active session to close")
        lock_container(Path(self.session_manager.session_path), user_id=self.logged_user or "sad")
        self.session_manager.close_session()
        self.status_label.setText("closed")


def test_session_control_button_workflow(qapp, tmp_path, monkeypatch):
    # Keep dialogs non-blocking in automation context
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.Yes))

    harness = _SessionWorkflowHarness(base_dir=tmp_path / "session_gui")
    harness.show()
    qapp.processEvents()

    # 1) Login sad
    QTest.mouseClick(harness.login_btn, Qt.LeftButton)
    assert harness.logged_user == "sad"

    # 2) Upload image (creates session + stores sample image)
    QTest.mouseClick(harness.upload_image_btn, Qt.LeftButton)
    assert harness.session_manager.is_session_active() is True
    session_path = Path(harness.session_manager.session_path)
    assert session_path.exists()

    # 3) Set 3 circle zones (sample_holder/include/exclude)
    QTest.mouseClick(harness.zones_btn, Qt.LeftButton)
    assert len(harness.created_zones) == 3

    # 4) Generate 5 points
    QTest.mouseClick(harness.points_btn, Qt.LeftButton)
    assert len(harness.generated_points) == 5

    # 5) Start measurements
    QTest.mouseClick(harness.measure_btn, Qt.LeftButton)
    assert harness.status_label.text() == "measured"

    # 6) Close container
    QTest.mouseClick(harness.close_btn, Qt.LeftButton)
    assert harness.status_label.text() == "closed"
    assert harness.session_manager.is_session_active() is False

    # 7) Validate session container against schema expectations
    with h5py.File(session_path, "r") as h5f:
        assert _decode(h5f.attrs.get(schema.ATTR_CONTAINER_TYPE)) == schema.CONTAINER_TYPE_SESSION
        assert _decode(h5f.attrs.get(schema.ATTR_SCHEMA_VERSION)) == schema.SCHEMA_VERSION
        assert bool(h5f.attrs.get("locked", False)) is True
        assert _decode(h5f.attrs.get(schema.ATTR_SAMPLE_ID)) == "SAMPLE_SESSION_001"
        assert _decode(h5f.attrs.get(schema.ATTR_STUDY_NAME)) == "STUDY_GUI"
        assert _decode(h5f.attrs.get(schema.ATTR_OPERATOR_ID)) == "sad"

        assert schema.GROUP_CALIBRATION_SNAPSHOT in h5f
        assert any(
            name.startswith("tech_evt_")
            for name in h5f[schema.GROUP_CALIBRATION_SNAPSHOT].keys()
        )
        assert f"{schema.GROUP_CALIBRATION_SNAPSHOT}/poni" in h5f
        assert schema.GROUP_IMAGES in h5f
        assert schema.GROUP_IMAGES_ZONES in h5f
        assert schema.GROUP_POINTS in h5f
        assert schema.GROUP_MEASUREMENTS in h5f

        # image uploaded
        assert "img_001" in h5f[schema.GROUP_IMAGES]

        # three zones with expected roles
        zone_ids = sorted(h5f[schema.GROUP_IMAGES_ZONES].keys())
        assert zone_ids == ["zone_001", "zone_002", "zone_003"]
        roles = [
            _decode(h5f[f"{schema.GROUP_IMAGES_ZONES}/{zone_id}"].attrs.get(schema.ATTR_ZONE_ROLE))
            for zone_id in zone_ids
        ]
        assert roles == ["sample_holder", "include", "exclude"]

        # five points + measured status
        point_ids = sorted(h5f[schema.GROUP_POINTS].keys())
        assert point_ids == ["pt_001", "pt_002", "pt_003", "pt_004", "pt_005"]
        for point_id in point_ids:
            point_group = h5f[f"{schema.GROUP_POINTS}/{point_id}"]
            assert _decode(point_group.attrs.get(schema.ATTR_POINT_STATUS)) == schema.POINT_STATUS_MEASURED

        # one measurement per point, with both detectors
        for point_id in point_ids:
            point_measurements = h5f[f"{schema.GROUP_MEASUREMENTS}/{point_id}"]
            measurement_ids = sorted(point_measurements.keys())
            assert len(measurement_ids) == 1
            measurement_group = point_measurements[measurement_ids[0]]
            detector_groups = sorted(name for name in measurement_group.keys() if name.startswith("det_"))
            assert detector_groups == ["det_primary", "det_secondary"]


def test_session_control_button_workflow_with_attenuation(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.Yes))

    harness = _SessionWorkflowHarness(base_dir=tmp_path / "session_gui_attenuation")
    harness.show()
    qapp.processEvents()

    QTest.mouseClick(harness.login_btn, Qt.LeftButton)
    QTest.mouseClick(harness.upload_image_btn, Qt.LeftButton)
    QTest.mouseClick(harness.zones_btn, Qt.LeftButton)
    QTest.mouseClick(harness.points_btn, Qt.LeftButton)

    # Attenuation-enabled branch
    QTest.mouseClick(harness.attenuation_btn, Qt.LeftButton)
    assert harness.status_label.text() == "attenuation_measured"

    QTest.mouseClick(harness.measure_btn, Qt.LeftButton)
    assert harness.status_label.text() == "measured"

    session_path = Path(harness.session_manager.session_path)
    QTest.mouseClick(harness.close_btn, Qt.LeftButton)
    assert harness.status_label.text() == "closed"

    with h5py.File(session_path, "r") as h5f:
        analytical_group = h5f[schema.GROUP_ANALYTICAL_MEASUREMENTS]
        analytical_ids = sorted(analytical_group.keys())
        assert len(analytical_ids) == 2

        for analytical_id in analytical_ids:
            analytical_path = f"{schema.GROUP_ANALYTICAL_MEASUREMENTS}/{analytical_id}"
            assert _decode(h5f[analytical_path].attrs.get(schema.ATTR_ANALYSIS_TYPE)) == "attenuation"

        point_ids = sorted(h5f[schema.GROUP_POINTS].keys())
        assert len(point_ids) == 5
        for point_id in point_ids:
            point_path = f"{schema.GROUP_POINTS}/{point_id}"
            assert schema.ATTR_ANALYTICAL_MEASUREMENT_IDS in h5f[point_path].attrs
            assert len(h5f[point_path].attrs[schema.ATTR_ANALYTICAL_MEASUREMENT_IDS]) == 2
