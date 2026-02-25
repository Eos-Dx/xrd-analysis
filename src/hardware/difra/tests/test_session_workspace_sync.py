"""Session workspace sync/restore regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from hardware.container.v0_2 import schema, technical_container
from hardware.container.v0_2.container_manager import lock_container
from hardware.difra.gui.main_window_ext.session_mixin import SessionMixin
from hardware.difra.gui.session_manager import SessionManager


class _SpinStub:
    def __init__(self, value):
        self._value = float(value)

    def value(self):
        return float(self._value)

    def setValue(self, value):
        self._value = float(value)


class _Harness(SessionMixin):
    def __init__(self, config):
        self.config = config
        self.session_manager = SessionManager(config=config)
        self.image_view = SimpleNamespace(current_image_path=None)
        self.state = {}
        self.pixel_to_mm_ratio = 1.0
        self.include_center = (0.0, 0.0)
        self.real_x_pos_mm = _SpinStub(0.0)
        self.real_y_pos_mm = _SpinStub(0.0)
        self.logs = []

    def _append_session_log(self, message: str):
        self.logs.append(str(message))

    def update_points_table(self):
        return None

    def update_shape_table(self):
        return None

    def update_coordinates(self):
        return None


def _make_locked_technical_container(folder):
    folder.mkdir(parents=True, exist_ok=True)
    detector_config = [
        {
            "id": "det_primary",
            "alias": "PRIMARY",
            "type": "Advacam",
            "size": {"width": 8, "height": 8},
            "pixel_size_um": [55.0, 55.0],
        }
    ]

    dark_npy = folder / "dark_primary.npy"
    np.save(dark_npy, np.full((8, 8), 5, dtype=np.float32))
    empty_npy = folder / "empty_primary.npy"
    np.save(empty_npy, np.full((8, 8), 7, dtype=np.float32))
    bkg_npy = folder / "background_primary.npy"
    np.save(bkg_npy, np.full((8, 8), 9, dtype=np.float32))
    agbh_npy = folder / "agbh_primary.npy"
    np.save(agbh_npy, np.full((8, 8), 11, dtype=np.float32))

    aux_measurements = {
        "DARK": {"PRIMARY": str(dark_npy)},
        "EMPTY": {"PRIMARY": str(empty_npy)},
        "BACKGROUND": {"PRIMARY": str(bkg_npy)},
        "AGBH": {"PRIMARY": str(agbh_npy)},
    }
    poni_content = (
        "Detector: Detector\n"
        "PixelSize1: 5.500e-05\n"
        "PixelSize2: 5.500e-05\n"
        "Distance: 0.170000\n"
        "Poni1: 0.01\n"
        "Poni2: 0.02\n"
        "Rot1: 0\n"
        "Rot2: 0\n"
        "Rot3: 0\n"
        "Wavelength: 1.5406e-10\n"
    )
    poni_data = {"PRIMARY": (poni_content, "primary.poni")}

    _cid, path = technical_container.generate_from_aux_table(
        folder=folder,
        aux_measurements=aux_measurements,
        poni_data=poni_data,
        detector_config=detector_config,
        active_detector_ids=["det_primary"],
        distances_cm=17.0,
    )
    lock_container(path, user_id="sad")
    return path


def test_workspace_sync_writes_physical_coordinates_and_keeps_image_immutable(tmp_path):
    technical_path = _make_locked_technical_container(tmp_path / "technical")
    config = {"technical_folder": str(Path(technical_path).parent), "operator_id": "sad"}
    harness = _Harness(config=config)

    _sid, session_path = harness.session_manager.create_session(
        folder=tmp_path / "sessions",
        distance_cm=17.0,
        sample_id="SYNC_SAMPLE",
        study_name="SYNC_STUDY",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-25",
    )

    harness.pixel_to_mm_ratio = 2.0
    harness.include_center = (100.0, 200.0)
    harness.real_x_pos_mm.setValue(1.5)
    harness.real_y_pos_mm.setValue(-2.0)

    image_calls = {"count": 0}

    def _image_source():
        image_calls["count"] += 1
        if image_calls["count"] == 1:
            return np.full((12, 12), 11, dtype=np.uint8)
        return np.full((12, 12), 99, dtype=np.uint8)

    harness._extract_current_image_array = _image_source
    harness.state = {
        "shapes": [
            {
                "id": 1,
                "type": "circle",
                "role": "include",
                "geometry": {"x": 10, "y": 10, "width": 40, "height": 40},
            }
        ],
        "zone_points": [
            {"id": 1, "x": 110.0, "y": 220.0, "type": "generated", "radius": 5},
            {"id": 2, "x": 90.0, "y": 180.0, "type": "generated", "radius": 5},
        ],
    }

    harness.sync_workspace_to_session_container(state=harness.state)
    harness.sync_workspace_to_session_container(state=harness.state)

    with h5py.File(session_path, "r") as h5f:
        image_data = h5f[f"{schema.GROUP_IMAGES}/img_001/data"][()]
        assert int(image_data[0, 0]) == 11

        mapping_raw = h5f[f"{schema.GROUP_IMAGES_MAPPING}/mapping"][()]
        if isinstance(mapping_raw, bytes):
            mapping_raw = mapping_raw.decode("utf-8")
        mapping = json.loads(mapping_raw)
        conversion = mapping.get("pixel_to_mm_conversion", {})
        assert float(conversion["ratio"]) == 2.0
        assert conversion.get("include_center_px") == [100.0, 200.0]
        assert conversion.get("stage_reference_mm") == [1.5, -2.0]

        pt_001 = h5f[f"{schema.GROUP_POINTS}/pt_001"]
        mm = pt_001.attrs[schema.ATTR_PHYSICAL_COORDINATES_MM]
        assert float(mm[0]) == 1.5 - (110.0 - 100.0) / 2.0
        assert float(mm[1]) == -2.0 - (220.0 - 200.0) / 2.0

    assert image_calls["count"] == 1


def test_workspace_restore_reapplies_mapping_origin(tmp_path):
    technical_path = _make_locked_technical_container(tmp_path / "technical_restore")
    config = {"technical_folder": str(Path(technical_path).parent), "operator_id": "sad"}
    harness = _Harness(config=config)

    _sid, session_path = harness.session_manager.create_session(
        folder=tmp_path / "sessions_restore",
        distance_cm=17.0,
        sample_id="RESTORE_SAMPLE",
        study_name="RESTORE_STUDY",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-25",
    )

    harness._extract_current_image_array = lambda: None
    harness.pixel_to_mm_ratio = 9.5
    harness.include_center = (123.0, 234.0)
    harness.real_x_pos_mm.setValue(7.25)
    harness.real_y_pos_mm.setValue(-3.75)
    harness.state = {
        "shapes": [
            {
                "id": 1,
                "type": "circle",
                "role": "include",
                "geometry": {"x": 1, "y": 2, "width": 3, "height": 4},
            }
        ],
        "zone_points": [
            {"id": 1, "x": 11.0, "y": 22.0, "type": "generated", "radius": 5}
        ],
    }
    harness.sync_workspace_to_session_container(state=harness.state)

    harness.pixel_to_mm_ratio = 1.0
    harness.include_center = (0.0, 0.0)
    harness.real_x_pos_mm.setValue(0.0)
    harness.real_y_pos_mm.setValue(0.0)

    harness._restore_session_workspace_from_container(Path(session_path))

    assert harness.pixel_to_mm_ratio == 9.5
    assert harness.include_center == (123.0, 234.0)
    assert harness.real_x_pos_mm.value() == 7.25
    assert harness.real_y_pos_mm.value() == -3.75
