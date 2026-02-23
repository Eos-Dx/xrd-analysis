from pathlib import Path

import h5py
import numpy as np

from hardware.container.v0_2 import schema, technical_container


def _make_detector_config():
    return [
        {
            "id": "det_primary",
            "alias": "PRIMARY",
            "type": "Advacam",
            "size": {"width": 8, "height": 8},
            "pixel_size_um": [55.0, 55.0],
        }
    ]


def test_technical_event_stores_acquisition_metadata(tmp_path):
    data_path = Path(tmp_path) / "dark_primary.npy"
    np.save(data_path, np.ones((8, 8), dtype=np.float32))

    aux_measurements = {
        "DARK": {
            "PRIMARY": {
                "file_path": str(data_path),
                "integration_time_ms": 2500.0,
                "n_frames": 7,
                "thickness": 1.3,
            }
        }
    }

    _container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(tmp_path),
        aux_measurements=aux_measurements,
        poni_data={},
        detector_config=_make_detector_config(),
        active_detector_ids=["det_primary"],
        distances_cm={"PRIMARY": 17.0},
    )

    with h5py.File(file_path, "r") as file_handle:
        event_id = schema.format_technical_event_id(1)
        det_group = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary"]
        assert float(det_group.attrs[schema.ATTR_INTEGRATION_TIME_MS]) == 2500.0
        assert int(det_group.attrs[schema.ATTR_N_FRAMES]) == 7
        assert float(det_group.attrs[schema.ATTR_THICKNESS]) == 1.3


def test_technical_event_parses_exposure_frames_from_filename_and_applies_default_thickness(
    tmp_path,
):
    data_path = Path(tmp_path) / "dark_001_20260223_120000_0.500000s_12frames_PRIMARY.npy"
    np.save(data_path, np.ones((8, 8), dtype=np.float32))

    aux_measurements = {"DARK": {"PRIMARY": str(data_path)}}

    _container_id, file_path = technical_container.generate_from_aux_table(
        folder=str(tmp_path),
        aux_measurements=aux_measurements,
        poni_data={},
        detector_config=_make_detector_config(),
        active_detector_ids=["det_primary"],
        distances_cm={"PRIMARY": 17.0},
        technical_thickness_mm=2.5,
    )

    with h5py.File(file_path, "r") as file_handle:
        event_id = schema.format_technical_event_id(1)
        det_group = file_handle[f"{schema.GROUP_TECHNICAL}/{event_id}/det_primary"]
        assert float(det_group.attrs[schema.ATTR_INTEGRATION_TIME_MS]) == 500.0
        assert int(det_group.attrs[schema.ATTR_N_FRAMES]) == 12
        assert float(det_group.attrs[schema.ATTR_THICKNESS]) == 2.5
