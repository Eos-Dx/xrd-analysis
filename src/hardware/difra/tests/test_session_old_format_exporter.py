"""Tests for exporting legacy old-format folders from session containers."""

import json
from pathlib import Path

import h5py
import numpy as np

from hardware.container.v0_2 import technical_container, writer as session_writer
from hardware.difra.gui.session_old_format_exporter import SessionOldFormatExporter


def _create_session_with_technical_and_measurement(tmp_path: Path) -> Path:
    technical_dir = tmp_path / "technical"
    measurements_dir = tmp_path / "measurements"

    _tech_id, tech_path = technical_container.create_technical_container(
        folder=technical_dir,
        distance_cm=17.0,
        container_id="aaaaaaaaaaaaaaaa",
    )
    technical_container.write_detector_config(
        file_path=tech_path,
        detectors_config=[
            {
                "id": "DET-PRIMARY",
                "alias": "PRIMARY",
                "type": "Pixet",
                "size": {"width": 4, "height": 4},
                "pixel_size_um": [55.0, 55.0],
            }
        ],
        active_detector_ids=["DET-PRIMARY"],
    )
    technical_container.write_poni_datasets(
        file_path=tech_path,
        poni_data={"PRIMARY": ("Distance: 0.17\nPoni1: 0.001\nPoni2: 0.002\n", "primary.poni")},
        distances_cm={"PRIMARY": 17.0},
        detector_id_by_alias={"PRIMARY": "DET-PRIMARY"},
    )
    technical_container.add_technical_event(
        file_path=tech_path,
        event_index=1,
        technical_type="AGBH",
        measurements={
            "PRIMARY": {
                "data": np.ones((4, 4), dtype=np.float32),
                "detector_id": "DET-PRIMARY",
                "integration_time_ms": 5000.0,
            }
        },
        timestamp="2026-02-24 10:00:00",
        distances_cm={"PRIMARY": 17.0},
    )

    _sid, session_path = session_writer.create_session_container(
        folder=measurements_dir,
        sample_id="SAMPLE_OLD_FMT",
        study_name="STUDY_OLD_FMT",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-24",
        container_id="bbbbbbbbbbbbbbbb",
    )
    session_writer.copy_technical_to_session(tech_path, session_path)

    session_writer.add_point(
        file_path=session_path,
        point_index=1,
        pixel_coordinates=[123.0, 456.0],
        physical_coordinates_mm=[1.25, 2.5],
    )
    session_writer.add_measurement(
        file_path=session_path,
        point_index=1,
        measurement_data={"DET-PRIMARY": np.arange(16).reshape(4, 4).astype(np.float32)},
        detector_metadata={"DET-PRIMARY": {"integration_time_ms": 5000.0}},
        poni_alias_map={"PRIMARY": "DET-PRIMARY"},
        raw_files={
            "DET-PRIMARY": {
                "raw_txt": b"1 2 3\n4 5 6\n",
                "raw_dsc": b"[F0]\nType=i16\n",
            }
        },
    )

    with h5py.File(session_path, "a") as h5f:
        h5f.attrs["meta_json"] = json.dumps({"from_state_file": True})

    return Path(session_path)


def test_export_session_to_old_format_creates_expected_layout(tmp_path):
    session_path = _create_session_with_technical_and_measurement(tmp_path)
    old_format_root = tmp_path / "Data" / "difra" / "Old_format"

    summary = SessionOldFormatExporter.export_from_session_container(
        session_path,
        config={"old_format_export_folder": str(old_format_root)},
    )

    assert summary.export_dir.exists() is True
    assert summary.export_dir.parent == old_format_root
    assert summary.state_path.exists() is True
    assert summary.raw_file_count >= 3  # npy + txt + dsc
    assert summary.technical_file_count >= 2  # poni + technical event data

    state = json.loads(summary.state_path.read_text(encoding="utf-8"))
    assert state.get("sample_id") == "SAMPLE_OLD_FMT"
    assert state.get("measurements_meta")

    root_files = {path.name for path in summary.export_dir.iterdir() if path.is_file()}
    assert any(name.endswith(".npy") for name in root_files)
    assert any(name.endswith(".txt") for name in root_files)
    assert any(name.endswith(".dsc") for name in root_files)

    technical_dir = summary.export_dir / "technical_measurements"
    assert technical_dir.exists() is True
    tech_files = {path.name for path in technical_dir.iterdir() if path.is_file()}
    assert any(name.endswith(".poni") for name in tech_files)
    assert any(name.endswith(".npy") for name in tech_files)
    assert "technical_meta_legacy.json" in tech_files
