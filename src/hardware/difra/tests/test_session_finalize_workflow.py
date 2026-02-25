"""Tests for active-session finalization workflow service."""

import json
from pathlib import Path

import h5py

from hardware.container.v0_2 import container_manager, writer as session_writer
from hardware.difra.gui.session_finalize_workflow import SessionFinalizeWorkflow


def _create_session_file(folder: Path, sample_id: str = "SAMPLE_A"):
    session_id, session_path = session_writer.create_session_container(
        folder=folder,
        sample_id=sample_id,
        study_name="STUDY_A",
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-16",
    )
    return session_id, Path(session_path)


def test_store_json_state_in_container_embeds_meta_json(tmp_path):
    measurements = tmp_path / "measurements"
    _sid, session_path = _create_session_file(measurements, "SAMPLE_META")

    state_path = measurements / "SAMPLE_META_state.json"
    payload = {"shapes": [{"id": 1}], "zone_points": [{"id": 2}]}
    state_path.write_text(json.dumps(payload))

    stored = SessionFinalizeWorkflow.store_json_state_in_container(
        session_path=session_path,
        measurements_folder=measurements,
        sample_id="SAMPLE_META",
    )
    assert stored is True

    with h5py.File(session_path, "r") as h5f:
        assert "meta_json" in h5f.attrs
        decoded = json.loads(h5f.attrs["meta_json"])
    assert decoded == payload


def test_archive_measurement_files_moves_only_matching_patterns(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)

    (measurements / "sample.txt").write_text("txt")
    (measurements / "sample.dsc").write_text("dsc")
    (measurements / "sample.npy").write_text("npy")
    (measurements / "sample.t3pa").write_text("t3pa")
    (measurements / "sample.poni").write_text("poni")
    (measurements / "SAMPLE_PAT_state.json").write_text("{}")
    (measurements / "keep.md").write_text("keep")
    nested = measurements / "nested"
    nested.mkdir()
    (nested / "nested.txt").write_text("nested")

    archive_dest, archived_count = SessionFinalizeWorkflow.archive_measurement_files(
        measurements_folder=measurements,
        sample_id="SAMPLE_PAT",
        study_name="STUDY_A",
        operator_id="sad",
        config={"measurements_archive_folder": str(tmp_path / "archive" / "measurements")},
    )

    assert archive_dest.exists() is True
    assert archived_count == 7
    assert (measurements / "keep.md").exists() is True
    assert (archive_dest / "sample.txt").exists() is True
    assert (archive_dest / "sample.dsc").exists() is True
    assert (archive_dest / "sample.npy").exists() is True
    assert (archive_dest / "sample.t3pa").exists() is True
    assert (archive_dest / "sample.poni").exists() is True
    assert (archive_dest / "SAMPLE_PAT_state.json").exists() is True
    assert (archive_dest / "nested" / "nested.txt").exists() is True
    assert archive_dest.name.startswith("session_sad_SAMPLE_PAT_STUDY_A_")


def test_finalize_session_runs_lock_archive_and_bundle(tmp_path):
    measurements = tmp_path / "measurements"
    session_id, session_path = _create_session_file(measurements, "SAMPLE_FINAL")

    (measurements / "SAMPLE_FINAL_state.json").write_text('{"meta": true}')
    (measurements / "raw.txt").write_text("raw")

    result = SessionFinalizeWorkflow.finalize_session(
        session_path=session_path,
        measurements_folder=measurements,
        sample_id="SAMPLE_FINAL",
        container_manager=container_manager,
        lock_user="sad",
        config={"measurements_archive_folder": str(tmp_path / "archive" / "measurements")},
    )

    assert result.session_path.parent == result.archive_dest
    assert result.session_path.name == session_path.name
    assert result.state_json_embedded is True
    assert result.archive_dest.exists() is True
    assert result.archived_count == 2
    assert result.archive_dest.name.startswith(
        f"{session_id}_sad_SAMPLE_FINAL_STUDY_A_"
    )
    assert session_path.exists() is False
    assert container_manager.is_container_locked(result.session_path) is True
    assert result.bundle_path is not None
    assert result.bundle_path.exists() is True
