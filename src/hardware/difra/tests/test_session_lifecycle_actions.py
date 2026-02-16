"""Tests for high-level session lifecycle workflow actions."""

from pathlib import Path

from hardware.container.v0_2 import container_manager, writer as session_writer
from hardware.difra.gui.session_lifecycle_actions import SessionLifecycleActions


def _create_session_file(folder: Path, sample_id: str):
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


def test_finalize_session_container_locks_once(tmp_path):
    _sid, session_path = _create_session_file(tmp_path / "measurements", "SAMPLE_A")

    changed = SessionLifecycleActions.finalize_session_container(
        session_path=session_path,
        container_manager=container_manager,
        lock_user="sad",
    )
    assert changed is True
    assert container_manager.is_container_locked(session_path) is True

    changed_again = SessionLifecycleActions.finalize_session_container(
        session_path=session_path,
        container_manager=container_manager,
        lock_user="sad",
    )
    assert changed_again is False


def test_send_and_archive_session_containers_tracks_active_session(tmp_path):
    measurements = tmp_path / "measurements"
    archive_folder = tmp_path / "archive" / "measurements"
    sid_a, path_a = _create_session_file(measurements, "SAMPLE_A")
    sid_b, path_b = _create_session_file(measurements, "SAMPLE_B")

    result = SessionLifecycleActions.send_and_archive_session_containers(
        container_paths=[path_a, path_b],
        container_manager=container_manager,
        archive_folder=archive_folder,
        active_session_path=path_a,
        lock_user="sad",
        session_ids={str(path_a): sid_a, str(path_b): sid_b},
    )

    assert result.failed == []
    assert result.moved == 2
    assert result.archived_active_session is True
    assert len(result.archived_paths) == 2
    assert all(path.exists() for path in result.archived_paths)
    assert path_a.exists() is False
    assert path_b.exists() is False
    parent_names = {p.parent.name for p in result.archived_paths}
    assert any(name.startswith(f"{sid_a}_") for name in parent_names)
    assert any(name.startswith(f"{sid_b}_") for name in parent_names)
