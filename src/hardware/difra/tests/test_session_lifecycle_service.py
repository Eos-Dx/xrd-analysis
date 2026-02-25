"""Tests for shared session lifecycle helper service."""

from pathlib import Path

from hardware.container.v0_2 import writer as session_writer
from hardware.container.v0_2.container_manager import (
    is_container_locked,
    lock_container,
)
from hardware.difra.gui.session_lifecycle_service import SessionLifecycleService


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


def test_resolve_archive_folder_prefers_configured_measurements_archive(tmp_path):
    configured = tmp_path / "cfg_archive" / "measurements"
    resolved = SessionLifecycleService.resolve_archive_folder(
        config={"measurements_archive_folder": str(configured)},
        measurements_folder=tmp_path / "measurements",
    )
    assert resolved == configured


def test_resolve_archive_folder_fallback_uses_measurements_parent(tmp_path):
    measurements = tmp_path / "measurements"
    resolved = SessionLifecycleService.resolve_archive_folder(
        config={},
        measurements_folder=measurements,
    )
    assert resolved == tmp_path / "archive" / "measurements"


def test_resolve_archive_folder_fallback_uses_session_path(tmp_path):
    session_path = tmp_path / "measurements" / "session_demo.nxs.h5"
    resolved = SessionLifecycleService.resolve_archive_folder(
        config={},
        session_path=session_path,
    )
    assert resolved == tmp_path / "archive" / "measurements"


def test_lock_container_if_needed_is_idempotent(tmp_path):
    sessions = tmp_path / "measurements"
    session_id, session_path = _create_session_file(sessions)

    from hardware.container.v0_2 import container_manager

    changed = SessionLifecycleService.lock_container_if_needed(
        container_path=session_path,
        container_manager=container_manager,
        user_id="sad",
    )
    assert changed is True
    assert is_container_locked(session_path) is True

    changed_second = SessionLifecycleService.lock_container_if_needed(
        container_path=session_path,
        container_manager=container_manager,
        user_id="sad",
    )
    assert changed_second is False


def test_archive_session_container_moves_file_to_timestamped_folder(tmp_path):
    sessions = tmp_path / "measurements"
    archive_root = tmp_path / "archive" / "measurements"
    session_id, session_path = _create_session_file(sessions, sample_id="SAMPLE_Z")

    lock_container(session_path, user_id="sad")

    destination = SessionLifecycleService.archive_session_container(
        session_path=session_path,
        session_id=session_id,
        archive_folder=archive_root,
        timestamp="20260216_123000",
    )

    assert session_path.exists() is False
    assert destination.exists() is True
    assert destination.name.startswith("session_")
    assert (
        destination.parent.name
        == f"{session_id}_sad_SAMPLE_Z_STUDY_A_20260216_123000"
    )
