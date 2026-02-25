"""Tests for high-level session lifecycle workflow actions."""

from pathlib import Path
from unittest.mock import patch

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
    old_format_folder = tmp_path / "Data" / "difra" / "Old_format"
    sid_a, path_a = _create_session_file(measurements, "SAMPLE_A")
    sid_b, path_b = _create_session_file(measurements, "SAMPLE_B")

    result = SessionLifecycleActions.send_and_archive_session_containers(
        container_paths=[path_a, path_b],
        container_manager=container_manager,
        archive_folder=archive_folder,
        active_session_path=path_a,
        lock_user="sad",
        session_ids={str(path_a): sid_a, str(path_b): sid_b},
        config={"old_format_export_folder": str(old_format_folder)},
    )

    assert result.failed == []
    assert result.moved == 2
    assert result.archived_active_session is True
    assert len(result.archived_paths) == 2
    assert all(path.exists() for path in result.archived_paths)
    assert result.old_format_failed == []
    assert len(result.old_format_paths) == 2
    assert all(path.exists() for path in result.old_format_paths)
    assert all(path.parent == old_format_folder for path in result.old_format_paths)
    assert path_a.exists() is False
    assert path_b.exists() is False
    parent_names = {p.parent.name for p in result.archived_paths}
    assert any(name.startswith(f"{sid_a}_") for name in parent_names)
    assert any(name.startswith(f"{sid_b}_") for name in parent_names)


def test_send_and_archive_cleans_measurement_artifacts(tmp_path):
    measurements = tmp_path / "measurements"
    archive_folder = tmp_path / "archive" / "measurements"
    old_format_folder = tmp_path / "Data" / "difra" / "Old_format"
    sid_a, path_a = _create_session_file(measurements, "SAMPLE_A")

    (measurements / "capture.txt").write_text("txt")
    (measurements / "capture.dsc").write_text("dsc")
    (measurements / "capture.npy").write_text("npy")
    (measurements / "SAMPLE_A_state.json").write_text("{}")
    grpc_folder = measurements / "grpc_exposures"
    grpc_folder.mkdir(parents=True, exist_ok=True)
    (grpc_folder / "tmp.txt").write_text("tmp")

    result = SessionLifecycleActions.send_and_archive_session_containers(
        container_paths=[path_a],
        container_manager=container_manager,
        archive_folder=archive_folder,
        lock_user="sad",
        session_ids={str(path_a): sid_a},
        config={"old_format_export_folder": str(old_format_folder)},
    )

    assert result.failed == []
    assert result.moved == 1
    archived_folder = result.archived_paths[0].parent
    assert (archived_folder / "capture.txt").exists() is True
    assert (archived_folder / "capture.dsc").exists() is True
    assert (archived_folder / "capture.npy").exists() is True
    assert (archived_folder / "SAMPLE_A_state.json").exists() is True
    assert (archived_folder / "grpc_exposures" / "tmp.txt").exists() is True
    assert (measurements / "capture.txt").exists() is False
    assert (measurements / "capture.dsc").exists() is False
    assert (measurements / "capture.npy").exists() is False
    assert (measurements / "SAMPLE_A_state.json").exists() is False
    assert grpc_folder.exists() is False


def test_send_and_archive_exports_old_format_before_archive_move(tmp_path):
    measurements = tmp_path / "measurements"
    archive_folder = tmp_path / "archive" / "measurements"
    sid_a, path_a = _create_session_file(measurements, "SAMPLE_A")

    call_order = []

    class _Summary:
        def __init__(self, export_dir: Path):
            self.export_dir = export_dir

    def _export_stub(session_path, **kwargs):
        call_order.append(("export", Path(session_path)))
        return _Summary(tmp_path / "old_format" / "mock")

    def _archive_stub(session_path, **kwargs):
        call_order.append(("archive", Path(session_path)))
        destination = (
            archive_folder
            / "mock_session_sad_SAMPLE_A_STUDY_A_20260225_100000"
            / Path(session_path).name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        Path(session_path).rename(destination)
        return destination

    with patch(
        "hardware.difra.gui.session_lifecycle_actions.SessionOldFormatExporter.export_from_session_container",
        side_effect=_export_stub,
    ), patch(
        "hardware.difra.gui.session_lifecycle_actions.SessionLifecycleService.archive_session_container",
        side_effect=_archive_stub,
    ):
        result = SessionLifecycleActions.send_and_archive_session_containers(
            container_paths=[path_a],
            container_manager=container_manager,
            archive_folder=archive_folder,
            lock_user="sad",
            session_ids={str(path_a): sid_a},
            config={"old_format_export_folder": str(tmp_path / "old_format")},
            export_old_format=True,
        )

    assert result.failed == []
    assert [item[0] for item in call_order] == ["export", "archive"]
    assert call_order[0][1] == path_a
