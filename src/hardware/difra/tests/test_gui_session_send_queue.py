"""GUI tests for Session tab send queue and archive list."""

import os
from pathlib import Path

import h5py
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QCheckBox, QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from hardware.container.v0_2 import schema, writer as session_writer
from hardware.container.v0_2.container_manager import is_container_locked
from hardware.difra.gui.main_window_ext.zone_measurements.session_tab_mixin import SessionTabMixin


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeSessionManager:
    def __init__(self):
        self.session_path = None
        self.sample_id = None
        self.session_id = None
        self.operator_id = "sad"
        self.study_name = None
        self.close_calls = 0

    def is_session_active(self):
        return self.session_path is not None

    def close_session(self):
        self.session_path = None
        self.sample_id = None
        self.session_id = None
        self.study_name = None
        self.close_calls += 1

    def get_session_info(self):
        if not self.is_session_active():
            return {"active": False}
        return {
            "active": True,
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "sample_id": self.sample_id or "UNKNOWN",
            "study_name": self.study_name or "UNSPECIFIED",
            "operator_id": self.operator_id,
            "machine_name": "DIFRA_TEST",
            "beam_energy_kev": 17.5,
            "is_locked": is_container_locked(Path(self.session_path)),
            "i0_recorded": False,
            "i_recorded": False,
            "attenuation_complete": False,
        }


class _SessionQueueHarness(QMainWindow, SessionTabMixin):
    def __init__(self, config, session_manager):
        super().__init__()
        self.config = config
        self.session_manager = session_manager
        self.status_updates = 0

        container = QWidget(self)
        layout = QVBoxLayout(container)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.setCentralWidget(container)

        self.create_session_tab()

    def update_session_status(self):
        self.status_updates += 1


def _create_session_file(folder: Path, sample: str, study: str) -> Path:
    _session_id, session_path = session_writer.create_session_container(
        folder=folder,
        sample_id=sample,
        study_name=study,
        operator_id="sad",
        site_id="ULSTER",
        machine_name="DIFRA_TEST",
        beam_energy_keV=17.5,
        acquisition_date="2026-02-13",
    )
    return Path(session_path)


def _row_checkbox(table, row):
    widget = table.cellWidget(row, 0)
    assert widget is not None
    checkbox = widget.findChild(QCheckBox)
    assert checkbox is not None
    return checkbox


def test_session_queue_send_selected_and_all(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.Ok))

    measurements_folder = tmp_path / "measurements"
    measurements_folder.mkdir(parents=True, exist_ok=True)
    archive_folder = tmp_path / "archive" / "measurements"
    old_format_folder = tmp_path / "Data" / "difra" / "Old_format"
    first_session = _create_session_file(measurements_folder, "SAMPLE_A", "STUDY_A")
    second_session = _create_session_file(measurements_folder, "SAMPLE_B", "STUDY_B")

    session_manager = _FakeSessionManager()
    session_manager.session_path = first_session
    session_manager.sample_id = "SAMPLE_A"
    session_manager.study_name = "STUDY_A"
    session_manager.session_id = "active"

    harness = _SessionQueueHarness(
        config={
            "measurements_folder": str(measurements_folder),
            "measurements_archive_folder": str(archive_folder),
            "old_format_export_folder": str(old_format_folder),
        },
        session_manager=session_manager,
    )
    harness.show()
    qapp.processEvents()

    harness._refresh_session_container_lists()
    assert harness.pending_sessions_table.rowCount() == 2
    assert harness.archived_sessions_table.rowCount() == 0

    # Send one selected container
    _row_checkbox(harness.pending_sessions_table, 0).setChecked(True)
    harness._on_send_selected_sessions()
    qapp.processEvents()

    # Selected one was moved + locked and removed from queue
    assert harness.pending_sessions_table.rowCount() == 1
    assert harness.archived_sessions_table.rowCount() == 1
    old_dirs = [path for path in old_format_folder.glob("*") if path.is_dir()]
    assert len(old_dirs) == 1
    archived_files = sorted(archive_folder.rglob("session_*.nxs.h5"))
    assert len(archived_files) == 1
    with h5py.File(archived_files[0], "r") as h5f:
        assert bool(h5f.attrs.get("locked", False)) is True
        assert h5f.attrs.get(schema.ATTR_SAMPLE_ID) in {"SAMPLE_A", "SAMPLE_B"}
        assert h5f.attrs.get(schema.ATTR_STUDY_NAME) in {"STUDY_A", "STUDY_B"}

    # Active session got closed if it was the one sent
    assert session_manager.close_calls in {0, 1}

    # Send remaining queue
    harness._on_send_all_sessions()
    qapp.processEvents()

    assert harness.pending_sessions_table.rowCount() == 0
    assert harness.archived_sessions_table.rowCount() == 2
    old_dirs = [path for path in old_format_folder.glob("*") if path.is_dir()]
    assert len(old_dirs) == 2
    archived_files = sorted(archive_folder.rglob("session_*.nxs.h5"))
    assert len(archived_files) == 2
    for archived in archived_files:
        with h5py.File(archived, "r") as h5f:
            assert bool(h5f.attrs.get("locked", False)) is True


def test_session_tab_close_finalize_active_session(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: QMessageBox.Ok))

    measurements_folder = tmp_path / "measurements"
    measurements_folder.mkdir(parents=True, exist_ok=True)
    archive_folder = tmp_path / "archive" / "measurements"
    old_format_folder = tmp_path / "Data" / "difra" / "Old_format"

    active_session = _create_session_file(measurements_folder, "SAMPLE_FINAL", "STUDY_FINAL")
    (measurements_folder / "SAMPLE_FINAL_state.json").write_text('{"demo": true}')
    (measurements_folder / "capture.txt").write_text("raw")
    (measurements_folder / "capture.npy").write_text("processed")

    session_manager = _FakeSessionManager()
    session_manager.session_path = active_session
    session_manager.sample_id = "SAMPLE_FINAL"
    session_manager.study_name = "STUDY_FINAL"
    session_manager.session_id = "session_final"

    harness = _SessionQueueHarness(
        config={
            "measurements_folder": str(measurements_folder),
            "measurements_archive_folder": str(archive_folder),
            "old_format_export_folder": str(old_format_folder),
        },
        session_manager=session_manager,
    )
    harness.show()
    qapp.processEvents()

    harness._on_close_finalize_session()
    qapp.processEvents()

    assert session_manager.close_calls == 1
    assert active_session.exists() is False
    archived_sessions = sorted(archive_folder.rglob("session_*.nxs.h5"))
    assert archived_sessions, "Expected archived session container after finalize"
    assert is_container_locked(archived_sessions[-1]) is True

    archived_dir = archived_sessions[-1].parent
    assert archived_dir.exists() is True
    assert (archived_dir / "SAMPLE_FINAL_state.json").exists() is True
    assert (archived_dir / "capture.txt").exists() is True
    assert (archived_dir / "capture.npy").exists() is True

    bundle_zip = archived_dir.with_suffix(".zip")
    assert bundle_zip.exists() is True
