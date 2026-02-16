"""Unit tests for Session tab presenter helpers."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QTableWidget

from hardware.container.v0_2 import schema, writer as session_writer
from hardware.container.v0_2.container_manager import is_container_locked, lock_container
from hardware.difra.gui.session_tab_presenter import SessionTabPresenter


class _ContainerManagerStub:
    @staticmethod
    def is_container_locked(path: Path) -> bool:
        return is_container_locked(Path(path))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


def test_build_pending_rows_reads_metadata_and_lock_status(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)

    unlocked = _create_session_file(measurements, "SAMPLE_A", "STUDY_A")
    locked = _create_session_file(measurements, "SAMPLE_B", "STUDY_B")
    lock_container(locked, user_id="sad")

    rows = SessionTabPresenter.build_pending_rows(
        measurements,
        schema=schema,
        container_manager=_ContainerManagerStub(),
    )

    assert len(rows) == 2
    by_sample = {row["sample_id"]: row for row in rows}
    assert by_sample["SAMPLE_A"]["status"] == "UNLOCKED"
    assert by_sample["SAMPLE_B"]["status"] == "LOCKED"
    assert by_sample["SAMPLE_A"]["path"] == str(unlocked)


def test_build_archived_rows_scans_nested_archive_tree(tmp_path):
    archive_root = tmp_path / "archive" / "measurements"
    archived_dir = archive_root / "SAMPLE_Z_20260213_120000"
    archived_dir.mkdir(parents=True, exist_ok=True)

    _create_session_file(archived_dir, "SAMPLE_Z", "STUDY_Z")

    rows = SessionTabPresenter.build_archived_rows(
        archive_root,
        schema=schema,
        container_manager=_ContainerManagerStub(),
    )

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "SAMPLE_Z"
    assert rows[0]["archived"]


def test_presenter_populates_pending_and_archive_tables(qapp):
    pending_table = QTableWidget()
    pending_table.setColumnCount(8)

    archive_table = QTableWidget()
    archive_table.setColumnCount(7)

    pending_rows = [
        {
            "file_name": "session_a.nxs.h5",
            "sample_id": "SAMPLE_A",
            "study_name": "STUDY_A",
            "operator_id": "sad",
            "created": "2026-02-13",
            "status": "LOCKED",
            "path": "/tmp/session_a.nxs.h5",
        }
    ]
    archive_rows = [
        {
            "file_name": "session_b.nxs.h5",
            "sample_id": "SAMPLE_B",
            "study_name": "STUDY_B",
            "operator_id": "sad",
            "created": "2026-02-13",
            "archived": "20260213_120000",
            "path": "/tmp/archive/session_b.nxs.h5",
        }
    ]

    SessionTabPresenter.populate_pending_table(pending_table, pending_rows)
    SessionTabPresenter.populate_archive_table(archive_table, archive_rows)

    assert pending_table.rowCount() == 1
    assert archive_table.rowCount() == 1

    pending_checkbox_widget = pending_table.cellWidget(0, 0)
    assert pending_checkbox_widget is not None
    assert pending_checkbox_widget.findChild(QCheckBox) is not None

    pending_file_item = pending_table.item(0, 1)
    assert pending_file_item.text() == "session_a.nxs.h5"
    assert pending_file_item.flags() == (Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    assert pending_table.item(0, 7).text() == "/tmp/session_a.nxs.h5"

    assert archive_table.item(0, 0).text() == "session_b.nxs.h5"
    assert archive_table.item(0, 5).text() == "20260213_120000"
    assert archive_table.item(0, 6).text() == "/tmp/archive/session_b.nxs.h5"


def test_build_active_session_view_state():
    inactive = SessionTabPresenter.build_active_session_view_state({"active": False})
    assert inactive.info_text == "No active session"
    assert inactive.close_enabled is False
    assert inactive.upload_enabled is False

    active_unlocked = SessionTabPresenter.build_active_session_view_state(
        {
            "active": True,
            "sample_id": "SAMPLE_X",
            "study_name": "STUDY_X",
            "session_id": "session_x",
            "operator_id": "sad",
            "session_path": "/tmp/session_x.nxs.h5",
            "is_locked": False,
        }
    )
    assert "SAMPLE_X" in active_unlocked.info_text
    assert active_unlocked.close_enabled is True
    assert active_unlocked.upload_enabled is False

    active_locked = SessionTabPresenter.build_active_session_view_state(
        {
            "active": True,
            "sample_id": "SAMPLE_X",
            "study_name": "STUDY_X",
            "session_id": "session_x",
            "operator_id": "sad",
            "session_path": "/tmp/session_x.nxs.h5",
            "is_locked": True,
        }
    )
    assert active_locked.close_enabled is False
    assert active_locked.upload_enabled is True
