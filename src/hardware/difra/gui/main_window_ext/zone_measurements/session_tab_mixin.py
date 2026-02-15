"""Session management tab for Zone Measurements."""

import shutil
import time
from pathlib import Path
from typing import Dict, List

import h5py
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hardware.difra.gui.container_api import get_container_manager, get_schema
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionTabMixin:
    """Mixin for session management tab in Zone Measurements."""

    def _container_schema(self):
        return get_schema(self.config if hasattr(self, "config") else None)

    def _container_manager(self):
        return get_container_manager(self.config if hasattr(self, "config") else None)

    def create_session_tab(self):
        """Create session management tab with queue and archive views."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info_group = QGroupBox("Active Session Information")
        info_layout = QVBoxLayout(info_group)
        self.session_info_label = QLabel("No active session")
        self.session_info_label.setStyleSheet("padding: 10px;")
        info_layout.addWidget(self.session_info_label)
        layout.addWidget(info_group)

        actions_group = QGroupBox("Active Session Actions")
        actions_layout = QVBoxLayout(actions_group)

        close_layout = QHBoxLayout()
        self.close_session_btn = QPushButton("Close && Finalize Active Session")
        self.close_session_btn.setToolTip(
            "Lock active session container and archive measurement files."
        )
        self.close_session_btn.clicked.connect(self._on_close_finalize_session)
        self.close_session_btn.setEnabled(False)
        close_layout.addWidget(self.close_session_btn)
        actions_layout.addLayout(close_layout)

        upload_layout = QHBoxLayout()
        self.upload_session_btn = QPushButton("Upload Active Session (Fake)")
        self.upload_session_btn.setToolTip(
            "Fake upload for currently active session. Use queue actions for batch sending."
        )
        self.upload_session_btn.clicked.connect(self._on_upload_session)
        self.upload_session_btn.setEnabled(False)
        upload_layout.addWidget(self.upload_session_btn)
        actions_layout.addLayout(upload_layout)

        layout.addWidget(actions_group)

        queue_group = QGroupBox("Session Containers Ready To Close/Send")
        queue_layout = QVBoxLayout(queue_group)

        queue_btn_layout = QHBoxLayout()
        self.refresh_sessions_btn = QPushButton("Refresh")
        self.refresh_sessions_btn.clicked.connect(self._refresh_session_container_lists)
        queue_btn_layout.addWidget(self.refresh_sessions_btn)

        self.select_all_sessions_btn = QPushButton("Select All")
        self.select_all_sessions_btn.clicked.connect(
            lambda: self._set_all_pending_selection(True)
        )
        queue_btn_layout.addWidget(self.select_all_sessions_btn)

        self.clear_sessions_selection_btn = QPushButton("Clear Selection")
        self.clear_sessions_selection_btn.clicked.connect(
            lambda: self._set_all_pending_selection(False)
        )
        queue_btn_layout.addWidget(self.clear_sessions_selection_btn)

        self.send_selected_sessions_btn = QPushButton("Close && Send Selected")
        self.send_selected_sessions_btn.clicked.connect(self._on_send_selected_sessions)
        queue_btn_layout.addWidget(self.send_selected_sessions_btn)

        self.send_all_sessions_btn = QPushButton("Close && Send All")
        self.send_all_sessions_btn.clicked.connect(self._on_send_all_sessions)
        queue_btn_layout.addWidget(self.send_all_sessions_btn)
        queue_layout.addLayout(queue_btn_layout)

        self.pending_sessions_table = QTableWidget()
        self.pending_sessions_table.setColumnCount(8)
        self.pending_sessions_table.setHorizontalHeaderLabels(
            [
                "Select",
                "File",
                "Sample",
                "Study",
                "Operator",
                "Created",
                "Status",
                "Path",
            ]
        )
        self.pending_sessions_table.setColumnHidden(7, True)
        queue_layout.addWidget(self.pending_sessions_table)

        layout.addWidget(queue_group)

        archive_group = QGroupBox("Archived Session Containers")
        archive_layout = QVBoxLayout(archive_group)

        self.archive_path_label = QLabel("")
        self.archive_path_label.setStyleSheet("color: #555; padding: 4px;")
        archive_layout.addWidget(self.archive_path_label)

        self.archived_sessions_table = QTableWidget()
        self.archived_sessions_table.setColumnCount(7)
        self.archived_sessions_table.setHorizontalHeaderLabels(
            ["File", "Sample", "Study", "Operator", "Created", "Archived", "Path"]
        )
        self.archived_sessions_table.setColumnHidden(6, True)
        archive_layout.addWidget(self.archived_sessions_table)

        layout.addWidget(archive_group)
        layout.addStretch()

        if hasattr(self, "tabs"):
            self.tabs.addTab(tab, "Session")

        self._update_session_tab_info()
        self._refresh_session_container_lists()

    @staticmethod
    def _decode_attr(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def _get_measurements_folder_for_queue(self) -> Path:
        if hasattr(self, "config") and self.config:
            folder = self.config.get("measurements_folder") or self.config.get(
                "session_folder"
            )
            if folder:
                return Path(folder)

        if hasattr(self, "folderLineEdit"):
            folder = (self.folderLineEdit.text() or "").strip()
            if folder:
                return Path(folder)

        if (
            hasattr(self, "session_manager")
            and self.session_manager
            and getattr(self.session_manager, "session_path", None)
        ):
            return Path(self.session_manager.session_path).parent

        return Path.home() / "difra_measurements"

    def _get_session_archive_folder(self) -> Path:
        if hasattr(self, "config") and self.config:
            archive = self.config.get("measurements_archive_folder")
            if archive:
                return Path(archive)
            archive = self.config.get("session_archive_folder")
            if archive:
                return Path(archive)

        measurements_folder = self._get_measurements_folder_for_queue()
        return measurements_folder.parent / "archive" / "measurements"

    def _scan_pending_session_containers(self) -> List[Path]:
        measurements_folder = self._get_measurements_folder_for_queue()
        if not measurements_folder.exists():
            return []
        return sorted(
            [path for path in measurements_folder.glob("session_*.nxs.h5") if path.is_file()]
        )

    def _scan_archived_session_containers(self) -> List[Path]:
        archive_folder = self._get_session_archive_folder()
        if not archive_folder.exists():
            return []
        return sorted(
            [path for path in archive_folder.rglob("session_*.nxs.h5") if path.is_file()]
        )

    def _read_session_container_metadata(self, container_path: Path) -> Dict[str, str]:
        info: Dict[str, str] = {
            "file_name": container_path.name,
            "path": str(container_path),
            "sample_id": "UNKNOWN",
            "study_name": "UNSPECIFIED",
            "operator_id": "UNKNOWN",
            "created": "",
            "status": "UNKNOWN",
            "session_id": "",
            "archived": "",
        }

        try:
            schema = self._container_schema()
            with h5py.File(container_path, "r") as h5f:
                info["sample_id"] = str(
                    self._decode_attr(h5f.attrs.get(schema.ATTR_SAMPLE_ID, "UNKNOWN"))
                )
                info["study_name"] = str(
                    self._decode_attr(
                        h5f.attrs.get(schema.ATTR_STUDY_NAME, "UNSPECIFIED")
                    )
                )
                info["operator_id"] = str(
                    self._decode_attr(h5f.attrs.get(schema.ATTR_OPERATOR_ID, "UNKNOWN"))
                )
                info["created"] = str(
                    self._decode_attr(
                        h5f.attrs.get(schema.ATTR_CREATION_TIMESTAMP, "")
                    )
                )
                info["session_id"] = str(
                    self._decode_attr(h5f.attrs.get(schema.ATTR_SESSION_ID, ""))
                )
                locked = self._container_manager().is_container_locked(container_path)
                info["status"] = "LOCKED" if locked else "UNLOCKED"
        except Exception as exc:
            info["status"] = f"ERROR ({exc})"

        try:
            parent_name = container_path.parent.name
            if "_" in parent_name:
                info["archived"] = parent_name.rsplit("_", 1)[-1]
        except Exception:
            pass
        if not info["archived"]:
            info["archived"] = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(container_path.stat().st_mtime)
            )

        return info

    def _populate_pending_table(self, containers: List[Path]):
        self.pending_sessions_table.setRowCount(0)
        for row, container_path in enumerate(containers):
            info = self._read_session_container_metadata(container_path)
            self.pending_sessions_table.insertRow(row)

            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.addWidget(checkbox)
            self.pending_sessions_table.setCellWidget(row, 0, checkbox_widget)

            for col, key in enumerate(
                ["file_name", "sample_id", "study_name", "operator_id", "created", "status"],
                start=1,
            ):
                item = QTableWidgetItem(str(info.get(key, "")))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.pending_sessions_table.setItem(row, col, item)

            path_item = QTableWidgetItem(str(container_path))
            path_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.pending_sessions_table.setItem(row, 7, path_item)

    def _populate_archive_table(self, containers: List[Path]):
        self.archived_sessions_table.setRowCount(0)
        for row, container_path in enumerate(containers):
            info = self._read_session_container_metadata(container_path)
            self.archived_sessions_table.insertRow(row)
            for col, key in enumerate(
                ["file_name", "sample_id", "study_name", "operator_id", "created", "archived"]
            ):
                item = QTableWidgetItem(str(info.get(key, "")))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.archived_sessions_table.setItem(row, col, item)

            path_item = QTableWidgetItem(str(container_path))
            path_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.archived_sessions_table.setItem(row, 6, path_item)

    def _refresh_session_container_lists(self):
        if not hasattr(self, "pending_sessions_table") or not hasattr(
            self, "archived_sessions_table"
        ):
            return

        pending = self._scan_pending_session_containers()
        archived = self._scan_archived_session_containers()
        self._populate_pending_table(pending)
        self._populate_archive_table(archived)

        archive_folder = self._get_session_archive_folder()
        self.archive_path_label.setText(f"Archive folder: {archive_folder}")

    def _set_all_pending_selection(self, checked: bool):
        for row in range(self.pending_sessions_table.rowCount()):
            checkbox_widget = self.pending_sessions_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(checked)

    def _selected_pending_containers(self) -> List[Path]:
        selected: List[Path] = []
        for row in range(self.pending_sessions_table.rowCount()):
            checkbox_widget = self.pending_sessions_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox) if checkbox_widget else None
            if checkbox is None or not checkbox.isChecked():
                continue
            path_item = self.pending_sessions_table.item(row, 7)
            if path_item is None:
                continue
            selected.append(Path(path_item.text()))
        return selected

    def _all_pending_containers(self) -> List[Path]:
        containers: List[Path] = []
        for row in range(self.pending_sessions_table.rowCount()):
            path_item = self.pending_sessions_table.item(row, 7)
            if path_item is not None:
                containers.append(Path(path_item.text()))
        return containers

    def _send_and_archive_sessions(self, container_paths: List[Path]):
        if not container_paths:
            QMessageBox.information(self, "No Containers", "No session containers selected.")
            return

        archive_folder = self._get_session_archive_folder()
        archive_folder.mkdir(parents=True, exist_ok=True)

        moved = 0
        failed = []

        for container_path in container_paths:
            try:
                if not container_path.exists():
                    continue

                info = self._read_session_container_metadata(container_path)
                was_active = False
                if (
                    hasattr(self, "session_manager")
                    and self.session_manager
                    and getattr(self.session_manager, "session_path", None)
                ):
                    active_path = Path(self.session_manager.session_path)
                    was_active = active_path.resolve() == container_path.resolve()

                container_manager = self._container_manager()
                if not container_manager.is_container_locked(container_path):
                    lock_user = None
                    if hasattr(self, "session_manager") and self.session_manager:
                        lock_user = getattr(self.session_manager, "operator_id", None)
                    container_manager.lock_container(container_path, user_id=lock_user)

                # Fake cloud send for development mode: keep explicit log marker,
                # but apply real post-send lifecycle (lock + archive move).
                logger.info(
                    "Fake cloud send completed",
                    session_path=str(container_path),
                    sample_id=info.get("sample_id"),
                )

                session_id = info.get("session_id") or container_path.stem
                archive_stamp = time.strftime("%Y%m%d_%H%M%S")
                session_archive_dir = archive_folder / f"{session_id}_{archive_stamp}"
                session_archive_dir.mkdir(parents=True, exist_ok=True)
                destination = session_archive_dir / container_path.name
                shutil.move(str(container_path), str(destination))
                moved += 1

                if was_active:
                    self.session_manager.close_session()
            except Exception as exc:
                failed.append(f"{container_path.name}: {exc}")

        summary = [f"Sent+archived {moved} session container(s)."]
        if failed:
            summary.append("")
            summary.append("Failures:")
            summary.extend(failed[:8])
            if len(failed) > 8:
                summary.append(f"... and {len(failed) - 8} more")

        QMessageBox.information(self, "Session Send Queue", "\n".join(summary))
        self._refresh_session_container_lists()
        if hasattr(self, "update_session_status"):
            self.update_session_status()

    def _on_send_selected_sessions(self):
        selected = self._selected_pending_containers()
        if not selected:
            QMessageBox.warning(
                self,
                "No Selection",
                "Select one or more session containers from the queue.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Close && Send Selected",
            f"Close, fake-send, and archive {len(selected)} selected session container(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._send_and_archive_sessions(selected)

    def _on_send_all_sessions(self):
        all_containers = self._all_pending_containers()
        if not all_containers:
            QMessageBox.information(
                self, "Queue Empty", "No session containers found in measurements folder."
            )
            return

        reply = QMessageBox.question(
            self,
            "Close && Send All",
            f"Close, fake-send, and archive ALL {len(all_containers)} queued session container(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._send_and_archive_sessions(all_containers)

    def _update_session_tab_info(self):
        """Update active-session info and button states."""
        if not hasattr(self, "session_manager") or not hasattr(
            self, "session_info_label"
        ):
            return

        info = self.session_manager.get_session_info()

        if info["active"]:
            info_text = f"<b>Sample ID:</b> {info['sample_id']}<br>"
            info_text += f"<b>Study:</b> {info.get('study_name', 'UNSPECIFIED')}<br>"
            info_text += f"<b>Session ID:</b> {info['session_id']}<br>"
            info_text += f"<b>Operator:</b> {info['operator_id']}<br>"
            info_text += f"<b>Container:</b> {Path(info['session_path']).name}<br>"
            info_text += (
                f"<b>Status:</b> {'🔒 Locked' if info['is_locked'] else '🔓 Unlocked'}"
            )
            self.session_info_label.setText(info_text)
            is_locked = info["is_locked"]
            self.close_session_btn.setEnabled(not is_locked)
            self.upload_session_btn.setEnabled(is_locked)
        else:
            self.session_info_label.setText("No active session")
            self.close_session_btn.setEnabled(False)
            self.upload_session_btn.setEnabled(False)

        self._refresh_session_container_lists()

    def _on_close_finalize_session(self):
        """Close and finalize the active session container and archive measurement files."""
        if not hasattr(self, "session_manager") or not self.session_manager.is_session_active():
            QMessageBox.warning(self, "No Active Session", "No session is currently active.")
            return

        info = self.session_manager.get_session_info()
        reply = QMessageBox.question(
            self,
            "Close and Finalize Session?",
            f"Close and finalize session '{info['sample_id']}'?\n\n"
            f"This will:\n"
            f"• Lock the session container (read-only)\n"
            f"• Archive measurement files\n"
            f"• Close the active session\n\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            session_path = Path(info["session_path"])
            measurements_folder = session_path.parent

            self._store_json_state_in_container(
                session_path, measurements_folder, info["sample_id"]
            )

            logger.info("Locking session container", session_path=str(session_path))
            self._container_manager().lock_container(session_path)

            archive_dest, archived_count = self._archive_measurement_files(
                measurements_folder, info["sample_id"]
            )
            bundle_path = self._create_session_bundle_zip(session_path, archive_dest)

            self.session_manager.close_session()

            details = [
                f"Session '{info['sample_id']}' has been finalized.",
                "",
                f"Container: {session_path.name}",
                f"Archived files: {archived_count}",
                f"Archive folder: {archive_dest}",
            ]
            if bundle_path:
                details.append(f"ZIP bundle: {bundle_path}")

            QMessageBox.information(self, "Session Finalized", "\n".join(details))
            logger.info("Session finalized and closed", sample_id=info["sample_id"])

            self._update_session_tab_info()
            if hasattr(self, "update_session_status"):
                self.update_session_status()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Finalization Failed",
                f"Failed to finalize session:\n\n{str(exc)}",
            )
            logger.error(f"Failed to finalize session: {exc}", exc_info=True)

    def _store_json_state_in_container(
        self, session_path: Path, measurements_folder: Path, sample_id: str
    ):
        """Store JSON state file in session container as attribute."""
        import json

        state_file = measurements_folder / f"{sample_id}_state.json"
        if not state_file.exists():
            logger.warning(f"State JSON file not found: {state_file}")
            return

        try:
            with open(state_file, "r") as file_handle:
                state_data = json.load(file_handle)
            with h5py.File(session_path, "a") as h5f:
                h5f.attrs["meta_json"] = json.dumps(state_data)
            logger.info(
                "Stored state JSON in container",
                session_path=str(session_path),
                state_file=str(state_file),
            )
        except Exception as exc:
            logger.error(f"Failed to store state JSON in container: {exc}", exc_info=True)

    def _archive_measurement_files(self, measurements_folder: Path, sample_id: str):
        """Archive raw and NPY measurement files, including state JSON."""
        from fnmatch import fnmatch

        if hasattr(self, "config") and self.config:
            archive_base = self.config.get("measurements_archive_folder")
            if archive_base:
                archive_folder = Path(archive_base)
            else:
                archive_folder = measurements_folder.parent / "archive" / "measurements"
        else:
            archive_folder = measurements_folder.parent / "archive" / "measurements"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_dest = archive_folder / f"{sample_id}_{timestamp}"
        archive_dest.mkdir(parents=True, exist_ok=True)

        patterns = ["*.txt", "*.dsc", "*.npy", "*.t3pa", "*_state.json"]
        archived_count = 0
        for file_path in sorted(measurements_folder.rglob("*")):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(measurements_folder)
            relative_str = relative_path.as_posix()
            matches_pattern = any(
                fnmatch(file_path.name, pattern) or fnmatch(relative_str, pattern)
                for pattern in patterns
            )
            if not matches_pattern:
                continue

            try:
                dest_path = archive_dest / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(dest_path))
                archived_count += 1
            except Exception as exc:
                logger.warning(f"Failed to archive {relative_path}: {exc}")

        logger.info(
            f"Archived {archived_count} measurement files",
            archive_folder=str(archive_dest),
        )
        return archive_dest, archived_count

    def _create_session_bundle_zip(self, session_path: Path, archive_folder: Path):
        """Create ZIP bundle with locked session container and archived files."""
        from hardware.container import create_container_bundle

        try:
            output_zip = archive_folder.with_suffix(".zip")
            bundle_path = create_container_bundle(
                container_file=session_path,
                source_folder=archive_folder,
                output_zip=output_zip,
                source_arcname=archive_folder.name,
            )
            logger.info("Created session ZIP bundle", bundle_path=str(bundle_path))
            return bundle_path
        except Exception as exc:
            logger.warning(f"Failed to create session ZIP bundle: {exc}")
            return None

    def _on_upload_session(self):
        """Fake upload action for currently active session."""
        if not hasattr(self, "session_manager") or not self.session_manager.is_session_active():
            QMessageBox.warning(self, "No Active Session", "No session is currently active.")
            return

        info = self.session_manager.get_session_info()
        if not info["is_locked"]:
            QMessageBox.warning(
                self,
                "Session Not Finalized",
                "Session must be closed and finalized before uploading.",
            )
            return

        QMessageBox.information(
            self,
            "Upload to Cloud (Fake)",
            f"Fake cloud upload executed for active session '{info['sample_id']}'.\n\n"
            f"Use 'Close && Send Selected/All' in the queue for archival transfer.",
        )
        logger.info("Cloud upload requested", sample_id=info["sample_id"])
