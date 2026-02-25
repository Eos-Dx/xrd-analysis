"""Session management tab for Zone Measurements."""

from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from hardware.difra.gui.container_api import get_container_manager, get_schema
from hardware.difra.gui.session_finalize_workflow import SessionFinalizeWorkflow
from hardware.difra.gui.session_lifecycle_actions import SessionLifecycleActions
from hardware.difra.gui.session_lifecycle_service import SessionLifecycleService
from hardware.difra.gui.session_tab_presenter import SessionTabPresenter
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

        self.load_session_from_path_btn = QPushButton("Load Container…")
        self.load_session_from_path_btn.clicked.connect(
            self._on_load_session_container_from_dialog
        )
        queue_btn_layout.addWidget(self.load_session_from_path_btn)

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
        self.pending_sessions_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pending_sessions_table.customContextMenuRequested.connect(
            self._show_pending_sessions_context_menu
        )
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
        self.archived_sessions_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.archived_sessions_table.customContextMenuRequested.connect(
            self._show_archived_sessions_context_menu
        )
        archive_layout.addWidget(self.archived_sessions_table)

        layout.addWidget(archive_group)
        layout.addStretch()

        if hasattr(self, "tabs"):
            self.tabs.addTab(tab, "Session")

        self._update_session_tab_info()
        self._refresh_session_container_lists()

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
        measurements_folder = self._get_measurements_folder_for_queue()
        return SessionLifecycleService.resolve_archive_folder(
            config=self.config if hasattr(self, "config") else None,
            measurements_folder=measurements_folder,
        )

    def _refresh_session_container_lists(self):
        if not hasattr(self, "pending_sessions_table") or not hasattr(
            self, "archived_sessions_table"
        ):
            return

        schema = self._container_schema()
        container_manager = self._container_manager()
        pending_rows = SessionTabPresenter.build_pending_rows(
            self._get_measurements_folder_for_queue(),
            schema=schema,
            container_manager=container_manager,
        )
        archived_rows = SessionTabPresenter.build_archived_rows(
            self._get_session_archive_folder(),
            schema=schema,
            container_manager=container_manager,
        )
        SessionTabPresenter.populate_pending_table(self.pending_sessions_table, pending_rows)
        SessionTabPresenter.populate_archive_table(self.archived_sessions_table, archived_rows)

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

    def _path_from_table_row(self, table: QTableWidget, row: int, path_col: int):
        if row < 0:
            return None
        path_item = table.item(row, path_col)
        if path_item is None:
            return None
        raw = (path_item.text() or "").strip()
        if not raw:
            return None
        return Path(raw)

    def _open_session_container_path(self, container_path: Path):
        if container_path is None:
            return
        if not container_path.exists():
            QMessageBox.warning(
                self,
                "Container Missing",
                f"Session container not found:\n{container_path}",
            )
            return
        if hasattr(self, "load_session_container_from_path"):
            self.load_session_container_from_path(container_path)
        else:
            QMessageBox.warning(
                self,
                "Load Not Available",
                "Session loading API is not available in this window build.",
            )

    def _show_pending_sessions_context_menu(self, pos):
        table = self.pending_sessions_table
        row = table.rowAt(pos.y())
        if row < 0:
            return
        container_path = self._path_from_table_row(table, row, 7)
        if container_path is None:
            return

        menu = QMenu(table)
        load_action = menu.addAction("Load Container")
        selected = menu.exec_(table.viewport().mapToGlobal(pos))
        if selected == load_action:
            self._open_session_container_path(container_path)

    def _show_archived_sessions_context_menu(self, pos):
        table = self.archived_sessions_table
        row = table.rowAt(pos.y())
        if row < 0:
            return
        container_path = self._path_from_table_row(table, row, 6)
        if container_path is None:
            return

        menu = QMenu(table)
        load_action = menu.addAction("Load Container")
        selected = menu.exec_(table.viewport().mapToGlobal(pos))
        if selected == load_action:
            self._open_session_container_path(container_path)

    def _on_load_session_container_from_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session Container",
            str(self._get_measurements_folder_for_queue()),
            "NeXus HDF5 Files (*.nxs.h5 *.h5);;All Files (*)",
        )
        if not file_path:
            return
        self._open_session_container_path(Path(file_path))

    def _send_and_archive_sessions(self, container_paths: List[Path]):
        if not container_paths:
            QMessageBox.information(self, "No Containers", "No session containers selected.")
            return

        schema = self._container_schema()
        container_manager = self._container_manager()
        archive_folder = self._get_session_archive_folder()
        archive_folder.mkdir(parents=True, exist_ok=True)

        active_session_path = None
        if (
            hasattr(self, "session_manager")
            and self.session_manager
            and getattr(self.session_manager, "session_path", None)
        ):
            active_session_path = Path(self.session_manager.session_path)
        batch_session_ids = {}

        for container_path in container_paths:
            if not Path(container_path).exists():
                continue
            info = SessionTabPresenter.read_session_container_metadata(
                Path(container_path),
                schema=schema,
                container_manager=container_manager,
            )
            logger.info(
                "Fake cloud send completed",
                session_path=str(container_path),
                sample_id=info.get("sample_id"),
            )
            batch_session_ids[str(Path(container_path))] = (
                info.get("session_id") or Path(container_path).stem
            )

        lock_user = None
        if hasattr(self, "session_manager") and self.session_manager:
            lock_user = getattr(self.session_manager, "operator_id", None)

        workflow_result = SessionLifecycleActions.send_and_archive_session_containers(
            container_paths=container_paths,
            container_manager=container_manager,
            archive_folder=archive_folder,
            active_session_path=active_session_path,
            lock_user=lock_user,
            session_ids=batch_session_ids,
            config=self.config if hasattr(self, "config") else None,
        )

        if workflow_result.archived_active_session and hasattr(self, "session_manager"):
            self.session_manager.close_session()

        summary = [f"Sent+archived {workflow_result.moved} session container(s)."]
        summary.append(f"Cleaned measurement artifacts: {workflow_result.cleaned_artifacts}")
        summary.append(f"Old-format exports: {len(workflow_result.old_format_paths)}")
        if workflow_result.old_format_paths:
            summary.append(f"Old-format folder: {workflow_result.old_format_paths[-1]}")
        if workflow_result.failed:
            summary.append("")
            summary.append("Failures:")
            summary.extend(workflow_result.failed[:8])
            if len(workflow_result.failed) > 8:
                summary.append(f"... and {len(workflow_result.failed) - 8} more")
        if workflow_result.old_format_failed:
            summary.append("")
            summary.append("Old-format export failures:")
            summary.extend(workflow_result.old_format_failed[:8])
            if len(workflow_result.old_format_failed) > 8:
                summary.append(
                    f"... and {len(workflow_result.old_format_failed) - 8} more"
                )

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
        view_state = SessionTabPresenter.build_active_session_view_state(info)
        self.session_info_label.setText(view_state.info_text)
        self.close_session_btn.setEnabled(view_state.close_enabled)
        self.upload_session_btn.setEnabled(view_state.upload_enabled)

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

            lock_user = getattr(self.session_manager, "operator_id", None)
            workflow_result = SessionFinalizeWorkflow.finalize_session(
                session_path=session_path,
                measurements_folder=measurements_folder,
                sample_id=info["sample_id"],
                container_manager=self._container_manager(),
                lock_user=lock_user,
                config=self.config if hasattr(self, "config") else None,
                logger=logger,
            )

            self.session_manager.close_session()

            details = [
                f"Session '{info['sample_id']}' has been finalized.",
                "",
                f"Container: {session_path.name}",
                f"Archived files: {workflow_result.archived_count}",
                f"Archive folder: {workflow_result.archive_dest}",
            ]
            if workflow_result.bundle_path:
                details.append(f"ZIP bundle: {workflow_result.bundle_path}")

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
