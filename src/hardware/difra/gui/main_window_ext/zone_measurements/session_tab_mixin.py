"""Session management tab for Zone Measurements.

Provides UI for:
- Closing/finalizing session containers
- Uploading to cloud
- Archiving measurement files
"""

from pathlib import Path

from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class SessionTabMixin:
    """Mixin for session management tab in Zone Measurements."""
    
    def create_session_tab(self):
        """Create session management tab with close/finalize and upload buttons."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Session info group
        info_group = QGroupBox("Session Information")
        info_layout = QVBoxLayout(info_group)
        
        self.session_info_label = QLabel("No active session")
        self.session_info_label.setStyleSheet("padding: 10px;")
        info_layout.addWidget(self.session_info_label)
        
        layout.addWidget(info_group)
        
        # Session actions group
        actions_group = QGroupBox("Session Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Close/finalize button
        close_layout = QHBoxLayout()
        self.close_session_btn = QPushButton("Close && Finalize Session")
        self.close_session_btn.setToolTip(
            "Lock the session container and archive measurement files.\n"
            "This will:\n"
            "1. Lock the session container (read-only)\n"
            "2. Move raw and .npy files to difra/archive/measurements/\n"
            "3. Close the active session"
        )
        self.close_session_btn.clicked.connect(self._on_close_finalize_session)
        self.close_session_btn.setEnabled(False)
        close_layout.addWidget(self.close_session_btn)
        actions_layout.addLayout(close_layout)
        
        # Upload button
        upload_layout = QHBoxLayout()
        self.upload_session_btn = QPushButton("Upload to Cloud")
        self.upload_session_btn.setToolTip(
            "Upload session container to cloud storage.\n"
            "Session must be closed and finalized first."
        )
        self.upload_session_btn.clicked.connect(self._on_upload_session)
        self.upload_session_btn.setEnabled(False)
        upload_layout.addWidget(self.upload_session_btn)
        actions_layout.addLayout(upload_layout)
        
        layout.addWidget(actions_group)
        layout.addStretch()
        
        # Add tab to tabs widget
        if hasattr(self, 'tabs'):
            self.tabs.addTab(tab, "Session")
        
        # Update session info on creation
        self._update_session_tab_info()
    
    def _update_session_tab_info(self):
        """Update session info display and button states."""
        if not hasattr(self, 'session_manager'):
            return
        
        info = self.session_manager.get_session_info()
        
        if info['active']:
            # Update info label
            info_text = f"<b>Sample ID:</b> {info['sample_id']}<br>"
            info_text += f"<b>Session ID:</b> {info['session_id']}<br>"
            info_text += f"<b>Operator:</b> {info['operator_id']}<br>"
            info_text += f"<b>Container:</b> {Path(info['session_path']).name}<br>"
            info_text += f"<b>Status:</b> {'🔒 Locked' if info['is_locked'] else '🔓 Unlocked'}"
            
            self.session_info_label.setText(info_text)
            
            # Enable/disable buttons based on lock status
            is_locked = info['is_locked']
            self.close_session_btn.setEnabled(not is_locked)
            self.upload_session_btn.setEnabled(is_locked)
        else:
            self.session_info_label.setText("No active session")
            self.close_session_btn.setEnabled(False)
            self.upload_session_btn.setEnabled(False)
    
    def _on_close_finalize_session(self):
        """Close and finalize the session container, archive measurement files."""
        if not hasattr(self, 'session_manager') or not self.session_manager.is_session_active():
            QMessageBox.warning(
                self,
                "No Active Session",
                "No session is currently active.",
            )
            return
        
        info = self.session_manager.get_session_info()
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Close and Finalize Session?",
            f"Close and finalize session '{info['sample_id']}'?\n\n"
            f"This will:\n"
            f"• Lock the session container (read-only)\n"
            f"• Archive measurement files to difra/archive/measurements/\n"
            f"• Close the active session\n\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            from hardware.container.v0_1.container_manager import lock_container
            session_path = Path(info['session_path'])
            measurements_folder = session_path.parent
            
            # Store JSON state file in container before archiving
            self._store_json_state_in_container(session_path, measurements_folder, info['sample_id'])
            
            logger.info("Locking session container", session_path=str(session_path))
            lock_container(session_path)
            
            # Archive measurement files
            archive_dest, archived_count = self._archive_measurement_files(
                measurements_folder, info['sample_id']
            )
            bundle_path = self._create_session_bundle_zip(session_path, archive_dest)
            
            # Close session
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

            QMessageBox.information(
                self,
                "Session Finalized",
                "\n".join(details),
            )
            
            logger.info("Session finalized and closed", sample_id=info['sample_id'])
            
            # Update UI
            self._update_session_tab_info()
            if hasattr(self, 'update_session_status'):
                self.update_session_status()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Finalization Failed",
                f"Failed to finalize session:\n\n{str(e)}",
            )
            logger.error(f"Failed to finalize session: {e}", exc_info=True)
    
    def _store_json_state_in_container(self, session_path: Path, measurements_folder: Path, sample_id: str):
        """Store JSON state file in session container as attribute.
        
        Args:
            session_path: Path to session container
            measurements_folder: Folder containing state JSON file
            sample_id: Sample ID for finding state file
        """
        import json
        import h5py
        
        # Find state JSON file
        state_file = measurements_folder / f"{sample_id}_state.json"
        
        if not state_file.exists():
            logger.warning(f"State JSON file not found: {state_file}")
            return
        
        try:
            # Read JSON state
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Store as root attribute in container
            with h5py.File(session_path, 'a') as h5f:
                # Store as JSON string attribute
                h5f.attrs['meta_json'] = json.dumps(state_data)
            
            logger.info(
                f"Stored state JSON in container",
                session_path=str(session_path),
                state_file=str(state_file)
            )
            
        except Exception as e:
            logger.error(f"Failed to store state JSON in container: {e}", exc_info=True)
    
    def _archive_measurement_files(self, measurements_folder: Path, sample_id: str):
        """Archive raw and .npy measurement files, including state JSON.
        
        Args:
            measurements_folder: Folder containing measurement files
            sample_id: Sample ID for archive folder naming

        Returns:
            Tuple of (archive destination folder, archived file count)
        """
        from fnmatch import fnmatch
        import shutil
        import time
        
        # Get archive folder from config, with fallback
        if hasattr(self, 'config') and self.config:
            archive_base = self.config.get('measurements_archive_folder')
            if archive_base:
                archive_folder = Path(archive_base)
            else:
                # Fallback: difra_base/archive/measurements
                difra_base = measurements_folder.parent
                archive_folder = difra_base / "archive" / "measurements"
        else:
            # No config, use default
            difra_base = measurements_folder.parent
            archive_folder = difra_base / "archive" / "measurements"
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_dest = archive_folder / f"{sample_id}_{timestamp}"
        archive_dest.mkdir(parents=True, exist_ok=True)
        
        # Find measurement files (raw, .npy, and state JSON), preserving
        # operator-created folder structure.
        patterns = ['*.txt', '*.dsc', '*.npy', '*.t3pa', '*_state.json']
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
                logger.debug(f"Archived: {relative_path}")
            except Exception as e:
                logger.warning(f"Failed to archive {relative_path}: {e}")
        
        logger.info(
            f"Archived {archived_count} measurement files",
            archive_folder=str(archive_dest)
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
        except Exception as e:
            logger.warning(f"Failed to create session ZIP bundle: {e}")
            return None
    
    def _on_upload_session(self):
        """Upload session container to cloud."""
        if not hasattr(self, 'session_manager') or not self.session_manager.is_session_active():
            QMessageBox.warning(
                self,
                "No Active Session",
                "No session is currently active.",
            )
            return
        
        info = self.session_manager.get_session_info()
        
        if not info['is_locked']:
            QMessageBox.warning(
                self,
                "Session Not Finalized",
                "Session must be closed and finalized before uploading.\n\n"
                "Click 'Close & Finalize Session' first.",
            )
            return
        
        # TODO: Implement cloud upload
        QMessageBox.information(
            self,
            "Upload to Cloud",
            f"Cloud upload for session '{info['sample_id']}' not yet implemented.\n\n"
            f"Container: {Path(info['session_path']).name}",
        )
        
        logger.info("Cloud upload requested", sample_id=info['sample_id'])
