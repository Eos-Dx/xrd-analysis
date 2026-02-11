"""H5 Management Mixin - Container validation, locking, archiving, and loading."""
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Qt for type hints and usage
try:
    from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
except ImportError:
    # Test stubs
    class QFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return "", ""
    
    class QInputDialog:
        @staticmethod
        def getText(*args, **kwargs):
            return "", False
    
    class QMessageBox:
        Yes, No = 1, 0
        
        @staticmethod
        def question(*args, **kwargs):
            return QMessageBox.Yes
        
        @staticmethod
        def information(*args, **kwargs):
            pass
        
        @staticmethod
        def warning(*args, **kwargs):
            return QMessageBox.Yes
        
        @staticmethod
        def critical(*args, **kwargs):
            pass


class H5ManagementMixin:
    """Mixin for H5 container management operations.
    
    Handles:
    - Container validation
    - Container locking
    - Automatic archiving of old containers
    - Loading existing containers
    """
    
    def _validate_and_prompt_lock(self, container_path: str, container_id: str):
        """Validate container and prompt user to lock it.
        
        Args:
            container_path: Path to generated container
            container_id: Container ID
        """
        from hardware.difra.data.hdf5.technical_validator import validate_technical_container
        import h5py
        
        # Validate container
        try:
            is_valid, errors, warnings = validate_technical_container(container_path, strict=False)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Failed to validate container:\n{e}"
            )
            self._log_technical_event(f"Validation error: {e}")
            return
        
        # Check schema version
        expected_version = self.config.get("expected_technical_schema_version", "1.0")
        try:
            with h5py.File(container_path, 'r') as f:
                actual_version = f.attrs.get("schema_version", "unknown")
                if isinstance(actual_version, bytes):
                    actual_version = actual_version.decode('utf-8')
                
                if actual_version != expected_version:
                    errors.append(
                        f"Schema version mismatch: container has {actual_version}, expected {expected_version}"
                    )
                    is_valid = False
        except Exception as e:
            errors.append(f"Failed to check schema version: {e}")
            is_valid = False
        
        # Build validation summary
        status_icon = "✅" if is_valid else ("⚠️" if errors else "✅")
        summary_lines = [
            f"{status_icon} Container Validation Results",
            "",
            f"Container ID: {container_id}",
            f"Location: {os.path.basename(container_path)}",
            f"Schema Version: {actual_version}",
            "",
        ]
        
        if errors:
            summary_lines.append(f"❌ {len(errors)} Error(s):")
            for i, error in enumerate(errors[:5], 1):
                summary_lines.append(f"  {i}. {error}")
            if len(errors) > 5:
                summary_lines.append(f"  ... and {len(errors) - 5} more")
            summary_lines.append("")
        
        if warnings:
            summary_lines.append(f"⚠️  {len(warnings)} Warning(s):")
            for i, warning in enumerate(warnings[:3], 1):
                summary_lines.append(f"  {i}. {warning}")
            if len(warnings) > 3:
                summary_lines.append(f"  ... and {len(warnings) - 3} more")
            summary_lines.append("")
        
        if not errors and not warnings:
            summary_lines.append("✅ No issues found")
            summary_lines.append("")
        
        # Show validation results
        if is_valid:
            summary_lines.append("Container is valid and ready to lock.")
            summary_lines.append("\nLock this container for session measurements?")
            
            reply = QMessageBox.question(
                self,
                "Validation Passed",
                "\n".join(summary_lines),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            
            if reply == QMessageBox.Yes:
                # Lock the container
                self._lock_container(container_path, container_id)
            else:
                QMessageBox.information(
                    self,
                    "Container Saved",
                    f"Container saved without locking.\n\nLocation: {container_path}",
                )
        else:
            summary_lines.append("Container has validation errors.")
            summary_lines.append("\nYou can still use this container, but it may not work correctly.")
            summary_lines.append("\nSave anyway?")
            
            reply = QMessageBox.warning(
                self,
                "Validation Failed",
                "\n".join(summary_lines),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            
            if reply == QMessageBox.Yes:
                self._log_technical_event(f"User saved container {container_id} despite validation errors")
                QMessageBox.information(
                    self,
                    "Container Saved",
                    f"Container saved with errors.\n\nLocation: {container_path}",
                )
    
    def _archive_existing_containers(self, storage_folder: str) -> int:
        """Archive any existing .h5 containers in storage folder before creating new one.
        
        Prompts user about unvalidated/unlocked containers to determine if they were
        created by error, then archives with appropriate metadata.
        
        Archives both the .h5 container and any raw data files (.txt, .dsc) to:
        difra/archive/technical/<container_id>_<timestamp>/
        
        Args:
            storage_folder: Technical storage folder path
            
        Returns:
            Number of containers archived
        """
        from .helpers import _get_technical_archive_folder
        from hardware.container.v0_1.container_manager import is_container_locked
        
        storage_path = Path(storage_folder)
        if not storage_path.exists():
            return 0
        
        # Find all .h5 files in storage folder
        h5_files = list(storage_path.glob("*.h5"))
        if not h5_files:
            return 0
        
        archive_base = Path(_get_technical_archive_folder(
            self.config if hasattr(self, "config") else None
        ))
        
        archived_count = 0
        for h5_file in h5_files:
            try:
                # Extract container ID from filename (format: technical_<id>_<distance>.h5)
                filename = h5_file.stem  # Remove .h5 extension
                parts = filename.split('_')
                if len(parts) >= 2:
                    container_id = parts[1]  # Extract ID from technical_<id>_...
                else:
                    container_id = filename  # Fallback to full name
                
                # Check if container is locked
                is_locked = is_container_locked(h5_file)
                
                # If unlocked, prompt user about error status
                created_by_error = False
                error_reason = ""
                
                if not is_locked:
                    # Show dialog asking if this container was created by error
                    reply = QMessageBox.question(
                        self,
                        "Unvalidated Technical Container",
                        f"Found unvalidated technical container:\n\n"
                        f"Container ID: {container_id}\n"
                        f"File: {h5_file.name}\n\n"
                        f"You are about to create a new technical container.\n"
                        f"The existing container will be archived.\n\n"
                        f"Was this container created by error?\n\n"
                        f"Select 'Yes' to mark as error (you can provide a reason).\n"
                        f"Select 'No' to archive without error marking.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    
                    if reply == QMessageBox.Yes:
                        created_by_error = True
                        # Prompt for error reason
                        reason, ok = QInputDialog.getText(
                            self,
                            "Error Reason",
                            f"Why was container {container_id} created by error?\n\n"
                            f"(Optional - provide brief description)",
                        )
                        if ok and reason.strip():
                            error_reason = reason.strip()
                        else:
                            error_reason = "User marked as error without specifying reason"
                        
                        self._log_technical_event(
                            f"Container {container_id} marked as created_by_error: {error_reason}"
                        )
                
                # Create timestamped archive folder
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                archive_folder = archive_base / f"{container_id}_{timestamp}"
                archive_folder.mkdir(parents=True, exist_ok=True)
                
                # Move .h5 container to archive
                dest_h5 = archive_folder / h5_file.name
                shutil.move(str(h5_file), str(dest_h5))
                
                # Write error metadata if applicable
                if created_by_error:
                    import h5py
                    try:
                        with h5py.File(dest_h5, 'a') as f:
                            f.attrs['created_by_error'] = True
                            f.attrs['error_reason'] = error_reason
                            f.attrs['archived_timestamp'] = timestamp
                        self._log_technical_event(
                            f"Added error attributes to archived container: {h5_file.name}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add error attributes to {h5_file.name}: {e}")
                
                self._log_technical_event(
                    f"Archived H5 container: {h5_file.name} -> {archive_folder.name}/" +
                    (f" [ERROR: {error_reason}]" if created_by_error else "")
                )
                
                # Move any associated RAW data files (.txt, .dsc) from same folder
                # Skip .npy as it's processed data already stored in H5 container
                raw_file_count = 0
                for pattern in ["*.txt", "*.dsc"]:
                    for raw_file in storage_path.glob(pattern):
                        try:
                            dest_raw = archive_folder / raw_file.name
                            shutil.move(str(raw_file), str(dest_raw))
                            raw_file_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to archive {raw_file.name}: {e}")
                
                if raw_file_count > 0:
                    self._log_technical_event(
                        f"Archived {raw_file_count} raw data file(s) with container"
                    )
                
                archived_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to archive {h5_file.name}: {e}")
                self._log_technical_event(f"Warning: Could not archive {h5_file.name}: {e}")
        
        return archived_count
    
    def _lock_container(self, container_path: str, container_id: str):
        """Lock the technical container and archive raw data.
        
        Args:
            container_path: Path to container
            container_id: Container ID
        """
        from hardware.container.v0_1.container_manager import lock_technical_container
        from hardware.difra.gui.operator_manager import OperatorManager
        from .helpers import _get_technical_archive_folder
        
        # Get current operator
        operator_manager = OperatorManager()
        operator_id = operator_manager.get_current_operator_id()
        
        if not operator_id:
            operator_id = "unknown"
        
        # Lock the container
        try:
            lock_technical_container(
                Path(container_path),
                locked_by=operator_id,
                notes="Auto-locked after generation and validation"
            )
            
            self._log_technical_event(
                f"Container {container_id} locked by {operator_id}"
            )
            
            # After successful locking, archive raw data files
            archived_count = 0
            try:
                container_dir = Path(container_path).parent
                archive_folder = Path(_get_technical_archive_folder(
                    self.config if hasattr(self, "config") else None
                ))
                
                # Create timestamped subfolder in archive for this container
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                archive_subdir = archive_folder / f"{container_id}_{timestamp}"
                archive_subdir.mkdir(parents=True, exist_ok=True)
                
                # Find and move all RAW data files (.txt, .dsc) from the container directory
                # Skip .npy as it's processed data already stored in H5 container
                for pattern in ["*.txt", "*.dsc"]:
                    for data_file in container_dir.glob(pattern):
                        try:
                            dest = archive_subdir / data_file.name
                            shutil.move(str(data_file), str(dest))
                            archived_count += 1
                            self._log_technical_event(
                                f"Archived raw data: {data_file.name} -> {archive_subdir.name}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to archive {data_file.name}: {e}")
                
                if archived_count > 0:
                    self._log_technical_event(
                        f"Archived {archived_count} raw measurement file(s) to {archive_subdir}"
                    )
            except Exception as e:
                logger.warning(f"Failed to archive raw data files: {e}")
                self._log_technical_event(f"Warning: Could not archive raw data: {e}")
                # Non-fatal - container is still locked
            
            QMessageBox.information(
                self,
                "Container Locked",
                f"✅ Container locked successfully!\n\n"
                f"Container ID: {container_id}\n"
                f"Locked by: {operator_id}\n"
                f"Location: {container_path}\n"
                f"Raw data archived: {archived_count} file(s)\n\n"
                f"This container is now ready for session measurements.",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Lock Failed",
                f"Failed to lock container:\n{e}\n\nContainer location: {container_path}"
            )
            self._log_technical_event(f"Failed to lock container: {e}")
    
    def load_technical_h5(self):
        """Load and validate an existing technical HDF5 container.
        
        Automatically validates the container and displays its contents in the aux table.
        """
        from hardware.difra.data.hdf5.technical_validator import validate_technical_container
        from .helpers import _get_default_folder
        
        self._log_technical_event("Opening file dialog to load HDF5 container...")
        
        # Get folder from UI
        folder = (self.folderLE.text() or "").strip()
        if not folder:
            folder = _get_default_folder(self.config if hasattr(self, "config") else None)
        
        # Open file dialog to select HDF5 file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Technical HDF5 Container",
            folder,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if not file_path:
            self._log_technical_event("Load cancelled by user")
            return
        
        self._log_technical_event(f"Loading and validating: {os.path.basename(file_path)}")
        
        # Perform automatic validation
        try:
            is_valid, errors, warnings = validate_technical_container(file_path, strict=False)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Failed to validate container:\n{e}"
            )
            self._log_technical_event(f"Validation error: {e}")
            return
        
        # Show validation results
        if not is_valid:
            msg_parts = [
                f"Container validation failed with {len(errors)} error(s).",
                "",
                "Errors:"
            ]
            for i, error in enumerate(errors[:5], 1):
                msg_parts.append(f"  {i}. {error}")
            if len(errors) > 5:
                msg_parts.append(f"  ... and {len(errors) - 5} more")
            
            msg_parts.append("")
            msg_parts.append("Do you want to load this container anyway?")
            
            reply = QMessageBox.question(
                self,
                "⚠️ Container Validation Failed",
                "\n".join(msg_parts),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            
            if reply != QMessageBox.Yes:
                self._log_technical_event("Load cancelled due to validation errors")
                return
        
        # Load container and populate aux table
        try:
            self._populate_aux_table_from_h5(file_path)
            
            # Show success message
            status_icon = "✅" if is_valid else "⚠️"
            msg_parts = [
                f"{status_icon} Container loaded: {os.path.basename(file_path)}",
                "",
            ]
            
            if warnings:
                msg_parts.append(f"⚠️  {len(warnings)} warning(s):")
                for i, warning in enumerate(warnings[:3], 1):
                    msg_parts.append(f"  {i}. {warning}")
                if len(warnings) > 3:
                    msg_parts.append(f"  ... and {len(warnings) - 3} more")
            else:
                msg_parts.append("Status: ✅ VALID")
            
            QMessageBox.information(self, "Container Loaded", "\n".join(msg_parts))
            self._log_technical_event(f"Container loaded successfully: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load container contents:\n{e}"
            )
            self._log_technical_event(f"Load error: {e}")
            logger.error(f"Error loading container: {e}", exc_info=True)
    
    def _populate_aux_table_from_h5(self, h5_path: str):
        """Populate aux table from a technical HDF5 container.
        
        Args:
            h5_path: Path to the technical HDF5 container
        """
        import h5py
        from hardware.container.v0_1 import schema
        from PyQt5.QtWidgets import QComboBox
        
        # Clear existing table
        self.auxTable.setRowCount(0)
        
        with h5py.File(h5_path, "r") as f:
            tech_group = f.get("technical")
            if not tech_group:
                raise ValueError("No /technical group found in container")
            
            # Iterate through technical events
            for evt_name in sorted(tech_group.keys()):
                if not evt_name.startswith("tech_evt_"):
                    continue
                
                evt_group = tech_group[evt_name]
                tech_type = evt_group.attrs.get("technical_type", "UNKNOWN")
                
                # Iterate through detectors in this event
                for det_name in evt_group.keys():
                    if not det_name.startswith("det_"):
                        continue
                    
                    det_group = evt_group[det_name]
                    
                    # Get detector alias from attributes
                    detector_id = det_group.attrs.get("detector_id", det_name.replace("det_", ""))
                    
                    # Get measurement file path if stored
                    file_path = det_group.attrs.get("source_file", "")
                    if not file_path:
                        file_path = f"[H5: {evt_name}/{det_name}]"
                    
                    # Add to table
                    alias = detector_id.upper() if detector_id else "UNKNOWN"
                    self._add_aux_item_to_list(alias, file_path)
                    
                    # Set type in the newly added row
                    row_idx = self.auxTable.rowCount() - 1
                    type_cb = self.auxTable.cellWidget(row_idx, 1)
                    if type_cb and isinstance(type_cb, QComboBox):
                        idx = type_cb.findText(tech_type)
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
        
        self._log_technical_event(
            f"Loaded {self.auxTable.rowCount()} measurements from container"
        )
