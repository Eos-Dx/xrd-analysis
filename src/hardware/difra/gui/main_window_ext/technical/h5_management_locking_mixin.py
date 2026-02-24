"""Technical H5 validation/locking responsibilities."""

from . import h5_management_mixin as _module

os = _module.os
logger = _module.logger
QInputDialog = _module.QInputDialog
QMessageBox = _module.QMessageBox
QFileDialog = _module.QFileDialog
Path = _module.Path
shutil = _module.shutil
time = _module.time
get_container_manager = _module.get_container_manager
get_technical_validator = _module.get_technical_validator


class H5ManagementLockingMixin:
    def _validate_and_prompt_lock(self, container_path: str, container_id: str):
        """Validate container and prompt user to lock it.
        
        Args:
            container_path: Path to generated container
            container_id: Container ID
        """
        import h5py
        technical_validator = get_technical_validator(
            self.config if hasattr(self, "config") else None
        )
        validate_technical_container = technical_validator.validate_technical_container
        
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
        expected_version = self.config.get(
            "expected_technical_schema_version",
            self.config.get("container_version", "0.2"),
        )
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
        container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
        
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
                is_locked = container_manager.is_container_locked(h5_file)
                
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
                archive_operator = "unknown"
                try:
                    import h5py

                    with h5py.File(h5_file, "r") as f:
                        raw_operator = (
                            f.attrs.get("locked_by")
                            or f.attrs.get("operator_id")
                        )
                        if isinstance(raw_operator, bytes):
                            raw_operator = raw_operator.decode("utf-8", errors="replace")
                        archive_operator = (
                            "".join(
                                ch if ch.isalnum() or ch in ("-", "_") else "_"
                                for ch in str(raw_operator or "")
                            ).strip("_")
                            or "unknown"
                        )
                except Exception:
                    archive_operator = "unknown"

                archive_folder = archive_base / f"{container_id}_{archive_operator}_{timestamp}"
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
                
                # Move any associated data files using container module function
                # Get patterns from config (detector-specific)
                file_patterns = None
                if hasattr(self, 'config') and self.config:
                    file_patterns = self.config.get(
                        'technical_archive_patterns',
                        ['*.txt', '*.dsc', '*.npy', '*.poni', '*_state.json'],
                    )
                
                try:
                    archive_technical_data_files = container_manager.archive_technical_data_files
                    # Create a dummy container path in the storage folder to use with the function
                    dummy_container_path = storage_path / h5_file.name
                    raw_file_count = archive_technical_data_files(
                        container_path=dummy_container_path,
                        archive_folder=archive_folder,
                        file_patterns=file_patterns
                    )
                    
                    if raw_file_count > 0:
                        self._log_technical_event(
                            f"Archived {raw_file_count} data file(s) with container"
                        )
                except Exception as e:
                    logger.warning(f"Failed to archive data files: {e}")
                
                archived_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to archive {h5_file.name}: {e}")
                self._log_technical_event(f"Warning: Could not archive {h5_file.name}: {e}")
        
        return archived_count

    def _update_aux_table_paths_after_archive(self, archive_folder: Path) -> int:
        """Remap aux table file paths to archived locations for visualization."""
        try:
            from hardware.difra.gui.main_window_ext import technical_measurements as tm
        except Exception:
            return 0

        if not hasattr(self, "auxTable") or self.auxTable is None:
            return 0

        updated = 0
        archive_folder = Path(archive_folder)
        for row in range(self.auxTable.rowCount()):
            file_item = self.auxTable.item(row, 1)
            if file_item is None:
                continue

            old_path = str(file_item.data(tm.Qt.UserRole) or "").strip()
            if not old_path:
                continue

            old_file = Path(old_path)
            if old_file.exists():
                continue

            candidate = archive_folder / old_file.name
            if not candidate.exists():
                continue

            file_item.setData(tm.Qt.UserRole, str(candidate))
            updated += 1

        if updated > 0:
            self._log_technical_event(
                f"Updated {updated} technical table path(s) to archive folder: {archive_folder.name}"
            )
        return updated

    def create_new_technical_container(self):
        """Legacy API kept for compatibility; uses container-first creation flow."""
        if hasattr(self, "_create_new_active_technical_container"):
            created = self._create_new_active_technical_container(clear_table=True)
            if created is not None:
                QMessageBox.information(
                    self,
                    "Technical Container",
                    f"Created new technical container:\n{created}",
                )
            return
        QMessageBox.information(
            self,
            "Removed Workflow",
            "Legacy 'New Container' flow is removed.",
        )

    def lock_active_technical_container(self):
        """Lock currently active technical container."""
        import h5py

        if hasattr(self, "_sync_active_technical_container_from_table"):
            self._sync_active_technical_container_from_table(show_errors=True)

        active_path = getattr(self, "_active_technical_container_path", "")
        active_path = str(active_path or "").strip()
        if not active_path:
            QMessageBox.warning(
                self,
                "No Active Container",
                "No active technical container loaded or created.",
            )
            return

        container_path = Path(active_path)
        if not container_path.exists():
            QMessageBox.warning(
                self,
                "Container Missing",
                f"Technical container not found:\n{container_path}",
            )
            return

        container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
        if container_manager.is_container_locked(container_path):
            QMessageBox.information(
                self,
                "Already Locked",
                f"Container is already locked:\n{container_path.name}",
            )
            return

        container_id = container_path.stem
        try:
            with h5py.File(container_path, "r") as h5f:
                raw_id = h5f.attrs.get("container_id")
                if isinstance(raw_id, bytes):
                    raw_id = raw_id.decode("utf-8", errors="replace")
                if raw_id:
                    container_id = str(raw_id)
        except Exception:
            pass

        self._lock_container(str(container_path), container_id)
        self._active_technical_container_locked = True
    
    def _lock_container(self, container_path: str, container_id: str):
        """Lock the technical container and archive raw data.
        
        Args:
            container_path: Path to container
            container_id: Container ID
        """
        from hardware.difra.gui.operator_manager import OperatorManager
        from .helpers import _get_technical_archive_folder
        container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
        
        # Get current operator
        operator_manager = OperatorManager()
        operator_id = operator_manager.get_current_operator_id()
        
        if not operator_id:
            operator_id = "unknown"
        
        # Lock the container
        try:
            logger.info(
                "Locking technical container: id=%s path=%s operator=%s",
                container_id,
                str(container_path),
                str(operator_id),
            )
            container_manager.lock_technical_container(
                Path(container_path),
                locked_by=operator_id,
                notes="Auto-locked after generation and validation"
            )
            
            self._log_technical_event(
                f"Container {container_id} locked by {operator_id}"
            )
            logger.info(
                "Technical container locked: id=%s operator=%s",
                container_id,
                str(operator_id),
            )
            
            # After successful locking, archive data files
            archived_count = 0
            try:
                archive_folder = Path(_get_technical_archive_folder(
                    self.config if hasattr(self, "config") else None
                ))
                
                # Create timestamped subfolder in archive for this container
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                operator_token = (
                    "".join(
                        ch if ch.isalnum() or ch in ("-", "_") else "_"
                        for ch in str(operator_id or "")
                    ).strip("_")
                    or "unknown"
                )
                archive_subdir = archive_folder / f"{container_id}_{operator_token}_{timestamp}"
                
                # Get file patterns from config (detector-specific)
                # Default to Advacam patterns if not configured
                file_patterns = None
                if hasattr(self, 'config') and self.config:
                    file_patterns = self.config.get(
                        'technical_archive_patterns',
                        ['*.txt', '*.dsc', '*.npy', '*.poni', '*_state.json'],
                    )
                
                # Use container module function to archive files
                archived_count = container_manager.archive_technical_data_files(
                    container_path=Path(container_path),
                    archive_folder=archive_subdir,
                    file_patterns=file_patterns
                )
                self._update_aux_table_paths_after_archive(archive_subdir)
                
                if archived_count > 0:
                    self._log_technical_event(
                        f"Archived {archived_count} data file(s) to {archive_subdir.name}"
                    )
                logger.info(
                    "Archived technical container companion files: id=%s archived=%d folder=%s",
                    container_id,
                    int(archived_count),
                    str(archive_subdir),
                )
            except Exception as e:
                logger.warning(f"Failed to archive data files: {e}")
                self._log_technical_event(f"Warning: Could not archive data files: {e}")
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
            logger.error(
                "Technical container lock failed: id=%s path=%s error=%s",
                container_id,
                str(container_path),
                str(e),
            )
    
