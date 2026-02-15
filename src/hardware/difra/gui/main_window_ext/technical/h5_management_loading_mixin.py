"""Technical H5 loading/table population responsibilities."""

from . import h5_management_mixin as _module

os = _module.os
logger = _module.logger
QInputDialog = _module.QInputDialog
QMessageBox = _module.QMessageBox
QFileDialog = _module.QFileDialog
get_container_manager = _module.get_container_manager
get_schema = _module.get_schema
get_technical_validator = _module.get_technical_validator


class H5ManagementLoadingMixin:
    def validate_technical_h5(self):
        """Validate an existing technical HDF5 container without loading it.
        
        Displays validation results in a dialog.
        """
        from .helpers import _get_default_folder
        import h5py
        technical_validator = get_technical_validator(
            self.config if hasattr(self, "config") else None
        )
        validate_technical_container = technical_validator.validate_technical_container
        container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
        
        self._log_technical_event("Opening file dialog to validate HDF5 container...")
        
        # Get folder from UI
        folder = (self.folderLE.text() or "").strip()
        if not folder:
            folder = _get_default_folder(self.config if hasattr(self, "config") else None)
        
        # Open file dialog to select HDF5 file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Validate Technical HDF5 Container",
            folder,
            "NeXus HDF5 Files (*.nxs.h5 *.h5 *.hdf5);;All Files (*)"
        )
        
        if not file_path:
            self._log_technical_event("Validation cancelled by user")
            return
        
        self._log_technical_event(f"Validating: {os.path.basename(file_path)}")
        
        # Perform validation
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
        
        # Check lock status
        is_locked = container_manager.is_container_locked(file_path)
        lock_status = "🔒 LOCKED" if is_locked else "🔓 UNLOCKED"
        
        # Check schema version
        expected_version = self.config.get(
            "expected_technical_schema_version",
            self.config.get("container_version", "0.2"),
        )
        try:
            with h5py.File(file_path, 'r') as f:
                actual_version = f.attrs.get("schema_version", "unknown")
                if isinstance(actual_version, bytes):
                    actual_version = actual_version.decode('utf-8')
                container_id = f.attrs.get("container_id", "unknown")
                if isinstance(container_id, bytes):
                    container_id = container_id.decode('utf-8')
        except Exception as e:
            actual_version = "unknown"
            container_id = "unknown"
            errors.append(f"Failed to read container metadata: {e}")
        
        # Build validation report
        status_icon = "✅" if is_valid else "❌"
        report_lines = [
            f"{status_icon} Validation Report",
            "",
            f"Container: {os.path.basename(file_path)}",
            f"Container ID: {container_id}",
            f"Lock Status: {lock_status}",
            f"Schema Version: {actual_version} (expected: {expected_version})",
            "",
        ]
        
        if errors:
            report_lines.append(f"❌ {len(errors)} Error(s):")
            for i, error in enumerate(errors[:10], 1):
                report_lines.append(f"  {i}. {error}")
            if len(errors) > 10:
                report_lines.append(f"  ... and {len(errors) - 10} more")
            report_lines.append("")
        
        if warnings:
            report_lines.append(f"⚠️  {len(warnings)} Warning(s):")
            for i, warning in enumerate(warnings[:5], 1):
                report_lines.append(f"  {i}. {warning}")
            if len(warnings) > 5:
                report_lines.append(f"  ... and {len(warnings) - 5} more")
            report_lines.append("")
        
        if not errors and not warnings:
            report_lines.append("✅ No issues found")
            report_lines.append("")
        
        # Show report
        if is_valid:
            QMessageBox.information(
                self,
                "✅ Validation Passed",
                "\n".join(report_lines)
            )
        else:
            QMessageBox.warning(
                self,
                "❌ Validation Failed",
                "\n".join(report_lines)
            )
        
        self._log_technical_event(f"Validation complete: {status_icon} {len(errors)} errors, {len(warnings)} warnings")
    
    def load_technical_h5(self):
        """Load and validate an existing technical HDF5 container.
        
        Automatically validates the container and displays its contents in the aux table.
        Works with both locked and unlocked containers.
        """
        from .helpers import _get_default_folder
        technical_validator = get_technical_validator(
            self.config if hasattr(self, "config") else None
        )
        validate_technical_container = technical_validator.validate_technical_container
        container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
        
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
            "NeXus HDF5 Files (*.nxs.h5 *.h5 *.hdf5);;All Files (*)"
        )
        
        if not file_path:
            self._log_technical_event("Load cancelled by user")
            return
        
        self._log_technical_event(f"Loading and validating: {os.path.basename(file_path)}")
        
        # Check lock status
        is_locked = container_manager.is_container_locked(file_path)
        lock_status = "🔒 LOCKED" if is_locked else "🔓 UNLOCKED"
        
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
                f"Lock Status: {lock_status}",
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

            # Notify host window so active session containers can optionally
            # refresh embedded technical data.
            if hasattr(self, "on_technical_container_loaded"):
                try:
                    self.on_technical_container_loaded(file_path, is_locked=is_locked)
                except Exception as callback_error:
                    logger.warning(
                        f"Technical-load callback failed: {callback_error}",
                        exc_info=True,
                    )
            
            # Show success message
            status_icon = "✅" if is_valid else "⚠️"
            msg_parts = [
                f"{status_icon} Container loaded: {os.path.basename(file_path)}",
                f"Lock Status: {lock_status}",
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
            self._log_technical_event(f"Container loaded successfully: {os.path.basename(file_path)} ({lock_status})")
            
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
        
        Also extracts and sets detector distances from the container.
        
        Args:
            h5_path: Path to the technical HDF5 container
        """
        import h5py
        schema = get_schema(self.config if hasattr(self, "config") else None)
        from PyQt5.QtWidgets import QCheckBox, QComboBox

        def _as_text(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value) if value is not None else ""

        def _find_column_index(label: str, default_index: int) -> int:
            label_norm = label.strip().lower()
            for index in range(self.auxTable.columnCount()):
                header_item = self.auxTable.horizontalHeaderItem(index)
                if header_item and header_item.text().strip().lower() == label_norm:
                    return index
            return default_index
        
        # Clear existing table
        self.auxTable.setRowCount(0)
        primary_col = _find_column_index("Primary", 0)
        type_col = _find_column_index("Type", 2)
        
        # Extract detector distances from container
        extracted_distances = {}
        detector_configs = self.config.get("detectors", []) if hasattr(self, "config") else []
        alias_to_detector_id = {
            cfg.get("alias"): cfg.get("id")
            for cfg in detector_configs
            if cfg.get("alias") and cfg.get("id")
        }
        detector_id_to_alias = {
            cfg.get("id"): cfg.get("alias")
            for cfg in detector_configs
            if cfg.get("id") and cfg.get("alias")
        }
        
        # Track loaded items for logging
        loaded_count = 0
        
        with h5py.File(h5_path, "r") as f:
            tech_group = f.get(schema.GROUP_TECHNICAL)
            if tech_group is None:
                tech_group = f.get(f"{schema.GROUP_CALIBRATION_SNAPSHOT}/events")
            if not tech_group:
                raise ValueError(
                    f"No technical event group found in container. "
                    f"Expected {schema.GROUP_TECHNICAL} or "
                    f"{schema.GROUP_CALIBRATION_SNAPSHOT}/events."
                )
            
            # Iterate through technical events
            for evt_name in sorted(tech_group.keys()):
                if not evt_name.startswith("tech_evt_"):
                    continue
                
                evt_group = tech_group[evt_name]
                # Read technical type from event-level attrs.
                # Current writers use "type"; keep ATTR_TECHNICAL_TYPE as fallback.
                tech_type = _as_text(evt_group.attrs.get(
                    "type",
                    evt_group.attrs.get(schema.ATTR_TECHNICAL_TYPE, "UNKNOWN"),
                )).strip().upper()
                
                # Iterate through detectors in this event
                for det_name in evt_group.keys():
                    if not det_name.startswith("det_"):
                        continue
                    
                    det_group = evt_group[det_name]
                    
                    # Recover detector identity from canonical attributes.
                    detector_id = det_group.attrs.get(schema.ATTR_DETECTOR_ID, "")
                    detector_alias = det_group.attrs.get(schema.ATTR_DETECTOR_ALIAS, "")
                    if isinstance(detector_id, bytes):
                        detector_id = detector_id.decode("utf-8")
                    if isinstance(detector_alias, bytes):
                        detector_alias = detector_alias.decode("utf-8")
                    if not detector_alias:
                        detector_alias = detector_id_to_alias.get(detector_id, "")
                    if not detector_alias:
                        try:
                            detector_alias = schema.parse_detector_role(det_name)
                        except Exception:
                            detector_alias = det_name.replace("det_", "").upper()
                    if not detector_id:
                        detector_id = alias_to_detector_id.get(detector_alias, detector_alias)
                    
                    # Extract distance from this detector's measurement
                    distance_cm = det_group.attrs.get("distance_cm", None)
                    if distance_cm is not None and detector_id:
                        extracted_distances[detector_id] = float(distance_cm)
                    
                    # Get measurement file path if stored
                    file_path = det_group.attrs.get("source_file", "")
                    if not file_path:
                        file_path = f"[H5: {evt_name}/{det_name}]"
                    
                    # Add to table
                    alias = detector_alias if detector_alias else "UNKNOWN"
                    try:
                        self._add_aux_item_to_list(alias, file_path)
                        loaded_count += 1
                        
                        # Log each item added
                        self._log_technical_event(
                            f"Loading: {tech_type} measurement for {alias}"
                        )
                    except Exception as add_err:
                        logger.error(f"Failed to add item to table: {add_err}", exc_info=True)
                        self._log_technical_event(f"Error adding {alias}: {add_err}")
                        continue
                    
                    # Set type and primary status in the newly added row
                    row_idx = self.auxTable.rowCount() - 1
                    
                    # Set Type combobox (column 2)
                    type_cb = self.auxTable.cellWidget(row_idx, type_col)
                    if type_cb and isinstance(type_cb, QComboBox):
                        idx = -1
                        for item_idx in range(type_cb.count()):
                            item_text = _as_text(type_cb.itemText(item_idx)).strip().upper()
                            if item_text == tech_type:
                                idx = item_idx
                                break
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
                    
                    # Loaded technical container rows are canonical primaries.
                    try:
                        checkbox_widget = self.auxTable.cellWidget(row_idx, primary_col)
                        if checkbox_widget:
                            checkbox = (
                                checkbox_widget
                                if isinstance(checkbox_widget, QCheckBox)
                                else checkbox_widget.findChild(QCheckBox)
                            )
                            if checkbox is not None:
                                checkbox.setChecked(True)
                    except Exception as e:
                        logger.warning(f"Failed to set primary checkbox: {e}")
        
        # Force table update
        try:
            self.auxTable.viewport().update()
        except Exception:
            pass
        
        # Set extracted distances
        if extracted_distances:
            self._detector_distances = extracted_distances
            self._log_technical_event(
                f"Extracted distances from container: {extracted_distances}"
            )
            
            # Update window title and button states
            if hasattr(self, '_update_window_title_with_distances'):
                self._update_window_title_with_distances()
            if hasattr(self, '_update_distance_dependent_controls'):
                self._update_distance_dependent_controls()
        
        # Final summary
        row_count = self.auxTable.rowCount()
        if row_count == 0:
            self._log_technical_event(
                "WARNING: No measurements were loaded from container (table is empty)"
            )
            logger.warning(f"No items loaded from H5 container: {h5_path}")
        else:
            self._log_technical_event(
                f"Successfully loaded {row_count} measurements from container"
            )
