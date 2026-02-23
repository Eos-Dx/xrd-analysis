"""Technical HDF5 container generation responsibilities."""

from . import h5_generation_mixin as _module

json = _module.json
logger = _module.logger
os = _module.os
re = _module.re
uuid = _module.uuid

QComboBox = _module.QComboBox
QDialog = _module.QDialog
QFileDialog = _module.QFileDialog
QInputDialog = _module.QInputDialog
QMessageBox = _module.QMessageBox
QCheckBox = _module.QCheckBox
Qt = _module.Qt

get_schema = _module.get_schema
get_technical_container = _module.get_technical_container


class H5GenerationContainerMixin:
    def generate_technical_h5(self):
        """Generate technical HDF5 container from measurements in aux table."""
        schema = get_schema(self.config if hasattr(self, "config") else None)
        technical_container = get_technical_container(
            self.config if hasattr(self, "config") else None
        )
        from .helpers import _get_technical_temp_folder
        from hardware.difra.gui.main_window_ext.technical_measurements import PoniFileSelectionDialog

        self._log_technical_event("Generating technical HDF5 container...")

        # Use ALL rows in table (no selection required)
        # User marks which are primary via the checkbox column
        rows = list(range(self.auxTable.rowCount()))
        if not rows:
            self._log_technical_event("Error: No rows in table for HDF5 generation")
            QMessageBox.warning(
                self, "No Data", "No measurements in the Aux table."
            )
            return

        # Validate folder
        folder = (self.folderLE.text() or "").strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Select a valid save folder.")
            return
        if not os.access(folder, os.W_OK):
            QMessageBox.warning(
                self,
                "Folder Not Writable",
                "Selected save folder is not writable. Choose a different folder.",
            )
            return
        logger.info(
            "Technical container generation requested: rows=%d folder=%s",
            len(rows),
            folder,
        )

        aux_measurements = {}
        primary_measurements = {}  # Track which measurements are marked as primary: {(type, alias): [is_prim1, is_prim2, ...]}
        gui_integration_ms = None
        gui_n_frames = None
        try:
            gui_integration_ms = float(self.integrationTimeSpin.value()) * 1000.0
        except Exception:
            gui_integration_ms = None
        try:
            gui_n_frames = int(self.captureFramesSpin.value())
        except Exception:
            gui_n_frames = None

        # Get active detector aliases for validation
        try:
            active_aliases = self._get_active_detector_aliases()
        except Exception:
            active_aliases = []

        for row in rows:
            # Check if primary checkbox is checked
            checkbox_widget = self.auxTable.cellWidget(row, 0)
            is_primary = False
            if checkbox_widget:
                # Find the QCheckBox within the widget
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    is_primary = checkbox.isChecked()
            
            file_item = self.auxTable.item(row, 1)
            if not file_item:
                continue
            file_path = file_item.data(Qt.UserRole)
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(
                    self, "Missing File", f"Row {row+1}: file path does not exist."
                )
                return

            # Type
            type_cb = self.auxTable.cellWidget(row, 2)
            if (
                not isinstance(type_cb, QComboBox)
                or type_cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Type", f"Row {row+1}: select measurement type."
                )
                return
            typ_ui = type_cb.currentText()
            typ = self._normalize_technical_type(typ_ui)

            # Alias (must be selected)
            cb = self.auxTable.cellWidget(row, 3)
            if (
                not isinstance(cb, QComboBox)
                or cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Alias", f"Row {row+1}: select an alias."
                )
                return
            alias = cb.currentText()

            if typ not in schema.ALL_TECHNICAL_TYPES:
                QMessageBox.warning(
                    self,
                    "Invalid Type",
                    f"Type '{typ_ui}' is not supported for HDF5.\n"
                    f"Supported: {', '.join(schema.ALL_TECHNICAL_TYPES)}",
                )
                return

            if typ_ui == "SPECIAL":
                self._log_technical_event("Mapping type SPECIAL → WATER for HDF5")

            # Allow multiple measurements per (type, alias) pair
            # Only PRIMARY measurements will be used in H5, supplementary are ignored
            # We validate PRIMARY uniqueness later
            # 
            # If this is a primary measurement, it will be used for H5
            if is_primary:
                entry = {"file_path": file_path}
                try:
                    row_metadata = self._get_aux_row_metadata(
                        row,
                        str(file_path),
                        include_filename_fallback=False,
                    )
                except Exception:
                    row_metadata = {}
                if isinstance(row_metadata, dict):
                    for key, value in row_metadata.items():
                        if value is not None:
                            entry[key] = value
                if entry.get("integration_time_ms") is None and gui_integration_ms is not None:
                    entry["integration_time_ms"] = gui_integration_ms
                if entry.get("n_frames") is None and gui_n_frames is not None:
                    entry["n_frames"] = gui_n_frames
                aux_measurements.setdefault(typ, {})[alias] = entry
            
            # Track primary/supplementary status for this row
            pair = (typ, alias)
            if pair not in primary_measurements:
                primary_measurements[pair] = []
            primary_measurements[pair].append(is_primary)
            
            self._log_technical_event(
                f"Row {row+1}: {typ_ui} for {alias} - {'PRIMARY' if is_primary else 'supplementary'}"
            )

        # Enforce completeness: all REQUIRED measurement types must be present, and for each alias
        required_types = set(
            getattr(self, "REQUIRED_TYPE_OPTIONS", None)
            or schema.REQUIRED_TECHNICAL_TYPES
        )

        # 1) Ensure at least one row selected for each required type
        types_in_meta = {t for t in aux_measurements.keys() if t in required_types}
        missing_types = sorted(required_types - types_in_meta)
        if missing_types:
            QMessageBox.warning(
                self,
                "Missing Measurement Types",
                "The following measurement types are missing from your selection:\n\n"
                + ", ".join(missing_types)
                + "\n\nPlease include at least one measurement for each required type before generating HDF5.",
            )
            return

        # 2) Ensure per-alias coverage for each required type
        aliases_in_selection = set()
        for type_map in aux_measurements.values():
            if isinstance(type_map, dict):
                aliases_in_selection.update(type_map.keys())
        aliases_to_check = active_aliases or sorted(aliases_in_selection)

        missing_pairs = []
        for t in sorted(required_types):
            type_map = aux_measurements.get(t, {})
            for a in aliases_to_check:
                if a not in type_map:
                    missing_pairs.append(f"{t} → {a}")

        if missing_pairs:
            QMessageBox.warning(
                self,
                "Incomplete Technical Set",
                "All measurement types must be provided for each detector alias.\n\nMissing combinations:\n"
                + "\n".join(missing_pairs),
            )
            return
        
        # 3) Validate primary selections: max one primary per measurement type per detector
        primary_violations = []
        for typ in required_types:
            for alias in aliases_to_check:
                pair = (typ, alias)
                if pair in primary_measurements:
                    # Count how many rows for this (type, alias) pair are marked as primary
                    primary_count = sum(1 for is_prim in primary_measurements[pair] if is_prim)
                    if primary_count > 1:
                        primary_violations.append(f"{typ} → {alias}: {primary_count} primary files")
        
        if primary_violations:
            QMessageBox.warning(
                self,
                "Primary Selection Error",
                "Each measurement type can have at most ONE primary file per detector.\n\n"
                "Violations found:\n" + "\n".join(primary_violations) +
                "\n\nPlease uncheck some primary selections before generating H5.",
            )
            return

        default_thickness_mm, accepted = self._prompt_technical_thickness_mm(
            default_mm=float(getattr(self, "_last_technical_thickness_mm", 0.0) or 0.0)
        )
        if not accepted:
            self._log_technical_event("Technical HDF5 generation cancelled (thickness prompt)")
            return
        if default_thickness_mm is not None:
            self._last_technical_thickness_mm = float(default_thickness_mm)
            for type_map in aux_measurements.values():
                if not isinstance(type_map, dict):
                    continue
                for alias, entry in list(type_map.items()):
                    if not isinstance(entry, dict):
                        entry = {"file_path": entry}
                    if entry.get("thickness") in (None, ""):
                        entry["thickness"] = float(default_thickness_mm)
                    type_map[alias] = entry

        # Collect PONI data (prefer file selection, fallback to in-memory PONI)
        poni_data = {}
        missing_poni = []
        selected_poni_files = {}
        
        # Check if dev mode is enabled
        dev_mode = self.config.get("DEV", False) if hasattr(self, "config") else False

        if aliases_to_check:
            current_poni_files = getattr(self, "poni_files", {})
            poni_dialog = PoniFileSelectionDialog(
                aliases=sorted(aliases_to_check),
                current_poni_files=current_poni_files,
                parent=self,
            )

            if poni_dialog.exec_() == QDialog.Accepted:
                selected_poni_files = poni_dialog.get_poni_files() or {}
            else:
                res = QMessageBox.question(
                    self,
                    "PONI Files",
                    "Use currently loaded PONI values instead of selecting files?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if res != QMessageBox.Yes:
                    return

        for alias in aliases_to_check:
            poni_content = None
            poni_filename = None

            file_path = selected_poni_files.get(alias)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        poni_content = f.read()
                    poni_filename = os.path.basename(file_path)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "PONI File Read Error",
                        f"Failed to read PONI file for {alias}:\n{file_path}\n\nError: {e}\n\n"
                        "Falling back to current PONI values if available.",
                    )

            if not poni_content:
                try:
                    poni_content = (getattr(self, "ponis", {}) or {}).get(alias)
                    poni_meta = (getattr(self, "poni_files", {}) or {}).get(alias, {})
                    poni_filename = poni_meta.get("name") or f"{alias}.poni"
                except Exception:
                    poni_content = None

            if poni_content:
                poni_data[alias] = (poni_content, poni_filename or f"{alias}.poni")
            else:
                missing_poni.append(alias)
        logger.info(
            "Technical container inputs prepared: aliases=%s types=%s with_poni=%d missing_poni=%d",
            list(aliases_to_check),
            sorted(list(aux_measurements.keys())),
            len(poni_data),
            len(missing_poni),
        )

        if missing_poni:
            res = QMessageBox.question(
                self,
                "Missing PONI Data",
                "No PONI data found for:\n"
                + ", ".join(missing_poni)
                + "\n\nContinue without these PONI datasets?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        # Check if per-detector distances have been configured
        if hasattr(self, '_detector_distances') and self._detector_distances:
            # Use pre-configured per-detector distances (keyed by detector ID)
            # Convert to alias-keyed dict for use in container generation
            user_distances_cm = {}
            for detector_id, distance_cm in self._detector_distances.items():
                # Find detector config by ID to get alias
                detector_config = next(
                    (d for d in self.config.get('detectors', []) if d.get('id') == detector_id),
                    None
                )
                if detector_config:
                    alias = detector_config.get('alias', detector_id)
                    user_distances_cm[alias] = distance_cm
            
            self._log_technical_event(
                f"Using pre-configured per-detector distances: {user_distances_cm}"
            )
            
            # Validate that distances are set for ALL active detectors
            missing_distance_aliases = [a for a in aliases_to_check if a not in user_distances_cm]
            if missing_distance_aliases:
                QMessageBox.warning(
                    self,
                    "Incomplete Distance Configuration",
                    f"Distances must be configured for ALL active detectors.\n\n"
                    f"Missing distances for: {', '.join(missing_distance_aliases)}\n\n"
                    f"Please click 'Distances...' to configure all detector distances.",
                )
                return
        else:
            # No distances configured - require user to configure them first
            if not dev_mode:
                QMessageBox.warning(
                    self,
                    "Distances Not Configured",
                    "Please click the 'Distances...' button to configure detector distances before generating H5 container.",
                )
                return
            
            # Dev mode: use single distance prompt as fallback
            self._log_technical_event("Dev mode: no pre-configured distances, prompting user")
            user_distance_cm = self._prompt_distance_cm(default_cm=17.0)
            if user_distance_cm is None:
                return
            # Convert to dict for uniform processing
            user_distances_cm = {alias: user_distance_cm for alias in aliases_to_check}
        
        # Extract PONI distances per detector for validation
        poni_distances_cm = {}
        for alias in aliases_to_check:
            if alias in poni_data:
                poni_content, _fname = poni_data[alias]
                d = self._parse_poni_distance_m(poni_content)
                if d is not None:
                    poni_distances_cm[alias] = d * 100.0  # Convert meters to cm
        
        # In dev mode, generate fake PONI files matching user distances (within 3%)
        if dev_mode:
            self._log_technical_event(
                f"Dev mode: generating fake PONI files with distances within ±3%: {user_distances_cm}"
            )
            # Generate fake PONIs per detector
            fake_poni_data = {}
            for alias in aliases_to_check:
                distance_cm = user_distances_cm.get(alias, 17.0)
                # Generate single detector fake PONI
                single_poni = self._generate_fake_poni_data([alias], distance_cm)
                if alias in single_poni:
                    fake_poni_data[alias] = single_poni[alias]
            poni_data = fake_poni_data

        # Get technical temp folder for HDF5 generation
        tech_temp_folder = _get_technical_temp_folder(self.config if hasattr(self, "config") else None)
        
        # Generate HDF5 container in temp folder
        try:
            container_id, temp_file_path = technical_container.generate_from_aux_table(
                folder=tech_temp_folder,
                aux_measurements=aux_measurements,
                poni_data=poni_data,
                detector_config=self.config.get("detectors", []),
                active_detector_ids=self._get_active_detector_ids(),
                distances_cm=user_distances_cm,  # Pass per-detector distances dict
                poni_distances_cm=poni_distances_cm if poni_distances_cm else None,  # Pass per-detector PONI distances
                technical_thickness_mm=default_thickness_mm,
                producer_software=str(self.config.get("producer_software") or "difra"),
                producer_version=str(
                    self.config.get("producer_version")
                    or self.config.get("container_version")
                    or "unknown"
                ),
            )
        except Exception as e:
            QMessageBox.critical(
                self, "HDF5 Write Error", f"Failed to generate HDF5 container:\n{e}"
            )
            return
        logger.info(
            "Technical container generated: id=%s temp=%s aliases=%s",
            container_id,
            str(temp_file_path),
            list(aliases_to_check),
        )

        self._log_technical_event(
            f"Technical HDF5 generated in temp: {os.path.basename(temp_file_path)}"
        )
        
        # Archive any existing containers in storage folder before copying new one
        archived_count = self._archive_existing_containers(folder)
        if archived_count > 0:
            self._log_technical_event(
                f"Archived {archived_count} existing container(s) to make room for new one"
            )
        
        # Copy to user-specified storage folder
        import shutil
        try:
            storage_file_path = os.path.join(folder, os.path.basename(temp_file_path))
            shutil.copy2(temp_file_path, storage_file_path)
            self._log_technical_event(
                f"Copied to storage: {os.path.basename(storage_file_path)}"
            )
            final_path = storage_file_path
            logger.info(
                "Technical container copied to storage: id=%s path=%s",
                container_id,
                str(storage_file_path),
            )
        except Exception as e:
            logger.warning(f"Failed to copy to storage folder: {e}")
            self._log_technical_event(
                f"Warning: Could not copy to storage folder, file remains in temp: {temp_file_path}"
            )
            final_path = temp_file_path
            logger.warning(
                "Technical container copy to storage failed, using temp file: id=%s temp=%s error=%s",
                container_id,
                str(temp_file_path),
                str(e),
            )
        
        # Auto-validate container if configured
        should_validate = self.config.get("validate_containers_before_locking", True)
        if should_validate:
            self._log_technical_event("Auto-validating generated container...")
            logger.info(
                "Technical container auto-validation requested: id=%s path=%s",
                container_id,
                str(final_path),
            )
            self._validate_and_prompt_lock(final_path, container_id)
        else:
            QMessageBox.information(
                self,
                "HDF5 Generated",
                f"Container generated successfully!\n\nLocation: {final_path}\n\nContainer ID: {container_id}",
            )

    def _prompt_technical_thickness_mm(self, default_mm: float = 0.0):
        thickness_mm, ok = QInputDialog.getDouble(
            self,
            "Technical Thickness",
            "Enter technical thickness in mm (0 = unknown/not applicable):",
            float(default_mm or 0.0),
            0.0,
            100000.0,
            3,
        )
        if not ok:
            return None, False
        if float(thickness_mm) <= 0.0:
            return None, True
        return float(thickness_mm), True
