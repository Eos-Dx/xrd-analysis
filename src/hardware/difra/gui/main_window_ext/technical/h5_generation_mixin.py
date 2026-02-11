"""H5 Generation Mixin - Technical container and metadata generation."""
import json
import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

# Import Qt for type hints and usage
try:
    from PyQt5.QtWidgets import QComboBox, QDialog, QFileDialog, QInputDialog, QMessageBox, QCheckBox
    from PyQt5.QtCore import Qt
except ImportError:
    # Test stubs
    class QComboBox:
        def __init__(self, *args, **kwargs):
            pass
    
    class QDialog:
        Accepted = 1
        Rejected = 0
    
    class QFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return "", ""
    
    class QInputDialog:
        @staticmethod
        def getDouble(*args, **kwargs):
            return 17.0, True
    
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
            pass
        
        @staticmethod
        def critical(*args, **kwargs):
            pass
    
    class QCheckBox:
        def __init__(self, *args, **kwargs):
            pass
        
        def isChecked(self):
            return False
    
    class Qt:
        UserRole = 32


class H5GenerationMixin:
    """Mixin for H5 container and metadata generation operations.
    
    Handles:
    - Technical metadata JSON generation
    - Technical HDF5 container generation
    - PONI data parsing and fake generation
    - Distance prompting
    """
    
    def _parse_poni_distance_m(self, poni_text: str):
        """Parse Distance from PONI text in meters. Returns None if not found/invalid."""
        if not poni_text:
            return None
        try:
            m = re.search(r"^Distance:\s*([0-9.eE+-]+)", poni_text, flags=re.MULTILINE)
            return float(m.group(1)) if m else None
        except Exception:
            return None
    
    def _generate_fake_poni_data(self, aliases, user_distance_cm):
        """Generate fake PONI data for dev mode with distances within ±3% of user value.
        
        Args:
            aliases: List of detector aliases
            user_distance_cm: User-specified distance in cm
        
        Returns:
            Dict mapping alias to tuple of (poni_content, poni_filename)
        """
        import random
        import time
        
        pony_data = {}
        
        for alias in aliases:
            # Get detector config
            detector_config = None
            for d in self.config.get("detectors", []):
                if d.get("alias") == alias:
                    detector_config = d
                    break
            
            if not detector_config:
                detector_config = {"alias": alias}
            
            # Generate distance within ±3% margin (inside the 5% validation tolerance)
            random.seed(hash(alias))  # Consistent values for same detector
            margin = random.uniform(-0.03, 0.03)
            fake_distance_m = (user_distance_cm / 100.0) * (1 + margin)
            
            # Get detector size or use defaults
            size = detector_config.get("size", {"width": 256, "height": 256})
            width = size.get("width", 256)
            height = size.get("height", 256)
            
            # Generate slightly different parameters for each detector
            poni1 = round(random.uniform(0.005, 0.010), 6)
            poni2 = round(random.uniform(0.0008, 0.0030), 6)
            
            # Generate pixel sizes (typically 55um or 100um)
            pixel_size = detector_config.get("pixel_size_um", [55, 55])
            pixel1 = pixel_size[0] * 1e-6 if len(pixel_size) > 0 else 5.5e-05
            pixel2 = pixel_size[1] * 1e-6 if len(pixel_size) > 1 else 5.5e-05
            
            wavelength = 1.5406e-10  # Typical Cu Kα wavelength
            
            current_time = time.strftime("%a %b %d %H:%M:%S %Y")
            
            poni_content = f"""# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis
# Calibration done on {current_time} (DEV MODE - FAKE DATA)
poni_version: 2.1
Detector: Detector
Detector_config: {{"pixel1": {pixel1}, "pixel2": {pixel2}, "max_shape": [{height}, {width}], "orientation": 3}}
Distance: {fake_distance_m}
Poni1: {poni1}
Poni2: {poni2}
Rot1: 0
Rot2: 0
Rot3: 0
Wavelength: {wavelength}
# Calibrant: AgBh (DEV MODE)
# Detector: {alias} (DEV MODE - FAKE DATA)
# User specified: {user_distance_cm:.2f} cm, Generated: {fake_distance_m*100:.2f} cm (margin: {margin*100:.1f}%)
"""
            
            poni_filename = f"{alias.lower()}_fake_h5gen.poni"
            pony_data[alias] = (poni_content, poni_filename)
            
            logger.info(
                f"Generated fake PONI for {alias}: distance={fake_distance_m*100:.2f} cm "
                f"(user: {user_distance_cm:.2f} cm, margin: {margin*100:.1f}%)"
            )
        
        return pony_data

    def _prompt_distance_cm(self, default_cm: float = None):
        """Prompt user for sample-detector distance in cm. Returns None if canceled."""
        default_val = 17.0 if default_cm is None else float(default_cm)
        dist_cm, ok = QInputDialog.getDouble(
            self,
            "Sample-Detector Distance",
            "Enter sample-detector distance (cm):",
            default_val,
            0.01,
            100000.0,
            3,
        )
        return float(dist_cm) if ok else None
    
    def generate_technical_meta(self):
        """Generate technical metadata JSON file from selected measurements."""
        from pathlib import Path

        self._log_technical_event("Generating technical metadata...")

        # Validate selection
        sel = (
            self.auxTable.selectionModel().selectedRows()
            if self.auxTable.selectionModel()
            else []
        )
        rows = [idx.row() for idx in sel]
        if not rows:
            self._log_technical_event("Error: No rows selected for metadata generation")
            QMessageBox.warning(
                self, "No Selection", "Select one or more rows in the Aux table."
            )
            return

        # Validate name and folder
        name = (self.auxNameLE.text() or "").strip()
        if not name:
            QMessageBox.warning(
                self, "Missing Name", "Enter a name in the Aux Measurement field."
            )
            return
        safe_name = name.replace(" ", "_")
        # Use the explicit folder path for meta generation.
        # (Do not auto-fallback to CWD; if the folder is invalid/unwritable we should stop.)
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

        out_path = os.path.join(folder, f"technical_meta_{safe_name}.json")
        if os.path.exists(out_path):
            res = QMessageBox.question(
                self,
                "Overwrite?",
                f"File exists:\n{out_path}\n\nDo you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        meta = {}
        seen_pairs = set()  # (type, alias)

        # Get active detector aliases for validation
        try:
            active_aliases = self._get_active_detector_aliases()
        except Exception:
            active_aliases = []

        for row in rows:
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
            typ = type_cb.currentText()

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
            al = cb.currentText()

            base = os.path.basename(file_path)
            dst = meta.setdefault(typ, {})
            pair = (typ, al)
            if pair in seen_pairs or al in dst:
                QMessageBox.warning(
                    self,
                    "Duplicate Assignment",
                    f"Measurement for type '{typ}' and alias '{al}' is already assigned.",
                )
                return
            dst[al] = base
            seen_pairs.add(pair)

        # Enforce completeness: all REQUIRED measurement types must be present, and for each alias
        required_types = set(
            getattr(self, "REQUIRED_TYPE_OPTIONS", None)
            or ["AGBH", "DARK", "EMPTY", "BACKGROUND"]
        )

        # 1) Ensure at least one row selected for each required type
        types_in_meta = {t for t in meta.keys() if t in required_types}
        missing_types = sorted(required_types - types_in_meta)
        if missing_types:
            QMessageBox.warning(
                self,
                "Missing Measurement Types",
                "The following measurement types are missing from your selection:\n\n"
                + ", ".join(missing_types)
                + "\n\nPlease include at least one measurement for each required type before generating the meta file.",
            )
            return

        # 2) Ensure per-alias coverage for each required type
        # Prefer active aliases from config; if unavailable, fall back to aliases seen in selection
        aliases_in_selection = set()
        for type_map in meta.values():
            if isinstance(type_map, dict):
                aliases_in_selection.update(type_map.keys())
        aliases_to_check = active_aliases or sorted(aliases_in_selection)

        missing_pairs = []
        for t in sorted(required_types):
            type_map = meta.get(t, {})
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

        # Get unique aliases from selected measurements for PONI file selection
        unique_aliases = set()
        for row in rows:
            cb = self.auxTable.cellWidget(row, 3)
            if (
                isinstance(cb, QComboBox)
                and cb.currentText() != self.NO_SELECTION_LABEL
            ):
                unique_aliases.add(cb.currentText())

        # Show PONI file selection dialog if we have aliases
        poni_lab = {}
        if unique_aliases:
            # Import here to avoid circular dependency
            from hardware.difra.gui.main_window_ext.technical_measurements import PoniFileSelectionDialog
            
            # Get current PONI files if available
            current_poni_files = getattr(self, "poni_files", {})

            poni_dialog = PoniFileSelectionDialog(
                aliases=sorted(unique_aliases),
                current_poni_files=current_poni_files,
                parent=self,
            )

            if poni_dialog.exec_() == QDialog.Accepted:
                selected_poni_files = poni_dialog.get_poni_files()
                poni_lab_path = {}
                poni_lab_values = {}

                # Process each selected PONI file
                for alias, file_path in selected_poni_files.items():
                    # Store filename for PONI_LAB
                    poni_lab[alias] = os.path.basename(file_path)

                    # Store full path for PONI_LAB_PATH
                    poni_lab_path[alias] = file_path

                    # Read and store PONI file content for PONI_LAB_VALUES
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            poni_content = f.read()
                            poni_lab_values[alias] = poni_content
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "PONI File Read Error",
                            f"Failed to read PONI file for {alias}:\n{file_path}\n\nError: {e}\n\nContinuing without this PONI file content.",
                        )
                        # Still include the filename and path, but mark content as unavailable
                        poni_lab_values[alias] = (
                            f"# ERROR: Could not read PONI file content\n# File: {file_path}\n# Error: {str(e)}"
                        )

                # Store additional PONI data for later use
                self._temp_poni_lab_path = poni_lab_path
                self._temp_poni_lab_values = poni_lab_values
            else:
                # User cancelled PONI selection, ask if they want to continue without PONI files
                res = QMessageBox.question(
                    self,
                    "No PONI Files Selected",
                    "Do you want to generate the technical meta file without PONI calibration files?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res != QMessageBox.Yes:
                    return

        # Add PONI sections to meta if any PONI files were selected
        if poni_lab:
            meta["PONI_LAB"] = poni_lab

        # Add PONI_LAB_PATH section if available
        if hasattr(self, "_temp_poni_lab_path") and self._temp_poni_lab_path:
            meta["PONI_LAB_PATH"] = self._temp_poni_lab_path

        # Add PONI_LAB_VALUES section if available
        if hasattr(self, "_temp_poni_lab_values") and self._temp_poni_lab_values:
            meta["PONI_LAB_VALUES"] = self._temp_poni_lab_values

        # Add or reuse a calibration group hash so multiple files can be grouped together
        try:
            group_hash = getattr(self, "calibration_group_hash", None)
            if not group_hash:
                group_hash = uuid.uuid4().hex[:16]
                setattr(self, "calibration_group_hash", group_hash)
            meta["CALIBRATION_GROUP_HASH"] = group_hash
        except Exception:
            # Non-fatal; proceed without the group hash if something unexpected happens
            pass

        # Write JSON
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            QMessageBox.critical(
                self, "Write Error", f"Failed to write meta file:\n{e}"
            )
            return
        
        # Clean up temporary PONI data variables
        if hasattr(self, "_temp_poni_lab_path"):
            delattr(self, "_temp_poni_lab_path")
        if hasattr(self, "_temp_poni_lab_values"):
            delattr(self, "_temp_poni_lab_values")

        # Summary
        try:
            summary_lines = []
            for k, v in meta.items():
                if isinstance(v, dict):
                    summary_lines.append(f"{k}: {len(v)} file(s)")
                else:
                    summary_lines.append(f"{k}: {v}")
            summary = "\n".join(summary_lines) or "(empty)"
        except Exception:
            summary = "(summary unavailable)"
        
        self._log_technical_event(
            f"Technical metadata generated: {os.path.basename(out_path)}"
        )
        
        QMessageBox.information(
            self, "Meta Generated", f"Saved to:\n{out_path}\n\nSummary:\n{summary}"
        )
        
        # Now generate HDF5 container
        self.generate_technical_h5()

    def generate_technical_h5(self):
        """Generate technical HDF5 container from measurements in aux table."""
        from hardware.container.v0_1 import schema, technical_container
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

        aux_measurements = {}
        primary_measurements = {}  # Track which measurements are marked as primary: {(type, alias): [is_prim1, is_prim2, ...]}

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
                aux_measurements.setdefault(typ, {})[alias] = file_path
            
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
            or ["AGBH", "DARK", "EMPTY", "BACKGROUND"]
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

        # Collect PONI data (prefer file selection, fallback to in-memory PONI)
        pony_data = {}
        missing_pony = []
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
                pony_data[alias] = (poni_content, poni_filename or f"{alias}.poni")
            else:
                missing_pony.append(alias)

        if missing_pony:
            res = QMessageBox.question(
                self,
                "Missing PONI Data",
                "No PONI data found for:\n"
                + ", ".join(missing_pony)
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
            if alias in pony_data:
                poni_content, _fname = pony_data[alias]
                d = self._parse_poni_distance_m(poni_content)
                if d is not None:
                    poni_distances_cm[alias] = d * 100.0  # Convert meters to cm
        
        # In dev mode, generate fake PONI files matching user distances (within 3%)
        if dev_mode:
            self._log_technical_event(
                f"Dev mode: generating fake PONI files with distances within ±3%: {user_distances_cm}"
            )
            # Generate fake PONIs per detector
            fake_pony_data = {}
            for alias in aliases_to_check:
                distance_cm = user_distances_cm.get(alias, 17.0)
                # Generate single detector fake PONI
                single_pony = self._generate_fake_poni_data([alias], distance_cm)
                if alias in single_pony:
                    fake_pony_data[alias] = single_pony[alias]
            pony_data = fake_pony_data

        # Get technical temp folder for HDF5 generation
        tech_temp_folder = _get_technical_temp_folder(self.config if hasattr(self, "config") else None)
        
        # Generate HDF5 container in temp folder
        try:
            container_id, temp_file_path = technical_container.generate_from_aux_table(
                folder=tech_temp_folder,
                aux_measurements=aux_measurements,
                pony_data=pony_data,
                detector_config=self.config.get("detectors", []),
                active_detector_ids=self._get_active_detector_ids(),
                distances_cm=user_distances_cm,  # Pass per-detector distances dict
                poni_distances_cm=poni_distances_cm if poni_distances_cm else None,  # Pass per-detector PONI distances
            )
        except Exception as e:
            QMessageBox.critical(
                self, "HDF5 Write Error", f"Failed to generate HDF5 container:\n{e}"
            )
            return

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
        except Exception as e:
            logger.warning(f"Failed to copy to storage folder: {e}")
            self._log_technical_event(
                f"Warning: Could not copy to storage folder, file remains in temp: {temp_file_path}"
            )
            final_path = temp_file_path
        
        # Auto-validate container if configured
        should_validate = self.config.get("validate_containers_before_locking", True)
        if should_validate:
            self._log_technical_event("Auto-validating generated container...")
            self._validate_and_prompt_lock(final_path, container_id)
        else:
            QMessageBox.information(
                self,
                "HDF5 Generated",
                f"Container generated successfully!\n\nLocation: {final_path}\n\nContainer ID: {container_id}",
            )
