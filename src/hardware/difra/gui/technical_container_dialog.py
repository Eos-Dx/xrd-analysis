"""Technical Container Generation Dialog.

Dialog for selecting operator and confirming distance when generating
a technical HDF5 container from auxiliary measurements.
"""

import logging
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from hardware.difra.gui.operator_manager import OperatorManager, NewOperatorDialog

logger = logging.getLogger(__name__)


class TechnicalContainerDialog(QDialog):
    """Dialog for technical container generation parameters.
    
    Prompts for:
    - Per-detector distances (with PONI validation and apply-to-all option)
    - Operator (required, with option to add new)
    
    Supports multiple detectors (e.g., SAXS at 100cm, WAXS at 17cm).
    """
    
    def __init__(
        self,
        operator_manager: OperatorManager,
        detector_configs: List[Dict],
        poni_distances: Optional[Dict[str, float]] = None,
        parent=None
    ):
        """Initialize dialog.
        
        Args:
            operator_manager: Operator manager instance
            detector_configs: List of detector config dicts with 'id', 'alias', etc.
            poni_distances: Dict mapping detector_id to distance from PONI (if available)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.operator_manager = operator_manager
        self.detector_configs = detector_configs
        self.poni_distances = poni_distances or {}
        self.selected_operator_id: Optional[str] = None
        self.distance_edits: Dict[str, QLineEdit] = {}
        self.apply_to_all_checkbox: Optional[QCheckBox] = None
        
        self.setWindowTitle("Generate Technical Container")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(
            "<h3>Technical Container Generation</h3>"
            "Configure distances for each detector and select the operator who performed "
            "the technical measurements (DARK, EMPTY, BACKGROUND, etc.)."
        )
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Distance group
        distance_group = QGroupBox("Detector Distance Configuration")
        distance_layout = QFormLayout(distance_group)
        
        # Apply to all checkbox (shown only if multiple detectors)
        if len(self.detector_configs) > 1:
            self.apply_to_all_checkbox = QCheckBox("Apply first distance to all detectors")
            self.apply_to_all_checkbox.setChecked(False)
            self.apply_to_all_checkbox.stateChanged.connect(self._on_apply_to_all_changed)
            distance_layout.addRow("", self.apply_to_all_checkbox)
            
            # Info label
            info_label = QLabel(
                "<i>Note: WAXS and SAXS typically have different distances</i>"
            )
            info_label.setStyleSheet("color: #888; font-size: 10px;")
            distance_layout.addRow("", info_label)
        
        # Create distance input for each detector
        for detector_config in self.detector_configs:
            detector_id = detector_config.get('id', 'unknown')
            detector_alias = detector_config.get('alias', detector_id)
            
            # Get PONI distance for this detector if available
            poni_dist = self.poni_distances.get(detector_id)
            default_dist = poni_dist if poni_dist else 17.0
            
            # Create label with PONI info
            if poni_dist:
                label_text = f"<b>{detector_alias}*:</b> (PONI: {poni_dist:.2f} cm)"
            else:
                label_text = f"<b>{detector_alias}*:</b>"
            
            label = QLabel(label_text)
            
            # Create distance input
            distance_edit = QLineEdit()
            distance_edit.setText(str(default_dist))
            distance_edit.setPlaceholderText("e.g. 17.0, 25.0, 100.0")
            distance_edit.setProperty("detector_id", detector_id)
            
            # Store reference
            self.distance_edits[detector_id] = distance_edit
            
            distance_layout.addRow(label, distance_edit)
        
        layout.addWidget(distance_group)
        
        # Operator selection group
        operator_group = QGroupBox("Operator Selection")
        operator_layout = QFormLayout(operator_group)
        
        self.operator_combo = QComboBox()
        self.operator_combo.currentIndexChanged.connect(self._on_operator_changed)
        self._populate_operator_combo()
        operator_layout.addRow("Operator*:", self.operator_combo)
        
        # Operator details display
        self.operator_details_label = QLabel()
        self.operator_details_label.setWordWrap(True)
        self.operator_details_label.setStyleSheet(
            "color: #555; background-color: #f0f0f0; padding: 5px; border-radius: 3px; font-size: 10px;"
        )
        operator_layout.addRow("Details:", self.operator_details_label)
        
        # Add new operator button
        new_operator_btn = QPushButton("Add New Operator...")
        new_operator_btn.clicked.connect(self._on_add_new_operator)
        operator_layout.addRow("", new_operator_btn)
        
        layout.addWidget(operator_group)
        
        # Info label
        info_label = QLabel(
            "* Required fields\n\n"
            "<b>Note:</b> The operator information will be stored in the technical "
            "container and can be different from the session container operator."
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Update operator details for initial selection
        self._update_operator_details()
    
    def _populate_operator_combo(self):
        """Populate operator combo box."""
        self.operator_combo.clear()
        
        operators = self.operator_manager.get_all_operators()
        
        if not operators:
            self.operator_combo.addItem("No operators defined", None)
            return
        
        # Add operators
        current_id = self.operator_manager.get_current_operator_id()
        current_index = 0
        
        for i, (op_id, op_info) in enumerate(sorted(operators.items())):
            display_name = self.operator_manager.get_operator_display_name(op_id)
            self.operator_combo.addItem(display_name, op_id)
            
            # Pre-select current operator
            if op_id == current_id:
                current_index = i
        
        if current_id and current_index < self.operator_combo.count():
            self.operator_combo.setCurrentIndex(current_index)
    
    def _update_operator_details(self):
        """Update operator details display."""
        operator_id = self.operator_combo.currentData()
        
        if not operator_id:
            self.operator_details_label.setText("No operator selected")
            return
        
        operator = self.operator_manager.get_operator(operator_id)
        if not operator:
            self.operator_details_label.setText("Operator not found")
            return
        
        details = f"{operator['name']} {operator['surname']} | {operator.get('email', 'N/A')}"
        if operator.get('institution'):
            details += f" | {operator['institution']}"
        
        self.operator_details_label.setText(details)
    
    def _on_operator_changed(self):
        """Handle operator selection change."""
        self._update_operator_details()
    
    def _on_add_new_operator(self):
        """Handle add new operator button."""
        dialog = NewOperatorDialog(self.operator_manager, self)
        
        if dialog.exec_() == QDialog.Accepted:
            new_operator_id = dialog.get_operator_id()
            
            # Refresh combo box
            self._populate_operator_combo()
            
            # Select the new operator
            for i in range(self.operator_combo.count()):
                if self.operator_combo.itemData(i) == new_operator_id:
                    self.operator_combo.setCurrentIndex(i)
                    break
    
    def _on_apply_to_all_changed(self, state):
        """Handle apply-to-all checkbox state change."""
        if not self.distance_edits:
            return
        
        # Get first detector distance
        first_detector_id = list(self.distance_edits.keys())[0]
        first_distance = self.distance_edits[first_detector_id].text()
        
        if state == Qt.Checked:
            # Apply first distance to all other detectors
            for i, detector_id in enumerate(self.distance_edits.keys()):
                if i == 0:
                    continue  # Skip first
                
                self.distance_edits[detector_id].setText(first_distance)
                self.distance_edits[detector_id].setEnabled(False)
        else:
            # Re-enable all distance inputs
            for i, (detector_id, edit) in enumerate(self.distance_edits.items()):
                if i > 0:
                    edit.setEnabled(True)
    
    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        # Validate all detector distances
        distances = {}
        warnings = []
        
        for detector_id, distance_edit in self.distance_edits.items():
            distance_text = distance_edit.text().strip()
            
            if not distance_text:
                detector_alias = next(
                    (d.get('alias', d['id']) for d in self.detector_configs if d['id'] == detector_id),
                    detector_id
                )
                QMessageBox.warning(
                    self,
                    "Missing Distance",
                    f"Please enter a distance for {detector_alias}.",
                )
                return
            
            try:
                distance_cm = float(distance_text)
            except ValueError:
                detector_alias = next(
                    (d.get('alias', d['id']) for d in self.detector_configs if d['id'] == detector_id),
                    detector_id
                )
                QMessageBox.warning(
                    self,
                    "Invalid Distance",
                    f"Distance for {detector_alias} must be a number.",
                )
                return
            
            distances[detector_id] = distance_cm
            
            # Validate against PONI distance if available
            poni_dist = self.poni_distances.get(detector_id)
            if poni_dist is not None:
                tolerance = 0.05  # 5%
                min_dist = poni_dist * (1 - tolerance)
                max_dist = poni_dist * (1 + tolerance)
                
                if not (min_dist <= distance_cm <= max_dist):
                    detector_alias = next(
                        (d.get('alias', d['id']) for d in self.detector_configs if d['id'] == detector_id),
                        detector_id
                    )
                    warnings.append(
                        f"{detector_alias}: {distance_cm:.2f} cm (PONI: {poni_dist:.2f} cm, "
                        f"expected: {min_dist:.2f}-{max_dist:.2f} cm)"
                    )
        
        # Show warnings if any distances mismatch PONI
        if warnings:
            warning_msg = "The following distances differ from PONI by more than 5%:\n\n"
            warning_msg += "\n".join(warnings)
            warning_msg += "\n\nContinue anyway?"
            
            reply = QMessageBox.warning(
                self,
                "Distance Mismatch",
                warning_msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # Validate operator selection
        operator_id = self.operator_combo.currentData()
        if not operator_id:
            QMessageBox.warning(
                self,
                "No Operator Selected",
                "Please select an operator or add a new one.",
            )
            return
        
        self.selected_operator_id = operator_id
        self.detector_distances = distances
        self.accept()
    
    def get_parameters(self) -> Tuple[Dict[str, float], str]:
        """Get technical container parameters.
        
        Returns:
            Tuple of (distances_dict, operator_id) where:
            - distances_dict: Dict mapping detector_id to distance_cm
            - operator_id: Selected operator ID
        """
        return self.detector_distances, self.selected_operator_id
