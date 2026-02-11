"""Detector Distance Configuration Dialog.

Dialog for setting detector distances before starting technical measurements.
User must configure distances for all detectors before capturing DARK, EMPTY, etc.
"""

import logging
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class DetectorDistanceConfigDialog(QDialog):
    """Dialog for configuring detector distances before technical measurements.
    
    User must set distances for all active detectors. Can use apply-to-all
    for convenience when all detectors are at the same distance.
    """
    
    def __init__(
        self,
        detector_configs: List[Dict],
        current_distances: Optional[Dict[str, float]] = None,
        parent=None
    ):
        """Initialize dialog.
        
        Args:
            detector_configs: List of detector config dicts with 'id', 'alias', etc.
            current_distances: Dict mapping detector_id to current distance (if set)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.detector_configs = detector_configs
        self.current_distances = current_distances or {}
        self.distance_edits: Dict[str, QLineEdit] = {}
        self.apply_to_all_checkbox: Optional[QCheckBox] = None
        self.detector_distances: Dict[str, float] = {}
        
        self.setWindowTitle("Configure Detector Distances")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(
            "<h3>Configure Detector Distances</h3>"
            "Set the sample-to-detector distance for each detector before "
            "starting technical measurements (DARK, EMPTY, BACKGROUND, AgBH)."
        )
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Distance group
        distance_group = QGroupBox("Detector Distances")
        distance_layout = QFormLayout(distance_group)
        
        # Apply to all checkbox (shown only if multiple detectors)
        if len(self.detector_configs) > 1:
            self.apply_to_all_checkbox = QCheckBox("Use same distance for all detectors")
            self.apply_to_all_checkbox.setChecked(False)
            self.apply_to_all_checkbox.stateChanged.connect(self._on_apply_to_all_changed)
            distance_layout.addRow("", self.apply_to_all_checkbox)
            
            # Info label
            info_label = QLabel(
                "<i>Note: WAXS and SAXS typically have different distances.<br>"
                "Check this box only if all detectors are at the same distance.</i>"
            )
            info_label.setStyleSheet("color: #888; font-size: 10px;")
            distance_layout.addRow("", info_label)
        
        # Create distance input for each detector
        for detector_config in self.detector_configs:
            detector_id = detector_config.get('id', 'unknown')
            detector_alias = detector_config.get('alias', detector_id)
            
            # Get current distance if available
            current_dist = self.current_distances.get(detector_id, 17.0)
            
            # Create label
            label = QLabel(f"<b>{detector_alias} (cm)*:</b>")
            
            # Create distance input
            distance_edit = QLineEdit()
            distance_edit.setText(str(current_dist))
            distance_edit.setPlaceholderText("e.g. 17.0, 25.0, 100.0")
            distance_edit.setProperty("detector_id", detector_id)
            
            # Store reference
            self.distance_edits[detector_id] = distance_edit
            
            distance_layout.addRow(label, distance_edit)
        
        layout.addWidget(distance_group)
        
        # Info label
        info_label = QLabel(
            "* Required fields\n\n"
            "<b>Important:</b> These distances will be stored with your technical "
            "measurements and must match the actual detector positions.\n\n"
            "Technical measurements will be associated with these distances "
            "in the HDF5 container."
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
            
            # Also connect signal to sync when first changes
            first_edit = self.distance_edits[first_detector_id]
            first_edit.textChanged.connect(self._sync_distances)
        else:
            # Re-enable all distance inputs
            for i, (detector_id, edit) in enumerate(self.distance_edits.items()):
                if i > 0:
                    edit.setEnabled(True)
            
            # Disconnect signal
            first_edit = self.distance_edits[first_detector_id]
            try:
                first_edit.textChanged.disconnect(self._sync_distances)
            except TypeError:
                pass  # Not connected
    
    def _sync_distances(self, text):
        """Sync first distance to all others when apply-to-all is checked."""
        if self.apply_to_all_checkbox and self.apply_to_all_checkbox.isChecked():
            for i, edit in enumerate(self.distance_edits.values()):
                if i > 0:
                    edit.setText(text)
    
    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        # Validate all detector distances
        distances = {}
        
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
                if distance_cm <= 0:
                    raise ValueError("Distance must be positive")
            except ValueError as e:
                detector_alias = next(
                    (d.get('alias', d['id']) for d in self.detector_configs if d['id'] == detector_id),
                    detector_id
                )
                QMessageBox.warning(
                    self,
                    "Invalid Distance",
                    f"Distance for {detector_alias} must be a positive number.",
                )
                return
            
            distances[detector_id] = distance_cm
        
        self.detector_distances = distances
        self.accept()
    
    def get_distances(self) -> Dict[str, float]:
        """Get configured detector distances.
        
        Returns:
            Dict mapping detector_id to distance_cm
        """
        return self.detector_distances
