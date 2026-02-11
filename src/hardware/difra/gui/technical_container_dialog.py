"""Technical Container Generation Dialog.

Dialog for selecting operator and confirming distance when generating
a technical HDF5 container from auxiliary measurements.
"""

import logging
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
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
    - Distance (required, with PONI validation)
    - Operator (required, with option to add new)
    """
    
    def __init__(
        self,
        operator_manager: OperatorManager,
        poni_distance_cm: Optional[float] = None,
        parent=None
    ):
        """Initialize dialog.
        
        Args:
            operator_manager: Operator manager instance
            poni_distance_cm: Distance from PONI file (if available)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.operator_manager = operator_manager
        self.poni_distance_cm = poni_distance_cm
        self.selected_operator_id: Optional[str] = None
        
        self.setWindowTitle("Generate Technical Container")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(
            "<h3>Technical Container Generation</h3>"
            "Please confirm the distance and select the operator who performed "
            "the technical measurements (DARK, EMPTY, BACKGROUND, etc.)."
        )
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Distance group
        distance_group = QGroupBox("Distance Configuration")
        distance_layout = QFormLayout(distance_group)
        
        # Show PONI distance if available
        if poni_distance_cm is not None:
            poni_label = QLabel(f"<b>{poni_distance_cm:.2f} cm</b>")
            poni_label.setStyleSheet("color: #007acc;")
            distance_layout.addRow("PONI Distance:", poni_label)
            
            default_distance = poni_distance_cm
        else:
            no_poni_label = QLabel("No PONI distance available")
            no_poni_label.setStyleSheet("color: #888; font-style: italic;")
            distance_layout.addRow("PONI Distance:", no_poni_label)
            
            default_distance = 17.0
        
        # Distance input
        distance_label = QLabel(
            "<b>Container Distance (cm)*:</b><br>"
            "<span style='color: #555; font-size: 10px;'>"
            "Distance for this technical container<br>"
            "Should match PONI distance (±5% tolerance)"
            "</span>"
        )
        self.distance_edit = QLineEdit()
        self.distance_edit.setText(str(default_distance))
        self.distance_edit.setPlaceholderText("e.g. 17.0, 25.0, 50.0")
        distance_layout.addRow(distance_label, self.distance_edit)
        
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
    
    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        # Validate distance
        if not self.distance_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Distance",
                "Please enter a distance value.",
            )
            return
        
        try:
            distance_cm = float(self.distance_edit.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Distance",
                "Distance must be a number.",
            )
            return
        
        # Validate against PONI distance if available
        if self.poni_distance_cm is not None:
            tolerance = 0.05  # 5%
            min_dist = self.poni_distance_cm * (1 - tolerance)
            max_dist = self.poni_distance_cm * (1 + tolerance)
            
            if not (min_dist <= distance_cm <= max_dist):
                reply = QMessageBox.warning(
                    self,
                    "Distance Mismatch",
                    f"Entered distance ({distance_cm:.2f} cm) differs from PONI distance "
                    f"({self.poni_distance_cm:.2f} cm) by more than 5%.\n\n"
                    f"Expected range: {min_dist:.2f} - {max_dist:.2f} cm\n\n"
                    f"Continue anyway?",
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
        self.accept()
    
    def get_parameters(self) -> Tuple[float, str]:
        """Get technical container parameters.
        
        Returns:
            Tuple of (distance_cm, operator_id)
        """
        distance_cm = float(self.distance_edit.text())
        return distance_cm, self.selected_operator_id
