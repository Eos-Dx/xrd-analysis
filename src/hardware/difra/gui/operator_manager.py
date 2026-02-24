"""Operator Management for DIFRA.

Manages operator information stored in JSON format with contact details.
Provides dialog for operator selection/creation on startup.
"""

import json
import hashlib
import hmac
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)

DEFAULT_MODIFICATION_PASSWORD_HASH = (
    "64ae5ac9f98ac4a2bb67a66cc913909022d4d0bb7d673fcf76d1999c33debd93"
)


def _hash_password(password: str) -> str:
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


class OperatorManager:
    """Manages operator database and selection."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize operator manager.
        
        Args:
            config_path: Path to operator JSON file. If None, uses default location.
        """
        if config_path is None:
            # Default to config directory in resources
            config_path = Path(__file__).parent.parent / "resources" / "config" / "operators.json"
        
        self.config_path = Path(config_path)
        self.operators: Dict[str, Dict[str, str]] = {}
        self.current_operator_id: Optional[str] = None
        self.operator_modify_password_hash: str = DEFAULT_MODIFICATION_PASSWORD_HASH
        
        # Load operators from file
        self.load_operators()
    
    def load_operators(self) -> None:
        """Load operators from JSON file."""
        if not self.config_path.exists():
            logger.info(f"Operator config not found, creating default: {self.config_path}")
            self.create_default_operators()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.operators = data.get('operators', {})
                self.current_operator_id = data.get('current_operator_id')
                loaded_hash = data.get("operator_modify_password_hash")
                if isinstance(loaded_hash, str) and loaded_hash.strip():
                    self.operator_modify_password_hash = loaded_hash.strip()
                else:
                    self.operator_modify_password_hash = DEFAULT_MODIFICATION_PASSWORD_HASH
                    # Persist upgraded config format (without plaintext password).
                    self.save_operators()
            
            logger.info(f"Loaded {len(self.operators)} operators from {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to load operators: {e}", exc_info=True)
            QMessageBox.warning(
                None,
                "Operator Config Error",
                f"Failed to load operator configuration:\n{e}\n\nCreating default config.",
            )
            self.create_default_operators()
    
    def create_default_operators(self) -> None:
        """Create default operators file with example operator."""
        self.operators = {
            "default_operator": {
                "name": "Default Operator",
                "surname": "User",
                "email": "operator@example.com",
                "phone": "",
                "institution": "",
            }
        }
        self.current_operator_id = "default_operator"
        self.operator_modify_password_hash = DEFAULT_MODIFICATION_PASSWORD_HASH
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        self.save_operators()
        logger.info(f"Created default operator config: {self.config_path}")
    
    def save_operators(self) -> None:
        """Save operators to JSON file."""
        try:
            data = {
                "operators": self.operators,
                "current_operator_id": self.current_operator_id,
                "operator_modify_password_hash": self.operator_modify_password_hash,
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved operators to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save operators: {e}", exc_info=True)
            raise
    
    def get_operator(self, operator_id: str) -> Optional[Dict[str, str]]:
        """Get operator information by ID.
        
        Args:
            operator_id: Operator identifier
            
        Returns:
            Operator dict with name, surname, email, etc., or None if not found
        """
        return self.operators.get(operator_id)
    
    def get_all_operators(self) -> Dict[str, Dict[str, str]]:
        """Get all operators.
        
        Returns:
            Dict mapping operator_id to operator info
        """
        return self.operators.copy()
    
    def add_operator(
        self,
        operator_id: str,
        name: str,
        surname: str,
        email: str,
        phone: str = "",
        institution: str = "",
    ) -> None:
        """Add or update an operator.
        
        Args:
            operator_id: Unique operator identifier (e.g., username)
            name: First name
            surname: Last name
            email: Email address
            phone: Phone number (optional)
            institution: Institution/organization (optional)
        """
        self.operators[operator_id] = {
            "name": name,
            "surname": surname,
            "email": email,
            "phone": phone,
            "institution": institution,
        }
        
        self.save_operators()
        logger.info(f"Added/updated operator: {operator_id} ({name} {surname})")
    
    def remove_operator(self, operator_id: str) -> bool:
        """Remove an operator.
        
        Args:
            operator_id: Operator to remove
            
        Returns:
            True if removed, False if not found
        """
        if operator_id in self.operators:
            del self.operators[operator_id]
            
            # Clear current if it was this operator
            if self.current_operator_id == operator_id:
                self.current_operator_id = None
            
            self.save_operators()
            logger.info(f"Removed operator: {operator_id}")
            return True
        
        return False
    
    def set_current_operator(self, operator_id: str) -> bool:
        """Set the current operator.
        
        Args:
            operator_id: Operator ID to set as current
            
        Returns:
            True if set successfully, False if operator not found
        """
        if operator_id not in self.operators:
            logger.warning(f"Cannot set current operator - not found: {operator_id}")
            return False
        
        self.current_operator_id = operator_id
        self.save_operators()
        logger.info(f"Set current operator: {operator_id}")
        return True
    
    def get_current_operator(self) -> Optional[Dict[str, str]]:
        """Get current operator information.
        
        Returns:
            Current operator dict, or None if not set
        """
        if self.current_operator_id:
            return self.operators.get(self.current_operator_id)
        return None
    
    def get_current_operator_id(self) -> Optional[str]:
        """Get current operator ID.
        
        Returns:
            Current operator ID, or None if not set
        """
        return self.current_operator_id
    
    def get_operator_display_name(self, operator_id: str) -> str:
        """Get display name for operator.
        
        Args:
            operator_id: Operator ID
            
        Returns:
            Formatted name (e.g., "John Doe (john@example.com)")
        """
        operator = self.get_operator(operator_id)
        if operator:
            name = f"{operator['name']} {operator['surname']}"
            email = operator.get('email', '')
            if email:
                return f"{name} ({email})"
            return name
        return operator_id

    def verify_modify_password(self, password: str) -> bool:
        if not password:
            return False
        expected = str(self.operator_modify_password_hash or "").strip()
        provided = _hash_password(password)
        return bool(expected) and hmac.compare_digest(expected, provided)


class OperatorSelectionDialog(QDialog):
    """Dialog for selecting or creating an operator on startup."""
    
    def __init__(self, operator_manager: OperatorManager, parent=None):
        super().__init__(parent)
        
        self.operator_manager = operator_manager
        self.selected_operator_id: Optional[str] = None
        
        self.setWindowTitle("Select Operator")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Welcome message
        welcome_label = QLabel(
            "Welcome to DIFRA!\n\n"
            "Please select your operator profile or create a new one.\n"
            "This information will be stored with your measurements."
        )
        welcome_label.setWordWrap(True)
        layout.addWidget(welcome_label)
        
        # Operator selection group
        select_group = QGroupBox("Select Existing Operator")
        select_layout = QFormLayout(select_group)
        
        self.operator_combo = QComboBox()
        self._populate_operator_combo()
        select_layout.addRow("Operator:", self.operator_combo)
        
        # Operator details display
        self.operator_details_label = QLabel()
        self.operator_details_label.setWordWrap(True)
        self.operator_details_label.setStyleSheet(
            "color: #555; background-color: #f0f0f0; padding: 8px; border-radius: 4px;"
        )
        select_layout.addRow("Details:", self.operator_details_label)
        self.operator_combo.currentIndexChanged.connect(self._on_operator_selected)
        
        layout.addWidget(select_group)
        
        # New operator button
        new_operator_btn = QPushButton("Create New Operator...")
        new_operator_btn.clicked.connect(self._on_create_new_operator)
        layout.addWidget(new_operator_btn)

        # Modify selected operator button
        edit_operator_btn = QPushButton("Modify Selected Operator...")
        edit_operator_btn.clicked.connect(self._on_edit_operator)
        layout.addWidget(edit_operator_btn)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Update details for initial selection
        self._update_operator_details()
    
    def _populate_operator_combo(self):
        """Populate the operator combo box."""
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
            
            if op_id == current_id:
                current_index = i
        
        # Select current operator if set
        if current_id and current_index < self.operator_combo.count():
            self.operator_combo.setCurrentIndex(current_index)
    
    def _update_operator_details(self):
        """Update operator details display."""
        if not hasattr(self, "operator_details_label"):
            return

        operator_id = self.operator_combo.currentData()
        
        if not operator_id:
            self.operator_details_label.setText("No operator selected")
            return
        
        operator = self.operator_manager.get_operator(operator_id)
        if not operator:
            self.operator_details_label.setText("Operator not found")
            return
        
        details = f"<b>{operator['name']} {operator['surname']}</b><br>"
        details += f"Email: {operator.get('email', 'N/A')}<br>"
        
        if operator.get('phone'):
            details += f"Phone: {operator['phone']}<br>"
        if operator.get('institution'):
            details += f"Institution: {operator['institution']}<br>"
        
        details += f"<br><i>Operator ID: {operator_id}</i>"
        
        self.operator_details_label.setText(details)
    
    def _on_operator_selected(self):
        """Handle operator selection change."""
        self._update_operator_details()
    
    def _on_create_new_operator(self):
        """Handle create new operator button."""
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

    def _on_edit_operator(self):
        operator_id = self.operator_combo.currentData()
        if not operator_id:
            QMessageBox.warning(self, "No Operator Selected", "Please select an operator to modify.")
            return

        dialog = NewOperatorDialog(
            self.operator_manager,
            self,
            existing_operator_id=str(operator_id),
        )
        if dialog.exec_() == QDialog.Accepted:
            updated_operator_id = dialog.get_operator_id() or str(operator_id)
            self._populate_operator_combo()
            for i in range(self.operator_combo.count()):
                if self.operator_combo.itemData(i) == updated_operator_id:
                    self.operator_combo.setCurrentIndex(i)
                    break
    
    def _on_accept(self):
        """Validate and accept."""
        operator_id = self.operator_combo.currentData()
        
        if not operator_id:
            QMessageBox.warning(
                self,
                "No Operator Selected",
                "Please select an operator or create a new one.",
            )
            return
        
        self.selected_operator_id = operator_id
        
        # Set as current operator
        self.operator_manager.set_current_operator(operator_id)
        
        self.accept()
    
    def get_selected_operator_id(self) -> Optional[str]:
        """Get the selected operator ID.
        
        Returns:
            Selected operator ID, or None if cancelled
        """
        return self.selected_operator_id


class NewOperatorDialog(QDialog):
    """Dialog for creating a new operator."""
    
    def __init__(
        self,
        operator_manager: OperatorManager,
        parent=None,
        existing_operator_id: Optional[str] = None,
    ):
        super().__init__(parent)
        
        self.operator_manager = operator_manager
        self.new_operator_id: Optional[str] = None
        self._existing_operator_id: Optional[str] = existing_operator_id
        
        self.setWindowTitle("Modify Operator" if existing_operator_id else "Create New Operator")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Operator ID (username)
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("e.g., john_doe, operator_123")
        form_layout.addRow("Operator ID*:", self.id_edit)
        
        # Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., John")
        form_layout.addRow("First Name*:", self.name_edit)
        
        # Surname
        self.surname_edit = QLineEdit()
        self.surname_edit.setPlaceholderText("e.g., Doe")
        form_layout.addRow("Last Name*:", self.surname_edit)
        
        # Email
        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("e.g., john.doe@example.com")
        form_layout.addRow("Email*:", self.email_edit)
        
        # Phone (optional)
        self.phone_edit = QLineEdit()
        self.phone_edit.setPlaceholderText("Optional")
        form_layout.addRow("Phone:", self.phone_edit)
        
        # Institution (optional)
        self.institution_edit = QLineEdit()
        self.institution_edit.setPlaceholderText("Optional")
        form_layout.addRow("Institution:", self.institution_edit)
        
        layout.addLayout(form_layout)
        
        # Info label
        info_label = QLabel("* Required fields")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if self._existing_operator_id:
            self._load_operator_for_edit(self._existing_operator_id)

    def _load_operator_for_edit(self, operator_id: str) -> None:
        operator = self.operator_manager.get_operator(operator_id)
        if not operator:
            return
        self.id_edit.setText(operator_id)
        self.id_edit.setReadOnly(True)
        self.name_edit.setText(str(operator.get("name", "")))
        self.surname_edit.setText(str(operator.get("surname", "")))
        self.email_edit.setText(str(operator.get("email", "")))
        self.phone_edit.setText(str(operator.get("phone", "")))
        self.institution_edit.setText(str(operator.get("institution", "")))

    def _confirm_modify_password(self) -> bool:
        password, ok = QInputDialog.getText(
            self,
            "Password Required",
            "Enter password to modify operator data:",
            QLineEdit.Password,
        )
        if not ok:
            return False
        if not self.operator_manager.verify_modify_password(password):
            QMessageBox.warning(self, "Invalid Password", "Incorrect password.")
            return False
        return True
    
    def _on_accept(self):
        """Validate and accept."""
        # Validate required fields
        operator_id = self.id_edit.text().strip()
        name = self.name_edit.text().strip()
        surname = self.surname_edit.text().strip()
        email = self.email_edit.text().strip()
        
        if not operator_id:
            QMessageBox.warning(self, "Missing Field", "Please enter an Operator ID.")
            return
        
        if not name:
            QMessageBox.warning(self, "Missing Field", "Please enter a First Name.")
            return
        
        if not surname:
            QMessageBox.warning(self, "Missing Field", "Please enter a Last Name.")
            return
        
        if not email:
            QMessageBox.warning(self, "Missing Field", "Please enter an Email.")
            return
        
        if self._existing_operator_id and operator_id != self._existing_operator_id:
            QMessageBox.warning(
                self,
                "Operator ID Locked",
                "Operator ID cannot be changed in modify mode.",
            )
            return

        existing = self.operator_manager.get_operator(operator_id)
        is_modify = existing is not None
        if is_modify and not self._confirm_modify_password():
            return
        
        # Add operator
        try:
            self.operator_manager.add_operator(
                operator_id=operator_id,
                name=name,
                surname=surname,
                email=email,
                phone=self.phone_edit.text().strip(),
                institution=self.institution_edit.text().strip(),
            )
            
            self.new_operator_id = operator_id
            
            QMessageBox.information(
                self,
                "Operator Updated" if is_modify else "Operator Created",
                (
                    f"Operator '{name} {surname}' updated successfully!"
                    if is_modify
                    else f"Operator '{name} {surname}' created successfully!"
                ),
            )
            
            self.accept()
        
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Creating Operator",
                f"Failed to create operator:\n\n{str(e)}",
            )
            logger.error(f"Failed to create operator: {e}", exc_info=True)
    
    def get_operator_id(self) -> Optional[str]:
        """Get the created operator ID.
        
        Returns:
            New operator ID, or None if cancelled
        """
        return self.new_operator_id
