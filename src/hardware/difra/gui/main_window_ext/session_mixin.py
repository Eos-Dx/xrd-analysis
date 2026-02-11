"""Session Management Mixin for DIFRA Main Window.

Integrates SessionManager for HDF5 container-based data storage.
"""

import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from hardware.difra.gui.session_manager import SessionManager
from hardware.difra.gui.operator_manager import OperatorManager, OperatorSelectionDialog

logger = logging.getLogger(__name__)


class SessionMixin:
    """Mixin for session management functionality."""
    
    def init_session_manager(self):
        """Initialize SessionManager and add UI actions."""
        logger.info("Initializing SessionManager")
        
        # Initialize operator manager first
        self.operator_manager = OperatorManager()
        
        # Show operator selection dialog on startup
        self.show_operator_selection_dialog()
        
        # Create SessionManager instance with config (including operator)
        config = self.config if hasattr(self, 'config') else {}
        
        # Add current operator to config
        if self.operator_manager.get_current_operator_id():
            config['operator_id'] = self.operator_manager.get_current_operator_id()
        
        self.session_manager = SessionManager(config=config)
        
        # Add session menu actions
        self.add_session_menu_actions()
        
        logger.info("SessionManager initialized")
    
    def add_session_menu_actions(self):
        """Add session-related actions to File menu."""
        # Get or create File menu
        menu_bar = self.menuBar()
        file_menu = None
        
        for action in menu_bar.actions():
            if action.text() == "File":
                file_menu = action.menu()
                break
        
        if not file_menu:
            file_menu = menu_bar.addMenu("File")
        
        # Add separator
        file_menu.addSeparator()
        
        # New Session action
        new_session_action = QAction("New Session...", self)
        new_session_action.triggered.connect(self.on_new_session)
        new_session_action.setStatusTip("Create a new measurement session")
        file_menu.addAction(new_session_action)
        
        # Close Session action
        close_session_action = QAction("Close Session", self)
        close_session_action.triggered.connect(self.on_close_session)
        close_session_action.setStatusTip("Close the current session")
        file_menu.addAction(close_session_action)
        
        # Session Info action
        session_info_action = QAction("Session Info", self)
        session_info_action.triggered.connect(self.on_session_info)
        session_info_action.setStatusTip("Show current session information")
        file_menu.addAction(session_info_action)
        
        # Add separator
        file_menu.addSeparator()
        
        # Finalize & Send Session action
        finalize_session_action = QAction("Finalize && Send Session", self)
        finalize_session_action.triggered.connect(self.on_finalize_session)
        finalize_session_action.setStatusTip("Finalize session and prepare for upload")
        file_menu.addAction(finalize_session_action)
        
        # Add separator
        file_menu.addSeparator()
        
        # Restore/Open Session action
        restore_session_action = QAction("Open Existing Session...", self)
        restore_session_action.triggered.connect(self.on_restore_session)
        restore_session_action.setStatusTip("Open an existing session container for analysis")
        file_menu.addAction(restore_session_action)
        
        logger.debug("Session menu actions added")
    
    def show_operator_selection_dialog(self):
        """Show operator selection dialog on startup."""
        dialog = OperatorSelectionDialog(self.operator_manager, self)
        
        if dialog.exec_() == QDialog.Accepted:
            operator_id = dialog.get_selected_operator_id()
            logger.info(f"Operator selected: {operator_id}")
        else:
            # User cancelled - use default or show warning
            logger.warning("Operator selection cancelled")
            QMessageBox.warning(
                self,
                "No Operator Selected",
                "No operator selected. Using default operator.\n\n"
                "You can change this later from File → Operator Settings...",
            )
    
    def on_new_session(self):
        """Handle New Session action."""
        # Check if session already active
        if self.session_manager.is_session_active():
            reply = QMessageBox.question(
                self,
                "Close Current Session?",
                f"Close current session '{self.session_manager.sample_id}' and create new session?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            self.session_manager.close_session()
        
        # Show dialog to get session parameters
        dialog = NewSessionDialog(self.operator_manager, self)
        
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            
            # Get session folder from config or file dialog
            session_folder = self.get_session_folder()
            if not session_folder:
                return
            
            try:
                # Create session
                session_id, session_path = self.session_manager.create_session(
                    folder=session_folder,
                    sample_id=params['sample_id'],
                    distance_cm=params['distance_cm'],
                    operator_id=params.get('operator_id'),
                    beam_energy_kev=params.get('beam_energy_kev'),
                )
                
                QMessageBox.information(
                    self,
                    "Session Created",
                    f"Session created successfully!\n\n"
                    f"Sample ID: {params['sample_id']}\n"
                    f"Container: {session_path.name}",
                )
                
                logger.info(
                    f"Created new session: {session_id} for sample {params['sample_id']}"
                )
                
                # Update UI
                self.update_session_status()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Session Creation Failed",
                    f"Failed to create session:\n\n{str(e)}",
                )
                logger.error(f"Failed to create session: {e}", exc_info=True)
    
    def on_close_session(self):
        """Handle Close Session action."""
        if not self.session_manager.is_session_active():
            QMessageBox.information(
                self,
                "No Active Session",
                "No session is currently active.",
            )
            return
        
        reply = QMessageBox.question(
            self,
            "Close Session?",
            f"Close session '{self.session_manager.sample_id}'?\n\n"
            f"The container has been saved.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.session_manager.close_session()
            QMessageBox.information(
                self,
                "Session Closed",
                "Session closed successfully.",
            )
            logger.info("Session closed by user")
            
            # Update UI
            self.update_session_status()
    
    def on_session_info(self):
        """Handle Session Info action."""
        info = self.session_manager.get_session_info()
        
        if not info['active']:
            QMessageBox.information(
                self,
                "No Active Session",
                "No session is currently active.\n\n"
                "Create a new session from File → New Session...",
            )
            return
        
        # Build info message
        msg = f"Sample ID: {info['sample_id']}\n"
        msg += f"Session ID: {info['session_id']}\n"
        msg += f"Operator: {info['operator_id']}\n"
        msg += f"Machine: {info['machine_name']}\n"
        msg += f"Beam Energy: {info['beam_energy_kev']} keV\n\n"
        msg += f"Container: {Path(info['session_path']).name}\n\n"
        msg += "Attenuation Status:\n"
        msg += f"  I₀ recorded: {'✓' if info['i0_recorded'] else '✗'}\n"
        msg += f"  I recorded: {'✓' if info['i_recorded'] else '✗'}\n"
        msg += f"  Complete: {'✓' if info['attenuation_complete'] else '✗'}\n"
        
        QMessageBox.information(
            self,
            "Session Information",
            msg,
        )
    
    def get_session_folder(self) -> Path:
        """Get session folder from config or user selection.
        
        Returns:
            Path to session folder, or None if cancelled
        """
        # Try to get from config
        if hasattr(self, 'config') and self.config:
            folder = self.config.get('session_folder')
            if folder:
                return Path(folder)
        
        # Prompt user
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Session Folder",
            str(Path.home()),
        )
        
        if folder:
            return Path(folder)
        
        return None
    
    def update_session_status(self):
        """Update UI to reflect current session status."""
        info = self.session_manager.get_session_info()
        
        # Update window title
        if info['active']:
            self.setWindowTitle(f"DIFRA - {info['sample_id']}")
        else:
            self.setWindowTitle("DIFRA")
        
        # Update status bar if present
        if hasattr(self, 'statusBar'):
            if info['active']:
                status_msg = f"Session: {info['sample_id']}"
                if info['attenuation_complete']:
                    status_msg += " | Attenuation: Complete"
                elif info['i0_recorded']:
                    status_msg += " | Attenuation: I₀ recorded"
                self.statusBar().showMessage(status_msg)
            else:
                self.statusBar().showMessage("No active session")
    
    def _handle_new_sample_image(self, image_path: str):
        """Handle loading/capturing a new sample image - auto-creates session.
        
        Args:
            image_path: Path to the loaded/captured image
        """
        from pathlib import Path
        import numpy as np
        from PyQt5.QtGui import QPixmap
        
        # Check if session already active - ask user if they want to close it
        if self.session_manager.is_session_active():
            reply = QMessageBox.question(
                self,
                "Close Current Session?",
                f"Close current session '{self.session_manager.sample_id}' and create new session for this sample?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                logger.info("User chose to keep existing session")
                return
            
            self.session_manager.close_session()
        
        # Show dialog to get sample information
        dialog = NewSessionDialog(self.operator_manager, self)
        
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            
            # Get session folder from config or use image directory
            session_folder = self.get_session_folder()
            if not session_folder:
                # Default to image directory
                session_folder = Path(image_path).parent
                logger.info(f"Using image directory as session folder: {session_folder}")
            
            try:
                # Create session
                session_id, session_path = self.session_manager.create_session(
                    folder=session_folder,
                    sample_id=params['sample_id'],
                    distance_cm=params['distance_cm'],
                    operator_id=params.get('operator_id'),
                    beam_energy_kev=params.get('beam_energy_kev'),
                )
                
                # Add image to session container
                try:
                    # Load image as numpy array
                    pixmap = QPixmap(image_path)
                    from PyQt5.QtCore import QBuffer, QIODevice
                    import cv2
                    
                    # Convert QPixmap to numpy array via cv2
                    # For now, store the image path - actual conversion can be done later
                    # if needed, or we can store the raw image file
                    
                    # Simple approach: read image with cv2 or PIL
                    try:
                        import cv2
                        image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    except ImportError:
                        from PIL import Image
                        image_array = np.array(Image.open(image_path).convert('L'))
                    
                    if image_array is not None:
                        self.session_manager.add_sample_image(
                            image_data=image_array,
                            image_index=1,
                            image_type="sample",
                        )
                        logger.info(f"Added sample image to session container")
                    else:
                        logger.warning(f"Failed to load image as array: {image_path}")
                        
                except Exception as e:
                    logger.warning(
                        f"Failed to add image to session container: {e}",
                        exc_info=True,
                    )
                
                QMessageBox.information(
                    self,
                    "Session Created",
                    f"Session created successfully!\n\n"
                    f"Sample ID: {params['sample_id']}\n"
                    f"Container: {session_path.name}\n\n"
                    f"Sample image added to container.",
                )
                
                logger.info(
                    f"Created new session: {session_id} for sample {params['sample_id']}",
                    image_path=image_path,
                )
                
                # Update UI
                self.update_session_status()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Session Creation Failed",
                    f"Failed to create session:\n\n{str(e)}",
                )
                logger.error(f"Failed to create session: {e}", exc_info=True)
        else:
            logger.info("User cancelled session creation")
    
    def _add_zones_to_session(self):
        """Add zones from shapes to session container.
        
        Called after points are generated to store zone definitions.
        """
        if not hasattr(self, 'session_manager') or not self.session_manager.is_session_active():
            return
        
        if not hasattr(self, 'image_view') or not hasattr(self.image_view, 'shapes'):
            logger.warning("No image view or shapes available")
            return
        
        try:
            zone_index = 1
            for shape in self.image_view.shapes:
                role = shape.get('role', 'include')
                item = shape.get('item')
                
                if item is None:
                    continue
                
                # Get geometry from shape
                from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem
                from PyQt5.QtCore import QPointF
                
                geometry_px = None
                shape_type = "polygon"
                
                if isinstance(item, QGraphicsEllipseItem):
                    rect = item.rect()
                    geometry_px = [
                        rect.x(),
                        rect.y(),
                        rect.width(),
                        rect.height(),
                    ]
                    shape_type = "circle"
                elif hasattr(item, 'polygon'):
                    # Polygon shape
                    poly = item.polygon()
                    geometry_px = [[p.x(), p.y()] for p in poly]
                    shape_type = "polygon"
                elif isinstance(item, QGraphicsRectItem):
                    rect = item.rect()
                    geometry_px = [
                        rect.x(),
                        rect.y(),
                        rect.width(),
                        rect.height(),
                    ]
                    shape_type = "rectangle"
                
                if geometry_px:
                    # Map role to zone_role
                    zone_role_map = {
                        'include': 'sample_holder',
                        'exclude': 'exclude',
                    }
                    zone_role = zone_role_map.get(role, 'sample_holder')
                    
                    # Get holder diameter if it's a sample_holder
                    holder_diameter_mm = None
                    if zone_role == 'sample_holder' and hasattr(self, 'pixel_to_mm_ratio'):
                        # Calculate diameter from geometry
                        if shape_type == 'circle' and len(geometry_px) == 4:
                            diameter_px = max(geometry_px[2], geometry_px[3])
                            holder_diameter_mm = diameter_px / self.pixel_to_mm_ratio
                    
                    self.session_manager.add_zone(
                        zone_index=zone_index,
                        geometry_px=geometry_px,
                        shape=shape_type,
                        zone_role=zone_role,
                        image_index=1,
                        holder_diameter_mm=holder_diameter_mm,
                    )
                    
                    logger.info(
                        f"Added zone {zone_index} to session",
                        role=zone_role,
                        shape=shape_type,
                    )
                    zone_index += 1
            
            if zone_index > 1:
                logger.info(f"Added {zone_index - 1} zones to session container")
        
        except Exception as e:
            logger.error(
                f"Failed to add zones to session container: {e}",
                exc_info=True,
            )
    
    def on_finalize_session(self):
        """Finalize session - close, lock, and prepare for upload."""
        if not self.session_manager.is_session_active():
            QMessageBox.information(
                self,
                "No Active Session",
                "No session is currently active.",
            )
            return
        
        # Get session info
        info = self.session_manager.get_session_info()
        
        # Confirm finalization
        msg = (
            f"Finalize session '{info['sample_id']}'?\n\n"
            f"This will:\n"
            f"  • Close the session container\n"
            f"  • Mark it as read-only (locked)\n"
            f"  • No more data can be added\n\n"
            f"Container: {Path(info['session_path']).name}\n\n"
            f"After finalization, the container can be uploaded to the cloud."
        )
        
        reply = QMessageBox.question(
            self,
            "Finalize Session?",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                from hardware.container.v0_1.container_manager import lock_container
                
                session_path = self.session_manager.session_path
                
                # Close session
                self.session_manager.close_session()
                
                # Lock the container (mark read-only)
                lock_container(session_path)
                
                logger.info(
                    f"Session finalized and locked: {session_path.name}"
                )
                
                # Show success message with container location
                QMessageBox.information(
                    self,
                    "Session Finalized",
                    f"Session finalized successfully!\n\n"
                    f"Container: {session_path.name}\n"
                    f"Location: {session_path.parent}\n\n"
                    f"The container is now locked and ready for upload.\n"
                    f"No more data can be added to this session.",
                )
                
                # Update UI
                self.update_session_status()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Finalization Failed",
                    f"Failed to finalize session:\n\n{str(e)}",
                )
                logger.error(f"Failed to finalize session: {e}", exc_info=True)
    
    def on_restore_session(self):
        """Open an existing session container (including locked ones) for analysis."""
        from pathlib import Path
        
        # Close current session if active
        if self.session_manager.is_session_active():
            reply = QMessageBox.question(
                self,
                "Close Current Session?",
                f"Close current session '{self.session_manager.sample_id}' and open existing session?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            self.session_manager.close_session()
        
        # Get session file from user
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session Container",
            str(Path.home()),
            "HDF5 Files (*.h5);;All Files (*)",
        )
        
        if not file_path:
            return
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            QMessageBox.critical(
                self,
                "File Not Found",
                f"Container file not found:\n{file_path}",
            )
            return
        
        try:
            import h5py
            from hardware.container.v0_1 import schema
            from hardware.container.v0_1.container_manager import is_container_locked
            
            # Check if locked
            is_locked = is_container_locked(file_path)
            
            # Open container to read metadata
            with h5py.File(file_path, 'r') as f:
                sample_id = f.attrs.get(schema.ATTR_SAMPLE_ID, 'Unknown')
                session_id = f.attrs.get(schema.ATTR_SESSION_ID, 'Unknown')
                operator_id = f.attrs.get(schema.ATTR_OPERATOR_ID, 'Unknown')
                distance_cm = f.attrs.get(schema.ATTR_DISTANCE_CM, None)
                beam_energy_kev = f.attrs.get(schema.ATTR_BEAM_ENERGY_KEV, None)
                
                # Count points and measurements
                num_points = len(f.get(schema.GROUP_POINTS, {}).keys())
                
                # Get all measurements
                meas_group = f.get(schema.GROUP_MEASUREMENTS, {})
                num_measurements = 0
                for point_group in meas_group.values():
                    num_measurements += len(list(point_group.keys()))
            
            # Show container info
            lock_status = "🔒 LOCKED (read-only)" if is_locked else "🔓 Unlocked (editable)"
            msg = (
                f"Container Information:\n\n"
                f"Sample ID: {sample_id}\n"
                f"Session ID: {session_id}\n"
                f"Operator: {operator_id}\n"
                f"Status: {lock_status}\n\n"
                f"Data Summary:\n"
                f"  Points: {num_points}\n"
                f"  Measurements: {num_measurements}\n\n"
            )
            
            if distance_cm is not None:
                msg += f"Distance: {distance_cm} cm\n"
            if beam_energy_kev is not None:
                msg += f"Beam Energy: {beam_energy_kev} keV\n\n"
            
            if is_locked:
                msg += "This container is locked and will be opened in read-only mode.\n"
                msg += "You can analyze the data but cannot add new measurements."
            else:
                msg += "This container is unlocked. You can add new measurements."
            
            reply = QMessageBox.information(
                self,
                "Session Container Opened",
                msg,
                QMessageBox.Ok
            )
            
            # Set session manager state (read-only reference)
            self.session_manager.session_path = file_path
            self.session_manager.sample_id = str(sample_id)
            self.session_manager.session_id = str(session_id)
            
            logger.info(
                f"Opened existing session container",
                sample_id=sample_id,
                locked=is_locked,
                path=str(file_path),
            )
            
            # Update UI
            self.update_session_status()
            
            QMessageBox.information(
                self,
                "Container Ready",
                f"Session container opened successfully!\n\n"
                f"You can now analyze the data in DIFRA.\n\n"
                f"Note: {'Read-only mode (locked)' if is_locked else 'Editable mode'}",
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Failed to Open Container",
                f"Failed to open session container:\n\n{str(e)}",
            )
            logger.error(f"Failed to open session container: {e}", exc_info=True)


class NewSessionDialog(QDialog):
    """Dialog for creating a new session.
    
    Prompts user for:
    - Sample ID (required)
    - Distance in cm (required)
    - Operator (with option to use current or select different)
    
    Beam energy is read from global config.
    """
    
    def __init__(self, operator_manager: OperatorManager, parent=None, default_distance: float = None):
        super().__init__(parent)
        
        self.operator_manager = operator_manager
        self.selected_operator_id = None
        
        self.setWindowTitle("New Session")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Form layout for parameters
        form_layout = QFormLayout()
        
        # Sample ID (required)
        self.sample_id_edit = QLineEdit()
        self.sample_id_edit.setPlaceholderText("e.g. SAMPLE_001")
        form_layout.addRow("Sample ID*:", self.sample_id_edit)
        
        # Distance (required) - with explicit prompt
        distance_label = QLabel(
            "<b>Distance (cm)*:</b><br>"
            "<span style='color: #555; font-size: 10px;'>"
            "Sample-to-detector distance (must match technical container)"
            "</span>"
        )
        self.distance_edit = QLineEdit()
        if default_distance:
            self.distance_edit.setText(str(default_distance))
        else:
            self.distance_edit.setText("17.0")
        self.distance_edit.setPlaceholderText("e.g. 17.0, 25.0, 50.0")
        form_layout.addRow(distance_label, self.distance_edit)
        
        layout.addLayout(form_layout)
        
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
            "Beam energy: Read from global config\n"
            "<b>Note:</b> Distance must match technical container distance."
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
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
        from hardware.difra.gui.operator_manager import NewOperatorDialog
        
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
    
    def validate_and_accept(self):
        """Validate inputs before accepting."""
        if not self.sample_id_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Sample ID",
                "Please enter a Sample ID.",
            )
            return
        
        if not self.distance_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Distance",
                "Please enter a distance value.",
            )
            return
        
        try:
            float(self.distance_edit.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Distance",
                "Distance must be a number.",
            )
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
    
    def get_parameters(self):
        """Get session parameters from dialog."""
        params = {
            'sample_id': self.sample_id_edit.text().strip(),
            'distance_cm': float(self.distance_edit.text()),
            'operator_id': self.selected_operator_id,
        }
        
        return params
