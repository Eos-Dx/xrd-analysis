"""Session Management Mixin for DIFRA Main Window.

Integrates SessionManager for HDF5 container-based data storage.
"""

import json
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

from hardware.difra.gui.container_api import (
    get_container_manager,
    get_schema,
    get_writer,
)
from hardware.difra.gui.session_manager import SessionManager
from hardware.difra.gui.operator_manager import OperatorManager, OperatorSelectionDialog

logger = logging.getLogger(__name__)


class SessionMixin:
    """Mixin for session management functionality."""

    @staticmethod
    def _decode_attr(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
    
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
                # Create session with schema-driven parameters
                # All attributes come from params dict or SessionManager defaults
                session_id, session_path = self.session_manager.create_session(
                    folder=session_folder,
                    distance_cm=params['distance_cm'],
                    sample_id=params['sample_id'],
                    operator_id=params.get('operator_id'),
                    # Any other schema attributes can be passed from params
                    **{k: v for k, v in params.items() if k not in ['sample_id', 'operator_id', 'distance_cm']},
                )
                
                QMessageBox.information(
                    self,
                    "Session Created",
                    f"Session created successfully!\n\n"
                    f"Sample ID: {params['sample_id']}\n"
                    f"Study: {params.get('study_name', 'UNSPECIFIED')}\n"
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
        msg += f"Study: {info.get('study_name', 'UNSPECIFIED')}\n"
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
        """Get session (measurements) folder from config.
        
        Reads measurements_folder from global.json config.
        User can change this later from Zone Measurements panel.
        
        Returns:
            Path to measurements folder from config
        """
        # Get measurements folder from config
        if hasattr(self, 'config') and self.config:
            # Try measurements_folder first (preferred)
            folder = self.config.get('measurements_folder')
            if folder:
                folder_path = Path(folder)
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using measurements folder from config: {folder_path}")
                return folder_path
            
            # Fallback to session_folder for backward compatibility
            folder = self.config.get('session_folder')
            if folder:
                folder_path = Path(folder)
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using session folder from config: {folder_path}")
                return folder_path
        
        # No config - use default under difra_base_folder
        if hasattr(self, 'config') and self.config:
            base = self.config.get('difra_base_folder')
            if base:
                folder_path = Path(base) / 'measurements'
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using default measurements folder: {folder_path}")
                return folder_path
        
        # Last resort: use home directory
        folder_path = Path.home() / 'difra_measurements'
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"No config found, using fallback: {folder_path}")
        return folder_path
    
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
        
        # Update Zone Measurements panel Sample ID if present
        if hasattr(self, 'fileNameLineEdit'):
            if info['active']:
                self.fileNameLineEdit.setText(info['sample_id'])
                # Update lock indicator
                if hasattr(self, 'sampleIdLockLabel'):
                    is_locked = info.get('is_locked', False)
                    if is_locked:
                        self.sampleIdLockLabel.setText("🔒 Locked")
                        self.sampleIdLockLabel.setStyleSheet("color: #d32f2f; font-size: 9px;")
                    else:
                        self.sampleIdLockLabel.setText("")
            else:
                self.fileNameLineEdit.setText("")
                if hasattr(self, 'sampleIdLockLabel'):
                    self.sampleIdLockLabel.setText("")
        
        # Update Session tab if present
        if hasattr(self, '_update_session_tab_info'):
            self._update_session_tab_info()

    def on_technical_container_loaded(self, technical_path: str, is_locked: bool = False):
        """Optionally update embedded technical data in active unlocked session."""
        if not hasattr(self, "session_manager"):
            return
        if not self.session_manager.is_session_active():
            return

        if self.session_manager.is_locked():
            QMessageBox.information(
                self,
                "Session Locked",
                "Active session is locked. Technical data cannot be updated.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Update Session Technical Data?",
            f"Technical container loaded:\n{Path(technical_path).name}\n\n"
            "Do you want to replace embedded technical data in the active session?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self.session_manager.replace_technical_container(Path(technical_path))
            status = "locked" if is_locked else "unlocked"
            QMessageBox.information(
                self,
                "Technical Updated",
                f"Session technical data updated from {status} container:\n"
                f"{Path(technical_path).name}",
            )
            logger.info(
                "Updated session technical data from loaded container: "
                f"session={self.session_manager.session_path} technical={technical_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Technical Update Failed",
                f"Failed to update session technical data:\n\n{e}",
            )
            logger.error(
                f"Failed technical update from loaded container: {e}",
                exc_info=True,
            )

    def _set_image_from_array(self, image_array):
        """Render numpy image array into image_view when available."""
        if not hasattr(self, "image_view"):
            return False

        try:
            import numpy as np
            from PyQt5.QtGui import QImage, QPixmap

            array = np.asarray(image_array)
            if array.ndim == 2:
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                height, width = array.shape
                qimage = QImage(
                    array.data, width, height, array.strides[0], QImage.Format_Grayscale8
                ).copy()
            elif array.ndim == 3 and array.shape[2] in (3, 4):
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                height, width, channels = array.shape
                fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
                qimage = QImage(
                    array.data, width, height, array.strides[0], fmt
                ).copy()
            else:
                return False

            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                return False

            self.image_view.set_image(pixmap, image_path=None)
            return True
        except Exception as exc:
            logger.warning(f"Failed to set image from session array: {exc}")
            return False

    def _restore_session_workspace_from_container(self, session_path: Path):
        """Restore image/zones/points from an existing session container into GUI."""
        if not hasattr(self, "state"):
            self.state = {}

        try:
            import h5py
            schema = get_schema(self.config if hasattr(self, "config") else None)

            restored_shapes = []
            restored_points = []
            restored_image = False
            restored_ratio = None

            with h5py.File(session_path, "r") as h5f:
                # Restore sample image (use first available image dataset)
                images_group = h5f.get(schema.GROUP_IMAGES)
                if images_group:
                    image_keys = sorted(
                        key for key in images_group.keys() if key.startswith("img_")
                    )
                    if image_keys:
                        image_group = images_group[image_keys[0]]
                        if "data" in image_group:
                            restored_image = self._set_image_from_array(image_group["data"][:])

                # Restore zones -> state shape structure
                zones_group = h5f.get(schema.GROUP_IMAGES_ZONES)
                if zones_group:
                    for index, zone_id in enumerate(sorted(zones_group.keys()), start=1):
                        zone_group = zones_group[zone_id]
                        zone_role = str(
                            self._decode_attr(
                                zone_group.attrs.get(schema.ATTR_ZONE_ROLE, "sample_holder")
                            )
                        ).lower()
                        shape_name = str(
                            self._decode_attr(zone_group.attrs.get(schema.ATTR_ZONE_SHAPE, "circle"))
                        ).lower()
                        geometry_value = None
                        if "geometry_px" in zone_group:
                            raw_geometry = zone_group["geometry_px"][()]
                            if isinstance(raw_geometry, bytes):
                                raw_geometry = raw_geometry.decode("utf-8", errors="replace")
                            geometry_value = raw_geometry

                        x = y = width = height = 0.0
                        if isinstance(geometry_value, str):
                            parsed = json.loads(geometry_value)
                            if isinstance(parsed, dict):
                                if "center" in parsed and "radius" in parsed:
                                    center_x, center_y = parsed.get("center", [0, 0])
                                    radius = float(parsed.get("radius", 0))
                                    x = float(center_x) - radius
                                    y = float(center_y) - radius
                                    width = height = radius * 2.0
                                else:
                                    x = float(parsed.get("x", 0))
                                    y = float(parsed.get("y", 0))
                                    width = float(parsed.get("width", 0))
                                    height = float(parsed.get("height", 0))
                            elif isinstance(parsed, list) and len(parsed) >= 4:
                                x, y, width, height = [float(parsed[i]) for i in range(4)]
                        elif geometry_value is not None:
                            values = list(geometry_value)
                            if len(values) >= 4:
                                x, y, width, height = [float(values[i]) for i in range(4)]

                        ui_role = "include" if zone_role == "sample_holder" else zone_role
                        restored_shapes.append(
                            {
                                "id": index,
                                "type": "circle" if shape_name == "circle" else "rectangle",
                                "role": ui_role,
                                "geometry": {
                                    "x": x,
                                    "y": y,
                                    "width": width,
                                    "height": height,
                                },
                            }
                        )

                # Restore points as generated points
                points_group = h5f.get(schema.GROUP_POINTS)
                if points_group:
                    for point_id in sorted(points_group.keys()):
                        point_group = points_group[point_id]
                        pixel_coords = point_group.attrs.get(schema.ATTR_PIXEL_COORDINATES, [])
                        if len(pixel_coords) < 2:
                            continue
                        point_index = int(point_id.split("_")[-1])
                        restored_points.append(
                            {
                                "id": point_index,
                                "x": float(pixel_coords[0]),
                                "y": float(pixel_coords[1]),
                                "type": "generated",
                                "radius": 5.0,
                            }
                        )

                # Restore mapping ratio if available
                mapping_ds = h5f.get(f"{schema.GROUP_IMAGES_MAPPING}/mapping")
                if mapping_ds is not None:
                    mapping_raw = mapping_ds[()]
                    if isinstance(mapping_raw, bytes):
                        mapping_raw = mapping_raw.decode("utf-8", errors="replace")
                    mapping = json.loads(mapping_raw)
                    conversion = mapping.get("pixel_to_mm_conversion", {})
                    if "ratio" in conversion:
                        restored_ratio = float(conversion["ratio"])

            self.state["shapes"] = restored_shapes
            self.state["zone_points"] = restored_points
            if restored_ratio is not None:
                self.pixel_to_mm_ratio = restored_ratio

            if hasattr(self, "_restore_shapes"):
                self._restore_shapes(restored_shapes)
            if hasattr(self, "_restore_points"):
                self._restore_points(restored_points)
            if hasattr(self, "_refresh_id_counter"):
                self._refresh_id_counter()

            if hasattr(self, "update_points_table"):
                self.update_points_table()
            if hasattr(self, "update_shape_table"):
                self.update_shape_table()
            if hasattr(self, "update_coordinates"):
                self.update_coordinates()

            logger.info(
                f"Restored workspace from session container: session={session_path} "
                f"image_loaded={restored_image} shapes={len(restored_shapes)} "
                f"points={len(restored_points)}"
            )
        except Exception as exc:
            logger.warning(
                f"Session workspace restore failed for {session_path}: {exc}",
                exc_info=True,
            )

    def _extract_current_image_array(self):
        """Read current sample image into numpy array for session sync."""
        if not hasattr(self, "image_view"):
            return None

        image_path = getattr(self.image_view, "current_image_path", None)
        if not image_path:
            return None

        try:
            import cv2

            image_array = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image_array is not None:
                return image_array
        except Exception:
            pass

        try:
            import numpy as np
            from PIL import Image

            return np.array(Image.open(image_path))
        except Exception as exc:
            logger.warning(f"Failed to load current image for session sync: {exc}")
            return None

    def sync_workspace_to_session_container(self, state=None):
        """Persist image/zones/points snapshot into active unlocked session container."""
        if not hasattr(self, "session_manager"):
            return
        if not self.session_manager.is_session_active():
            return
        if self.session_manager.is_locked():
            return

        if state is None:
            state = getattr(self, "state", None) or {}

        try:
            import h5py
            schema = get_schema(self.config if hasattr(self, "config") else None)
            writer = get_writer(self.config if hasattr(self, "config") else None)

            session_path = self.session_manager.session_path

            image_array = self._extract_current_image_array()
            if image_array is not None:
                with h5py.File(session_path, "a") as h5f:
                    if schema.GROUP_IMAGES in h5f and "img_001" in h5f[schema.GROUP_IMAGES]:
                        del h5f[f"{schema.GROUP_IMAGES}/img_001"]
                writer.add_image(
                    file_path=session_path,
                    image_index=1,
                    image_data=image_array,
                    image_type="sample",
                )

            shapes = state.get("shapes", [])
            with h5py.File(session_path, "a") as h5f:
                if schema.GROUP_IMAGES_ZONES in h5f:
                    del h5f[schema.GROUP_IMAGES_ZONES]
                h5f.create_group(schema.GROUP_IMAGES_ZONES)

            for zone_index, shape in enumerate(shapes, start=1):
                role = str(shape.get("role", "include")).lower()
                zone_role = "exclude" if role == "exclude" else "sample_holder"
                shape_type = str(shape.get("type", "circle")).lower()
                geometry = shape.get("geometry", {})
                geometry_px = [
                    float(geometry.get("x", 0)),
                    float(geometry.get("y", 0)),
                    float(geometry.get("width", 0)),
                    float(geometry.get("height", 0)),
                ]
                holder_diameter_mm = None
                if zone_role == "sample_holder" and hasattr(self, "pixel_to_mm_ratio"):
                    diameter_px = max(geometry_px[2], geometry_px[3])
                    if getattr(self, "pixel_to_mm_ratio", 0):
                        holder_diameter_mm = diameter_px / float(self.pixel_to_mm_ratio)

                writer.add_zone(
                    file_path=session_path,
                    zone_index=zone_index,
                    zone_role=zone_role,
                    geometry_px=geometry_px,
                    shape=shape_type,
                    holder_diameter_mm=holder_diameter_mm,
                )

            if hasattr(self, "pixel_to_mm_ratio"):
                writer.add_image_mapping(
                    file_path=session_path,
                    sample_holder_zone_id="zone_001",
                    pixel_to_mm_conversion={
                        "ratio": float(self.pixel_to_mm_ratio),
                        "units": "mm/pixel",
                    },
                    orientation="standard",
                    mapping_version=schema.SCHEMA_VERSION,
                )

            # Only rewrite points while there are no recorded measurements.
            has_measurements = False
            with h5py.File(session_path, "r") as h5f:
                measurements = h5f.get(schema.GROUP_MEASUREMENTS)
                if measurements:
                    for point_group in measurements.values():
                        if len(point_group.keys()) > 0:
                            has_measurements = True
                            break

            if not has_measurements:
                points = state.get("zone_points", [])
                with h5py.File(session_path, "a") as h5f:
                    if schema.GROUP_POINTS in h5f:
                        del h5f[schema.GROUP_POINTS]
                    h5f.create_group(schema.GROUP_POINTS)

                for point_index, point in enumerate(points, start=1):
                    x = float(point.get("x", 0))
                    y = float(point.get("y", 0))
                    writer.add_point(
                        file_path=session_path,
                        point_index=point_index,
                        pixel_coordinates=[x, y],
                        physical_coordinates_mm=[0.0, 0.0],
                        point_status=schema.POINT_STATUS_PENDING,
                    )

        except Exception as exc:
            logger.warning(
                f"Workspace snapshot sync to session failed: {exc}",
                exc_info=True,
            )
    
    def _handle_session_replacement(self) -> bool:
        """Handle replacement of existing session with error checking.
        
        Returns:
            True if session was closed/archived, False if user cancelled
        """
        from PyQt5.QtWidgets import QInputDialog
        import h5py
        import time
        container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
        
        if not self.session_manager.is_session_active():
            return True
        
        info = self.session_manager.get_session_info()
        session_path = Path(info['session_path'])
        sample_id = info['sample_id']
        session_id = info['session_id']
        
        # Check if container is locked/finalized
        is_locked = container_manager.is_container_locked(session_path)
        
        # Check if measurements exist
        has_measurements = False
        try:
            with h5py.File(session_path, 'r') as f:
                schema = get_schema(self.config if hasattr(self, "config") else None)
                if schema.GROUP_MEASUREMENTS in f:
                    meas_group = f[schema.GROUP_MEASUREMENTS]
                    # Check if any point groups exist
                    has_measurements = any(key.startswith('pt_') for key in meas_group.keys())
        except Exception:
            pass
        
        # Build status message
        status_lines = [
            f"Sample ID: {sample_id}",
            f"Session ID: {session_id}",
            f"Status: {'Finalized (locked)' if is_locked else 'Unfinalized (unlocked)'}",
            f"Measurements: {'Yes' if has_measurements else 'None recorded'}",
        ]
        
        # Check attenuation status
        if info.get('i0_recorded'):
            status_lines.append(f"Attenuation: I₀ recorded")
        if info.get('attenuation_complete'):
            status_lines.append(f"Attenuation: Complete")
        
        status_str = "\n".join(status_lines)
        
        # Show different dialogs based on status
        if is_locked:
            # Container is locked - simple replacement
            reply = QMessageBox.question(
                self,
                "Replace Finalized Session?",
                f"Current session is finalized and locked:\n\n{status_str}\n\n"
                f"Close this session and create new one for new sample?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.session_manager.close_session()
                return True
            else:
                return False
        
        # Unlocked container - potential error scenario
        msg = (
            f"⚠️  Found unfinalized session:\n\n{status_str}\n\n"
            f"You are about to load a new sample image.\n"
            f"The current session will be archived.\n\n"
        )
        
        # Warn about incomplete data
        if not has_measurements:
            msg += "⚠️  WARNING: No measurements recorded in this session!\n"
        
        if not info.get('attenuation_complete'):
            msg += "⚠️  WARNING: Attenuation not complete!\n"
        
        msg += "\nWas this session created by error?"
        
        # Create custom dialog with three buttons
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Replace Unfinalized Session?")
        msg_box.setText(msg)
        msg_box.setIcon(QMessageBox.Warning)
        
        # Add buttons
        mark_error_btn = msg_box.addButton("Yes - Mark as Error", QMessageBox.YesRole)
        continue_btn = msg_box.addButton("No - Archive Normally", QMessageBox.NoRole)
        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        msg_box.setDefaultButton(cancel_btn)
        
        msg_box.exec_()
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == cancel_btn:
            logger.info("User cancelled session replacement")
            return False
        
        # Determine error status
        created_by_error = (clicked_button == mark_error_btn)
        error_reason = ""
        
        if created_by_error:
            # Prompt for error reason
            reason, ok = QInputDialog.getText(
                self,
                "Error Reason",
                f"Why was session '{sample_id}' created by error?\n\n"
                f"(Optional - provide brief description)",
            )
            if ok and reason.strip():
                error_reason = reason.strip()
            else:
                error_reason = "User marked as error without specifying reason"
            
            logger.info(
                f"Session {session_id} marked as created_by_error: {error_reason}"
            )
        
        # Close session and archive with metadata
        try:
            # Add error attributes before closing if needed
            if created_by_error:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                try:
                    with h5py.File(session_path, 'a') as f:
                        f.attrs['created_by_error'] = True
                        f.attrs['error_reason'] = error_reason
                        f.attrs['archived_timestamp'] = timestamp
                    logger.info(f"Added error attributes to session container: {session_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to add error attributes: {e}")
            
            # Close session (this will keep the file in place)
            self.session_manager.close_session()
            
            # Archive the container to session_archive folder
            self._archive_session_container(session_path, session_id, created_by_error, error_reason)
            
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Archive Failed",
                f"Failed to archive session:\n{e}",
            )
            logger.error(f"Failed to archive session: {e}", exc_info=True)
            return False
    
    def _archive_session_container(self, session_path: Path, session_id: str, 
                                   created_by_error: bool = False, error_reason: str = ""):
        """Archive session container to session_archive folder.
        
        Args:
            session_path: Path to session container
            session_id: Session container ID
            created_by_error: Whether marked as error
            error_reason: Optional error reason
        """
        import shutil
        import time
        
        # Create archive folder from config (preferred) with deterministic fallback
        archive_base = None
        try:
            if hasattr(self, "config") and self.config:
                configured = self.config.get("measurements_archive_folder")
                if configured:
                    archive_base = Path(configured)
                elif self.config.get("session_archive_folder"):
                    archive_base = Path(self.config.get("session_archive_folder"))
        except Exception:
            archive_base = None

        if archive_base is None:
            archive_base = session_path.parent.parent / "archive" / "measurements"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_folder = archive_base / f"{session_id}_{timestamp}"
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        # Move container to archive
        dest_path = archive_folder / session_path.name
        try:
            shutil.move(str(session_path), str(dest_path))
            logger.info(
                f"Archived session container: {session_path.name} -> {archive_folder.name}/" +
                (f" [ERROR: {error_reason}]" if created_by_error else "")
            )
        except Exception as e:
            logger.error(f"Failed to move session to archive: {e}")
            raise
    
    def _handle_new_sample_image(self, image_path: str):
        """Handle loading/capturing a new sample image - auto-creates session.
        
        Prompts user about existing unfinalized sessions and handles archiving
        with error marking if needed.
        
        Args:
            image_path: Path to the loaded/captured image
        """
        from pathlib import Path
        import numpy as np
        from PyQt5.QtGui import QPixmap
        
        # Check if session already active - show detailed dialog
        if self.session_manager.is_session_active():
            if not self._handle_session_replacement():
                # User cancelled replacement
                return
        
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
                # Create session with schema-driven parameters
                session_id, session_path = self.session_manager.create_session(
                    folder=session_folder,
                    distance_cm=params['distance_cm'],
                    sample_id=params['sample_id'],
                    operator_id=params.get('operator_id'),
                    # Pass all other schema attributes from params
                    **{k: v for k, v in params.items() if k not in ['sample_id', 'operator_id', 'distance_cm']},
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
                    f"Study: {params.get('study_name', 'UNSPECIFIED')}\n"
                    f"Container: {session_path.name}\n\n"
                    f"Sample image added to container.",
                )
                
                logger.info(
                    f"Created new session: {session_id} for sample {params['sample_id']} "
                    f"with image: {image_path}"
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
        """Add zones from state to session container.
        
        Called when measurements start to store zone definitions from state.
        """
        if not hasattr(self, 'session_manager') or not self.session_manager.is_session_active():
            return
        
        # Get shapes from state instead of image_view
        if not hasattr(self, 'state') or 'shapes' not in self.state:
            logger.warning("No shapes found in state")
            return
        
        shapes_data = self.state.get('shapes', [])
        if not shapes_data:
            logger.warning("Shapes list is empty in state")
            return
        
        try:
            zone_index = 1
            for shape in shapes_data:
                # Get shape data from state structure
                role = shape.get('role', 'include')
                shape_type = shape.get('type', 'Circle').lower()
                geometry = shape.get('geometry', {})
                
                # Convert geometry dict to list format [x, y, width, height]
                geometry_px = [
                    geometry.get('x', 0),
                    geometry.get('y', 0),
                    geometry.get('width', 0),
                    geometry.get('height', 0),
                ]
                
                # Map role to zone_role
                zone_role_map = {
                    'include': 'sample_holder',
                    'sample holder': 'sample_holder',
                    'exclude': 'exclude',
                }
                zone_role = zone_role_map.get(role.lower(), 'sample_holder')
                
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
                    holder_diameter_mm=holder_diameter_mm,
                )
                
                logger.info(
                    f"Added zone {zone_index} to session: role={zone_role}, shape={shape_type}, geometry={geometry_px}"
                )
                zone_index += 1
            
            if zone_index > 1:
                logger.info(f"Added {zone_index - 1} zones to session container")
            else:
                logger.warning("No zones were added to session container")
        
        except Exception as e:
            logger.error(
                f"Failed to add zones to session container: {e}",
                exc_info=True,
            )
    
    def _add_mapping_to_session(self):
        """Add image mapping (pixel-to-mm conversion) to session container.
        
        Called when measurements start to store the coordinate transformation.
        """
        if not hasattr(self, 'session_manager') or not self.session_manager.is_session_active():
            return
        
        if not hasattr(self, 'pixel_to_mm_ratio'):
            logger.warning("No pixel_to_mm_ratio available for mapping")
            return
        
        try:
            writer = get_writer(self.config if hasattr(self, "config") else None)
            schema = get_schema(self.config if hasattr(self, "config") else None)
            
            # Find sample_holder zone ID (first zone with sample_holder role)
            sample_holder_zone_id = "zone_001"  # Default to first zone
            
            # Create pixel-to-mm conversion dict
            pixel_to_mm_conversion = {
                "ratio": float(self.pixel_to_mm_ratio),
                "units": "mm/pixel",
            }
            
            # Add orientation if available
            orientation = "standard"
            
            # Call writer to add mapping with overwrite=True
            writer.add_image_mapping(
                file_path=self.session_manager.session_path,
                sample_holder_zone_id=sample_holder_zone_id,
                pixel_to_mm_conversion=pixel_to_mm_conversion,
                orientation=orientation,
                mapping_version=schema.SCHEMA_VERSION,
            )
            
            logger.info(
                f"Added image mapping to session container: ratio={self.pixel_to_mm_ratio}"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to add mapping to session container: {e}",
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
                container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
                
                session_path = self.session_manager.session_path
                
                # Close session
                self.session_manager.close_session()
                
                # Lock the container (mark read-only)
                container_manager.lock_container(session_path)
                
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
            "NeXus HDF5 Files (*.nxs.h5 *.h5);;All Files (*)",
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
            schema = get_schema(self.config if hasattr(self, "config") else None)
            container_manager = get_container_manager(self.config if hasattr(self, "config") else None)
            
            # Check if locked
            is_locked = container_manager.is_container_locked(file_path)
            
            # Open container to read metadata
            with h5py.File(file_path, 'r') as f:
                sample_id = self._decode_attr(f.attrs.get(schema.ATTR_SAMPLE_ID, 'Unknown'))
                study_name = self._decode_attr(f.attrs.get(schema.ATTR_STUDY_NAME, 'UNSPECIFIED'))
                session_id = self._decode_attr(f.attrs.get(schema.ATTR_SESSION_ID, 'Unknown'))
                operator_id = self._decode_attr(f.attrs.get(schema.ATTR_OPERATOR_ID, 'Unknown'))
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
                f"Study: {study_name}\n"
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
            
            QMessageBox.information(
                self,
                "Session Container Opened",
                msg,
                QMessageBox.Ok
            )
            
            # Load session manager state from container metadata.
            self.session_manager.open_existing_session(file_path)
            
            logger.info(
                "Opened existing session container: sample_id=%s locked=%s path=%s",
                sample_id,
                is_locked,
                str(file_path),
            )

            # Restore workspace data from session container when UI supports it.
            self._restore_session_workspace_from_container(file_path)

            # If technical table exists, restore from embedded calibration snapshot data.
            if hasattr(self, "_populate_aux_table_from_h5"):
                try:
                    self._populate_aux_table_from_h5(str(file_path))
                except Exception as tech_restore_error:
                    logger.warning(
                        f"Failed to restore technical table from session: {tech_restore_error}"
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
    - Study (required)
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

        # Study (required)
        self.study_name_edit = QLineEdit()
        self.study_name_edit.setPlaceholderText("e.g. STUDY_2026_A")
        form_layout.addRow("Study*:", self.study_name_edit)
        
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

        if not self.study_name_edit.text().strip():
            QMessageBox.warning(
                self,
                "Missing Study",
                "Please enter a Study name.",
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
            'study_name': self.study_name_edit.text().strip(),
            'distance_cm': float(self.distance_edit.text()),
            'operator_id': self.selected_operator_id,
        }
        
        return params
