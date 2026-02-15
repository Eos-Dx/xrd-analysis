"""Session lifecycle flow helpers for SessionMixin."""

from . import session_mixin as _session_module

Path = _session_module.Path
QMessageBox = _session_module.QMessageBox
QFileDialog = _session_module.QFileDialog
get_container_manager = _session_module.get_container_manager
get_schema = _session_module.get_schema
get_writer = _session_module.get_writer
logger = _session_module.logger


class SessionFlowMixin:
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
        from hardware.difra.gui.main_window_ext.session_mixin import NewSessionDialog
        
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

