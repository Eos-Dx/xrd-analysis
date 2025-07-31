# zone_measurements/logic/stage_control_mixin.py

from PyQt5.QtCore import Qt


class StageControlMixin:
    def toggle_hardware(self):
        """
        Toggle hardware initialization state. Dynamically (re)builds detector param tab widgets
        for only active detectors after hardware is initialized.
        """
        if not getattr(self, "hardware_initialized", False):
            # --- Initialize hardware using your config-driven HardwareController ---
            from hardware.Ulster.hardware.hardware_control import (
                HardwareController,
            )

            self.hardware_controller = HardwareController(self.config)
            res_xystage, res_det = self.hardware_controller.initialize()

            # Use updated controllers from hardware_controller
            self.stage_controller = self.hardware_controller.stage_controller
            self.detector_controller = (
                self.hardware_controller.detectors
            )  # dict: {alias: controller}

            # Update indicators
            self.xyStageIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
                if res_xystage
                else "background-color: red; border-radius: 10px;"
            )
            self.cameraIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
                if res_det
                else "background-color: red; border-radius: 10px;"
            )

            ok = res_xystage and res_det
            self.start_btn.setEnabled(ok)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            # Enable X/Y controls if hardware is initialized
            self.xPosSpin.setEnabled(ok)
            self.yPosSpin.setEnabled(ok)
            self.gotoBtn.setEnabled(ok)

            if ok:
                self.populate_detector_param_tab()  # Populate "Detector param" tab now
                self.initializeBtn.setText("Deinitialize Hardware")
                self.hardware_initialized = True
                if hasattr(self, "hardware_state_changed"):
                    self.hardware_state_changed.emit(True)
        else:
            # --- Deinitialize hardware and clean up ---
            try:
                self.hardware_controller.deinitialize()
            except Exception as e:
                print(f"Error deinitializing hardware: {e}")

            # Clear the param tab UI
            self.clear_detector_param_tab()

            self.xyStageIndicator.setStyleSheet(
                "background-color: gray; border-radius: 10px;"
            )
            self.cameraIndicator.setStyleSheet(
                "background-color: gray; border-radius: 10px;"
            )
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.xPosSpin.setEnabled(False)
            self.yPosSpin.setEnabled(False)
            self.gotoBtn.setEnabled(False)
            self.initializeBtn.setText("Initialize Hardware")
            self.hardware_initialized = False
            if hasattr(self, "hardware_state_changed"):
                self.hardware_state_changed.emit(False)

    def update_xy_pos(self):
        """
        Updates the X/Y position display in the UI from the hardware controller.
        Also updates beam cross overlay on the scene.
        """
        if getattr(self, "hardware_initialized", False) and hasattr(
            self, "stage_controller"
        ):
            try:
                x, y = self.stage_controller.get_xy_position()
                self.xPosSpin.setValue(x)
                self.yPosSpin.setValue(y)
            except Exception as e:
                print("Error reading stage pos:", e)
                x, y = 0, 0
                self.xPosSpin.setValue(0.0)
                self.yPosSpin.setValue(0.0)
        else:
            x, y = 0, 0
            self.xPosSpin.setValue(0.0)
            self.yPosSpin.setValue(0.0)

        # Remove old beam cross
        old = self.image_view.points_dict.get("beam", [])
        for itm in old:
            self.image_view.scene.removeItem(itm)

        x_pix, y_pix = self.mm_to_pixels(x, y)

        if x_pix >= 0 and y_pix >= 0:
            size = 15
            from PyQt5.QtGui import QPen

            pen = QPen(Qt.black, 5)
            hl = self._add_beam_line(
                x_pix - size, y_pix, x_pix + size, y_pix, pen
            )
            vl = self._add_beam_line(
                x_pix, y_pix - size, x_pix, y_pix + size, pen
            )
            self.image_view.points_dict["beam"] = [hl, vl]
        else:
            self.image_view.points_dict["beam"] = []

    def goto_stage_position(self):
        """
        Moves the stage to the user-specified X/Y coordinates.
        Updates X/Y spin boxes and calls the controller.
        """
        if hasattr(self, "stage_controller") and getattr(
            self, "hardware_initialized", False
        ):
            x = self.xPosSpin.value()
            y = self.yPosSpin.value()
            try:
                new_x, new_y = self.stage_controller.move_stage(x, y)
                self.xPosSpin.setValue(new_x)
                self.yPosSpin.setValue(new_y)
                self.update_xy_pos()
            except Exception as e:
                print("Error moving stage:", e)
        else:
            print("Stage not initialized; cannot GoTo.")

    def home_stage_button_clicked(self):
        """
        Moves the XY stage to home using the controller.
        """
        if (
            hasattr(self, "stage_controller")
            and self.stage_controller is not None
        ):
            x, y = self.stage_controller.home_stage(home_timeout=10)
            print(f"Home position reached: ({x}, {y})")
            self.xyStageIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
            )
        else:
            print("Stage not initialized.")

    def load_position_button_clicked(self):
        """
        Moves the XY stage to a fixed or user-defined load position.
        """
        if (
            hasattr(self, "stage_controller")
            and self.stage_controller is not None
        ):
            new_x, new_y = self.stage_controller.move_stage(
                -15, -6, move_timeout=10
            )
            print(f"Loaded position: ({new_x}, {new_y})")
        else:
            print("Stage not initialized.")
