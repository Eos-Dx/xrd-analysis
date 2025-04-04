import os
import numpy as np
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QDockWidget, QLabel,
    QSpinBox, QLineEdit, QFileDialog,
    QSpacerItem, QSizePolicy, QProgressBar,
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QHBoxLayout
from PyQt5.QtCore import  QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

from xrdanalysis.data_processing.azimuthal_integration import initialize_azimuthal_integrator_df
from hardware.Ulster.hardware.hardware_control import *
from hardware.Ulster.hardware.auxiliary import encode_image_to_base64

class ZoneMeasurementsMixin:

    def createZoneMeasurementsWidget(self):
        """
        Creates a dock widget for zone measurements.
        This widget (placed at the bottom right) includes:
          - An Initialize button (to set up hardware),
          - Start, Pause, and Stop buttons,
          - An Integration Time (sec) control,
          - A Repeat control,
          - Folder and File Name controls,
          - LED indicators for the XY stage and Camera,
          - A progress indicator showing percentage complete and estimated time remaining.
        """
        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Hardware control buttons.
        buttonLayout = QHBoxLayout()
        self.initializeBtn = QPushButton("Initialize")
        self.initializeBtn.clicked.connect(self.initializeHardware)
        self.startBtn = QPushButton("Start measurement")
        self.startBtn.clicked.connect(self.startMeasurements)
        self.pauseBtn = QPushButton("Pause")
        self.pauseBtn.clicked.connect(self.pauseMeasurements)
        self.stopBtn = QPushButton("Stop")
        self.stopBtn.clicked.connect(self.stopMeasurements)
        # Initially, buttons are disabled until hardware is initialized.
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        buttonLayout.addWidget(self.initializeBtn)
        buttonLayout.addWidget(self.startBtn)
        buttonLayout.addWidget(self.pauseBtn)
        buttonLayout.addWidget(self.stopBtn)
        layout.addLayout(buttonLayout)

        # Hardware status indicators.
        statusLayout = QHBoxLayout()
        xyLabel = QLabel("XY Stage:")
        self.xyStageIndicator = QLabel()
        self.xyStageIndicator.setFixedSize(20, 20)
        self.xyStageIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        statusLayout.addWidget(xyLabel)
        statusLayout.addWidget(self.xyStageIndicator)
        cameraLabel = QLabel("Camera:")
        self.cameraIndicator = QLabel()
        self.cameraIndicator.setFixedSize(20, 20)
        self.cameraIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        statusLayout.addWidget(cameraLabel)
        statusLayout.addWidget(self.cameraIndicator)
        # Spacer to push items left.
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        statusLayout.addItem(spacer)
        layout.addLayout(statusLayout)

        # Integration time control.
        integrationLayout = QHBoxLayout()
        integrationLabel = QLabel("Integration Time (sec):")
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(60)
        self.integrationSpinBox.setValue(1)
        integrationLayout.addWidget(integrationLabel)
        integrationLayout.addWidget(self.integrationSpinBox)
        layout.addLayout(integrationLayout)

        # Repeat control.
        repeatLayout = QHBoxLayout()
        repeatLabel = QLabel("Repeat:")
        self.repeatSpinBox = QSpinBox()
        self.repeatSpinBox.setMinimum(1)
        self.repeatSpinBox.setMaximum(10)
        self.repeatSpinBox.setValue(1)
        repeatLayout.addWidget(repeatLabel)
        repeatLayout.addWidget(self.repeatSpinBox)
        layout.addLayout(repeatLayout)

        # Folder selection control.
        folderLayout = QHBoxLayout()
        folderLabel = QLabel("Save Folder:")
        self.folderLineEdit = QLineEdit()
        default_folder = self.config.get("default_folder", "") if hasattr(self, "config") else ""
        self.folderLineEdit.setText(default_folder)
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browseFolder)
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.folderLineEdit)
        folderLayout.addWidget(self.browseBtn)
        layout.addLayout(folderLayout)

        # File name control.
        fileNameLayout = QHBoxLayout()
        fileNameLabel = QLabel("File Name:")
        self.fileNameLineEdit = QLineEdit()
        fileNameLayout.addWidget(fileNameLabel)
        fileNameLayout.addWidget(self.fileNameLineEdit)
        layout.addLayout(fileNameLayout)

        # Progress indicator.
        progressLayout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)  # Will be reset when measurements start.
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        layout.addLayout(progressLayout)

        container.setLayout(layout)
        self.zoneMeasurementsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)

        # Ensure the unified dictionary is initialized.
        self.image_view.points_dict = {
            "generated": {"points": [], "zones": []},
            "user": {"points": [], "zones": []}
        }

    def browseFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folderLineEdit.setText(folder)

    def initializeHardware(self):
        """
        Initializes the XY stage and Pixet camera.
        Updates LED indicators and enables the Start button if both hardware components are active.
        """
        try:
            from hardware.Ulster.hardware import hardware_control as hc
        except Exception as e:
            print("Error importing hardware control module:", e)
            self.xyStageIndicator.setStyleSheet("background-color: red; border-radius: 10px;")
            self.cameraIndicator.setStyleSheet("background-color: red; border-radius: 10px;")
            return

        try:
            # Retrieve serial number from the configuration JSON and convert to byte string.
            serial_str = self.config.get("serial_number_XY", "default_serial")
            c_serial_number = c_char_p(serial_str.encode('utf-8'))
            self.serial_number = c_serial_number  # Save for later use.
            self.xystage_lib = hc.init_stage(sim_en=True, serial_num=c_serial_number, x_chan=1, y_chan=2)
            print("XY stage initialized.")
            self.xyStageIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        except Exception as e:
            print("Error initializing XY stage:", e)
            self.xyStageIndicator.setStyleSheet("background-color: red; border-radius: 10px;")

        try:
            self.pixet, self.detector = hc.init_detector(capture_enabled=True)
            print("Pixet camera initialized.")
            self.cameraIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        except Exception as e:
            print("Error initializing Pixet camera:", e)
            self.cameraIndicator.setStyleSheet("background-color: red; border-radius: 10px;")

        if ("green" in self.xyStageIndicator.styleSheet() and
                "green" in self.cameraIndicator.styleSheet()):
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)

        # Store the hardware control module reference for later use.
        self.hc = hc

    def startMeasurements(self):
        """
        Sorts all points (both generated and user-defined) by increasing X (mm) then Y (mm)
        and begins the measurement sequence. Progress is visualized.
        """
        self.validate_folder()

        self.manualSaveState()
        with open(Path(self.measurement_folder) / f'state.json', "w") as f:
            # Save the current state of the points_dict to a JSON file.
            self.state['image_base64'] = encode_image_to_base64(self.image_view.current_image_path)
            json.dump(self.state, f, indent=4)
        if self.pointsTable.rowCount() == 0:
            print("No points available for measurement.")
            return

        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        self.stopBtn.setEnabled(True)
        self.stopped = False
        self.paused = False

        # Use the unified dictionary.
        generated_points = self.image_view.points_dict["generated"]["points"]
        user_points = self.image_view.points_dict["user"]["points"]
        all_points = []
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            x_mm = center.x() / self.pixel_to_mm_ratio
            y_mm = center.y() / self.pixel_to_mm_ratio
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = center.x() / self.pixel_to_mm_ratio
            y_mm = center.y() / self.pixel_to_mm_ratio
            all_points.append((offset + j, x_mm, y_mm))
        # Sort by increasing X then Y.
        all_points_sorted = sorted(all_points, key=lambda tup: (tup[1], tup[2]))
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        self.current_measurement_sorted_index = 0

        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        self.integration_time = self.integrationSpinBox.value()
        self.initial_estimate = self.total_points * self.integration_time
        self.measurementStartTime = time.time()
        self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")
        print("Starting measurements in sorted order...")
        self.measureNextPoint()

    def measureNextPoint(self):
        """
        Proceeds to measure the next point by:
          - Moving the stage,
          - Capturing data (dummy capture writes a 10x10 random matrix as a txt file),
          - Converting the txt file to an npy file,
          - Adding a clickable measurement button to the table.
        The filename is built as: self.measurement_folder + base_file_name + X_Y coordinates + timestamp.
        """
        if self.stopped:
            print("Measurement stopped.")
            return
        if self.paused:
            print("Measurement is paused. Waiting for resume.")
            return
        if self.current_measurement_sorted_index >= self.total_points:
            print("All points measured.")
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)
            return

        # Determine which point to measure.
        index = self.sorted_indices[self.current_measurement_sorted_index]
        gp = self.image_view.points_dict["generated"]["points"]
        up = self.image_view.points_dict["user"]["points"]
        if index < len(gp):
            point_item = gp[index]
            zone_item = self.image_view.points_dict["generated"]["zones"][index]
        else:
            user_index = index - len(gp)
            point_item = up[user_index]
            zone_item = self.image_view.points_dict["user"]["zones"][user_index]

        # Get point coordinates in mm.
        center = point_item.sceneBoundingRect().center()
        x_mm = center.x() / self.pixel_to_mm_ratio
        y_mm = center.y() / self.pixel_to_mm_ratio

        # Move the stage.
        x_chan = 1
        y_chan = 2
        new_x, new_y = self.hc.move_stage(
            self.xystage_lib, self.serial_number, x_chan, y_chan, x_mm, y_mm, move_timeout=1
        )

        # Build the filename using the self.measurement_folder, base file name, coordinates, and a timestamp.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = self.fileNameLineEdit.text().strip()
        txt_filename = os.path.join(self.measurement_folder, f"{base_name}_{x_mm:.2f}_{y_mm:.2f}_{timestamp}.txt")

        # Capture the measurement data (dummy capture writes a 10x10 matrix to txt).
        self.hc.capture_point(self.detector, self.pixet, 1, self.integration_time, txt_filename)

        # Convert the captured txt file into an npy file.
        try:
            data = np.loadtxt(txt_filename)
            npy_filename = os.path.join(self.measurement_folder, f"{base_name}_{x_mm:.2f}_{y_mm:.2f}_{timestamp}.npy")
            np.save(npy_filename, data)
            print(f"Converted {txt_filename} to {npy_filename}")
        except Exception as e:
            print(f"Error converting file: {e}")
            npy_filename = txt_filename  # fallback

        # Update the measurement column: add a clickable button.
        table_row = self.sorted_indices[self.current_measurement_sorted_index]
        self.addMeasurementToTable(table_row, npy_filename)

        # Visual feedback.
        green_brush = QColor(0, 255, 0)
        point_item.setBrush(green_brush)
        if zone_item:
            green_zone = QColor(0, 255, 0)
            green_zone.setAlphaF(0.2)
            zone_item.setBrush(green_zone)

        # Proceed to the next point after a delay.
        QTimer.singleShot(1000, self.measurementFinished)

    def addMeasurementToTable(self, row, measurement_filename):
        """
        Adds a clickable button to the "Measurement" cell of the table at the given row.
        Clicking the button will open the measurement analysis window.
        """
        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton

        # Check if a custom widget already exists in the cell.
        widget = self.pointsTable.cellWidget(row, 5)
        if widget is None:
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)
            self.pointsTable.setCellWidget(row, 5, widget)
        else:
            layout = widget.layout()

        # Create a new button for this measurement.
        btn = QPushButton(os.path.basename(measurement_filename))
        btn.setStyleSheet(
            "Text-align:left; border: none; color: blue; text-decoration: underline;"
        )
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self.showMeasurement(measurement_filename))
        layout.addWidget(btn)

    def showMeasurement(self, measurement_filename):
        """
        Opens a new window that loads the given npy file,
        displays the 2D image (using imshow),
        and to its right shows the azimuthal integration using the real
        integration code from the azimuthal integration module.
        The window is modeless (non-blocking) so the main application remains responsive.
        """
        # Load the measurement data.
        data = np.load(measurement_filename)

        # --- Calibration parameters ---
        pixel_size = 55e-6  # in meters (55 µm)
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        center_row, center_column = max_idx  # row corresponds to y, column to x
        wavelength = 1.54  # in angstroms
        sample_distance_mm = 100.0

        # --- Initialize the integrator ---
        ai = initialize_azimuthal_integrator_df(
            pixel_size,
            center_column,
            center_row,
            wavelength,
            sample_distance_mm
        )

        # --- Perform the integration ---
        npt = 100  # Number of integration points
        try:
            result = ai.integrate1d(data, npt, unit="q_nm^-1", error_model="azimuthal")
            radial = result.radial
            intensity = result.intensity
        except Exception as e:
            print("Error integrating data:", e)
            return

        # --- Create the dialog to display the plots ---
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Azimuthal Integration: {os.path.basename(measurement_filename)}")
        layout = QHBoxLayout(dialog)

        # Left plot: Display the raw 2D image.
        fig1 = Figure(figsize=(5, 5))
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        im = ax1.imshow(data, cmap="viridis")
        ax1.set_title("2D Image")
        fig1.colorbar(im, ax=ax1)

        # Right plot: Display the azimuthal integration result.
        fig2 = Figure(figsize=(5, 5))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        ax2.plot(radial, intensity, '-o')
        ax2.set_title("Azimuthal Integration")
        ax2.set_xlabel("q (nm^-1)")
        ax2.set_ylabel("Intensity")

        layout.addWidget(canvas1)
        layout.addWidget(canvas2)

        dialog.resize(1000, 500)

        # Ensure dialogs remain in memory by storing them in a list attribute.
        if not hasattr(self, "_open_measurement_windows"):
            self._open_measurement_windows = []
        self._open_measurement_windows.append(dialog)

        # Remove the dialog from the list when it is closed.
        dialog.finished.connect(lambda _: self._open_measurement_windows.remove(dialog))

        # Show the dialog as a modeless window.
        dialog.show()

    def pauseMeasurements(self):
        """
        Toggles the pause/resume state.
        When paused, the measurement loop halts and the button text changes to "Resume".
        The button remains enabled.
        """
        if not hasattr(self, 'paused'):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pauseBtn.setText("Resume")
            print("Measurements paused.")
        else:
            self.paused = False
            self.pauseBtn.setText("Pause")
            print("Measurements resumed.")
            self.measureNextPoint()

    def stopMeasurements(self):
        """
        Stops the measurement process and resets state.
        """
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurement stopped.")
        self.startBtn.setEnabled(True)
        self.pauseBtn.setText("Pause")
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        print("Measurements stopped and reset.")

    def measurementFinished(self):
        """
        Called after a measurement completes.
        Updates progress visualization and proceeds to the next point if not paused or stopped.
        """
        if self.stopped:
            print("Measurement stopped.")
            return

        self.current_measurement_sorted_index += 1
        self.progressBar.setValue(self.current_measurement_sorted_index)
        elapsed = time.time() - self.measurementStartTime
        if self.current_measurement_sorted_index > 0:
            avg_time = elapsed / self.current_measurement_sorted_index
            remaining = avg_time * (self.total_points - self.current_measurement_sorted_index)
            percent_complete = (self.current_measurement_sorted_index / self.total_points) * 100
            self.timeRemainingLabel.setText(f"{percent_complete:.0f}% done, {remaining:.0f} sec remaining")

        if self.current_measurement_sorted_index < self.total_points and not self.paused and not self.stopped:
            self.measureNextPoint()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                print("All points measured.")
                self.pauseBtn.setEnabled(False)
                self.stopBtn.setEnabled(False)
                self.startBtn.setEnabled(True)

    def validate_folder(self):
        # Validate the self.measurement_folder.
        self.measurement_folder = self.folderLineEdit.text().strip()
        if not self.measurement_folder:
            self.measurement_folder = os.getcwd()
        if not os.path.exists(self.measurement_folder):
            try:
                os.makedirs(self.measurement_folder, exist_ok=True)
            except Exception as e:
                print(f"Error creating self.measurement_folder {self.measurement_folder}: {e}. Using current directory.")
                self.measurement_folder = os.getcwd()
        if not os.access(self.measurement_folder, os.W_OK):
            print(f"Folder {self.measurement_folder} is not writable. Using current directory.")
            self.measurement_folder = os.getcwd()