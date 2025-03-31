from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import LogNorm


class VisualizationMixin:
    def init_visualization_zone(self):
        """
        Initializes the Measurements Zone as a dockable widget which includes:
          - A label for measurement information.
          - Three plot panels:
              * Plot 1: Raw 2D data.
              * Plot 2: XY Plot (Azimuthal Integration) with user-controlled horizontal and vertical lines.
              * Plot 3: AgBH Plot.
        """
        # Create a dock widget for measurements.
        self.visualization_dock = QDockWidget("Measurements Zone", self)
        self.visualization_dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        # Create the main widget for the dock.
        self.visualization_widget = QWidget()
        self.visualization_layout = QVBoxLayout(self.visualization_widget)

        # --- Measurements Zone Content ---
        # Measurement information label.
        self.info_label = QLabel("Measurement Info")
        self.visualization_layout.addWidget(self.info_label)

        # --- Plots Area ---
        plots_layout = QHBoxLayout()

        # Plot Panel 1: Raw 2D Data
        self.plot1_widget = QWidget()
        plot1_layout = QVBoxLayout(self.plot1_widget)
        controls1_layout = QHBoxLayout()
        label1 = QLabel("Raw 2D Scale:")
        self.scale_combo_raw = QComboBox()
        self.scale_combo_raw.addItems(["Log", "Linear"])
        self.scale_combo_raw.setCurrentIndex(0)
        controls1_layout.addWidget(label1)
        controls1_layout.addWidget(self.scale_combo_raw)
        plot1_layout.addLayout(controls1_layout)
        self.fig_raw = Figure(figsize=(4, 4))
        self.canvas_raw = FigureCanvas(self.fig_raw)
        plot1_layout.addWidget(self.canvas_raw)
        self.toolbar_raw = NavigationToolbar2QT(self.canvas_raw, self)
        plot1_layout.addWidget(self.toolbar_raw)
        plots_layout.addWidget(self.plot1_widget)

        # Plot Panel 2: XY Plot (Azimuthal Integration)
        self.plot2_widget = QWidget()
        plot2_layout = QVBoxLayout(self.plot2_widget)
        controls2_layout = QHBoxLayout()
        label2 = QLabel("Azimuthal Y-Scale:")
        self.scale_combo_xy = QComboBox()
        self.scale_combo_xy.addItems(["Log", "Linear"])
        self.scale_combo_xy.setCurrentIndex(0)
        controls2_layout.addWidget(label2)
        controls2_layout.addWidget(self.scale_combo_xy)

        # Horizontal line control.
        hline_label = QLabel("HLine Value:")
        self.hline_line_edit = QLineEdit()
        self.hline_line_edit.setPlaceholderText("Enter horizontal line value")
        controls2_layout.addWidget(hline_label)
        controls2_layout.addWidget(self.hline_line_edit)
        # Update the plot when the horizontal line value is changed.
        self.hline_line_edit.editingFinished.connect(self.update_horizontal_line)

        # Vertical line control.
        vline_label = QLabel("Vertical Line Value:")
        self.vertical_line_line_edit = QLineEdit()
        self.vertical_line_line_edit.setPlaceholderText("Enter vertical line value")
        controls2_layout.addWidget(vline_label)
        controls2_layout.addWidget(self.vertical_line_line_edit)

        self.add_vline_button = QPushButton("+Vertical")
        self.remove_vline_button = QPushButton("-Vertical")
        controls2_layout.addWidget(self.add_vline_button)
        controls2_layout.addWidget(self.remove_vline_button)

        # Connect vertical line buttons.
        self.add_vline_button.clicked.connect(self.add_vertical_line)
        self.remove_vline_button.clicked.connect(self.remove_vertical_line)

        plot2_layout.addLayout(controls2_layout)
        self.fig_xy = Figure(figsize=(4, 4))
        self.canvas_xy = FigureCanvas(self.fig_xy)
        plot2_layout.addWidget(self.canvas_xy)
        self.toolbar_xy = NavigationToolbar2QT(self.canvas_xy, self)
        plot2_layout.addWidget(self.toolbar_xy)
        plots_layout.addWidget(self.plot2_widget)

        # Plot Panel 3: AgBH Plot
        self.plot3_widget = QWidget()
        plot3_layout = QVBoxLayout(self.plot3_widget)
        controls3_layout = QHBoxLayout()
        label3 = QLabel("AgBH Scale:")
        self.scale_combo_agbh = QComboBox()
        self.scale_combo_agbh.addItems(["Log", "Linear"])
        self.scale_combo_agbh.setCurrentIndex(0)
        controls3_layout.addWidget(label3)
        controls3_layout.addWidget(self.scale_combo_agbh)
        plot3_layout.addLayout(controls3_layout)
        self.fig_agbh = Figure(figsize=(4, 4))
        self.canvas_agbh = FigureCanvas(self.fig_agbh)
        plot3_layout.addWidget(self.canvas_agbh)
        self.toolbar_agbh = NavigationToolbar2QT(self.canvas_agbh, self)
        plot3_layout.addWidget(self.toolbar_agbh)
        plots_layout.addWidget(self.plot3_widget)

        self.visualization_layout.addLayout(plots_layout)

        # Set the dock widget's main widget.
        self.visualization_dock.setWidget(self.visualization_widget)
        # Add the dock widget to the main window.
        self.addDockWidget(Qt.LeftDockWidgetArea, self.visualization_dock)

        # Initialize the current measurement index and container for vertical lines.
        self.current_index = 0
        self.vertical_lines_values = []

    def display_measurement(self, index):
        """
        Updates the Measurements Zone with data from the transformed DataFrame at the given index.
        Expected columns in self.transformed_df include:
          - 'measurement_data': 2D array for Plot 1.
          - 'radial_profile_data' and 'q_range': for Plot 2.
          - 'calib_name': for Plot 3 to retrieve calibration data.
          - 'id', 'externalId', 'isCancerDiagnosed' for the info label.
        """
        if self.transformed_df is None:
            return

        if index < 0 or index >= len(self.transformed_df):
            return

        self.current_index = index
        row = self.transformed_df.iloc[index]

        # Update measurement information.
        measurement_id = row.get('id', 'N/A')
        patient_id = row.get('name', 'N/A')
        cancer_diagnosis = row.get('isCancerDiagnosed', 'N/A')
        info_text = (f"Meusurement ID: {measurement_id} | "
                     f"Cancer Diagnosis: {cancer_diagnosis} | "
                     f"Patient ID: {patient_id}")
        self.info_label.setText(info_text)

        # --- Plot 1: Raw 2D Data ---
        self.fig_raw.clf()
        ax_raw = self.fig_raw.add_subplot(111)
        measurement_data = row.get('measurement_data')
        if measurement_data is not None:
            norm = LogNorm() if self.scale_combo_raw.currentText() == "Log" else None
            ax_raw.imshow(measurement_data, aspect='equal', norm=norm)
            ax_raw.set_title("Raw 2D data")
            ax_raw.set_aspect('equal')
        self.canvas_raw.setMinimumSize(200, 200)
        self.canvas_raw.draw()

        # --- Plot 2: XY Plot (Azimuthal Integration) ---
        self.fig_xy.clf()
        ax_xy = self.fig_xy.add_subplot(111)
        radial_profile_data = row.get('radial_profile_data')
        q_range = row.get('q_range')
        if radial_profile_data is not None and q_range is not None:
            ax_xy.plot(q_range, radial_profile_data)
            ax_xy.set_title("Azimuthal integration")
            ax_xy.set_xlabel("q_range (nm-1)")
            ax_xy.set_ylabel("radial_profile_data")
            if self.scale_combo_xy.currentText() == "Log":
                ax_xy.set_yscale("log")
            else:
                ax_xy.set_yscale("linear")
            ax_xy.set_xlim(min(q_range), max(q_range))

            # Draw horizontal line using user-defined value (if valid).
            try:
                h_value = float(self.hline_line_edit.text())
                ax_xy.axhline(y=h_value, color='red', linestyle='--')
            except ValueError:
                pass

            # Draw vertical lines based on stored values.
            for v in self.vertical_lines_values:
                ax_xy.axvline(x=v, color='blue', linestyle='--')
        self.canvas_xy.setMinimumSize(200, 150)
        self.canvas_xy.draw()

        # --- Plot 3: AgBH Plot ---
        self.fig_agbh.clf()
        ax_agbh = self.fig_agbh.add_subplot(111)
        calib_name = row.get('calib_name')
        if self.calibration_df is not None and calib_name is not None:
            calib_rows = self.calibration_df[self.calibration_df['calib_name'] == calib_name]
            if not calib_rows.empty:
                calib_row = calib_rows.iloc[0]
                agbh_data = calib_row.get('measurement_data')
                if agbh_data is not None:
                    norm = LogNorm() if self.scale_combo_agbh.currentText() == "Log" else None
                    ax_agbh.imshow(agbh_data, aspect='equal', norm=norm)
                    ax_agbh.set_title("AgBH")
                    ax_agbh.set_aspect('equal')
                else:
                    ax_agbh.text(0.5, 0.5, "No AgBH data available",
                                 ha="center", va="center", transform=ax_agbh.transAxes)
            else:
                ax_agbh.text(0.5, 0.5, "No matching calibration data",
                             ha="center", va="center", transform=ax_agbh.transAxes)
        else:
            ax_agbh.text(0.5, 0.5, "No AgBH data available",
                         ha="center", va="center", transform=ax_agbh.transAxes)
        self.canvas_agbh.setMinimumSize(200, 200)
        self.canvas_agbh.draw()

    def update_horizontal_line(self):
        """
        Updates the XY plot when the horizontal line value is changed.
        """
        self.display_measurement(self.current_index)

    def add_vertical_line(self):
        """
        Reads the vertical line value from the input field and adds it to the list of vertical lines.
        Redraws the XY plot to include the new vertical line.
        """
        try:
            v_value = float(self.vertical_line_line_edit.text())
        except ValueError:
            return
        self.vertical_lines_values.append(v_value)
        self.display_measurement(self.current_index)

    def remove_vertical_line(self):
        """
        Removes the last added vertical line value and redraws the XY plot.
        """
        if self.vertical_lines_values:
            self.vertical_lines_values.pop()
            self.display_measurement(self.current_index)
