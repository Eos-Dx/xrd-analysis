from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import LogNorm


class VisualizationMixin:
    def init_visualization_zone(self):
        """
        Initializes the central visualization zone which includes:
          - A label for measurement information.
          - Three plot panels (each with a toggle for log/linear scaling and a navigation toolbar for zooming):
              * Plot 1: Raw 2D data (imshow of 'measurement_data')
              * Plot 2: XY Plot (plot of 'radial_profile_data' vs 'q_range' in nm-1)
              * Plot 3: AgBH plot (data from calibration_df based on matching calib_name)
          - Navigation buttons (Previous, Next, Exclude_measurement)
          - A textedit for entering exclusion reasons.
        """
        # Create a central widget and set its layout
        self.visualization_widget = QWidget()
        main_layout = QVBoxLayout(self.visualization_widget)

        # --- Top: Measurement Information ---
        self.info_label = QLabel("Measurement Info")
        main_layout.addWidget(self.info_label)

        # --- Middle: Plots Area ---
        plots_layout = QHBoxLayout()

        # --- Plot Panel 1: Raw 2D Data ---
        self.plot1_widget = QWidget()
        plot1_layout = QVBoxLayout(self.plot1_widget)
        controls1_layout = QHBoxLayout()
        label1 = QLabel("Raw 2D Scale:")
        self.scale_combo_raw = QComboBox()
        self.scale_combo_raw.addItems(["Log", "Linear"])
        self.scale_combo_raw.setCurrentIndex(0)  # Default to Log
        controls1_layout.addWidget(label1)
        controls1_layout.addWidget(self.scale_combo_raw)
        plot1_layout.addLayout(controls1_layout)
        self.fig_raw = Figure(figsize=(4, 4))
        self.canvas_raw = FigureCanvas(self.fig_raw)
        plot1_layout.addWidget(self.canvas_raw)
        self.toolbar_raw = NavigationToolbar2QT(self.canvas_raw, self.visualization_widget)
        plot1_layout.addWidget(self.toolbar_raw)
        plots_layout.addWidget(self.plot1_widget)

        # --- Plot Panel 2: XY Plot (Azimuthal Integration) ---
        self.plot2_widget = QWidget()
        plot2_layout = QVBoxLayout(self.plot2_widget)
        controls2_layout = QHBoxLayout()
        label2 = QLabel("Azimuthal Y-Scale:")
        self.scale_combo_xy = QComboBox()
        self.scale_combo_xy.addItems(["Log", "Linear"])
        self.scale_combo_xy.setCurrentIndex(0)  # Default to Log
        controls2_layout.addWidget(label2)
        controls2_layout.addWidget(self.scale_combo_xy)
        plot2_layout.addLayout(controls2_layout)
        self.fig_xy = Figure(figsize=(4, 4))
        self.canvas_xy = FigureCanvas(self.fig_xy)
        plot2_layout.addWidget(self.canvas_xy)
        self.toolbar_xy = NavigationToolbar2QT(self.canvas_xy, self.visualization_widget)
        plot2_layout.addWidget(self.toolbar_xy)
        plots_layout.addWidget(self.plot2_widget)

        # --- Plot Panel 3: AgBH Plot ---
        self.plot3_widget = QWidget()
        plot3_layout = QVBoxLayout(self.plot3_widget)
        controls3_layout = QHBoxLayout()
        label3 = QLabel("AgBH Scale:")
        self.scale_combo_agbh = QComboBox()
        self.scale_combo_agbh.addItems(["Log", "Linear"])
        self.scale_combo_agbh.setCurrentIndex(0)  # Default to Log
        controls3_layout.addWidget(label3)
        controls3_layout.addWidget(self.scale_combo_agbh)
        plot3_layout.addLayout(controls3_layout)
        self.fig_agbh = Figure(figsize=(4, 4))
        self.canvas_agbh = FigureCanvas(self.fig_agbh)
        plot3_layout.addWidget(self.canvas_agbh)
        self.toolbar_agbh = NavigationToolbar2QT(self.canvas_agbh, self.visualization_widget)
        plot3_layout.addWidget(self.toolbar_agbh)
        plots_layout.addWidget(self.plot3_widget)

        main_layout.addLayout(plots_layout)

        # --- Bottom: Navigation Buttons and Exclusion Reason ---
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.exclude_button = QPushButton("Exclude_measurement")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.exclude_button)
        main_layout.addLayout(nav_layout)

        self.reason_textedit = QTextEdit()
        self.reason_textedit.setPlaceholderText("Reason")
        main_layout.addWidget(self.reason_textedit)

        # Set the visualization widget as the central widget of the MainWindow
        self.setCentralWidget(self.visualization_widget)

        # Initialize the current measurement index
        self.current_index = 0

        # Connect buttons to their respective actions
        self.prev_button.clicked.connect(self.show_previous_measurement)
        self.next_button.clicked.connect(self.show_next_measurement)
        self.exclude_button.clicked.connect(self.exclude_current_measurement)

    def display_measurement(self, index):
        """
        Updates the visualization zone with data from the transformed DataFrame at the given index.
        Expected DataFrame columns in self.transformed_df:
            - 'measurement_data': 2D numpy array for imshow in Plot 1.
            - 'radial_profile_data': 1D array for the y-axis of the XY plot.
            - 'q_range': 1D array for the x-axis of the XY plot (in nm-1).
            - 'patient_id', 'cancer_diagnosis', 'calibration_measurement_id', 'calib_name' for measurement info.
        """
        if self.transformed_df is None:
            return

        if index < 0 or index >= len(self.transformed_df):
            return

        self.current_index = index
        row = self.transformed_df.iloc[index]

        # Update measurement information label
        patient_id = row.get('id', 'N/A')
        externalId = row.get('externalId', 'N/A')
        cancer_diagnosis = row.get('isCancerDiagnosed', 'N/A')

        info_text = (f"Patient ID: {patient_id} | "
                     f"Cancer Diagnosis: {cancer_diagnosis} | "
                     f"External ID: {externalId}")
        self.info_label.setText(info_text)

        # --- Plot 1: Raw 2D Data ---
        self.fig_raw.clf()
        ax_raw = self.fig_raw.add_subplot(111)
        measurement_data = row.get('measurement_data')
        if measurement_data is not None:
            norm = LogNorm() if self.scale_combo_raw.currentText() == "Log" else None
            ax_raw.imshow(measurement_data, aspect='auto', norm=norm)
            ax_raw.set_title("Raw 2D data")
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
            # Toggle y-axis scale: log or linear
            if self.scale_combo_xy.currentText() == "Log":
                ax_xy.set_yscale("log")
            else:
                ax_xy.set_yscale("linear")
            # Zoom into full q_range and set maximum y value to 10**4
            ax_xy.set_xlim(min(q_range), max(q_range))
            current_ylim = ax_xy.get_ylim()
            #ax_xy.set_ylim(current_ylim[0], 10 ** 4)
            # Draw horizontal dashed red line at y=100
            ax_xy.axhline(y=100, color='red', linestyle='--')
        self.canvas_xy.draw()

        # --- Plot 3: AgBH Plot ---
        self.fig_agbh.clf()
        ax_agbh = self.fig_agbh.add_subplot(111)
        # Retrieve the calibration data corresponding to the current measurement's calib_name
        calib_name = row.get('calib_name')
        if self.calibration_df is not None and calib_name is not None:
            calib_rows = self.calibration_df[self.calibration_df['calib_name'] == calib_name]
            if not calib_rows.empty:
                calib_row = calib_rows.iloc[0]
                agbh_data = calib_row.get('measurement_data')
                if agbh_data is not None:
                    norm = LogNorm() if self.scale_combo_agbh.currentText() == "Log" else None
                    ax_agbh.imshow(agbh_data, aspect='auto', norm=norm)
                    ax_agbh.set_title("AgBH")
                else:
                    ax_agbh.text(0.5, 0.5, "No AgBH data available",
                                 ha="center", va="center", transform=ax_agbh.transAxes)
            else:
                ax_agbh.text(0.5, 0.5, "No matching calibration data",
                             ha="center", va="center", transform=ax_agbh.transAxes)
        else:
            ax_agbh.text(0.5, 0.5, "No AgBH data available",
                         ha="center", va="center", transform=ax_agbh.transAxes)
        self.canvas_agbh.draw()

        # Clear the reason text edit when displaying a new measurement
        self.reason_textedit.clear()

    def show_previous_measurement(self):
        """Display the previous measurement."""
        new_index = self.current_index - 1
        if new_index >= 0:
            self.display_measurement(new_index)

    def show_next_measurement(self):
        """Display the next measurement."""
        new_index = self.current_index + 1
        if self.transformed_df is not None and new_index < len(self.transformed_df):
            self.display_measurement(new_index)

    def exclude_current_measurement(self):
        """
        Append the current measurement's ID and the entered exclusion reason to 'exclude.txt'.
        The measurement ID is taken from the 'calibration_measurement_id' column.
        """
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        calibration_measurement_id = row.get('calibration_measurement_id', 'N/A')
        reason = self.reason_textedit.toPlainText().strip()
        if reason:
            line = f"{calibration_measurement_id}: {reason}\n"
            with open("exclude.txt", "a") as f:
                f.write(line)
