from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class VisualizationMixin:
    def init_visualization_zone(self):
        """
        Initializes the central visualization zone which includes:
          - A label for measurement information.
          - Three plots:
              * Raw 2D data (imshow of 'measurement_data')
              * XY plot (plot of 'radial_profile_data' vs 'q_range' in nm-1)
              * AgBH plot (placeholder for now)
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
        # Plot 1: Raw 2D Data
        self.fig_raw = Figure(figsize=(4, 4))
        self.canvas_raw = FigureCanvas(self.fig_raw)
        plots_layout.addWidget(self.canvas_raw)

        # Plot 2: XY Plot (Azimuthal Integration)
        self.fig_xy = Figure(figsize=(4, 4))
        self.canvas_xy = FigureCanvas(self.fig_xy)
        plots_layout.addWidget(self.canvas_xy)

        # Plot 3: AgBH (Placeholder)
        self.fig_agbh = Figure(figsize=(4, 4))
        self.canvas_agbh = FigureCanvas(self.fig_agbh)
        plots_layout.addWidget(self.canvas_agbh)

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
        Expected DataFrame columns:
            - 'measurement_data': 2D numpy array for imshow.
            - 'radial_profile_data': 1D array for the y-axis of the XY plot.
            - 'q_range': 1D array for the x-axis of the XY plot (in nm-1).
            - 'patient_id', 'cancer_diagnosis', 'calibration_measurement_id' for measurement info.
        """
        if self.transformed_df is None:
            return

        if index < 0 or index >= len(self.transformed_df):
            return

        self.current_index = index
        row = self.transformed_df.iloc[index]

        # Update measurement information label
        patient_id = row.get('patient_id', 'N/A')
        cancer_diagnosis = row.get('cancer_diagnosis', 'N/A')
        calibration_measurement_id = row.get('calibration_measurement_id', 'N/A')
        info_text = (f"Patient ID: {patient_id} | "
                     f"Cancer Diagnosis: {cancer_diagnosis} | "
                     f"Calibration Measurement ID: {calibration_measurement_id}")
        self.info_label.setText(info_text)

        # --- Plot 1: Raw 2D Data ---
        self.fig_raw.clf()
        ax_raw = self.fig_raw.add_subplot(111)
        measurement_data = row.get('measurement_data')
        if measurement_data is not None:
            ax_raw.imshow(measurement_data, aspect='auto')
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
        self.canvas_xy.draw()

        # --- Plot 3: AgBH (Placeholder) ---
        self.fig_agbh.clf()
        ax_agbh = self.fig_agbh.add_subplot(111)
        ax_agbh.text(0.5, 0.5, "AgBH plot (placeholder)",
                     ha="center", va="center", transform=ax_agbh.transAxes)
        ax_agbh.set_title("AgBH")
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
