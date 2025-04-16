from PyQt5.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QLineEdit, QPushButton, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import LogNorm
import seaborn as sns  # Added import for seaborn
import numpy as np

class VisualizationMixin:

    def init_visualization_zone(self):
        """
        Initializes the Measurements Zone as a dockable widget which includes:
          - A label for measurement information.
          - Two rows of plots:
              * First row:
                  - Left: Plot 1 (Raw 2D data)
                  - Right: Plot 2 (XY Plot: Azimuthal Integration with user-controlled lines)
              * Second row:
                  - Left: Plot 3 (AgBH heatmap)
                  - Right: Plot 4 (Cake Representation placeholder)
        The dock and its components are configured so that the minimum configuration fits in 600 (width) x 400 (height) pixels.
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

        # --- Plots Area: Two Rows Layout ---
        plots_main_layout = QVBoxLayout()

        # ----- First Row -----
        first_row_layout = QHBoxLayout()

        # Plot Panel 1: Raw 2D Data (left column)
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
        self.fig_raw = Figure(figsize=(2.5, 2.5))
        self.canvas_raw = FigureCanvas(self.fig_raw)
        # Adjust minimum size for the canvas.
        plot1_layout.addWidget(self.canvas_raw)
        self.toolbar_raw = NavigationToolbar2QT(self.canvas_raw, self)
        plot1_layout.addWidget(self.toolbar_raw)
        first_row_layout.addWidget(self.plot1_widget)

        # Plot Panel 2: XY Plot (Azimuthal Integration) (right column)
        self.plot2_widget = QWidget()
        plot2_layout = QVBoxLayout(self.plot2_widget)
        controls2_layout = QHBoxLayout()
        label2 = QLabel("Y-Scale:")
        self.scale_combo_xy = QComboBox()
        self.scale_combo_xy.addItems(["Log", "Linear"])
        self.scale_combo_xy.setCurrentIndex(0)
        controls2_layout.addWidget(label2)
        controls2_layout.addWidget(self.scale_combo_xy)
        # Horizontal line control.
        hline_label = QLabel("HLine")
        self.hline_line_edit = QLineEdit()
        self.hline_line_edit.setPlaceholderText("")
        controls2_layout.addWidget(hline_label)
        controls2_layout.addWidget(self.hline_line_edit)
        spacer = QSpacerItem(10, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        controls2_layout.addItem(spacer)
        self.hline_line_edit.editingFinished.connect(self.update_horizontal_line)

        # Vertical line control.
        vline_label = QLabel("VLine")
        self.vertical_line_line_edit = QLineEdit()
        self.vertical_line_line_edit.setPlaceholderText("")
        controls2_layout.addWidget(vline_label)
        controls2_layout.addWidget(self.vertical_line_line_edit)
        self.add_vline_button = QPushButton("+V")
        self.remove_vline_button = QPushButton("-V")
        controls2_layout.addWidget(self.add_vline_button)
        controls2_layout.addWidget(self.remove_vline_button)
        self.add_vline_button.clicked.connect(self.add_vertical_line)
        self.remove_vline_button.clicked.connect(self.remove_vertical_line)
        plot2_layout.addLayout(controls2_layout)
        self.fig_xy = Figure(figsize=(2.5, 2.5))
        self.canvas_xy = FigureCanvas(self.fig_xy)
        plot2_layout.addWidget(self.canvas_xy)
        self.toolbar_xy = NavigationToolbar2QT(self.canvas_xy, self)
        plot2_layout.addWidget(self.toolbar_xy)
        first_row_layout.addWidget(self.plot2_widget)

        plots_main_layout.addLayout(first_row_layout)

        # ----- Second Row -----
        second_row_layout = QHBoxLayout()

        # Plot Panel 3: AgBH Plot (left column) - modified to use sns.heatmap
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
        self.fig_agbh = Figure(figsize=(2.5, 2.5))
        self.canvas_agbh = FigureCanvas(self.fig_agbh)
        plot3_layout.addWidget(self.canvas_agbh)
        self.toolbar_agbh = NavigationToolbar2QT(self.canvas_agbh, self)
        plot3_layout.addWidget(self.toolbar_agbh)
        second_row_layout.addWidget(self.plot3_widget)

        # Plot Panel 4: Cake Representation Placeholder (right column)
        self.plot4_widget = QWidget()
        plot4_layout = QVBoxLayout(self.plot4_widget)
        self.fig_cake = Figure(figsize=(2.5, 2.5))
        self.canvas_cake = FigureCanvas(self.fig_cake)
        plot4_layout.addWidget(self.canvas_cake)
        second_row_layout.addWidget(self.plot4_widget)
        self.toolbar_cake = NavigationToolbar2QT(self.canvas_cake, self)
        plot4_layout.addWidget(self.toolbar_cake)

        plots_main_layout.addLayout(second_row_layout)
        self.visualization_layout.addLayout(plots_main_layout)

        # Set the dock widget's main widget and add it to the main window.
        self.visualization_dock.setWidget(self.visualization_widget)
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
          - Other keys for measurement info.
        """
        for fig in [self.fig_raw, self.fig_xy, self.fig_agbh, self.fig_cake]:
            fig.clf()
        if self.transformed_df is None or index < 0 or index >= len(self.transformed_df):
            return
        self.current_index = index
        row = self.transformed_df.iloc[index]
        row2D = self.transformed_df2D.iloc[index]

        # Update measurement information.
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        patient_id = row.get('patientId', 'N/A')
        cancer_diagnosis = row.get('isCancerDiagnosed', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        meas_name = row.get('meas_name', 'N/A')
        info_text = (f"Measurement group ID: {measurement_group_id} | "
                     f"Patient DB ID: {patient_db_id} | "
                     f"Patient ID: {patient_id} | " 
                     f"Specimen DB ID: {specimen_db_id} | "
                     f"Cancer diagnosed: {cancer_diagnosis} | "
                     f"Measurement name: {meas_name}")
        self.info_label.setText(info_text)

        # --- Plot 1: Raw 2D Data ---
        self.fig_raw.clf()
        ax_raw = self.fig_raw.add_subplot(111)
        measurement_data = row.get('measurement_data')
        if measurement_data is not None:
            norm = LogNorm() if self.scale_combo_raw.currentText() == "Log" else None
            #ax_raw.imshow(measurement_data, aspect='equal', norm=norm)
            sns.heatmap(measurement_data, ax=ax_raw, robust=True, square=True, cbar=True)
            ax_raw.set_aspect('equal')
        self.canvas_raw.draw_idle()

        # --- Plot 2: XY Plot (Azimuthal Integration) ---
        self.fig_xy.clf()
        ax_xy = self.fig_xy.add_subplot(111)
        radial_profile_data = row.get('radial_profile_data')
        a = row2D.get('radial_profile_data')
        radial_profile_data2D = np.sum(row2D.get('radial_profile_data'), axis=0)
        q_range2D = row2D.get('q_range')
        q_range = row.get('q_range')
        if radial_profile_data is not None and q_range is not None:
            ax_xy.plot(q_range, radial_profile_data, label='Azimuthal Integration')
            ax_xy.plot(q_range2D, radial_profile_data2D, label='Cake sum')
            ax_xy.set_xlabel("q_range (nm-1)")
            ax_xy.set_ylabel("radial_profile_data")
            ax_xy.set_yscale("log" if self.scale_combo_xy.currentText() == "Log" else "linear")
            ax_xy.legend()
            ax_xy.set_xlim(min(q_range), max(q_range))
            try:
                h_value = float(self.hline_line_edit.text())
                ax_xy.axhline(y=h_value, color='red', linestyle='--')
            except ValueError:
                pass
            for v in self.vertical_lines_values:
                ax_xy.axvline(x=v, color='blue', linestyle='--')
        self.canvas_xy.draw_idle()

        # --- Plot 3: AgBH Plot using sns.heatmap ---
        self.fig_agbh.clf()
        ax_agbh = self.fig_agbh.add_subplot(111)
        calib_name = row.get('calib_name')
        if self.calibration_df is not None and calib_name is not None:
            calib_rows = self.calibration_df[(self.calibration_df['calib_name'] == calib_name) &
                                             (self.calibration_df['cal_name'].str.contains('AgBh'))]
            if not calib_rows.empty:
                calib_row = calib_rows.iloc[0]
                agbh_data = calib_row.get('measurement_data')
                if agbh_data is not None:
                    # Use seaborn's heatmap with robust scaling and square cells.
                    sns.heatmap(agbh_data, ax=ax_agbh, robust=True, square=True, cbar=True)
                else:
                    ax_agbh.text(0.5, 0.5, "No AgBH data available",
                                 ha="center", va="center", transform=ax_agbh.transAxes)
            else:
                ax_agbh.text(0.5, 0.5, "No matching calibration data",
                             ha="center", va="center", transform=ax_agbh.transAxes)
        else:
            ax_agbh.text(0.5, 0.5, "No AgBH data available",
                         ha="center", va="center", transform=ax_agbh.transAxes)
        self.canvas_agbh.draw_idle()

        # --- Plot 4: Cake Representation ---
        self.fig_cake.clf()
        ax_cake = self.fig_cake.add_subplot(111)
        try:
            cake_data = calib_row.get('radial_profile_data')
            azimuthal_positions = calib_row.get('azimuthal_positions')
            q_range = calib_row.get('q_range')
        except Exception:
            cake_data = None
        if cake_data is not None and azimuthal_positions is not None and q_range is not None:
            extent = [min(q_range), max(q_range), min(azimuthal_positions), max(azimuthal_positions)]
            im = ax_cake.imshow(cake_data, extent=extent, aspect='auto', origin='lower',
                                interpolation='none', norm=LogNorm())
            ax_cake.set_xlabel("q_range (nm-1)")
            ax_cake.set_ylabel("Azimuthal Positions")
        else:
            ax_cake.text(0.5, 0.5, "Cake Representation data not available",
                         ha="center", va="center", transform=ax_cake.transAxes)
        self.canvas_cake.draw_idle()

    def update_horizontal_line(self):
        """Updates the XY plot when the horizontal line value is changed."""
        self.display_measurement(self.current_index)

    def add_vertical_line(self):
        """Adds a vertical line based on the input value and updates the XY plot."""
        try:
            v_value = float(self.vertical_line_line_edit.text())
        except ValueError:
            return
        self.vertical_lines_values.append(v_value)
        self.display_measurement(self.current_index)

    def remove_vertical_line(self):
        """Removes the last added vertical line and updates the XY plot."""
        if self.vertical_lines_values:
            self.vertical_lines_values.pop()
            self.display_measurement(self.current_index)
