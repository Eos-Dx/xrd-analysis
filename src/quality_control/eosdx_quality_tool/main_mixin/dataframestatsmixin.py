from PyQt5.QtWidgets import QDockWidget, QTextEdit, QWidget, QHBoxLayout, QListWidget
from PyQt5.QtCore import Qt


class DataFrameStatsMixin:
    def init_df_stats_zone(self):
        """
        Initializes a dockable widget to display DataFrame statistics and a list of measurements.
        The widget contains:
          - A read-only QTextEdit for DataFrame statistics.
          - A QListWidget that displays the "meas_name" column of all measurements in self.transformed_df.
        """
        self.df_stats_dock = QDockWidget("DataFrame Statistics", self)

        # Create a container widget with a horizontal layout.
        container = QWidget()
        layout = QHBoxLayout(container)

        # QTextEdit for DataFrame statistics.
        self.df_stats_text = QTextEdit()
        self.df_stats_text.setReadOnly(True)
        layout.addWidget(self.df_stats_text)

        # QListWidget for displaying measurements (meas_name).
        self.measurements_list_widget = QListWidget()
        layout.addWidget(self.measurements_list_widget)

        # Connect item click to open the measurement.
        self.measurements_list_widget.itemClicked.connect(self.open_measurement_from_list)

        self.df_stats_dock.setWidget(container)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.df_stats_dock)

    def update_df_stats(self, text):
        """
        Updates the DataFrame statistics zone with the provided text.
        """
        self.df_stats_text.setText(text)

    def update_measurements_list(self):
        """
        Updates the measurements list widget using the "meas_name" column from self.transformed_df.
        """
        self.measurements_list_widget.clear()
        if self.transformed_df is not None:
            print("DataFrame loaded, columns:", self.transformed_df.columns)
            if 'meas_name' in self.transformed_df.columns:
                for measurement_name in self.transformed_df['meas_name']:
                    self.measurements_list_widget.addItem(str(measurement_name))
            else:
                print("Column 'meas_name' not found in DataFrame.")
        else:
            print("self.transformed_df is None.")

    def open_measurement_from_list(self, item):
        """
        When a measurement (identified by 'meas_name') in the list is clicked,
        find its corresponding row in self.transformed_df and redraw the VisualizationMixin zone.
        """
        meas_name = item.text()
        if self.transformed_df is not None and 'meas_name' in self.transformed_df.columns:
            indices = self.transformed_df.index[self.transformed_df['meas_name'] == meas_name].tolist()
            if indices:
                # Call the VisualizationMixin's display_measurement method with the first matching index.
                self.display_measurement(indices[0])
            else:
                print("Measurement with meas_name '{}' not found.".format(meas_name))
        else:
            print("self.transformed_df is None or does not contain 'meas_name'.")
