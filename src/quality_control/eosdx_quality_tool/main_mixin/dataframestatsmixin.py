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
        container = QWidget()
        layout = QHBoxLayout(container)
        self.df_stats_text = QTextEdit()
        self.df_stats_text.setReadOnly(True)
        layout.addWidget(self.df_stats_text)
        self.measurements_list_widget = QListWidget()
        layout.addWidget(self.measurements_list_widget)
        self.measurements_list_widget.itemClicked.connect(self.open_measurement_from_list)
        self.df_stats_dock.setWidget(container)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.df_stats_dock)

    def update_df_stats(self, text):
        """Updates the DataFrame statistics zone."""
        self.df_stats_text.setText(text)

    def update_measurements_list(self):
        """Updates the measurements list widget using the 'meas_name' column."""
        self.measurements_list_widget.clear()
        if self.transformed_df is not None:
            for measurement_name in self.transformed_df['meas_name']:
                self.measurements_list_widget.addItem(str(measurement_name))
        else:
            print("self.transformed_df is None.")

    def open_measurement_from_list(self, item):
        """
        When a measurement is double-clicked, find its row in self.transformed_df,
        display it, update the Exclude Zone's status label, and if the measurement is
        present in the labels file (_labels.txt), select it in the Labels Zone list.
        """
        meas_name = item.text()
        if self.transformed_df is not None and 'meas_name' in self.transformed_df.columns:
            indices = self.transformed_df.index[self.transformed_df['meas_name'] == meas_name].tolist()
            if indices:
                self.display_measurement(indices[0])
                # Update the status label in the Exclude Zone if available.
                if hasattr(self, 'update_status_label'):
                    self.update_status_label()
                # If the Labels Zone is present, select the corresponding label.
                if hasattr(self, 'excluded_list_widget'):
                    for i in range(self.excluded_list_widget.count()):
                        label_item = self.excluded_list_widget.item(i)
                        text = label_item.text().strip()
                        # Remove the prefix if it exists.
                        for prefix in ("Excluded: ", "Included: ", "Suspicious: "):
                            if text.startswith(prefix):
                                text = text[len(prefix):]
                                break
                        # Now, assume the first token (before the first colon) is the measurement name.
                        label_meas = text.split(":", 1)[0].strip()
                        if label_meas == meas_name:
                            self.excluded_list_widget.setCurrentItem(label_item)
                            break
            else:
                print(f"Measurement '{meas_name}' not found.")
        else:
            print("self.transformed_df is None or missing 'meas_name'.")

