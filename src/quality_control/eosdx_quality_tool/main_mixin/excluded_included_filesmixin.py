from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from quality_control.eosdx_quality_tool.config import REASON
from pathlib import Path


class ExcludedIncludedFilesMixin:
    def init_excluded_included_files_zone(self):
        """
        Initializes the Labels Zone as a dockable widget.
        This zone shows all records from the labels file.
        Entries are clickable to open the corresponding measurement.
        A delete button allows removal of a record; when a record is deleted,
        the corresponding measurement is reset to gray.
        """
        # Create a dock widget for the labels.
        self.excluded_files_dock = QDockWidget("Labels Zone", self)
        self.excluded_files_dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        # Main widget and layout.
        self.excluded_files_widget = QWidget()
        layout = QVBoxLayout(self.excluded_files_widget)

        # Title label.
        title_label = QLabel("Measurements Labels (Excluded, Included, Suspicious):")
        layout.addWidget(title_label)

        # List widget to show labels.
        self.excluded_list_widget = QListWidget()
        layout.addWidget(self.excluded_list_widget)

        # Delete button to remove selected entries.
        self.delete_entry_button = QPushButton("Delete Selected Entry")
        layout.addWidget(self.delete_entry_button)
        self.delete_entry_button.clicked.connect(self.delete_selected_entry)

        self.excluded_files_dock.setWidget(self.excluded_files_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.excluded_files_dock)

        # Connect item click to open the measurement.
        self.excluded_list_widget.itemClicked.connect(self.open_excluded_included_measurement)

        # Load labels from the file and apply them.
        self.load_excluded_included_files()
        self.apply_labels_from_file()

    def load_excluded_included_files(self):
        """Loads the labels file and populates the list widget."""
        self.excluded_list_widget.clear()
        try:
            self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        except Exception:
            return

        if self.labels_filename.exists():
            with open(self.labels_filename, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    self.excluded_list_widget.addItem(line)

    def apply_labels_from_file(self):
        """
        Reads the labels file and applies the corresponding colors to the measurement list items.
        Each record is expected to start with one of "Excluded: ", "Included: ", or "Suspicious: ".
        """
        try:
            labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        except Exception:
            return

        if labels_filename.exists():
            with open(labels_filename, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Excluded: "):
                    color = "red"
                    content = line[len("Excluded: "):]
                elif line.startswith("Included: "):
                    color = "green"
                    content = line[len("Included: "):]
                elif line.startswith("Suspicious: "):
                    color = "yellow"
                    content = line[len("Suspicious: "):]
                else:
                    continue
                parts = content.split(":", 1)
                if parts:
                    meas_name = parts[0].strip()
                    self.mark_measurement_in_list(meas_name, color)

    def remove_line_from_file(self, filename, line_to_remove):
        """Removes a specific line from the given file."""
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
            with open(filename, "w") as f:
                for line in lines:
                    if line.strip() != line_to_remove:
                        f.write(line)
        except Exception:
            pass

    def delete_selected_entry(self):
        """
        Deletes the selected record from the list and from the labels file.
        Also resets the corresponding measurement’s color to gray.
        """
        selected_items = self.excluded_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            text = item.text()
            if text.startswith("Excluded: "):
                actual_line = text[len("Excluded: "):]
            elif text.startswith("Included: "):
                actual_line = text[len("Included: "):]
            elif text.startswith("Suspicious: "):
                actual_line = text[len("Suspicious: "):]
            else:
                actual_line = text
            parts = actual_line.split(":", 1)
            if parts:
                meas_name = parts[0].strip()
                self.mark_measurement_in_list(meas_name, "gray")
            file_path = self.labels_filename
            # Remove only the part after the first colon and space.
            self.remove_line_from_file(file_path, text.split(": ", 1)[1])
            row = self.excluded_list_widget.row(item)
            self.excluded_list_widget.takeItem(row)

    def open_excluded_included_measurement(self, item):
        """
        Parses the measurement name from the selected record and opens that measurement.
        After opening the measurement, updates the Exclude Zone's status label.
        """
        text = item.text()
        if text.startswith("Excluded: "):
            text = text[len("Excluded: "):]
        elif text.startswith("Included: "):
            text = text[len("Included: "):]
        elif text.startswith("Suspicious: "):
            text = text[len("Suspicious: "):]
        if ':' in text:
            meas_name = text.split(":", 1)[0].strip()
            if self.transformed_df is not None:
                indices = self.transformed_df.index[self.transformed_df['meas_name'] == meas_name].tolist()
                if indices:
                    self.display_measurement(indices[0])
                    # Update the Exclude Zone's status label.
                    if hasattr(self, 'update_status_label'):
                        self.update_status_label()


    def mark_measurement_in_list(self, meas_name, color):
        """
        Iterates over the measurements list widget items and sets the background color
        for the item that matches the given meas_name.
        """
        for i in range(self.measurements_list_widget.count()):
            item = self.measurements_list_widget.item(i)
            if item.text() == meas_name:
                item.setBackground(QBrush(QColor(color)))
                break
