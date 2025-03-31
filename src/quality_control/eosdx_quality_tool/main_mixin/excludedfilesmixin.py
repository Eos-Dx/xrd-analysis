from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel
from PyQt5.QtCore import Qt

class ExcludedFilesMixin:
    def init_excluded_files_zone(self):
        """
        Initializes the Excluded Files Zone as a dockable widget.
        If an exclusion file exists (derived from self.file_path), its contents are loaded
        into a list. Each list entry is clickable to open the corresponding measurement,
        and entries can be deleted using the provided delete button.
        """
        # Create a dock widget for the excluded files.
        self.excluded_files_dock = QDockWidget("Excluded Files Zone", self)
        self.excluded_files_dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        # Create the main widget for the dock.
        self.excluded_files_widget = QWidget()
        layout = QVBoxLayout(self.excluded_files_widget)

        # Title label.
        title_label = QLabel("Excluded Measurements:")
        layout.addWidget(title_label)

        # List widget to show exclusion file entries.
        self.excluded_list_widget = QListWidget()
        layout.addWidget(self.excluded_list_widget)

        # Delete button to remove selected entries.
        self.delete_entry_button = QPushButton("Delete Selected Entry")
        layout.addWidget(self.delete_entry_button)
        self.delete_entry_button.clicked.connect(self.delete_selected_entry)

        # Set the dock widget's widget and add it to the main window.
        self.excluded_files_dock.setWidget(self.excluded_files_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.excluded_files_dock)

        # Connect list item click to opening the measurement.
        self.excluded_list_widget.itemClicked.connect(self.open_excluded_measurement)

        # Load the file (if it exists) into the list.
        self.load_excluded_file()

    def load_excluded_file(self):
        """
        Loads the exclusion file (derived from self.file_path) and populates the list widget.
        Each line should be in the format "measurement_id: reason".
        If the file does not exist, it is created as an empty file.
        """
        self.excluded_list_widget.clear()
        try:
            # Assuming self.file_path is a pathlib.Path object.
            filename = self.file_path.parent / f"{self.file_path.stem}_exclusion.txt"
        except Exception:
            return

        if not filename.exists():
            # Create an empty file so that future writes work as expected.
            filename.touch()
            return

        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                self.excluded_list_widget.addItem(line)

    def update_excluded_file(self):
        """
        Updates the exclusion file with the current list widget contents.
        """
        try:
            filename = self.excluded_filename
        except Exception:
            return

        items = []
        for i in range(self.excluded_list_widget.count()):
            items.append(self.excluded_list_widget.item(i).text())
        with open(filename, "w") as f:
            for line in items:
                f.write(line + "\n")

    def delete_selected_entry(self):
        """
        Deletes the selected list entry and updates the exclusion file.
        """
        selected_items = self.excluded_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            row = self.excluded_list_widget.row(item)
            self.excluded_list_widget.takeItem(row)
        self.update_excluded_file()

    def open_excluded_measurement(self, item):
        """
        Parses the measurement_id from the selected list entry (assumed format: "measurement_id: reason")
        and opens the corresponding measurement by calling display_measurement.
        """
        text = item.text()
        if ':' in text:
            meas_name = text.split(":", 1)[0].strip()
            if self.transformed_df is not None:
                # Look for a row with a matching 'id' value.
                indices = self.transformed_df.index[self.transformed_df['meas_name'] == meas_name].tolist()
                if indices:
                    self.display_measurement(indices[0])
