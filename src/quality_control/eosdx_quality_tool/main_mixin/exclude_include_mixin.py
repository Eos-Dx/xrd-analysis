from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from quality_control.eosdx_quality_tool.config import REASON
from pathlib import Path


class ExcludeIncludeMixin:
    def init_exclude_zone(self):
        """
        Initializes the Exclude Zone as a dockable widget which includes:
          - Navigation buttons (Previous, Suspicious, Exclude, Next, Include, Ethalon).
          - A text edit for entering a reason.
          - Buttons to add or remove a reason.
          - A list widget showing saved reasons.
        Existing reasons from the file defined in REASON are loaded into the list.
        Also, a status label is displayed showing the current measurement’s label.
        """
        # Create a dock widget for the exclude controls.
        self.exclude_dock = QDockWidget("Exclude Zone", self)
        self.exclude_dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        # Main widget and layout for the dock.
        self.exclude_widget = QWidget()
        exclude_layout = QVBoxLayout(self.exclude_widget)

        # Navigation buttons: Previous, Suspicious, Exclude, Next, Include, Ethalon.
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.suspicious_button = QPushButton("Suspicious")
        self.exclude_button = QPushButton("Exclude")
        self.next_button = QPushButton("Next")
        self.include_button = QPushButton("Include")
        self.ethalon_button = QPushButton("Ethalon")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.suspicious_button)
        nav_layout.addWidget(self.exclude_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.include_button)
        nav_layout.addWidget(self.ethalon_button)
        exclude_layout.addLayout(nav_layout)

        # Status label to show current measurement status.
        self.status_label = QLabel("Status: None")
        exclude_layout.addWidget(self.status_label)

        # Reason entry.
        reason_label = QLabel("Enter Reason:")
        exclude_layout.addWidget(reason_label)
        self.reason_textedit = QTextEdit()
        self.reason_textedit.setPlaceholderText("Type reason here")
        exclude_layout.addWidget(self.reason_textedit)

        # Add Reason button.
        self.add_reason_button = QPushButton("Add Reason")
        exclude_layout.addWidget(self.add_reason_button)
        self.add_reason_button.clicked.connect(self.add_reason_to_list)

        # Remove Reason button.
        self.remove_reason_button = QPushButton("Remove Reason")
        exclude_layout.addWidget(self.remove_reason_button)
        self.remove_reason_button.clicked.connect(self.remove_reason_from_list)

        # List widget for saved reasons.
        list_label = QLabel("Saved Reasons:")
        exclude_layout.addWidget(list_label)
        self.reason_list_widget = QListWidget()
        exclude_layout.addWidget(self.reason_list_widget)
        self.reason_list_widget.itemClicked.connect(self.populate_reason_textedit)

        # Load existing reasons from the REASON file.
        try:
            with open(REASON, "r") as f:
                file_reasons = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            file_reasons = []
        for r in file_reasons:
            self.reason_list_widget.addItem(r)

        self.exclude_dock.setWidget(self.exclude_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.exclude_dock)

        # Connect navigation and action buttons.
        self.prev_button.clicked.connect(self.show_previous_measurement)
        self.suspicious_button.clicked.connect(self.suspicious_current_measurement)
        self.exclude_button.clicked.connect(self.exclude_current_measurement)
        self.next_button.clicked.connect(self.show_next_measurement)
        self.include_button.clicked.connect(self.include_current_measurement)
        self.ethalon_button.clicked.connect(self.ethalon_current_measurement)

    def add_reason_to_list(self):
        """(Unchanged) Adds a reason to the list and file."""
        reason = self.reason_textedit.toPlainText().strip()
        if reason:
            try:
                with open(REASON, "r") as f:
                    file_reasons = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                file_reasons = []
            widget_reasons = [self.reason_list_widget.item(i).text() for i in range(self.reason_list_widget.count())]
            existing_reasons = set(file_reasons + widget_reasons)
            if reason not in existing_reasons:
                self.reason_list_widget.addItem(reason)
                with open(REASON, "a") as f:
                    f.write(reason + "\n")

    def remove_reason_from_list(self):
        """(Unchanged) Removes a reason from the list and file."""
        selected_items = self.reason_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            reason_text = item.text()
            row = self.reason_list_widget.row(item)
            self.reason_list_widget.takeItem(row)
            try:
                with open(REASON, "r") as f:
                    lines = f.readlines()
                with open(REASON, "w") as f:
                    for line in lines:
                        if line.strip() != reason_text:
                            f.write(line)
            except FileNotFoundError:
                pass

    def populate_reason_textedit(self, item):
        """(Unchanged) Populates the reason text edit."""
        self.reason_textedit.setPlainText(item.text())

    def show_previous_measurement(self):
        new_index = self.current_index - 1
        if new_index >= 0:
            self.display_measurement(new_index)
            self.update_status_label()

    def show_next_measurement(self):
        new_index = self.current_index + 1
        if self.transformed_df is not None and new_index < len(self.transformed_df):
            self.display_measurement(new_index)
            self.update_status_label()

    def remove_existing_label_for_measurement(self, meas_name):
        """
        Removes any existing record for the given measurement name from the labels file.
        """
        self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        if not self.labels_filename.exists():
            return
        with open(self.labels_filename, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Expecting format: "Status: meas_name: ..." (e.g., "Excluded: my_measurement: ...")
            parts = line.split(":")
            if len(parts) >= 2:
                existing_meas = parts[1].strip()
                if existing_meas == meas_name:
                    continue  # Skip this record.
            new_lines.append(line)
        with open(self.labels_filename, "w") as f:
            for line in new_lines:
                f.write(line + "\n")

    def update_status_label(self):
        """
        Checks the labels file for the current measurement and updates the status label.
        """
        status = "None"
        self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        if self.labels_filename.exists():
            with open(self.labels_filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(":")
                    if len(parts) >= 2:
                        # parts[0] is the status ("Excluded", "Included", "Suspicious", "Ethalon")
                        # parts[1] is the measurement name
                        if parts[1].strip() == self.transformed_df.iloc[self.current_index].get('meas_name', 'N/A'):
                            status = parts[0].strip()
                            break
        self.status_label.setText(f"Status: {status}")

    def suspicious_current_measurement(self):
        """
        For a suspicious measurement, a non-empty reason is required.
        Removes any existing label, then writes a suspicious record and colors the measurement yellow.
        """
        reason = self.reason_textedit.toPlainText().strip()
        if not reason:
            self.status_label.setText("Error: Reason required for suspicious measurement.")
            return
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        meas_name = row.get('meas_name', 'N/A')
        self.remove_existing_label_for_measurement(meas_name)
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        line = f"Suspicious: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
        with open(self.labels_filename, "a") as f:
            f.write(line)
        if hasattr(self, 'load_excluded_included_files'):
            self.load_excluded_included_files()
        self.mark_measurement_in_list(meas_name, "yellow")
        self.update_status_label()

    def exclude_current_measurement(self):
        """
        For an excluded measurement, a non-empty reason is required.
        Removes any existing label, then writes an excluded record and colors the measurement red.
        """
        reason = self.reason_textedit.toPlainText().strip()
        if not reason:
            self.status_label.setText("Error: Reason required for excluded measurement.")
            return
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        meas_name = row.get('meas_name', 'N/A')
        self.remove_existing_label_for_measurement(meas_name)
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        line = f"Excluded: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
        with open(self.labels_filename, "a") as f:
            f.write(line)
        if hasattr(self, 'load_excluded_included_files'):
            self.load_excluded_included_files()
        self.mark_measurement_in_list(meas_name, "red")
        self.update_status_label()

    def include_current_measurement(self):
        """
        For an included measurement, the reason is optional.
        Removes any existing label, then writes an included record and colors the measurement green.
        """
        # For include, if no reason is provided, default to "No reason".
        reason = self.reason_textedit.toPlainText().strip() or "No reason"
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        meas_name = row.get('meas_name', 'N/A')
        self.remove_existing_label_for_measurement(meas_name)
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        line = f"Included: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
        with open(self.labels_filename, "a") as f:
            f.write(line)
        if hasattr(self, 'load_excluded_included_files'):
            self.load_excluded_included_files()
        self.mark_measurement_in_list(meas_name, "green")
        self.update_status_label()

    def ethalon_current_measurement(self):
        """
        For an ethalon measurement, a non-empty reason is required.
        Removes any existing label, then writes an ethalon record and colors the measurement deep gold.
        The labels file is marked with 'Ethalon:'.
        """
        reason = self.reason_textedit.toPlainText().strip()
        if not reason:
            self.status_label.setText("Error: Reason required for ethalon measurement.")
            return
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        meas_name = row.get('meas_name', 'N/A')
        self.remove_existing_label_for_measurement(meas_name)
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
        line = f"Ethalon: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
        with open(self.labels_filename, "a") as f:
            f.write(line)
        if hasattr(self, 'load_excluded_included_files'):
            self.load_excluded_included_files()
        # Mark the measurement with a deep gold color.
        self.mark_measurement_in_list(meas_name, "gold")
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
