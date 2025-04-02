from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from quality_control.eosdx_quality_tool.config import REASON
from pathlib import Path


class ExcludeIncludeMixin:
    def init_exclude_zone(self):
        """
        Initializes the Exclude Zone as a dockable widget which includes:
          - Navigation buttons (Previous, Suspicious, Exclude, Next, Include).
          - A text edit for entering a reason.
          - Buttons to add or remove a reason.
          - A list widget showing saved reasons.
        Existing reasons from the file defined in REASON are loaded into the list.
        """
        # Create a dock widget for the exclude controls.
        self.exclude_dock = QDockWidget("Exclude Zone", self)
        self.exclude_dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        # Main widget and layout for the dock.
        self.exclude_widget = QWidget()
        exclude_layout = QVBoxLayout(self.exclude_widget)

        # Navigation buttons in order: Previous, Suspicious, Exclude, Next, Include.
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.suspicious_button = QPushButton("Suspicious")
        self.exclude_button = QPushButton("Exclude")
        self.next_button = QPushButton("Next")
        self.include_button = QPushButton("Include")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.suspicious_button)
        nav_layout.addWidget(self.exclude_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.include_button)
        exclude_layout.addLayout(nav_layout)

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

    def add_reason_to_list(self):
        """
        Adds the text from the reason text edit to the list widget if not empty.
        It also appends the reason to the REASON file if it is not already present.
        """
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
        """
        Removes the selected reason from the list widget and updates the REASON file.
        """
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
        """
        Populates the reason text edit with the text of the clicked reason.
        """
        self.reason_textedit.setPlainText(item.text())

    def show_previous_measurement(self):
        new_index = self.current_index - 1
        if new_index >= 0:
            self.display_measurement(new_index)

    def show_next_measurement(self):
        new_index = self.current_index + 1
        if self.transformed_df is not None and new_index < len(self.transformed_df):
            self.display_measurement(new_index)

    def suspicious_current_measurement(self):
        """
        Writes a suspicious record into the labels file and colors the measurement yellow.
        """
        self.add_reason_to_list()
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        meas_name = row.get('meas_name', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        reason = self.reason_textedit.toPlainText().strip()
        if reason:
            self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
            line = f"Suspicious: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
            with open(self.labels_filename, "a") as f:
                f.write(line)
            if hasattr(self, 'load_excluded_included_files'):
                self.load_excluded_included_files()
            self.mark_measurement_in_list(meas_name, "yellow")

    def exclude_current_measurement(self):
        """
        Writes an excluded record into the labels file and colors the measurement red.
        """
        self.add_reason_to_list()
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        meas_name = row.get('meas_name', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        reason = self.reason_textedit.toPlainText().strip()
        if reason:
            self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
            line = f"Excluded: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
            with open(self.labels_filename, "a") as f:
                f.write(line)
            if hasattr(self, 'load_excluded_included_files'):
                self.load_excluded_included_files()
            self.mark_measurement_in_list(meas_name, "red")

    def include_current_measurement(self):
        """
        Writes an included record into the labels file and colors the measurement green.
        """
        self.add_reason_to_list()
        if self.transformed_df is None:
            return
        row = self.transformed_df.iloc[self.current_index]
        measurement_group_id = row.get('measurementsGroupId', 'N/A')
        patient_db_id = row.get('patientDBId', 'N/A')
        meas_name = row.get('meas_name', 'N/A')
        specimen_db_id = row.get('specimenDBId', 'N/A')
        reason = self.reason_textedit.toPlainText().strip()
        if reason:
            self.labels_filename = self.file_path.parent / f"{self.file_path.stem}_labels.txt"
            line = f"Included: {meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
            with open(self.labels_filename, "a") as f:
                f.write(line)
            if hasattr(self, 'load_excluded_included_files'):
                self.load_excluded_included_files()
            self.mark_measurement_in_list(meas_name, "green")

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
