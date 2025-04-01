from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QListWidget
from PyQt5.QtCore import Qt
from quality_control.eosdx_quality_tool.config import REASON

from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit, QListWidget
from PyQt5.QtCore import Qt

class ExcludeMixin:
    def init_exclude_zone(self):
        """
        Initializes the Exclude Zone as a dockable widget which includes:
          - Navigation buttons (Previous, Next, Exclude_measurement).
          - A text edit for entering exclusion reasons.
          - A button to add a new reason to the table.
          - A table (list widget) showing saved reasons.
          Existing reasons from the file defined in REASON are loaded into the list widget.
        """
        # Create a dock widget for the exclude controls.
        self.exclude_dock = QDockWidget("Exclude Zone", self)
        self.exclude_dock.setAllowedAreas(Qt.AllDockWidgetAreas)

        # Create main widget for the dock.
        self.exclude_widget = QWidget()
        exclude_layout = QVBoxLayout(self.exclude_widget)

        # Navigation buttons.
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.exclude_button = QPushButton("Exclude_measurement")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.exclude_button)
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

        # List widget for saved reasons.
        list_label = QLabel("Saved Reasons:")
        exclude_layout.addWidget(list_label)
        self.reason_list_widget = QListWidget()
        exclude_layout.addWidget(self.reason_list_widget)
        self.reason_list_widget.itemClicked.connect(self.populate_reason_textedit)

        # Load existing reasons from file defined in REASON.
        from quality_control.eosdx_quality_tool.config import REASON
        try:
            with open(REASON, "r") as f:
                file_reasons = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            file_reasons = []
        for r in file_reasons:
            self.reason_list_widget.addItem(r)

        self.exclude_dock.setWidget(self.exclude_widget)
        # Add the dock widget to the main window.
        self.addDockWidget(Qt.RightDockWidgetArea, self.exclude_dock)

        # Connect navigation and exclude buttons.
        self.prev_button.clicked.connect(self.show_previous_measurement)
        self.next_button.clicked.connect(self.show_next_measurement)
        self.exclude_button.clicked.connect(self.exclude_current_measurement)

    def add_reason_to_list(self):
        """
        Adds the text from the reason_textedit to the list widget if not empty.
        First, it reads the REASON file to determine existing reasons. If the new reason
        is not already present in either the file or the list widget, it is added and
        appended to the file.
        """
        reason = self.reason_textedit.toPlainText().strip()
        if reason:
            from quality_control.eosdx_quality_tool.config import REASON

            # Read existing reasons from the file (if it exists)
            try:
                with open(REASON, "r") as f:
                    file_reasons = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                file_reasons = []

            # Get reasons currently in the list widget.
            widget_reasons = [self.reason_list_widget.item(i).text() for i in range(self.reason_list_widget.count())]

            # Combine both lists and convert to a set to remove duplicates.
            existing_reasons = set(file_reasons + widget_reasons)

            if reason not in existing_reasons:
                self.reason_list_widget.addItem(reason)
                with open(REASON, "a") as f:
                    f.write(reason + "\n")

    def populate_reason_textedit(self, item):
        """
        When a reason in the list widget is clicked, populate the reason_textedit with that text.
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

    def exclude_current_measurement(self):
        """
        Writes the pair of patient_id and reason to a txt file.
        The file name is derived from the loaded h5 file (self.h5_filename) with the suffix '_exclusion.txt'.
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
            self.excluded_filename = self.file_path.parent / f"{self.file_path.stem}_exclusion.txt"
            line = f"{meas_name}: {measurement_group_id} : {patient_db_id} : {specimen_db_id} : {reason}\n"
            with open(self.excluded_filename, "a") as f:
                f.write(line)
            self.load_excluded_file()
