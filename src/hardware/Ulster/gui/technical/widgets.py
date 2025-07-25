from PyQt5.QtWidgets import (
    QDialog,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hardware.Ulster.gui.technical.capture import show_measurement_window


class MeasurementHistoryWidget(QWidget):
    def __init__(self, mask=None, poni=None, parent=None):
        super().__init__(parent)
        self.measurements = []
        self.mask = mask
        self.poni = poni
        self.parent_window = parent  # main window, for dialog parent
        self.layout = QVBoxLayout(self)
        self.summary_btn = QPushButton("No measurements")
        self.summary_btn.clicked.connect(self.show_history_dialog)
        self.layout.addWidget(self.summary_btn)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.update_summary()

    def add_measurement(
        self,
        waxs_filename,
        saxs_filename,
        goodness_waxs,
        goodness_saxs,
        timestamp,
    ):
        self.measurements.append(
            {
                "timestamp": timestamp,
                "WAXS_filename": waxs_filename,
                "SAXS_filename": saxs_filename,
                "WAXS_goodness": goodness_waxs,
                "SAXS_goodness": goodness_saxs,
            }
        )
        self.update_summary()

    def update_summary(self):
        n = len(self.measurements)
        if n == 0:
            self.summary_btn.setText("No measurements")
        else:
            last = self.measurements[-1]
            self.summary_btn.setText(
                f"{n} measurement(s), last: {last['timestamp']}"
            )

    def show_history_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Measurement History")
        layout = QVBoxLayout(dlg)
        table = QTableWidget()
        table.setRowCount(len(self.measurements))
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            [
                "Timestamp",
                "WAXS File",
                "WAXS Goodness",
                "SAXS File",
                "SAXS Goodness",
            ]
        )
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, m in enumerate(self.measurements):
            table.setItem(i, 0, QTableWidgetItem(str(m["timestamp"])))
            waxs_item = QTableWidgetItem(m["WAXS_filename"])
            saxs_item = QTableWidgetItem(m["SAXS_filename"])
            table.setItem(i, 1, waxs_item)
            table.setItem(i, 2, QTableWidgetItem(f"{m['WAXS_goodness']:.1f}%"))
            table.setItem(i, 3, saxs_item)
            table.setItem(i, 4, QTableWidgetItem(f"{m['SAXS_goodness']:.1f}%"))
        layout.addWidget(table)
        dlg.setLayout(layout)
        dlg.resize(700, 400)

        # Double-click logic
        def cell_double_clicked(row, col):
            if col == 1:  # WAXS File
                filename = self.measurements[row]["WAXS_filename"]
            elif col == 3:  # SAXS File
                filename = self.measurements[row]["SAXS_filename"]
            else:
                return  # Ignore non-file columns
            # Call the measurement viewer
            show_measurement_window(
                filename,
                self.mask,
                self.poni,
                self.parent_window,  # for dialog parent
            )

        table.cellDoubleClicked.connect(cell_double_clicked)
        dlg.exec_()
