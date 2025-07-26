from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QDialog, QTableWidget, QTableWidgetItem, QHeaderView

class MeasurementHistoryWidget(QWidget):
    def __init__(self, masks, ponis, parent=None):
        super().__init__(parent)
        self.measurements = []  # List[dict] with {timestamp, results}
        self.masks = masks
        self.ponis = ponis
        self.parent_window = parent
        self.layout = QVBoxLayout(self)
        self.summary_btn = QPushButton("No measurements")
        self.summary_btn.clicked.connect(self.show_history_dialog)
        self.layout.addWidget(self.summary_btn)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.update_summary()

    def add_measurement(self, results, timestamp):
        # results: {alias: {'filename':..., 'goodness':...}}
        self.measurements.append({
            "timestamp": timestamp,
            "results": results
        })
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
        if not self.measurements:
            return
        all_aliases = sorted({alias for m in self.measurements for alias in m["results"].keys()})
        ncols = 1 + 2 * len(all_aliases)
        table = QTableWidget()
        table.setRowCount(len(self.measurements))
        table.setColumnCount(ncols)
        headers = ["Timestamp"]
        for alias in all_aliases:
            headers.append(f"{alias} File")
            headers.append(f"{alias} Goodness")
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, m in enumerate(self.measurements):
            table.setItem(i, 0, QTableWidgetItem(str(m["timestamp"])))
            for j, alias in enumerate(all_aliases):
                res = m["results"].get(alias, {})
                filename = res.get("filename", "")
                goodness = res.get("goodness")
                table.setItem(i, 1 + 2 * j, QTableWidgetItem(str(filename)))
                good_str = f"{goodness:.1f}%" if goodness is not None else "-"
                table.setItem(i, 2 + 2 * j, QTableWidgetItem(good_str))
        layout.addWidget(table)
        dlg.setLayout(layout)
        dlg.resize(800, 400)

        def cell_double_clicked(row, col):
            # Only respond to file columns (ignore timestamp/goodness)
            if col == 0 or (col - 1) % 2 != 0:
                return
            alias_idx = (col - 1) // 2
            try:
                alias = all_aliases[alias_idx]
            except IndexError:
                print(f"[DoubleClick] Alias index out of range for col={col}")
                return
            res = self.measurements[row]["results"].get(alias, {})
            filename = res.get("filename")
            if filename:
                try:
                    from hardware.Ulster.gui.technical.capture import show_measurement_window
                    show_measurement_window(
                        filename,
                        self.masks.get(alias),
                        self.ponis.get(alias),
                        self.parent_window
                    )
                except Exception as e:
                    print(f"Error opening measurement window: {e}")

        table.cellDoubleClicked.connect(cell_double_clicked)
        dlg.exec_()
