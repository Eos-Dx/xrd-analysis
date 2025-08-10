import logging

from PyQt5.QtWidgets import (
    QDialog,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MeasurementHistoryWidget(QWidget):
    def __init__(self, masks, ponis, parent=None, point_id=None):
        super().__init__(parent)
        self.measurements = []
        self.masks = masks if isinstance(masks, dict) else {}
        self.ponis = ponis if isinstance(ponis, dict) else {}
        self.point_id = point_id  # Store the ID
        self.parent_window = parent
        self.x_mm = None
        self.y_mm = None
        self.layout = QVBoxLayout(self)
        try:
            # Set window title to include point_id and, if available, X:Y (mm)
            self._update_title_with_coordinates()
            self.summary_btn = QPushButton("No measurements")
            self.summary_btn.clicked.connect(self.show_history_dialog)
            self.layout.addWidget(self.summary_btn)
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(self.layout)
            self.update_summary()
        except Exception as e:
            logging.getLogger(__name__).exception(
                "Error initializing MeasurementHistoryWidget: %s", e
            )

    def add_measurement(self, results, timestamp):
        try:
            # results: {alias: {'filename':..., 'goodness':...}}
            self.measurements.append({"timestamp": timestamp, "results": results or {}})
            self.update_summary()
        except Exception as e:
            logging.getLogger(__name__).exception("Error adding measurement: %s", e)

    def set_mm_coordinates(self, x_mm: float, y_mm: float):
        """Optionally set coordinates (in mm) and refresh the title."""
        try:
            self.x_mm = x_mm
            self.y_mm = y_mm
            self._update_title_with_coordinates()
        except Exception:
            pass

    def _update_title_with_coordinates(self):
        """Set window title to include point_id and, if known, X:Y in mm.
        Attempts to read coordinates from parent.pointsTable if not already set.
        """
        try:
            title_base = "Measurement History"
            if self.point_id is not None:
                title_base = f"Measurement History: Point #{self.point_id}"
            x_mm = self.x_mm
            y_mm = self.y_mm
            # Try to derive coordinates from parent table if not set yet
            if (
                (x_mm is None or y_mm is None)
                and getattr(self.parent_window, "pointsTable", None) is not None
                and self.point_id is not None
            ):
                table = self.parent_window.pointsTable
                try:
                    from PyQt5.QtWidgets import QTableWidgetItem  # noqa: F401

                    rows = table.rowCount()
                    for r in range(rows):
                        it = table.item(r, 0)
                        if (
                            it is not None
                            and it.text().strip().isdigit()
                            and int(it.text()) == int(self.point_id)
                        ):
                            itx = table.item(r, 3)
                            ity = table.item(r, 4)
                            if itx is not None and ity is not None:
                                try:
                                    x_mm = (
                                        float(itx.text())
                                        if itx.text() not in (None, "", "N/A")
                                        else None
                                    )
                                    y_mm = (
                                        float(ity.text())
                                        if ity.text() not in (None, "", "N/A")
                                        else None
                                    )
                                except Exception:
                                    pass
                            break
                except Exception:
                    pass
            if x_mm is not None and y_mm is not None:
                self.setWindowTitle(f"{title_base} {x_mm:.2f}:{y_mm:.2f} mm")
            else:
                self.setWindowTitle(title_base)
        except Exception:
            # Fallback plain title
            if self.point_id is not None:
                self.setWindowTitle(f"Measurement History: Point #{self.point_id}")
            else:
                self.setWindowTitle("Measurement History")

    def update_summary(self):
        try:
            n = len(self.measurements)
            if n == 0:
                self.summary_btn.setText("No measurements")
            else:
                last = self.measurements[-1]
                ts = last.get("timestamp", "-")
                self.summary_btn.setText(f"{n} measurement(s), last: {ts}")
        except Exception as e:
            logging.getLogger(__name__).exception("Error updating summary: %s", e)

    def show_history_dialog(self):
        try:
            # make the dialog a child of the main window (or None), NOT the cell widget
            parent_for_dialog = (
                self.parent_window if getattr(self, "parent_window", None) else None
            )
            dlg = QDialog(parent_for_dialog)
            # Ensure title reflects current ID and coordinates
            self._update_title_with_coordinates()
            dlg.setWindowTitle(self.windowTitle())
            layout = QVBoxLayout(dlg)
            if not self.measurements:
                return
            all_aliases = sorted(
                {
                    alias
                    for m in self.measurements
                    for alias in (m.get("results") or {}).keys()
                }
            )
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
                table.setItem(i, 0, QTableWidgetItem(str(m.get("timestamp", "-"))))
                res_map = m.get("results") or {}
                for j, alias in enumerate(all_aliases):
                    res = res_map.get(alias, {})
                    filename = res.get("filename", "")
                    goodness = res.get("goodness")
                    table.setItem(i, 1 + 2 * j, QTableWidgetItem(str(filename)))
                    good_str = (
                        f"{goodness:.1f}%"
                        if isinstance(goodness, (int, float))
                        else "-"
                    )
                    table.setItem(i, 2 + 2 * j, QTableWidgetItem(good_str))
            layout.addWidget(table)
            dlg.setLayout(layout)
            dlg.resize(800, 400)

            def cell_double_clicked(row, col):
                try:
                    if col == 0 or (col - 1) % 2 != 0:
                        return
                    alias_idx = (col - 1) // 2
                    try:
                        alias = all_aliases[alias_idx]
                    except IndexError:
                        logging.getLogger(__name__).exception(
                            "Alias index out of range for col=%s", col
                        )
                        return
                    res_map = self.measurements[row].get("results") or {}
                    res = res_map.get(alias, {})
                    filename = res.get("filename")
                    if filename:
                        try:
                            from hardware.Ulster.gui.technical.capture import (
                                show_measurement_window,
                            )

                            show_measurement_window(
                                filename,
                                (self.masks or {}).get(alias),
                                (self.ponis or {}).get(alias),
                                self.parent_window,
                            )
                        except Exception as e:
                            logging.getLogger(__name__).exception(
                                "Error opening measurement window: %s", e
                            )
                except Exception as e:
                    logging.getLogger(__name__).exception(
                        "Error handling cell double click: %s", e
                    )

            table.cellDoubleClicked.connect(cell_double_clicked)
            dlg.exec_()
        except Exception as e:
            logging.getLogger(__name__).exception("Error showing history dialog: %s", e)
