import os
import re
from pathlib import Path

import numpy as np


def _tm():
    from hardware.difra.gui.main_window_ext import technical_measurements as tm

    return tm


class TechnicalAuxTableMixin:
    def _add_aux_item_to_list(self, alias, npy_path):
        tm = _tm()

        try:
            if not self._validate_timestamp_before_alias(npy_path):
                tm.QMessageBox.warning(
                    self,
                    "Filename format",
                    "File name should include timestamp before detector alias\n"
                    "Expected pattern like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext",
                )
        except Exception:
            pass

        row = self.auxTable.rowCount()
        self.auxTable.insertRow(row)

        primary_checkbox = tm.QCheckBox()
        primary_checkbox.setChecked(False)
        primary_checkbox.setToolTip("Mark as primary measurement (unchecked = supplementary)")
        checkbox_widget = tm.QWidget()
        checkbox_layout = tm.QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(primary_checkbox)
        checkbox_layout.setAlignment(tm.Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.auxTable.setCellWidget(row, 0, checkbox_widget)

        display = f"{alias}: {Path(npy_path).name}"
        file_item = tm.QTableWidgetItem(display)
        file_item.setFlags(tm.Qt.ItemIsSelectable | tm.Qt.ItemIsEnabled)
        file_item.setData(tm.Qt.UserRole, str(npy_path))
        self.auxTable.setItem(row, 1, file_item)

        type_cb = self._make_type_combobox()
        self.auxTable.setCellWidget(row, 2, type_cb)

        alias_cb = self._make_alias_combobox(preselect=alias)
        self.auxTable.setCellWidget(row, 3, alias_cb)

        try:
            inferred_type = self._infer_type_from_filename(npy_path)
            if inferred_type:
                type_cb = self.auxTable.cellWidget(row, 2)
                if type_cb and hasattr(type_cb, "findText"):
                    idx = type_cb.findText(inferred_type)
                    if idx >= 0:
                        type_cb.setCurrentIndex(idx)
                        self._log_technical_event(
                            f"Added {inferred_type} measurement: {alias} ({os.path.basename(npy_path)})"
                        )
        except Exception:
            pass

    def _file_base(self, typ: str) -> str:
        le = getattr(self, f"{typ.lower()}NameLE")
        txt = le.text().strip().replace(" ", "_")
        return txt or typ.lower()

    def _validate_timestamp_before_alias(self, file_path: str) -> bool:
        base = os.path.basename(file_path)
        name, _ext = os.path.splitext(base)
        toks = name.split("_")
        if len(toks) < 3:
            return False
        stamp_match = re.search(r"\d{8}_\d{6}", name)
        if not stamp_match:
            return False
        alias = toks[-1]
        if not re.fullmatch(r"[A-Za-z0-9]+", alias):
            return False
        return stamp_match.start() < (len(name) - len(alias))

    def _infer_alias_from_filename(self, file_path: str) -> str:
        base = os.path.basename(file_path)
        alias = os.path.splitext(base)[0].split("_")[-1]
        try:
            active_aliases = self._get_active_detector_aliases()
            if alias in active_aliases:
                return alias
        except Exception:
            pass
        return alias

    def _infer_type_from_filename(self, file_path: str) -> str:
        base = os.path.basename(file_path).lower()
        for type_option in self.TYPE_OPTIONS:
            if type_option.lower() in base:
                return type_option
        if "background" in base:
            return "BACKGROUND"
        if "dark" in base:
            return "DARK"
        if "empty" in base:
            return "EMPTY"
        if "agbh" in base:
            return "AGBH"
        return None

    def load_technical_files(self):
        tm = _tm()
        files, _ = tm.QFileDialog.getOpenFileNames(
            self,
            "Load Technical Measurement Files",
            str(self.folderLE.text() or ""),
            "NumPy Arrays (*.npy);;Text Files (*.txt);;All Files (*)",
        )
        if not files:
            return

        self._log_technical_event(f"Loading {len(files)} technical files...")

        for fpath in files:
            path_to_use = fpath
            try:
                if fpath.lower().endswith(".txt"):
                    data = np.loadtxt(fpath)
                    npy_path = os.path.splitext(fpath)[0] + ".npy"
                    np.save(npy_path, data)
                    path_to_use = npy_path
            except Exception as e:
                tm.QMessageBox.warning(
                    self,
                    "Conversion failed",
                    f"Failed to convert TXT to NPY for:\n{fpath}\nError: {e}",
                )
                continue

            if not self._validate_timestamp_before_alias(path_to_use):
                tm.QMessageBox.warning(
                    self,
                    "Filename format",
                    "File name should include timestamp before detector alias\n"
                    "Expected pattern like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext",
                )

            alias = self._infer_alias_from_filename(path_to_use)
            self._add_aux_item_to_list(alias, path_to_use)

            try:
                inferred_type = self._infer_type_from_filename(path_to_use)
                if inferred_type:
                    row_idx = self.auxTable.rowCount() - 1
                    type_cb = self.auxTable.cellWidget(row_idx, 2)
                    if type_cb and hasattr(type_cb, "findText"):
                        idx = type_cb.findText(inferred_type)
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
            except Exception:
                pass

    def build_aux_state(self):
        tm = _tm()
        rows = []
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return rows
            for r in range(self.auxTable.rowCount()):
                file_item = self.auxTable.item(r, self.AUX_COL_FILE)
                file_path = file_item.data(tm.Qt.UserRole) if file_item is not None else None

                is_primary = False
                primary_widget = self.auxTable.cellWidget(r, self.AUX_COL_PRIMARY)
                try:
                    if primary_widget is not None:
                        primary_checkbox = primary_widget.findChild(tm.QCheckBox)
                        if primary_checkbox is not None:
                            is_primary = bool(primary_checkbox.isChecked())
                except Exception:
                    pass

                type_cb = self.auxTable.cellWidget(r, self.AUX_COL_TYPE)
                type_text = None
                try:
                    if type_cb is not None:
                        t = type_cb.currentText()
                        if t and t != self.NO_SELECTION_LABEL:
                            type_text = t
                except Exception:
                    pass

                alias_cb = self.auxTable.cellWidget(r, self.AUX_COL_ALIAS)
                alias_text = None
                try:
                    if alias_cb is not None:
                        a = alias_cb.currentText()
                        if a and a != self.NO_SELECTION_LABEL:
                            alias_text = a
                except Exception:
                    pass
                rows.append(
                    {
                        "file_path": file_path,
                        "type": type_text,
                        "alias": alias_text,
                        "is_primary": is_primary,
                    }
                )
        except Exception as e:
            print(f"Error building aux state: {e}")
        return rows

    def restore_technical_aux_rows(self, rows):
        tm = _tm()
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return
            self.auxTable.setRowCount(0)
            for row in rows or []:
                fpath = row.get("file_path")
                alias = row.get("alias") or self._infer_alias_from_filename(fpath or "")
                self._add_aux_item_to_list(alias or "", fpath or "")
                try:
                    rix = self.auxTable.rowCount() - 1
                    type_cb = self.auxTable.cellWidget(rix, self.AUX_COL_TYPE)
                    if type_cb is not None and row.get("type"):
                        idx = type_cb.findText(row["type"]) if hasattr(type_cb, "findText") else -1
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)

                    if row.get("is_primary"):
                        primary_widget = self.auxTable.cellWidget(rix, self.AUX_COL_PRIMARY)
                        if primary_widget is not None:
                            primary_checkbox = primary_widget.findChild(tm.QCheckBox)
                            if primary_checkbox is not None:
                                primary_checkbox.setChecked(True)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error restoring aux rows: {e}")

    def delete_selected_aux_rows(self):
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return
            sel_model = self.auxTable.selectionModel()
            if not sel_model:
                return
            rows = sorted({ix.row() for ix in sel_model.selectedRows()}, reverse=True)
            if not rows:
                return
            for r in rows:
                try:
                    self.auxTable.removeRow(r)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error deleting selected aux rows: {e}")

    def eventFilter(self, source, event):
        tm = _tm()
        if source is getattr(self, "auxTable", None) and event.type() == tm.QEvent.KeyPress:
            try:
                if event.key() == tm.Qt.Key_Delete:
                    self.delete_selected_aux_rows()
                    return True
            except Exception:
                pass
        return super().eventFilter(source, event)

    def _get_active_detector_aliases(self):
        dev_mode = self.config.get("DEV", False)
        ids = self.config.get("dev_active_detectors", []) if dev_mode else self.config.get("active_detectors", [])
        return [d.get("alias") for d in self.config.get("detectors", []) if d.get("id") in ids]

    def _get_active_detector_ids(self):
        dev_mode = self.config.get("DEV", False)
        return self.config.get("dev_active_detectors", []) if dev_mode else self.config.get("active_detectors", [])

    def _normalize_technical_type(self, typ: str) -> str:
        if typ == "SPECIAL":
            return "WATER"
        return typ

    def _make_type_combobox(self):
        tm = _tm()
        cb = tm.QComboBox()
        cb.addItem(self.NO_SELECTION_LABEL, None)
        for t in self.TYPE_OPTIONS:
            cb.addItem(t, t)
        cb.currentTextChanged.connect(self._on_type_changed)
        return cb

    def _on_type_changed(self, new_type):
        tm = _tm()
        if new_type == self.NO_SELECTION_LABEL:
            return

        sender = self.sender()
        if not isinstance(sender, tm.QComboBox):
            return

        trigger_row = None
        for row in range(self.auxTable.rowCount()):
            if self.auxTable.cellWidget(row, 2) is sender:
                trigger_row = row
                break
        if trigger_row is None:
            return

        file_item = self.auxTable.item(trigger_row, 1)
        if not file_item:
            return

        file_path = file_item.data(tm.Qt.UserRole)
        if not file_path:
            return

        base_name = Path(file_path).stem
        parts = base_name.split("_")
        if len(parts) < 2:
            return
        measurement_name = "_".join(parts[:-1])

        for row in range(self.auxTable.rowCount()):
            if row == trigger_row:
                continue
            row_file_item = self.auxTable.item(row, 1)
            if not row_file_item:
                continue
            row_file_path = row_file_item.data(tm.Qt.UserRole)
            if not row_file_path:
                continue

            row_base_name = Path(row_file_path).stem
            row_parts = row_base_name.split("_")
            if len(row_parts) < 2:
                continue

            row_measurement_name = "_".join(row_parts[:-1])
            if row_measurement_name == measurement_name:
                type_cb = self.auxTable.cellWidget(row, 2)
                if isinstance(type_cb, tm.QComboBox):
                    type_cb.blockSignals(True)
                    type_cb.setCurrentText(new_type)
                    type_cb.blockSignals(False)
                    self._log_technical_event(f"Auto-synced type to {new_type} for row {row + 1}")

    def _make_alias_combobox(self, preselect=None):
        tm = _tm()
        cb = tm.QComboBox()
        cb.addItem(self.NO_SELECTION_LABEL, None)
        for alias in self._get_active_detector_aliases():
            cb.addItem(alias, alias)
        if preselect:
            idx = cb.findText(preselect)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        return cb

    def refresh_aux_table_alias_models(self):
        aliases = self._get_active_detector_aliases()
        for row in range(self.auxTable.rowCount()):
            cb = self.auxTable.cellWidget(row, 3)
            if not hasattr(cb, "addItem"):
                continue
            current = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItem(self.NO_SELECTION_LABEL, None)
            for a in aliases:
                cb.addItem(a, a)
            if current and current in aliases:
                cb.setCurrentText(current)
            cb.blockSignals(False)
