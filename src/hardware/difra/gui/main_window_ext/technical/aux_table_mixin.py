import os
import re
from pathlib import Path


def _tm():
    from hardware.difra.gui.main_window_ext import technical_measurements as tm

    return tm


class TechnicalAuxTableMixin:
    @staticmethod
    def _aux_metadata_role():
        tm = _tm()
        return tm.Qt.UserRole + 1

    @staticmethod
    def _aux_source_info_role():
        tm = _tm()
        return tm.Qt.UserRole + 2

    def _extract_capture_metadata_from_path(self, file_path: str):
        metadata = {}
        if not file_path:
            return metadata

        stem = Path(str(file_path)).stem

        integration_match = re.search(
            r"(?:^|_)(\d+(?:\.\d+)?)s(?:_|$)",
            stem,
            flags=re.IGNORECASE,
        )
        if integration_match:
            try:
                metadata["integration_time_ms"] = float(integration_match.group(1)) * 1000.0
            except Exception:
                pass

        frames_match = re.search(r"(?:^|_)(\d+)frames(?:_|$)", stem, flags=re.IGNORECASE)
        if frames_match:
            try:
                metadata["n_frames"] = int(frames_match.group(1))
            except Exception:
                pass

        thickness_match = re.search(
            r"(?:^|_)(\d+(?:\.\d+)?)mm(?:_|$)",
            stem,
            flags=re.IGNORECASE,
        )
        if thickness_match:
            try:
                metadata["thickness"] = float(thickness_match.group(1))
            except Exception:
                pass

        return metadata

    def _get_aux_row_metadata(
        self,
        row: int,
        fallback_path: str = "",
        include_filename_fallback: bool = True,
    ):
        tm = _tm()
        metadata = {}
        file_path = fallback_path

        try:
            file_item = self.auxTable.item(row, self.AUX_COL_FILE)
            if file_item is not None:
                if not file_path:
                    file_path = file_item.data(tm.Qt.UserRole) or ""
                stored = file_item.data(self._aux_metadata_role())
                if isinstance(stored, dict):
                    metadata.update(stored)
        except Exception:
            pass

        if include_filename_fallback:
            parsed = self._extract_capture_metadata_from_path(str(file_path or ""))
            for key, value in parsed.items():
                metadata.setdefault(key, value)

        return metadata

    def _add_aux_item_to_list(
        self,
        alias,
        npy_path,
        *,
        source_kind: str = "file",
        source_container: str = "",
        source_dataset: str = "",
        technical_type: str = None,
        is_primary: bool = False,
        source_row_id: str = "",
        explicit_metadata: dict = None,
    ):
        tm = _tm()

        try:
            if source_kind == "file" and not self._validate_timestamp_before_alias(npy_path):
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
        primary_checkbox.setChecked(bool(is_primary))
        primary_checkbox.setToolTip("Mark as primary measurement (unchecked = supplementary)")
        primary_checkbox.stateChanged.connect(self._on_aux_row_updated)
        checkbox_widget = tm.QWidget()
        checkbox_layout = tm.QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(primary_checkbox)
        checkbox_layout.setAlignment(tm.Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.auxTable.setCellWidget(row, 0, checkbox_widget)

        source_ref = str(npy_path or "")
        if source_kind == "container" and source_container and source_dataset:
            source_ref = f"h5ref://{source_container}#{source_dataset}"

        display_name = Path(str(npy_path or "")).name or source_row_id or "from_container"
        display = f"{alias}: {display_name}"
        file_item = tm.QTableWidgetItem(display)
        file_item.setFlags(tm.Qt.ItemIsSelectable | tm.Qt.ItemIsEnabled)
        file_item.setData(tm.Qt.UserRole, source_ref)
        metadata = self._extract_capture_metadata_from_path(str(npy_path))
        pending_metadata = getattr(self, "_pending_aux_capture_metadata", None)
        if isinstance(pending_metadata, dict):
            for key, value in pending_metadata.items():
                if value is not None:
                    metadata[key] = value
        if isinstance(explicit_metadata, dict):
            for key, value in explicit_metadata.items():
                if value is not None:
                    metadata[key] = value
        if metadata:
            file_item.setData(self._aux_metadata_role(), metadata)
        file_item.setData(
            self._aux_source_info_role(),
            {
                "source_kind": str(source_kind or "file"),
                "source_path": str(npy_path or ""),
                "container_path": str(source_container or ""),
                "dataset_path": str(source_dataset or ""),
                "row_id": str(source_row_id or ""),
            },
        )
        self.auxTable.setItem(row, 1, file_item)

        type_cb = self._make_type_combobox()
        self.auxTable.setCellWidget(row, 2, type_cb)

        alias_cb = self._make_alias_combobox(preselect=alias)
        self.auxTable.setCellWidget(row, 3, alias_cb)
        if hasattr(alias_cb, "currentTextChanged"):
            alias_cb.currentTextChanged.connect(self._on_aux_row_updated)

        if technical_type:
            idx = type_cb.findText(technical_type) if hasattr(type_cb, "findText") else -1
            if idx >= 0:
                type_cb.setCurrentIndex(idx)

        try:
            inferred_type = (
                technical_type
                if technical_type
                else self._infer_type_from_filename(str(npy_path or ""))
            )
            if inferred_type:
                type_cb = self.auxTable.cellWidget(row, 2)
                if type_cb and hasattr(type_cb, "findText"):
                        idx = type_cb.findText(inferred_type)
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
                            self._log_technical_event(
                                f"Added {inferred_type} measurement: {alias} "
                                f"({os.path.basename(str(npy_path or ''))})"
                            )
        except Exception:
            pass

        self._on_aux_row_updated()

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
        tm.QMessageBox.information(
            self,
            "Removed Workflow",
            "Loading technical data directly from files is removed.\n"
            "Use technical container workflow: set distances, measure, and load containers.",
        )
        self._log_technical_event(
            "Load files action is removed; technical workflow is container-first"
        )

    def build_aux_state(self):
        tm = _tm()
        rows = []
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return rows
            for r in range(self.auxTable.rowCount()):
                file_item = self.auxTable.item(r, self.AUX_COL_FILE)
                file_path = file_item.data(tm.Qt.UserRole) if file_item is not None else None
                source_info = (
                    file_item.data(self._aux_source_info_role())
                    if file_item is not None
                    else {}
                )

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
                        "capture_metadata": self._get_aux_row_metadata(r, str(file_path or "")),
                        "source_info": source_info if isinstance(source_info, dict) else {},
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
            self._restoring_aux_table = True
            self.auxTable.setRowCount(0)
            for row in rows or []:
                fpath = row.get("file_path")
                alias = row.get("alias") or self._infer_alias_from_filename(fpath or "")
                source_info = row.get("source_info") if isinstance(row, dict) else {}
                if not isinstance(source_info, dict):
                    source_info = {}
                self._add_aux_item_to_list(
                    alias or "",
                    fpath or "",
                    source_kind=source_info.get("source_kind", "file"),
                    source_container=source_info.get("container_path", ""),
                    source_dataset=source_info.get("dataset_path", ""),
                    technical_type=row.get("type"),
                    is_primary=bool(row.get("is_primary")),
                    source_row_id=source_info.get("row_id", ""),
                    explicit_metadata=row.get("capture_metadata")
                    if isinstance(row.get("capture_metadata"), dict)
                    else None,
                )
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
                    capture_metadata = row.get("capture_metadata")
                    if isinstance(capture_metadata, dict):
                        file_item = self.auxTable.item(rix, self.AUX_COL_FILE)
                        if file_item is not None:
                            file_item.setData(self._aux_metadata_role(), capture_metadata)
                except Exception:
                    pass
            self._restoring_aux_table = False
            self._on_aux_row_updated()
        except Exception as e:
            self._restoring_aux_table = False
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
            self._on_aux_row_updated()
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
        try:
            from PyQt5.QtWidgets import QComboBox as _QtComboBox, QWidget as _QtWidget

            if not isinstance(cb, _QtWidget):
                cb = _QtComboBox()
        except Exception:
            pass
        if hasattr(cb, "addItem"):
            cb.addItem(self.NO_SELECTION_LABEL, None)
            for t in self.TYPE_OPTIONS:
                cb.addItem(t, t)
        if hasattr(cb, "currentTextChanged"):
            cb.currentTextChanged.connect(self._on_type_changed)
        return cb

    def _on_type_changed(self, new_type):
        tm = _tm()
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

        if new_type == self.NO_SELECTION_LABEL:
            self._on_aux_row_updated()
            return

        file_item = self.auxTable.item(trigger_row, 1)
        if not file_item:
            self._on_aux_row_updated()
            return

        file_path = file_item.data(tm.Qt.UserRole)
        if not file_path:
            self._on_aux_row_updated()
            return

        # Auto-sync type across detector aliases for the same captured file only for
        # regular file-backed rows. Container-backed rows use explicit row identity.
        if str(file_path).startswith("h5ref://"):
            self._on_aux_row_updated()
            return

        base_name = Path(file_path).stem
        parts = base_name.split("_")
        if len(parts) < 2:
            self._on_aux_row_updated()
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
        self._on_aux_row_updated()

    def _make_alias_combobox(self, preselect=None):
        tm = _tm()
        cb = tm.QComboBox()
        try:
            from PyQt5.QtWidgets import QComboBox as _QtComboBox, QWidget as _QtWidget

            if not isinstance(cb, _QtWidget):
                cb = _QtComboBox()
        except Exception:
            pass
        if hasattr(cb, "addItem"):
            cb.addItem(self.NO_SELECTION_LABEL, None)
            for alias in self._get_active_detector_aliases():
                cb.addItem(alias, alias)
        if preselect:
            if hasattr(cb, "findText") and hasattr(cb, "setCurrentIndex"):
                idx = cb.findText(preselect)
                if idx >= 0:
                    cb.setCurrentIndex(idx)
        if hasattr(cb, "currentTextChanged"):
            cb.currentTextChanged.connect(self._on_aux_row_updated)
        return cb

    def _on_aux_row_updated(self, *_args):
        """Notify container workflow that table state changed."""
        if getattr(self, "_restoring_aux_table", False):
            return
        sync_fn = getattr(self, "_sync_active_technical_container_from_table", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                pass

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
