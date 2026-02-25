# zone_measurements/attenuation_mixin.py

import os
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hardware.difra.gui.main_window_ext.zone_measurements.logic.beam_center_utils import (
    get_beam_center,
)
from hardware.difra.gui.container_api import get_container_version
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


def _place_raw_capture_file(src_raw: str, target_txt: Path, allow_move: bool = True) -> None:
    """Place raw detector output at target path, preferring move over copy."""
    src_path = Path(src_raw)
    target_txt = Path(target_txt)
    target_txt.parent.mkdir(parents=True, exist_ok=True)
    src_dsc = src_path.with_suffix(".dsc")
    dst_dsc = target_txt.with_suffix(".dsc")

    if src_path.resolve() == target_txt.resolve():
        if src_dsc.exists() and not dst_dsc.exists():
            shutil.copy2(src_dsc, dst_dsc)
        return

    moved = False
    if allow_move:
        try:
            shutil.move(str(src_path), str(target_txt))
            moved = True
        except Exception:
            moved = False

    if not moved:
        shutil.copy2(src_path, target_txt)

    if src_dsc.exists():
        if moved:
            try:
                shutil.move(str(src_dsc), str(dst_dsc))
            except Exception:
                shutil.copy2(src_dsc, dst_dsc)
        else:
            shutil.copy2(src_dsc, dst_dsc)


class AttenuationMixin:
    """Handles 'Attenuation' tab and measurement logic."""
    
    def get_poni_file(self, alias):
        """Get PONI file path for a detector alias.
        
        Args:
            alias: Detector alias (e.g. 'SAXS', 'WAXS')
            
        Returns:
            Path to PONI file, or None if not found
        """
        if hasattr(self, 'poni_files') and alias in self.poni_files:
            return self.poni_files[alias].get('path')
        return None

    def create_attenuation_tab(self):
        """Creates attenuation tab UI and connects signals."""
        atten_tab = QWidget()
        atten_layout = QVBoxLayout(atten_tab)
        self.tabs.addTab(atten_tab, "Attenuation")

        # --- 1st row: Number of frames & Integration time ---
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Number of frames:"))
        self.n_repeat_spin = QSpinBox()
        self.n_repeat_spin.setMinimum(1)
        self.n_repeat_spin.setMaximum(1000)
        self.n_repeat_spin.setValue(100)
        frames_layout.addWidget(self.n_repeat_spin)

        frames_layout.addWidget(QLabel("Integration time (s):"))
        self.integration_time_spin = QDoubleSpinBox()
        self.integration_time_spin.setDecimals(6)
        self.integration_time_spin.setSuffix(" s")
        self.integration_time_spin.setValue(0.00005)
        frames_layout.addWidget(self.integration_time_spin)

        atten_layout.addLayout(frames_layout)

        # --- 2nd row: Measure buttons ---
        measure_btn_layout = QHBoxLayout()
        self.measure_without_sample_btn = QPushButton("Measure Without Sample")
        self.measure_with_sample_btn = QPushButton("Measure With Sample")
        measure_btn_layout.addWidget(self.measure_without_sample_btn)
        measure_btn_layout.addWidget(self.measure_with_sample_btn)
        atten_layout.addLayout(measure_btn_layout)

        # --- 3rd row: Integration radius ---
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Integration radius (pixels):"))
        self.integration_radius_spin = QSpinBox()
        self.integration_radius_spin.setRange(1, 20)
        self.integration_radius_spin.setValue(2)
        radius_layout.addWidget(self.integration_radius_spin)
        atten_layout.addLayout(radius_layout)

        # --- 4th row: Calculate and Load buttons ---
        action_btn_layout = QHBoxLayout()
        self.calc_atten_btn = QPushButton("Calculate Attenuation (all pairs)")
        self.load_atten_btn = QPushButton("Load Attenuation Data")
        action_btn_layout.addWidget(self.calc_atten_btn)
        action_btn_layout.addWidget(self.load_atten_btn)
        atten_layout.addLayout(action_btn_layout)

        # --- Results and measurement list ---
        self.result_label = QLabel("No results yet.")
        atten_layout.addWidget(self.result_label)

        self.attenuationList = QListWidget()
        atten_layout.addWidget(QLabel("Measurements:"))
        atten_layout.addWidget(self.attenuationList)

        # --- Connections ---
        self.attenuationList.itemActivated.connect(self.open_attenuation_measurement)
        self.measure_without_sample_btn.clicked.connect(self.measure_without_sample)
        self.measure_with_sample_btn.clicked.connect(self.measure_with_sample)
        self.calc_atten_btn.clicked.connect(self.calculate_all_attenuations)
        self.load_atten_btn.clicked.connect(self.load_attenuation_data)

    def measure_without_sample(self):
        """Captures attenuation measurement without sample."""
        self._attenuation_measure("without")

    def measure_with_sample(self):
        """Captures attenuation measurement with sample."""
        self._attenuation_measure("with")

    def _attenuation_measure(self, mode):
        """Core logic for performing attenuation measurement."""
        # Check if session is active
        if not self.session_manager.is_session_active():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Active Session",
                "Please create a session before measuring attenuation.\n\n"
                "Go to File → New Session...",
            )
            return
        
        N = self.n_repeat_spin.value()
        t_exp = self.integration_time_spin.value()
        save_folder = self.folderLineEdit.text().strip()
        os.makedirs(save_folder, exist_ok=True)

        # Get container version from config
        container_version = get_container_version(
            self.config if hasattr(self, "config") else None
        )
        
        results = {}
        all_data = {}  # Collect data for all detectors

        if getattr(self, "hardware_client", None) is None:
            logger.error(
                "Hardware client is required for attenuation measurement; direct detector calls are disabled"
            )
            return

        try:
            raw_outputs = self.hardware_client.capture_exposure(
                exposure_s=float(t_exp),
                frames=max(int(N), 1),
                timeout_s=max(30.0, float(t_exp) * max(int(N), 1) + 30.0),
            )
        except Exception as e:
            logger.error(
                "Attenuation acquisition via hardware client failed",
                mode=mode,
                error=str(e),
            )
            return

        source_usage = Counter()
        fallback_single = next(iter(raw_outputs.values())) if len(raw_outputs) == 1 else None
        for alias in self.detector_controller.keys():
            src_raw = raw_outputs.get(alias) or fallback_single
            if not src_raw:
                continue
            try:
                source_usage[str(Path(src_raw).resolve())] += 1
            except Exception:
                source_usage[str(src_raw)] += 1

        for alias, detector in self.detector_controller.items():
            center_x, center_y = self.get_beam_center(alias)
            size = detector.size if hasattr(detector, "size") else (256, 256)
            if not (0 <= center_x < size[0] and 0 <= center_y < size[1]):
                logger.error(
                    "Invalid beam center for attenuation measurement",
                    mode=mode,
                    detector=alias,
                    center=(center_x, center_y),
                    detector_size=size,
                )
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = os.path.join(
                save_folder, f"attenuation_{mode}_{alias}_{timestamp}"
            )

            src_raw = raw_outputs.get(alias)
            if src_raw is None and len(raw_outputs) == 1:
                src_raw = next(iter(raw_outputs.values()))
            if not src_raw:
                logger.error(
                    "Attenuation acquisition result not found",
                    mode=mode,
                    detector=alias,
                )
                continue

            txt_file = filename_base + ".txt"
            src_path = Path(src_raw)
            dst_path = Path(txt_file)
            key = str(src_path.resolve())
            allow_move = source_usage.get(key, 0) <= 1
            _place_raw_capture_file(src_raw=src_raw, target_txt=dst_path, allow_move=allow_move)
            if key in source_usage and source_usage[key] > 0:
                source_usage[key] -= 1
            if not os.path.isfile(txt_file):
                logger.error(
                    "Attenuation acquisition failed or file not found",
                    mode=mode,
                    detector=alias,
                    file=txt_file,
                )
                continue

            # Detector converts raw file to container format
            try:
                final_npy_file = detector.convert_to_container_format(txt_file, container_version)
            except Exception as e:
                logger.error(
                    "Failed to convert to container format",
                    mode=mode,
                    detector=alias,
                    file=txt_file,
                    error=str(e),
                )
                continue

            try:
                frame = np.load(final_npy_file)
                radius = self.integration_radius_spin.value()
                value = self.integrate_central_area(
                    frame, center_x, center_y, size=radius
                )
                results[alias] = value
                
                # Store data for session container
                all_data[alias] = frame
                
                logger.info(
                    "Attenuation measurement completed",
                    mode=mode,
                    detector=alias,
                    value=results[alias],
                    file=str(final_npy_file),
                )
            except Exception as e:
                logger.error(
                    "Error loading/integrating attenuation file",
                    mode=mode,
                    detector=alias,
                    file=str(final_npy_file),
                    error=str(e),
                )
                continue

            # Add measurement to attenuationList
            item = QListWidgetItem(f"{mode}: {alias} {timestamp}")
            item.setData(Qt.UserRole, final_npy_file)
            item.setData(
                Qt.UserRole + 1,
                {
                    "mode": mode,
                    "alias": alias,
                    "timestamp": timestamp,
                    "center": (center_x, center_y),
                    "radius": radius,
                    "value": value,
                },
            )
            self.attenuationList.addItem(item)

        # Add to session container
        if all_data:
            try:
                detector_lookup = {
                    d.get("alias"): d for d in self.config.get("detectors", [])
                }
                measurement_data = {}
                detector_metadata = {}
                poni_alias_map = {}
                for alias, signal in all_data.items():
                    detector_meta = detector_lookup.get(alias, {})
                    detector_id = detector_meta.get("id", alias)
                    measurement_data[detector_id] = signal
                    detector_metadata[detector_id] = {
                        "integration_time_ms": t_exp * 1000,
                        "detector_id": detector_id,
                        "timestamp": timestamp,
                        "integration_radius_px": radius,
                        "n_frames": N,
                    }
                    poni_alias_map[alias] = detector_id

                # Add to session container
                self.session_manager.add_attenuation_measurement(
                    measurement_data=measurement_data,
                    detector_metadata=detector_metadata,
                    poni_alias_map=poni_alias_map,
                    mode=mode,
                )
                
                logger.info(
                    f"Added {mode} attenuation to session container",
                    detectors=list(all_data.keys()),
                )
                
                # Update session status in UI
                if hasattr(self, 'update_session_status'):
                    self.update_session_status()
                    
            except Exception as e:
                logger.error(
                    f"Failed to add attenuation to session container: {e}",
                    exc_info=True,
                )

        setattr(self, f"atten_{mode}_results", results)
        self.display_attenuation_result()

    def load_attenuation_data(self):
        """Loads attenuation data from file(s)."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Attenuation Data Files",
            "",
            "Numpy files (*.npy);;All Files (*)",
        )
        if not files:
            return

        import re

        for file_path in files:
            file_name = os.path.basename(file_path)
            match = re.match(
                r"attenuation_(with|without)_([A-Za-z0-9]+)_(\d{8}_\d{6})",
                file_name,
            )
            from datetime import datetime

            if match:
                mode, alias, timestamp = match.groups()
            else:
                mode = "unknown"
                alias = "unknown"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            try:
                frame = np.load(file_path)
                center_x, center_y = self.get_beam_center(alias)
                radius = self.integration_radius_spin.value()
                value = self.integrate_central_area(
                    frame, center_x, center_y, size=radius
                )
            except Exception as e:
                logger.error(
                    "Failed to load/integrate attenuation data",
                    file=file_path,
                    error=str(e),
                )
                value = None

            item = QListWidgetItem(f"{mode}: {alias} {timestamp}")
            item.setData(Qt.UserRole, file_path)
            item.setData(
                Qt.UserRole + 1,
                {
                    "mode": mode,
                    "alias": alias,
                    "timestamp": timestamp,
                    "center": (center_x, center_y),
                    "radius": radius,
                    "value": value,
                },
            )
            self.attenuationList.addItem(item)

    def get_beam_center(self, alias):
        import json
        import re

        poni = self.ponis.get(alias, "")
        # Parse lines
        lines = poni.splitlines()
        poni1 = poni2 = pixel1 = pixel2 = None
        # First, extract Poni1 and Poni2 (in meters)
        for line in lines:
            if line.startswith("Poni1:"):
                poni1 = float(line.split(":")[1].strip())
            if line.startswith("Poni2:"):
                poni2 = float(line.split(":")[1].strip())
            if line.startswith("Detector_config:"):
                # Value is a JSON dict after the colon
                m = re.search(r"Detector_config:\s*(\{.*\})", line)
                if m:
                    cfg = json.loads(m.group(1))
                    pixel1 = float(cfg["pixel1"])
                    pixel2 = float(cfg["pixel2"])
        # Calculate center coordinates in pixels (col, row)
        if None not in (poni1, poni2, pixel1, pixel2):
            center_y = poni1 / pixel1  # row
            center_x = poni2 / pixel2  # col
            return center_x, center_y  # (x, y), both in pixels
        # Fallback to image center if any value missing
        size = self.detector_controller[alias].size
        return (size[0] // 2, size[1] // 2)

    def open_attenuation_measurement(self, item):
        """Opens and displays an attenuation measurement."""
        from hardware.difra.gui.technical.capture import show_measurement_window

        file_path = item.data(Qt.UserRole)
        meta = item.data(Qt.UserRole + 1) or {}
        alias = meta.get("alias", "unknown")

        try:
            center_x, center_y = self.get_beam_center(alias)
            center = (center_y, center_x)  # for plotting: (y, x)
        except Exception:
            frame = np.load(file_path)
            center = (frame.shape[0] // 2, frame.shape[1] // 2)

        radius = meta.get("radius", self.integration_radius_spin.value())
        show_measurement_window(
            measurement_filename=file_path,
            mask=None,
            poni_text=(
                self.ponis.get(alias)
                if hasattr(self, "ponis") and alias in self.ponis
                else None
            ),
            parent=self,
            columns_to_remove=30,
            goodness=0.0,
            center=center,
            integration_radius=radius,
        )

    def display_attenuation_result(self):
        """Displays attenuation result summary."""
        if hasattr(self, "atten_without_results") and hasattr(
            self, "atten_with_results"
        ):
            texts = []
            for alias in self.atten_without_results:
                I0 = self.atten_without_results[alias]
                I = self.atten_with_results.get(alias)
                if I is None or I0 <= 0 or I <= 0:
                    alpha = "N/A"
                else:
                    alpha = -np.log10(I / I0)
                texts.append(f"{alias}: I0={I0:.1f}, I={I:.1f}, α={alpha}")
            self.result_label.setText("\n".join(texts))
        else:
            self.result_label.setText("Need both measurements for attenuation.")

    def integrate_file(self, file_path, alias):
        """Integrates loaded file for central area."""
        frame = np.load(file_path)
        center_x, center_y = self.get_beam_center(alias)
        radius = self.integration_radius_spin.value()
        return self.integrate_central_area(frame, center_x, center_y, size=radius)

    def calculate_all_attenuations(self):
        """Calculates attenuation for all loaded measurements."""
        from collections import defaultdict

        by_alias = defaultdict(lambda: {"with": [], "without": []})
        for i in range(self.attenuationList.count()):
            item = self.attenuationList.item(i)
            meta = item.data(Qt.UserRole + 1)
            file_path = item.data(Qt.UserRole)
            if meta:
                by_alias[meta["alias"]][meta["mode"]].append(
                    (meta["timestamp"], file_path)
                )
        results = []
        for alias, modes in by_alias.items():
            withs = sorted(modes["with"])
            withouts = sorted(modes["without"])
            for (t_with, file_with), (t_wo, file_wo) in zip(withs, withouts):
                I = self.integrate_file(file_with, alias)
                I0 = self.integrate_file(file_wo, alias)
                if I > 0 and I0 > 0:
                    alpha = -np.log10(I / I0)
                else:
                    alpha = "N/A"
                results.append((alias, t_with, t_wo, I, I0, alpha))
        txts = [
            f"{alias}: I0={I0:.1f}, I={I:.1f}, α={alpha}"
            for alias, t_with, t_wo, I, I0, alpha in results
        ]
        self.result_label.setText("\n".join(txts))

    def integrate_central_area(self, frame, center_x, center_y, size=2):
        """Integrates square region in a frame."""
        x0 = int(round(center_x)) - size
        x1 = int(round(center_x)) + size
        y0 = int(round(center_y)) - size
        y1 = int(round(center_y)) + size
        x0, x1 = max(x0, 0), min(x1, frame.shape[1])
        y0, y1 = max(y0, 0), min(y1, frame.shape[0])
        return np.sum(frame[y0:y1, x0:x1])
