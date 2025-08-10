# zone_measurements/detector_param_mixin.py

import os
from functools import partial
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Path setup is handled in Ulster.__init__.py

from xrdanalysis.data_processing.utility_functions import create_mask


class DetectorParamMixin:
    """Handles 'Detector param' tab, mask/poni file loading, display, save."""

    def setup_detector_param_tab(self):
        """Create the Detector param tab and layout if not already done."""
        if not hasattr(self, "param_tab"):
            self.param_tab = QWidget()
            self.param_layout = QVBoxLayout(self.param_tab)
            self.tabs.addTab(self.param_tab, "Detector param")

    def populate_detector_param_tab(self):
        """Populates detector param tab with widgets for each detector."""
        self.clear_detector_param_tab()
        self.detector_aliases = self.hardware_controller.active_detector_aliases

        for alias in self.detector_aliases:
            # Mask widget
            mask_layout = QHBoxLayout()
            mask_layout.addWidget(QLabel(f"{alias} Mask:"))
            mask_lineedit = QLineEdit()
            setattr(self, f"{alias.lower()}_mask_lineedit", mask_lineedit)
            mask_layout.addWidget(mask_lineedit)
            mask_btn = QPushButton("Browse...")
            mask_btn.clicked.connect(partial(self.browse_mask_file, detector=alias))
            mask_layout.addWidget(mask_btn)
            self.param_layout.addLayout(mask_layout)

            # PONI widget
            poni_layout = QHBoxLayout()
            poni_layout.addWidget(QLabel(f"{alias} PONI:"))
            poni_lineedit = QLineEdit()
            setattr(self, f"{alias.lower()}_poni_lineedit", poni_lineedit)
            poni_layout.addWidget(poni_lineedit)
            poni_btn = QPushButton("Browse...")
            poni_btn.clicked.connect(partial(self.browse_poni_file, detector=alias))
            poni_layout.addWidget(poni_btn)
            self.param_layout.addLayout(poni_layout)

    def clear_detector_param_tab(self):
        """Removes all widgets from the detector param tab."""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                sub_layout = item.layout()
                while sub_layout.count():
                    sub_item = sub_layout.takeAt(0)
                    sub_widget = sub_item.widget()
                    if sub_widget:
                        sub_widget.deleteLater()
                del sub_layout

    def browse_mask_file(self, detector):
        """Opens file dialog and loads mask file."""
        mask_file, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {detector} Mask File",
            "",
            "Mask Files (*.mask *.npy *.txt);;All Files (*)",
        )
        if mask_file:
            getattr(self, f"{detector.lower()}_mask_lineedit").setText(mask_file)
            self.load_mask_file(mask_file, detector)

    def browse_poni_file(self, detector):
        """Opens file dialog and loads poni file."""
        poni_file, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {detector} PONI File",
            "",
            "PONI Files (*.poni);;All Files (*)",
        )
        if poni_file:
            getattr(self, f"{detector.lower()}_poni_lineedit").setText(poni_file)
            self.load_poni_file(poni_file, detector)

    def load_default_masks_and_ponis(self):
        """Loads default masks/ponis from config for each detector alias."""
        self.masks = {}
        self.ponis = {}
        self.detector_aliases = []

        detectors = self.config.get("detectors", [])
        resource_dir = (
            Path(__file__).resolve().parent.parent.parent.parent / "resources"
        )

        for det_cfg in detectors:
            alias = det_cfg["alias"]
            self.detector_aliases.append(alias)

            # Get detector size from config, fallback to (256, 256)
            det_size = (
                tuple(det_cfg.get("size", {}).values())
                if "size" in det_cfg
                else (256, 256)
            )
            if len(det_size) != 2:
                det_size = (256, 256)

            # Faulty pixels mask (optional)
            mask_file = det_cfg.get("faulty_pixels")
            mask_path = None
            if mask_file:
                mask_path = Path(mask_file)
                if not mask_path.is_absolute():
                    mask_path = resource_dir / mask_path
                try:
                    faulty_pixels = np.load(str(mask_path), allow_pickle=True)
                    mask = create_mask(faulty_pixels, size=det_size)
                    self.masks[alias] = mask
                except Exception as e:
                    print(f"Failed to load faulty pixels for {alias}: {e}")
                    self.masks[alias] = None
            else:
                self.masks[alias] = None

            # PONI: prefer poni_file, fallback to default_poni, fallback to empty
            poni = ""
            poni_file = det_cfg.get("poni_file")
            if poni_file:
                poni_path = Path(poni_file)
                if not poni_path.is_absolute():
                    poni_path = resource_dir / poni_path
                try:
                    with open(str(poni_path), "r") as f:
                        poni = f.read()
                except Exception as e:
                    print(f"Failed to load poni for {alias}: {e}")
                    poni = ""
            else:
                poni = det_cfg.get("default_poni", "")
            self.ponis[alias] = poni

    def load_mask_file(self, mask_file, detector):
        """Loads a mask file for a given detector."""
        try:
            data = np.load(mask_file)
            # you may want to adjust the mask creation depending on your logic:
            self.masks[detector] = create_mask(data)
        except Exception as e:
            print(f"Error loading mask file for {detector}:", e)

    def load_poni_file(self, poni_file, detector):
        """Loads a poni file for a given detector."""
        try:
            with open(poni_file, "r") as f:
                self.ponis[detector] = f.read()
        except Exception as e:
            print(f"Error loading PONI file for {detector}:", e)
