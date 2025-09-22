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

from xrdanalysis.data_processing.utility_functions import create_mask


class DetectorParamMixin:
    """Handles detector tabs with PONI/mask settings, one tab per active detector alias."""

    def setup_detector_param_tabs(self):
        """Creates detector tabs based on active detector aliases."""
        # Store references to detector tabs for rebuilding
        if not hasattr(self, "detector_tabs"):
            self.detector_tabs = {}

    def populate_detector_param_tabs(self):
        """Creates one tab per active detector alias with PONI/mask settings."""
        self.clear_detector_param_tabs()

        try:
            self.detector_aliases = self.hardware_controller.active_detector_aliases
        except AttributeError:
            # Fallback when hardware controller not available
            dev_mode = self.config.get("DEV", False)
            active_ids = (
                self.config.get("dev_active_detectors", [])
                if dev_mode
                else self.config.get("active_detectors", [])
            )
            self.detector_aliases = [
                d.get("alias")
                for d in self.config.get("detectors", [])
                if d.get("id") in active_ids
            ]

        for alias in self.detector_aliases:
            # Create a new tab for this detector
            detector_tab = QWidget()
            detector_layout = QVBoxLayout(detector_tab)

            # Mask section
            mask_layout = QHBoxLayout()
            mask_layout.addWidget(QLabel(f"Mask file:"))
            mask_lineedit = QLineEdit()
            setattr(self, f"{alias.lower()}_mask_lineedit", mask_lineedit)
            mask_layout.addWidget(mask_lineedit)
            mask_btn = QPushButton("Browse...")
            mask_btn.clicked.connect(partial(self.browse_mask_file, detector=alias))
            mask_layout.addWidget(mask_btn)
            detector_layout.addLayout(mask_layout)

            # PONI section
            poni_layout = QHBoxLayout()
            poni_layout.addWidget(QLabel(f"PONI file:"))
            poni_lineedit = QLineEdit()
            # Prefill with known path if available
            try:
                if hasattr(self, "poni_files"):
                    meta = self.poni_files.get(alias, {})
                    pth = meta.get("path")
                    if pth:
                        poni_lineedit.setText(str(pth))
            except Exception:
                pass
            setattr(self, f"{alias.lower()}_poni_lineedit", poni_lineedit)
            poni_layout.addWidget(poni_lineedit)
            poni_btn = QPushButton("Browse...")
            poni_btn.clicked.connect(partial(self.browse_poni_file, detector=alias))
            poni_layout.addWidget(poni_btn)
            detector_layout.addLayout(poni_layout)

            # Add some spacing
            detector_layout.addStretch()

            # Add the tab to the tab widget using the detector alias as the tab name
            tab_index = self.tabs.addTab(detector_tab, alias)
            self.detector_tabs[alias] = {"widget": detector_tab, "index": tab_index}

    def clear_detector_param_tabs(self):
        """Removes all detector tabs (tabs created from detector aliases)."""
        if not hasattr(self, "detector_tabs"):
            return

        # Remove tabs in reverse order to maintain indices
        tabs_to_remove = []
        for alias, tab_info in self.detector_tabs.items():
            tabs_to_remove.append((tab_info["index"], alias))

        # Sort by index descending
        tabs_to_remove.sort(key=lambda x: x[0], reverse=True)

        for tab_index, alias in tabs_to_remove:
            # Remove the tab from the tab widget
            widget = self.tabs.widget(tab_index)
            if widget:
                self.tabs.removeTab(tab_index)
                widget.deleteLater()

        # Clear the tracking dictionary
        self.detector_tabs.clear()

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

    def _create_fake_poni_file(self, alias: str, detector_config: dict) -> str:
        """Creates a fake PONI file content for demo mode detectors."""
        import random
        import time

        # Get detector size or use defaults
        size = detector_config.get("size", {"width": 256, "height": 256})
        width = size.get("width", 256)
        height = size.get("height", 256)

        # Generate slightly different parameters for each detector
        random.seed(hash(alias))  # Consistent values based on alias

        # Different distances and positions for different detector types
        if "SAXS" in alias.upper():
            distance = round(random.uniform(0.15, 0.25), 6)  # Longer distance for SAXS
            poni1 = round(random.uniform(0.006, 0.008), 6)
            poni2 = round(random.uniform(0.001, 0.003), 6)
        elif "WAXS" in alias.upper():
            distance = round(random.uniform(0.08, 0.12), 6)  # Shorter distance for WAXS
            poni1 = round(random.uniform(0.005, 0.009), 6)
            poni2 = round(random.uniform(0.0008, 0.0025), 6)
        else:
            # Generic values for other detectors
            distance = round(random.uniform(0.10, 0.20), 6)
            poni1 = round(random.uniform(0.005, 0.010), 6)
            poni2 = round(random.uniform(0.0008, 0.0030), 6)

        # Generate pixel sizes (typically 55um or 100um)
        pixel_size = detector_config.get("pixel_size_um", [55, 55])
        pixel1 = pixel_size[0] * 1e-6 if len(pixel_size) > 0 else 5.5e-05
        pixel2 = pixel_size[1] * 1e-6 if len(pixel_size) > 1 else 5.5e-05

        wavelength = 1.5406e-10  # Typical Cu Kα wavelength

        current_time = time.strftime("%a %b %d %H:%M:%S %Y")

        poni_content = f"""# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis
# Calibration done on {current_time} (DEMO MODE - FAKE DATA)
poni_version: 2.1
Detector: Detector
Detector_config: {{"pixel1": {pixel1}, "pixel2": {pixel2}, "max_shape": [{height}, {width}], "orientation": 3}}
Distance: {distance}
Poni1: {poni1}
Poni2: {poni2}
Rot1: 0
Rot2: 0
Rot3: 0
Wavelength: {wavelength}
# Calibrant: AgBh (DEMO)
# Detector: {alias} (DEMO MODE)
# Image: demo://fake_calibration_image_{alias.lower()}.npy
"""
        return poni_content

    def _create_demo_poni_files(self, detectors: list, resource_dir: Path):
        """Creates fake PONI files for demo mode detectors."""
        demo_poni_dir = resource_dir / "demo_poni_files"
        demo_poni_dir.mkdir(exist_ok=True)

        for det_cfg in detectors:
            alias = det_cfg["alias"]
            poni_filename = f"{alias.lower()}_demo.poni"
            poni_path = demo_poni_dir / poni_filename

            # Generate fake PONI content
            poni_content = self._create_fake_poni_file(alias, det_cfg)

            # Write the fake PONI file
            try:
                with open(poni_path, "w") as f:
                    f.write(poni_content)
                print(f"Created demo PONI file: {poni_path}")
            except Exception as e:
                print(f"Failed to create demo PONI file for {alias}: {e}")

    def load_default_masks_and_ponis(self):
        """Loads default masks/ponis from config for each ACTIVE detector alias.
        In demo mode, creates fake PONI files automatically."""
        self.masks = {}
        self.ponis = {}
        self.poni_files = {}
        self.detector_aliases = []

        detectors_all = self.config.get("detectors", [])
        dev_mode = self.config.get("DEV", False)
        active_ids = (
            self.config.get("dev_active_detectors", [])
            if dev_mode
            else self.config.get("active_detectors", [])
        )
        detectors = [d for d in detectors_all if d.get("id") in active_ids]

        resource_dir = (
            Path(__file__).resolve().parent.parent.parent.parent / "resources"
        )

        # Create fake PONI files for demo mode
        if dev_mode:
            self._create_demo_poni_files(detectors, resource_dir)

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

            # PONI: prefer poni_file, then demo PONI file (if dev mode), then default_poni, then empty
            poni_value = ""
            meta_path = None
            meta_name = None
            poni_file = det_cfg.get("poni_file")

            # Try explicit poni_file first
            if poni_file:
                poni_path = Path(poni_file)
                if not poni_path.is_absolute():
                    poni_path = resource_dir / poni_file
                try:
                    with open(str(poni_path), "r") as f:
                        poni_value = f.read()
                    meta_path = str(poni_path)
                    meta_name = os.path.basename(meta_path)
                except Exception as e:
                    print(f"Failed to load explicit poni file for {alias}: {e}")

            # If dev mode and no explicit poni file, try the generated demo PONI file
            if not poni_value and dev_mode:
                demo_poni_dir = resource_dir / "demo_poni_files"
                demo_poni_file = demo_poni_dir / f"{alias.lower()}_demo.poni"
                if demo_poni_file.exists():
                    try:
                        with open(str(demo_poni_file), "r") as f:
                            poni_value = f.read()
                        meta_path = str(demo_poni_file)
                        meta_name = os.path.basename(meta_path)
                        print(f"Using demo PONI file for {alias}: {meta_name}")
                    except Exception as e:
                        print(f"Failed to load demo PONI for {alias}: {e}")

            # Fallback to default_poni if still no value
            if not poni_value:
                poni_value = det_cfg.get("default_poni", "")
                if poni_value:
                    print(f"Using default_poni for {alias}")
            self.ponis[alias] = poni_value
            self.poni_files[alias] = {
                "path": meta_path,
                "name": meta_name,
            }

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
            # Track meta (filename and path)
            if not hasattr(self, "poni_files"):
                self.poni_files = {}
            self.poni_files[detector] = {
                "path": str(poni_file),
                "name": os.path.basename(str(poni_file)),
            }
        except Exception as e:
            print(f"Error loading PONI file for {detector}:", e)

    def get_current_poni_settings(self):
        """Captures current PONI settings from UI line edits for active detectors."""
        current_settings = {}
        try:
            active_aliases = getattr(self, "detector_aliases", [])
            for alias in active_aliases:
                poni_lineedit = getattr(self, f"{alias.lower()}_poni_lineedit", None)
                if poni_lineedit:
                    path = poni_lineedit.text().strip()
                    current_settings[alias] = {
                        "path": path if path else None,
                        "name": os.path.basename(path) if path else None,
                        "value": self.ponis.get(alias, ""),
                    }
        except Exception as e:
            print(f"Error capturing current PONI settings: {e}")
        return current_settings

    def refresh_detector_tabs_for_mode_switch(self):
        """Explicitly refresh detector tabs when switching between demo/production modes.
        This can be called manually or automatically when config changes."""
        print("Refreshing detector tabs for mode switch...")

        # Reload masks and ponis for new active detectors
        self.load_default_masks_and_ponis()

        # Rebuild the tabs with new detector aliases
        self.populate_detector_param_tabs()

        print(
            f"Detector tabs refreshed. Active detectors: {getattr(self, 'detector_aliases', [])}"
        )

    def on_config_mode_changed(self, dev_mode: bool):
        """Called when DEV mode is toggled in the config.
        Updates detector tabs to reflect the new mode without requiring hardware restart.
        """
        print(f"Config mode changed to: {'DEMO' if dev_mode else 'PRODUCTION'}")

        # Update the config DEV flag if needed
        self.config["DEV"] = dev_mode

        # Refresh detector tabs immediately for the new mode
        self.refresh_detector_tabs_for_mode_switch()

        # If hardware is already initialized, we may want to reinitialize it
        # to switch to the correct detector controllers
        if getattr(self, "hardware_initialized", False):
            print(
                "Hardware is initialized. Consider reinitializing to switch detector controllers."
            )
            # Optionally auto-reinitialize:
            # self.toggle_hardware()  # deinitialize
            # self.toggle_hardware()  # reinitialize with new mode
