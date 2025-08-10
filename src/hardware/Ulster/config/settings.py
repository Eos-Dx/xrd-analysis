"""Configuration settings management for Ulster application."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logging import get_module_logger

logger = get_module_logger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for a detector."""

    alias: str
    type: str  # "dummy" or "pixet"
    id: str
    size: tuple  # (width, height)
    pixel_size_um: Optional[float] = None
    faulty_pixels: Optional[list] = None


@dataclass
class StageConfig:
    """Configuration for a stage controller."""

    type: str  # "dummy" or "brushless"
    x_serial: Optional[str] = None
    y_serial: Optional[str] = None
    x_limits: Optional[tuple] = None  # (min, max)
    y_limits: Optional[tuple] = None  # (min, max)


@dataclass
class AppSettings:
    """Main application settings."""

    dev_mode: bool = True
    default_folder: str = ""
    default_image_folder: str = ""
    default_image: str = ""
    log_level: str = "INFO"
    log_to_file: bool = True


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "resources" / "config" / "main.json"
            )

        self.config_path = config_path
        self._config_data = {}
        self._detectors = []
        self._stage_config = None
        self._app_settings = None

        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    self._config_data = json.load(f)

                self._parse_config()
                logger.info("Configuration loaded", path=str(self.config_path))
            else:
                logger.warning(
                    "Config file not found, using defaults", path=str(self.config_path)
                )
                self._create_default_config()

        except Exception as e:
            logger.error(
                "Error loading configuration", error=str(e), path=str(self.config_path)
            )
            self._create_default_config()

    def save_config(self):
        """Save current configuration to file."""
        try:
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Build config data
            config_data = {
                "DEV": self._app_settings.dev_mode if self._app_settings else True,
                "default_folder": (
                    self._app_settings.default_folder if self._app_settings else ""
                ),
                "default_image_folder": (
                    self._app_settings.default_image_folder
                    if self._app_settings
                    else ""
                ),
                "default_image": (
                    self._app_settings.default_image if self._app_settings else ""
                ),
                "log_level": (
                    self._app_settings.log_level if self._app_settings else "INFO"
                ),
                "log_to_file": (
                    self._app_settings.log_to_file if self._app_settings else True
                ),
                "detectors": [asdict(det) for det in self._detectors],
                "stage": asdict(self._stage_config) if self._stage_config else {},
            }

            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=4)

            logger.info("Configuration saved", path=str(self.config_path))

        except Exception as e:
            logger.error(
                "Error saving configuration", error=str(e), path=str(self.config_path)
            )

    def _parse_config(self):
        """Parse loaded configuration data."""
        # Parse app settings
        self._app_settings = AppSettings(
            dev_mode=self._config_data.get("DEV", True),
            default_folder=self._config_data.get("default_folder", ""),
            default_image_folder=self._config_data.get("default_image_folder", ""),
            default_image=self._config_data.get("default_image", ""),
            log_level=self._config_data.get("log_level", "INFO"),
            log_to_file=self._config_data.get("log_to_file", True),
        )

        # Parse detectors
        self._detectors = []
        for det_data in self._config_data.get("detectors", []):
            detector = DetectorConfig(
                alias=det_data["alias"],
                type=det_data["type"],
                id=det_data["id"],
                size=tuple(det_data["size"]),
                pixel_size_um=det_data.get("pixel_size_um"),
                faulty_pixels=det_data.get("faulty_pixels"),
            )
            self._detectors.append(detector)

        # Parse stage config
        stage_data = self._config_data.get("stage", {})
        if stage_data:
            self._stage_config = StageConfig(
                type=stage_data.get("type", "dummy"),
                x_serial=stage_data.get("x_serial"),
                y_serial=stage_data.get("y_serial"),
                x_limits=(
                    tuple(stage_data["x_limits"]) if "x_limits" in stage_data else None
                ),
                y_limits=(
                    tuple(stage_data["y_limits"]) if "y_limits" in stage_data else None
                ),
            )

    def _create_default_config(self):
        """Create default configuration."""
        self._app_settings = AppSettings()
        self._detectors = [
            DetectorConfig(
                alias="DUMMY_DETECTOR", type="dummy", id="dummy_001", size=(256, 256)
            )
        ]
        self._stage_config = StageConfig(type="dummy")

        logger.info("Created default configuration")

    @property
    def app_settings(self) -> AppSettings:
        """Get application settings."""
        return self._app_settings

    @property
    def detectors(self) -> list:
        """Get detector configurations."""
        return self._detectors

    @property
    def stage_config(self) -> Optional[StageConfig]:
        """Get stage configuration."""
        return self._stage_config

    @property
    def dev_mode(self) -> bool:
        """Check if in development mode."""
        return self._app_settings.dev_mode if self._app_settings else True

    def get_detector_config(self, alias: str) -> Optional[DetectorConfig]:
        """Get configuration for a specific detector."""
        for detector in self._detectors:
            if detector.alias == alias:
                return detector
        return None

    def update_dev_mode(self, dev_mode: bool):
        """Update development mode setting."""
        if self._app_settings:
            self._app_settings.dev_mode = dev_mode
            self.save_config()
            logger.info("Dev mode updated", dev_mode=dev_mode)


# Global configuration instance
config_manager = ConfigManager()
