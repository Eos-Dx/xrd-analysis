# hardware/manager.py
from utils.logging import get_module_logger

from .controllers.detector import (
    DetectorController,
    DummyDetectorController,
    PixetDetectorController,
)
from .controllers.stage import (
    BrushlessStageController,
    DummyStageController,
    StageController,
)

logger = get_module_logger(__name__)

# Mapping types from config to classes
DETECTOR_CLASSES = {
    "Pixet": PixetDetectorController,
    "DummyDetector": DummyDetectorController,
}

STAGE_CLASSES = {
    "brushless": BrushlessStageController,
    "dummy": DummyStageController,
}


class HardwareController:
    def __init__(self, config):
        self.config = config
        self.detectors = {}  # alias → DetectorController
        self.stage_controller: StageController = None
        self.hardware_initialized = False

    @property
    def active_detector_aliases(self):
        dev_mode = self.config.get("DEV", False)
        active_ids = (
            self.config.get("dev_active_detectors", [])
            if dev_mode
            else self.config.get("active_detectors", [])
        )
        aliases = [
            det_cfg["alias"]
            for det_cfg in self.config.get("detectors", [])
            if det_cfg["id"] in active_ids
        ]
        return aliases

    def initialize(self):
        dev_mode = self.config.get("DEV", True)

        # --- Initialize Detectors ---
        detector_list = self.config.get("detectors", [])
        selected_ids = (
            self.config.get("dev_active_detectors", [])
            if dev_mode
            else self.config.get("active_detectors", [])
        )
        selected_detectors = [d for d in detector_list if d["id"] in selected_ids]

        for det_cfg in selected_detectors:
            det_type = det_cfg.get("type")
            det_class = DETECTOR_CLASSES.get(det_type)
            if not det_class:
                raise ValueError(f"Unknown detector type: {det_type}")
            # Prepare init kwargs
            alias = det_cfg.get("alias", det_cfg["id"])
            size = (det_cfg["size"]["width"], det_cfg["size"]["height"])
            # You may add more config fields if your detector needs them
            if det_type == "DummyDetector":
                controller = det_class(alias=alias, size=size)
            elif det_type == "Pixet":
                controller = det_class(
                    alias=alias, size=size, config=det_cfg
                )  # Adjust as needed
            else:
                controller = det_class(alias=alias, size=size)
            success = controller.init_detector()
            if not success:
                raise RuntimeError(f"Failed to initialize detector {det_cfg['id']}")
            self.detectors[alias] = controller

        # --- Initialize Stage ---
        stage_list = self.config.get("translation_stages", [])
        selected_stage_ids = (
            self.config.get("dev_active_stages", [])
            if dev_mode
            else self.config.get("active_translation_stages", [])
        )
        selected_stage = next(
            (s for s in stage_list if s["id"] in selected_stage_ids), None
        )

        if selected_stage:
            stage_type = selected_stage.get("type")
            stage_class = STAGE_CLASSES.get(stage_type)
            if not stage_class:
                raise ValueError(f"Unknown stage type: {stage_type}")
            self.stage_controller = stage_class(config=selected_stage)
            stage_success = self.stage_controller.init_stage()
        else:
            logger.warning("No translation stage selected")
            stage_success = False

        self.hardware_initialized = stage_success and bool(self.detectors)
        return stage_success, bool(self.detectors)

    def deinitialize(self):
        if self.stage_controller:
            try:
                self.stage_controller.deinit()
            except Exception as e:
                logger.error("Stage deinitialization error", error=str(e))
        for alias, detector in self.detectors.items():
            try:
                detector.deinit_detector()
            except Exception as e:
                logger.error(
                    "Detector deinitialization error", detector=alias, error=str(e)
                )
        self.hardware_initialized = False

    def get_xy_position(self):
        if self.stage_controller:
            return self.stage_controller.get_xy_position()
        return 0.0, 0.0

    def move_stage(self, x, y, timeout=10):
        if self.stage_controller:
            return self.stage_controller.move_stage(x, y, move_timeout=timeout)
        return x, y

    def home_stage(self, timeout=10):
        if self.stage_controller:
            return self.stage_controller.home_stage(timeout_s=timeout)
        return 0.0, 0.0

    def get_detector(self, alias: str) -> DetectorController:
        return self.detectors.get(alias)

    def list_detectors(self):
        return list(self.detectors.keys())

    def is_initialized(self):
        return self.hardware_initialized
