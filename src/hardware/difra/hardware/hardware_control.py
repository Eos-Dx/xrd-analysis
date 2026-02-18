# hardware_control.py
import os

from hardware.difra.hardware.detectors import (
    DetectorController,
    DummyDetectorController,
    PixetDetectorController,
    PixetLegacyDetectorController,
    PixetSidecarDetectorController,
)
from hardware.difra.hardware.xystages import (
    BaseStageController,
    DummyStageController,
    MarlinStageController,
    XYStageLibController,
)

# Mapping types from config to classes
DETECTOR_CLASSES = {
    "Pixet": PixetDetectorController,
    "PixetLegacy": PixetLegacyDetectorController,
    "PixetSidecar": PixetSidecarDetectorController,
    "DummyDetector": DummyDetectorController,
}

STAGE_CLASSES = {
    "Kinesis": XYStageLibController,
    "Marlin": MarlinStageController,
    "DummyStage": DummyStageController,
}


class HardwareController:
    def __init__(self, config):
        self.config = config
        self.detectors = {}  # alias → DetectorController
        self.stage_controller: BaseStageController = None
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

    def initialize(self, init_stage: bool = True, init_detector: bool = True):
        dev_mode = self.config.get("DEV", True)
        detector_success = bool(self.detectors)
        stage_success = bool(self.stage_controller)

        if init_detector:
            # --- Initialize Detectors ---
            detector_list = self.config.get("detectors", [])
            selected_ids = (
                self.config.get("dev_active_detectors", [])
                if dev_mode
                else self.config.get("active_detectors", [])
            )
            selected_detectors = [d for d in detector_list if d["id"] in selected_ids]

            def _resolve_detector_class(det_cfg):
                det_type = str(det_cfg.get("type", "")).strip()
                env_detector_backend = str(
                    os.environ.get("DETECTOR_BACKEND", "")
                ).lower().strip()
                detector_backend = str(
                    env_detector_backend
                    or det_cfg.get("detector_backend", det_cfg.get("backend", ""))
                ).lower().strip()

                if det_type in {"Pixet", "PixetLegacy", "PixetSidecar"}:
                    if detector_backend not in {"sidecar", "socket", "ipc"}:
                        print(
                            "⚠ Pixet detectors are restricted to legacy sidecar mode; "
                            f"forcing DETECTOR_BACKEND=sidecar (was '{detector_backend or 'unset'}')."
                        )
                    os.environ["DETECTOR_BACKEND"] = "sidecar"
                    os.environ["PIXET_BACKEND"] = "sidecar"
                    return "PixetSidecar", PixetSidecarDetectorController

                if detector_backend in {"sidecar", "socket", "ipc"} and det_type in {
                    "Pixet",
                    "DummyDetector",
                }:
                    return "PixetSidecar", PixetSidecarDetectorController
                return det_type, DETECTOR_CLASSES.get(det_type)

            self.detectors = {}
            for det_cfg in selected_detectors:
                det_type, det_class = _resolve_detector_class(det_cfg)
                if not det_class:
                    print(f"⚠ Unknown detector type: {det_type}")
                    continue
                alias = det_cfg.get("alias", det_cfg["id"])
                size = (det_cfg["size"]["width"], det_cfg["size"]["height"])
                try:
                    if det_type == "DummyDetector":
                        controller = det_class(alias=alias, size=size)
                    elif det_type in {"Pixet", "PixetLegacy", "PixetSidecar"}:
                        controller = det_class(alias=alias, size=size, config=det_cfg)
                    else:
                        controller = det_class(alias=alias, size=size)
                    success = controller.init_detector()
                    if success:
                        self.detectors[alias] = controller
                        print(f"✓ Detector '{alias}' ({det_type}) initialized successfully")
                    else:
                        print(f"✗ Detector '{alias}' ({det_type}) failed to initialize")
                except Exception as e:
                    print(f"✗ Error initializing detector '{alias}' ({det_type}): {e}")

            detector_success = bool(self.detectors)

        if init_stage:
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

            if self.stage_controller is not None:
                try:
                    self.stage_controller.deinit()
                except Exception as e:
                    print(f"[Stage Reinit Warning] {e}")
                finally:
                    self.stage_controller = None

            if selected_stage:
                stage_type = selected_stage.get("type")
                stage_class = STAGE_CLASSES.get(stage_type)
                if not stage_class:
                    print(f"⚠ Unknown stage type: {stage_type}")
                    stage_success = False
                else:
                    try:
                        self.stage_controller = stage_class(config=selected_stage)
                        stage_success = self.stage_controller.init_stage()
                        if stage_success:
                            print(f"✓ Stage '{selected_stage.get('alias')}' ({stage_type}) initialized successfully")
                        else:
                            print(f"✗ Stage '{selected_stage.get('alias')}' ({stage_type}) failed to initialize")
                    except Exception as e:
                        print(f"✗ Error initializing stage '{selected_stage.get('alias')}' ({stage_type}): {e}")
                        stage_success = False
            else:
                print("⚠ No translation stage selected.")
                stage_success = False

        # Consider hardware initialized if at least one component succeeded
        self.hardware_initialized = stage_success or detector_success
        return stage_success, detector_success

    def deinitialize(self):
        if self.stage_controller:
            try:
                self.stage_controller.deinit()
            except Exception as e:
                print(f"[Stage Deinit Error] {e}")
        for alias, detector in self.detectors.items():
            try:
                detector.deinit_detector()
            except Exception as e:
                print(f"[Detector '{alias}' Deinit Error] {e}")
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
