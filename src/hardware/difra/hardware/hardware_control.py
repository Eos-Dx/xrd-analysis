# hardware_control.py
import logging
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

logger = logging.getLogger(__name__)


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
        dev_mode = self.config.get("DEV", False)
        detector_success = bool(self.detectors)
        stage_success = bool(self.stage_controller)

        if dev_mode:
            logger.warning(
                "DEV mode is enabled: using dev_active_detectors/dev_active_stages"
            )
            print(
                "[WARN] DEV mode is enabled. Hardware selection uses dev_active_detectors/dev_active_stages."
            )

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

                # Dummy detector should stay local/demo even when detector backend is sidecar.
                if det_type == "DummyDetector":
                    return "DummyDetector", DummyDetectorController, "demo"

                if det_type in {"Pixet", "PixetLegacy", "PixetSidecar"}:
                    if detector_backend not in {"sidecar", "socket", "ipc"}:
                        print(
                            "⚠ Pixet detectors are restricted to legacy sidecar mode; "
                            f"forcing DETECTOR_BACKEND=sidecar (was '{detector_backend or 'unset'}')."
                        )
                    os.environ["DETECTOR_BACKEND"] = "sidecar"
                    os.environ["PIXET_BACKEND"] = "sidecar"
                    return "PixetSidecar", PixetSidecarDetectorController, "legacy-sidecar"

                if detector_backend in {"sidecar", "socket", "ipc"} and det_type in {
                    "Pixet",
                }:
                    return "PixetSidecar", PixetSidecarDetectorController, "legacy-sidecar"

                protocol = "direct"
                if det_type == "Pixet":
                    protocol = "ctypes-direct"
                elif det_type == "PixetLegacy":
                    protocol = "legacy-direct"
                elif det_type == "DummyDetector":
                    protocol = "demo"
                return det_type, DETECTOR_CLASSES.get(det_type), protocol

            self.detectors = {}
            for det_cfg in selected_detectors:
                det_type, det_class, protocol = _resolve_detector_class(det_cfg)
                if not det_class:
                    print(f"⚠ Unknown detector type: {det_type}")
                    continue
                alias = det_cfg.get("alias", det_cfg["id"])
                size = (det_cfg["size"]["width"], det_cfg["size"]["height"])
                backend = str(os.environ.get("DETECTOR_BACKEND", "")).lower().strip() or "unset"
                logger.info(
                    "Detector init requested: alias=%s id=%s cfg_type=%s resolved_type=%s class=%s protocol=%s backend=%s size=%sx%s",
                    alias,
                    det_cfg.get("id", "unknown"),
                    det_cfg.get("type", "unknown"),
                    det_type,
                    getattr(det_class, "__name__", str(det_class)),
                    protocol,
                    backend,
                    size[0],
                    size[1],
                )
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
                        logger.info(
                            "Detector initialized: alias=%s class=%s protocol=%s",
                            alias,
                            controller.__class__.__name__,
                            protocol,
                        )
                    else:
                        print(f"✗ Detector '{alias}' ({det_type}) failed to initialize")
                        logger.error(
                            "Detector initialization failed: alias=%s type=%s class=%s protocol=%s",
                            alias,
                            det_type,
                            getattr(det_class, "__name__", str(det_class)),
                            protocol,
                        )
                except Exception as e:
                    print(f"✗ Error initializing detector '{alias}' ({det_type}): {e}")
                    logger.exception(
                        "Detector initialization error: alias=%s type=%s class=%s protocol=%s",
                        alias,
                        det_type,
                        getattr(det_class, "__name__", str(det_class)),
                        protocol,
                    )

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
                        if stage_type == "Kinesis":
                            stage_protocol = "ctypes-dll"
                        elif stage_type == "Marlin":
                            stage_protocol = "serial"
                        elif stage_type == "DummyStage":
                            stage_protocol = "demo"
                        else:
                            stage_protocol = "direct"
                        logger.info(
                            "Stage init requested: alias=%s id=%s type=%s class=%s protocol=%s",
                            selected_stage.get("alias", "unknown"),
                            selected_stage.get("id", "unknown"),
                            stage_type,
                            stage_class.__name__,
                            stage_protocol,
                        )
                        stage_success = self.stage_controller.init_stage()
                        if stage_success:
                            print(f"✓ Stage '{selected_stage.get('alias')}' ({stage_type}) initialized successfully")
                            logger.info(
                                "Stage initialized: alias=%s class=%s protocol=%s",
                                selected_stage.get("alias", "unknown"),
                                self.stage_controller.__class__.__name__,
                                stage_protocol,
                            )
                        else:
                            print(f"✗ Stage '{selected_stage.get('alias')}' ({stage_type}) failed to initialize")
                            logger.error(
                                "Stage initialization failed: alias=%s type=%s class=%s protocol=%s",
                                selected_stage.get("alias", "unknown"),
                                stage_type,
                                stage_class.__name__,
                                stage_protocol,
                            )
                    except Exception as e:
                        print(f"✗ Error initializing stage '{selected_stage.get('alias')}' ({stage_type}): {e}")
                        logger.exception(
                            "Stage initialization error: alias=%s type=%s class=%s",
                            selected_stage.get("alias", "unknown"),
                            stage_type,
                            stage_class.__name__ if stage_class else "unknown",
                        )
                        stage_success = False
            else:
                print("⚠ No translation stage selected.")
                stage_success = False

        # Consider hardware initialized if at least one component succeeded
        self.hardware_initialized = stage_success or detector_success
        stage_summary = "skipped" if not init_stage else str(stage_success)
        detector_summary = "skipped" if not init_detector else str(detector_success)
        logger.info(
            "Hardware initialize summary: init_stage=%s init_detector=%s stage=%s detector=%s mode=%s active_detectors=%d",
            init_stage,
            init_detector,
            stage_summary,
            detector_summary,
            "demo" if dev_mode else "production",
            len(self.detectors),
        )
        return stage_success, detector_success

    def deinitialize(self):
        logger.info(
            "Hardware deinitialize requested: stage_present=%s detectors=%d",
            bool(self.stage_controller),
            len(self.detectors),
        )
        if self.stage_controller:
            try:
                self.stage_controller.deinit()
            except Exception as e:
                print(f"[Stage Deinit Error] {e}")
                logger.exception("Stage deinitialize error")
        for alias, detector in self.detectors.items():
            try:
                detector.deinit_detector()
            except Exception as e:
                print(f"[Detector '{alias}' Deinit Error] {e}")
                logger.exception("Detector deinitialize error: alias=%s", alias)
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
