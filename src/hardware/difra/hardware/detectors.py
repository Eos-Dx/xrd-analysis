"""Backward-compatible detector controller exports.

This module intentionally re-exports detector controller classes from their
smaller focused modules. Existing imports should continue to use:
`hardware.difra.hardware.detectors`.
"""

import socket

from hardware.difra.hardware.detector_controller_base import DetectorController
from hardware.difra.hardware.detector_dummy_controller import DummyDetectorController
from hardware.difra.hardware.detector_pixet_ctypes_controller import (
    PixetDetectorController,
)
from hardware.difra.hardware.detector_pixet_legacy_controller import (
    PixetLegacyDetectorController,
)
from hardware.difra.hardware.detector_pixet_sidecar_controller import (
    PixetSidecarDetectorController,
    PixetSidecarError,
)

__all__ = [
    "DetectorController",
    "DummyDetectorController",
    "PixetDetectorController",
    "PixetLegacyDetectorController",
    "PixetSidecarDetectorController",
    "PixetSidecarError",
    "socket",
]
