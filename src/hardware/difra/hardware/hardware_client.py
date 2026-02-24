"""Backward-compatible hardware client facade.

This module re-exports hardware client types and implementations from focused
modules to keep file sizes manageable while preserving existing import paths.
"""

from hardware.difra.hardware.hardware_client_axis import normalize_axis as _normalize_axis
from hardware.difra.hardware.hardware_client_direct import DirectHardwareClient
from hardware.difra.hardware.hardware_client_dual import DualPathHardwareClient
from hardware.difra.hardware.hardware_client_factory import create_hardware_client
from hardware.difra.hardware.hardware_client_grpc import (
    FALLBACK_GRPC_EXCEPTIONS as _FALLBACK_GRPC_EXCEPTIONS,
)
from hardware.difra.hardware.hardware_client_grpc import (
    GrpcHardwareClient,
    Timestamp,
    _command_context,
    _timestamp_now,
    grpc,
    hub_pb2,
    hub_pb2_grpc,
)
from hardware.difra.hardware.hardware_client_types import (
    CommandReadiness,
    HardwareClient,
)

__all__ = [
    "CommandReadiness",
    "HardwareClient",
    "DirectHardwareClient",
    "GrpcHardwareClient",
    "DualPathHardwareClient",
    "create_hardware_client",
]
