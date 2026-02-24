from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from hardware.difra.hardware.hardware_client_direct import DirectHardwareClient
from hardware.difra.hardware.hardware_client_dual import DualPathHardwareClient
from hardware.difra.hardware.hardware_client_grpc import (
    GrpcHardwareClient,
    grpc_import_error,
    grpc_runtime_available,
)
from hardware.difra.hardware.hardware_client_types import HardwareClient

LOGGER = logging.getLogger(__name__)


def create_hardware_client(config: Dict[str, Any]) -> HardwareClient:
    protocol_cfg = (config or {}).get("hardware_protocol", {})
    detector_backend = str(os.environ.get("DETECTOR_BACKEND", "")).lower().strip()
    if detector_backend not in {"sidecar", "socket", "ipc"}:
        LOGGER.warning(
            "Detector backend '%s' is not allowed; forcing sidecar.",
            detector_backend or "unset",
        )
        os.environ["DETECTOR_BACKEND"] = "sidecar"
        os.environ["PIXET_BACKEND"] = "sidecar"
        detector_backend = "sidecar"

    default_mode = "grpc"
    mode_override = str(
        os.environ.get("HARDWARE_CLIENT_MODE")
        or os.environ.get("DIFRA_HARDWARE_CLIENT_MODE")
        or ""
    ).lower().strip()
    requested_mode = str(mode_override or protocol_cfg.get("client_mode", default_mode)).lower().strip()
    if requested_mode != "grpc":
        LOGGER.warning(
            "Hardware client mode '%s' is not allowed; forcing grpc.",
            requested_mode or "unset",
        )
    mode = "grpc"

    sync_direct_detectors_cfg = protocol_cfg.get("sync_direct_detectors")
    if sync_direct_detectors_cfg is None:
        sync_direct_detectors = detector_backend in {"sidecar", "socket", "ipc"}
    else:
        sync_direct_detectors = bool(sync_direct_detectors_cfg)

    direct_client = DirectHardwareClient(config)

    grpc_client: Optional[GrpcHardwareClient] = None
    if mode in {"dual", "grpc"}:
        if not grpc_runtime_available():
            import_error = grpc_import_error()
            msg = "grpcio/protobuf stubs unavailable in this environment" + (
                f": {import_error}" if import_error else ""
            )
            if mode == "grpc":
                raise RuntimeError(msg)
            LOGGER.warning("%s; falling back to direct mode", msg)
            mode = "direct"
            return DualPathHardwareClient(
                direct_client=direct_client,
                grpc_client=None,
                mode=mode,
                sync_direct_detectors=sync_direct_detectors,
            )

        host = str(os.environ.get("DIFRA_GRPC_HOST") or protocol_cfg.get("grpc_host", "127.0.0.1"))
        port = int(os.environ.get("DIFRA_GRPC_PORT") or protocol_cfg.get("grpc_port", 50061))
        timeout_s = float(
            os.environ.get("DIFRA_GRPC_TIMEOUT_S")
            or protocol_cfg.get("grpc_timeout_s", 3.0)
        )
        user = str(
            os.environ.get("DIFRA_GRPC_USER") or protocol_cfg.get("grpc_user", "difra_gui")
        )

        try:
            grpc_client = GrpcHardwareClient(
                host=host,
                port=port,
                timeout_s=timeout_s,
                user=user,
            )
        except Exception as exc:
            if mode == "grpc":
                raise
            LOGGER.warning(
                "Failed to initialize gRPC hardware client, using direct mode: %s",
                exc,
            )
            mode = "direct"

    return DualPathHardwareClient(
        direct_client=direct_client,
        grpc_client=grpc_client,
        mode=mode,
        sync_direct_detectors=sync_direct_detectors,
    )
