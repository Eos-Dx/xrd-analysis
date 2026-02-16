from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from hardware.difra.hardware.hardware_control import HardwareController

_GENERATED_STUB_ROOT = Path(__file__).resolve().parent / "generated"
if str(_GENERATED_STUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENERATED_STUB_ROOT))

from hub.v1 import hub_pb2, hub_pb2_grpc

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _now_timestamp() -> Timestamp:
    now = datetime.now(timezone.utc)
    ts = Timestamp()
    ts.FromDatetime(now)
    return ts


def _default_context() -> hub_pb2.CommandContext:
    return hub_pb2.CommandContext(
        command_id=str(uuid.uuid4()),
        user="difra_gui",
        reason="",
        timestamp=_now_timestamp(),
        measurement_class=hub_pb2.SAMPLE,
    )


def _to_pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_") if part)


@dataclass
class CoreCommandDescriptor:
    command_id: str
    service_name: str
    command_name: str
    description: str
    request_fields: List[hub_pb2.FieldDescriptor]
    response_type: str
    safety_requirements: List[str]


from hardware.difra.grpc_server.difra_service_state import DifraServiceState
from hardware.difra.grpc_server.difra_services import (
    AcquisitionService,
    MotionService,
    DeviceInitializationService,
    CommandDiscoveryService,
    StateMonitorService,
)


class DifraGrpcServer:
    def __init__(self, config: Dict[str, Any], host: str = "127.0.0.1", port: int = 50061):
        self.config = config
        self.host = host
        self.port = port
        self.state = DifraServiceState(config)
        self.server = grpc.aio.server()

        hub_pb2_grpc.add_AcquisitionServicer_to_server(
            AcquisitionService(self.state), self.server
        )
        hub_pb2_grpc.add_MotionServicer_to_server(MotionService(self.state), self.server)
        hub_pb2_grpc.add_DeviceInitializationServicer_to_server(
            DeviceInitializationService(self.state), self.server
        )
        hub_pb2_grpc.add_CommandDiscoveryServicer_to_server(
            CommandDiscoveryService(self.state), self.server
        )
        hub_pb2_grpc.add_StateMonitorServicer_to_server(
            StateMonitorService(self.state), self.server
        )

        self.bound_port = self.server.add_insecure_port(f"{self.host}:{self.port}")
        if self.bound_port <= 0:
            raise RuntimeError(
                f"Failed to bind DiFRA gRPC server to {self.host}:{self.port}"
            )

    async def start(self) -> None:
        await self.server.start()

    async def stop(self, grace: float = 0.0) -> None:
        await self.server.stop(grace)


async def start_grpc_server(
    config: Dict[str, Any], host: str = "127.0.0.1", port: int = 50061
) -> DifraGrpcServer:
    server = DifraGrpcServer(config=config, host=host, port=port)
    await server.start()
    return server


def load_difra_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path:
        path = Path(config_path)
    else:
        config_dir = Path(__file__).resolve().parents[1] / "resources" / "config"
        main_name = "main_win.json" if os.name == "nt" else "main.json"
        candidate = config_dir / main_name
        path = candidate if candidate.exists() else config_dir / "main.json"

    if not path.exists():
        raise FileNotFoundError(f"DiFRA config not found: {path}")

    return json.loads(path.read_text())


async def _serve_forever(config_path: Optional[str], host: str, port: int) -> None:
    config = load_difra_config(config_path)
    server = await start_grpc_server(config=config, host=host, port=port)
    try:
        await server.server.wait_for_termination()
    finally:
        await server.stop(0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run DiFRA gRPC sidecar")
    parser.add_argument("--config", type=str, default=None, help="Path to DiFRA JSON config")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50061)
    args = parser.parse_args()

    asyncio.run(_serve_forever(args.config, args.host, args.port))


if __name__ == "__main__":
    main()
