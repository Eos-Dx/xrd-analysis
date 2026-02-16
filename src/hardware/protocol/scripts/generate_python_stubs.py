#!/usr/bin/env python3
"""Generate Python gRPC stubs from canonical hardware protocol."""

from __future__ import annotations

import subprocess
import sys
from importlib import resources
from pathlib import Path


def _ensure_init_files(base: Path, rel_pkg_path: Path) -> None:
    current = base
    for part in rel_pkg_path.parts:
        current = current / part
        current.mkdir(parents=True, exist_ok=True)
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")


def _run_protoc(proto_file: Path, include_dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        *(f"-I{inc}" for inc in include_dirs),
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    proto_file = repo_root / "src/hardware/protocol/hub/v1/hub.proto"

    try:
        grpc_include = Path(str(resources.files("grpc_tools") / "_proto"))
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"grpc_tools is required to generate stubs: {exc}", file=sys.stderr)
        return 2

    include_dirs = [repo_root / "src/hardware/protocol", grpc_include]

    # DiFRA sidecar stubs
    difra_out = repo_root / "src/hardware/difra/grpc_server/generated"
    _run_protoc(proto_file, include_dirs, difra_out)
    _ensure_init_files(difra_out, Path("hub/v1"))

    # Omniscan orchestrator client stubs (two historical import locations)
    orchestrator_root = (
        repo_root
        / "src/hardware/Omniscan/omniscan-orchestrator/src/omniscan_orchestrator"
    )
    _run_protoc(proto_file, include_dirs, orchestrator_root)
    _ensure_init_files(orchestrator_root, Path("hub/v1"))

    hub_generated = orchestrator_root / "hub" / "v1"
    proto_hub_generated = orchestrator_root / "proto" / "hub" / "v1"
    proto_hub_generated.mkdir(parents=True, exist_ok=True)
    _ensure_init_files(orchestrator_root, Path("proto/hub/v1"))
    for filename in ("hub_pb2.py", "hub_pb2_grpc.py"):
        (proto_hub_generated / filename).write_text(
            (hub_generated / filename).read_text()
        )

    # Keep legacy orchestrator proto copy synchronized.
    canonical_proto = proto_file.read_text()
    orchestrator_proto = (
        repo_root / "src/hardware/Omniscan/omniscan-orchestrator/proto/hub/v1/hub.proto"
    )
    orchestrator_proto.parent.mkdir(parents=True, exist_ok=True)
    orchestrator_proto.write_text(canonical_proto)

    # Keep legacy hardware-server proto copy synchronized.
    hw_proto = (
        repo_root / "src/hardware/Omniscan/omniscan-hw-server/proto/hub/v1/hub.proto"
    )
    hw_proto.parent.mkdir(parents=True, exist_ok=True)
    hw_proto.write_text(canonical_proto)

    print("Generated Python stubs from canonical protocol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
