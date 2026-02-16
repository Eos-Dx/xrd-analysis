# Hardware Protocol Source of Truth

This folder is the canonical protocol and command-schema source for:
- `src/hardware/Omniscan/omniscan-hw-server` (Rust gRPC server)
- `src/hardware/Omniscan/omniscan-orchestrator` (Python client/orchestrator)
- `src/hardware/difra/grpc_server` (Python sidecar server)

## Structure
- `hub/v1/hub.proto`: canonical gRPC protocol
- `commands/v1/*.toml`: command schema contracts used for discovery/readiness
- `scripts/`: protocol generation and sync utilities

## Core-8 Phase 1 Commands
- `DeviceInitialization.InitializeDetector`
- `DeviceInitialization.InitializeMotion`
- `Acquisition.GetState`
- `Motion.MoveTo`
- `Motion.Home`
- `Acquisition.StartExposure`
- `Acquisition.Stop`
- `Acquisition.Abort`
