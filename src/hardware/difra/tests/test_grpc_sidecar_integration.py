import asyncio
import os
import sys
import uuid

import grpc
import pytest
from google.protobuf.timestamp_pb2 import Timestamp

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.grpc_server.server import DifraGrpcServer, hub_pb2, hub_pb2_grpc


def _dummy_config():
    return {
        "DEV": True,
        "detectors": [
            {
                "alias": "PRIMARY",
                "type": "DummyDetector",
                "id": "DUMMY-DET-1",
                "size": {"width": 16, "height": 16},
            }
        ],
        "dev_active_detectors": ["DUMMY-DET-1"],
        "active_detectors": [],
        "translation_stages": [
            {
                "alias": "DUMMY_STAGE",
                "type": "DummyStage",
                "id": "DUMMY-STAGE-1",
                "settings": {
                    "limits_mm": {"x": [-14.0, 14.0], "y": [-14.0, 14.0]},
                    "home": [0.0, 0.0],
                    "load": [5.0, 5.0],
                },
            }
        ],
        "dev_active_stages": ["DUMMY-STAGE-1"],
        "active_translation_stages": [],
    }


def _ctx(
    reason: str,
    measurement_class: int = hub_pb2.SAMPLE,
) -> hub_pb2.CommandContext:
    ts = Timestamp()
    ts.GetCurrentTime()
    return hub_pb2.CommandContext(
        command_id=str(uuid.uuid4()),
        user="pytest",
        reason=reason,
        timestamp=ts,
        measurement_class=measurement_class,
    )


def test_sidecar_core8_and_discovery_flow():
    async def _scenario():
        server = DifraGrpcServer(config=_dummy_config(), host="127.0.0.1", port=0)
        await server.start()

        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
        await channel.channel_ready()

        init_stub = hub_pb2_grpc.DeviceInitializationStub(channel)
        motion_stub = hub_pb2_grpc.MotionStub(channel)
        acq_stub = hub_pb2_grpc.AcquisitionStub(channel)
        discovery_stub = hub_pb2_grpc.CommandDiscoveryStub(channel)
        state_stub = hub_pb2_grpc.StateMonitorStub(channel)

        before_readiness = await discovery_stub.GetCommandReadiness(hub_pb2.Empty())
        before_map = {
            (item.service_name, item.command_name): item for item in before_readiness.items
        }
        assert before_map[("Motion", "MoveTo")].ready is False

        init_motion = await init_stub.InitializeMotion(
            hub_pb2.InitializeMotionRequest(ctx=_ctx("init_motion"))
        )
        init_detector = await init_stub.InitializeDetector(
            hub_pb2.InitializeDetectorRequest(ctx=_ctx("init_detector"))
        )
        assert init_motion.initialized is True
        assert init_detector.initialized is True

        await motion_stub.MoveTo(
            hub_pb2.MoveToRequest(ctx=_ctx("move axis:x"), position_mm=3.0)
        )
        motion_state = await state_stub.GetMotionState(hub_pb2.Empty())
        assert motion_state.position_x == pytest.approx(3.0, abs=1e-3)

        await acq_stub.StartExposure(
            hub_pb2.StartExposureRequest(
                ctx=_ctx("start_exposure"),
                exposure_time_ms=1100,
                max_timeout_ms=5000,
            )
        )

        await asyncio.sleep(1.3)
        state = await acq_stub.GetState(hub_pb2.Empty())
        assert state.state in (hub_pb2.IDLE, hub_pb2.RUNNING)
        assert state.locks.device_locked is False
        assert state.locks.session_locked is False
        assert state.locks.technical_container_locked is False
        exposure = await acq_stub.GetLastExposureResult(hub_pb2.Empty())
        assert exposure.has_result is True
        assert bool(exposure.result.data_path) is True

        list_commands = await discovery_stub.ListCommands(hub_pb2.Empty())
        command_names = {
            (c.service_name, c.command_name)
            for c in list_commands.commands
        }
        expected_core8 = {
            ("DeviceInitialization", "InitializeDetector"),
            ("DeviceInitialization", "InitializeMotion"),
            ("Acquisition", "GetState"),
            ("Motion", "MoveTo"),
            ("Motion", "Home"),
            ("Acquisition", "StartExposure"),
            ("Acquisition", "Pause"),
            ("Acquisition", "Resume"),
            ("Acquisition", "Stop"),
            ("Acquisition", "Abort"),
        }
        assert expected_core8.issubset(command_names)

        compatibility = await discovery_stub.ValidateCompatibility(
            hub_pb2.ValidateCompatibilityRequest(
                client_version="1.0.0",
                client_protocol_version="1.1.0",
                required_commands=[
                    "Acquisition.GetState",
                    "Motion.MoveTo",
                    "DeviceInitialization.InitializeDetector",
                ],
            )
        )
        assert compatibility.compatible is True
        assert compatibility.protocol_compatible is True

        await channel.close()
        await server.stop(0)

    asyncio.run(_scenario())


def test_sidecar_reports_limit_violations():
    async def _scenario():
        server = DifraGrpcServer(config=_dummy_config(), host="127.0.0.1", port=0)
        await server.start()

        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
        await channel.channel_ready()

        init_stub = hub_pb2_grpc.DeviceInitializationStub(channel)
        motion_stub = hub_pb2_grpc.MotionStub(channel)

        await init_stub.InitializeMotion(hub_pb2.InitializeMotionRequest(ctx=_ctx("init_motion")))

        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await motion_stub.MoveTo(
                hub_pb2.MoveToRequest(ctx=_ctx("move_limit axis:x"), position_mm=999.0)
            )

        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

        await channel.close()
        await server.stop(0)

    asyncio.run(_scenario())


def test_sidecar_pause_resume_and_incident_stream():
    async def _scenario():
        server = DifraGrpcServer(config=_dummy_config(), host="127.0.0.1", port=0)
        await server.start()

        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
        await channel.channel_ready()

        init_stub = hub_pb2_grpc.DeviceInitializationStub(channel)
        acq_stub = hub_pb2_grpc.AcquisitionStub(channel)
        discovery_stub = hub_pb2_grpc.CommandDiscoveryStub(channel)
        state_stub = hub_pb2_grpc.StateMonitorStub(channel)

        await init_stub.InitializeDetector(
            hub_pb2.InitializeDetectorRequest(ctx=_ctx("init_detector"))
        )

        await acq_stub.StartExposure(
            hub_pb2.StartExposureRequest(
                ctx=_ctx("start_exposure_sample", measurement_class=hub_pb2.SAMPLE),
                exposure_time_ms=3500,
                max_timeout_ms=6000,
            )
        )
        await asyncio.sleep(0.4)

        await acq_stub.Pause(hub_pb2.PauseRequest(ctx=_ctx("pause")))
        paused_state = await acq_stub.GetState(hub_pb2.Empty())
        assert paused_state.state == hub_pb2.PAUSED

        readiness = await discovery_stub.GetCommandReadiness(hub_pb2.Empty())
        readiness_map = {
            (item.service_name, item.command_name): item for item in readiness.items
        }
        assert readiness_map[("Acquisition", "Resume")].ready is True

        await acq_stub.Resume(hub_pb2.ResumeRequest(ctx=_ctx("resume")))
        await asyncio.sleep(0.3)

        incident_call = state_stub.SubscribeIncidents(hub_pb2.Empty())
        incident_next = asyncio.create_task(incident_call.read())
        await asyncio.sleep(0.1)

        await server.state.set_device_locked(True)
        incident = await asyncio.wait_for(incident_next, timeout=2.0)

        assert incident.code == "DEVICE_LOCKED_DURING_MEASUREMENT"
        assert incident.severity == hub_pb2.HIGH
        assert incident.measurement_class == hub_pb2.SAMPLE

        locked_state = await acq_stub.GetState(hub_pb2.Empty())
        assert locked_state.locks.device_locked is True
        assert locked_state.state == hub_pb2.LOCKED

        await channel.close()
        await server.stop(0)

    asyncio.run(_scenario())


def test_motion_move_to_and_move_relative_support_axis_hints():
    async def _scenario():
        server = DifraGrpcServer(config=_dummy_config(), host="127.0.0.1", port=0)
        await server.start()
        try:
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
            await channel.channel_ready()
            init_stub = hub_pb2_grpc.DeviceInitializationStub(channel)
            motion_stub = hub_pb2_grpc.MotionStub(channel)
            state_stub = hub_pb2_grpc.StateMonitorStub(channel)

            init_motion = await init_stub.InitializeMotion(
                hub_pb2.InitializeMotionRequest(ctx=_ctx("init_motion"))
            )
            assert init_motion.initialized is True

            await motion_stub.MoveTo(
                hub_pb2.MoveToRequest(ctx=_ctx("move_to_x axis=1"), position_mm=2.5)
            )
            await motion_stub.MoveTo(
                hub_pb2.MoveToRequest(ctx=_ctx("move_to_y axis=2"), position_mm=-1.5)
            )
            await motion_stub.MoveRelative(
                hub_pb2.MoveRelativeRequest(
                    ctx=_ctx("move_relative_y axis:y"),
                    distance_mm=0.5,
                )
            )

            motion_state = await state_stub.GetMotionState(hub_pb2.Empty())
            assert motion_state.position_x == pytest.approx(2.5, abs=1e-3)
            assert motion_state.position_y == pytest.approx(-1.0, abs=1e-3)

            await channel.close()
        finally:
            await server.stop(0)

    asyncio.run(_scenario())


def test_motion_commands_require_axis_hint_in_context_reason():
    async def _scenario():
        server = DifraGrpcServer(config=_dummy_config(), host="127.0.0.1", port=0)
        await server.start()
        try:
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
            await channel.channel_ready()
            init_stub = hub_pb2_grpc.DeviceInitializationStub(channel)
            motion_stub = hub_pb2_grpc.MotionStub(channel)

            init_motion = await init_stub.InitializeMotion(
                hub_pb2.InitializeMotionRequest(ctx=_ctx("init_motion"))
            )
            assert init_motion.initialized is True

            with pytest.raises(grpc.aio.AioRpcError) as exc_move_to:
                await motion_stub.MoveTo(
                    hub_pb2.MoveToRequest(ctx=_ctx("missing_axis"), position_mm=1.0)
                )
            assert exc_move_to.value.code() == grpc.StatusCode.INVALID_ARGUMENT

            with pytest.raises(grpc.aio.AioRpcError) as exc_move_relative:
                await motion_stub.MoveRelative(
                    hub_pb2.MoveRelativeRequest(
                        ctx=_ctx("missing_axis"),
                        distance_mm=1.0,
                    )
                )
            assert exc_move_relative.value.code() == grpc.StatusCode.INVALID_ARGUMENT

            await channel.close()
        finally:
            await server.stop(0)

    asyncio.run(_scenario())
