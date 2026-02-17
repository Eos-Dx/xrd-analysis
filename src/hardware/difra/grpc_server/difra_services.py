"""DiFRA gRPC service implementations."""

from __future__ import annotations

from . import server as _module

grpc = _module.grpc
hub_pb2 = _module.hub_pb2
hub_pb2_grpc = _module.hub_pb2_grpc

_default_context = _module._default_context
_now_timestamp = _module._now_timestamp
DifraServiceState = _module.DifraServiceState


class AcquisitionService(hub_pb2_grpc.AcquisitionServicer):
    def __init__(self, state: DifraServiceState):
        self.state = state

    async def StartExposure(self, request, context):
        ctx = request.ctx if request.HasField("ctx") else _default_context()
        await self.state.emit_command_lifecycle(
            ctx.command_id,
            "Acquisition",
            "StartExposure",
            hub_pb2.COMMAND_STARTED,
            True,
        )
        try:
            await self.state.start_exposure(
                exposure_time_ms=request.exposure_time_ms,
                measurement_class=ctx.measurement_class,
            )
        except Exception as exc:
            await self.state.emit_command_lifecycle(
                ctx.command_id,
                "Acquisition",
                "StartExposure",
                hub_pb2.COMMAND_DONE,
                False,
                str(exc),
            )
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        await self.state.emit_command_lifecycle(
            ctx.command_id,
            "Acquisition",
            "StartExposure",
            hub_pb2.COMMAND_DONE,
            True,
        )
        return hub_pb2.Empty()

    async def Pause(self, request, context):
        ctx = request.ctx if request.HasField("ctx") else _default_context()
        try:
            await self.state.pause_exposure()
        except Exception as exc:
            await self.state.emit_command_lifecycle(
                ctx.command_id, "Acquisition", "Pause", hub_pb2.COMMAND_DONE, False, str(exc)
            )
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        await self.state.emit_command_lifecycle(
            ctx.command_id, "Acquisition", "Pause", hub_pb2.COMMAND_DONE, True
        )
        return hub_pb2.Empty()

    async def Resume(self, request, context):
        ctx = request.ctx if request.HasField("ctx") else _default_context()
        try:
            await self.state.resume_exposure()
        except Exception as exc:
            await self.state.emit_command_lifecycle(
                ctx.command_id, "Acquisition", "Resume", hub_pb2.COMMAND_DONE, False, str(exc)
            )
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        await self.state.emit_command_lifecycle(
            ctx.command_id, "Acquisition", "Resume", hub_pb2.COMMAND_DONE, True
        )
        return hub_pb2.Empty()

    async def Stop(self, request, context):
        ctx = request.ctx if request.HasField("ctx") else _default_context()
        try:
            await self.state.stop_exposure_graceful()
        except Exception as exc:
            await self.state.emit_command_lifecycle(
                ctx.command_id, "Acquisition", "Stop", hub_pb2.COMMAND_DONE, False, str(exc)
            )
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        await self.state.emit_command_lifecycle(
            ctx.command_id, "Acquisition", "Stop", hub_pb2.COMMAND_DONE, True
        )
        return hub_pb2.Empty()

    async def Abort(self, request, context):
        ctx = request.ctx if request.HasField("ctx") else _default_context()
        try:
            await self.state.abort_exposure()
        except Exception as exc:
            await self.state.emit_command_lifecycle(
                ctx.command_id, "Acquisition", "Abort", hub_pb2.COMMAND_DONE, False, str(exc)
            )
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        await self.state.emit_command_lifecycle(
            ctx.command_id, "Acquisition", "Abort", hub_pb2.COMMAND_DONE, True
        )
        return hub_pb2.Empty()

    async def GetState(self, request, context):
        interlocks = hub_pb2.InterlockStatus(
            emergency_stop=not self.state.device_locked,
            door_closed=True,
            radiation_safe=True,
            cooling_ok=True,
            power_ok=True,
            overall_safe=not self.state.device_locked,
            violation_reason="" if not self.state.device_locked else "device_locked",
            enable_button=not self.state.device_locked,
            key_switch=not self.state.device_locked,
        )
        return hub_pb2.GetStateResponse(
            state=self.state.server_state,
            detail="difra_sidecar",
            interlocks=interlocks,
            timestamp=_now_timestamp(),
            locks=self.state._locks_message(),
        )

    async def GetLastExposureResult(self, request, context):
        result = self.state.get_last_exposure_result()
        if result is None:
            return hub_pb2.GetExposureResultResponse(has_result=False)
        return hub_pb2.GetExposureResultResponse(has_result=True, result=result)

    async def StartBackgroundMeasurement(self, request, context):
        return hub_pb2.BackgroundMeasurementResponse(
            success=False,
            error_message="Not implemented in DiFRA sidecar",
        )

    async def SubscribeRunEvents(self, request, context):
        async for event in self.state.subscribe_run_events():
            yield event


class MotionService(hub_pb2_grpc.MotionServicer):
    def __init__(self, state: DifraServiceState):
        self.state = state

    @staticmethod
    def _axis_from_reason(reason: str):
        text = str(reason or "").lower()
        if (
            ("axis:x" in text)
            or ("axis=x" in text)
            or ("axis_x" in text)
            or ("axis=1" in text)
            or ("axis 1" in text)
            or (" x-axis" in text)
            or ("axis x" in text)
        ):
            return "x"
        if (
            ("axis:y" in text)
            or ("axis=y" in text)
            or ("axis_y" in text)
            or ("axis=2" in text)
            or ("axis 2" in text)
            or (" y-axis" in text)
            or ("axis y" in text)
        ):
            return "y"
        return None

    async def MoveTo(self, request, context):
        if not request.HasField("ctx"):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Protocol v1 MoveTo requires axis in ctx.reason",
            )
        axis = self._axis_from_reason(request.ctx.reason)
        if axis is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "MoveTo requires axis hint in ctx.reason: axis:x|axis:y|axis=1|axis=2",
            )
        try:
            if axis == "y":
                await self.state.move_to_y(request.position_mm)
            else:
                await self.state.move_to_x(request.position_mm)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return hub_pb2.Empty()

    async def MoveRelative(self, request, context):
        if not request.HasField("ctx"):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Protocol v1 MoveRelative requires axis in ctx.reason",
            )
        axis = self._axis_from_reason(request.ctx.reason)
        if axis is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "MoveRelative requires axis hint in ctx.reason: axis:x|axis:y|axis=1|axis=2",
            )
        try:
            x, y = await self.state.get_position()
            if axis == "y":
                await self.state.move_to_y(y + request.distance_mm)
            else:
                await self.state.move_to_x(x + request.distance_mm)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return hub_pb2.Empty()

    async def Home(self, request, context):
        try:
            await self.state.home()
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return hub_pb2.Empty()

    async def Stop(self, request, context):
        try:
            await self.state.abort_exposure()
        except Exception:
            pass
        return hub_pb2.Empty()

    async def SetVelocity(self, request, context):
        return hub_pb2.Empty()

    async def GetPosition(self, request, context):
        x, _ = await self.state.get_position()
        return hub_pb2.GetPositionResponse(position_mm=x, is_homed=self.state.stage_initialized)


class DeviceInitializationService(hub_pb2_grpc.DeviceInitializationServicer):
    def __init__(self, state: DifraServiceState):
        self.state = state

    async def InitializeDetector(self, request, context):
        _, detector_ok = await self.state.initialize_hardware(
            init_stage=False,
            init_detector=True,
        )
        return hub_pb2.DetectorStateResponse(
            powered=detector_ok,
            initialized=detector_ok,
            status=hub_pb2.DETECTOR_IDLE if detector_ok else hub_pb2.DETECTOR_ERROR,
            temperature=0.0,
            total_exposures=0,
        )

    async def InitializeMotion(self, request, context):
        stage_ok, _ = await self.state.initialize_hardware(
            init_stage=True,
            init_detector=False,
        )
        x, y = await self.state.get_position()
        return hub_pb2.MotionStateResponse(
            powered=stage_ok,
            initialized=stage_ok,
            is_homed=stage_ok,
            status=hub_pb2.MOTION_IDLE if stage_ok else hub_pb2.MOTION_ERROR,
            position_x=x,
            position_y=y,
            total_moves=0,
        )

    async def PowerOffDetector(self, request, context):
        await self.state.deinitialize_hardware()
        return hub_pb2.Empty()

    async def PowerOffMotion(self, request, context):
        await self.state.deinitialize_hardware()
        return hub_pb2.Empty()


class CommandDiscoveryService(hub_pb2_grpc.CommandDiscoveryServicer):
    def __init__(self, state: DifraServiceState):
        self.state = state

    async def GetServerCapabilities(self, request, context):
        return hub_pb2.GetServerCapabilitiesResponse(
            capabilities=hub_pb2.ServerCapabilities(
                server_version="0.2.0",
                protocol_version="1.2.0",
                build_time=_now_timestamp(),
                supported_features=[
                    "difra_sidecar",
                    "pause_resume",
                    "lock_status",
                    "incident_stream",
                ],
                device_type="difra_sidecar",
            )
        )

    async def ListCommands(self, request, context):
        return hub_pb2.ListCommandsResponse(
            commands=self.state.list_commands_response(),
            server_info=hub_pb2.ServerCapabilities(
                server_version="0.2.0",
                protocol_version="1.2.0",
                build_time=_now_timestamp(),
                supported_features=[
                    "difra_sidecar",
                    "pause_resume",
                    "lock_status",
                    "incident_stream",
                ],
                device_type="difra_sidecar",
            ),
        )

    async def ValidateCompatibility(self, request, context):
        available = {
            f"{d.service_name}.{d.command_name}" for d in self.state.list_commands_response()
        }
        missing = [cmd for cmd in request.required_commands if cmd not in available]
        protocol_ok = request.client_protocol_version.split(".")[0] == "1"
        compatible = protocol_ok and not missing

        message = "Compatible" if compatible else "Incompatible"
        if missing:
            message = f"Missing commands: {', '.join(missing)}"

        return hub_pb2.ValidateCompatibilityResponse(
            compatible=compatible,
            message=message,
            missing_commands=missing,
            version_warnings=[],
            protocol_compatible=protocol_ok,
        )

    async def GetCommandReadiness(self, request, context):
        return hub_pb2.GetCommandReadinessResponse(
            items=self.state.get_command_readiness_items()
        )


class StateMonitorService(hub_pb2_grpc.StateMonitorServicer):
    def __init__(self, state: DifraServiceState):
        self.state = state

    async def SubscribeToStateUpdates(self, request, context):
        async for notif in self.state.subscribe_state_updates():
            yield notif

    async def SubscribeIncidents(self, request, context):
        async for incident in self.state.subscribe_incidents():
            yield incident

    async def GetFullServerState(self, request, context):
        x, y = await self.state.get_position()
        return hub_pb2.FullServerStateResponse(
            safety_state=self.state.server_state,
            gpio=hub_pb2.GpioStateResponse(
                powered=True,
                key_switch_on=not self.state.device_locked,
                activation_button_active=not self.state.device_locked,
                interlocks=hub_pb2.InterlockStatus(
                    emergency_stop=not self.state.device_locked,
                    door_closed=True,
                    radiation_safe=True,
                    cooling_ok=True,
                    power_ok=True,
                    overall_safe=not self.state.device_locked,
                    violation_reason="" if not self.state.device_locked else "device_locked",
                    enable_button=not self.state.device_locked,
                    key_switch=not self.state.device_locked,
                ),
                main_led="Green" if not self.state.device_locked else "Red",
                radiation_led="Green" if not self.state.device_locked else "Red",
            ),
            detector=hub_pb2.DetectorStateResponse(
                powered=self.state.detector_initialized,
                initialized=self.state.detector_initialized,
                status=hub_pb2.DETECTOR_IDLE
                if self.state.detector_initialized
                else hub_pb2.DETECTOR_OFF,
                temperature=0.0,
                total_exposures=0,
            ),
            motion=hub_pb2.MotionStateResponse(
                powered=self.state.stage_initialized,
                initialized=self.state.stage_initialized,
                is_homed=self.state.stage_initialized,
                status=hub_pb2.MOTION_IDLE
                if self.state.stage_initialized
                else hub_pb2.MOTION_OFF,
                position_x=x,
                position_y=y,
                total_moves=0,
            ),
            timestamp=_now_timestamp(),
            locks=self.state._locks_message(),
        )

    async def GetGpioState(self, request, context):
        return (await self.GetFullServerState(request, context)).gpio

    async def GetDetectorState(self, request, context):
        return (await self.GetFullServerState(request, context)).detector

    async def GetMotionState(self, request, context):
        return (await self.GetFullServerState(request, context)).motion
