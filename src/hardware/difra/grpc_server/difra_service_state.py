"""DiFRA gRPC sidecar state model and command/readiness logic."""

from . import server as _module
import concurrent.futures
import tempfile

asyncio = _module.asyncio
functools = _module.functools
uuid = _module.uuid
Path = _module.Path
Any = _module.Any
Dict = _module.Dict
List = _module.List
Optional = _module.Optional
Tuple = _module.Tuple

hub_pb2 = _module.hub_pb2
HardwareController = _module.HardwareController
tomllib = _module.tomllib

_now_timestamp = _module._now_timestamp
_to_pascal_case = _module._to_pascal_case
CoreCommandDescriptor = _module.CoreCommandDescriptor


class DifraServiceState:
    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        self.config = config
        self._config_path = str(config_path) if config_path else None
        self.hardware_controller = HardwareController(config)
        self.stage_controller = None
        self.detector_controllers: Dict[str, Any] = {}

        self.stage_initialized = False
        self.detector_initialized = False
        self.server_state = hub_pb2.SAFE

        self.device_locked = bool(config.get("device_locked", False))
        self.session_locked = bool(config.get("session_locked", False))
        self.technical_container_locked = bool(
            config.get("technical_container_locked", False)
        )

        self._lock = asyncio.Lock()
        self._run_stream_subscribers: List[asyncio.Queue] = []
        self._state_stream_subscribers: List[asyncio.Queue] = []
        self._incident_stream_subscribers: List[asyncio.Queue] = []

        self._active_run_id: Optional[str] = None
        self._active_run_task: Optional[asyncio.Task] = None
        self._active_run_stop_requested = False
        self._active_run_abort_requested = False
        self._active_run_paused = False
        self._active_run_total_seconds = 0
        self._active_run_elapsed_seconds = 0
        self._active_measurement_class = hub_pb2.MEASUREMENT_CLASS_UNSPECIFIED
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._last_exposure_result: Optional[hub_pb2.ExposureResult] = None

        self._command_descriptors = self._load_command_descriptors()

    def _reload_config_from_disk(self) -> None:
        if not self._config_path:
            return
        try:
            refreshed = _module.load_difra_config(self._config_path)
        except Exception as exc:
            print(f"[WARN] Failed to reload gRPC config from {self._config_path}: {exc}")
            return

        self.config = refreshed
        self.hardware_controller = HardwareController(refreshed)
        self.stage_controller = None
        self.detector_controllers = {}
        self.stage_initialized = False
        self.detector_initialized = False

    def _protocol_command_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "protocol" / "commands" / "v1"

    def _load_command_descriptors(self) -> List[CoreCommandDescriptor]:
        command_dir = self._protocol_command_dir()
        if not command_dir.exists():
            raise FileNotFoundError(
                f"Canonical command schema directory does not exist: {command_dir}"
            )
        descriptors: List[CoreCommandDescriptor] = []

        response_map = {
            ("Acquisition", "GetState"): "GetStateResponse",
            ("DeviceInitialization", "InitializeDetector"): "DetectorStateResponse",
            ("DeviceInitialization", "InitializeMotion"): "MotionStateResponse",
            ("Acquisition", "StartBackgroundMeasurement"): "BackgroundMeasurementResponse",
            ("Acquisition", "GetLastExposureResult"): "GetExposureResultResponse",
        }

        for command_file in sorted(command_dir.glob("*.toml")):
            with command_file.open("rb") as f:
                data = tomllib.load(f)

            command_id = str(data["command_id"])
            service_name = str(data["service"])
            command_name = _to_pascal_case(command_id)

            fields: List[hub_pb2.FieldDescriptor] = []
            if not (service_name == "Acquisition" and command_name == "GetState"):
                fields.append(
                    hub_pb2.FieldDescriptor(
                        name="ctx",
                        type="CommandContext",
                        required=True,
                        description="Command context for audit trail",
                    )
                )

            for param in data.get("parameters", []):
                param_type = param.get("param_type", {}).get("type", "String")
                proto_type = {
                    "Float": "double",
                    "Integer": "int64",
                    "Boolean": "bool",
                    "Enum": "string",
                }.get(param_type, "string")
                fields.append(
                    hub_pb2.FieldDescriptor(
                        name=str(param["name"]),
                        type=proto_type,
                        required=bool(param.get("required", False)),
                        description=str(param.get("description", "")),
                        default_value=str(param.get("default_value", ""))
                        if param.get("default_value") is not None
                        else "",
                    )
                )

            safety = data.get("safety_requirements", {})
            safety_tokens: List[str] = []
            if safety.get("requires_key_switch"):
                safety_tokens.append("key_switch_on")
            if safety.get("requires_enable_button"):
                safety_tokens.append("activation_button")
            for check in safety.get("interlock_checks", []):
                name = check.get("name")
                if name:
                    safety_tokens.append(f"interlock:{name}")

            response_type = response_map.get((service_name, command_name), "Empty")
            descriptors.append(
                CoreCommandDescriptor(
                    command_id=command_id,
                    service_name=service_name,
                    command_name=command_name,
                    description=str(data.get("description", "")),
                    request_fields=fields,
                    response_type=response_type,
                    safety_requirements=safety_tokens,
                )
            )

        return descriptors

    def _locks_message(self) -> hub_pb2.LockStatus:
        return hub_pb2.LockStatus(
            device_locked=self.device_locked,
            session_locked=self.session_locked,
            technical_container_locked=self.technical_container_locked,
        )

    async def _emit_state_change(self, old_state: int, new_state: int, reason: str) -> None:
        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                state_changed=hub_pb2.StateChangedEvent(
                    old_state=old_state,
                    new_state=new_state,
                    reason=reason,
                ),
            )
        )
        await self.emit_state_change("server_state", reason)

    async def _set_state(self, new_state: int, reason: str) -> None:
        if self.server_state == new_state:
            return
        old_state = self.server_state
        self.server_state = new_state
        await self._emit_state_change(old_state, new_state, reason)

    def _is_run_active(self) -> bool:
        return self._active_run_task is not None and not self._active_run_task.done()

    def _guard_mutating_command(self) -> None:
        if self.device_locked:
            raise RuntimeError("DEVICE_LOCKED: hardware key lock active")

    async def initialize_hardware(
        self,
        init_stage: bool = True,
        init_detector: bool = True,
    ) -> Tuple[bool, bool]:
        self._guard_mutating_command()
        async with self._lock:
            if not self.stage_initialized and not self.detector_initialized:
                self._reload_config_from_disk()
            stage_ok, detector_ok = await asyncio.to_thread(
                self.hardware_controller.initialize,
                init_stage=init_stage,
                init_detector=init_detector,
            )
            if init_stage:
                self.stage_controller = self.hardware_controller.stage_controller
                self.stage_initialized = bool(stage_ok)
            if init_detector:
                self.detector_controllers = dict(self.hardware_controller.detectors)
                self.detector_initialized = bool(detector_ok)
            await self._set_state(
                hub_pb2.IDLE if (self.stage_initialized or self.detector_initialized) else hub_pb2.SAFE,
                "initialize_hardware",
            )
            return self.stage_initialized, self.detector_initialized

    async def deinitialize_hardware(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self.hardware_controller.deinitialize)
            self.stage_controller = None
            self.detector_controllers = {}
            self.stage_initialized = False
            self.detector_initialized = False
            await self._set_state(hub_pb2.SAFE, "deinitialize_hardware")

    async def get_position(self) -> Tuple[float, float]:
        if not self.stage_controller:
            return 0.0, 0.0
        return await asyncio.to_thread(self.stage_controller.get_xy_position)

    async def _move_to_axis_position(
        self,
        *,
        target_x: float,
        target_y: float,
        old_axis_position: float,
        new_axis_position: float,
        run_type: str,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        self._guard_mutating_command()
        if not self.stage_controller:
            raise RuntimeError("Motion stage is not initialized")

        run_id = str(uuid.uuid4())
        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                run_started=hub_pb2.RunStartedEvent(
                    run_id=run_id,
                    run_type=run_type,
                    total_seconds=1,
                ),
            )
        )

        move_call = functools.partial(
            self.stage_controller.move_stage,
            target_x,
            target_y,
            move_timeout=timeout_s,
        )
        move_future = asyncio.create_task(asyncio.to_thread(move_call))

        for tick in range(10):
            if move_future.done():
                break
            # Keep loop latency low so short exposures complete close to requested time.
            await asyncio.sleep(0.02)
            frac = (tick + 1) / 10.0
            interpolated = old_axis_position + (
                (new_axis_position - old_axis_position) * frac
            )
            await self.emit_system_event(
                hub_pb2.SystemEvent(
                    timestamp=_now_timestamp(),
                    run_progress=hub_pb2.RunProgressEvent(
                        run_id=run_id,
                        run_type=run_type,
                        elapsed_seconds=0,
                        total_seconds=1,
                        position_mm=interpolated,
                    ),
                )
            )

        new_x, new_y = await move_future
        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                run_completed=hub_pb2.RunCompletedEvent(
                    run_id=run_id,
                    run_type=run_type,
                    status="completed",
                    reason="",
                ),
            )
        )
        return float(new_x), float(new_y)

    async def move_to_x(self, x_mm: float, timeout_s: float = 25.0) -> Tuple[float, float]:
        old_x, old_y = await self.get_position()
        return await self._move_to_axis_position(
            target_x=float(x_mm),
            target_y=float(old_y),
            old_axis_position=float(old_x),
            new_axis_position=float(x_mm),
            run_type="motion_x",
            timeout_s=timeout_s,
        )

    async def move_to_y(self, y_mm: float, timeout_s: float = 25.0) -> Tuple[float, float]:
        old_x, old_y = await self.get_position()
        return await self._move_to_axis_position(
            target_x=float(old_x),
            target_y=float(y_mm),
            old_axis_position=float(old_y),
            new_axis_position=float(y_mm),
            run_type="motion_y",
            timeout_s=timeout_s,
        )

    async def home(self, timeout_s: float = 25.0) -> Tuple[float, float]:
        self._guard_mutating_command()
        if not self.stage_controller:
            raise RuntimeError("Motion stage is not initialized")

        run_id = str(uuid.uuid4())
        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                run_started=hub_pb2.RunStartedEvent(
                    run_id=run_id,
                    run_type="motion_home",
                    total_seconds=1,
                ),
            )
        )

        home_call = functools.partial(self.stage_controller.home_stage, timeout_s=timeout_s)
        new_x, new_y = await asyncio.to_thread(home_call)

        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                run_completed=hub_pb2.RunCompletedEvent(
                    run_id=run_id,
                    run_type="motion_home",
                    status="completed",
                    reason="",
                ),
            )
        )
        return float(new_x), float(new_y)

    async def _emit_measurement_incident_for_lock(self) -> None:
        severity = hub_pb2.MODERATE
        if self._active_measurement_class == hub_pb2.TECHNICAL:
            severity = hub_pb2.LOW
        elif self._active_measurement_class == hub_pb2.SAMPLE:
            severity = hub_pb2.HIGH
        elif self._active_measurement_class == hub_pb2.PATIENT:
            severity = hub_pb2.CRITICAL

        await self.emit_incident(
            code="DEVICE_LOCKED_DURING_MEASUREMENT",
            severity=severity,
            message="Hardware key lock activated during measurement",
            measurement_class=self._active_measurement_class,
            run_id=self._active_run_id,
            metadata={
                "device_locked": "true",
                "phase": "measurement",
            },
        )

    async def set_device_locked(self, locked: bool) -> None:
        async with self._lock:
            self.device_locked = bool(locked)
            if self.device_locked:
                await self._set_state(hub_pb2.LOCKED, "device_locked")
                if self._is_run_active():
                    self._active_run_abort_requested = True
                    self._pause_gate.set()
                    await self._emit_measurement_incident_for_lock()
            else:
                target = hub_pb2.IDLE if (self.stage_initialized or self.detector_initialized) else hub_pb2.SAFE
                await self._set_state(target, "device_unlocked")

    async def start_exposure(
        self,
        exposure_time_ms: int,
        measurement_class: int,
    ) -> str:
        self._guard_mutating_command()
        async with self._lock:
            if self._is_run_active():
                raise RuntimeError("Exposure already running")
            if not self.detector_initialized:
                raise RuntimeError("Detector is not initialized")
            if self.session_locked:
                raise RuntimeError("SESSION_LOCKED: create/select new session container first")

            run_id = str(uuid.uuid4())
            self._active_run_id = run_id
            self._active_measurement_class = measurement_class
            self._active_run_stop_requested = False
            self._active_run_abort_requested = False
            self._active_run_paused = False
            self._active_run_total_seconds = max(1, int(round(exposure_time_ms / 1000.0)))
            self._active_run_elapsed_seconds = 0
            self._pause_gate.set()
            self._last_exposure_result = None

            await self._set_state(hub_pb2.PENDING_ARMED, "start_exposure_queued")
            self._active_run_task = asyncio.create_task(
                self._run_exposure(run_id, exposure_time_ms)
            )
            return run_id

    def _capture_detectors(
        self,
        run_id: str,
        exposure_time_ms: int,
    ) -> Dict[str, str]:
        if not self.detector_controllers:
            raise RuntimeError("No initialized detector controllers")

        exposure_s = max(float(exposure_time_ms) / 1000.0, 0.001)
        base_root = (
            self.config.get("measurements_folder")
            or self.config.get("difra_base_folder")
            or tempfile.gettempdir()
        )
        # Keep gRPC raw outputs in the main measurements folder so the GUI
        # does not maintain a parallel "grpc_exposures" subtree.
        capture_root = Path(base_root)
        capture_root.mkdir(parents=True, exist_ok=True)

        def _capture_single(alias: str, controller: Any) -> Tuple[str, str]:
            alias_tag = str(alias).replace(" ", "_")
            filename_base = capture_root / f"{run_id}_{alias_tag}"
            ok = bool(
                controller.capture_point(
                    Nframes=1,
                    Nseconds=exposure_s,
                    filename_base=str(filename_base),
                )
            )
            if not ok:
                raise RuntimeError(f"Detector '{alias}' capture failed")

            txt_path = filename_base.with_suffix(".txt")
            if txt_path.exists():
                return alias, str(txt_path)

            fallback = sorted(capture_root.glob(f"{filename_base.name}.*"))
            if not fallback:
                raise RuntimeError(
                    f"Detector '{alias}' produced no output for base '{filename_base}'"
                )
            return alias, str(fallback[0])

        outputs: Dict[str, str] = {}
        max_workers = max(1, len(self.detector_controllers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_capture_single, alias, controller)
                for alias, controller in self.detector_controllers.items()
            ]
            for fut in concurrent.futures.as_completed(futures):
                alias, path = fut.result()
                outputs[alias] = path
        return outputs

    async def pause_exposure(self) -> None:
        self._guard_mutating_command()
        async with self._lock:
            if not self._is_run_active():
                raise RuntimeError("No active exposure")
            if self._active_run_paused:
                return
            self._active_run_paused = True
            self._pause_gate.clear()
            await self._set_state(hub_pb2.PAUSED, "pause_requested")
            await self.emit_system_event(
                hub_pb2.SystemEvent(
                    timestamp=_now_timestamp(),
                    exposure_event=hub_pb2.ExposureEvent(
                        type=hub_pb2.ExposureEvent.PAUSED,
                        exposure_time_ms=self._active_run_total_seconds * 1000,
                    ),
                )
            )

    async def resume_exposure(self) -> None:
        self._guard_mutating_command()
        async with self._lock:
            if not self._is_run_active():
                raise RuntimeError("No active exposure")
            if not self._active_run_paused:
                raise RuntimeError("Exposure is not paused")
            self._active_run_paused = False
            self._pause_gate.set()
            await self._set_state(hub_pb2.RUNNING, "resume_requested")
            await self.emit_system_event(
                hub_pb2.SystemEvent(
                    timestamp=_now_timestamp(),
                    exposure_event=hub_pb2.ExposureEvent(
                        type=hub_pb2.ExposureEvent.RESUMED,
                        exposure_time_ms=self._active_run_total_seconds * 1000,
                    ),
                )
            )

    async def stop_exposure_graceful(self) -> None:
        self._guard_mutating_command()
        async with self._lock:
            if not self._is_run_active():
                raise RuntimeError("No active exposure")
            self._active_run_stop_requested = True
            self._active_run_paused = False
            self._pause_gate.set()
            await self._set_state(hub_pb2.STOPPING, "stop_requested")

    async def abort_exposure(self) -> None:
        async with self._lock:
            if not self._is_run_active():
                raise RuntimeError("No active exposure")
            self._active_run_abort_requested = True
            self._active_run_paused = False
            self._pause_gate.set()
            await self._set_state(hub_pb2.STOPPING, "abort_requested")

    async def _run_exposure(self, run_id: str, exposure_time_ms: int) -> None:
        total_seconds = max(1, int(round(exposure_time_ms / 1000.0)))
        await self._set_state(hub_pb2.RUNNING, "run_started")

        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                run_started=hub_pb2.RunStartedEvent(
                    run_id=run_id,
                    run_type="measurement",
                    total_seconds=total_seconds,
                ),
            )
        )

        status = "completed"
        reason = ""
        captured_files: Dict[str, str] = {}
        capture_future = asyncio.create_task(
            asyncio.to_thread(self._capture_detectors, run_id, exposure_time_ms)
        )
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        last_elapsed = -1

        while not capture_future.done():
            if self._active_run_paused:
                await self._pause_gate.wait()
                continue

            await asyncio.sleep(0.1)
            elapsed = min(int(loop.time() - started_at), total_seconds)
            if elapsed > last_elapsed:
                last_elapsed = elapsed
                self._active_run_elapsed_seconds = elapsed
                await self.emit_system_event(
                    hub_pb2.SystemEvent(
                        timestamp=_now_timestamp(),
                        run_progress=hub_pb2.RunProgressEvent(
                            run_id=run_id,
                            run_type="measurement",
                            elapsed_seconds=elapsed,
                            total_seconds=total_seconds,
                        ),
                    )
                )

        try:
            captured_files = await capture_future
        except Exception as exc:
            status = "failed"
            reason = str(exc)

        if self._active_run_abort_requested:
            status = "interrupted"
            reason = "abort_requested"
        elif self._active_run_stop_requested:
            status = "stopped"
            reason = "stop_requested"
        elif status == "completed":
            # proto v1 has a single data_path, so expose the first detector output.
            first_path = next(iter(captured_files.values()), "")
            data_size = 0
            if first_path:
                try:
                    data_size = int(Path(first_path).stat().st_size)
                except Exception:
                    data_size = 0
            self._last_exposure_result = hub_pb2.ExposureResult(
                exposure_time_ms=int(exposure_time_ms),
                timestamp=_now_timestamp(),
                data_size=max(data_size, 0),
                data_path=str(first_path),
                detector_temp=0.0,
            )

        async with self._lock:
            self._active_run_id = None
            self._active_run_task = None
            self._active_run_stop_requested = False
            self._active_run_abort_requested = False
            self._active_run_paused = False
            self._active_run_total_seconds = 0
            self._active_run_elapsed_seconds = 0
            self._active_measurement_class = hub_pb2.MEASUREMENT_CLASS_UNSPECIFIED
            self._pause_gate.set()
            target = hub_pb2.LOCKED if self.device_locked else hub_pb2.IDLE
            await self._set_state(target, "run_finished")

        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                run_completed=hub_pb2.RunCompletedEvent(
                    run_id=run_id,
                    run_type="measurement",
                    status=status,
                    reason=reason,
                ),
            )
        )

    def get_last_exposure_result(self) -> Optional[hub_pb2.ExposureResult]:
        return self._last_exposure_result

    async def emit_command_lifecycle(
        self,
        command_id: str,
        service_name: str,
        command_name: str,
        phase: int,
        success: bool,
        reason: str = "",
    ) -> None:
        await self.emit_system_event(
            hub_pb2.SystemEvent(
                timestamp=_now_timestamp(),
                command_lifecycle=hub_pb2.CommandLifecycleEvent(
                    command_id=command_id,
                    service_name=service_name,
                    command_name=command_name,
                    phase=phase,
                    success=success,
                    reason=reason,
                ),
            )
        )

    async def emit_system_event(self, event: hub_pb2.SystemEvent) -> None:
        for queue in list(self._run_stream_subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def emit_incident(
        self,
        code: str,
        severity: int,
        message: str,
        measurement_class: int = hub_pb2.MEASUREMENT_CLASS_UNSPECIFIED,
        run_id: Optional[str] = None,
        point_index: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        incident = hub_pb2.IncidentEvent(
            incident_id=str(uuid.uuid4()),
            code=code,
            severity=severity,
            message=message,
            timestamp=_now_timestamp(),
            measurement_class=measurement_class,
            metadata=metadata or {},
        )
        if run_id is not None:
            incident.run_id = run_id
        if point_index is not None:
            incident.point_index = point_index

        for queue in list(self._incident_stream_subscribers):
            try:
                queue.put_nowait(incident)
            except asyncio.QueueFull:
                pass

    async def emit_state_change(self, component: str, change_type: str) -> None:
        notif = hub_pb2.StateChangeNotification(
            timestamp=_now_timestamp(), component=component, change_type=change_type
        )
        for queue in list(self._state_stream_subscribers):
            try:
                queue.put_nowait(notif)
            except asyncio.QueueFull:
                pass

    async def subscribe_run_events(self):
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._run_stream_subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            if queue in self._run_stream_subscribers:
                self._run_stream_subscribers.remove(queue)

    async def subscribe_state_updates(self):
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._state_stream_subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            if queue in self._state_stream_subscribers:
                self._state_stream_subscribers.remove(queue)

    async def subscribe_incidents(self):
        queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._incident_stream_subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            if queue in self._incident_stream_subscribers:
                self._incident_stream_subscribers.remove(queue)

    def get_command_readiness_items(self) -> List[hub_pb2.CommandReadiness]:
        running = self._is_run_active()
        paused = self.server_state == hub_pb2.PAUSED

        lock_reasons: List[str] = []
        if self.device_locked:
            lock_reasons.append("DEVICE_LOCKED: hardware key lock active")

        def with_lock(base_ready: bool, reasons: List[str]) -> Tuple[bool, List[str]]:
            if lock_reasons:
                return False, lock_reasons + reasons
            return base_ready, reasons

        readiness = {
            ("DeviceInitialization", "InitializeDetector"): with_lock(
                not self.detector_initialized,
                [] if not self.detector_initialized else ["Detector already initialized"],
            ),
            ("DeviceInitialization", "InitializeMotion"): with_lock(
                not self.stage_initialized,
                [] if not self.stage_initialized else ["Motion already initialized"],
            ),
            ("Acquisition", "GetState"): (True, []),
            ("Motion", "MoveTo"): with_lock(
                self.stage_initialized and not running and not paused,
                []
                if (self.stage_initialized and not running and not paused)
                else ["Motion stage is not initialized or acquisition is active"],
            ),
            ("Motion", "Home"): with_lock(
                self.stage_initialized and not running and not paused,
                []
                if (self.stage_initialized and not running and not paused)
                else ["Motion stage is not initialized or acquisition is active"],
            ),
            ("Acquisition", "StartExposure"): with_lock(
                self.detector_initialized and not running and not paused and not self.session_locked,
                []
                if (self.detector_initialized and not running and not paused and not self.session_locked)
                else [
                    "Detector is not initialized"
                    if not self.detector_initialized
                    else (
                        "SESSION_LOCKED: create/select new session container first"
                        if self.session_locked
                        else "Exposure already running"
                    )
                ],
            ),
            ("Acquisition", "Pause"): with_lock(
                running and not paused,
                [] if (running and not paused) else ["No active running exposure"],
            ),
            ("Acquisition", "Resume"): with_lock(
                running and paused,
                [] if (running and paused) else ["Exposure is not paused"],
            ),
            ("Acquisition", "Stop"): with_lock(
                running or paused,
                [] if (running or paused) else ["No active exposure"],
            ),
            ("Acquisition", "Abort"): (
                running or paused,
                [] if (running or paused) else ["No active exposure"],
            ),
        }

        items: List[hub_pb2.CommandReadiness] = []
        for desc in self._command_descriptors:
            ready, reasons = readiness.get((desc.service_name, desc.command_name), (True, []))
            items.append(
                hub_pb2.CommandReadiness(
                    service_name=desc.service_name,
                    command_name=desc.command_name,
                    ready=ready,
                    reasons=reasons,
                )
            )
        return items

    def list_commands_response(self) -> List[hub_pb2.CommandDescriptor]:
        return [
            hub_pb2.CommandDescriptor(
                service_name=desc.service_name,
                command_name=desc.command_name,
                description=desc.description,
                request_fields=desc.request_fields,
                response_type=desc.response_type,
                response_fields=[],
                required_permissions=[],
                safety_requirements=desc.safety_requirements,
            )
            for desc in self._command_descriptors
        ]
