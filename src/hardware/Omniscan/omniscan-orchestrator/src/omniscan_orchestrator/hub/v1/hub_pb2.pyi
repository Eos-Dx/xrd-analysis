import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ServerState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATE_UNSPECIFIED: _ClassVar[ServerState]
    IDLE: _ClassVar[ServerState]
    PENDING_ARMED: _ClassVar[ServerState]
    RUNNING: _ClassVar[ServerState]
    STOPPING: _ClassVar[ServerState]
    SAFE: _ClassVar[ServerState]

class DetectorStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DETECTOR_OFF: _ClassVar[DetectorStatus]
    DETECTOR_IDLE: _ClassVar[DetectorStatus]
    DETECTOR_EXPOSING: _ClassVar[DetectorStatus]
    DETECTOR_READING: _ClassVar[DetectorStatus]
    DETECTOR_ERROR: _ClassVar[DetectorStatus]

class MotionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MOTION_OFF: _ClassVar[MotionStatus]
    MOTION_IDLE: _ClassVar[MotionStatus]
    MOTION_MOVING: _ClassVar[MotionStatus]
    MOTION_HOMING: _ClassVar[MotionStatus]
    MOTION_ERROR: _ClassVar[MotionStatus]
    MOTION_LIMIT_HIT: _ClassVar[MotionStatus]
STATE_UNSPECIFIED: ServerState
IDLE: ServerState
PENDING_ARMED: ServerState
RUNNING: ServerState
STOPPING: ServerState
SAFE: ServerState
DETECTOR_OFF: DetectorStatus
DETECTOR_IDLE: DetectorStatus
DETECTOR_EXPOSING: DetectorStatus
DETECTOR_READING: DetectorStatus
DETECTOR_ERROR: DetectorStatus
MOTION_OFF: MotionStatus
MOTION_IDLE: MotionStatus
MOTION_MOVING: MotionStatus
MOTION_HOMING: MotionStatus
MOTION_ERROR: MotionStatus
MOTION_LIMIT_HIT: MotionStatus

class CommandContext(_message.Message):
    __slots__ = ("command_id", "user", "reason", "timestamp")
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    user: str
    reason: str
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, command_id: _Optional[str] = ..., user: _Optional[str] = ..., reason: _Optional[str] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InterlockStatus(_message.Message):
    __slots__ = ("emergency_stop", "door_closed", "radiation_safe", "cooling_ok", "power_ok", "overall_safe", "violation_reason", "enable_button", "key_switch")
    EMERGENCY_STOP_FIELD_NUMBER: _ClassVar[int]
    DOOR_CLOSED_FIELD_NUMBER: _ClassVar[int]
    RADIATION_SAFE_FIELD_NUMBER: _ClassVar[int]
    COOLING_OK_FIELD_NUMBER: _ClassVar[int]
    POWER_OK_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SAFE_FIELD_NUMBER: _ClassVar[int]
    VIOLATION_REASON_FIELD_NUMBER: _ClassVar[int]
    ENABLE_BUTTON_FIELD_NUMBER: _ClassVar[int]
    KEY_SWITCH_FIELD_NUMBER: _ClassVar[int]
    emergency_stop: bool
    door_closed: bool
    radiation_safe: bool
    cooling_ok: bool
    power_ok: bool
    overall_safe: bool
    violation_reason: str
    enable_button: bool
    key_switch: bool
    def __init__(self, emergency_stop: bool = ..., door_closed: bool = ..., radiation_safe: bool = ..., cooling_ok: bool = ..., power_ok: bool = ..., overall_safe: bool = ..., violation_reason: _Optional[str] = ..., enable_button: bool = ..., key_switch: bool = ...) -> None: ...

class DetectorHealth(_message.Message):
    __slots__ = ("powered", "temperature", "voltage", "status", "last_exposure_time", "total_exposures", "uptime_seconds")
    POWERED_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    VOLTAGE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LAST_EXPOSURE_TIME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EXPOSURES_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    powered: bool
    temperature: float
    voltage: float
    status: DetectorStatus
    last_exposure_time: int
    total_exposures: int
    uptime_seconds: int
    def __init__(self, powered: bool = ..., temperature: _Optional[float] = ..., voltage: _Optional[float] = ..., status: _Optional[_Union[DetectorStatus, str]] = ..., last_exposure_time: _Optional[int] = ..., total_exposures: _Optional[int] = ..., uptime_seconds: _Optional[int] = ...) -> None: ...

class MotionHealth(_message.Message):
    __slots__ = ("powered", "status", "position", "target_position", "is_homed", "total_moves", "uptime_seconds")
    POWERED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    TARGET_POSITION_FIELD_NUMBER: _ClassVar[int]
    IS_HOMED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MOVES_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    powered: bool
    status: MotionStatus
    position: float
    target_position: float
    is_homed: bool
    total_moves: int
    uptime_seconds: int
    def __init__(self, powered: bool = ..., status: _Optional[_Union[MotionStatus, str]] = ..., position: _Optional[float] = ..., target_position: _Optional[float] = ..., is_homed: bool = ..., total_moves: _Optional[int] = ..., uptime_seconds: _Optional[int] = ...) -> None: ...

class ExposureResult(_message.Message):
    __slots__ = ("exposure_time_ms", "timestamp", "data_size", "data_path", "detector_temp")
    EXPOSURE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DATA_SIZE_FIELD_NUMBER: _ClassVar[int]
    DATA_PATH_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_TEMP_FIELD_NUMBER: _ClassVar[int]
    exposure_time_ms: int
    timestamp: _timestamp_pb2.Timestamp
    data_size: int
    data_path: str
    detector_temp: float
    def __init__(self, exposure_time_ms: _Optional[int] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., data_size: _Optional[int] = ..., data_path: _Optional[str] = ..., detector_temp: _Optional[float] = ...) -> None: ...

class StartExposureRequest(_message.Message):
    __slots__ = ("ctx", "exposure_time_ms", "max_timeout_ms")
    CTX_FIELD_NUMBER: _ClassVar[int]
    EXPOSURE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    exposure_time_ms: int
    max_timeout_ms: int
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ..., exposure_time_ms: _Optional[int] = ..., max_timeout_ms: _Optional[int] = ...) -> None: ...

class StopRequest(_message.Message):
    __slots__ = ("ctx",)
    CTX_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ...) -> None: ...

class AbortRequest(_message.Message):
    __slots__ = ("ctx",)
    CTX_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ...) -> None: ...

class MoveToRequest(_message.Message):
    __slots__ = ("ctx", "position_mm")
    CTX_FIELD_NUMBER: _ClassVar[int]
    POSITION_MM_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    position_mm: float
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ..., position_mm: _Optional[float] = ...) -> None: ...

class MoveRelativeRequest(_message.Message):
    __slots__ = ("ctx", "distance_mm")
    CTX_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_MM_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    distance_mm: float
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ..., distance_mm: _Optional[float] = ...) -> None: ...

class HomeRequest(_message.Message):
    __slots__ = ("ctx",)
    CTX_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ...) -> None: ...

class SetVelocityRequest(_message.Message):
    __slots__ = ("ctx", "velocity_mm_s")
    CTX_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_MM_S_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    velocity_mm_s: float
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ..., velocity_mm_s: _Optional[float] = ...) -> None: ...

class PowerDeviceRequest(_message.Message):
    __slots__ = ("ctx", "device_type", "power_on")
    CTX_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TYPE_FIELD_NUMBER: _ClassVar[int]
    POWER_ON_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    device_type: str
    power_on: bool
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ..., device_type: _Optional[str] = ..., power_on: bool = ...) -> None: ...

class CalibrateDetectorRequest(_message.Message):
    __slots__ = ("ctx",)
    CTX_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ...) -> None: ...

class InitializeDetectorRequest(_message.Message):
    __slots__ = ("ctx",)
    CTX_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ...) -> None: ...

class InitializeMotionRequest(_message.Message):
    __slots__ = ("ctx",)
    CTX_FIELD_NUMBER: _ClassVar[int]
    ctx: CommandContext
    def __init__(self, ctx: _Optional[_Union[CommandContext, _Mapping]] = ...) -> None: ...

class GpioStateResponse(_message.Message):
    __slots__ = ("powered", "key_switch_on", "activation_button_active", "activation_remaining_secs", "interlocks", "main_led", "radiation_led")
    POWERED_FIELD_NUMBER: _ClassVar[int]
    KEY_SWITCH_ON_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_BUTTON_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_REMAINING_SECS_FIELD_NUMBER: _ClassVar[int]
    INTERLOCKS_FIELD_NUMBER: _ClassVar[int]
    MAIN_LED_FIELD_NUMBER: _ClassVar[int]
    RADIATION_LED_FIELD_NUMBER: _ClassVar[int]
    powered: bool
    key_switch_on: bool
    activation_button_active: bool
    activation_remaining_secs: int
    interlocks: InterlockStatus
    main_led: str
    radiation_led: str
    def __init__(self, powered: bool = ..., key_switch_on: bool = ..., activation_button_active: bool = ..., activation_remaining_secs: _Optional[int] = ..., interlocks: _Optional[_Union[InterlockStatus, _Mapping]] = ..., main_led: _Optional[str] = ..., radiation_led: _Optional[str] = ...) -> None: ...

class DetectorStateResponse(_message.Message):
    __slots__ = ("powered", "initialized", "status", "temperature", "total_exposures")
    POWERED_FIELD_NUMBER: _ClassVar[int]
    INITIALIZED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EXPOSURES_FIELD_NUMBER: _ClassVar[int]
    powered: bool
    initialized: bool
    status: DetectorStatus
    temperature: float
    total_exposures: int
    def __init__(self, powered: bool = ..., initialized: bool = ..., status: _Optional[_Union[DetectorStatus, str]] = ..., temperature: _Optional[float] = ..., total_exposures: _Optional[int] = ...) -> None: ...

class MotionStateResponse(_message.Message):
    __slots__ = ("powered", "initialized", "is_homed", "status", "position_x", "position_y", "total_moves")
    POWERED_FIELD_NUMBER: _ClassVar[int]
    INITIALIZED_FIELD_NUMBER: _ClassVar[int]
    IS_HOMED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    POSITION_X_FIELD_NUMBER: _ClassVar[int]
    POSITION_Y_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MOVES_FIELD_NUMBER: _ClassVar[int]
    powered: bool
    initialized: bool
    is_homed: bool
    status: MotionStatus
    position_x: float
    position_y: float
    total_moves: int
    def __init__(self, powered: bool = ..., initialized: bool = ..., is_homed: bool = ..., status: _Optional[_Union[MotionStatus, str]] = ..., position_x: _Optional[float] = ..., position_y: _Optional[float] = ..., total_moves: _Optional[int] = ...) -> None: ...

class FullServerStateResponse(_message.Message):
    __slots__ = ("safety_state", "gpio", "detector", "motion", "timestamp")
    SAFETY_STATE_FIELD_NUMBER: _ClassVar[int]
    GPIO_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_FIELD_NUMBER: _ClassVar[int]
    MOTION_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    safety_state: ServerState
    gpio: GpioStateResponse
    detector: DetectorStateResponse
    motion: MotionStateResponse
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, safety_state: _Optional[_Union[ServerState, str]] = ..., gpio: _Optional[_Union[GpioStateResponse, _Mapping]] = ..., detector: _Optional[_Union[DetectorStateResponse, _Mapping]] = ..., motion: _Optional[_Union[MotionStateResponse, _Mapping]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class StateChangeNotification(_message.Message):
    __slots__ = ("timestamp", "component", "change_type")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    CHANGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    timestamp: _timestamp_pb2.Timestamp
    component: str
    change_type: str
    def __init__(self, timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., component: _Optional[str] = ..., change_type: _Optional[str] = ...) -> None: ...

class GetStateResponse(_message.Message):
    __slots__ = ("state", "detail", "interlocks", "timestamp")
    STATE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    INTERLOCKS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    state: ServerState
    detail: str
    interlocks: InterlockStatus
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, state: _Optional[_Union[ServerState, str]] = ..., detail: _Optional[str] = ..., interlocks: _Optional[_Union[InterlockStatus, _Mapping]] = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetPositionResponse(_message.Message):
    __slots__ = ("position_mm", "is_homed")
    POSITION_MM_FIELD_NUMBER: _ClassVar[int]
    IS_HOMED_FIELD_NUMBER: _ClassVar[int]
    position_mm: float
    is_homed: bool
    def __init__(self, position_mm: _Optional[float] = ..., is_homed: bool = ...) -> None: ...

class GetExposureResultResponse(_message.Message):
    __slots__ = ("has_result", "result")
    HAS_RESULT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    has_result: bool
    result: ExposureResult
    def __init__(self, has_result: bool = ..., result: _Optional[_Union[ExposureResult, _Mapping]] = ...) -> None: ...

class HealthComponent(_message.Message):
    __slots__ = ("name", "ok", "detail", "last_check")
    NAME_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECK_FIELD_NUMBER: _ClassVar[int]
    name: str
    ok: bool
    detail: str
    last_check: _timestamp_pb2.Timestamp
    def __init__(self, name: _Optional[str] = ..., ok: bool = ..., detail: _Optional[str] = ..., last_check: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class AggregateHealth(_message.Message):
    __slots__ = ("ok", "components", "interlocks")
    OK_FIELD_NUMBER: _ClassVar[int]
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    INTERLOCKS_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    components: _containers.RepeatedCompositeFieldContainer[HealthComponent]
    interlocks: InterlockStatus
    def __init__(self, ok: bool = ..., components: _Optional[_Iterable[_Union[HealthComponent, _Mapping]]] = ..., interlocks: _Optional[_Union[InterlockStatus, _Mapping]] = ...) -> None: ...

class SystemEvent(_message.Message):
    __slots__ = ("timestamp", "state_changed", "interlock_event", "exposure_event", "motion_event", "error")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATE_CHANGED_FIELD_NUMBER: _ClassVar[int]
    INTERLOCK_EVENT_FIELD_NUMBER: _ClassVar[int]
    EXPOSURE_EVENT_FIELD_NUMBER: _ClassVar[int]
    MOTION_EVENT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    timestamp: _timestamp_pb2.Timestamp
    state_changed: StateChangedEvent
    interlock_event: InterlockEvent
    exposure_event: ExposureEvent
    motion_event: MotionEvent
    error: ErrorEvent
    def __init__(self, timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., state_changed: _Optional[_Union[StateChangedEvent, _Mapping]] = ..., interlock_event: _Optional[_Union[InterlockEvent, _Mapping]] = ..., exposure_event: _Optional[_Union[ExposureEvent, _Mapping]] = ..., motion_event: _Optional[_Union[MotionEvent, _Mapping]] = ..., error: _Optional[_Union[ErrorEvent, _Mapping]] = ...) -> None: ...

class StateChangedEvent(_message.Message):
    __slots__ = ("old_state", "new_state", "reason")
    OLD_STATE_FIELD_NUMBER: _ClassVar[int]
    NEW_STATE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    old_state: ServerState
    new_state: ServerState
    reason: str
    def __init__(self, old_state: _Optional[_Union[ServerState, str]] = ..., new_state: _Optional[_Union[ServerState, str]] = ..., reason: _Optional[str] = ...) -> None: ...

class InterlockEvent(_message.Message):
    __slots__ = ("status", "triggered_by")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TRIGGERED_BY_FIELD_NUMBER: _ClassVar[int]
    status: InterlockStatus
    triggered_by: str
    def __init__(self, status: _Optional[_Union[InterlockStatus, _Mapping]] = ..., triggered_by: _Optional[str] = ...) -> None: ...

class ExposureEvent(_message.Message):
    __slots__ = ("type", "exposure_time_ms")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STARTED: _ClassVar[ExposureEvent.Type]
        COMPLETED: _ClassVar[ExposureEvent.Type]
        STOPPED: _ClassVar[ExposureEvent.Type]
        FAILED: _ClassVar[ExposureEvent.Type]
    STARTED: ExposureEvent.Type
    COMPLETED: ExposureEvent.Type
    STOPPED: ExposureEvent.Type
    FAILED: ExposureEvent.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    EXPOSURE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    type: ExposureEvent.Type
    exposure_time_ms: int
    def __init__(self, type: _Optional[_Union[ExposureEvent.Type, str]] = ..., exposure_time_ms: _Optional[int] = ...) -> None: ...

class MotionEvent(_message.Message):
    __slots__ = ("type", "position_mm")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        MOVE_STARTED: _ClassVar[MotionEvent.Type]
        MOVE_COMPLETED: _ClassVar[MotionEvent.Type]
        HOME_COMPLETED: _ClassVar[MotionEvent.Type]
        STOPPED: _ClassVar[MotionEvent.Type]
        LIMIT_HIT: _ClassVar[MotionEvent.Type]
    MOVE_STARTED: MotionEvent.Type
    MOVE_COMPLETED: MotionEvent.Type
    HOME_COMPLETED: MotionEvent.Type
    STOPPED: MotionEvent.Type
    LIMIT_HIT: MotionEvent.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    POSITION_MM_FIELD_NUMBER: _ClassVar[int]
    type: MotionEvent.Type
    position_mm: float
    def __init__(self, type: _Optional[_Union[MotionEvent.Type, str]] = ..., position_mm: _Optional[float] = ...) -> None: ...

class ErrorEvent(_message.Message):
    __slots__ = ("component", "error_message", "error_code")
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    component: str
    error_message: str
    error_code: str
    def __init__(self, component: _Optional[str] = ..., error_message: _Optional[str] = ..., error_code: _Optional[str] = ...) -> None: ...
