"""
gRPC Client for Omniscan Hardware Server

Connects to the Rust gRPC server and provides high-level methods
for orchestrator operations.
"""

from __future__ import annotations

import grpc
import time
from typing import Any, Dict, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import uuid

from .hub.v1 import hub_pb2, hub_pb2_grpc

if TYPE_CHECKING:
    from .database import OrchestratorDatabase


class OmniscanGrpcClient:
    """gRPC client for Omniscan hardware server."""
    
    def __init__(
        self,
        server_address: str = "localhost:50051",
        database: Optional['OrchestratorDatabase'] = None,
        session_id: Optional[str] = None,
        client_cert_path: Optional[str] = None,
        client_key_path: Optional[str] = None,
        ca_cert_path: Optional[str] = None,
    ):
        self.server_address = server_address
        self.db = database
        self.session_id = session_id or str(uuid.uuid4())
        
        # Set up channel with mTLS if certificates provided
        if client_cert_path and client_key_path and ca_cert_path:
            # Read certificate files
            with open(ca_cert_path, 'rb') as f:
                ca_cert = f.read()
            with open(client_cert_path, 'rb') as f:
                client_cert = f.read()
            with open(client_key_path, 'rb') as f:
                client_key = f.read()
                
            # Create SSL credentials
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert,
            )
            self.channel = grpc.secure_channel(server_address, credentials)
        else:
            # Insecure channel for development
            self.channel = grpc.insecure_channel(server_address)
            
        # Create service stubs
        self.acquisition = hub_pb2_grpc.AcquisitionStub(self.channel)
        self.motion = hub_pb2_grpc.MotionStub(self.channel)
        self.device_control = hub_pb2_grpc.DeviceControlStub(self.channel)
        self.device_init = hub_pb2_grpc.DeviceInitializationStub(self.channel)
        self.state_monitor = hub_pb2_grpc.StateMonitorStub(self.channel)
        self.safety = hub_pb2_grpc.SafetyStub(self.channel)
        self.health = hub_pb2_grpc.HealthStub(self.channel)
        # Try to create CommandDiscovery stub (may not exist in older servers)
        try:
            self.command_discovery = hub_pb2_grpc.CommandDiscoveryStub(self.channel)
        except AttributeError:
            self.command_discovery = None
        
    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    # Helper method to create command context
    def _create_context(self, user: str, reason: str = "") -> hub_pb2.CommandContext:
        """Create a command context for audit trail."""
        return hub_pb2.CommandContext(
            command_id=str(uuid.uuid4()),
            user=user,
            reason=reason,
            timestamp=hub_pb2.google_dot_protobuf_dot_timestamp__pb2.Timestamp(
                seconds=int(datetime.now(timezone.utc).timestamp())
            )
        )
    
    def _log_command(
        self,
        command_type: str,
        hw_command_id: str,
        user: str,
        command_data: Dict[str, Any],
        start_time: float,
        result: str,
        error_message: Optional[str] = None
    ):
        """Helper to log command execution to database if available.
        
        Args:
            command_type: Service.Command format (e.g., "Acquisition.StartExposure")
            hw_command_id: Command ID sent to hw-server
            user: Operator/user ID
            command_data: Command parameters as dict
            start_time: time.time() when command started
            result: "success" or "failure"
            error_message: Error details if result is "failure"
        """
        if not self.db:
            return
        
        try:
            self.db.log_command(
                command_id=str(uuid.uuid4()),  # Orchestrator's own command ID
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                command_type=command_type,
                command_data=command_data,
                hw_server_command_id=hw_command_id,
                user_context=user,
                result=result,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_message
            )
        except Exception as e:
            # Don't let logging errors break the operation
            print(f"Warning: Failed to log command {command_type}: {e}")
        
    # Health and Status
    def get_aggregate_health(self) -> Dict[str, Any]:
        """Call Health/GetAggregateHealth and return full JSON-friendly dict."""
        try:
            resp = self.health.GetAggregateHealth(hub_pb2.Empty())
            # Convert timestamps to ISO8601 where available
            def ts_to_iso(ts):
                if ts and getattr(ts, 'seconds', None) is not None:
                    return datetime.fromtimestamp(ts.seconds, tz=timezone.utc).isoformat()
                return None
            return {
                "ok": resp.ok,
                "components": [
                    {
                        "name": c.name,
                        "ok": c.ok,
                        "detail": c.detail,
                        "last_check": ts_to_iso(c.last_check),
                    } for c in resp.components
                ],
                "interlocks": {
                    "emergency_stop": resp.interlocks.emergency_stop if resp.HasField('interlocks') else None,
                    "door_closed": resp.interlocks.door_closed if resp.HasField('interlocks') else None,
                    "radiation_safe": resp.interlocks.radiation_safe if resp.HasField('interlocks') else None,
                    "cooling_ok": resp.interlocks.cooling_ok if resp.HasField('interlocks') else None,
                    "power_ok": resp.interlocks.power_ok if resp.HasField('interlocks') else None,
                    "overall_safe": resp.interlocks.overall_safe if resp.HasField('interlocks') else None,
                    "violation_reason": resp.interlocks.violation_reason if resp.HasField('interlocks') else None,
                    "enable_button": resp.interlocks.enable_button if resp.HasField('interlocks') else None,
                    "key_switch": resp.interlocks.key_switch if resp.HasField('interlocks') else None,
                } if resp.HasField('interlocks') else None,
            }
        except grpc.RpcError as e:
            return {"error": str(e), "code": str(e.code()), "details": e.details()}

    def get_interlocks(self) -> Dict[str, Any]:
        """Call Safety/GetInterlockStatus."""
        try:
            r = self.safety.GetInterlockStatus(hub_pb2.Empty())
            return {
                "emergency_stop": r.emergency_stop,
                "door_closed": r.door_closed,
                "radiation_safe": r.radiation_safe,
                "cooling_ok": r.cooling_ok,
                "power_ok": r.power_ok,
                "overall_safe": r.overall_safe,
                "violation_reason": r.violation_reason,
                "enable_button": r.enable_button,
                "key_switch": r.key_switch,
            }
        except grpc.RpcError as e:
            return {"error": str(e)}

    def get_server_state(self) -> Dict[str, Any]:
        """Call Acquisition/GetState."""
        try:
            r = self.acquisition.GetState(hub_pb2.Empty())
            return {
                "state": hub_pb2.ServerState.Name(r.state),
                "detail": r.detail,
                "interlocks": {
                    "emergency_stop": r.interlocks.emergency_stop if r.HasField('interlocks') else None,
                    "door_closed": r.interlocks.door_closed if r.HasField('interlocks') else None,
                    "radiation_safe": r.interlocks.radiation_safe if r.HasField('interlocks') else None,
                    "cooling_ok": r.interlocks.cooling_ok if r.HasField('interlocks') else None,
                    "power_ok": r.interlocks.power_ok if r.HasField('interlocks') else None,
                    "overall_safe": r.interlocks.overall_safe if r.HasField('interlocks') else None,
                    "violation_reason": r.interlocks.violation_reason if r.HasField('interlocks') else None,
                    "enable_button": r.interlocks.enable_button if r.HasField('interlocks') else None,
                    "key_switch": r.interlocks.key_switch if r.HasField('interlocks') else None,
                } if r.HasField('interlocks') else None,
            }
        except grpc.RpcError as e:
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Legacy convenience: returns aggregate health, interlocks, and state in one dict."""
        try:
            health_response = self.health.GetAggregateHealth(hub_pb2.Empty())
            interlock_response = self.safety.GetInterlockStatus(hub_pb2.Empty())
            state_response = self.acquisition.GetState(hub_pb2.Empty())
            return {
                "ok": health_response.ok,
                "state": hub_pb2.ServerState.Name(state_response.state),
                "interlocks": {
                    "overall_safe": interlock_response.overall_safe,
                    "emergency_stop": interlock_response.emergency_stop,
                    "door_closed": interlock_response.door_closed,
                    "radiation_safe": interlock_response.radiation_safe,
                    "cooling_ok": interlock_response.cooling_ok,
                    "power_ok": interlock_response.power_ok,
                },
                "components": [
                    {
                        "name": comp.name,
                        "ok": comp.ok,
                        "detail": comp.detail
                    } for comp in health_response.components
                ]
            }
        except grpc.RpcError as e:
            return {
                "error": str(e),
                "code": e.code(),
                "details": e.details()
            }
            
    # Device Control

    def get_detector_health(self) -> Dict[str, Any]:
        """Wrapper around DeviceControl/GetDetectorHealth."""
        try:
            d = self.device_control.GetDetectorHealth(hub_pb2.Empty())
            return {
                "powered": d.powered,
                "temperature": d.temperature,
                "voltage": d.voltage,
                "status": hub_pb2.DetectorStatus.Name(d.status),
                "last_exposure_time": d.last_exposure_time,
                "total_exposures": d.total_exposures,
                "uptime_seconds": d.uptime_seconds,
            }
        except grpc.RpcError as e:
            return {"error": str(e)}

    def get_motion_health(self) -> Dict[str, Any]:
        """Wrapper around DeviceControl/GetMotionHealth."""
        try:
            m = self.device_control.GetMotionHealth(hub_pb2.Empty())
            return {
                "powered": m.powered,
                "status": hub_pb2.MotionStatus.Name(m.status),
                "position": m.position if m.HasField('position') else None,
                "target_position": m.target_position if m.HasField('target_position') else None,
                "is_homed": m.is_homed,
                "total_moves": m.total_moves,
                "uptime_seconds": m.uptime_seconds,
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
    def get_device_state(self) -> Dict[str, Any]:
        """Get device state (detector and motion health)."""
        try:
            detector_health = self.device_control.GetDetectorHealth(hub_pb2.Empty())
            motion_health = self.device_control.GetMotionHealth(hub_pb2.Empty())
            
            return {
                "detector": {
                    "powered": detector_health.powered,
                    "temperature": detector_health.temperature,
                    "voltage": detector_health.voltage,
                    "status": hub_pb2.DetectorStatus.Name(detector_health.status),
                    "total_exposures": detector_health.total_exposures,
                },
                "motion": {
                    "powered": motion_health.powered,
                    "status": hub_pb2.MotionStatus.Name(motion_health.status),
                    "position": motion_health.position if motion_health.HasField("position") else None,
                    "is_homed": motion_health.is_homed,
                    "total_moves": motion_health.total_moves,
                }
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
    
    def get_device_state_general(self, device_type: str) -> Dict[str, Any]:
        """Get general device state using unified GetDeviceState RPC.
        
        Args:
            device_type: One of "pdu", "gpio", "detector", "motion"
        
        Returns dict with:
        - device_type: str
        - powered: bool
        - status: str (e.g., "Active", "Off", "IDLE", "INIT")
        - uptime_seconds: int
        - outputs: dict (e.g., PDU: {"main_power": true})
        """
        try:
            request = hub_pb2.DeviceStateRequest(device_type=device_type)
            response = self.device_control.GetDeviceState(request)
            
            return {
                "device_type": response.device_type,
                "powered": response.powered,
                "status": response.status,
                "uptime_seconds": response.uptime_seconds,
                "outputs": dict(response.outputs) if response.outputs else {},
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
    
    def get_gpio_state(self) -> Dict[str, Any]:
        """Get full GPIO state including buttons, interlocks, and LEDs.
        
        Returns dict with:
        - powered: bool
        - key_switch_on: bool
        - activation_button_active: bool
        - activation_remaining_secs: Optional[int]
        - interlocks: dict
        - main_led: str
        - radiation_led: str
        """
        try:
            gpio_response = self.state_monitor.GetGpioState(hub_pb2.Empty())
            
            return {
                "powered": gpio_response.powered,
                "key_switch_on": gpio_response.key_switch_on,
                "activation_button_active": gpio_response.activation_button_active,
                "activation_remaining_secs": gpio_response.activation_remaining_secs if gpio_response.HasField("activation_remaining_secs") else None,
                "interlocks": {
                    "emergency_stop": gpio_response.interlocks.emergency_stop,
                    "door_closed": gpio_response.interlocks.door_closed,
                    "radiation_safe": gpio_response.interlocks.radiation_safe,
                    "cooling_ok": gpio_response.interlocks.cooling_ok,
                    "power_ok": gpio_response.interlocks.power_ok,
                    "overall_safe": gpio_response.interlocks.overall_safe,
                    "enable_button": gpio_response.interlocks.enable_button,
                    "key_switch": gpio_response.interlocks.key_switch,
                },
                "main_led": gpio_response.main_led,
                "radiation_led": gpio_response.radiation_led,
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
    
    def get_gpio_button_states(self) -> Dict[str, bool]:
        """Get GPIO button states (key switch and enable button).
        
        Convenience method that calls get_gpio_state and extracts button info.
        Returns dict with 'key_switch' and 'enable_button' booleans.
        """
        gpio_state = self.get_gpio_state()
        if "error" in gpio_state:
            return {"key_switch": False, "enable_button": False}
        
        return {
            "key_switch": gpio_state.get("key_switch_on", False),
            "enable_button": gpio_state.get("activation_button_active", False),
        }
    
    def check_enable_button(self) -> Dict[str, Any]:
        """Check if enable button is active.
        
        Returns:
            dict with 'active': bool, 'remaining_secs': Optional[int], 'error': Optional[str]
        """
        gpio_state = self.get_gpio_state()
        if "error" in gpio_state:
            return {"active": False, "error": gpio_state["error"]}
        
        return {
            "active": gpio_state.get("activation_button_active", False),
            "remaining_secs": gpio_state.get("activation_remaining_secs"),
        }
    
    def initialize_detector(self, user: str = "unknown") -> Dict[str, Any]:
        """Initialize detector (requires enable button active).
        
        Preconditions (per safety spec):
        - Key switch ON
        - Activation/Enable button ACTIVE (within 20s window)
        - Radiation SAFE (beam physically blocked)
        - Cooling OK
        - Power OK
        - Door state: ANY
        
        Returns:
            dict with detector state or error
        """
        start_time = time.time()
        ctx = self._create_context(user, "Initialize detector for imaging")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            # Check enable/activation button first
            button_check = self.check_enable_button()
            if not button_check.get("active", False):
                error_msg = "Enable/Activation button not active - required for detector initialization"
                self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {
                    "error": error_msg,
                    "detail": "Click ACTIVATE/ENABLE button and retry within 20 seconds"
                }

            # Check required interlocks and key switch
            gpio_state = self.get_gpio_state()
            if "error" in gpio_state:
                error_msg = f"Failed to read GPIO state: {gpio_state['error']}"
                self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}

            if not gpio_state.get("key_switch_on", False):
                error_msg = "Key switch must be ON for detector initialization"
                self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}

            interlocks = gpio_state.get("interlocks", {})
            if not interlocks.get("radiation_safe", False):
                error_msg = "Radiation is NOT SAFE (beam not blocked) - block beam before initializing detector"
                self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            if not interlocks.get("cooling_ok", False):
                error_msg = "Cooling is NOT OK - check cooling system before initializing detector"
                self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            if not interlocks.get("power_ok", False):
                error_msg = "Power is NOT OK - check power before initializing detector"
                self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            # Door can be any state for initialization per spec
            
            request = hub_pb2.InitializeDetectorRequest(ctx=ctx)
            response = self.device_init.InitializeDetector(request)
            
            self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "success")
            
            return {
                "status": "initialized",
                "powered": response.powered,
                "initialized": response.initialized,
                "detector_status": hub_pb2.DetectorStatus.Name(response.status),
                "temperature": response.temperature,
            }
        except grpc.RpcError as e:
            self._log_command("DeviceInitialization.InitializeDetector", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e), "code": e.code().name if hasattr(e, "code") else "UNKNOWN"}
    
    def initialize_motion(self, user: str = "unknown") -> Dict[str, Any]:
        """Initialize motion system (requires enable button active).
        
        Preconditions (per safety spec):
        - Key switch ON
        - Activation/Enable button ACTIVE (within 20s window)
        - Radiation SAFE (beam physically blocked)
        - Cooling OK
        - Power OK
        - Door state: ANY
        
        Returns:
            dict with motion state or error
        """
        start_time = time.time()
        ctx = self._create_context(user, "Initialize motion system")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            # Check enable/activation button first
            button_check = self.check_enable_button()
            if not button_check.get("active", False):
                error_msg = "Enable/Activation button not active - required for motion initialization"
                self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {
                    "error": error_msg,
                    "detail": "Click ACTIVATE/ENABLE button and retry within 20 seconds"
                }

            # Check required interlocks and key switch
            gpio_state = self.get_gpio_state()
            if "error" in gpio_state:
                error_msg = f"Failed to read GPIO state: {gpio_state['error']}"
                self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}

            if not gpio_state.get("key_switch_on", False):
                error_msg = "Key switch must be ON for motion initialization"
                self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}

            interlocks = gpio_state.get("interlocks", {})
            if not interlocks.get("radiation_safe", False):
                error_msg = "Radiation is NOT SAFE (beam not blocked) - block beam before initializing motion"
                self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            if not interlocks.get("cooling_ok", False):
                error_msg = "Cooling is NOT OK - check cooling system before initializing motion"
                self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            if not interlocks.get("power_ok", False):
                error_msg = "Power is NOT OK - check power before initializing motion"
                self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            # Door can be any state for initialization per spec
            
            request = hub_pb2.InitializeMotionRequest(ctx=ctx)
            response = self.device_init.InitializeMotion(request)
            
            self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "success")
            
            return {
                "status": "initialized",
                "powered": response.powered,
                "initialized": response.initialized,
                "is_homed": response.is_homed,
                "motion_status": hub_pb2.MotionStatus.Name(response.status),
            }
        except grpc.RpcError as e:
            self._log_command("DeviceInitialization.InitializeMotion", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e), "code": e.code().name if hasattr(e, "code") else "UNKNOWN"}
    
    def power_off_detector(self, user: str = "unknown") -> Dict[str, Any]:
        """Power off detector (safe operation, no enable button required)."""
        start_time = time.time()
        ctx = self._create_context(user, "Power off detector")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            self.device_init.PowerOffDetector(ctx)
            
            self._log_command("DeviceInitialization.PowerOffDetector", hw_command_id, user, command_data, start_time, "success")
            return {"status": "powered_off"}
        except grpc.RpcError as e:
            self._log_command("DeviceInitialization.PowerOffDetector", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e)}
    
    def power_off_motion(self, user: str = "unknown") -> Dict[str, Any]:
        """Power off motion system (safe operation, no enable button required)."""
        start_time = time.time()
        ctx = self._create_context(user, "Power off motion")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            self.device_init.PowerOffMotion(ctx)
            
            self._log_command("DeviceInitialization.PowerOffMotion", hw_command_id, user, command_data, start_time, "success")
            return {"status": "powered_off"}
        except grpc.RpcError as e:
            self._log_command("DeviceInitialization.PowerOffMotion", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e)}
            
    def power_device(self, device_type: str, power_on: bool, user: str = "unknown") -> Dict[str, Any]:
        """Control device power (DEPRECATED - use initialize_* or power_off_* methods).
        
        Note: This method is deprecated. For powering on devices, use:
        - initialize_detector() for detector
        - initialize_motion() for motion
        For powering off, use:
        - power_off_detector()
        - power_off_motion()
        """
        if power_on:
            if device_type == "detector":
                return self.initialize_detector(user)
            elif device_type == "motion":
                return self.initialize_motion(user)
            else:
                return {"error": f"Unknown device type: {device_type}"}
        else:
            if device_type == "detector":
                return self.power_off_detector(user)
            elif device_type == "motion":
                return self.power_off_motion(user)
            else:
                return {"error": f"Unknown device type: {device_type}"}
            
    # Measurement Operations
    def start_measurement(
        self,
        exposure_time_ms: int,
        user: str = "unknown",
        max_timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Start a measurement/exposure (requires enable button active).
        
        This is a HARMFUL operation that requires:
        - Key switch ON
        - Enable button ACTIVE (within 20s window)
        - All interlocks SAFE
        """
        start_time = time.time()
        ctx = self._create_context(user, "Start measurement")
        hw_command_id = ctx.command_id
        command_data = {
            "exposure_time_ms": exposure_time_ms,
            "user": user,
            "max_timeout_ms": max_timeout_ms or (exposure_time_ms + 5000)
        }
        
        try:
            # Check enable button first
            button_check = self.check_enable_button()
            if not button_check.get("active", False):
                error_msg = "Enable button not active - required to start X-ray exposure"
                self._log_command("Acquisition.StartExposure", hw_command_id, user, command_data, start_time, "failure", error_msg)
                return {
                    "error": error_msg,
                    "detail": "Click ENABLE button in GPIO panel and retry within 20 seconds"
                }
            
            request = hub_pb2.StartExposureRequest(
                ctx=ctx,
                exposure_time_ms=exposure_time_ms,
                max_timeout_ms=max_timeout_ms or (exposure_time_ms + 5000)
            )
            self.acquisition.StartExposure(request)
            
            # Log successful command
            self._log_command("Acquisition.StartExposure", hw_command_id, user, command_data, start_time, "success")
            
            return {
                "status": "started",
                "exposure_time_ms": exposure_time_ms,
                "measurement_id": hw_command_id  # Return for measurement_id linkage
            }
        except grpc.RpcError as e:
            self._log_command("Acquisition.StartExposure", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e)}
            
    def stop_measurement(self, user: str = "unknown") -> Dict[str, Any]:
        """Stop current measurement (safe operation, no enable button required)."""
        start_time = time.time()
        ctx = self._create_context(user, "Stop measurement")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            request = hub_pb2.StopRequest(ctx=ctx)
            self.acquisition.Stop(request)
            
            self._log_command("Acquisition.Stop", hw_command_id, user, command_data, start_time, "success")
            return {"status": "stopped"}
        except grpc.RpcError as e:
            self._log_command("Acquisition.Stop", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e)}

    def stop_motion(self, user: str = "unknown") -> Dict[str, Any]:
        """Stop any ongoing motion immediately (safe operation, no enable button required)."""
        start_time = time.time()
        ctx = self._create_context(user, "Stop motion")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            request = hub_pb2.StopRequest(ctx=ctx)
            self.motion.Stop(request)
            
            self._log_command("Motion.Stop", hw_command_id, user, command_data, start_time, "success")
            return {"status": "stopped"}
        except grpc.RpcError as e:
            self._log_command("Motion.Stop", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"error": str(e)}
            
    def get_measurement_status(self) -> Dict[str, Any]:
        """Get current measurement status."""
        try:
            state_response = self.acquisition.GetState(hub_pb2.Empty())
            return {
                "state": hub_pb2.ServerState.Name(state_response.state),
                "detail": state_response.detail,
                "timestamp": state_response.timestamp.seconds
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
            
    def get_measurement_result(self) -> Dict[str, Any]:
        """Get last measurement result."""
        try:
            result_response = self.acquisition.GetLastExposureResult(hub_pb2.Empty())
            
            if not result_response.has_result:
                return {"has_result": False}
                
            result = result_response.result
            return {
                "has_result": True,
                "exposure_time_ms": result.exposure_time_ms,
                "data_size": result.data_size,
                "data_path": result.data_path if result.HasField("data_path") else None,
                "detector_temp": result.detector_temp,
                "timestamp": result.timestamp.seconds
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
            
    def calibrate_detector(self, user: str = "unknown") -> Dict[str, Any]:
        """Calibrate the detector and return QC report."""
        start_time = time.time()
        ctx = self._create_context(user, "Calibrate detector")
        hw_command_id = ctx.command_id
        command_data = {"user": user}
        
        try:
            request = hub_pb2.CalibrateDetectorRequest(ctx=ctx)
            response = self.acquisition.CalibrateDetector(request)
            
            # Check if response has success field, otherwise assume old proto
            if hasattr(response, 'success'):
                if not response.success:
                    error_msg = response.error_message if hasattr(response, 'error_message') else "Calibration failed"
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                # Extract QC report if present
                if response.HasField('qc_report'):
                    qc = response.qc_report
                    
                    return {
                        "success": True,
                        "calibration_id": str(uuid.uuid4()),
                        "timestamp": datetime.fromtimestamp(qc.timestamp.seconds).isoformat(),
                        "calibrant_material": qc.calibrant_material,
                        "overall_pass": qc.overall_pass,
                        "qc_checks": {
                            "total_intensity": {
                                "passed": qc.total_intensity_check.passed,
                                "measured": qc.total_intensity_check.measured_value,
                                "threshold": qc.total_intensity_check.threshold,
                                "details": qc.total_intensity_check.details
                            },
                            "goodness_of_fit": {
                                "passed": qc.goodness_check.passed,
                                "measured": qc.goodness_check.measured_value,
                                "threshold": qc.goodness_check.threshold,
                                "details": qc.goodness_check.details
                            },
                            "snr": {
                                "passed": qc.snr_check.passed,
                                "measured": qc.snr_check.measured_value,
                                "threshold": qc.snr_check.threshold,
                                "details": qc.snr_check.details
                            },
                            "ring_quality": {
                                "passed": qc.ring_quality_check.passed,
                                "measured": qc.ring_quality_check.measured_value,
                                "threshold": qc.ring_quality_check.threshold,
                                "details": qc.ring_quality_check.details
                            },
                            "poni": {
                                "success": qc.poni_result.success,
                                "distance_mm": qc.poni_result.distance_mm if qc.poni_result.HasField("distance_mm") else None,
                                "beam_center_x": qc.poni_result.beam_center_x if qc.poni_result.HasField("beam_center_x") else None,
                                "beam_center_y": qc.poni_result.beam_center_y if qc.poni_result.HasField("beam_center_y") else None,
                                "wavelength_angstrom": qc.poni_result.wavelength_angstrom if qc.poni_result.HasField("wavelength_angstrom") else None
                            }
                        },
                        "formatted_report": qc.formatted_report
                    }
            
            # Fallback for old proto or missing qc_report
            self._log_command("Acquisition.CalibrateDetector", hw_command_id, user, command_data, start_time, "success")
            return {"success": True, "status": "calibrated"}
        except grpc.RpcError as e:
            self._log_command("Acquisition.CalibrateDetector", hw_command_id, user, command_data, start_time, "failure", str(e))
            return {"success": False, "error": str(e)}
    
    def get_last_calibration(self) -> Dict[str, Any]:
        """Get the last calibration QC report."""
        try:
            response = self.acquisition.GetLastCalibration(hub_pb2.Empty())
            
            if not response.has_calibration:
                return {
                    "has_calibration": False
                }
            
            qc = response.qc_report
            
            return {
                "has_calibration": True,
                "timestamp": datetime.fromtimestamp(qc.timestamp.seconds).isoformat(),
                "calibrant_material": qc.calibrant_material,
                "overall_pass": qc.overall_pass,
                "qc_checks": {
                    "total_intensity": {
                        "passed": qc.total_intensity_check.passed,
                        "measured": qc.total_intensity_check.measured_value,
                        "threshold": qc.total_intensity_check.threshold,
                        "details": qc.total_intensity_check.details
                    },
                    "goodness_of_fit": {
                        "passed": qc.goodness_check.passed,
                        "measured": qc.goodness_check.measured_value,
                        "threshold": qc.goodness_check.threshold,
                        "details": qc.goodness_check.details
                    },
                    "snr": {
                        "passed": qc.snr_check.passed,
                        "measured": qc.snr_check.measured_value,
                        "threshold": qc.snr_check.threshold,
                        "details": qc.snr_check.details
                    },
                    "ring_quality": {
                        "passed": qc.ring_quality_check.passed,
                        "measured": qc.ring_quality_check.measured_value,
                        "threshold": qc.ring_quality_check.threshold,
                        "details": qc.ring_quality_check.details
                    },
                    "poni": {
                        "success": qc.poni_result.success,
                        "distance_mm": qc.poni_result.distance_mm if qc.poni_result.HasField("distance_mm") else None,
                        "beam_center_x": qc.poni_result.beam_center_x if qc.poni_result.HasField("beam_center_x") else None,
                        "beam_center_y": qc.poni_result.beam_center_y if qc.poni_result.HasField("beam_center_y") else None,
                        "wavelength_angstrom": qc.poni_result.wavelength_angstrom if qc.poni_result.HasField("wavelength_angstrom") else None
                    }
                },
                "formatted_report": qc.formatted_report
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
            
    # Configuration (would need to be added to proto)
    def get_config(self) -> Dict[str, Any]:
        """Get configuration (placeholder - not in current proto)."""
        return {
            "error": "Configuration management not yet implemented in gRPC",
            "note": "This requires adding a Configuration service to hub.proto"
        }
        
    def set_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set configuration (placeholder)."""
        return {
            "error": "Configuration management not yet implemented in gRPC",
            "note": "This requires adding a Configuration service to hub.proto"
        }
        
    def patch_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Patch configuration (placeholder)."""
        return {
            "error": "Configuration management not yet implemented in gRPC",
            "note": "This requires adding a Configuration service to hub.proto"
        }
        
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration (placeholder)."""
        return {
            "error": "Configuration management not yet implemented in gRPC",
            "note": "This requires adding a Configuration service to hub.proto"
        }
        
    # Maintenance mode (would need to be added to proto)
    def enter_maintenance(self, ttl_seconds: int) -> Dict[str, Any]:
        """Enter maintenance mode (placeholder)."""
        return {
            "error": "Maintenance mode not yet implemented in gRPC",
            "note": "This requires adding a Maintenance service to hub.proto with enter/exit/renew methods"
        }
        
    def exit_maintenance(self) -> Dict[str, Any]:
        """Exit maintenance mode (placeholder)."""
        return {
            "error": "Maintenance mode not yet implemented in gRPC",
            "note": "This requires adding a Maintenance service to hub.proto"
        }
        
    def renew_maintenance(self, ttl_seconds: int) -> Dict[str, Any]:
        """Renew maintenance lease (placeholder)."""
        return {
            "error": "Maintenance mode not yet implemented in gRPC",
            "note": "This requires adding a Maintenance service to hub.proto"
        }
    
    # State Monitoring Methods
    def get_full_server_state(self) -> Dict[str, Any]:
        """Get complete server state including all devices.
        
        Returns comprehensive state from StateMonitor.GetFullServerState.
        """
        try:
            response = self.state_monitor.GetFullServerState(hub_pb2.Empty())
            
            return {
                "safety_state": hub_pb2.ServerState.Name(response.safety_state),
                "gpio": {
                    "powered": response.gpio.powered,
                    "key_switch_on": response.gpio.key_switch_on,
                    "activation_button_active": response.gpio.activation_button_active,
                    "activation_remaining_secs": response.gpio.activation_remaining_secs if response.gpio.HasField("activation_remaining_secs") else None,
                    "main_led": response.gpio.main_led,
                    "radiation_led": response.gpio.radiation_led,
                },
                "detector": {
                    "powered": response.detector.powered,
                    "initialized": response.detector.initialized,
                    "status": hub_pb2.DetectorStatus.Name(response.detector.status),
                    "temperature": response.detector.temperature,
                    "total_exposures": response.detector.total_exposures,
                },
                "motion": {
                    "powered": response.motion.powered,
                    "initialized": response.motion.initialized,
                    "is_homed": response.motion.is_homed,
                    "status": hub_pb2.MotionStatus.Name(response.motion.status),
                    "position_x": response.motion.position_x if response.motion.HasField("position_x") else None,
                    "position_y": response.motion.position_y if response.motion.HasField("position_y") else None,
                    "total_moves": response.motion.total_moves,
                },
                "timestamp": response.timestamp.seconds,
            }
        except grpc.RpcError as e:
            return {"error": str(e)}
    
    def subscribe_to_state_updates(self):
        """Subscribe to state change notifications.
        
        Returns a stream of StateChangeNotification events.
        Use this to receive push notifications when any component state changes.
        
        Example:
            for notification in client.subscribe_to_state_updates():
                print(f"State changed: {notification.component} - {notification.change_type}")
        """
        try:
            return self.state_monitor.SubscribeToStateUpdates(hub_pb2.Empty())
        except grpc.RpcError as e:
            print(f"Failed to subscribe to state updates: {e}")
            return None
    
    # Workflow-specific methods for daily operations
    async def get_key_switch_state(self) -> 'KeySwitchState':
        """Get current key switch state from hardware server.
        
        Returns:
            KeySwitchState with key_on boolean and last_change timestamp
            
        Note: This method requires server implementation of DeviceControl.GetKeySwitchState
        For now, returns stub data for development.
        """
        # TODO: Implement when server adds DeviceControl.GetKeySwitchState
        from dataclasses import dataclass
        from datetime import datetime
        
        @dataclass
        class KeySwitchState:
            key_on: bool
            last_change: datetime
        
        # STUB: Return key as ON for development
        return KeySwitchState(
            key_on=True,
            last_change=datetime.now()
        )
    
    async def activate(self, user_id: str, timeout_seconds: int = 20):
        """Trigger system activation (startup sequence).
        
        Args:
            user_id: User triggering activation
            timeout_seconds: Timeout for physical enable button press
            
        Raises:
            grpc.RpcError: If activation fails or times out
            
        Note: Requires server implementation of DeviceControl.Activate
        """
        # TODO: Implement when server adds DeviceControl.Activate
        # For now, stub that returns immediately
        pass
    
    async def stream_startup_progress(self):
        """Stream real-time startup progress from server.
        
        Yields:
            Dict with subsystem_name, progress_percent, status_message, completed, error
            
        Note: Requires server implementation of DeviceControl.StreamStartupProgress
        """
        # TODO: Implement when server adds streaming endpoint
        # For now, yield stub progress events
        subsystems = [
            "POWER_SUPPLY",
            "DETECTOR",
            "MOTION_CONTROL",
            "WATCHDOGS",
            "BEAM_BLOCK"
        ]
        
        import asyncio
        
        for subsystem in subsystems:
            await asyncio.sleep(0.5)
            yield {
                "subsystem_name": subsystem,
                "progress_percent": 100,
                "status_message": f"{subsystem} ready",
                "completed": True,
                "error": False
            }
    
    async def get_warmup_status(self) -> 'WarmupStatus':
        """Get current X-ray source warmup status.
        
        Returns:
            WarmupStatus with is_warming_up, elapsed_seconds, total_seconds
            
        Note: Requires server implementation of DeviceControl.GetWarmupStatus
        """
        from dataclasses import dataclass
        
        @dataclass
        class WarmupStatus:
            is_warming_up: bool
            elapsed_seconds: int
            total_seconds: int
        
        # TODO: Implement when server adds warmup tracking
        # STUB: Return not warming up
        return WarmupStatus(
            is_warming_up=False,
            elapsed_seconds=600,
            total_seconds=600
        )
    
    async def get_calibration_status(self) -> 'CalibrationStatus':
        """Get calibration validity status.
        
        Returns:
            CalibrationStatus with is_valid, last_calibration_time, time_until_expiry
            
        Note: Requires server implementation of calibration tracking
        """
        from dataclasses import dataclass
        from datetime import datetime
        from typing import Optional
        
        @dataclass
        class CalibrationStatus:
            is_valid: bool
            last_calibration_time: Optional[datetime]
            time_until_expiry: Optional[int]  # seconds
        
        # TODO: Implement when server adds calibration manager
        # STUB: Return valid calibration
        return CalibrationStatus(
            is_valid=True,
            last_calibration_time=datetime.now(),
            time_until_expiry=3600  # 1 hour remaining
        )
    
    async def start_exposure_with_uuid(
        self,
        measurement_id: str,
        operator_id: str,
        exposure_time_ms: int
    ) -> Dict[str, Any]:
        """Start measurement with UUID (no patient information).
        
        Preconditions (per safety spec):
        - Door CLOSED = True
        - Activation/Enable button ACTIVE (within 20s window)
        - Radiation SAFE before start (beam blocked); becomes FALSE during exposure
        
        CRITICAL: Only UUID and operator ID are sent to server.
        Patient information NEVER leaves the orchestrator.
        
        Args:
            measurement_id: UUID for this measurement (generated by orchestrator)
            operator_id: User ID performing measurement (no name)
            exposure_time_ms: Exposure time in milliseconds
            
        Returns:
            Dict with result data
        """
        start_time = time.time()
        command_data = {
            "measurement_id": measurement_id,
            "operator_id": operator_id,
            "exposure_time_ms": exposure_time_ms,
            "type": "patient_measurement"
        }
        
        try:
            # Verify activation/enable button
            button = self.check_enable_button()
            if not button.get("active", False):
                error_msg = "Enable/Activation button not active - required to start measurement"
                self._log_command("Acquisition.StartExposure", measurement_id, operator_id, command_data, start_time, "failure", error_msg)
                return {
                    "error": error_msg,
                    "detail": "Click ACTIVATE/ENABLE button and retry within 20 seconds"
                }

            # Verify door closed and radiation safe before starting
            gpio = self.get_gpio_state()
            if "error" in gpio:
                error_msg = f"Failed to read GPIO state: {gpio['error']}"
                self._log_command("Acquisition.StartExposure", measurement_id, operator_id, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            interlocks = gpio.get("interlocks", {})
            if not interlocks.get("door_closed", False):
                error_msg = "Door must be CLOSED to start measurement"
                self._log_command("Acquisition.StartExposure", measurement_id, operator_id, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}
            if not interlocks.get("radiation_safe", False):
                error_msg = "Radiation not in SAFE state before measurement - ensure beam is blocked"
                self._log_command("Acquisition.StartExposure", measurement_id, operator_id, command_data, start_time, "failure", error_msg)
                return {"error": error_msg}

            ctx = self._create_context(operator_id, "patient_measurement")
            # Override command_id with measurement UUID
            ctx.command_id = measurement_id
            
            request = hub_pb2.StartExposureRequest(
                ctx=ctx,
                exposure_time_ms=exposure_time_ms,
                max_timeout_ms=exposure_time_ms + 5000,
            )
            
            self.acquisition.StartExposure(request)
            
            # Log successful patient measurement start
            self._log_command("Acquisition.StartExposure", measurement_id, operator_id, command_data, start_time, "success")
            
            return {
                "status": "started",
                "measurement_id": measurement_id,
                "exposure_time_ms": exposure_time_ms
            }
        except grpc.RpcError as e:
            self._log_command("Acquisition.StartExposure", measurement_id, operator_id, command_data, start_time, "failure", str(e))
            return {"error": str(e)}
    
    # Command Discovery Methods
    def get_server_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities and version info."""
        if not self.command_discovery:
            return {"error": "CommandDiscovery service not available on this server"}
        
        try:
            resp = self.command_discovery.GetServerCapabilities(hub_pb2.Empty())
            caps = resp.capabilities
            return {
                "server_version": caps.server_version,
                "protocol_version": caps.protocol_version,
                "build_time": caps.build_time.seconds if caps.HasField('build_time') else None,
                "supported_features": list(caps.supported_features),
                "device_type": caps.device_type,
            }
        except grpc.RpcError as e:
            return {"error": str(e), "code": str(e.code()), "details": e.details()}
    
    def list_commands(self) -> Dict[str, Any]:
        """List all available commands with metadata."""
        if not self.command_discovery:
            return {"error": "CommandDiscovery service not available on this server"}
        
        try:
            resp = self.command_discovery.ListCommands(hub_pb2.Empty())
            
            commands = []
            for cmd in resp.commands:
                commands.append({
                    "service_name": cmd.service_name,
                    "command_name": cmd.command_name,
                    "description": cmd.description,
                    "request_fields": [
                        {
                            "name": f.name,
                            "type": f.type,
                            "required": f.required,
                            "description": f.description,
                            "default_value": f.default_value if f.HasField('default_value') else None,
                        } for f in cmd.request_fields
                    ],
                    "response_type": cmd.response_type,
                    "response_fields": [
                        {
                            "name": f.name,
                            "type": f.type,
                            "required": f.required,
                            "description": f.description,
                        } for f in cmd.response_fields
                    ],
                    "required_permissions": list(cmd.required_permissions),
                    "safety_requirements": list(cmd.safety_requirements),
                })
            
            return {
                "commands": commands,
                "server_info": {
                    "server_version": resp.server_info.server_version,
                    "protocol_version": resp.server_info.protocol_version,
                    "supported_features": list(resp.server_info.supported_features),
                    "device_type": resp.server_info.device_type,
                }
            }
        except grpc.RpcError as e:
            return {"error": str(e), "code": str(e.code()), "details": e.details()}
    
    def validate_compatibility(
        self,
        client_version: str = "0.1.0",
        protocol_version: str = "1.0.0",
        required_commands: Optional[list] = None
    ) -> Dict[str, Any]:
        """Validate compatibility with server."""
        if not self.command_discovery:
            return {"error": "CommandDiscovery service not available on this server"}
        
        if required_commands is None:
            # Default commands that orchestrator needs
            required_commands = [
                "Acquisition.StartExposure",
                "Acquisition.Stop",
                "Acquisition.GetState",
                "Motion.MoveTo",
                "Motion.Home",
                "Safety.GetInterlockStatus",
                "Health.GetAggregateHealth",
                "StateMonitor.GetFullServerState",
            ]
        
        try:
            req = hub_pb2.ValidateCompatibilityRequest(
                client_version=client_version,
                client_protocol_version=protocol_version,
                required_commands=required_commands,
            )
            resp = self.command_discovery.ValidateCompatibility(req)
            
            return {
                "compatible": resp.compatible,
                "message": resp.message,
                "missing_commands": list(resp.missing_commands),
                "version_warnings": list(resp.version_warnings),
                "protocol_compatible": resp.protocol_compatible,
            }
        except grpc.RpcError as e:
            return {"error": str(e), "code": str(e.code()), "details": e.details()}
