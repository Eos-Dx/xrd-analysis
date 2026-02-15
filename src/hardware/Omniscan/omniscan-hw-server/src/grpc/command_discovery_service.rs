use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;
use chrono::Utc;

use crate::grpc::hub::v1::*;
use crate::grpc::services::ServiceState;

/// Protocol version - increment when breaking changes are made to the protobuf schema
const PROTOCOL_VERSION: &str = "1.0.0";

/// CommandDiscovery service implementation
/// Provides introspection of available commands, their fields, and version information
pub struct CommandDiscoveryService {
    state: Arc<ServiceState>,
}

impl CommandDiscoveryService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }

    /// Compute readiness for all known commands based on current system state
    async fn compute_readiness(&self) -> Vec<CommandReadiness> {
        let commands = Self::get_all_commands();

        // Snapshot state
        let safety_sm = self.state.safety_state_machine.read().await;
        let current_state = safety_sm.get_current_state().await;
        drop(safety_sm);

        let interlocks = self.state.gpio.get_interlocks().await;
        let key_on = self.state.gpio.get_key_switch_state().await.unwrap_or(false);
        let enable_active = self.state.gpio.get_activation_button_active().await.unwrap_or(false);
        let det_health = self.state.detector.get_health().await;

        let safety_state_str = current_state.as_str().to_string();

        commands.into_iter().map(|c| {
            let mut reasons: Vec<String> = Vec::new();

            // Generic helpers
            let require = |cond: bool, msg: &str, reasons: &mut Vec<String>| {
                if !cond { reasons.push(msg.to_string()); }
            };

            // Generic checks inferred from descriptor safety_requirements strings
            for req in &c.safety_requirements {
                match req.as_str() {
                    "key_switch_on" => require(key_on, "Key switch is OFF", &mut reasons),
                    "activation_button" => require(enable_active, "Enable button is not active", &mut reasons),
                    "interlocks_safe" => require(interlocks.overall_safe, "Interlocks are not satisfied", &mut reasons),
                    "beam_closed" => require(interlocks.radiation_safe, "Beam is open (radiation not safe)", &mut reasons),
                    "state_armed_or_running" => {
                        let ok = matches!(safety_state_str.as_str(), "PENDING_ARMED" | "RUNNING");
                        require(ok, &format!("State is {} (need PENDING_ARMED or RUNNING)", safety_state_str), &mut reasons)
                    }
                    _ => {}
                }
            }

            // Command-specific rules aligned with TOML schemas
            match (c.service_name.as_str(), c.command_name.as_str()) {
                ("Acquisition", "CalibrateDetector") => {
                    // Allowed states: Idle or Locked
                    let ok_state = matches!(safety_state_str.as_str(), "IDLE" | "LOCKED");
                    require(ok_state, &format!("State {} not allowed for calibration", safety_state_str), &mut reasons);
                    require(interlocks.door_closed, "Safety door must be closed", &mut reasons);
                    require(interlocks.emergency_stop, "Emergency stop is pressed", &mut reasons);
                    // Detector ready (powered and Idle)
                    let det_ready = det_health.powered && matches!(det_health.status, crate::devices::detectors::DetectorStatus::Idle);
                    require(det_ready, "Detector is not ready", &mut reasons);
                }
                ("DeviceInitialization", "InitializeDetector") => {
                    require(interlocks.radiation_safe, "Beam must be closed (radiation safe=true)", &mut reasons);
                    require(interlocks.cooling_ok, "Cooling not OK", &mut reasons);
                    require(interlocks.power_ok, "Power not OK", &mut reasons);
                }
                ("DeviceInitialization", "InitializeMotion") => {
                    require(interlocks.radiation_safe, "Beam must be closed (radiation safe=true)", &mut reasons);
                    require(interlocks.power_ok, "Power not OK", &mut reasons);
                }
                _ => {}
            }

            CommandReadiness {
                service_name: c.service_name.clone(),
                command_name: c.command_name.clone(),
                ready: reasons.is_empty(),
                reasons,
            }
        }).collect()
    }

    /// Returns metadata about all available commands
    fn get_all_commands() -> Vec<CommandDescriptor> {
        vec![
            // Acquisition Service
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "StartExposure".to_string(),
                description: "Start an X-ray exposure with specified duration".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "exposure_time_ms".to_string(),
                        r#type: "uint32".to_string(),
                        required: true,
                        description: "Exposure duration in milliseconds".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "max_timeout_ms".to_string(),
                        r#type: "uint32".to_string(),
                        required: true,
                        description: "Safety timeout in milliseconds".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["interlocks_safe".to_string(), "state_armed_or_running".to_string()],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "Stop".to_string(),
                description: "Stop current acquisition gracefully".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "Abort".to_string(),
                description: "Immediately abort current acquisition".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "GetState".to_string(),
                description: "Get current server safety state".to_string(),
                request_fields: vec![],
                response_type: "GetStateResponse".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "state".to_string(),
                        r#type: "ServerState".to_string(),
                        required: true,
                        description: "Current safety state".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "detail".to_string(),
                        r#type: "string".to_string(),
                        required: true,
                        description: "State description".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "GetLastExposureResult".to_string(),
                description: "Get the last exposure result".to_string(),
                request_fields: vec![],
                response_type: "GetExposureResultResponse".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "has_result".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "Whether a result is available".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "result".to_string(),
                        r#type: "ExposureResult".to_string(),
                        required: false,
                        description: "Exposure result data".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "CalibrateDetector".to_string(),
                description: "Perform detector calibration with QC checks".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "CalibrateDetectorResponse".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "success".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "Calibration success status".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "qc_report".to_string(),
                        r#type: "CalibrationQcReport".to_string(),
                        required: false,
                        description: "Quality control report".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec!["key_switch_on".to_string(), "activation_button".to_string()],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "GetLastCalibration".to_string(),
                description: "Retrieve last calibration report".to_string(),
                request_fields: vec![],
                response_type: "GetLastCalibrationResponse".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "has_calibration".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "Whether calibration exists".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Acquisition".to_string(),
                command_name: "StartBackgroundMeasurement".to_string(),
                description: "Capture background/dark frame (requires beam closed)".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "exposure_time_ms".to_string(),
                        r#type: "uint32".to_string(),
                        required: true,
                        description: "Exposure time for dark frame".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "BackgroundMeasurementResponse".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "success".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "Measurement success status".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec!["beam_closed".to_string()],
            },
            // Motion Service
            CommandDescriptor {
                service_name: "Motion".to_string(),
                command_name: "MoveTo".to_string(),
                description: "Move to absolute position".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "position_mm".to_string(),
                        r#type: "double".to_string(),
                        required: true,
                        description: "Target position in millimeters".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["motion_homed".to_string(), "interlocks_safe".to_string()],
            },
            CommandDescriptor {
                service_name: "Motion".to_string(),
                command_name: "MoveRelative".to_string(),
                description: "Move relative distance from current position".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "distance_mm".to_string(),
                        r#type: "double".to_string(),
                        required: true,
                        description: "Relative distance in millimeters".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["motion_homed".to_string(), "interlocks_safe".to_string()],
            },
            CommandDescriptor {
                service_name: "Motion".to_string(),
                command_name: "Home".to_string(),
                description: "Home motion axes".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["motion_powered".to_string()],
            },
            CommandDescriptor {
                service_name: "Motion".to_string(),
                command_name: "Stop".to_string(),
                description: "Stop motion immediately".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Motion".to_string(),
                command_name: "SetVelocity".to_string(),
                description: "Set motion velocity".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "velocity_mm_s".to_string(),
                        r#type: "double".to_string(),
                        required: true,
                        description: "Velocity in mm/s".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Motion".to_string(),
                command_name: "GetPosition".to_string(),
                description: "Get current motion position".to_string(),
                request_fields: vec![],
                response_type: "GetPositionResponse".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "position_mm".to_string(),
                        r#type: "double".to_string(),
                        required: false,
                        description: "Current position in mm".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "is_homed".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "Whether motion is homed".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            // DeviceControl Service
            CommandDescriptor {
                service_name: "DeviceControl".to_string(),
                command_name: "PowerDevice".to_string(),
                description: "Power device on/off".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "device_type".to_string(),
                        r#type: "string".to_string(),
                        required: true,
                        description: "Device type: detector, motion, gpio".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "power_on".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "True to power on, false to power off".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["key_switch_on".to_string()],
            },
            CommandDescriptor {
                service_name: "DeviceControl".to_string(),
                command_name: "GetDetectorHealth".to_string(),
                description: "Get detector health information".to_string(),
                request_fields: vec![],
                response_type: "DetectorHealth".to_string(),
                response_fields: vec![
                    FieldDescriptor {
                        name: "powered".to_string(),
                        r#type: "bool".to_string(),
                        required: true,
                        description: "Power status".to_string(),
                        default_value: None,
                    },
                    FieldDescriptor {
                        name: "temperature".to_string(),
                        r#type: "float".to_string(),
                        required: true,
                        description: "Temperature in Celsius".to_string(),
                        default_value: None,
                    },
                ],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "DeviceControl".to_string(),
                command_name: "GetMotionHealth".to_string(),
                description: "Get motion system health information".to_string(),
                request_fields: vec![],
                response_type: "MotionHealth".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "DeviceControl".to_string(),
                command_name: "GetDeviceState".to_string(),
                description: "Get general device state (supports pdu, gpio, detector, motion)".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "device_type".to_string(),
                        r#type: "string".to_string(),
                        required: true,
                        description: "Device type to query".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "DeviceStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            // Safety Service
            CommandDescriptor {
                service_name: "Safety".to_string(),
                command_name: "GetInterlockStatus".to_string(),
                description: "Get current interlock status".to_string(),
                request_fields: vec![],
                response_type: "InterlockStatus".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Safety".to_string(),
                command_name: "ResetInterlocks".to_string(),
                description: "Reset interlock conditions".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["key_switch_on".to_string()],
            },
            CommandDescriptor {
                service_name: "Safety".to_string(),
                command_name: "CheckSafetyToOperate".to_string(),
                description: "Check if safe to operate".to_string(),
                request_fields: vec![],
                response_type: "InterlockStatus".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            // Health Service
            CommandDescriptor {
                service_name: "Health".to_string(),
                command_name: "Liveness".to_string(),
                description: "Basic liveness probe".to_string(),
                request_fields: vec![],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Health".to_string(),
                command_name: "Readiness".to_string(),
                description: "Readiness probe for operations".to_string(),
                request_fields: vec![],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "Health".to_string(),
                command_name: "GetAggregateHealth".to_string(),
                description: "Detailed health check of all components".to_string(),
                request_fields: vec![],
                response_type: "AggregateHealth".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            // StateMonitor Service
            CommandDescriptor {
                service_name: "StateMonitor".to_string(),
                command_name: "GetFullServerState".to_string(),
                description: "Get complete server state snapshot".to_string(),
                request_fields: vec![],
                response_type: "FullServerStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "StateMonitor".to_string(),
                command_name: "GetGpioState".to_string(),
                description: "Get GPIO state".to_string(),
                request_fields: vec![],
                response_type: "GpioStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "StateMonitor".to_string(),
                command_name: "GetDetectorState".to_string(),
                description: "Get detector state".to_string(),
                request_fields: vec![],
                response_type: "DetectorStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "StateMonitor".to_string(),
                command_name: "GetMotionState".to_string(),
                description: "Get motion state".to_string(),
                request_fields: vec![],
                response_type: "MotionStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            // DeviceInitialization Service
            CommandDescriptor {
                service_name: "DeviceInitialization".to_string(),
                command_name: "InitializeDetector".to_string(),
                description: "Initialize detector (requires key switch and activation)".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "DetectorStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["key_switch_on".to_string(), "activation_button".to_string(), "interlocks_safe".to_string()],
            },
            CommandDescriptor {
                service_name: "DeviceInitialization".to_string(),
                command_name: "InitializeMotion".to_string(),
                description: "Initialize motion system (requires key switch and activation)".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "MotionStateResponse".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec!["key_switch_on".to_string(), "activation_button".to_string(), "interlocks_safe".to_string()],
            },
            CommandDescriptor {
                service_name: "DeviceInitialization".to_string(),
                command_name: "PowerOffDetector".to_string(),
                description: "Power off detector".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
            CommandDescriptor {
                service_name: "DeviceInitialization".to_string(),
                command_name: "PowerOffMotion".to_string(),
                description: "Power off motion system".to_string(),
                request_fields: vec![
                    FieldDescriptor {
                        name: "ctx".to_string(),
                        r#type: "CommandContext".to_string(),
                        required: true,
                        description: "Command context for audit trail".to_string(),
                        default_value: None,
                    },
                ],
                response_type: "Empty".to_string(),
                response_fields: vec![],
                required_permissions: vec![],
                safety_requirements: vec![],
            },
        ]
    }

    /// Get server capabilities including version and features
    fn get_server_capabilities(&self) -> ServerCapabilities {
        let features = vec![
            "audit_logging".to_string(),
            "safety_interlocks".to_string(),
            "state_monitoring".to_string(),
            "calibration_qc".to_string(),
        ];

        #[cfg(feature = "gui")]
        let features = {
            let mut f = features;
            f.push("gui".to_string());
            f
        };

        ServerCapabilities {
            server_version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            build_time: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(), // In production, use build time
                nanos: 0,
            }),
            supported_features: features,
            device_type: "hardware_server".to_string(),
        }
    }

    /// Validate compatibility between client and server
    fn validate_compatibility(
        &self,
        request: ValidateCompatibilityRequest,
    ) -> ValidateCompatibilityResponse {
        let mut warnings = Vec::new();
        let mut missing_commands = Vec::new();
        
        let server_capabilities = self.get_server_capabilities();
        let all_commands = Self::get_all_commands();
        
        // Check protocol version compatibility
        let protocol_compatible = self.check_protocol_compatibility(
            &request.client_protocol_version,
            &server_capabilities.protocol_version,
        );
        
        if !protocol_compatible {
            warnings.push(format!(
                "Protocol version mismatch: client {} vs server {}",
                request.client_protocol_version, server_capabilities.protocol_version
            ));
        }
        
        // Check version compatibility (semver)
        if let Some(warning) = self.check_version_compatibility(
            &request.client_version,
            &server_capabilities.server_version,
        ) {
            warnings.push(warning);
        }
        
        // Check if all required commands are available
        for required_cmd in &request.required_commands {
            let parts: Vec<&str> = required_cmd.split('.').collect();
            if parts.len() == 2 {
                let service_name = parts[0];
                let command_name = parts[1];
                
                let found = all_commands.iter().any(|cmd| {
                    cmd.service_name == service_name && cmd.command_name == command_name
                });
                
                if !found {
                    missing_commands.push(required_cmd.clone());
                }
            }
        }
        
        let compatible = protocol_compatible && missing_commands.is_empty();
        
        let message = if compatible {
            if warnings.is_empty() {
                "Client and server are fully compatible".to_string()
            } else {
                format!("Compatible with warnings: {}", warnings.join(", "))
            }
        } else {
            let mut reasons = Vec::new();
            if !protocol_compatible {
                reasons.push("incompatible protocol version".to_string());
            }
            if !missing_commands.is_empty() {
                reasons.push(format!(
                    "missing commands: {}",
                    missing_commands.join(", ")
                ));
            }
            format!("Incompatible: {}", reasons.join("; "))
        };
        
        ValidateCompatibilityResponse {
            compatible,
            message,
            missing_commands,
            version_warnings: warnings,
            protocol_compatible,
        }
    }
    
    /// Check if protocol versions are compatible (major version must match)
    fn check_protocol_compatibility(&self, client_version: &str, server_version: &str) -> bool {
        let client_major = client_version.split('.').next().unwrap_or("0");
        let server_major = server_version.split('.').next().unwrap_or("0");
        client_major == server_major
    }
    
    /// Check version compatibility and return warning if needed
    fn check_version_compatibility(&self, client_version: &str, server_version: &str) -> Option<String> {
        // Parse semver versions
        let client_parts: Vec<&str> = client_version.split('.').collect();
        let server_parts: Vec<&str> = server_version.split('.').collect();
        
        if client_parts.len() < 2 || server_parts.len() < 2 {
            return Some("Invalid version format".to_string());
        }
        
        let client_major = client_parts[0].parse::<u32>().unwrap_or(0);
        let server_major = server_parts[0].parse::<u32>().unwrap_or(0);
        
        if client_major != server_major {
            return Some(format!(
                "Major version mismatch: client v{} may not work with server v{}",
                client_version, server_version
            ));
        }
        
        None
    }
}

#[tonic::async_trait]
impl command_discovery_server::CommandDiscovery for CommandDiscoveryService {
    async fn get_server_capabilities(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<GetServerCapabilitiesResponse>, Status> {
        info!("CommandDiscovery: GetServerCapabilities called");
        
        let capabilities = self.get_server_capabilities();
        
        Ok(Response::new(GetServerCapabilitiesResponse {
            capabilities: Some(capabilities),
        }))
    }
    
    async fn list_commands(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ListCommandsResponse>, Status> {
        info!("CommandDiscovery: ListCommands called");
        
        let commands = Self::get_all_commands();
        let server_info = self.get_server_capabilities();
        
        Ok(Response::new(ListCommandsResponse {
            commands,
            server_info: Some(server_info),
        }))
    }
    
    async fn validate_compatibility(
        &self,
        request: Request<ValidateCompatibilityRequest>,
    ) -> Result<Response<ValidateCompatibilityResponse>, Status> {
        let req = request.into_inner();
        
        info!(
            "CommandDiscovery: ValidateCompatibility called - client v{}, protocol v{}",
            req.client_version, req.client_protocol_version
        );
        
        let response = self.validate_compatibility(req);
        
        if !response.compatible {
            info!("Client compatibility check failed: {}", response.message);
        } else {
            info!("Client is compatible");
        }
        
        Ok(Response::new(response))
    }

    async fn get_command_readiness(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<GetCommandReadinessResponse>, Status> {
        let items = self.compute_readiness().await;
        Ok(Response::new(GetCommandReadinessResponse { items }))
    }
}
