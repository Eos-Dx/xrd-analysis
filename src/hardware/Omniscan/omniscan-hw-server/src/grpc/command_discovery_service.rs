use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use crate::commands::schema::{CommandSchema, InterlockCheck, ParameterType};
use crate::commands::CommandRegistry;
use crate::grpc::hub::v1::*;
use crate::grpc::services::ServiceState;

/// Protocol version - increment when breaking changes are made to the protobuf schema.
const PROTOCOL_VERSION: &str = "1.1.0";

#[derive(Clone)]
struct LoadedCommand {
    schema: CommandSchema,
    descriptor: CommandDescriptor,
}

/// CommandDiscovery service implementation.
/// Metadata is sourced from TOML command schemas under the shared protocol directory.
pub struct CommandDiscoveryService {
    state: Arc<ServiceState>,
    commands: Vec<LoadedCommand>,
}

impl CommandDiscoveryService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        let commands = Self::load_commands();
        Self { state, commands }
    }

    fn command_schema_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../protocol/commands/v1")
    }

    fn load_commands() -> Vec<LoadedCommand> {
        let schema_dir = Self::command_schema_dir();
        let registry = match CommandRegistry::load_from_directory(&schema_dir) {
            Ok(registry) => registry,
            Err(err) => {
                warn!(
                    "Failed to load command schemas from {}: {}",
                    schema_dir.display(),
                    err
                );
                return Vec::new();
            }
        };

        let mut command_ids = registry.list_commands();
        command_ids.sort();

        let mut loaded = Vec::new();
        for command_id in command_ids {
            if let Some(schema) = registry.get(&command_id) {
                let descriptor = Self::schema_to_descriptor(schema);
                loaded.push(LoadedCommand {
                    schema: schema.clone(),
                    descriptor,
                });
            }
        }

        loaded
    }

    fn schema_to_descriptor(schema: &CommandSchema) -> CommandDescriptor {
        let command_name = Self::to_pascal_case(&schema.command_id);

        let mut request_fields = Vec::new();
        if Self::command_requires_context(schema, &command_name) {
            request_fields.push(FieldDescriptor {
                name: "ctx".to_string(),
                r#type: "CommandContext".to_string(),
                required: true,
                description: "Command context for audit trail".to_string(),
                default_value: None,
            });
        }

        for param in &schema.parameters {
            request_fields.push(FieldDescriptor {
                name: param.name.clone(),
                r#type: Self::param_type_to_proto(&param.param_type),
                required: param.required,
                description: param.description.clone(),
                default_value: param.default_value.clone(),
            });
        }

        let response_type = Self::response_type_for(&schema.service, &command_name).to_string();

        CommandDescriptor {
            service_name: schema.service.clone(),
            command_name,
            description: schema.description.clone(),
            request_fields,
            response_type,
            response_fields: Vec::new(),
            required_permissions: Vec::new(),
            safety_requirements: Self::safety_requirements_tokens(schema),
        }
    }

    fn command_requires_context(schema: &CommandSchema, command_name: &str) -> bool {
        // Core-8 phase: GetState is the only command using Empty request.
        !(schema.service == "Acquisition" && command_name == "GetState")
    }

    fn response_type_for(service: &str, command_name: &str) -> &'static str {
        match (service, command_name) {
            ("Acquisition", "GetState") => "GetStateResponse",
            ("DeviceInitialization", "InitializeDetector") => "DetectorStateResponse",
            ("DeviceInitialization", "InitializeMotion") => "MotionStateResponse",
            _ => "Empty",
        }
    }

    fn param_type_to_proto(param_type: &ParameterType) -> String {
        match param_type {
            ParameterType::Float { .. } => "double".to_string(),
            ParameterType::Integer { .. } => "int64".to_string(),
            ParameterType::Boolean => "bool".to_string(),
            ParameterType::String { .. } => "string".to_string(),
            ParameterType::Enum { .. } => "string".to_string(),
        }
    }

    fn safety_requirements_tokens(schema: &CommandSchema) -> Vec<String> {
        let safety = &schema.safety_requirements;
        let mut tokens = Vec::new();

        if safety.requires_key_switch {
            tokens.push("key_switch_on".to_string());
        }
        if safety.requires_enable_button {
            tokens.push("activation_button".to_string());
        }
        if safety.requires_calibration {
            tokens.push("calibrated".to_string());
        }

        for state in &safety.allowed_states {
            tokens.push(format!("state:{}", state));
        }

        for check in &safety.interlock_checks {
            tokens.push(format!("interlock:{}", check.name));
        }

        tokens
    }

    fn to_pascal_case(value: &str) -> String {
        value
            .split('_')
            .filter(|part| !part.is_empty())
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => {
                        let mut out = String::new();
                        out.extend(first.to_uppercase());
                        out.push_str(chars.as_str());
                        out
                    }
                    None => String::new(),
                }
            })
            .collect::<String>()
    }

    fn list_descriptors(&self) -> Vec<CommandDescriptor> {
        self.commands
            .iter()
            .map(|loaded| loaded.descriptor.clone())
            .collect()
    }

    /// Compute readiness for all known commands based on current system state.
    async fn compute_readiness(&self) -> Vec<CommandReadiness> {
        let safety_sm = self.state.safety_state_machine.read().await;
        let current_state = safety_sm.get_current_state().await;
        let current_state_name = current_state.as_str().to_string();
        drop(safety_sm);

        let interlocks = self.state.gpio.get_interlocks().await;
        let key_on = self.state.gpio.get_key_switch_state().await.unwrap_or(false);
        let enable_active = self
            .state
            .gpio
            .get_activation_button_active()
            .await
            .unwrap_or(false);

        let mut results = Vec::new();

        for loaded in &self.commands {
            let schema = &loaded.schema;
            let mut reasons: Vec<String> = Vec::new();
            let safety = &schema.safety_requirements;

            if !safety.allowed_states.is_empty() {
                let allowed = safety
                    .allowed_states
                    .iter()
                    .any(|state| state.eq_ignore_ascii_case(&current_state_name));
                if !allowed {
                    reasons.push(format!(
                        "State {} not allowed (allowed: {})",
                        current_state_name,
                        safety.allowed_states.join(", ")
                    ));
                }
            }

            if safety.requires_key_switch && !key_on {
                reasons.push("Key switch is OFF".to_string());
            }

            if safety.requires_enable_button && !enable_active {
                reasons.push("Enable button is not active".to_string());
            }

            if safety.requires_calibration && !current_state_name.eq_ignore_ascii_case("Calibrated") {
                reasons.push("Calibration is required".to_string());
            }

            for check in &safety.interlock_checks {
                match Self::interlock_check_passes(check, &interlocks, key_on, enable_active) {
                    Some(true) => {}
                    Some(false) => reasons.push(check.error_message.clone()),
                    None => reasons.push(format!("Unknown interlock check: {}", check.name)),
                }
            }

            results.push(CommandReadiness {
                service_name: loaded.descriptor.service_name.clone(),
                command_name: loaded.descriptor.command_name.clone(),
                ready: reasons.is_empty(),
                reasons,
            });
        }

        results
    }

    fn interlock_check_passes(
        check: &InterlockCheck,
        interlocks: &crate::devices::gpio::InterlockStatus,
        key_on: bool,
        enable_active: bool,
    ) -> Option<bool> {
        let actual = match check.name.as_str() {
            // In TOML schemas, emergency_stop check uses active-low semantics:
            // required_value=false means "not pressed".
            "emergency_stop" => !interlocks.emergency_stop,
            "door_closed" => interlocks.door_closed,
            "beam_watchdog" | "radiation_safe" => interlocks.radiation_safe,
            "cooling_ok" => interlocks.cooling_ok,
            "power_ok" => interlocks.power_ok,
            "overall_safe" | "interlocks_safe" => interlocks.overall_safe,
            "key_switch" => key_on,
            "enable_button" => enable_active,
            _ => return None,
        };

        Some(actual == check.required_value)
    }

    /// Get server capabilities including version and features.
    fn get_server_capabilities(&self) -> ServerCapabilities {
        let features = vec![
            "audit_logging".to_string(),
            "safety_interlocks".to_string(),
            "state_monitoring".to_string(),
            "command_schemas".to_string(),
            "typed_events".to_string(),
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
                seconds: Utc::now().timestamp(),
                nanos: 0,
            }),
            supported_features: features,
            device_type: "hardware_server".to_string(),
        }
    }

    /// Validate compatibility between client and server.
    fn validate_compatibility(
        &self,
        request: ValidateCompatibilityRequest,
    ) -> ValidateCompatibilityResponse {
        let mut warnings = Vec::new();
        let mut missing_commands = Vec::new();

        let server_capabilities = self.get_server_capabilities();

        let available_commands: std::collections::HashSet<String> = self
            .commands
            .iter()
            .map(|cmd| {
                format!(
                    "{}.{}",
                    cmd.descriptor.service_name, cmd.descriptor.command_name
                )
            })
            .collect();

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

        if let Some(warning) = self.check_version_compatibility(
            &request.client_version,
            &server_capabilities.server_version,
        ) {
            warnings.push(warning);
        }

        for required_cmd in &request.required_commands {
            if !available_commands.contains(required_cmd) {
                missing_commands.push(required_cmd.clone());
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
                reasons.push(format!("missing commands: {}", missing_commands.join(", ")));
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

    /// Check if protocol versions are compatible (major version must match).
    fn check_protocol_compatibility(&self, client_version: &str, server_version: &str) -> bool {
        let client_major = client_version.split('.').next().unwrap_or("0");
        let server_major = server_version.split('.').next().unwrap_or("0");
        client_major == server_major
    }

    /// Check version compatibility and return warning if needed.
    fn check_version_compatibility(
        &self,
        client_version: &str,
        server_version: &str,
    ) -> Option<String> {
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

        let commands = self.list_descriptors();
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

#[cfg(test)]
mod tests {
    use super::CommandDiscoveryService;
    use std::collections::HashSet;

    #[test]
    fn to_pascal_case_maps_snake_case() {
        assert_eq!(CommandDiscoveryService::to_pascal_case("start_exposure"), "StartExposure");
        assert_eq!(CommandDiscoveryService::to_pascal_case("get_state"), "GetState");
    }

    #[test]
    fn load_commands_includes_core8() {
        let commands = CommandDiscoveryService::load_commands();
        let available: HashSet<String> = commands
            .into_iter()
            .map(|cmd| format!("{}.{}", cmd.descriptor.service_name, cmd.descriptor.command_name))
            .collect();

        let expected = [
            "DeviceInitialization.InitializeDetector",
            "DeviceInitialization.InitializeMotion",
            "Acquisition.GetState",
            "Motion.MoveTo",
            "Motion.Home",
            "Acquisition.StartExposure",
            "Acquisition.Stop",
            "Acquisition.Abort",
        ];

        for command in expected {
            assert!(
                available.contains(command),
                "missing expected command descriptor: {}",
                command
            );
        }
    }
}
