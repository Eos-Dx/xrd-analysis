use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::config::{ConfigManager, EnhancedDeviceConfig};
use crate::audit::{AuditLogger, CommandLog, CommandResult};

/// Maintenance mode state
pub struct MaintenanceState {
    config_manager: Arc<RwLock<ConfigManager>>,
    audit_logger: Arc<AuditLogger>,
    current_config: Arc<RwLock<EnhancedDeviceConfig>>,
    maintenance_active: Arc<RwLock<bool>>,
}

impl MaintenanceState {
    pub fn new(
        config_manager: ConfigManager,
        audit_logger: AuditLogger,
        config: EnhancedDeviceConfig,
    ) -> Self {
        Self {
            config_manager: Arc::new(RwLock::new(config_manager)),
            audit_logger: Arc::new(audit_logger),
            current_config: Arc::new(RwLock::new(config)),
            maintenance_active: Arc::new(RwLock::new(false)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MaintenanceStatusResponse {
    pub maintenance_mode: bool,
    pub config_version: u32,
    pub last_update: Option<chrono::DateTime<chrono::Utc>>,
    pub device_info: DeviceInfoResponse,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeviceInfoResponse {
    pub device_id: String,
    pub device_name: String,
    pub serial_number: String,
    pub firmware_version: String,
    pub location: String,
}

#[derive(Debug, Deserialize)]
pub struct EnableMaintenanceRequest {
    pub password: String,
    pub reason: String,
    pub engineer_id: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateConfigRequest {
    pub password: String,
    pub config_updates: serde_json::Value,
    pub reason: String,
    pub engineer_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ConfigQuery {
    pub section: Option<String>, // "motion", "detector", "gpio", "safety"
}

pub fn create_maintenance_router(state: MaintenanceState) -> Router {
    Router::new()
        .route("/status", get(get_maintenance_status))
        .route("/enable", post(enable_maintenance_mode))
        .route("/disable", post(disable_maintenance_mode))
        .route("/config", get(get_configuration))
        .route("/config", put(update_configuration))
        .route("/config/validate", post(validate_configuration))
        .route("/logs/commands", get(get_recent_commands))
        .with_state(Arc::new(state))
}

async fn get_maintenance_status(
    State(state): State<Arc<MaintenanceState>>,
) -> Result<Json<MaintenanceStatusResponse>, StatusCode> {
    let config = state.current_config.read().await;
    let maintenance_active = *state.maintenance_active.read().await;
    
    Ok(Json(MaintenanceStatusResponse {
        maintenance_mode: maintenance_active,
        config_version: config.maintenance.config_version,
        last_update: config.maintenance.last_config_update,
        device_info: DeviceInfoResponse {
            device_id: config.device_info.device_id.clone(),
            device_name: config.device_info.device_name.clone(),
            serial_number: config.device_info.serial_number.clone(),
            firmware_version: config.device_info.firmware_version.clone(),
            location: config.device_info.location.clone(),
        },
    }))
}

async fn enable_maintenance_mode(
    State(state): State<Arc<MaintenanceState>>,
    Json(request): Json<EnableMaintenanceRequest>,
) -> Result<Json<MaintenanceStatusResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    // Log the command attempt
    let mut command_log = CommandLog::new(
        "enable_maintenance_mode",
        serde_json::json!({
            "engineer_id": request.engineer_id,
            "reason": request.reason
        }),
        Some(request.engineer_id.clone()),
        serde_json::json!({"maintenance_mode": false}),
    );
    command_log.session_id = state.audit_logger.session_id().to_string();

    let config_manager = state.config_manager.read().await;
    let config = state.current_config.read().await.clone();
    
    match config_manager.enable_maintenance_mode(config, &request.password).await {
        Ok(updated_config) => {
            // Update state
            {
                let mut current_config = state.current_config.write().await;
                *current_config = updated_config.clone();
            }
            {
                let mut maintenance_active = state.maintenance_active.write().await;
                *maintenance_active = true;
            }

            // Log successful command
            command_log.device_state_after = Some(serde_json::json!({"maintenance_mode": true}));
            command_log.result = CommandResult::Success;
            command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
            
            if let Err(e) = state.audit_logger.log_command(command_log).await {
                error!("Failed to log maintenance mode enable: {}", e);
            }

            warn!("Maintenance mode ENABLED by engineer: {}", request.engineer_id);

            Ok(Json(MaintenanceStatusResponse {
                maintenance_mode: true,
                config_version: updated_config.maintenance.config_version,
                last_update: updated_config.maintenance.last_config_update,
                device_info: DeviceInfoResponse {
                    device_id: updated_config.device_info.device_id,
                    device_name: updated_config.device_info.device_name,
                    serial_number: updated_config.device_info.serial_number,
                    firmware_version: updated_config.device_info.firmware_version,
                    location: updated_config.device_info.location,
                },
            }))
        }
        Err(e) => {
            // Log failed command
            command_log.result = CommandResult::Failed { error: e.to_string() };
            command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
            
            if let Err(log_err) = state.audit_logger.log_command(command_log).await {
                error!("Failed to log maintenance mode enable failure: {}", log_err);
            }

            error!("Failed to enable maintenance mode: {}", e);
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

async fn disable_maintenance_mode(
    State(state): State<Arc<MaintenanceState>>,
) -> Result<Json<MaintenanceStatusResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    let mut command_log = CommandLog::new(
        "disable_maintenance_mode",
        serde_json::json!({}),
        Some("system".to_string()),
        serde_json::json!({"maintenance_mode": true}),
    );
    command_log.session_id = state.audit_logger.session_id().to_string();

    let config_manager = state.config_manager.read().await;
    let config = state.current_config.read().await.clone();
    
    match config_manager.disable_maintenance_mode(config).await {
        Ok(updated_config) => {
            // Update state
            {
                let mut current_config = state.current_config.write().await;
                *current_config = updated_config.clone();
            }
            {
                let mut maintenance_active = state.maintenance_active.write().await;
                *maintenance_active = false;
            }

            // Log successful command
            command_log.device_state_after = Some(serde_json::json!({"maintenance_mode": false}));
            command_log.result = CommandResult::Success;
            command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
            
            if let Err(e) = state.audit_logger.log_command(command_log).await {
                error!("Failed to log maintenance mode disable: {}", e);
            }

            info!("Maintenance mode disabled");

            Ok(Json(MaintenanceStatusResponse {
                maintenance_mode: false,
                config_version: updated_config.maintenance.config_version,
                last_update: updated_config.maintenance.last_config_update,
                device_info: DeviceInfoResponse {
                    device_id: updated_config.device_info.device_id,
                    device_name: updated_config.device_info.device_name,
                    serial_number: updated_config.device_info.serial_number,
                    firmware_version: updated_config.device_info.firmware_version,
                    location: updated_config.device_info.location,
                },
            }))
        }
        Err(e) => {
            command_log.result = CommandResult::Failed { error: e.to_string() };
            command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
            
            if let Err(log_err) = state.audit_logger.log_command(command_log).await {
                error!("Failed to log maintenance mode disable failure: {}", log_err);
            }

            error!("Failed to disable maintenance mode: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_configuration(
    State(state): State<Arc<MaintenanceState>>,
    Query(query): Query<ConfigQuery>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let maintenance_active = *state.maintenance_active.read().await;
    if !maintenance_active {
        return Err(StatusCode::FORBIDDEN);
    }

    let config = state.current_config.read().await;
    
    let response = match query.section.as_deref() {
        Some("motion") => serde_json::to_value(&config.motion),
        Some("detector") => serde_json::to_value(&config.detector),
        Some("gpio") => serde_json::to_value(&config.gpio),
        Some("safety") => serde_json::to_value(&config.safety),
        Some("device_info") => serde_json::to_value(&config.device_info),
        Some("maintenance") => serde_json::to_value(&config.maintenance),
        _ => serde_json::to_value(&*config),
    };

    match response {
        Ok(value) => Ok(Json(value)),
        Err(e) => {
            error!("Failed to serialize configuration: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn update_configuration(
    State(state): State<Arc<MaintenanceState>>,
    Json(request): Json<UpdateConfigRequest>,
) -> Result<Json<MaintenanceStatusResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    let maintenance_active = *state.maintenance_active.read().await;
    if !maintenance_active {
        return Err(StatusCode::FORBIDDEN);
    }

    let mut command_log = CommandLog::new(
        "update_configuration",
        request.config_updates.clone(),
        Some(request.engineer_id.clone()),
        serde_json::to_value(&*state.current_config.read().await).unwrap_or_default(),
    );
    command_log.session_id = state.audit_logger.session_id().to_string();

    // Apply configuration updates
    let mut config = state.current_config.read().await.clone();
    
    if let Err(e) = apply_config_updates(&mut config, request.config_updates) {
        command_log.result = CommandResult::Failed { error: e.to_string() };
        command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        if let Err(log_err) = state.audit_logger.log_command(command_log).await {
            error!("Failed to log config update failure: {}", log_err);
        }

        error!("Failed to apply configuration updates: {}", e);
        return Err(StatusCode::BAD_REQUEST);
    }

    // Save updated configuration
    let config_manager = state.config_manager.read().await;
    match config_manager.update_config_maintenance(config, &request.password).await {
        Ok(updated_config) => {
            // Update state
            {
                let mut current_config = state.current_config.write().await;
                *current_config = updated_config.clone();
            }

            // Log successful command
            command_log.device_state_after = Some(serde_json::to_value(&updated_config).unwrap_or_default());
            command_log.result = CommandResult::Success;
            command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
            
            if let Err(e) = state.audit_logger.log_command(command_log).await {
                error!("Failed to log config update: {}", e);
            }

            info!("Configuration updated by engineer: {} (version {})", 
                  request.engineer_id, updated_config.maintenance.config_version);

            Ok(Json(MaintenanceStatusResponse {
                maintenance_mode: true,
                config_version: updated_config.maintenance.config_version,
                last_update: updated_config.maintenance.last_config_update,
                device_info: DeviceInfoResponse {
                    device_id: updated_config.device_info.device_id,
                    device_name: updated_config.device_info.device_name,
                    serial_number: updated_config.device_info.serial_number,
                    firmware_version: updated_config.device_info.firmware_version,
                    location: updated_config.device_info.location,
                },
            }))
        }
        Err(e) => {
            command_log.result = CommandResult::Failed { error: e.to_string() };
            command_log.execution_time_ms = start_time.elapsed().as_millis() as u64;
            
            if let Err(log_err) = state.audit_logger.log_command(command_log).await {
                error!("Failed to log config update failure: {}", log_err);
            }

            error!("Failed to save configuration: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn validate_configuration(
    State(_state): State<Arc<MaintenanceState>>,
    Json(config_data): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Attempt to deserialize the configuration
    match serde_json::from_value::<EnhancedDeviceConfig>(config_data) {
        Ok(_config) => {
            Ok(Json(serde_json::json!({
                "valid": true,
                "message": "Configuration is valid"
            })))
        }
        Err(e) => {
            Ok(Json(serde_json::json!({
                "valid": false,
                "error": e.to_string(),
                "message": "Configuration validation failed"
            })))
        }
    }
}

async fn get_recent_commands(
    State(state): State<Arc<MaintenanceState>>,
) -> Result<Json<Vec<CommandLog>>, StatusCode> {
    let maintenance_active = *state.maintenance_active.read().await;
    if !maintenance_active {
        return Err(StatusCode::FORBIDDEN);
    }

    match state.audit_logger.get_recent_commands(50).await {
        Ok(commands) => Ok(Json(commands)),
        Err(e) => {
            error!("Failed to retrieve recent commands: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Apply configuration updates from JSON to the config structure
fn apply_config_updates(config: &mut EnhancedDeviceConfig, updates: serde_json::Value) -> anyhow::Result<()> {
    let updates_obj = updates.as_object().ok_or_else(|| anyhow::anyhow!("Updates must be an object"))?;

    for (key, value) in updates_obj {
        match key.as_str() {
            "motion" => {
                if let Ok(motion_updates) = serde_json::from_value(value.clone()) {
                    config.motion = motion_updates;
                } else {
                    return Err(anyhow::anyhow!("Invalid motion configuration"));
                }
            }
            "detector" => {
                if let Ok(detector_updates) = serde_json::from_value(value.clone()) {
                    config.detector = detector_updates;
                } else {
                    return Err(anyhow::anyhow!("Invalid detector configuration"));
                }
            }
            "gpio" => {
                if let Ok(gpio_updates) = serde_json::from_value(value.clone()) {
                    config.gpio = gpio_updates;
                } else {
                    return Err(anyhow::anyhow!("Invalid GPIO configuration"));
                }
            }
            "safety" => {
                if let Ok(safety_updates) = serde_json::from_value(value.clone()) {
                    config.safety = safety_updates;
                } else {
                    return Err(anyhow::anyhow!("Invalid safety configuration"));
                }
            }
            "device_info" => {
                if let Ok(device_info_updates) = serde_json::from_value(value.clone()) {
                    config.device_info = device_info_updates;
                } else {
                    return Err(anyhow::anyhow!("Invalid device info configuration"));
                }
            }
            _ => {
                return Err(anyhow::anyhow!("Unknown configuration section: {}", key));
            }
        }
    }

    Ok(())
}