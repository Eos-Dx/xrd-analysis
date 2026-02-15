use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{info, warn};

use crate::{
    config::ServerConfig,
    device::{DeviceInfo, DeviceState},
    measurement::{MeasurementManager, MeasurementRequest, MeasurementResult, MeasurementStatus},
};

pub mod device;
pub mod maintenance;
pub mod measurement;

// App state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub config: ServerConfig,
    pub device_state: DeviceState,
    pub measurement_manager: Arc<MeasurementManager>,
}

pub fn create_router(config: ServerConfig, device_state: DeviceState) -> Router {
    let measurement_manager = Arc::new(MeasurementManager::new(config.clone(), device_state.clone()));
    
    let state = AppState {
        config: config.clone(),
        device_state,
        measurement_manager,
    };

    // API v1 routes
    let api_v1 = Router::new()
        // Device endpoints
        .route("/device/state", get(device::get_state))
        .route("/device/power", post(device::set_power))
        
        // Measurement endpoints
        .route("/measure/start", post(measurement::start_measurement))
        .route("/measure/stop", post(measurement::stop_measurement))
        .route("/measure/status", get(measurement::get_status))
        .route("/measure/result", get(measurement::get_result))
        
        
        // Config endpoints
        .route("/config", get(get_config))
        .route("/config", put(set_config))
        .route("/config/validate", post(validate_config))
        
        .with_state(state);

    // Mount API under /api/v1
    Router::new()
        .nest("/api/v1", api_v1)
        .route("/health", get(health_check))
}

// Health check endpoint
async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let safety_sm = state.device_state.safety_state_machine.read().await;
    let calibration_is_valid = safety_sm.is_calibration_valid();
    let last_calibration = safety_sm.get_last_calibration();
    let time_until_expiry_hours = safety_sm.get_time_until_expiry_hours();
    let interlocks = safety_sm.get_interlock_status().await;
    drop(safety_sm);
    
    // Check device health and initialization status
    let detector_health = state.device_state.detector.get_health().await;
    let motion_health = state.device_state.motion.get_health().await;
    let gpio_interlocks = state.device_state.gpio.get_interlocks().await;
    
    // Hardware is ready only if:
    // 1. All devices are powered AND initialized AND ready to operate (Idle status)
    // 2. Motion: powered and Idle (homing will be done during calibration)
    // 3. All safety interlocks pass
    let detector_ready = detector_health.powered && matches!(detector_health.status, crate::devices::DetectorStatus::Idle);
    let motion_ready = motion_health.powered && matches!(motion_health.status, crate::devices::motions::MotionStatus::Idle);
    let all_devices_ready = detector_ready && motion_ready;
    let safety_checks_pass = interlocks.overall_safe;
    let is_hardware_ready = all_devices_ready && safety_checks_pass;
    
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "isHardwareReady": is_hardware_ready,
        "devices": {
            "detector": {
                "powered": detector_health.powered,
                "status": format!("{:?}", detector_health.status),
                "temperature": detector_health.temperature,
                "voltage": detector_health.voltage,
                "total_exposures": detector_health.total_exposures
            },
            "motion": {
                "powered": motion_health.powered,
                "status": format!("{:?}", motion_health.status),
                "is_homed": motion_health.is_homed,
                "total_moves": motion_health.total_moves
            }
        },
        "safety": {
            "overall_safe": interlocks.overall_safe,
            "emergency_stop": interlocks.emergency_stop,
            "door_closed": interlocks.door_closed,
            "beam_watchdog": interlocks.beam_watchdog,
            "violation_reason": interlocks.violation_reason
        },
        "calibration": {
            "valid": calibration_is_valid,
            "last_calibration_time": last_calibration.map(|dt| dt.to_rfc3339()),
            "hours_until_expiry": time_until_expiry_hours,
            "requires_calibration": !calibration_is_valid
        }
    }))
}

// Config endpoints
async fn get_config(State(state): State<AppState>) -> Json<Value> {
    Json(serde_json::to_value(&state.config).unwrap_or_default())
}

#[derive(Deserialize)]
struct ConfigRequest {
    #[serde(flatten)]
    config: Value,
}

async fn set_config(
    State(mut state): State<AppState>,
    Json(payload): Json<ConfigRequest>,
) -> Result<Json<Value>, StatusCode> {
    match serde_json::from_value::<ServerConfig>(payload.config) {
        Ok(new_config) => {
            if let Err(e) = new_config.validate() {
                warn!("Invalid config provided: {}", e);
                return Err(StatusCode::BAD_REQUEST);
            }
            
            state.config = new_config.clone();
            info!("Configuration updated successfully");
            Ok(Json(serde_json::to_value(&new_config).unwrap_or_default()))
        }
        Err(e) => {
            warn!("Failed to parse config: {}", e);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}

async fn validate_config(Json(payload): Json<ConfigRequest>) -> Result<Json<Value>, StatusCode> {
    match serde_json::from_value::<ServerConfig>(payload.config) {
        Ok(config) => {
            match config.validate() {
                Ok(_) => Ok(Json(json!({"valid": true}))),
                Err(e) => Ok(Json(json!({"valid": false, "error": e.to_string()}))),
            }
        }
        Err(e) => Ok(Json(json!({"valid": false, "error": e.to_string()}))),
    }
}