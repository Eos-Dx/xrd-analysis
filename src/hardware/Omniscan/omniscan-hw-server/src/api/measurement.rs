use axum::{extract::State, http::StatusCode, response::Json};
use serde_json::Value;
use tracing::{info, warn};

use crate::measurement::MeasurementRequest;

use super::AppState;

pub async fn start_measurement(
    State(state): State<AppState>,
    Json(payload): Json<MeasurementRequest>,
) -> Result<Json<Value>, StatusCode> {
    info!("Starting measurement: {:?}", payload);
    
    match state.measurement_manager.start_measurement(payload).await {
        Ok(meta) => {
            info!("Measurement started successfully: {}", meta.id);
            Ok(Json(serde_json::json!({
                "scheduled": true,
                "meta": meta
            })))
        }
        Err(e) => {
            warn!("Failed to start measurement: {}", e);
            Err(StatusCode::CONFLICT)
        }
    }
}

pub async fn stop_measurement(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    info!("Stopping measurement");
    
    match state.measurement_manager.stop_measurement().await {
        Ok(stopped) => {
            Ok(Json(serde_json::json!({
                "stopped": stopped,
                "reason": if stopped { "cancelled" } else { "no-active" }
            })))
        }
        Err(e) => {
            warn!("Failed to stop measurement: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_status(State(state): State<AppState>) -> Json<Value> {
    let status = state.measurement_manager.get_status().await;
    Json(serde_json::to_value(&status).unwrap_or_default())
}

pub async fn get_result(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    let device_info = state.device_state.get_state().await;
    
    if let Some(result_path) = device_info.last_result_path {
        match tokio::fs::read_to_string(&result_path).await {
            Ok(content) => {
                match serde_json::from_str::<Value>(&content) {
                    Ok(json_data) => Ok(Json(json_data)),
                    Err(e) => {
                        warn!("Failed to parse result file {}: {}", result_path, e);
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                }
            }
            Err(e) => {
                warn!("Failed to read result file {}: {}", result_path, e);
                Err(StatusCode::NOT_FOUND)
            }
        }
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}