use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use super::AppState;

#[derive(Deserialize)]
pub struct PowerRequest {
    pub on: bool,
}

pub async fn get_state(State(state): State<AppState>) -> Json<Value> {
    let device_info = state.device_state.get_state().await;
    Json(serde_json::to_value(&device_info).unwrap_or_default())
}

pub async fn set_power(
    State(state): State<AppState>,
    Json(payload): Json<PowerRequest>,
) -> Result<Json<Value>, StatusCode> {
    info!("Setting device power to: {}", payload.on);
    
    let device_info = state.device_state.set_power(payload.on).await;
    Ok(Json(serde_json::to_value(&device_info).unwrap_or_default()))
}