use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::ServerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub power_on: bool,
    pub interlocks_armed: bool,
    pub emergency_stop: bool,
    pub maintenance: bool,
    pub measurement_active: bool,
    pub last_result_path: Option<String>,
}

impl DeviceInfo {
    fn new(config: &ServerConfig) -> Self {
        Self {
            power_on: config.device.power_default_on,
            interlocks_armed: config.device.interlocks_armed,
            emergency_stop: config.device.emergency_stop,
            maintenance: false,
            measurement_active: false,
            last_result_path: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceState {
    inner: Arc<RwLock<DeviceInfo>>,
}

impl DeviceState {
    pub fn new(config: &ServerConfig) -> Self {
        let device_info = DeviceInfo::new(config);
        info!("Device initialized: {}", serde_json::to_string(&device_info).unwrap_or_default());
        
        Self {
            inner: Arc::new(RwLock::new(device_info)),
        }
    }

    pub async fn get_state(&self) -> DeviceInfo {
        self.inner.read().await.clone()
    }

    pub async fn set_power(&self, on: bool) -> DeviceInfo {
        let mut state = self.inner.write().await;
        state.power_on = on;
        info!("Device power set to: {}", on);
        state.clone()
    }

    pub async fn set_maintenance(&self, maintenance: bool) -> DeviceInfo {
        let mut state = self.inner.write().await;
        state.maintenance = maintenance;
        info!("Maintenance mode set to: {}", maintenance);
        state.clone()
    }

    pub async fn set_measurement_active(&self, active: bool) -> DeviceInfo {
        let mut state = self.inner.write().await;
        state.measurement_active = active;
        info!("Measurement active set to: {}", active);
        state.clone()
    }

    pub async fn set_last_result(&self, path: Option<String>) -> DeviceInfo {
        let mut state = self.inner.write().await;
        state.last_result_path = path.clone();
        if let Some(ref p) = path {
            info!("Last result path set to: {}", p);
        }
        state.clone()
    }

    pub async fn can_start_measurement(&self) -> (bool, String) {
        let state = self.inner.read().await;
        
        if state.emergency_stop {
            return (false, "Emergency stop engaged".to_string());
        }
        
        if !state.power_on {
            return (false, "Device power is off".to_string());
        }
        
        if !state.interlocks_armed {
            return (false, "Interlocks not armed".to_string());
        }
        
        if state.measurement_active {
            return (false, "Measurement already running".to_string());
        }
        
        (true, "Ready".to_string())
    }
}