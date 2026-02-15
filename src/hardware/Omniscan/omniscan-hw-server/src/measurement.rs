pub mod file_converter;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{sleep, Duration};
use tracing::{info, warn};
use uuid::Uuid;

use crate::config::ServerConfig;
use crate::device::DeviceState;

// Re-export file converter types
pub use file_converter::{FileConverter, ConverterConfig, ConversionResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementRequest {
    pub file_name: Option<String>,
    pub duration_s: Option<u32>,
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMeta {
    pub id: Uuid,
    pub file_name: String,
    pub duration_s: u32,
    pub mode: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub cancelled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    pub meta: MeasurementMeta,
    pub points: Vec<DataPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub t: u32,
    pub intensity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementStatus {
    pub active: bool,
    pub meta: Option<MeasurementMeta>,
    pub result_path: Option<String>,
}

#[derive(Debug)]
pub struct MeasurementManager {
    current_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    current_meta: Arc<RwLock<Option<MeasurementMeta>>>,
    device_state: DeviceState,
    config: ServerConfig,
}

impl MeasurementManager {
    pub fn new(config: ServerConfig, device_state: DeviceState) -> Self {
        Self {
            current_task: Arc::new(Mutex::new(None)),
            current_meta: Arc::new(RwLock::new(None)),
            device_state,
            config,
        }
    }

    pub async fn start_measurement(&self, request: MeasurementRequest) -> Result<MeasurementMeta, String> {
        let (can_start, reason) = self.device_state.can_start_measurement().await;
        if !can_start {
            return Err(reason);
        }

        let mut task_guard = self.current_task.lock().await;
        if let Some(task) = task_guard.as_ref() {
            if !task.is_finished() {
                return Err("Measurement already running".to_string());
            }
        }

        let duration = request.duration_s.unwrap_or(self.config.measurement.default_duration_s);
        let file_name = request.file_name.unwrap_or_else(|| {
            format!("measure_{}", chrono::Utc::now().timestamp())
        });
        let mode = request.mode.unwrap_or_else(|| "calibrant".to_string());

        let meta = MeasurementMeta {
            id: Uuid::new_v4(),
            file_name: file_name.clone(),
            duration_s: duration,
            mode,
            started_at: Utc::now(),
            completed_at: None,
            cancelled: false,
        };

        info!("Starting measurement: {:?}", meta);

        // Set device as active
        self.device_state.set_measurement_active(true).await;

        // Store current meta
        *self.current_meta.write().await = Some(meta.clone());

        // Start simulation task
        let task = {
            let meta = meta.clone();
            let device_state = self.device_state.clone();
            let current_meta = self.current_meta.clone();
            let output_dir = self.config.measurement.output_dir.clone();

            tokio::spawn(async move {
                let result = simulate_measurement(meta, &output_dir).await;
                
                // Update completion status
                if let Some(mut current) = current_meta.write().await.as_mut() {
                    current.completed_at = Some(Utc::now());
                }
                
                // Set device as inactive
                device_state.set_measurement_active(false).await;
                
                // Store result path
                if let Ok(path) = result {
                    device_state.set_last_result(Some(path)).await;
                }
            })
        };

        *task_guard = Some(task);
        Ok(meta)
    }

    pub async fn stop_measurement(&self) -> Result<bool, String> {
        let mut task_guard = self.current_task.lock().await;
        
        if let Some(task) = task_guard.as_ref() {
            if !task.is_finished() {
                task.abort();
                
                // Mark as cancelled
                if let Some(mut current) = self.current_meta.write().await.as_mut() {
                    current.cancelled = true;
                    current.completed_at = Some(Utc::now());
                }
                
                self.device_state.set_measurement_active(false).await;
                info!("Measurement stopped/cancelled");
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    pub async fn get_status(&self) -> MeasurementStatus {
        let task_guard = self.current_task.lock().await;
        let active = if let Some(task) = task_guard.as_ref() {
            !task.is_finished()
        } else {
            false
        };

        let meta = self.current_meta.read().await.clone();
        let result_path = self.device_state.get_state().await.last_result_path;

        MeasurementStatus {
            active,
            meta,
            result_path,
        }
    }
}

async fn simulate_measurement(mut meta: MeasurementMeta, output_dir: &str) -> Result<String, String> {
    let mut points = Vec::new();
    
    info!("Simulating measurement for {} seconds", meta.duration_s);
    
    for i in 0..meta.duration_s {
        sleep(Duration::from_secs(1)).await;
        
        // Simple synthetic data: increasing intensity with some noise
        let intensity = (i as f64 * 10.0) + 100.0 + (rand::random::<f64>() * 10.0);
        points.push(DataPoint {
            t: i,
            intensity,
        });
    }
    
    meta.completed_at = Some(Utc::now());
    
    let result = MeasurementResult { meta, points };
    
    // Write to file
    let output_path = PathBuf::from(output_dir).join(format!("{}.json", result.meta.file_name));
    let json_data = serde_json::to_string_pretty(&result)
        .map_err(|e| format!("Failed to serialize result: {}", e))?;
    
    tokio::fs::write(&output_path, json_data).await
        .map_err(|e| format!("Failed to write result file: {}", e))?;
    
    info!("Measurement completed, result saved to: {}", output_path.display());
    Ok(output_path.to_string_lossy().to_string())
}