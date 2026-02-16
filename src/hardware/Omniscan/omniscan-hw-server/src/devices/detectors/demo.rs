use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time;
use tracing::{info, error};

use super::{DetectorDevice, DetectorStatus, DetectorError, DetectorHealth, ExposureResult};

/// DEMO implementation of DetectorDevice for simulation
pub struct DemoDetector {
    pub state: Arc<RwLock<DemoDetectorState>>,
    exposure_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

pub struct DemoDetectorState {
    pub powered: bool,
    pub status: DetectorStatus,
    pub start_time: Instant,
    pub temperature: f32,
    pub voltage: f32,
    pub total_exposures: u64,
    pub last_exposure_time: Option<u32>,
    pub last_result: Option<ExposureResult>,
}

impl DemoDetector {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DemoDetectorState {
                powered: false,
                status: DetectorStatus::Off,
                start_time: Instant::now(),
                temperature: 22.5, // Room temperature
                voltage: 0.0,
                total_exposures: 0,
                last_exposure_time: None,
                last_result: None,
            })),
            exposure_task: Arc::new(Mutex::new(None)),
        }
    }

    async fn simulate_exposure(&self, exposure_time_ms: u32) {
        let state_clone = Arc::clone(&self.state);
        
        // Set status to exposing
        {
            let mut state = state_clone.write().await;
            state.status = DetectorStatus::Exposing;
            info!("DEMO Detector: Starting exposure for {}ms", exposure_time_ms);
        }

        // Wait for exposure time
        time::sleep(Duration::from_millis(exposure_time_ms as u64)).await;

        // Simulate reading phase
        {
            let mut state = state_clone.write().await;
            state.status = DetectorStatus::Reading;
            info!("DEMO Detector: Reading data...");
        }

        // Simulate read time (100ms)
        time::sleep(Duration::from_millis(100)).await;

        // Complete exposure
        {
            let mut state = state_clone.write().await;
            state.status = DetectorStatus::Idle;
            state.total_exposures += 1;
            state.last_exposure_time = Some(exposure_time_ms);
            
            // Simulate temperature rise during exposure
            state.temperature += 0.5;
            
            let result = ExposureResult {
                exposure_time_ms,
                timestamp: chrono::Utc::now(),
                data_size: (exposure_time_ms as usize) * 1024, // Simulate data size
                data_path: Some(format!("/data/exposure_{}.raw", state.total_exposures)),
                detector_temp: state.temperature,
            };
            
            state.last_result = Some(result);
            info!("DEMO Detector: Exposure completed (total: {})", state.total_exposures);
        }
    }
}

impl Default for DemoDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DetectorDevice for DemoDetector {
    async fn power_on(&self) -> Result<(), DetectorError> {
        let mut state = self.state.write().await;
        if state.powered {
            return Ok(());
        }
        
        state.powered = true;
        state.status = DetectorStatus::Idle;
        state.voltage = 12.0; // Simulate power voltage
        state.temperature = 25.0; // Slight warming when powered
        info!("DEMO Detector: Powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), DetectorError> {
        // Stop any ongoing exposure
        let mut task_guard = self.exposure_task.lock().await;
        if let Some(task) = task_guard.take() {
            task.abort();
        }

        let mut state = self.state.write().await;
        state.powered = false;
        state.status = DetectorStatus::Off;
        state.voltage = 0.0;
        state.temperature = 22.5; // Return to room temperature
        info!("DEMO Detector: Powered off");
        Ok(())
    }

    async fn is_powered(&self) -> bool {
        self.state.read().await.powered
    }

    async fn get_status(&self) -> DetectorStatus {
        self.state.read().await.status.clone()
    }

    async fn get_health(&self) -> DetectorHealth {
        let state = self.state.read().await;
        DetectorHealth {
            powered: state.powered,
            temperature: state.temperature,
            voltage: state.voltage,
            status: state.status.clone(),
            last_exposure_time: state.last_exposure_time,
            total_exposures: state.total_exposures,
            uptime: state.start_time.elapsed(),
        }
    }

    async fn start_exposure(&self, exposure_time_ms: u32) -> Result<(), DetectorError> {
        if exposure_time_ms == 0 || exposure_time_ms > 300_000 {
            return Err(DetectorError::InvalidExposureTime(exposure_time_ms));
        }

        {
            let state = self.state.read().await;
            if !state.powered {
                return Err(DetectorError::NotPowered);
            }
            
            if matches!(state.status, DetectorStatus::Exposing | DetectorStatus::Reading) {
                return Err(DetectorError::ExposureInProgress);
            }
        }

        // Cancel any existing exposure task
        let mut task_guard = self.exposure_task.lock().await;
        if let Some(task) = task_guard.take() {
            task.abort();
        }

        // Start new exposure task
        let detector_clone = DemoDetector {
            state: Arc::clone(&self.state),
            exposure_task: Arc::clone(&self.exposure_task),
        };

        let handle = tokio::spawn(async move {
            detector_clone.simulate_exposure(exposure_time_ms).await;
        });

        *task_guard = Some(handle);
        Ok(())
    }

    async fn stop_exposure(&self) -> Result<(), DetectorError> {
        let mut task_guard = self.exposure_task.lock().await;
        if let Some(task) = task_guard.take() {
            task.abort();
            
            let mut state = self.state.write().await;
            state.status = DetectorStatus::Idle;
            info!("DEMO Detector: Exposure stopped");
        }
        Ok(())
    }

    async fn get_last_result(&self) -> Option<ExposureResult> {
        self.state.read().await.last_result.clone()
    }

    async fn calibrate(&self) -> Result<(), DetectorError> {
        let state = self.state.read().await;
        if !state.powered {
            return Err(DetectorError::NotPowered);
        }
        drop(state);

        info!("DEMO Detector: Starting calibration measurement...");
        
        // Hardcoded measurement duration for calibration: 5 seconds for dev (300 sec in production)
        let calibration_duration_ms = 5000; // Development: 5 seconds (production would be 300_000)
        time::sleep(Duration::from_millis(calibration_duration_ms)).await;
        
        // Simulate 10% chance of calibration failure
        if rand::random::<f32>() < 0.1 {
            error!("DEMO Detector: Calibration failed");
            return Err(DetectorError::CalibrationError("Random simulation failure".to_string()));
        }

        info!("DEMO Detector: Calibration completed successfully");
        Ok(())
    }
}