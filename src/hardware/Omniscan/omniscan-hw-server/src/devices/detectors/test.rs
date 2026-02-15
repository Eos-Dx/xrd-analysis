use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

use super::{DetectorDevice, DetectorStatus, DetectorError, DetectorHealth, ExposureResult};

/// Test implementation of DetectorDevice for unit testing
pub struct TestDetector {
    state: Arc<RwLock<TestDetectorState>>,
}

struct TestDetectorState {
    powered: bool,
    status: DetectorStatus,
    start_time: Instant,
    total_exposures: u64,
    last_result: Option<ExposureResult>,
}

impl TestDetector {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(TestDetectorState {
                powered: false,
                status: DetectorStatus::Off,
                start_time: Instant::now(),
                total_exposures: 0,
                last_result: None,
            })),
        }
    }
}

impl Default for TestDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DetectorDevice for TestDetector {
    async fn power_on(&self) -> Result<(), DetectorError> {
        let mut state = self.state.write().await;
        state.powered = true;
        state.status = DetectorStatus::Idle;
        info!("Test Detector: Powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), DetectorError> {
        let mut state = self.state.write().await;
        state.powered = false;
        state.status = DetectorStatus::Off;
        info!("Test Detector: Powered off");
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
            temperature: 23.0,
            voltage: if state.powered { 12.0 } else { 0.0 },
            status: state.status.clone(),
            last_exposure_time: None,
            total_exposures: state.total_exposures,
            uptime: state.start_time.elapsed(),
        }
    }

    async fn start_exposure(&self, exposure_time_ms: u32) -> Result<(), DetectorError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(DetectorError::NotPowered);
        }
        
        state.status = DetectorStatus::Exposing;
        info!("Test Detector: Starting exposure for {}ms", exposure_time_ms);
        
        // Simulate instant completion for tests
        state.status = DetectorStatus::Idle;
        state.total_exposures += 1;
        
        let result = ExposureResult {
            exposure_time_ms,
            timestamp: chrono::Utc::now(),
            data_size: exposure_time_ms as usize,
            data_path: Some(format!("/test/exposure_{}.raw", state.total_exposures)),
            detector_temp: 23.5,
        };
        state.last_result = Some(result);
        
        Ok(())
    }

    async fn stop_exposure(&self) -> Result<(), DetectorError> {
        let mut state = self.state.write().await;
        state.status = DetectorStatus::Idle;
        info!("Test Detector: Exposure stopped");
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
        info!("Test Detector: Calibration completed");
        Ok(())
    }
}