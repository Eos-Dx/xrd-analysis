use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

use super::{MotionControlDevice, MotionStatus, MotionError, MotionHealth, MotionLimits};

/// Test implementation of MotionControlDevice for unit testing
pub struct TestMotion {
    state: Arc<RwLock<TestMotionState>>,
}

struct TestMotionState {
    powered: bool,
    status: MotionStatus,
    start_time: Instant,
    position: Option<f64>,
    is_homed: bool,
    total_moves: u64,
}

impl TestMotion {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(TestMotionState {
                powered: false,
                status: MotionStatus::Off,
                start_time: Instant::now(),
                position: None,
                is_homed: false,
                total_moves: 0,
            })),
        }
    }
}

impl Default for TestMotion {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MotionControlDevice for TestMotion {
    async fn power_on(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        state.powered = true;
        state.status = MotionStatus::Idle;
        info!("Test Motion: Powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        state.powered = false;
        state.status = MotionStatus::Off;
        state.position = None;
        state.is_homed = false;
        info!("Test Motion: Powered off");
        Ok(())
    }

    async fn is_powered(&self) -> bool {
        self.state.read().await.powered
    }

    async fn get_status(&self) -> MotionStatus {
        self.state.read().await.status.clone()
    }

    async fn get_health(&self) -> MotionHealth {
        let state = self.state.read().await;
        MotionHealth {
            powered: state.powered,
            status: state.status.clone(),
            position: state.position,
            target_position: None,
            is_homed: state.is_homed,
            total_moves: state.total_moves,
            uptime: state.start_time.elapsed(),
        }
    }

    async fn home(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(MotionError::NotPowered);
        }
        
        state.position = Some(0.0);
        state.is_homed = true;
        info!("Test Motion: Homed to position 0.0");
        Ok(())
    }

    async fn move_to(&self, position_mm: f64) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(MotionError::NotPowered);
        }
        if !state.is_homed {
            return Err(MotionError::HomingRequired);
        }
        
        state.position = Some(position_mm);
        state.total_moves += 1;
        info!("Test Motion: Moved to position {}mm", position_mm);
        Ok(())
    }

    async fn move_relative(&self, distance_mm: f64) -> Result<(), MotionError> {
        let current_pos = self.get_position().await.ok_or(MotionError::HomingRequired)?;
        self.move_to(current_pos + distance_mm).await
    }

    async fn stop_motion(&self) -> Result<(), MotionError> {
        info!("Test Motion: Motion stopped");
        Ok(())
    }

    async fn get_position(&self) -> Option<f64> {
        self.state.read().await.position
    }

    async fn get_limits(&self) -> MotionLimits {
        MotionLimits {
            min_position: 0.0,
            max_position: 100.0,
            max_velocity: 50.0,
            max_acceleration: 100.0,
        }
    }

    async fn set_velocity(&self, _velocity_mm_s: f64) -> Result<(), MotionError> {
        info!("Test Motion: Velocity set");
        Ok(())
    }
}