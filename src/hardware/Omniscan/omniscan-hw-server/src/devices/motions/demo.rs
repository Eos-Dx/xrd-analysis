use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

use super::{MotionControlDevice, MotionStatus, MotionError, MotionHealth, MotionLimits};
use crate::config::device_config::MotionConfig;

/// Enhanced DEMO implementation of MotionControlDevice with XY plane movement
pub struct DemoMotion {
    state: Arc<RwLock<DemoMotionState>>,
    #[allow(dead_code)]
    config: MotionConfig,
}

#[derive(Debug, Clone)]
pub struct Position2D {
    pub x: f64,
    pub y: f64,
}

struct DemoMotionState {
    powered: bool,
    status: MotionStatus,
    start_time: Instant,
    position: Option<f64>, // Current position in mm (1D for compatibility)
    target_position: Option<f64>,
    is_homed: bool,
    total_moves: u64,
    velocity: f64, // mm/s
    limits: MotionLimits,
    #[allow(dead_code)]
    current_move_task: Option<tokio::task::JoinHandle<()>>,
}

impl DemoMotion {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DemoMotionState {
                powered: false,
                status: MotionStatus::Off,
                start_time: Instant::now(),
                position: None, // Unknown until homed
                target_position: None,
                is_homed: false,
                total_moves: 0,
                velocity: 10.0, // Default 10 mm/s
                limits: MotionLimits {
                    min_position: 0.0,
                    max_position: 100.0, // 100mm travel
                    max_velocity: 50.0,   // 50 mm/s max
                    max_acceleration: 100.0, // 100 mm/s²
                },
                current_move_task: None,
            })),
            config: MotionConfig::default(), // Use default config
        }
    }
}

impl Default for DemoMotion {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MotionControlDevice for DemoMotion {
    async fn power_on(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        if state.powered {
            return Ok(());
        }
        
        state.powered = true;
        state.status = MotionStatus::Idle;
        info!("DEMO Motion: Powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        state.powered = false;
        state.status = MotionStatus::Off;
        state.position = None;
        state.is_homed = false;
        info!("DEMO Motion: Powered off");
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
            target_position: state.target_position,
            is_homed: state.is_homed,
            total_moves: state.total_moves,
            uptime: state.start_time.elapsed(),
        }
    }

    async fn home(&self) -> Result<(), MotionError> {
        {
            let state = self.state.read().await;
            if !state.powered {
                return Err(MotionError::NotPowered);
            }
            
            if state.status == MotionStatus::Moving {
                return Err(MotionError::MotionInProgress);
            }
        }

        // Set status to homing
        {
            let mut state = self.state.write().await;
            state.status = MotionStatus::Homing;
            info!("DEMO Motion: Starting homing sequence");
        }

        // Simulate homing time
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Complete homing
        {
            let mut state = self.state.write().await;
            state.status = MotionStatus::Idle;
            state.position = Some(0.0); // Home position is 0
            state.is_homed = true;
            info!("DEMO Motion: Homing completed at position 0.0mm");
        }

        Ok(())
    }

    async fn move_to(&self, position_mm: f64) -> Result<(), MotionError> {
        {
            let state = self.state.read().await;
            if !state.powered {
                return Err(MotionError::NotPowered);
            }
            
            if !state.is_homed {
                return Err(MotionError::HomingRequired);
            }
            
            if state.status == MotionStatus::Moving {
                return Err(MotionError::MotionInProgress);
            }
            
            if position_mm < state.limits.min_position || position_mm > state.limits.max_position {
                return Err(MotionError::InvalidPosition(position_mm));
            }
        }

        // Set status to moving
        {
            let mut state = self.state.write().await;
            state.status = MotionStatus::Moving;
            state.target_position = Some(position_mm);
            info!("DEMO Motion: Moving to position {}mm", position_mm);
        }

        // Simulate movement time based on distance and velocity
        let current_pos = self.get_position().await.unwrap_or(0.0);
        let distance = (position_mm - current_pos).abs();
        let velocity = self.state.read().await.velocity;
        let move_time_ms = ((distance / velocity) * 1000.0) as u64;
        
        tokio::time::sleep(Duration::from_millis(move_time_ms.max(100))).await;

        // Complete movement
        {
            let mut state = self.state.write().await;
            state.status = MotionStatus::Idle;
            state.position = Some(position_mm);
            state.target_position = None;
            state.total_moves += 1;
            info!("DEMO Motion: Reached position {}mm (total moves: {})", position_mm, state.total_moves);
        }

        Ok(())
    }

    async fn move_relative(&self, distance_mm: f64) -> Result<(), MotionError> {
        let current_position = self.get_position().await
            .ok_or(MotionError::HomingRequired)?;
        
        let target_position = current_position + distance_mm;
        self.move_to(target_position).await
    }

    async fn stop_motion(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        if state.status == MotionStatus::Moving {
            state.status = MotionStatus::Idle;
            state.target_position = None;
            info!("DEMO Motion: Motion stopped");
        }
        Ok(())
    }

    async fn get_position(&self) -> Option<f64> {
        self.state.read().await.position
    }

    async fn get_limits(&self) -> MotionLimits {
        self.state.read().await.limits.clone()
    }

    async fn set_velocity(&self, velocity_mm_s: f64) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(MotionError::NotPowered);
        }
        
        if velocity_mm_s <= 0.0 || velocity_mm_s > state.limits.max_velocity {
            return Err(MotionError::InvalidPosition(velocity_mm_s));
        }
        
        state.velocity = velocity_mm_s;
        info!("DEMO Motion: Velocity set to {}mm/s", velocity_mm_s);
        Ok(())
    }
}