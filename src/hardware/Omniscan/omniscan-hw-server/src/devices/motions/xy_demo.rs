use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tracing::info;

use super::{MotionControlDevice, MotionStatus, MotionError, MotionHealth, MotionLimits};
use crate::config::device_config::MotionConfig;

/// Enhanced XY Motion DEMO implementation with realistic movement simulation
pub struct XYDemoMotion {
    pub state: Arc<RwLock<XYMotionState>>,
    pub config: Arc<RwLock<MotionConfig>>,
    movement_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

#[derive(Debug, Clone)]
pub struct Position2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone)]
pub struct Velocity2D {
    #[allow(dead_code)]
    pub x: f64,
    #[allow(dead_code)]
    pub y: f64,
}

pub struct XYMotionState {
    pub powered: bool,
    pub status: MotionStatus,
    pub start_time: Instant,
    pub position: Option<Position2D>,
    pub target_position: Option<Position2D>,
    #[allow(dead_code)]
    pub velocity: Velocity2D,
    pub is_homed: bool,
    pub x_homed: bool,
    pub y_homed: bool,
    pub total_moves: u64,
    pub current_move_id: Option<String>,
    pub last_move_time: Option<Instant>,
}

impl XYDemoMotion {
    pub fn new(config: MotionConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(XYMotionState {
                powered: false,
                status: MotionStatus::Off,
                start_time: Instant::now(),
                position: None,
                target_position: None,
                velocity: Velocity2D { x: 0.0, y: 0.0 },
                is_homed: false,
                x_homed: false,
                y_homed: false,
                total_moves: 0,
                current_move_id: None,
                last_move_time: None,
            })),
            config: Arc::new(RwLock::new(config)),
            movement_task: Arc::new(Mutex::new(None)),
        }
    }

    /// Move to XY position
    pub async fn move_to_xy(&self, x: f64, y: f64) -> Result<(), MotionError> {
        let config = self.config.read().await;
        let mut state = self.state.write().await;

        if !state.powered {
            return Err(MotionError::NotPowered);
        }

        if !state.is_homed {
            return Err(MotionError::HomingRequired);
        }

        if state.status == MotionStatus::Moving {
            return Err(MotionError::MotionInProgress);
        }

        // Check limits
        if config.x_axis.enabled {
            if x < config.x_axis.min_position || x > config.x_axis.max_position {
                return Err(MotionError::InvalidPosition(x));
            }
        }

        if config.y_axis.enabled {
            if y < config.y_axis.min_position || y > config.y_axis.max_position {
                return Err(MotionError::InvalidPosition(y));
            }
        }

        let target = Position2D { x, y };
        state.target_position = Some(target.clone());
        state.status = MotionStatus::Moving;
        state.current_move_id = Some(uuid::Uuid::new_v4().to_string());

        info!("XY Motion: Moving to position ({:.3}, {:.3})", x, y);

        // Start movement task
        let motion_clone = Self {
            state: Arc::clone(&self.state),
            config: Arc::clone(&self.config),
            movement_task: Arc::clone(&self.movement_task),
        };

        let mut task_guard = self.movement_task.lock().await;
        if let Some(old_task) = task_guard.take() {
            old_task.abort();
        }

        let handle = tokio::spawn(async move {
            motion_clone.simulate_xy_movement(target).await;
        });

        *task_guard = Some(handle);
        drop(task_guard);
        drop(state);

        Ok(())
    }

    /// Home both axes
    /// Homing is required before calibration and any measurements
    pub async fn home_xy(&self) -> Result<(), MotionError> {
        let config = self.config.read().await;
        let mut state = self.state.write().await;

        if !state.powered {
            return Err(MotionError::NotPowered);
        }

        if state.status == MotionStatus::Moving {
            return Err(MotionError::MotionInProgress);
        }

        state.status = MotionStatus::Homing;
        state.x_homed = false;
        state.y_homed = false;
        state.is_homed = false;

        info!("XY Motion: Starting homing sequence");
        drop(state);
        drop(config);

        // Hardcoded homing time - both axes simultaneously: 2 seconds
        let home_time = 2000; // ms
        tokio::time::sleep(Duration::from_millis(home_time)).await;

        let config = self.config.read().await;
        let mut state = self.state.write().await;

        // Set to home positions
        state.position = Some(Position2D {
            x: config.x_axis.home_position,
            y: config.y_axis.home_position,
        });
        
        state.x_homed = config.x_axis.enabled;
        state.y_homed = config.y_axis.enabled;
        state.is_homed = state.x_homed && state.y_homed;
        state.status = MotionStatus::Idle;

        info!("XY Motion: Homing completed at ({:.3}, {:.3})", 
              config.x_axis.home_position, config.y_axis.home_position);

        Ok(())
    }

    /// Simulate realistic XY movement with acceleration
    async fn simulate_xy_movement(&self, target: Position2D) {
        let config = self.config.read().await.clone();
        let start_pos = {
            let state = self.state.read().await;
            state.position.clone().unwrap_or(Position2D { x: 0.0, y: 0.0 })
        };

        // Calculate movement parameters
        let dx = target.x - start_pos.x;
        let dy = target.y - start_pos.y;
        let distance = (dx * dx + dy * dy).sqrt();
        
        if distance < 0.001 {
            // Already at target
            let mut state = self.state.write().await;
            state.status = MotionStatus::Idle;
            state.target_position = None;
            return;
        }

        // Calculate movement time based on trapezoidal motion profile
        let max_vel = config.max_speed.min(config.x_axis.max_velocity.min(config.y_axis.max_velocity));
        let acceleration = config.acceleration;
        
        // Trapezoidal motion profile calculation
        let accel_time = max_vel / acceleration;
        let accel_distance = 0.5 * acceleration * accel_time * accel_time;
        
        let (total_time, cruise_time) = if distance <= 2.0 * accel_distance {
            // Triangular profile (no cruise phase)
            let time = (2.0 * distance / acceleration).sqrt();
            (time, 0.0)
        } else {
            // Trapezoidal profile
            let cruise_distance = distance - 2.0 * accel_distance;
            let cruise_time = cruise_distance / max_vel;
            (2.0 * accel_time + cruise_time, cruise_time)
        };

        info!("XY Motion: Movement profile - distance: {:.3}mm, time: {:.3}s", distance, total_time);

        // Simulate movement in steps
        let steps = ((total_time * 50.0) as usize).max(10); // 50 Hz update rate
        let step_time = Duration::from_millis((total_time * 1000.0 / steps as f64) as u64);

        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let progress = self.calculate_motion_progress(t, total_time, accel_time, cruise_time, max_vel, acceleration);
            
            let current_pos = Position2D {
                x: start_pos.x + dx * progress,
                y: start_pos.y + dy * progress,
            };

            // Update position
            {
                let mut state = self.state.write().await;
                state.position = Some(current_pos.clone());
                state.last_move_time = Some(Instant::now());
            }

            tokio::time::sleep(step_time).await;
        }

        // Complete movement
        let mut state = self.state.write().await;
        state.position = Some(target.clone());
        state.target_position = None;
        state.status = MotionStatus::Idle;
        state.total_moves += 1;
        state.current_move_id = None;

        info!("XY Motion: Reached target position ({:.3}, {:.3}) - total moves: {}", 
              target.x, target.y, state.total_moves);
    }

    /// Calculate motion progress using trapezoidal profile
    fn calculate_motion_progress(&self, t: f64, total_time: f64, accel_time: f64, cruise_time: f64, max_vel: f64, acceleration: f64) -> f64 {
        let current_time = t * total_time;
        
        if current_time <= accel_time {
            // Acceleration phase
            0.5 * acceleration * current_time * current_time / (accel_time * max_vel)
        } else if current_time <= accel_time + cruise_time {
            // Cruise phase
            let accel_progress = 0.5;
            let cruise_progress = (current_time - accel_time) / total_time;
            accel_progress + cruise_progress
        } else {
            // Deceleration phase
            let decel_time = current_time - accel_time - cruise_time;
            let remaining_time = total_time - accel_time - cruise_time;
            let decel_progress = (max_vel * decel_time - 0.5 * acceleration * decel_time * decel_time) / (remaining_time * max_vel);
            0.5 + cruise_time / total_time + decel_progress
        }
    }

    /// Get current XY position
    pub async fn get_xy_position(&self) -> Option<Position2D> {
        self.state.read().await.position.clone()
    }

    /// Update motion configuration
    #[allow(dead_code)]
    pub async fn update_config(&self, new_config: MotionConfig) {
        let mut config = self.config.write().await;
        *config = new_config;
        info!("XY Motion: Configuration updated");
    }
}

impl Default for XYDemoMotion {
    fn default() -> Self {
        Self::new(MotionConfig {
            name: "XY Demo Motion".to_string(),
            enabled: true,
            x_axis: crate::config::device_config::AxisConfig {
                enabled: true,
                min_position: 0.0,
                max_position: 100.0,
                home_position: 0.0,
                steps_per_mm: 1000.0,
                max_velocity: 50.0,
                acceleration: 100.0,
            },
            y_axis: crate::config::device_config::AxisConfig {
                enabled: true,
                min_position: 0.0,
                max_position: 100.0,
                home_position: 0.0,
                steps_per_mm: 1000.0,
                max_velocity: 50.0,
                acceleration: 100.0,
            },
            homing_speed: 10.0,
            max_speed: 50.0,
            acceleration: 100.0,
            backlash_compensation: 0.01,
        })
    }
}

#[async_trait]
impl MotionControlDevice for XYDemoMotion {
    async fn power_on(&self) -> Result<(), MotionError> {
        let mut state = self.state.write().await;
        if state.powered {
            return Ok(());
        }
        
        state.powered = true;
        state.status = MotionStatus::Idle;
        info!("XY Motion: Powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), MotionError> {
        // Stop any ongoing movement
        let mut task_guard = self.movement_task.lock().await;
        if let Some(task) = task_guard.take() {
            task.abort();
        }

        let mut state = self.state.write().await;
        state.powered = false;
        state.status = MotionStatus::Off;
        state.position = None;
        state.is_homed = false;
        state.x_homed = false;
        state.y_homed = false;
        info!("XY Motion: Powered off");
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
            position: state.position.as_ref().map(|p| p.x), // Return X position for compatibility
            target_position: state.target_position.as_ref().map(|p| p.x),
            is_homed: state.is_homed,
            total_moves: state.total_moves,
            uptime: state.start_time.elapsed(),
        }
    }

    async fn home(&self) -> Result<(), MotionError> {
        self.home_xy().await
    }

    async fn move_to(&self, position_mm: f64) -> Result<(), MotionError> {
        // For compatibility, move only X axis
        let current_pos = self.get_xy_position().await.unwrap_or(Position2D { x: 0.0, y: 0.0 });
        self.move_to_xy(position_mm, current_pos.y).await
    }

    async fn move_relative(&self, distance_mm: f64) -> Result<(), MotionError> {
        let current_pos = self.get_xy_position().await.ok_or(MotionError::HomingRequired)?;
        self.move_to_xy(current_pos.x + distance_mm, current_pos.y).await
    }

    async fn stop_motion(&self) -> Result<(), MotionError> {
        let mut task_guard = self.movement_task.lock().await;
        if let Some(task) = task_guard.take() {
            task.abort();
        }

        let mut state = self.state.write().await;
        if state.status == MotionStatus::Moving {
            state.status = MotionStatus::Idle;
            state.target_position = None;
            state.current_move_id = None;
            info!("XY Motion: Motion stopped");
        }
        Ok(())
    }

    async fn get_position(&self) -> Option<f64> {
        self.state.read().await.position.as_ref().map(|p| p.x)
    }

    async fn get_limits(&self) -> MotionLimits {
        let config = self.config.read().await;
        MotionLimits {
            min_position: config.x_axis.min_position,
            max_position: config.x_axis.max_position,
            max_velocity: config.x_axis.max_velocity,
            max_acceleration: config.x_axis.acceleration,
        }
    }

    async fn set_velocity(&self, velocity_mm_s: f64) -> Result<(), MotionError> {
        let mut config = self.config.write().await;
        let limits = MotionLimits {
            min_position: config.x_axis.min_position,
            max_position: config.x_axis.max_position,
            max_velocity: config.max_speed,
            max_acceleration: config.acceleration,
        };
        
        if !self.is_powered().await {
            return Err(MotionError::NotPowered);
        }
        
        if velocity_mm_s <= 0.0 || velocity_mm_s > limits.max_velocity {
            return Err(MotionError::InvalidPosition(velocity_mm_s));
        }
        
        config.max_speed = velocity_mm_s;
        info!("XY Motion: Velocity set to {}mm/s", velocity_mm_s);
        Ok(())
    }
}