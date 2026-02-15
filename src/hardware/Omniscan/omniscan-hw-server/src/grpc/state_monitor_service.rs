use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::grpc::hub::v1::*;
use crate::grpc::services::ServiceState;

pub struct StateMonitorService {
    state: Arc<ServiceState>,
}

impl StateMonitorService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl state_monitor_server::StateMonitor for StateMonitorService {
    async fn get_gpio_state(&self, _request: Request<Empty>) -> Result<Response<GpioStateResponse>, Status> {
        let gpio = &self.state.gpio;
        
        // Get current GPIO state
        let powered = gpio.is_powered().await;
        let key_switch_on = gpio.get_key_switch_state().await.unwrap_or(false);
        let activation_button_active = gpio.get_activation_button_active().await.unwrap_or(false);
        let interlocks = gpio.get_interlocks().await;
        
        // Get LED states - for now use simple values since downcast is complex
        // TODO: Add LED getters to GpioDevice trait for production
        let main_led = if key_switch_on { "Green".to_string() } else { "Red".to_string() };
        let radiation_led = if interlocks.overall_safe { "Green".to_string() } else { "Red".to_string() };
        
        // Get activation remaining time
        let activation_remaining_secs = gpio.get_activation_button_remaining_time().await;
        
        let response = GpioStateResponse {
            powered,
            key_switch_on,
            activation_button_active,
            activation_remaining_secs: activation_remaining_secs.map(|s| s as u32),
            interlocks: Some(InterlockStatus {
                emergency_stop: interlocks.emergency_stop,
                door_closed: interlocks.door_closed,
                radiation_safe: interlocks.radiation_safe,
                cooling_ok: interlocks.cooling_ok,
                power_ok: interlocks.power_ok,
                overall_safe: interlocks.overall_safe,
                violation_reason: String::new(),
                enable_button: activation_button_active,
                key_switch: key_switch_on,
            }),
            main_led,
            radiation_led,
        };
        
        // Log removed to reduce noise
        Ok(Response::new(response))
    }
    
    async fn get_full_server_state(&self, _request: Request<Empty>) -> Result<Response<FullServerStateResponse>, Status> {
        let safety_sm = self.state.safety_state_machine.read().await;
        let safety_state = safety_sm.get_current_state().await;
        
        // Get GPIO state
        let gpio_response = self.get_gpio_state(Request::new(Empty {})).await?.into_inner();
        
        // Get detector state
        let detector = &self.state.detector;
        let detector_health = detector.get_health().await;
        
        // Get motion state
        let motion = &self.state.motion;
        let motion_health = motion.get_health().await;
        let position = motion.get_position().await;
        
        // Convert detector status to proto enum
        let detector_status = match detector_health.status {
            crate::devices::detectors::DetectorStatus::Off => DetectorStatus::DetectorOff as i32,
            crate::devices::detectors::DetectorStatus::Init => DetectorStatus::DetectorInit as i32,
            crate::devices::detectors::DetectorStatus::Idle => DetectorStatus::DetectorIdle as i32,
            crate::devices::detectors::DetectorStatus::Exposing => DetectorStatus::DetectorExposing as i32,
            crate::devices::detectors::DetectorStatus::Reading => DetectorStatus::DetectorReading as i32,
            crate::devices::detectors::DetectorStatus::Error(_) => DetectorStatus::DetectorError as i32,
        };
        
        // Convert motion status to proto enum
        let motion_status = match motion_health.status {
            crate::devices::motions::MotionStatus::Off => MotionStatus::MotionOff as i32,
            crate::devices::motions::MotionStatus::Init => MotionStatus::MotionInit as i32,
            crate::devices::motions::MotionStatus::Idle => MotionStatus::MotionIdle as i32,
            crate::devices::motions::MotionStatus::Moving => MotionStatus::MotionMoving as i32,
            crate::devices::motions::MotionStatus::Homing => MotionStatus::MotionHoming as i32,
            crate::devices::motions::MotionStatus::Error(_) => MotionStatus::MotionError as i32,
            crate::devices::motions::MotionStatus::LimitHit => MotionStatus::MotionLimitHit as i32,
        };
        
        let response = FullServerStateResponse {
            safety_state: crate::grpc::services::safety_state_to_proto(safety_state) as i32,
            gpio: Some(gpio_response),
            detector: Some(DetectorStateResponse {
                powered: detector_health.powered,
                initialized: false, // TODO: Add initialized field to DetectorHealth
                status: detector_status,
                temperature: detector_health.temperature,
                total_exposures: detector_health.total_exposures,
            }),
            motion: Some(MotionStateResponse {
                powered: motion_health.powered,
                initialized: false, // TODO: Add initialized field to MotionHealth
                is_homed: motion_health.is_homed,
                status: motion_status,
                position_x: position,
                position_y: None, // Single axis for now
                total_moves: motion_health.total_moves,
            }),
            timestamp: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };
        
        // Log removed to reduce noise
        Ok(Response::new(response))
    }
    
    type SubscribeToStateUpdatesStream = tokio_stream::wrappers::ReceiverStream<Result<StateChangeNotification, Status>>;
    
    async fn subscribe_to_state_updates(&self, _request: Request<Empty>) -> Result<Response<Self::SubscribeToStateUpdatesStream>, Status> {
        // Subscribe to the broadcast channel
        let mut rx_broadcast = self.state.state_notifications.subscribe();
        
        // Create mpsc channel for tonic stream
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        // Spawn task to forward broadcast messages to stream
        tokio::spawn(async move {
            loop {
                match rx_broadcast.recv().await {
                    Ok(notification) => {
                        if tx.send(Ok(notification)).await.is_err() {
                            // Client disconnected
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!("Client lagged, skipped {} notifications", skipped);
                        // Continue receiving
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        // Channel closed, end stream
                        break;
                    }
                }
            }
            tracing::debug!("State update subscription ended");
        });
        
        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
    
    async fn get_detector_state(&self, _request: Request<Empty>) -> Result<Response<DetectorStateResponse>, Status> {
        let detector = &self.state.detector;
        let health = detector.get_health().await;
        
        let status = match health.status {
            crate::devices::detectors::DetectorStatus::Off => DetectorStatus::DetectorOff as i32,
            crate::devices::detectors::DetectorStatus::Init => DetectorStatus::DetectorInit as i32,
            crate::devices::detectors::DetectorStatus::Idle => DetectorStatus::DetectorIdle as i32,
            crate::devices::detectors::DetectorStatus::Exposing => DetectorStatus::DetectorExposing as i32,
            crate::devices::detectors::DetectorStatus::Reading => DetectorStatus::DetectorReading as i32,
            crate::devices::detectors::DetectorStatus::Error(_) => DetectorStatus::DetectorError as i32,
        };
        
        Ok(Response::new(DetectorStateResponse {
            powered: health.powered,
            initialized: false, // TODO: Add initialized field to DetectorHealth
            status,
            temperature: health.temperature,
            total_exposures: health.total_exposures,
        }))
    }
    
    async fn get_motion_state(&self, _request: Request<Empty>) -> Result<Response<MotionStateResponse>, Status> {
        let motion = &self.state.motion;
        let health = motion.get_health().await;
        let position = motion.get_position().await;
        
        let status = match health.status {
            crate::devices::motions::MotionStatus::Off => MotionStatus::MotionOff as i32,
            crate::devices::motions::MotionStatus::Init => MotionStatus::MotionInit as i32,
            crate::devices::motions::MotionStatus::Idle => MotionStatus::MotionIdle as i32,
            crate::devices::motions::MotionStatus::Moving => MotionStatus::MotionMoving as i32,
            crate::devices::motions::MotionStatus::Homing => MotionStatus::MotionHoming as i32,
            crate::devices::motions::MotionStatus::Error(_) => MotionStatus::MotionError as i32,
            crate::devices::motions::MotionStatus::LimitHit => MotionStatus::MotionLimitHit as i32,
        };
        
        Ok(Response::new(MotionStateResponse {
            powered: health.powered,
            initialized: false, // TODO: Add initialized field to MotionHealth
            is_homed: health.is_homed,
            status,
            position_x: position,
            position_y: None, // Single axis for now
            total_moves: health.total_moves,
        }))
    }
}
