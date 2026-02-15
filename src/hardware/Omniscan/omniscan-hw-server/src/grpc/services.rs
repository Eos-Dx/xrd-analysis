use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::{wrappers::ReceiverStream};
use tonic::{Request, Response, Status};
use tracing::{info, warn, error};
use chrono::Utc;
use uuid::Uuid;

use crate::certificates::ClientCertInfo;
use crate::grpc::hub::v1::*;
use crate::safety::{SafetyStateMachine, SafetyState, InterlockStatus as SafetyInterlockStatus};
use crate::devices::DetectorDevice;
use crate::devices::motions::MotionControlDevice;
use crate::devices::gpio::GpioDevice;
use crate::devices::pdu::DemoPdu;
use crate::audit::{AuditLogger, CommandLog, CommandResult, MeasurementRecord, MeasurementStatus};
use crate::calibration_qc::{CalibrationQualityController, CalibrationImageData, CalibrationQcResults};

/// Convert calibration QC results to protobuf message
fn qc_results_to_proto(results: &CalibrationQcResults, calibrant_material: &str) -> CalibrationQcReport {
    CalibrationQcReport {
        calibrant_material: calibrant_material.to_string(),
        overall_pass: results.overall_pass,
        timestamp: Some(prost_types::Timestamp {
            seconds: Utc::now().timestamp(),
            nanos: 0,
        }),
        total_intensity_check: Some(QcCheckResult {
            passed: results.total_intensity_check.passed,
            check_name: results.total_intensity_check.check_name.clone(),
            measured_value: results.total_intensity_check.measured_value,
            threshold: results.total_intensity_check.threshold,
            details: results.total_intensity_check.details.clone(),
        }),
        goodness_check: Some(QcCheckResult {
            passed: results.goodness_check.passed,
            check_name: results.goodness_check.check_name.clone(),
            measured_value: results.goodness_check.measured_value,
            threshold: results.goodness_check.threshold,
            details: results.goodness_check.details.clone(),
        }),
        snr_check: Some(QcCheckResult {
            passed: results.snr_check.passed,
            check_name: results.snr_check.check_name.clone(),
            measured_value: results.snr_check.measured_value,
            threshold: results.snr_check.threshold,
            details: results.snr_check.details.clone(),
        }),
        ring_quality_check: Some(QcCheckResult {
            passed: results.ring_quality_check.passed,
            check_name: results.ring_quality_check.check_name.clone(),
            measured_value: results.ring_quality_check.measured_value,
            threshold: results.ring_quality_check.threshold,
            details: results.ring_quality_check.details.clone(),
        }),
        poni_result: Some(PoniCalculationResult {
            success: results.poni_calculation.success,
            distance_mm: results.poni_calculation.distance,
            beam_center_x: results.poni_calculation.poni1,
            beam_center_y: results.poni_calculation.poni2,
            rot1: results.poni_calculation.rot1,
            rot2: results.poni_calculation.rot2,
            rot3: results.poni_calculation.rot3,
            pixel_size_1_m: results.poni_calculation.pixel_size_1,
            pixel_size_2_m: results.poni_calculation.pixel_size_2,
            wavelength_angstrom: results.poni_calculation.wavelength,
            poni_file_content: results.poni_calculation.poni_file_content.clone(),
        }),
        formatted_report: "".to_string(), // Will be set by caller
    }
}
/// Broadcast channel for state change notifications
pub type StateNotificationSender = tokio::sync::broadcast::Sender<StateChangeNotification>;

/// Convert safety state to protobuf enum
#[allow(dead_code)]
pub fn safety_state_to_proto(state: SafetyState) -> ServerState {
    match state {
        SafetyState::Locked => ServerState::Locked,              // System locked - requires calibration
        SafetyState::Initialized => ServerState::Idle,           // Initialized after key switch ON
        SafetyState::Idle => ServerState::Idle,
        SafetyState::WarmingUp => ServerState::Idle,             // Warmup in progress (stays IDLE)
        SafetyState::Calibrated => ServerState::Idle,            // Calibrated and ready
        SafetyState::PendingArmed => ServerState::PendingArmed,
        SafetyState::Running => ServerState::Running,
        SafetyState::Stopping => ServerState::Stopping,
        SafetyState::Safe => ServerState::Safe,
        SafetyState::Calibration => ServerState::Calibration,    // Calibration in progress
        SafetyState::Maintenance => ServerState::Maintenance,    // Maintenance mode active
    }
}

/// Convert safety interlock status to protobuf
#[allow(dead_code)]
fn interlocks_to_proto(interlocks: SafetyInterlockStatus) -> InterlockStatus {
    InterlockStatus {
        emergency_stop: interlocks.emergency_stop,
        door_closed: interlocks.door_closed,
        radiation_safe: interlocks.beam_watchdog, // Beam watchdog represents radiation safety
        cooling_ok: interlocks.over_temperature.unwrap_or(true),
        power_ok: true, // TODO: Add actual power monitoring
        overall_safe: interlocks.overall_safe,
        violation_reason: interlocks.violation_reason.unwrap_or_default(),
        enable_button: false, // Placeholder - will be updated by caller with GPIO state
        key_switch: false,    // Placeholder - will be updated by caller with GPIO state
    }
}

/// Extract client certificate information from request if available
#[allow(dead_code)]
fn extract_client_cert_info<T>(_request: &Request<T>) -> Option<ClientCertInfo> {
    // In mTLS mode, tonic provides peer certificates through request extensions
    // Note: This requires the connection info to be available
    // For now, we'll return None and log a warning if certificates are expected
    // Full implementation would require accessing the TLS peer certificates from the request
    
    // TODO: Extract from request.peer_certs() when available in tonic
    None
}

/// Validate client certificate matches the expected device UUID
#[allow(dead_code)]
fn validate_client_cert(cert_info: &ClientCertInfo, expected_device_uuid: &str) -> Result<(), Status> {
    if cert_info.device_uuid != expected_device_uuid {
        error!(
            "Certificate device UUID mismatch: client cert for '{}', but server is '{}'",
            cert_info.device_uuid, expected_device_uuid
        );
        return Err(Status::permission_denied(
            "Client certificate device UUID does not match this server"
        ));
    }
    
    info!("Client authenticated: Engineer {} for device {}", 
        cert_info.engineer_id, cert_info.device_uuid);
    
    Ok(())
}

/// Shared state for gRPC services
pub struct ServiceState {
    pub safety_state_machine: Arc<RwLock<SafetyStateMachine>>,
    pub detector: Arc<dyn DetectorDevice + Send + Sync>,
    pub motion: Arc<dyn MotionControlDevice + Send + Sync>,
    pub gpio: Arc<dyn GpioDevice + Send + Sync>,
    pub pdu: Arc<DemoPdu>,
    pub audit_logger: Arc<AuditLogger>,
    pub device_uuid: Option<String>,
    pub state_notifications: StateNotificationSender,
}

impl ServiceState {
    /// Emit a state change notification to all subscribers
    pub fn notify_state_change(&self, component: &str, change_type: &str) {
        let notification = StateChangeNotification {
            component: component.to_string(),
            change_type: change_type.to_string(),
            timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: 0,
            }),
        };
        
        // Best-effort broadcast - ignore if no subscribers
        let _ = self.state_notifications.send(notification);
    }
}

/// Acquisition service implementation (StartExposure, Stop, Abort, etc.)
pub struct AcquisitionService {
    state: Arc<ServiceState>,
}

impl AcquisitionService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }

    async fn log_command(&self, ctx: &CommandContext, service: &str, command: &str, result: CommandResult, execution_time_ms: u64) {
        // Extract orchestrator_id from device_uuid if mTLS is enabled (placeholder for now)
        let orchestrator_id = self.state.device_uuid.clone().unwrap_or_else(|| "unknown".to_string());
        
        let command_log = CommandLog {
            id: ctx.command_id.clone(),
            session_id: "grpc_session".to_string(), // TODO: Use actual session management
            timestamp: if let Some(ts) = &ctx.timestamp {
                chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or(Utc::now())
            } else {
                Utc::now()
            },
            command_type: format!("{}.{}", service, command), // Service.Command format per COMMANDS_SPEC.md
            command_data: serde_json::json!({
                "operator_id": &ctx.user,
                "orchestrator_id": orchestrator_id,
                "reason": &ctx.reason,
                "command_id": &ctx.command_id,
                "service": service,
                "command": command
            }),
            user_context: Some(ctx.user.clone()),
            device_state_before: serde_json::json!({"grpc_command": true}),
            device_state_after: None, // TODO: Add actual device state capture
            result,
            execution_time_ms,
        };

        if let Err(e) = self.state.audit_logger.log_command(command_log).await {
            error!("Failed to log command {}.{}: {}", service, command, e);
        }
    }
}

#[tonic::async_trait]
impl acquisition_server::Acquisition for AcquisitionService {
    async fn start_exposure(&self, request: Request<StartExposureRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        
        // Validate client certificate if mTLS is enabled
        if let Some(device_uuid) = &self.state.device_uuid {
            if let Some(cert_info) = extract_client_cert_info(&request) {
                validate_client_cert(&cert_info, device_uuid)?;
            } else {
                warn!("mTLS enabled but no client certificate found in request");
            }
        }
        
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("StartExposure requested by user: {} ({})", ctx.user, ctx.command_id);

        // Check safety state machine
        let mut safety_sm = self.state.safety_state_machine.write().await;
        let can_start = match safety_sm.start_measurement(ctx.command_id.clone(), ctx.user.clone()).await {
            Ok(result) => result,
            Err(e) => {
                error!("Safety check failed for StartExposure: {}", e);
                self.log_command(&ctx, "Acquisition", "StartExposure", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                return Err(Status::failed_precondition(format!("Safety check failed: {}", e)));
            }
        };

        if !can_start {
            let current_state = safety_sm.get_current_state().await;
            let msg = format!("Cannot start exposure from state: {}", current_state.as_str());
            warn!("{}", msg);
            self.log_command(&ctx, "Acquisition", "StartExposure", CommandResult::Failed { error: msg.clone() }, start_time.elapsed().as_millis() as u64).await;
            return Err(Status::failed_precondition(msg));
        }

        drop(safety_sm);

        // Start detector exposure
        match self.state.detector.start_exposure(req.exposure_time_ms).await {
            Ok(_) => {
                info!("Exposure started successfully: {} ms", req.exposure_time_ms);

                // Log measurement record per COMMANDS_SPEC.md line 104
                let measurement_record = crate::audit::MeasurementRecord {
                    id: Uuid::new_v4().to_string(),
                    session_id: self.state.audit_logger.session_id().to_string(),
                    measurement_id: ctx.command_id.clone(),
                    patient_id: None, // Server never receives patient_id
                    study_id: None,
                    timestamp_start: Utc::now(),
                    timestamp_end: None,
                    exposure_time_ms: req.exposure_time_ms,
                    detector_config: serde_json::json!({"type": "demo"}),
                    motion_position: serde_json::json!({"x": 0.0, "y": 0.0}),
                    data_files: vec![],
                    data_size_bytes: 0,
                    checksum: None,
                    operator: Some(ctx.user.clone()),
                    notes: Some(ctx.reason.clone()),
                    status: crate::audit::MeasurementStatus::Started,
                };
                
                if let Err(e) = self.state.audit_logger.log_measurement(measurement_record).await {
                    error!("Failed to log measurement record: {}", e);
                }

                // Broadcast non-blocking run start + per-second progress updates for orchestrator subscribers
                let run_id = ctx.command_id.clone();
                let notifier = self.state.state_notifications.clone();
                let total_secs: u32 = (req.exposure_time_ms / 1000).max(1);
                let audit_logger = self.state.audit_logger.clone();
                tokio::spawn(async move {
                    // START
                    let _ = notifier.send(StateChangeNotification {
                        component: "RUN".to_string(),
                        change_type: format!("START:measurement:{}:{}", run_id, total_secs),
                        timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                    });
                    // TICKS
                    for elapsed in 1..=total_secs {
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        let _ = notifier.send(StateChangeNotification {
                            component: "RUN".to_string(),
                            change_type: format!("TICK:measurement:{}:{}/{}", run_id, elapsed, total_secs),
                            timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                        });
                    }
                    // DONE (best-effort; detector may have finished earlier)
                    let _ = notifier.send(StateChangeNotification {
                        component: "RUN".to_string(),
                        change_type: format!("DONE:measurement:{}:completed", run_id),
                        timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                    });
                    
                    // Update measurement status to Completed
                    // TODO: Query actual final status from detector
                });

                self.log_command(&ctx, "Acquisition", "StartExposure", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Failed to start exposure: {}", e);
                // Transition back to safe state on hardware failure
                let mut safety_sm = self.state.safety_state_machine.write().await;
                let _ = safety_sm.abort(ctx.command_id.clone()).await;
                
                self.log_command(&ctx, "Acquisition", "StartExposure", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Hardware error: {}", e)))
            }
        }
    }

    async fn stop(&self, request: Request<StopRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("Stop requested by user: {} ({})", ctx.user, ctx.command_id);

        // Update safety state machine
        let mut safety_sm = self.state.safety_state_machine.write().await;
        if let Err(e) = safety_sm.stop(ctx.command_id.clone()).await {
            error!("Safety state machine stop failed: {}", e);
            self.log_command(&ctx, "Acquisition", "Stop", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
            return Err(Status::internal(format!("Safety error: {}", e)));
        }
        drop(safety_sm);

        // Stop detector
        match self.state.detector.stop_exposure().await {
            Ok(_) => {
                // Complete the stop in safety state machine
                let mut safety_sm = self.state.safety_state_machine.write().await;
                let _ = safety_sm.complete_stop().await;
                drop(safety_sm);

                // After successful stop, persist measurement to audit DB (best-effort)
                if let Some(result) = self.state.detector.get_last_result().await {
                    let measurement_id = ctx.command_id.clone();
                    let end_ts = result.timestamp;
                    let start_ts = end_ts - chrono::Duration::milliseconds(result.exposure_time_ms as i64);
                    let data_files = match result.data_path.clone() {
                        Some(p) => vec![p],
                        None => Vec::new(),
                    };
                    let record = MeasurementRecord {
                        id: uuid::Uuid::new_v4().to_string(),
                        session_id: self.state.audit_logger.session_id().to_string(),
                        measurement_id,
                        patient_id: None,
                        study_id: None,
                        timestamp_start: start_ts,
                        timestamp_end: Some(end_ts),
                        exposure_time_ms: result.exposure_time_ms,
                        detector_config: serde_json::json!({}),
                        motion_position: serde_json::json!({}),
                        data_files,
                        data_size_bytes: result.data_size as u64,
                        checksum: None,
                        operator: Some(ctx.user.clone()),
                        notes: None,
                        status: MeasurementStatus::Completed,
                    };
                    if let Err(e) = self.state.audit_logger.log_measurement(record).await {
                        error!("Failed to persist measurement record: {}", e);
                    }
                }

                info!("Stop completed successfully");
                self.log_command(&ctx, "Acquisition", "Stop", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Failed to stop exposure: {}", e);
                self.log_command(&ctx, "Acquisition", "Stop", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Hardware error: {}", e)))
            }
        }
    }

    async fn abort(&self, request: Request<AbortRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        warn!("ABORT requested by user: {} ({})", ctx.user, ctx.command_id);

        // Emergency abort - update safety state machine first
        let mut safety_sm = self.state.safety_state_machine.write().await;
        if let Err(e) = safety_sm.abort(ctx.command_id.clone()).await {
            error!("Safety state machine abort failed: {}", e);
        }
        drop(safety_sm);

        // Abort detector (use stop_exposure for abort)
        match self.state.detector.stop_exposure().await {
            Ok(_) => {
                warn!("Emergency abort completed");
                self.log_command(&ctx, "Acquisition", "Abort", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Failed to abort exposure: {}", e);
                self.log_command(&ctx, "Acquisition", "Abort", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Hardware error: {}", e)))
            }
        }
    }

    async fn get_state(&self, _request: Request<Empty>) -> Result<Response<GetStateResponse>, Status> {
        let safety_sm = self.state.safety_state_machine.read().await;
        let current_state = safety_sm.get_current_state().await;
        let interlocks = safety_sm.get_interlock_status().await;
        
        let response = GetStateResponse {
            state: safety_state_to_proto(current_state.clone()) as i32,
            detail: format!("System state: {}", current_state.as_str()),
            interlocks: Some(interlocks_to_proto(interlocks.clone())),
            timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn get_last_exposure_result(&self, _request: Request<Empty>) -> Result<Response<GetExposureResultResponse>, Status> {
        // TODO: Implement exposure result retrieval from detector
        let response = GetExposureResultResponse {
            has_result: false,
            result: None,
        };
        Ok(Response::new(response))
    }

    async fn calibrate_detector(&self, request: Request<CalibrateDetectorRequest>) -> Result<Response<CalibrateDetectorResponse>, Status> {
        let start_time = std::time::Instant::now();
        
        // Validate client certificate if mTLS is enabled
        if let Some(device_uuid) = &self.state.device_uuid {
            if let Some(cert_info) = extract_client_cert_info(&request) {
                validate_client_cert(&cert_info, device_uuid)?;
            } else {
                warn!("mTLS enabled but no client certificate found in request");
            }
        }
        
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("Calibrate detector requested by user: {} ({})", ctx.user, ctx.command_id);

        // Track whether we opened the beam so we can close it afterwards
        let mut opened_beam_this_call = false;

        // Safety checks for calibration measurement
        // Calibration requires X-ray beam to be OPEN (radiation_safe = false)
        let interlocks = self.state.gpio.get_interlocks().await;
        let key_switch_on = self.state.gpio.get_key_switch_state().await
            .map_err(|e| Status::internal(format!("Failed to check key switch: {}", e)))?;

        // Check 1: Key switch must be ON
        if !key_switch_on {
            let error_msg = "Calibration blocked: Key switch must be ON to activate the system";
            error!("{}", error_msg);
            self.log_command(&ctx, "Acquisition", "CalibrateDetector", CommandResult::Failed { error: error_msg.to_string() }, start_time.elapsed().as_millis() as u64).await;
            return Err(Status::failed_precondition(error_msg));
        }

        // Check 2: Require beam OPEN; if closed but enable button is active, actively open beam-stop
        if interlocks.radiation_safe {
            let enable_active = self.state.gpio.get_activation_button_active().await
                .map_err(|e| Status::internal(format!("Failed to read activation button: {}", e)))?;
            if enable_active {
                info!("Calibration: Enable button active - opening beam-stop (shutter) for calibration");
                if let Err(e) = self.state.gpio.open_beam_stop().await {
                    let error_msg = format!("Calibration blocked: Failed to open beam-stop: {}", e);
                    error!("{}", error_msg);
                    self.log_command(&ctx, "Acquisition", "CalibrateDetector", CommandResult::Failed { error: error_msg.clone() }, start_time.elapsed().as_millis() as u64).await;
                    return Err(Status::failed_precondition(error_msg));
                }
                opened_beam_this_call = true;
            } else {
                let error_msg = "Calibration blocked: X-ray beam is closed (radiation safe = true). Click 'Activation' button to open the beam.";
                error!("{}", error_msg);
                self.log_command(&ctx, "Acquisition", "CalibrateDetector", CommandResult::Failed { error: error_msg.to_string() }, start_time.elapsed().as_millis() as u64).await;
                return Err(Status::failed_precondition(error_msg));
            }
        }

        // Check 3: Other interlocks must be safe
        if !interlocks.emergency_stop || !interlocks.door_closed || !interlocks.cooling_ok || !interlocks.power_ok {
            let error_msg = format!("Calibration blocked: Safety interlocks not satisfied - E-stop: {}, Door: {}, Cooling: {}, Power: {}",
                interlocks.emergency_stop, interlocks.door_closed, interlocks.cooling_ok, interlocks.power_ok);
            error!("{}", error_msg);
            self.log_command(&ctx, "Acquisition", "CalibrateDetector", CommandResult::Failed { error: error_msg.clone() }, start_time.elapsed().as_millis() as u64).await;
            // Close beam if we opened it
            if opened_beam_this_call {
                let _ = self.state.gpio.close_beam_stop().await;
            }
            return Err(Status::failed_precondition(error_msg));
        }

        info!("✅ Calibration safety checks passed: Key switch ON, X-ray beam OPEN, interlocks safe");

        // Broadcast RUN start, then spawn background task to perform calibration and stream TICK/DONE
        let run_id = ctx.command_id.clone();
        let notifier = self.state.state_notifications.clone();
        let safety_state = self.state.safety_state_machine.clone();
        let gpio = self.state.gpio.clone();
        let motion = self.state.motion.clone();
        let detector = self.state.detector.clone();
        let audit = self.state.audit_logger.clone();
        let device_uuid = self.state.device_uuid.clone();
        tokio::spawn(async move {
            // START
            let _ = notifier.send(StateChangeNotification {
                component: "RUN".to_string(),
                change_type: format!("START:calibration:{}", run_id),
                timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
            });

            // Launch a ticker in parallel to emit TICK every second until completion
            let (tick_tx, mut tick_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
            let notifier_clone = notifier.clone();
            let run_id_clone = run_id.clone();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                            let _ = notifier_clone.send(StateChangeNotification {
                                component: "RUN".to_string(),
                                change_type: format!("TICK:calibration:{}", run_id_clone),
                                timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                            });
                        }
                        _ = tick_rx.recv() => {
                            break;
                        }
                    }
                }
            });

            // Do homing + calibration + QC in background
            let mut opened_beam = opened_beam_this_call;
            let mut qc_result_overall_pass: Option<bool> = None;
            // Home motion
            if let Err(e) = motion.home().await {
                let _ = notifier.send(StateChangeNotification {
                    component: "RUN".to_string(),
                    change_type: format!("DONE:calibration:{}:failed:home:{}", run_id, e),
                    timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                });
                let _ = tick_tx.send(());
                // Attempt to close beam if we opened it
                if opened_beam { let _ = gpio.close_beam_stop().await; }
                return;
            }
            // Perform detector calibration (blocking within this task)
            if let Err(e) = detector.calibrate().await {
                let _ = notifier.send(StateChangeNotification {
                    component: "RUN".to_string(),
                    change_type: format!("DONE:calibration:{}:failed:detector:{}", run_id, e),
                    timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                });
                let _ = tick_tx.send(());
                if opened_beam { let _ = gpio.close_beam_stop().await; }
                return;
            }
            // QC
            let qc_controller = CalibrationQualityController::new("LaB6".to_string());
            let image_data = CalibrationImageData {
                raw_data: vec![0u8; 1024],
                width: 1024,
                height: 1024,
                exposure_time_ms: 5000,
                timestamp: Utc::now(),
            };
            match qc_controller.run_quality_control(&image_data).await {
                Ok(qc_results) => {
                    qc_result_overall_pass = Some(qc_results.overall_pass);
                    let qc_json = serde_json::to_string(&qc_results).unwrap_or_else(|_| "{}".to_string());
                    // Record into safety state machine (only if pass)
                    if qc_results.overall_pass {
                        let mut sm = safety_state.write().await;
                        let _ = sm.record_calibration_with_qc(qc_json.clone()).await;
                    }
                    // Persist audit record best-effort
                    let _ = audit.log_calibration_record(
                        uuid::Uuid::new_v4().to_string(),
                        Utc::now(),
                        None,
                        "LaB6".to_string(),
                        Some(5000),
                        device_uuid.clone(),
                        if qc_results.overall_pass { "VALID".to_string() } else { "FAILED".to_string() },
                        qc_json,
                        qc_results.poni_calculation.poni_file_content.clone(),
                    ).await;
                }
                Err(e) => {
                    let _ = notifier.send(StateChangeNotification {
                        component: "RUN".to_string(),
                        change_type: format!("DONE:calibration:{}:failed:qc:{}", run_id, e),
                        timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
                    });
                    let _ = tick_tx.send(());
                    if opened_beam { let _ = gpio.close_beam_stop().await; }
                    return;
                }
            }

            // Close beam-stop if we opened it
            if opened_beam { let _ = gpio.close_beam_stop().await; }

            // DONE
            let status_str = match qc_result_overall_pass { Some(true) => "passed", Some(false) => "failed", None => "unknown" };
            let _ = notifier.send(StateChangeNotification {
                component: "RUN".to_string(),
                change_type: format!("DONE:calibration:{}:{}", run_id, status_str),
                timestamp: Some(prost_types::Timestamp { seconds: Utc::now().timestamp(), nanos: 0 }),
            });
            let _ = tick_tx.send(());
        });

        // Return immediately; orchestrator will receive progress via SubscribeToStateUpdates and can fetch report via GetLastCalibration
        self.log_command(&ctx, "Acquisition", "CalibrateDetector", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
        Ok(Response::new(CalibrateDetectorResponse { success: true, qc_report: None, error_message: "".to_string() }))
    }

    type SubscribeRunEventsStream = ReceiverStream<Result<SystemEvent, Status>>;

    async fn subscribe_run_events(&self, _request: Request<Empty>) -> Result<Response<Self::SubscribeRunEventsStream>, Status> {
        // TODO: Implement real-time event streaming
        let (_tx, rx) = tokio::sync::mpsc::channel(128);
        
        // For now, return empty stream
        info!("Event subscription requested - streaming not yet implemented");
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_last_calibration(&self, _request: Request<Empty>) -> Result<Response<GetLastCalibrationResponse>, Status> {
        let safety_sm = self.state.safety_state_machine.read().await;
        let last_calibration = safety_sm.get_last_calibration();
        let qc_json = safety_sm.get_last_calibration_qc_json();
        
        if last_calibration.is_none() {
            return Ok(Response::new(GetLastCalibrationResponse {
                has_calibration: false,
                qc_report: None,
            }));
        }
        
        // Deserialize stored QC results and convert to protobuf
        let qc_report = if let Some(json_str) = qc_json {
            match serde_json::from_str::<crate::calibration_qc::CalibrationQcResults>(&json_str) {
                Ok(qc_results) => {
                    let qc_controller = crate::calibration_qc::CalibrationQualityController::new("LaB6".to_string());
                    let formatted_report = qc_controller.generate_report(&qc_results);
                    let mut proto_report = qc_results_to_proto(&qc_results, "LaB6");
                    proto_report.formatted_report = formatted_report;
                    Some(proto_report)
                },
                Err(e) => {
                    warn!("Failed to deserialize stored QC report: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        let response = GetLastCalibrationResponse {
            has_calibration: true,
            qc_report,
        };
        
        Ok(Response::new(response))
    }

    async fn start_background_measurement(&self, request: Request<StartBackgroundMeasurementRequest>) -> Result<Response<BackgroundMeasurementResponse>, Status> {
        let start_time = std::time::Instant::now();
        
        // Validate client certificate if mTLS is enabled
        if let Some(device_uuid) = &self.state.device_uuid {
            if let Some(cert_info) = extract_client_cert_info(&request) {
                validate_client_cert(&cert_info, device_uuid)?;
            } else {
                warn!("mTLS enabled but no client certificate found in request");
            }
        }
        
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("Background measurement requested by user: {} ({})", ctx.user, ctx.command_id);

        // Safety checks for background measurement (dark frame)
        // Background measurement requires X-ray beam to be CLOSED (radiation_safe = true)
        let interlocks = self.state.gpio.get_interlocks().await;
        let key_switch_on = self.state.gpio.get_key_switch_state().await
            .map_err(|e| Status::internal(format!("Failed to check key switch: {}", e)))?;

        // Check 1: Key switch must be ON
        if !key_switch_on {
            let error_msg = "Background measurement blocked: Key switch must be ON to activate the system";
            error!("{}", error_msg);
            self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Failed { error: error_msg.to_string() }, start_time.elapsed().as_millis() as u64).await;
            return Ok(Response::new(BackgroundMeasurementResponse { success: false, error_message: error_msg.to_string(), result: None }));
        }

        // Check 2: X-ray beam must be CLOSED (radiation_safe = true)
        if !interlocks.radiation_safe {
            let error_msg = "Background measurement blocked: X-ray beam is open (radiation safe = false). Close the beam before taking dark frames.";
            error!("{}", error_msg);
            self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Failed { error: error_msg.to_string() }, start_time.elapsed().as_millis() as u64).await;
            return Ok(Response::new(BackgroundMeasurementResponse { success: false, error_message: error_msg.to_string(), result: None }));
        }

        // Check 3: Other interlocks must be safe
        if !interlocks.emergency_stop || !interlocks.door_closed || !interlocks.cooling_ok || !interlocks.power_ok {
            let error_msg = format!("Background measurement blocked: Safety interlocks not satisfied - E-stop: {}, Door: {}, Cooling: {}, Power: {}",
                interlocks.emergency_stop, interlocks.door_closed, interlocks.cooling_ok, interlocks.power_ok);
            error!("{}", error_msg);
            self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Failed { error: error_msg.clone() }, start_time.elapsed().as_millis() as u64).await;
            return Ok(Response::new(BackgroundMeasurementResponse { success: false, error_message: error_msg, result: None }));
        }

        info!("✅ Background measurement safety checks passed: Key switch ON, X-ray beam CLOSED, interlocks safe");

        // Check safety state machine
        let mut safety_sm = self.state.safety_state_machine.write().await;
        let can_start = match safety_sm.start_measurement(ctx.command_id.clone(), ctx.user.clone()).await {
            Ok(result) => result,
            Err(e) => {
                error!("Safety check failed for background measurement: {}", e);
                self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                return Ok(Response::new(BackgroundMeasurementResponse { success: false, error_message: format!("Safety check failed: {}", e), result: None }));
            }
        };

        if !can_start {
            let current_state = safety_sm.get_current_state().await;
            let msg = format!("Cannot start background measurement from state: {}", current_state.as_str());
            warn!("{}", msg);
            self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Failed { error: msg.clone() }, start_time.elapsed().as_millis() as u64).await;
            return Ok(Response::new(BackgroundMeasurementResponse { success: false, error_message: msg, result: None }));
        }

        drop(safety_sm);

        // Start detector exposure for background (dark frame)
        match self.state.detector.start_exposure(req.exposure_time_ms).await {
            Ok(_) => {
                info!("Background measurement started successfully: {} ms (dark frame)", req.exposure_time_ms);
                
                // Wait for exposure to complete or implement async handling
                // For now, we'll return success immediately - in production this would be async
                let result = ExposureResult {
                    exposure_time_ms: req.exposure_time_ms,
                    timestamp: Some(prost_types::Timestamp {
                        seconds: Utc::now().timestamp(),
                        nanos: 0,
                    }),
                    data_size: 0, // Will be filled by detector
                    data_path: None, // Will be filled by detector
                    detector_temp: 20.0, // Will be filled by detector
                };
                
                self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(BackgroundMeasurementResponse { 
                    success: true, 
                    error_message: "".to_string(), 
                    result: Some(result) 
                }))
            }
            Err(e) => {
                error!("Failed to start background measurement: {}", e);
                // Transition back to safe state on hardware failure
                let mut safety_sm = self.state.safety_state_machine.write().await;
                let _ = safety_sm.abort(ctx.command_id.clone()).await;
                
                self.log_command(&ctx, "Acquisition", "StartBackgroundMeasurement", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(BackgroundMeasurementResponse { 
                    success: false, 
                    error_message: format!("Hardware error: {}", e), 
                    result: None 
                }))
            }
        }
    }
}

/// Motion service implementation
pub struct MotionService {
    state: Arc<ServiceState>,
}

impl MotionService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }

    async fn log_command(&self, ctx: &CommandContext, service: &str, command: &str, result: CommandResult, execution_time_ms: u64) {
        let orchestrator_id = self.state.device_uuid.clone().unwrap_or_else(|| "unknown".to_string());
        
        let command_log = CommandLog {
            id: ctx.command_id.clone(),
            session_id: "grpc_session".to_string(),
            timestamp: if let Some(ts) = &ctx.timestamp {
                chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or(Utc::now())
            } else {
                Utc::now()
            },
            command_type: format!("{}.{}", service, command),
            command_data: serde_json::json!({
                "operator_id": &ctx.user,
                "orchestrator_id": orchestrator_id,
                "reason": &ctx.reason,
                "command_id": &ctx.command_id,
                "service": service,
                "command": command
            }),
            user_context: Some(ctx.user.clone()),
            device_state_before: serde_json::json!({"grpc_command": true}),
            device_state_after: None,
            result,
            execution_time_ms,
        };

        if let Err(e) = self.state.audit_logger.log_command(command_log).await {
            error!("Failed to log command {}.{}: {}", service, command, e);
        }
    }
}

#[tonic::async_trait]
impl motion_server::Motion for MotionService {
    async fn move_to(&self, request: Request<MoveToRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("MoveTo requested: {} mm by user: {} ({})", req.position_mm, ctx.user, ctx.command_id);

        match self.state.motion.move_to(req.position_mm).await {
            Ok(_) => {
                info!("MoveTo completed successfully");
                self.log_command(&ctx, "Motion", "MoveTo", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("MoveTo failed: {}", e);
                self.log_command(&ctx, "Motion", "MoveTo", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion error: {}", e)))
            }
        }
    }

    async fn move_relative(&self, request: Request<MoveRelativeRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("MoveRelative requested: {} mm by user: {} ({})", req.distance_mm, ctx.user, ctx.command_id);

        match self.state.motion.move_relative(req.distance_mm).await {
            Ok(_) => {
                info!("MoveRelative completed successfully");
                self.log_command(&ctx, "Motion", "MoveRelative", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("MoveRelative failed: {}", e);
                self.log_command(&ctx, "Motion", "MoveRelative", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion error: {}", e)))
            }
        }
    }

    async fn home(&self, request: Request<HomeRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("Home requested by user: {} ({})", ctx.user, ctx.command_id);

        match self.state.motion.home().await {
            Ok(_) => {
                info!("Homing completed successfully");
                self.log_command(&ctx, "Motion", "Home", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Homing failed: {}", e);
                self.log_command(&ctx, "Motion", "Home", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion error: {}", e)))
            }
        }
    }

    async fn stop(&self, request: Request<StopRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("Motion stop requested by user: {} ({})", ctx.user, ctx.command_id);

        match self.state.motion.stop_motion().await {
            Ok(_) => {
                info!("Motion stopped successfully");
                self.log_command(&ctx, "Motion", "Stop", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Motion stop failed: {}", e);
                self.log_command(&ctx, "Motion", "Stop", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion error: {}", e)))
            }
        }
    }

    async fn set_velocity(&self, request: Request<SetVelocityRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("SetVelocity requested: {} mm/s by user: {} ({})", req.velocity_mm_s, ctx.user, ctx.command_id);

        match self.state.motion.set_velocity(req.velocity_mm_s).await {
            Ok(_) => {
                info!("Velocity set successfully");
                self.log_command(&ctx, "Motion", "SetVelocity", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("SetVelocity failed: {}", e);
                self.log_command(&ctx, "Motion", "SetVelocity", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion error: {}", e)))
            }
        }
    }

    async fn get_position(&self, _request: Request<Empty>) -> Result<Response<GetPositionResponse>, Status> {
        let position = self.state.motion.get_position().await;
        let is_homed = position.is_some();

        let response = GetPositionResponse {
            position_mm: position,
            is_homed,
        };

        Ok(Response::new(response))
    }
}

/// Health service implementation
pub struct HealthService {
    state: Arc<ServiceState>,
}

impl HealthService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl health_server::Health for HealthService {
    async fn liveness(&self, _request: Request<Empty>) -> Result<Response<Empty>, Status> {
        // Basic liveness check - server is responding
        Ok(Response::new(Empty {}))
    }

    async fn readiness(&self, _request: Request<Empty>) -> Result<Response<Empty>, Status> {
        // Check if system is ready for operations
        let safety_sm = self.state.safety_state_machine.read().await;
        let current_state = safety_sm.get_current_state().await;
        
        match current_state {
            SafetyState::Locked => Err(Status::unavailable("System locked - calibration required")),
            SafetyState::Safe => Err(Status::unavailable("System in safe state - check interlocks")),
            SafetyState::Maintenance => Err(Status::unavailable("System in maintenance mode")),
            _ => Ok(Response::new(Empty {}))
        }
    }

    async fn get_aggregate_health(&self, _request: Request<Empty>) -> Result<Response<AggregateHealth>, Status> {
        let safety_sm = self.state.safety_state_machine.read().await;
        let current_state = safety_sm.get_current_state().await;
        drop(safety_sm);

        // Get real-time GPIO interlock status (updated by watchdog every 10ms)
        let gpio_interlocks = self.state.gpio.get_interlocks().await;
        let key_switch_on = self.state.gpio.get_key_switch_state().await.unwrap_or(false);
        let activation_button_active = self.state.gpio.get_activation_button_active().await.unwrap_or(false);

        let system_ok = matches!(current_state, SafetyState::Idle | SafetyState::PendingArmed | SafetyState::Running);

        let interlocks_ok = gpio_interlocks.overall_safe;
        let interlocks_detail = if interlocks_ok {
            "All interlocks satisfied".to_string()
        } else {
            let mut violations = Vec::new();
            if !gpio_interlocks.emergency_stop { violations.push("E-stop pressed"); }
            if !gpio_interlocks.door_closed { violations.push("Door open"); }
            if !gpio_interlocks.radiation_safe { violations.push("Radiation unsafe"); }
            if !gpio_interlocks.cooling_ok { violations.push("Cooling issue"); }
            if !gpio_interlocks.power_ok { violations.push("Power issue"); }
            violations.join(", ")
        };
        
        let components = vec![
            HealthComponent {
                name: "Safety State Machine".to_string(),
                ok: system_ok,
                detail: format!("Current state: {}", current_state.as_str()),
                last_check: Some(prost_types::Timestamp {
                    seconds: Utc::now().timestamp(),
                    nanos: 0,
                }),
            },
            HealthComponent {
                name: "Interlocks".to_string(),
                ok: interlocks_ok,
                detail: interlocks_detail.clone(),
                last_check: Some(prost_types::Timestamp {
                    seconds: Utc::now().timestamp(),
                    nanos: 0,
                }),
            },
        ];

        let response = AggregateHealth {
            ok: system_ok && interlocks_ok,
            components,
            interlocks: Some(InterlockStatus {
                emergency_stop: gpio_interlocks.emergency_stop,
                door_closed: gpio_interlocks.door_closed,
                radiation_safe: gpio_interlocks.radiation_safe,
                cooling_ok: gpio_interlocks.cooling_ok,
                power_ok: gpio_interlocks.power_ok,
                overall_safe: gpio_interlocks.overall_safe,
                violation_reason: if interlocks_ok { String::new() } else { interlocks_detail },
                enable_button: activation_button_active,
                key_switch: key_switch_on,
            }),
        };

        Ok(Response::new(response))
    }
}

/// Safety service implementation
pub struct SafetyService {
    state: Arc<ServiceState>,
}

impl SafetyService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl safety_server::Safety for SafetyService {
    async fn get_interlock_status(&self, _request: Request<Empty>) -> Result<Response<InterlockStatus>, Status> {
        // Get real-time GPIO interlock status (updated by watchdog every 10ms)
        let gpio_interlocks = self.state.gpio.get_interlocks().await;
        let key_switch_on = self.state.gpio.get_key_switch_state().await.unwrap_or(false);
        let activation_button_active = self.state.gpio.get_activation_button_active().await.unwrap_or(false);
        
        let response = InterlockStatus {
            emergency_stop: gpio_interlocks.emergency_stop,
            door_closed: gpio_interlocks.door_closed,
            radiation_safe: gpio_interlocks.radiation_safe,
            cooling_ok: gpio_interlocks.cooling_ok,
            power_ok: gpio_interlocks.power_ok,
            overall_safe: gpio_interlocks.overall_safe,
            violation_reason: if gpio_interlocks.overall_safe {
                String::new()
            } else {
                let mut violations = Vec::new();
                if !gpio_interlocks.emergency_stop { violations.push("E-stop pressed"); }
                if !gpio_interlocks.door_closed { violations.push("Door open"); }
                if !gpio_interlocks.radiation_safe { violations.push("Radiation unsafe"); }
                if !gpio_interlocks.cooling_ok { violations.push("Cooling issue"); }
                if !gpio_interlocks.power_ok { violations.push("Power issue"); }
                violations.join(", ")
            },
            enable_button: activation_button_active,
            key_switch: key_switch_on,
        };
        
        Ok(Response::new(response))
    }

    async fn reset_interlocks(&self, request: Request<CommandContext>) -> Result<Response<Empty>, Status> {
        let ctx = request.into_inner();
        info!("Reset interlocks requested by user: {} ({})", ctx.user, ctx.command_id);
        
        // TODO: Implement interlock reset logic
        warn!("Interlock reset not yet implemented");
        
        Ok(Response::new(Empty {}))
    }

    async fn check_safety_to_operate(&self, _request: Request<Empty>) -> Result<Response<InterlockStatus>, Status> {
        let safety_sm = self.state.safety_state_machine.read().await;
        let current_state = safety_sm.get_current_state().await;
        drop(safety_sm);
        
        // Get real-time GPIO interlock status (updated by watchdog every 10ms)
        let gpio_interlocks = self.state.gpio.get_interlocks().await;
        let key_switch_on = self.state.gpio.get_key_switch_state().await.unwrap_or(false);
        let activation_button_active = self.state.gpio.get_activation_button_active().await.unwrap_or(false);
        
        let mut response = InterlockStatus {
            emergency_stop: gpio_interlocks.emergency_stop,
            door_closed: gpio_interlocks.door_closed,
            radiation_safe: gpio_interlocks.radiation_safe,
            cooling_ok: gpio_interlocks.cooling_ok,
            power_ok: gpio_interlocks.power_ok,
            overall_safe: gpio_interlocks.overall_safe,
            violation_reason: if gpio_interlocks.overall_safe {
                String::new()
            } else {
                let mut violations = Vec::new();
                if !gpio_interlocks.emergency_stop { violations.push("E-stop pressed"); }
                if !gpio_interlocks.door_closed { violations.push("Door open"); }
                if !gpio_interlocks.radiation_safe { violations.push("Radiation unsafe"); }
                if !gpio_interlocks.cooling_ok { violations.push("Cooling issue"); }
                if !gpio_interlocks.power_ok { violations.push("Power issue"); }
                violations.join(", ")
            },
            enable_button: activation_button_active,
            key_switch: key_switch_on,
        };
        
        // Override safety based on current state
        if matches!(current_state, SafetyState::Locked | SafetyState::Safe) {
            response.overall_safe = false;
            if response.violation_reason.is_empty() {
                response.violation_reason = format!("System in {} state", current_state.as_str());
            } else {
                response.violation_reason = format!("{}, System in {} state", response.violation_reason, current_state.as_str());
            }
        }
        
        Ok(Response::new(response))
    }
}

/// Device Control service implementation
pub struct DeviceControlService {
    state: Arc<ServiceState>,
}

impl DeviceControlService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }

    async fn log_command(&self, ctx: &CommandContext, service: &str, command: &str, result: CommandResult, execution_time_ms: u64) {
        let orchestrator_id = self.state.device_uuid.clone().unwrap_or_else(|| "unknown".to_string());
        
        let command_log = CommandLog {
            id: ctx.command_id.clone(),
            session_id: "grpc_session".to_string(),
            timestamp: if let Some(ts) = &ctx.timestamp {
                chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or(Utc::now())
            } else {
                Utc::now()
            },
            command_type: format!("{}.{}", service, command),
            command_data: serde_json::json!({
                "operator_id": &ctx.user,
                "orchestrator_id": orchestrator_id,
                "reason": &ctx.reason,
                "command_id": &ctx.command_id,
                "service": service,
                "command": command
            }),
            user_context: Some(ctx.user.clone()),
            device_state_before: serde_json::json!({"grpc_command": true}),
            device_state_after: None,
            result,
            execution_time_ms,
        };

        if let Err(e) = self.state.audit_logger.log_command(command_log).await {
            error!("Failed to log command {}.{}: {}", service, command, e);
        }
    }
}

#[tonic::async_trait]
impl device_control_server::DeviceControl for DeviceControlService {
    async fn power_device(&self, request: Request<PowerDeviceRequest>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        
        // Validate client certificate if mTLS is enabled
        if let Some(device_uuid) = &self.state.device_uuid {
            if let Some(cert_info) = extract_client_cert_info(&request) {
                validate_client_cert(&cert_info, device_uuid)?;
            } else {
                warn!("mTLS enabled but no client certificate found in request");
            }
        }
        
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("PowerDevice requested: {} {} by user: {} ({})", 
            req.device_type, 
            if req.power_on { "ON" } else { "OFF" },
            ctx.user, 
            ctx.command_id
        );

        // Route to appropriate device
        let result = match req.device_type.as_str() {
            "detector" => {
                let res = if req.power_on {
                    self.state.detector.power_on().await
                        .map_err(|e| Status::internal(format!("Detector power on failed: {}", e)))
                } else {
                    self.state.detector.power_off().await
                        .map_err(|e| Status::internal(format!("Detector power off failed: {}", e)))
                };
                if res.is_ok() {
                    self.state.notify_state_change("DETECTOR", "POWER_CHANGED");
                }
                res
            }
            "motion" => {
                let res = if req.power_on {
                    self.state.motion.power_on().await
                        .map_err(|e| Status::internal(format!("Motion power on failed: {}", e)))
                } else {
                    self.state.motion.power_off().await
                        .map_err(|e| Status::internal(format!("Motion power off failed: {}", e)))
                };
                if res.is_ok() {
                    self.state.notify_state_change("MOTION", "POWER_CHANGED");
                }
                res
            }
            "gpio" => {
                let res = if req.power_on {
                    self.state.gpio.power_on().await
                        .map_err(|e| Status::internal(format!("GPIO power on failed: {}", e)))
                } else {
                    self.state.gpio.power_off().await
                        .map_err(|e| Status::internal(format!("GPIO power off failed: {}", e)))
                };
                if res.is_ok() {
                    self.state.notify_state_change("GPIO", "POWER_CHANGED");
                }
                res
            }
            _ => {
                let msg = format!("Unknown device type: {}", req.device_type);
                error!("{}", msg);
                self.log_command(&ctx, "DeviceControl", "PowerDevice", CommandResult::Failed { error: msg.clone() }, start_time.elapsed().as_millis() as u64).await;
                return Err(Status::invalid_argument(msg));
            }
        };

        match result {
            Ok(_) => {
                info!("PowerDevice completed successfully");
                self.log_command(&ctx, "DeviceControl", "PowerDevice", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("PowerDevice failed: {}", e);
                self.log_command(&ctx, "DeviceControl", "PowerDevice", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(e)
            }
        }
    }

    async fn get_detector_health(&self, _request: Request<Empty>) -> Result<Response<DetectorHealth>, Status> {
        let health = self.state.detector.get_health().await;
        
        // Convert detector status to protobuf enum
        let status = match health.status {
            crate::devices::DetectorStatus::Off => DetectorStatus::DetectorOff,
            crate::devices::DetectorStatus::Init => DetectorStatus::DetectorIdle,
            crate::devices::DetectorStatus::Idle => DetectorStatus::DetectorIdle,
            crate::devices::DetectorStatus::Exposing => DetectorStatus::DetectorExposing,
            crate::devices::DetectorStatus::Reading => DetectorStatus::DetectorReading,
            crate::devices::DetectorStatus::Error(_) => DetectorStatus::DetectorError,
        };

        let response = DetectorHealth {
            powered: health.powered,
            temperature: health.temperature,
            voltage: health.voltage,
            status: status as i32,
            last_exposure_time: health.last_exposure_time.unwrap_or(0),
            total_exposures: health.total_exposures,
            uptime_seconds: health.uptime.as_secs(),
        };

        Ok(Response::new(response))
    }

    async fn get_motion_health(&self, _request: Request<Empty>) -> Result<Response<MotionHealth>, Status> {
        let health = self.state.motion.get_health().await;
        
        // Convert motion status to protobuf enum
        let status = match health.status {
            crate::devices::motions::MotionStatus::Off => MotionStatus::MotionOff,
            crate::devices::motions::MotionStatus::Init => MotionStatus::MotionIdle,
            crate::devices::motions::MotionStatus::Idle => MotionStatus::MotionIdle,
            crate::devices::motions::MotionStatus::Moving => MotionStatus::MotionMoving,
            crate::devices::motions::MotionStatus::Homing => MotionStatus::MotionHoming,
            crate::devices::motions::MotionStatus::Error(_) => MotionStatus::MotionError,
            crate::devices::motions::MotionStatus::LimitHit => MotionStatus::MotionLimitHit,
        };

        let response = MotionHealth {
            powered: health.powered,
            status: status as i32,
            position: health.position,
            target_position: health.target_position,
            is_homed: health.is_homed,
            total_moves: health.total_moves,
            uptime_seconds: health.uptime.as_secs(),
        };

        Ok(Response::new(response))
    }

    async fn get_device_state(&self, request: Request<DeviceStateRequest>) -> Result<Response<DeviceStateResponse>, Status> {
        let req = request.into_inner();
        let device_type = req.device_type.to_lowercase();

        match device_type.as_str() {
            "pdu" => {
                let pdu_state = self.state.pdu.state.read().await;
                let mut outputs = std::collections::HashMap::new();
                outputs.insert("main_power".to_string(), pdu_state.powered);
                
                Ok(Response::new(DeviceStateResponse {
                    device_type: "pdu".to_string(),
                    powered: pdu_state.powered,
                    status: if pdu_state.powered { "Active" } else { "Off" }.to_string(),
                    uptime_seconds: pdu_state.last_changed.elapsed().as_secs(),
                    outputs,
                }))
            }
            "gpio" => {
                let gpio_powered = self.state.gpio.is_powered().await;
                Ok(Response::new(DeviceStateResponse {
                    device_type: "gpio".to_string(),
                    powered: gpio_powered,
                    status: if gpio_powered { "Active" } else { "Off" }.to_string(),
                    uptime_seconds: 0, // GPIO doesn't track uptime separately
                    outputs: std::collections::HashMap::new(),
                }))
            }
            "detector" => {
                let health = self.state.detector.get_health().await;
                let status_str = match health.status {
                    crate::devices::DetectorStatus::Off => "OFF",
                    crate::devices::DetectorStatus::Init => "INIT",
                    crate::devices::DetectorStatus::Idle => "IDLE",
                    crate::devices::DetectorStatus::Exposing => "EXPOSING",
                    crate::devices::DetectorStatus::Reading => "READING",
                    crate::devices::DetectorStatus::Error(_) => "ERROR",
                };
                Ok(Response::new(DeviceStateResponse {
                    device_type: "detector".to_string(),
                    powered: health.powered,
                    status: status_str.to_string(),
                    uptime_seconds: health.uptime.as_secs(),
                    outputs: std::collections::HashMap::new(),
                }))
            }
            "motion" => {
                let health = self.state.motion.get_health().await;
                let status_str = match health.status {
                    crate::devices::motions::MotionStatus::Off => "OFF",
                    crate::devices::motions::MotionStatus::Init => "INIT",
                    crate::devices::motions::MotionStatus::Idle => "IDLE",
                    crate::devices::motions::MotionStatus::Moving => "MOVING",
                    crate::devices::motions::MotionStatus::Homing => "HOMING",
                    crate::devices::motions::MotionStatus::Error(_) => "ERROR",
                    crate::devices::motions::MotionStatus::LimitHit => "LIMIT_HIT",
                };
                Ok(Response::new(DeviceStateResponse {
                    device_type: "motion".to_string(),
                    powered: health.powered,
                    status: status_str.to_string(),
                    uptime_seconds: health.uptime.as_secs(),
                    outputs: std::collections::HashMap::new(),
                }))
            }
            _ => {
                Err(Status::invalid_argument(format!("Unknown device type: {}. Supported: pdu, gpio, detector, motion", device_type)))
            }
        }
    }
}

/// DeviceInitialization service implementation
pub struct DeviceInitializationService {
    state: Arc<ServiceState>,
}

impl DeviceInitializationService {
    pub fn new(state: Arc<ServiceState>) -> Self {
        Self { state }
    }

    async fn log_command(&self, ctx: &CommandContext, service: &str, command: &str, result: CommandResult, execution_time_ms: u64) {
        let orchestrator_id = self.state.device_uuid.clone().unwrap_or_else(|| "unknown".to_string());
        
        let command_log = CommandLog {
            id: ctx.command_id.clone(),
            session_id: "grpc_session".to_string(),
            timestamp: if let Some(ts) = &ctx.timestamp {
                chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or(Utc::now())
            } else {
                Utc::now()
            },
            command_type: format!("{}.{}", service, command),
            command_data: serde_json::json!({
                "operator_id": &ctx.user,
                "orchestrator_id": orchestrator_id,
                "reason": &ctx.reason,
                "command_id": &ctx.command_id,
                "service": service,
                "command": command
            }),
            user_context: Some(ctx.user.clone()),
            device_state_before: serde_json::json!({"grpc_command": true}),
            device_state_after: None,
            result,
            execution_time_ms,
        };

        if let Err(e) = self.state.audit_logger.log_command(command_log).await {
            error!("Failed to log command {}.{}: {}", service, command, e);
        }
    }

    async fn check_initialization_preconditions(&self) -> Result<(), Status> {
        // Check key switch
        let key_switch = self.state.gpio.get_key_switch_state().await
            .map_err(|e| Status::internal(format!("Failed to read key switch: {}", e)))?;
        
        if !key_switch {
            return Err(Status::failed_precondition("Key switch must be ON for device initialization"));
        }

        // Check enable/activation button
        let enable_button = self.state.gpio.get_activation_button_active().await
            .map_err(|e| Status::internal(format!("Failed to read activation button: {}", e)))?;
        
        if !enable_button {
            return Err(Status::failed_precondition(
                "Enable/Activation button not active - required for device initialization. Click ACTIVATE button and retry within 20 seconds."
            ));
        }

        // Check safety interlocks
        let interlocks = self.state.gpio.get_interlocks().await;
        
        if !interlocks.radiation_safe {
            return Err(Status::failed_precondition(
                "Radiation is NOT SAFE (beam not blocked) - block beam before initializing devices"
            ));
        }
        
        if !interlocks.cooling_ok {
            return Err(Status::failed_precondition(
                "Cooling is NOT OK - check cooling system before initializing devices"
            ));
        }
        
        // Note: door_closed not required for initialization per spec
        
        Ok(())
    }
}

#[tonic::async_trait]
impl device_initialization_server::DeviceInitialization for DeviceInitializationService {
    async fn initialize_detector(&self, request: Request<InitializeDetectorRequest>) -> Result<Response<DetectorStateResponse>, Status> {
        let start_time = std::time::Instant::now();
        
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("InitializeDetector requested by user: {} ({})", ctx.user, ctx.command_id);

        // Check preconditions (key switch, enable button, interlocks)
        self.check_initialization_preconditions().await?;

        // Power on and initialize detector
        match self.state.detector.power_on().await {
            Ok(_) => {
                info!("Detector powered on and initialized successfully");
                self.state.notify_state_change("DETECTOR", "INITIALIZED");
                
                // Get current detector state
                let health = self.state.detector.get_health().await;
                let status = match health.status {
                    crate::devices::DetectorStatus::Off => DetectorStatus::DetectorOff,
                    crate::devices::DetectorStatus::Init => DetectorStatus::DetectorIdle,
                    crate::devices::DetectorStatus::Idle => DetectorStatus::DetectorIdle,
                    crate::devices::DetectorStatus::Exposing => DetectorStatus::DetectorExposing,
                    crate::devices::DetectorStatus::Reading => DetectorStatus::DetectorReading,
                    crate::devices::DetectorStatus::Error(_) => DetectorStatus::DetectorError,
                };

                let response = DetectorStateResponse {
                    powered: health.powered,
                    initialized: health.powered, // If powered, it's initialized
                    status: status as i32,
                    temperature: health.temperature,
                    total_exposures: health.total_exposures,
                };

                self.log_command(&ctx, "DeviceInitialization", "InitializeDetector", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Failed to initialize detector: {}", e);
                self.log_command(&ctx, "DeviceInitialization", "InitializeDetector", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Detector initialization failed: {}", e)))
            }
        }
    }

    async fn initialize_motion(&self, request: Request<InitializeMotionRequest>) -> Result<Response<MotionStateResponse>, Status> {
        let start_time = std::time::Instant::now();
        
        let req = request.into_inner();
        let ctx = req.ctx.ok_or_else(|| Status::invalid_argument("Missing command context"))?;

        info!("InitializeMotion requested by user: {} ({})", ctx.user, ctx.command_id);

        // Check preconditions (key switch, enable button, interlocks)
        self.check_initialization_preconditions().await?;

        // Power on and initialize motion controller
        match self.state.motion.power_on().await {
            Ok(_) => {
                info!("Motion controller powered on and initialized successfully");
                self.state.notify_state_change("MOTION", "INITIALIZED");
                
                // Get current motion state
                let health = self.state.motion.get_health().await;
                let status = match health.status {
                    crate::devices::motions::MotionStatus::Off => MotionStatus::MotionOff,
                    crate::devices::motions::MotionStatus::Init => MotionStatus::MotionIdle,
                    crate::devices::motions::MotionStatus::Idle => MotionStatus::MotionIdle,
                    crate::devices::motions::MotionStatus::Moving => MotionStatus::MotionMoving,
                    crate::devices::motions::MotionStatus::Homing => MotionStatus::MotionHoming,
                    crate::devices::motions::MotionStatus::Error(_) => MotionStatus::MotionError,
                    crate::devices::motions::MotionStatus::LimitHit => MotionStatus::MotionLimitHit,
                };

                let response = MotionStateResponse {
                    powered: health.powered,
                    initialized: health.powered, // If powered, it's initialized
                    is_homed: health.is_homed,
                    status: status as i32,
                    position_x: health.position,
                    position_y: None, // Single axis for now
                    total_moves: health.total_moves,
                };

                self.log_command(&ctx, "DeviceInitialization", "InitializeMotion", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Failed to initialize motion controller: {}", e);
                self.log_command(&ctx, "DeviceInitialization", "InitializeMotion", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion initialization failed: {}", e)))
            }
        }
    }

    async fn power_off_detector(&self, request: Request<CommandContext>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let ctx = request.into_inner();

        info!("PowerOffDetector requested by user: {} ({})", ctx.user, ctx.command_id);

        // Power off is a safe operation - no preconditions needed
        match self.state.detector.power_off().await {
            Ok(_) => {
                info!("Detector powered off successfully");
                self.state.notify_state_change("DETECTOR", "POWERED_OFF");
                self.log_command(&ctx, "DeviceInitialization", "PowerOffDetector", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Failed to power off detector: {}", e);
                self.log_command(&ctx, "DeviceInitialization", "PowerOffDetector", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Detector power off failed: {}", e)))
            }
        }
    }

    async fn power_off_motion(&self, request: Request<CommandContext>) -> Result<Response<Empty>, Status> {
        let start_time = std::time::Instant::now();
        let ctx = request.into_inner();

        info!("PowerOffMotion requested by user: {} ({})", ctx.user, ctx.command_id);

        // Power off is a safe operation - no preconditions needed
        match self.state.motion.power_off().await {
            Ok(_) => {
                info!("Motion controller powered off successfully");
                self.state.notify_state_change("MOTION", "POWERED_OFF");
                self.log_command(&ctx, "DeviceInitialization", "PowerOffMotion", CommandResult::Success, start_time.elapsed().as_millis() as u64).await;
                Ok(Response::new(Empty {}))
            }
            Err(e) => {
                error!("Failed to power off motion controller: {}", e);
                self.log_command(&ctx, "DeviceInitialization", "PowerOffMotion", CommandResult::Failed { error: e.to_string() }, start_time.elapsed().as_millis() as u64).await;
                Err(Status::internal(format!("Motion power off failed: {}", e)))
            }
        }
    }
}
