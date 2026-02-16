use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use super::{DetectorDevice, DetectorError, DetectorHealth, DetectorStatus, ExposureResult};

/// AXSCOM Protocol Client for Bruker Detectors
/// 
/// Implements the Bruker AXSCOM (AXS Communication) protocol for controlling
/// X-ray detectors via TCP sockets. Based on reverse-engineered protocol from
/// EosDx legacy system logs.
/// 
/// Protocol Details:
/// - Three TCP sockets: Command, File, Status
/// - Text-based protocol with [VERB /ARG=VAL]\n format
/// - Command handshaking: _YES, _NO, _MV responses
/// - Operational commands: _START, _DONE, _ERR, _SKIP
pub struct BrukerAxscomDetector {
    config: AxscomConfig,
    command_socket: Arc<Mutex<Option<TcpStream>>>,
    file_socket: Arc<Mutex<Option<TcpStream>>>,
    status_socket: Arc<Mutex<Option<TcpStream>>>,
    status: Arc<RwLock<DetectorStatus>>,
    health: Arc<RwLock<DetectorHealth>>,
    last_result: Arc<RwLock<Option<ExposureResult>>>,
    current_sample: Arc<RwLock<Option<SampleInfo>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxscomConfig {
    pub server_ip: String,
    pub command_port: u16,
    pub file_port: u16,
    pub status_port: u16,
    pub connect_timeout_ms: u64,
    pub command_timeout_ms: u64,
    pub status_poll_interval_ms: u64,
}

impl Default for AxscomConfig {
    fn default() -> Self {
        Self {
            server_ip: "192.168.23.3".to_string(),
            command_port: 49153,
            file_port: 49154,
            status_port: 49155,
            connect_timeout_ms: 10000,
            command_timeout_ms: 30000,
            status_poll_interval_ms: 10000,
        }
    }
}

#[derive(Debug, Clone)]
struct SampleInfo {
    name: String,
    status: AxscomSampleStatus,
    program: String,
    position: String,
}

/// AXSCOM Sample Status Codes (from log analysis)
#[derive(Debug, Clone, PartialEq)]
enum AxscomSampleStatus {
    Unknown,
    MeasurementRunning,
    MeasurementFinishedWaitingForEvaluation,
    MeasurementCancelledByHost,
    MeasurementCancelledByOperatorOrDefect,
    NoMeasurementActive,
}

impl AxscomSampleStatus {
    fn from_code(code: u8) -> Self {
        match code {
            0 => Self::NoMeasurementActive,
            1 => Self::MeasurementRunning,
            11 => Self::MeasurementFinishedWaitingForEvaluation,
            14 => Self::MeasurementCancelledByHost,
            15 => Self::MeasurementCancelledByOperatorOrDefect,
            _ => Self::Unknown,
        }
    }
}

/// AXSCOM Command Response
#[derive(Debug, Clone)]
enum CommandResponse {
    Yes,   // Command accepted
    No,    // Command rejected (syntax error)
    Mv,    // Mastership violation
    Start, // Operational command started
    Done,  // Operational command completed
    Err,   // Operational command failed
    Skip,  // Operational command skipped
}

impl BrukerAxscomDetector {
    pub fn new(config: AxscomConfig) -> Self {
        Self {
            config,
            command_socket: Arc::new(Mutex::new(None)),
            file_socket: Arc::new(Mutex::new(None)),
            status_socket: Arc::new(Mutex::new(None)),
            status: Arc::new(RwLock::new(DetectorStatus::Off)),
            health: Arc::new(RwLock::new(DetectorHealth {
                powered: false,
                temperature: 0.0,
                voltage: 0.0,
                status: DetectorStatus::Off,
                last_exposure_time: None,
                total_exposures: 0,
                uptime: Duration::from_secs(0),
            })),
            last_result: Arc::new(RwLock::new(None)),
            current_sample: Arc::new(RwLock::new(None)),
        }
    }

    /// Connect to AXSCOM server (all three sockets)
    async fn connect(&self) -> Result<(), DetectorError> {
        info!("Connecting to AXSCOM server at {}...", self.config.server_ip);

        // Connect command socket
        let cmd_addr = format!("{}:{}", self.config.server_ip, self.config.command_port);
        let cmd_stream = timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            TcpStream::connect(&cmd_addr),
        )
        .await
        .map_err(|_| DetectorError::HardwareError("Command socket connection timeout".to_string()))?
        .map_err(|e| DetectorError::HardwareError(format!("Command socket connection failed: {}", e)))?;
        
        *self.command_socket.lock().await = Some(cmd_stream);
        info!("Command socket connected");

        // Connect file socket
        let file_addr = format!("{}:{}", self.config.server_ip, self.config.file_port);
        let file_stream = timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            TcpStream::connect(&file_addr),
        )
        .await
        .map_err(|_| DetectorError::HardwareError("File socket connection timeout".to_string()))?
        .map_err(|e| DetectorError::HardwareError(format!("File socket connection failed: {}", e)))?;
        
        *self.file_socket.lock().await = Some(file_stream);
        info!("File socket connected");

        // Connect status socket
        let status_addr = format!("{}:{}", self.config.server_ip, self.config.status_port);
        let status_stream = timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            TcpStream::connect(&status_addr),
        )
        .await
        .map_err(|_| DetectorError::HardwareError("Status socket connection timeout".to_string()))?
        .map_err(|e| DetectorError::HardwareError(format!("Status socket connection failed: {}", e)))?;
        
        *self.status_socket.lock().await = Some(status_stream);
        info!("Status socket connected");

        // Start status listener task
        self.start_status_listener();

        Ok(())
    }

    /// Send AXSCOM command
    /// 
    /// Format: [VERB /ARG1=VAL1 /ARG2=VAL2]\n
    async fn send_command(
        &self,
        verb: &str,
        args: &HashMap<String, String>,
    ) -> Result<CommandResponse, DetectorError> {
        let mut cmd_sock = self.command_socket.lock().await;
        let socket = cmd_sock
            .as_mut()
            .ok_or_else(|| DetectorError::HardwareError("Command socket not connected".to_string()))?;

        // Build command string
        let mut command = format!("[{}", verb.to_uppercase());
        for (key, value) in args {
            command.push_str(&format!(" /{}={}", key.to_uppercase(), value));
        }
        command.push_str("]\n");

        debug!("Sending AXSCOM command: {}", command.trim());

        // Send command
        socket
            .write_all(command.as_bytes())
            .await
            .map_err(|e| DetectorError::HardwareError(format!("Failed to send command: {}", e)))?;

        // Read response with timeout
        let mut buffer = vec![0u8; 1024];
        let bytes_read = timeout(
            Duration::from_millis(self.config.command_timeout_ms),
            socket.read(&mut buffer),
        )
        .await
        .map_err(|_| DetectorError::HardwareError("Command response timeout".to_string()))?
        .map_err(|e| DetectorError::HardwareError(format!("Failed to read response: {}", e)))?;

        let response = String::from_utf8_lossy(&buffer[..bytes_read]);
        debug!("AXSCOM response: {}", response.trim());

        // Parse response
        if response.contains("_YES") {
            Ok(CommandResponse::Yes)
        } else if response.contains("_NO") {
            Err(DetectorError::HardwareError("Command rejected (syntax error)".to_string()))
        } else if response.contains("_MV") {
            Err(DetectorError::HardwareError("Command rejected (mastership violation)".to_string()))
        } else if response.contains("_START") {
            Ok(CommandResponse::Start)
        } else if response.contains("_DONE") {
            Ok(CommandResponse::Done)
        } else if response.contains("_ERR") {
            Err(DetectorError::HardwareError("Command failed on device".to_string()))
        } else if response.contains("_SKIP") {
            Ok(CommandResponse::Skip)
        } else {
            warn!("Unknown AXSCOM response: {}", response);
            Ok(CommandResponse::Yes) // Assume success
        }
    }

    /// Start background task to listen for status updates
    fn start_status_listener(&self) {
        let status_socket = Arc::clone(&self.status_socket);
        let status = Arc::clone(&self.status);

        tokio::spawn(async move {
            loop {
                let mut sock = status_socket.lock().await;
                if let Some(socket) = sock.as_mut() {
                    let mut buffer = vec![0u8; 4096];
                    match socket.read(&mut buffer).await {
                        Ok(bytes_read) if bytes_read > 0 => {
                            let message = String::from_utf8_lossy(&buffer[..bytes_read]);
                            debug!("Status update: {}", message.trim());
                            
                            // Parse status updates here
                            // TODO: Implement status parsing based on protocol
                        }
                        Ok(_) => {
                            warn!("Status socket closed by server");
                            break;
                        }
                        Err(e) => {
                            error!("Status socket read error: {}", e);
                            break;
                        }
                    }
                } else {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        });
    }

    /// Get instrument status
    async fn get_instrument_status(&self) -> Result<String, DetectorError> {
        let response = self.send_command("STATUS", &HashMap::new()).await?;
        // TODO: Parse and return structured status
        Ok("OK".to_string())
    }

    /// Get sample list status
    async fn get_sample_list_status(&self) -> Result<Vec<String>, DetectorError> {
        let mut args = HashMap::new();
        args.insert("ALL".to_string(), "$ALL".to_string());
        
        let response = self.send_command("STATUS", &args).await?;
        // TODO: Parse sample list from response
        Ok(Vec::new())
    }

    /// Start measurement (MEASMP command)
    async fn start_measurement(
        &self,
        sample_name: &str,
        program: &str,
        position: &str,
    ) -> Result<(), DetectorError> {
        let mut args = HashMap::new();
        args.insert("NAME".to_string(), format!("\"{}\"", sample_name));
        args.insert("PROGRAM".to_string(), program.to_string());
        args.insert("POS".to_string(), position.to_string());
        args.insert("MODE".to_string(), "UN".to_string());
        args.insert("REP".to_string(), "0".to_string());
        args.insert("SCALE".to_string(), "1".to_string());
        args.insert("PRIORITY".to_string(), "0".to_string());
        args.insert("SCANTIME".to_string(), "10".to_string());
        args.insert("FLAGS".to_string(), "00".to_string());

        self.send_command("MEASMP", &args).await?;
        
        // Store current sample info
        *self.current_sample.write().await = Some(SampleInfo {
            name: sample_name.to_string(),
            status: AxscomSampleStatus::MeasurementRunning,
            program: program.to_string(),
            position: position.to_string(),
        });

        Ok(())
    }

    /// Cancel sample
    async fn cancel_sample(&self, sample_name: &str) -> Result<(), DetectorError> {
        let mut args = HashMap::new();
        args.insert("NAME".to_string(), format!("\"{}\"", sample_name));
        
        self.send_command("CANCEL", &args).await?;
        Ok(())
    }

    /// Poll sample status
    async fn poll_sample_status(&self, sample_name: &str) -> Result<AxscomSampleStatus, DetectorError> {
        let mut args = HashMap::new();
        args.insert("NAME".to_string(), format!("\"{}\"", sample_name));
        
        self.send_command("STATUS", &args).await?;
        
        // TODO: Parse status code from response
        // For now, return running
        Ok(AxscomSampleStatus::MeasurementRunning)
    }
}

#[async_trait]
impl DetectorDevice for BrukerAxscomDetector {
    async fn power_on(&self) -> Result<(), DetectorError> {
        info!("Powering on Bruker AXSCOM detector");
        
        // Connect to AXSCOM server
        self.connect().await?;
        
        // Get initial status
        let _ = self.get_instrument_status().await?;
        
        // Update status
        *self.status.write().await = DetectorStatus::Idle;
        let mut health = self.health.write().await;
        health.powered = true;
        health.status = DetectorStatus::Idle;
        
        info!("Bruker AXSCOM detector powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), DetectorError> {
        info!("Powering off Bruker AXSCOM detector");
        
        // Cancel any active samples
        if let Some(sample) = self.current_sample.read().await.as_ref() {
            let _ = self.cancel_sample(&sample.name).await;
        }
        
        // Close sockets
        *self.command_socket.lock().await = None;
        *self.file_socket.lock().await = None;
        *self.status_socket.lock().await = None;
        
        // Update status
        *self.status.write().await = DetectorStatus::Off;
        let mut health = self.health.write().await;
        health.powered = false;
        health.status = DetectorStatus::Off;
        
        info!("Bruker AXSCOM detector powered off");
        Ok(())
    }

    async fn is_powered(&self) -> bool {
        self.health.read().await.powered
    }

    async fn get_status(&self) -> DetectorStatus {
        self.status.read().await.clone()
    }

    async fn get_health(&self) -> DetectorHealth {
        self.health.read().await.clone()
    }

    async fn start_exposure(&self, exposure_time_ms: u32) -> Result<(), DetectorError> {
        info!("Starting exposure: {}ms", exposure_time_ms);
        
        if !self.is_powered().await {
            return Err(DetectorError::NotPowered);
        }

        // Generate sample name with timestamp
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let sample_name = format!("SAMPLE_{}", timestamp);
        
        // Start measurement
        self.start_measurement(
            &sample_name,
            "EOSDX_JOBTEMPLATE",
            "1A01",
        )
        .await?;
        
        // Update status
        *self.status.write().await = DetectorStatus::Exposing;
        
        // Start polling task
        let detector = self.clone();
        let sample_name_clone = sample_name.clone();
        tokio::spawn(async move {
            detector.poll_measurement_completion(&sample_name_clone).await;
        });
        
        Ok(())
    }

    async fn stop_exposure(&self) -> Result<(), DetectorError> {
        info!("Stopping exposure");
        
        if let Some(sample) = self.current_sample.read().await.as_ref() {
            self.cancel_sample(&sample.name).await?;
        }
        
        *self.status.write().await = DetectorStatus::Idle;
        Ok(())
    }

    async fn get_last_result(&self) -> Option<ExposureResult> {
        self.last_result.read().await.clone()
    }

    async fn calibrate(&self) -> Result<(), DetectorError> {
        info!("Calibrating Bruker AXSCOM detector");
        
        // Calibration would follow similar pattern to measurements
        // but use calibration-specific sample naming and parameters
        
        Ok(())
    }
}

impl Clone for BrukerAxscomDetector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            command_socket: Arc::clone(&self.command_socket),
            file_socket: Arc::clone(&self.file_socket),
            status_socket: Arc::clone(&self.status_socket),
            status: Arc::clone(&self.status),
            health: Arc::clone(&self.health),
            last_result: Arc::clone(&self.last_result),
            current_sample: Arc::clone(&self.current_sample),
        }
    }
}

impl BrukerAxscomDetector {
    /// Poll measurement until completion
    async fn poll_measurement_completion(&self, sample_name: &str) {
        loop {
            tokio::time::sleep(Duration::from_millis(self.config.status_poll_interval_ms)).await;
            
            match self.poll_sample_status(sample_name).await {
                Ok(status) => {
                    match status {
                        AxscomSampleStatus::MeasurementFinishedWaitingForEvaluation => {
                            info!("Measurement completed: {}", sample_name);
                            
                            // Cancel to remove from queue
                            let _ = self.cancel_sample(sample_name).await;
                            
                            // Update status
                            *self.status.write().await = DetectorStatus::Idle;
                            
                            // Store result
                            *self.last_result.write().await = Some(ExposureResult {
                                exposure_time_ms: 0, // TODO: Get from sample info
                                timestamp: chrono::Utc::now(),
                                data_size: 0,
                                data_path: Some(format!("{}.gfrm", sample_name)),
                                detector_temp: 0.0,
                            });
                            
                            break;
                        }
                        AxscomSampleStatus::MeasurementCancelledByHost |
                        AxscomSampleStatus::MeasurementCancelledByOperatorOrDefect => {
                            warn!("Measurement cancelled: {}", sample_name);
                            *self.status.write().await = DetectorStatus::Idle;
                            break;
                        }
                        _ => {
                            debug!("Measurement in progress: {}", sample_name);
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to poll measurement status: {}", e);
                    *self.status.write().await = DetectorStatus::Error(e.to_string());
                    break;
                }
            }
        }
    }
}
