// ============================================================================
// EXTERNAL CRATE IMPORTS
// ============================================================================
use anyhow::{Result, anyhow};      // Error handling with context
use clap::Parser;                   // CLI argument parsing
use std::net::SocketAddr;           // Network socket address binding
use std::path::PathBuf;             // File path handling
use std::sync::Arc;                 // Atomic reference counting for thread-safe sharing
use tokio::sync::RwLock;            // Async-aware read-write lock for shared mutable state
use tonic::transport::Server;       // gRPC server framework
use tracing::{info, warn, error};   // Structured logging macros

// ============================================================================
// MODULE DECLARATIONS (internal modules that will be compiled into this binary)
// ============================================================================
mod audit;              // FDA compliance audit logging - tracks all commands for regulatory requirements
mod calibration;        // Device calibration routines - not directly used in main, but available for gRPC services
mod calibration_qc;     // QC (quality control) for calibration - supports calibration module
mod certificates;       // mTLS certificate management - used for secure device communication
mod commands;           // Declarative command schemas and registry
mod config;             // Server configuration loading/parsing - loads TOML config file
mod devices;            // Hardware device abstractions (detector, motion, GPIO, PDU)
mod grpc;               // gRPC service definitions and handlers - exposes all RPCs to clients
#[cfg(feature = "gui")]  // Conditional compilation: only if "gui" feature is enabled
mod gui;                // Desktop GUI using ImGui - alternative to headless server
mod logging;            // Structured logging setup and configuration
mod safety;             // Safety state machine for FDA interlocks and emergency stops

// ============================================================================
// SPECIFIC IMPORTS FROM INTERNAL MODULES (actually used in main.rs)
// ============================================================================
use crate::audit::{AuditLogger, CommandLog, CommandResult};
  // AuditLogger: Creates/manages audit database for FDA compliance
  // CommandLog: Represents a logged command (startup, operations, etc.)
  // CommandResult: Enum for command success/failure status

use crate::config::ServerConfig;
  // Loaded from TOML file via --config argument
  // Contains device settings, logging paths, certificate config, workflow settings

use crate::devices::detectors::DemoDetector;
  // Mock/demo X-ray detector - provides simulated detection capabilities
  // Used to initialize hardware devices

use crate::devices::motions::XYDemoMotion;
  // Mock/demo XY motion controller - simulates gantry/stage movement
  // Used to initialize hardware devices

use crate::devices::gpio::DemoGpio;
  // Mock/demo GPIO controller - simulates digital I/O and interlocks
  // Reads door sensors, emergency stops, radiation safety signals
  // Emits state notifications that both GUI and server listen to

use crate::devices::pdu::DemoPdu;
  // Mock/demo Power Distribution Unit - simulates power management
  // Controls power to medical device components

use crate::grpc::services::*;
  // All gRPC service implementations:
  // - AcquisitionService (X-ray scans)
  // - MotionService (gantry movement)
  // - DeviceControlService (general device commands)
  // - DeviceInitializationService (setup/calibration)
  // - HealthService (system status)
  // - SafetyService (interlock management)
  // - StateMonitorService (real-time notifications)
  // - CommandDiscoveryService (lists available commands)

use crate::grpc::*;
  // Additional gRPC utilities:
  // - ServiceState: Shared state passed to all gRPC handlers
  // - Server wrappers (AcquisitionServer, MotionServer, etc.)
  // - Proto definitions

use crate::logging::setup_logging;
  // Configures structured logging (tracing) based on config file
  // Sets up log files and console output

use crate::safety::SafetyStateMachine;
  // FDA safety state machine - enforces interlocks and emergency protocols
  // Validates device is safe before operations (door closed, e-stop not triggered, etc.)

async fn run_grpc_server(
    args: Arc<Args>,
    config: Arc<ServerConfig>,
    demo_gpio: Arc<DemoGpio>,
    demo_detector: Arc<DemoDetector>,
    demo_motion: Arc<XYDemoMotion>,
    demo_pdu: Arc<DemoPdu>,
    state_notifications_tx: tokio::sync::broadcast::Sender<crate::grpc::hub::v1::StateChangeNotification>,
) -> Result<()> {
    // Initialize audit logger for FDA compliance
    let audit_db_path = format!("{}/omniscan_audit.db", config.logging.log_dir);
    let audit_logger = match AuditLogger::new(&audit_db_path, config.certificates.device_certificate_number.clone()).await {
        Ok(logger) => {
            info!("📋 Medical device audit logger initialized");
            Arc::new(logger)
        }
        Err(e) => {
            error!("❌ CRITICAL: Failed to initialize audit logger: {}", e);
            return Err(anyhow!("Audit logging is required for FDA compliance: {}", e));
        }
    };

    // Use pre-initialized device components (cast to trait objects for gRPC services)
    let detector = demo_detector as Arc<dyn crate::devices::DetectorDevice + Send + Sync>;
    let motion = demo_motion as Arc<dyn crate::devices::motions::MotionControlDevice + Send + Sync>;
    let gpio = demo_gpio.clone() as Arc<dyn crate::devices::gpio::GpioDevice + Send + Sync>;
    
    // GPIO is always ready (hardware starts with computer)
    info!("🔌 GPIO hardware ready - reading interlocks");
    
    if config.device.interlocks_armed {
        info!("🔧 DEV MODE: Arming all interlocks for testing");
        if let Err(e) = demo_gpio.arm_all_interlocks().await {
            warn!("Failed to arm interlocks: {}", e);
        }
    }
    
    info!("🔬 Medical device components initialized: Detector, Motion, GPIO");
    
    // Notification sender was already configured in main() before cloning
    info!("✅ Using shared GPIO notification sender");
    
    // Initialize safety state machine
    let mut safety_state_machine = SafetyStateMachine::new(audit_logger.clone());
    let gpio_interlocks = gpio.get_interlocks().await;
    let safety_interlocks = crate::safety::InterlockStatus {
        key_switch: true,
        enable_button: true,
        emergency_stop: gpio_interlocks.emergency_stop,
        door_closed: gpio_interlocks.door_closed,
        beam_watchdog: gpio_interlocks.radiation_safe,
        over_temperature: Some(gpio_interlocks.cooling_ok),
        overall_safe: false,
        violation_reason: None,
        last_check: chrono::Utc::now(),
    };
    if let Err(e) = safety_state_machine.update_interlocks(safety_interlocks).await {
        warn!("Failed to sync GPIO interlocks: {}", e);
    }
    
    if let Err(e) = safety_state_machine.initialize().await {
        error!("❌ CRITICAL: Failed to initialize safety state machine: {}", e);
        return Err(anyhow!("Safety system initialization failed: {}", e));
    }
    let safety_state_machine = Arc::new(RwLock::new(safety_state_machine));
    
    info!("🛡️  Safety state machine initialized");

    // Log system startup
    let startup_log = CommandLog {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: "medical_device_session".to_string(),
        timestamp: chrono::Utc::now(),
        command_type: "system_startup".to_string(),
        command_data: serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "config_file": args.config.display().to_string(),
            "grpc_address": &args.grpc_addr,
            "skip_interlocks": args.skip_interlocks,
            "maintenance_mode": args.maintenance_mode,
            "compliance": "FDA/IEC 62304 Class B"
        }),
        user_context: Some("system".to_string()),
        device_state_before: serde_json::json!({"status": "initializing", "safety_state": "LOCKED"}),
        device_state_after: Some(serde_json::json!({"status": "ready", "safety_state": "IDLE"})),
        result: CommandResult::Success,
        execution_time_ms: 0,
    };
    
    if let Err(e) = audit_logger.log_command(startup_log).await {
        error!("❌ Failed to log system startup: {}", e);
        return Err(anyhow!("Audit logging failure: {}", e));
    }

    // Create shared state for gRPC services
    let device_uuid = if config.certificates.enable_mtls {
        Some(config.certificates.device_uuid.clone())
    } else {
        None
    };
    
    // Broadcast channel was created earlier and set on GPIO
    let service_state = Arc::new(ServiceState {
        safety_state_machine,
        detector,
        motion,
        gpio,
        pdu: demo_pdu,
        audit_logger,
        device_uuid,
        state_notifications: state_notifications_tx,
    });

    // Initialize gRPC services
    let acquisition_service = AcquisitionServer::new(AcquisitionService::new(service_state.clone()));
    let motion_service = MotionServer::new(MotionService::new(service_state.clone()));
    let device_control_service = DeviceControlServer::new(DeviceControlService::new(service_state.clone()));
    let device_init_service = DeviceInitializationServer::new(DeviceInitializationService::new(service_state.clone()));
    let health_service = HealthServer::new(HealthService::new(service_state.clone()));
    let safety_service = SafetyServer::new(SafetyService::new(service_state.clone()));
    let state_monitor_service = StateMonitorServer::new(StateMonitorService::new(service_state.clone()));
    let command_discovery_service = CommandDiscoveryServer::new(CommandDiscoveryService::new(service_state.clone()));

    info!("🌐 gRPC services initialized (including StateMonitor, DeviceInitialization, and CommandDiscovery)");

    // Parse gRPC address
    let grpc_addr: SocketAddr = args.grpc_addr.parse()
        .map_err(|e| anyhow!("Invalid gRPC address '{}': {}", args.grpc_addr, e))?;

    info!("🚀 Starting gRPC server on {}", grpc_addr);
    
    // Configure TLS if enabled
    let mut server_builder = Server::builder();
    
    if config.certificates.enable_mtls {
        warn!("⚠️  mTLS configuration requested but requires tonic >= 0.12");
        warn!("⚠️  Running in plaintext mode");
    } else {
        info!("ℹ️  Running in plaintext mode (development only)");
    }
    
    info!("🛡️  Safety Authority: Rust Hardware Server");

    // Start gRPC server
    server_builder
        .add_service(acquisition_service)
        .add_service(motion_service)
        .add_service(device_control_service)
        .add_service(device_init_service)
        .add_service(health_service)
        .add_service(safety_service)
        .add_service(state_monitor_service)
        .add_service(command_discovery_service)
        .serve(grpc_addr)
        .await
        .map_err(|e| anyhow!("gRPC server error: {}", e))?;

    Ok(())
}

#[derive(Parser)]
#[command(name = "omniscan-hw-server")]
#[command(about = "OMNIScan medical device hardware server - gRPC only")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/server.toml")]
    config: PathBuf,

    /// gRPC server bind address
    #[arg(long, default_value = "[::1]:50051")] // IPv6 localhost, standard gRPC port
    grpc_addr: String,

    /// Skip interlock checks (DANGEROUS - for testing only)
    #[arg(long)]
    skip_interlocks: bool,

    /// Start in maintenance mode
    #[arg(long)]
    maintenance_mode: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let args = Arc::new(args);
    
    // Load config first (synchronous)
    let config = ServerConfig::load_from_file(&args.config)?;
    let config = Arc::new(config);
    
    // Setup logging
    setup_logging(&config)?;

    info!("🩺 Starting OMNIScan Medical Device Hardware Server v{}", env!("CARGO_PKG_VERSION"));
    info!("📁 Config loaded from: {}", args.config.display());
    info!("🔒 FDA/IEC 62304 Class B Medical Software");
    
    if args.skip_interlocks {
        warn!("⚠️  DANGER: Interlock checks DISABLED - Testing mode only!");
    }
    
    if args.maintenance_mode {
        warn!("🔧 Starting in maintenance mode");
    }
    
    // Create a Tokio runtime early so background tasks can be spawned during device init
    let runtime = tokio::runtime::Runtime::new()?;
    let _rt_guard = runtime.enter();
    
    // Initialize devices (needed by both GUI and server)
    let demo_gpio = Arc::new(DemoGpio::with_config(config.workflow.enable_button_timeout));
    let demo_detector = Arc::new(DemoDetector::new());
    let demo_motion = Arc::new(XYDemoMotion::default());
    let demo_pdu = Arc::new(DemoPdu::new());
    
    // Create broadcast channel for state change notifications EARLY (before any cloning)
    // So both GUI and server share the same notifier
    let (state_notifications_tx, _) = tokio::sync::broadcast::channel(100);
    
    // Set notifier on devices BEFORE cloning for GUI
    // IMPORTANT: Must set it synchronously before any cloning happens
    demo_gpio.set_notifier_sync(state_notifications_tx.clone());
    demo_pdu.set_notifier_sync(state_notifications_tx.clone());
    info!("✅ Notification sender configured for GPIO and PDU (shared with GUI)");
    
    #[cfg(feature = "gui")]
    if config.device.gui_mode {
        // GUI MODE: Run GUI on main thread, server in background
        let is_demo_mode = config.device.name == "GPIO";
        
        if is_demo_mode {
            info!("🖥️  Starting in DEMO mode with GUI");
        } else {
            info!("🖥️  Starting in Production mode with GUI");
        }
        
        // Spawn gRPC server in background thread
        let args_clone = args.clone();
        let config_clone = config.clone();
        let gpio_clone = demo_gpio.clone();
        let detector_clone = demo_detector.clone();
        let motion_clone = demo_motion.clone();
        let pdu_clone = demo_pdu.clone();
        let state_notifications_clone = state_notifications_tx.clone();
        
        std::thread::spawn(move || {
            // Create Tokio runtime for server thread
            let runtime = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
            
            runtime.block_on(async {
                if let Err(e) = run_grpc_server(args_clone, config_clone, gpio_clone, detector_clone, motion_clone, pdu_clone, state_notifications_clone).await {
                    error!("gRPC server failed: {}", e);
                }
            });
        });
        
        // Give server time to start
        std::thread::sleep(std::time::Duration::from_millis(1000));
        
        // Run GUI on main thread (blocks until GUI closes)
        let server_gui = crate::gui::ServerGui::with_notifier(
            demo_gpio,
            demo_detector,
            demo_motion,
            demo_pdu,
            args.grpc_addr.clone(),
            args.config.display().to_string(),
            is_demo_mode,
            Some(state_notifications_tx.clone()),
        );
        
        info!("🖥️  Launching Server GUI...");
        if let Err(e) = server_gui.run() {
            error!("GUI error: {}", e);
            return Err(anyhow!("GUI failed to start: {}", e));
        }
        
        info!("GUI closed. Server continues running in background.");
        info!("Press Ctrl+C to stop the server.");
        
        // Keep main thread alive
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    } else {
        // NO GUI MODE: Run server normally on main thread
        info!("🚀 Starting server without GUI");
        
        runtime.block_on(async {
            run_grpc_server(args, config, demo_gpio, demo_detector, demo_motion, demo_pdu.clone(), state_notifications_tx.clone()).await
        })?;
    }
    
    #[cfg(not(feature = "gui"))]
    {
        // GUI feature not enabled: Run server normally
        info!("🚀 Starting server (GUI feature not enabled)");
        
        runtime.block_on(async {
            run_grpc_server(args, config, demo_gpio, demo_detector, demo_motion, demo_pdu, state_notifications_tx.clone()).await
        })?;
    }
    
    Ok(())
}
