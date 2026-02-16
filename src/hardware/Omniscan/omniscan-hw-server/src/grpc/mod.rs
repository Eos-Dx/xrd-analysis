// Generated protobuf code
pub mod hub {
    pub mod v1 {
        tonic::include_proto!("hub.v1");
        
        // Re-export server traits for easier access
        pub use acquisition_server::{Acquisition, AcquisitionServer};
        pub use motion_server::{Motion, MotionServer};
        pub use device_control_server::DeviceControlServer;
        pub use device_initialization_server::{DeviceInitialization, DeviceInitializationServer};
        pub use health_server::HealthServer;
        pub use safety_server::{Safety, SafetyServer};
        pub use state_monitor_server::{StateMonitor, StateMonitorServer};
        pub use command_discovery_server::{CommandDiscovery, CommandDiscoveryServer};
        
    }
}

// Service implementations
pub mod services;
pub mod state_monitor_service;
pub mod command_discovery_service;

// Re-export for convenience
#[allow(unused_imports)]
pub use hub::v1::{AcquisitionServer, MotionServer, DeviceControlServer, DeviceInitializationServer, HealthServer, SafetyServer, StateMonitorServer, CommandDiscoveryServer};
#[allow(unused_imports)]
pub use services::{ServiceState, AcquisitionService, MotionService, DeviceControlService, DeviceInitializationService, HealthService, SafetyService};
#[allow(unused_imports)]
pub use state_monitor_service::StateMonitorService;
#[allow(unused_imports)]
pub use command_discovery_service::CommandDiscoveryService;
