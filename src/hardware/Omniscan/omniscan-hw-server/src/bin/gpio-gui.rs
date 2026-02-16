use omniscan_hw_server::devices::gpio::{DemoGpio, GpioControlGui};
use std::sync::Arc;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), eframe::Error> {
    // Initialize tracing subscriber for console output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    // Initialize a DemoGpio instance (auto-powered, simulating hardware)
    let gpio = Arc::new(DemoGpio::new());
    
    // Arm all interlocks for testing
    if let Err(e) = gpio.arm_all_interlocks().await {
        eprintln!("Failed to arm interlocks: {}", e);
    }
    
    println!("🖥️  Launching GPIO Control GUI in DEMO mode...");
    println!("📝 GPIO hardware ready (auto-initialized)");
    println!("🔴 Click ACTIVATE button for 20-second timer");
    
    // Launch the GUI in DEMO mode (full control enabled)
    GpioControlGui::run(gpio, true)
}
