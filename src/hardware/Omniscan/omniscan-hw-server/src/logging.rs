use anyhow::Result;
use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

use crate::config::ServerConfig;

pub fn setup_logging(config: &ServerConfig) -> Result<()> {
    let log_level = match config.logging.log_level.to_uppercase().as_str() {
        "TRACE" => Level::TRACE,
        "DEBUG" => Level::DEBUG,
        "INFO" => Level::INFO,
        "WARN" => Level::WARN,
        "ERROR" => Level::ERROR,
        _ => Level::INFO,
    };

    // Ensure log directory exists
    std::fs::create_dir_all(&config.logging.log_dir)?;

    // File appender (daily rotation)
    let file_appender = tracing_appender::rolling::daily(
        &config.logging.log_dir,
        "server.log",
    );

    // Console layer (structured but readable)
    let console_layer = fmt::Layer::new()
        .with_target(true)
        .with_thread_ids(false)
        .with_level(true)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(EnvFilter::from_default_env().add_directive(log_level.into()));

    // File layer (JSON format for machine processing)
    let file_layer = fmt::Layer::new()
        .json()
        .with_writer(file_appender)
        .with_current_span(true)
        .with_span_list(true)
        .with_filter(EnvFilter::from_default_env().add_directive(log_level.into()));

    // Initialize subscriber
    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    tracing::info!(
        "Logging initialized - Level: {}, Directory: {}",
        log_level,
        config.logging.log_dir
    );

    Ok(())
}