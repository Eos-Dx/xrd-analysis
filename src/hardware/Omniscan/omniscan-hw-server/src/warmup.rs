use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Warmup manager for X-ray source heat-up tracking
pub struct WarmupManager {
    start_time: Option<Instant>,
    duration: Duration,
    // Future: add temperature and current monitoring
    // temperature_sensor: Option<Arc<TemperatureSensor>>,
    // current_monitor: Option<Arc<CurrentMonitor>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupStatus {
    pub is_warming_up: bool,
    pub elapsed_seconds: u32,
    pub total_seconds: u32,
    pub progress_percent: u32,
    // Future: real sensor data
    pub current_temperature: f64,
    pub current_current: f64,
    pub target_temperature: f64,
    pub target_current: f64,
}

impl WarmupManager {
    /// Create a new warmup manager with default 10-minute duration
    pub fn new() -> Self {
        Self::with_duration(Duration::from_secs(600)) // 10 minutes
    }
    
    /// Create a warmup manager with custom duration
    pub fn with_duration(duration: Duration) -> Self {
        Self {
            start_time: None,
            duration,
        }
    }
    
    /// Start the warmup timer
    pub fn start_warmup(&mut self) {
        self.start_time = Some(Instant::now());
        info!("Warmup started: {} seconds duration", self.duration.as_secs());
    }
    
    /// Check if warmup is complete
    pub fn is_complete(&self) -> bool {
        match self.start_time {
            Some(start) => start.elapsed() >= self.duration,
            None => false,
        }
    }
    
    /// Check if warmup is in progress
    pub fn is_warming_up(&self) -> bool {
        match self.start_time {
            Some(start) => start.elapsed() < self.duration,
            None => false,
        }
    }
    
    /// Get current warmup status
    pub fn get_status(&self) -> WarmupStatus {
        match self.start_time {
            Some(start) => {
                let elapsed = start.elapsed();
                let elapsed_secs = elapsed.as_secs() as u32;
                let total_secs = self.duration.as_secs() as u32;
                let progress_percent = if total_secs > 0 {
                    ((elapsed_secs as f64 / total_secs as f64) * 100.0).min(100.0) as u32
                } else {
                    100
                };
                
                WarmupStatus {
                    is_warming_up: elapsed < self.duration,
                    elapsed_seconds: elapsed_secs,
                    total_seconds: total_secs,
                    progress_percent,
                    current_temperature: self.get_temperature(),
                    current_current: self.get_current(),
                    target_temperature: 0.0, // Future: from config
                    target_current: 0.0,      // Future: from config
                }
            },
            None => WarmupStatus {
                is_warming_up: false,
                elapsed_seconds: 0,
                total_seconds: self.duration.as_secs() as u32,
                progress_percent: 0,
                current_temperature: 0.0,
                current_current: 0.0,
                target_temperature: 0.0,
                target_current: 0.0,
            },
        }
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }
    
    /// Reset warmup timer
    pub fn reset(&mut self) {
        info!("Warmup reset");
        self.start_time = None;
    }
    
    // Future: read from actual temperature sensor
    fn get_temperature(&self) -> f64 {
        0.0
    }
    
    // Future: read from actual current monitor
    fn get_current(&self) -> f64 {
        0.0
    }
}

impl Default for WarmupManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_warmup_not_started() {
        let manager = WarmupManager::new();
        assert!(!manager.is_warming_up());
        assert!(!manager.is_complete());
        
        let status = manager.get_status();
        assert!(!status.is_warming_up);
        assert_eq!(status.elapsed_seconds, 0);
        assert_eq!(status.progress_percent, 0);
    }
    
    #[test]
    fn test_warmup_start() {
        let mut manager = WarmupManager::with_duration(Duration::from_secs(2));
        manager.start_warmup();
        
        assert!(manager.is_warming_up());
        assert!(!manager.is_complete());
        
        let status = manager.get_status();
        assert!(status.is_warming_up);
    }
    
    #[test]
    fn test_warmup_complete() {
        let mut manager = WarmupManager::with_duration(Duration::from_millis(100));
        manager.start_warmup();
        
        assert!(manager.is_warming_up());
        thread::sleep(Duration::from_millis(150));
        
        assert!(!manager.is_warming_up());
        assert!(manager.is_complete());
        
        let status = manager.get_status();
        assert!(!status.is_warming_up);
        assert_eq!(status.progress_percent, 100);
    }
    
    #[test]
    fn test_warmup_reset() {
        let mut manager = WarmupManager::with_duration(Duration::from_secs(10));
        manager.start_warmup();
        
        assert!(manager.is_warming_up());
        
        manager.reset();
        assert!(!manager.is_warming_up());
        assert!(!manager.is_complete());
    }
}
