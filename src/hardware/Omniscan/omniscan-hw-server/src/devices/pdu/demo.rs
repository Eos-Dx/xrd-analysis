use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

type StateNotificationSender = tokio::sync::broadcast::Sender<crate::grpc::hub::v1::StateChangeNotification>;

/// Simple DEMO PDU (Power Distribution Unit) model
/// - Tracks overall machine power state (ON/OFF)
/// - GUI controls toggle this state
/// - In demo, GUI is responsible for syncing GPIO.power_ok with PDU state
pub struct DemoPdu {
    pub state: Arc<RwLock<DemoPduState>>,
    notifier: Arc<RwLock<Option<StateNotificationSender>>>,
}

#[derive(Debug, Clone)]
pub struct DemoPduState {
    pub powered: bool,
    pub last_changed: Instant,
}

impl DemoPdu {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DemoPduState {
                powered: false,
                last_changed: Instant::now(),
            })),
            notifier: Arc::new(RwLock::new(None)),
        }
    }

    /// Configure state change notification sender (sync)
    pub fn set_notifier_sync(&self, notifier: StateNotificationSender) {
        let mut n = self.notifier.blocking_write();
        *n = Some(notifier);
    }

    fn emit_notification(&self, component: &str, change_type: &str) {
        if let Ok(notifier_lock) = self.notifier.try_read() {
            if let Some(tx) = notifier_lock.as_ref() {
                let notification = crate::grpc::hub::v1::StateChangeNotification {
                    component: component.to_string(),
                    change_type: change_type.to_string(),
                    timestamp: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                };
                let _ = tx.send(notification);
            }
        }
    }

    /// Synchronous helper for GUI: set power state
    pub fn set_powered_sync(&self, on: bool) {
        let mut changed = false;
        if let Ok(mut s) = self.state.try_write() {
            if s.powered != on {
                s.powered = on;
                s.last_changed = Instant::now();
                changed = true;
                info!("DEMO PDU: Power {}", if on { "ON" } else { "OFF" });
            }
        } else {
            // Fallback to blocking write if try_write fails
            let mut s = self.state.blocking_write();
            if s.powered != on {
                s.powered = on;
                s.last_changed = Instant::now();
                changed = true;
                info!("DEMO PDU: Power {}", if on { "ON" } else { "OFF" });
            }
        }
        if changed {
            // Emit PDU power change notification
            self.emit_notification("PDU", "POWER_CHANGED");
        }
    }

    /// Synchronous helper for GUI: read power state
    pub fn is_powered_sync(&self) -> bool {
        if let Ok(s) = self.state.try_read() {
            s.powered
        } else {
            self.state.blocking_read().powered
        }
    }
}