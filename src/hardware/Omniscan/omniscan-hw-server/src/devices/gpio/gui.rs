use eframe::egui;
use std::sync::Arc;
use super::{DemoGpio, LedColor, InterlockStatus};

type StateNotificationSender = tokio::sync::broadcast::Sender<crate::grpc::hub::v1::StateChangeNotification>;

/// GPIO Control GUI - Works in both DEMO and Production modes
/// - DEMO mode: Full control + visualization
/// - Production mode: Visualization only (controls disabled)
pub struct GpioControlGui {
    gpio: Arc<DemoGpio>,
    is_demo_mode: bool,  // Enable controls in DEMO, disable in PROD
    // Cached state to avoid async calls in GUI thread
    key_switch: bool,
    interlocks: InterlockStatus,
    main_led: LedColor,
    radiation_led: LedColor,
    activation_button_active: bool,
    activation_button_remaining: Option<u64>,
    // Notification sender (optional - may not be set in standalone mode)
    notifier: Option<StateNotificationSender>,
}

impl GpioControlGui {
    pub fn new(gpio: Arc<DemoGpio>, is_demo_mode: bool) -> Self {
        Self::with_notifier(gpio, is_demo_mode, None)
    }
    
    pub fn with_notifier(gpio: Arc<DemoGpio>, is_demo_mode: bool, notifier: Option<StateNotificationSender>) -> Self {
        Self {
            gpio,
            is_demo_mode,
            key_switch: false,
            interlocks: InterlockStatus {
                emergency_stop: true,
                door_closed: true,
                radiation_safe: true,
                cooling_ok: true,
                power_ok: true,
                overall_safe: true,
            },
            main_led: LedColor::Red,
            radiation_led: LedColor::Green,
            activation_button_active: false,
            activation_button_remaining: None,
            notifier,
        }
    }
    
    fn emit_notification(&self, component: &str, change_type: &str) {
        if let Some(tx) = &self.notifier {
            let notification = crate::grpc::hub::v1::StateChangeNotification {
                component: component.to_string(),
                change_type: change_type.to_string(),
                timestamp: Some(prost_types::Timestamp {
                    seconds: chrono::Utc::now().timestamp(),
                    nanos: 0,
                }),
            };
            let _ = tx.send(notification); // Best-effort
        }
    }

    /// Launch the GUI window (blocking call - run in separate thread)
    /// - is_demo_mode: true = Enable controls, false = Visualization only
    pub fn run(gpio: Arc<DemoGpio>, is_demo_mode: bool) -> Result<(), eframe::Error> {
        Self::run_with_notifier(gpio, is_demo_mode, None)
    }
    
    /// Launch the GUI window with state change notification support
    pub fn run_with_notifier(gpio: Arc<DemoGpio>, is_demo_mode: bool, notifier: Option<StateNotificationSender>) -> Result<(), eframe::Error> {
        let title = if is_demo_mode {
            "GPIO Control Panel - DEMO Mode (Full Control)"
        } else {
            "GPIO Monitor - Production Mode (Visualization Only)"
        };
        
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([680.0, 650.0])
                .with_title(title),
            ..Default::default()
        };

        eframe::run_native(
            "GPIO Control",
            options,
            Box::new(move |_cc| Ok(Box::new(GpioControlGui::with_notifier(gpio, is_demo_mode, notifier)))),
        )
    }
}

impl eframe::App for GpioControlGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous updates for real-time display
        ctx.request_repaint();

        // Update cached state from GPIO (non-blocking)
        let gpio = self.gpio.clone();
        if let Ok(state) = gpio.state.try_read() {
            self.key_switch = state.key_switch_on;
            // Map key_led (bool) to LedColor for backwards compatibility
            self.main_led = if state.key_led { LedColor::Green } else { LedColor::Red };
            self.radiation_led = state.radiation_led;
            self.interlocks = InterlockStatus {
                emergency_stop: state.emergency_stop_ok,
                door_closed: state.door_closed_ok,
                radiation_safe: state.radiation_safe_input,  // Use INPUT sensor
                cooling_ok: state.cooling_ok,
                power_ok: state.power_ok,
                overall_safe: state.emergency_stop_ok && state.door_closed_ok &&
                             state.radiation_safe_input && state.cooling_ok && state.power_ok,
            };
            
            // Update activation button state
            self.activation_button_active = state.activation_button_active;
            if state.activation_button_active {
                if let Some(expires_at) = state.activation_button_expires_at {
                    let now = std::time::Instant::now();
                    if now < expires_at {
                        self.activation_button_remaining = Some(expires_at.duration_since(now).as_secs());
                    } else {
                        self.activation_button_remaining = None;
                    }
                } else {
                    self.activation_button_remaining = None;
                }
            } else {
                // Button is inactive - clear remaining time
                self.activation_button_remaining = None;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_demo_mode {
                ui.heading(egui::RichText::new("GPIO Control Panel - DEMO Mode").color(egui::Color32::from_rgb(255, 165, 0)));
                ui.label(egui::RichText::new("✏️ Full Control Enabled").color(egui::Color32::GREEN));
            } else {
                ui.heading(egui::RichText::new("GPIO Monitor - Production Mode").color(egui::Color32::LIGHT_BLUE));
                ui.label(egui::RichText::new("👁️ Visualization Only - Controls Disabled").color(egui::Color32::YELLOW));
            }
            ui.separator();

            let key_switch = self.key_switch;
            let interlocks = self.interlocks.clone();
            let main_led = self.main_led;
            let radiation_led = self.radiation_led;

            egui::ScrollArea::vertical().show(ui, |ui| {
                // 2-Column Layout: LEFT = GPIO Inputs (readings), RIGHT = Sensor Simulators
                egui::Grid::new("gpio_grid")
                    .num_columns(2)
                    .spacing([15.0, 10.0])
                    .min_col_width(300.0)
                    .striped(false)
                    .show(ui, |ui| {
                        // === LEFT COLUMN: GPIO INPUT READINGS (What the card reads) ===
                        ui.vertical(|ui| {
                            ui.heading(egui::RichText::new("📥 GPIO INPUTS (Pin Readings)").color(egui::Color32::LIGHT_BLUE).strong());
                            ui.label(egui::RichText::new("Read-only sensor states").small().italics().color(egui::Color32::GRAY));
                            ui.add_space(10.0);

                            // Input Pin States Group
                            ui.group(|ui| {
                                ui.set_min_width(300.0);
                                ui.heading("🔌 Input Pins");
                                ui.add_space(5.0);

                                let input_row = |ui: &mut egui::Ui, pin: u8, label: &str, state: bool| {
                                    ui.horizontal(|ui| {
                                        ui.label(format!("Pin {}: {}", pin, label));
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            if state {
                                                ui.colored_label(egui::Color32::GREEN, "✅ HIGH");
                                            } else {
                                                ui.colored_label(egui::Color32::RED, "❌ LOW");
                                            }
                                        });
                                    });
                                };

                                input_row(ui, 1, "E-Stop Released", interlocks.emergency_stop);
                                input_row(ui, 2, "Door Closed", interlocks.door_closed);
                                input_row(ui, 3, "Radiation Safe (Sensor)", interlocks.radiation_safe);
                                input_row(ui, 4, "Cooling OK", interlocks.cooling_ok);
                                input_row(ui, 5, "Power OK", interlocks.power_ok);
                                input_row(ui, 7, "Key Switch ON", key_switch);
                            });

                            ui.add_space(10.0);

                            // Overall Interlock Status
                            ui.group(|ui| {
                                ui.set_min_width(300.0);
                                ui.heading("🛡️ Overall Interlock Status");
                                ui.add_space(5.0);
                                
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("System Status:").strong());
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if interlocks.overall_safe {
                                            ui.colored_label(
                                                egui::Color32::GREEN,
                                                egui::RichText::new("✅ SAFE").size(16.0).strong()
                                            );
                                        } else {
                                            ui.colored_label(
                                                egui::Color32::RED,
                                                egui::RichText::new("❌ UNSAFE").size(16.0).strong()
                                            );
                                        }
                                    });
                                });
                            });

                            ui.add_space(10.0);

                            // LED Status (Output readings)
                            ui.group(|ui| {
                                ui.set_min_width(300.0);
                                ui.heading("💡 LED Outputs");
                                ui.add_space(5.0);
                                
                                ui.horizontal(|ui| {
                                    ui.label("Main Status LED:");
                                    let led_color = match main_led {
                                        LedColor::Red => egui::Color32::RED,
                                        LedColor::Orange => egui::Color32::from_rgb(255, 165, 0),
                                        LedColor::Green => egui::Color32::GREEN,
                                        LedColor::Off => egui::Color32::GRAY,
                                    };
                                    ui.colored_label(led_color, format!("● {:?}", main_led));
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Radiation Warning LED:");
                                    let led_color = match radiation_led {
                                        LedColor::Red => egui::Color32::RED,
                                        LedColor::Orange => egui::Color32::from_rgb(255, 165, 0),
                                        LedColor::Green => egui::Color32::GREEN,
                                        LedColor::Off => egui::Color32::GRAY,
                                    };
                                    ui.colored_label(led_color, format!("● {:?}", radiation_led));
                                });
                            });
                        });

                        // === RIGHT COLUMN: SENSOR SIMULATORS (Physical events) ===
                        ui.vertical(|ui| {
                            ui.heading(egui::RichText::new("🎮 SENSOR SIMULATORS").color(egui::Color32::from_rgb(255, 165, 0)).strong());
                            ui.label(egui::RichText::new("Emulate physical sensor changes").small().italics().color(egui::Color32::GRAY));
                            ui.add_space(10.0);

                            // Key Switch Simulator
                            ui.group(|ui| {
                                ui.set_min_width(300.0);
                                ui.heading("🔑 Key Switch Simulator");
                                if !self.is_demo_mode {
                                    ui.label(egui::RichText::new("🔒 Disabled in Production").color(egui::Color32::GRAY).italics());
                                }
                                ui.add_space(5.0);
                                
                                ui.add_enabled_ui(self.is_demo_mode, |ui| {
                                    ui.horizontal(|ui| {
                                        if ui.button("🔒 Turn OFF").clicked() {
                                            let gpio = self.gpio.clone();
                                            tokio::spawn(async move {
                                                if let Err(e) = gpio.set_key_switch(false).await {
                                                    tracing::error!("Failed to set key switch: {}", e);
                                                }
                                            });
                                            tracing::warn!("🔒 Key Switch turned OFF - System locked");
                                        }
                                        if ui.button("🔓 Turn ON").clicked() {
                                            let gpio = self.gpio.clone();
                                            tokio::spawn(async move {
                                                if let Err(e) = gpio.set_key_switch(true).await {
                                                    tracing::error!("Failed to set key switch: {}", e);
                                                }
                                            });
                                            tracing::info!("🔑 Key Switch turned ON - System unlocked");
                                        }
                                    });
                                    
                                    ui.label(if key_switch {
                                        egui::RichText::new("Current: ON (OPERATE)")
                                            .color(egui::Color32::GREEN)
                                    } else {
                                        egui::RichText::new("Current: OFF (LOCKED)")
                                            .color(egui::Color32::RED)
                                    });
                                });
                            });

                            ui.add_space(10.0);

                            // Safety Sensors Simulator
                            ui.group(|ui| {
                                ui.set_min_width(300.0);
                                ui.heading("🚨 Safety Sensor Simulators");
                                if !self.is_demo_mode {
                                    ui.label(egui::RichText::new("🔒 Disabled in Production").color(egui::Color32::GRAY).italics());
                                }
                                ui.add_space(5.0);

                                ui.add_enabled_ui(self.is_demo_mode, |ui| {
                                    ui.label(egui::RichText::new("Emergency Stop:").strong());
                                    ui.horizontal(|ui| {
                                        if ui.button("❌ Press E-Stop").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(1, super::GpioPinState::Low);
                                                tracing::warn!("🔴 E-Stop PRESSED (pin 1 -> LOW)");
                                            }
                                            self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                        }
                                        if ui.button("✅ Release E-Stop").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(1, super::GpioPinState::High);
                                                tracing::info!("✅ E-Stop RELEASED (pin 1 -> HIGH)");
                                            }
                                            self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                        }
                                    });

                                    ui.add_space(3.0);
                                    ui.label(egui::RichText::new("Door Sensor:").strong());
                                    ui.horizontal(|ui| {
                                        if ui.button("🚪 Open Door").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(2, super::GpioPinState::Low);
                                                tracing::warn!("🚪 Door OPENED (pin 2 -> LOW)");
                                            }
                                            self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                        }
                                        if ui.button("🚪 Close Door").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(2, super::GpioPinState::High);
                                                tracing::info!("🚪 Door CLOSED (pin 2 -> HIGH)");
                                            }
                                            self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                        }
                                    });

                                    ui.add_space(3.0);
                                    ui.label(egui::RichText::new("Beam-Stop Sensor Simulator:").strong());
                                    ui.label(egui::RichText::new("(Manually override sensor reading)").small().italics().color(egui::Color32::GRAY));
                                    ui.horizontal(|ui| {
                                        if ui.button("🔴 Remove Beam-Stop").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(3, super::GpioPinState::Low);
                                                tracing::warn!("🔴 Beam-stop REMOVED - Radiation UNSAFE (pin 3 -> LOW)");
                                            }
                                            self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                        }
                                        if ui.button("🟢 Insert Beam-Stop").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(3, super::GpioPinState::High);
                                                tracing::info!("🟢 Beam-stop INSERTED - Radiation SAFE (pin 3 -> HIGH)");
                                            }
                                            self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                        }
                                    });

                                    ui.add_space(5.0);
                                    ui.separator();
                                    ui.add_space(5.0);
                                    
                                    if ui.button("✅ ARM ALL SENSORS (Safe State)").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(1, super::GpioPinState::High);  // E-stop released
                                            state.simulated_pin_states.insert(2, super::GpioPinState::High);  // Door closed
                                            state.simulated_pin_states.insert(3, super::GpioPinState::High);  // Radiation safe
                                            state.simulated_pin_states.insert(4, super::GpioPinState::High);  // Cooling OK
                                            state.simulated_pin_states.insert(5, super::GpioPinState::High);  // Power OK
                                            tracing::info!("✅ ALL SENSORS ARMED (all pins -> HIGH)");
                                        }
                                        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
                                    }
                                }); // end add_enabled_ui
                            });

                            ui.add_space(10.0);

                            // Activation Button for Non-Safe Commands
                            ui.group(|ui| {
                                ui.set_min_width(300.0);
                                ui.heading("⚡ Non-Safe Activation Button");
                                let timeout_msg = format!("ℹ️ Activates for {} seconds", self.gpio.enable_button_timeout_secs);
                                ui.label(egui::RichText::new(timeout_msg).color(egui::Color32::GRAY).italics());
                                ui.add_space(5.0);
                                
                                ui.add_enabled_ui(self.is_demo_mode, |ui| {
                                    let button_text = if self.activation_button_active {
                                        if let Some(remaining) = self.activation_button_remaining {
                                            format!("✅ ACTIVE ({}s)", remaining)
                                        } else {
                                            "✅ ACTIVE".to_string()
                                        }
                                    } else {
                                        "🔴 ACTIVATE NOW".to_string()
                                    };
                                    
                                    let button_color = if self.activation_button_active {
                                        egui::Color32::GREEN
                                    } else {
                                        egui::Color32::from_rgb(200, 200, 200)
                                    };
                                    
                                    let button = egui::Button::new(
                                        egui::RichText::new(button_text)
                                            .size(18.0)
                                            .strong()
                                            .color(button_color)
                                    );
                                    
                                    if ui.add_sized([ui.available_width(), 50.0], button).clicked() {
                                        let gpio = self.gpio.clone();
                                        tokio::spawn(async move {
                                            if let Err(e) = gpio.activate_enable_button().await {
                                                tracing::error!("Failed to activate enable button: {}", e);
                                            }
                                        });
                                    }
                                });
                                
                                ui.add_space(5.0);
                                
                                if self.activation_button_active {
                                    ui.colored_label(
                                        egui::Color32::GREEN,
                                        egui::RichText::new("✅ Non-safe commands ENABLED").size(14.0).strong()
                                    );
                                    ui.label(
                                        egui::RichText::new("Detector/Motion init allowed").small().color(egui::Color32::GRAY)
                                    );
                                } else {
                                    ui.colored_label(
                                        egui::Color32::RED,
                                        egui::RichText::new("🔒 Non-safe commands BLOCKED").size(14.0)
                                    );
                                }
                            });
                        });
                    });

                    ui.add_space(10.0);
            });
            
            ui.separator();
            if self.is_demo_mode {
                ui.label(
                    egui::RichText::new("⚠️ DEMO Mode - Full Control Enabled - For Testing Only")
                        .italics()
                        .color(egui::Color32::from_rgb(255, 165, 0))
                );
            } else {
                ui.label(
                    egui::RichText::new("🛡️ Production Mode - Visualization Only - Physical GPIO Active")
                        .italics()
                        .color(egui::Color32::LIGHT_BLUE)
                );
            }
        });
    }
}
