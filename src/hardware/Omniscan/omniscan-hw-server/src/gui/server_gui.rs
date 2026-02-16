use eframe::egui;
use std::sync::Arc;
use std::time::Instant;
use crate::devices::gpio::DemoGpio;
use crate::devices::detectors::DemoDetector;
use crate::devices::motions::XYDemoMotion;
use crate::devices::pdu::DemoPdu;

type StateNotificationSender = tokio::sync::broadcast::Sender<crate::grpc::hub::v1::StateChangeNotification>;

/// Main Server Control GUI - Shows server status and GPIO controls
pub struct ServerGui {
    gpio: Arc<DemoGpio>,
    detector: Arc<DemoDetector>,
    motion: Arc<XYDemoMotion>,
    pdu: Arc<DemoPdu>,
    server_start_time: Instant,
    grpc_address: String,
    config_file: String,
    is_demo_mode: bool,
    show_gpio_panel: bool,
    notifier: Option<StateNotificationSender>,
    // Settings UI state
    measurement_dir_input: String,
    storage_message: Option<String>,
}

impl ServerGui {
    pub fn new(
        gpio: Arc<DemoGpio>,
        detector: Arc<DemoDetector>,
        motion: Arc<XYDemoMotion>,
        pdu: Arc<DemoPdu>,
        grpc_address: String,
        config_file: String,
        is_demo_mode: bool,
    ) -> Self {
        Self::with_notifier(gpio, detector, motion, pdu, grpc_address, config_file, is_demo_mode, None)
    }
    
    pub fn with_notifier(
        gpio: Arc<DemoGpio>,
        detector: Arc<DemoDetector>,
        motion: Arc<XYDemoMotion>,
        pdu: Arc<DemoPdu>,
        grpc_address: String,
        config_file: String,
        is_demo_mode: bool,
        notifier: Option<StateNotificationSender>,
    ) -> Self {
        // Initialize measurement_dir_input from config file
        let measurement_dir_input = match crate::config::ServerConfig::load_from_file(&config_file) {
            Ok(cfg) => cfg.measurement.output_dir,
            Err(_) => "measurements".to_string(),
        };
        
        Self {
            gpio,
            detector,
            motion,
            pdu,
            server_start_time: Instant::now(),
            grpc_address,
            config_file,
            is_demo_mode,
            show_gpio_panel: false,
            notifier,
            measurement_dir_input,
            storage_message: None,
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

    /// Launch the Server GUI window
    pub fn run(self) -> Result<(), eframe::Error> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([780.0, 550.0])
                .with_title("Omniscan Hardware Server - Control Panel"),
            ..Default::default()
        };

        eframe::run_native(
            "Server Control",
            options,
            Box::new(move |_cc| Ok(Box::new(self))),
        )
    }
}

impl eframe::App for ServerGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header
            ui.heading(egui::RichText::new("🖥️ Omniscan Hardware Server").size(20.0).strong());
            ui.label(egui::RichText::new("Control Panel & Device Manager").color(egui::Color32::GRAY));
            ui.separator();

            // Mode indicator
            if self.is_demo_mode {
                ui.label(
                    egui::RichText::new("⚠️ DEMO MODE - Simulation Active")
                        .color(egui::Color32::from_rgb(255, 165, 0))
                        .size(14.0)
                        .strong()
                );
            } else {
                ui.label(
                    egui::RichText::new("🛡️ PRODUCTION MODE - Real Hardware")
                        .color(egui::Color32::LIGHT_BLUE)
                        .size(14.0)
                        .strong()
                );
            }

            ui.add_space(10.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                // 3-Column Layout: Server | Devices | Controls
                egui::Grid::new("main_grid")
                .num_columns(3)
                .spacing([15.0, 10.0])
                .min_col_width(180.0)
                .striped(false)
                .show(ui, |ui| {
                    // === COLUMN 1: Server & Services ===
                    ui.vertical(|ui| {
                        // Server Status
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("📊 Server");
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Status:");
                                ui.colored_label(egui::Color32::GREEN, "✅ RUNNING");
                            });

                            ui.horizontal(|ui| {
                                ui.label("Uptime:");
                                let uptime = self.server_start_time.elapsed();
                                ui.label(format!("{:.0}s", uptime.as_secs()));
                            });

                            ui.horizontal(|ui| {
                                ui.label("gRPC:");
                                ui.label(&self.grpc_address);
                            });

                            ui.horizontal(|ui| {
                                ui.label("Config:");
                                ui.label(&self.config_file);
                            });
                        });

                        ui.add_space(10.0);

                        // Services Status
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("🔧 Services");
                            ui.add_space(5.0);

                            let service_row = |ui: &mut egui::Ui, name: &str, active: bool| {
                                ui.horizontal(|ui| {
                                    ui.label(name);
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if active {
                                            ui.label("✅");
                                        } else {
                                            ui.label("❌");
                                        }
                                    });
                                });
                            };

                            service_row(ui, "gRPC Server", true);
                            service_row(ui, "Audit Logger", true);
                            service_row(ui, "Safety SM", true);
                            service_row(ui, "Device Mgr", true);
                        });

                        ui.add_space(10.0);

                        // Storage settings
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("💾 Storage");
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Measurements dir:");
                            });
                            ui.add(egui::TextEdit::singleline(&mut self.measurement_dir_input).hint_text("e.g. C:/data/omniscan/measurements"));
                            ui.add_space(4.0);
                            if ui.button("Save storage settings").clicked() {
                                match crate::config::ServerConfig::load_from_file(&self.config_file) {
                                    Ok(mut cfg) => {
                                        cfg.measurement.output_dir = self.measurement_dir_input.clone();
                                        if let Err(e) = std::fs::create_dir_all(&cfg.measurement.output_dir) {
                                            self.storage_message = Some(format!("Failed to create dir: {}", e));
                                        } else if let Err(e) = cfg.save_to_file(&self.config_file) {
                                            self.storage_message = Some(format!("Failed to save config: {}", e));
                                        } else {
                                            self.storage_message = Some("Saved. New measurements will use this folder.".to_string());
                                        }
                                    }
                                    Err(e) => {
                                        self.storage_message = Some(format!("Failed to load config: {}", e));
                                    }
                                }
                            }
                            if let Some(msg) = &self.storage_message {
                                ui.label(egui::RichText::new(msg).small().color(egui::Color32::LIGHT_GREEN));
                            }
                            ui.label(egui::RichText::new("Note: Running acquisitions use the folder at start time of each run.").small().color(egui::Color32::GRAY));
                        });

                        ui.add_space(10.0);

                        // Orchestrator Connection
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("🔗 Orchestrator");
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Status:");
                                ui.colored_label(egui::Color32::GRAY, "⚫ N/A");
                            });
                            
                            ui.add_space(5.0);
                            ui.label(egui::RichText::new("ℹ️ Coming soon").small().color(egui::Color32::GRAY).italics());
                        });
                    });

// === COLUMN 2: Devices (PDU, GPIO, Detector, Motion) ===
                    ui.vertical(|ui| {

// PDU (Power Distribution Unit)
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("⚡ PDU");
                            ui.add_space(5.0);

                            // Read key switch and PDU power state
                            let (key_switch_on, pdu_on) = {
                                let ks = if let Ok(s) = self.gpio.state.try_read() { s.key_switch_on } else { false };
                                let po = self.pdu.is_powered_sync();
                                (ks, po)
                            };

                            ui.horizontal(|ui| {
                                ui.label("Power:");
                                if pdu_on {
                                    ui.colored_label(egui::Color32::GREEN, "✅ ON");
                                } else {
                                    ui.colored_label(egui::Color32::GRAY, "⚫ OFF");
                                }
                            });

                            ui.add_space(4.0);

                            // Controls
                            ui.horizontal(|ui| {
                                if !pdu_on {
let btn = egui::Button::new("🔌 Power ON");
                                    if !key_switch_on {
                                        // Disable if key switch is OFF
                                        ui.add_enabled_ui(false, |ui| { ui.add(egui::Button::new("🔌 Power ON")); });
                                        ui.colored_label(egui::Color32::RED, "🔒 Key switch must be ON");
                                    } else {
                                        if ui.add(btn).clicked() {
                                            // Turn on PDU and set GPIO power_ok = true
                                            self.pdu.set_powered_sync(true);
                                            let _ = self.gpio.set_power_ok_sync(true);
                                        }
                                    }
                                } else {
                                    if ui.button("⏻ Power OFF").clicked() {
                                        self.pdu.set_powered_sync(false);
                                        let _ = self.gpio.set_power_ok_sync(false);
                                    }
                                }
                            });

                            ui.add_space(4.0);
                            ui.label(egui::RichText::new("PDU controls overall machine power").small().color(egui::Color32::GRAY));
                        });

                        ui.add_space(10.0);

                        // GPIO Device
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("🔌 GPIO");
                            ui.add_space(5.0);

                            if let Ok(gpio_state) = self.gpio.state.try_read() {
                                ui.horizontal(|ui| {
                                    ui.label("Power:");
                                    if gpio_state.powered {
                                        ui.colored_label(egui::Color32::GREEN, "✅ ON");
                                    } else {
                                        ui.colored_label(egui::Color32::GRAY, "⚫ OFF");
                                    }
                                });
                                
                                if gpio_state.powered {
                                    ui.horizontal(|ui| {
                                        ui.label("Safety:");
                                        if gpio_state.emergency_stop_ok && gpio_state.door_closed_ok &&
                                           gpio_state.radiation_safe_input && gpio_state.cooling_ok && gpio_state.power_ok {
                                            ui.colored_label(egui::Color32::GREEN, "✅ SAFE");
                                        } else {
                                            ui.colored_label(egui::Color32::RED, "❌ FAULT");
                                        }
                                    });
                                    
                                    ui.horizontal(|ui| {
                                        ui.label("Key:");
                                        if gpio_state.key_switch_on {
                                            ui.colored_label(egui::Color32::GREEN, "✅ ON");
                                        } else {
                                            ui.colored_label(egui::Color32::GRAY, "⚫ OFF");
                                        }
                                    });
                                } else {
                                    ui.label(egui::RichText::new("⏳ Initializing...").small().color(egui::Color32::GRAY).italics());
                                }
                            }
                        });

                        ui.add_space(10.0);

                        // Detector
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("🔬 Detector");
                            ui.add_space(5.0);

                            if let Ok(det_state) = self.detector.state.try_read() {
                                ui.horizontal(|ui| {
                                    ui.label("Power:");
                                    if det_state.powered {
                                        ui.colored_label(egui::Color32::GREEN, "✅ ON");
                                    } else {
                                        ui.colored_label(egui::Color32::GRAY, "⚫ OFF");
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Temp:");
                                    let temp_color = if det_state.temperature > 50.0 {
                                        egui::Color32::RED
                                    } else if det_state.temperature > 40.0 {
                                        egui::Color32::from_rgb(255, 165, 0)
                                    } else {
                                        egui::Color32::GREEN
                                    };
                                    ui.colored_label(temp_color, format!("{:.1}°C", det_state.temperature));
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Exposures:");
                                    ui.label(format!("{}", det_state.total_exposures));
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Status:");
                                    let status_text = match &det_state.status {
                                        crate::devices::detectors::DetectorStatus::Off => "⚫ Off",
                                        crate::devices::detectors::DetectorStatus::Init => "🟡 Init",
                                        crate::devices::detectors::DetectorStatus::Idle => "🟢 Idle",
                                        crate::devices::detectors::DetectorStatus::Exposing => "🔵 Exp",
                                        crate::devices::detectors::DetectorStatus::Reading => "🟡 Read",
                                        crate::devices::detectors::DetectorStatus::Error(_) => "🔴 Err",
                                    };
                                    ui.label(status_text);
                                });
                            }
                            
                            ui.add_space(5.0);
                            ui.label(egui::RichText::new("ℹ️ gRPC controlled").small().color(egui::Color32::GRAY).italics());
                        });

                        ui.add_space(10.0);

                        // Motion
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("🎯 Motion");
                            ui.add_space(5.0);

                            if let Ok(motion_state) = self.motion.state.try_read() {
                                ui.horizontal(|ui| {
                                    ui.label("Power:");
                                    if motion_state.powered {
                                        ui.colored_label(egui::Color32::GREEN, "✅ ON");
                                    } else {
                                        ui.colored_label(egui::Color32::GRAY, "⚫ OFF");
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Homing:");
                                    if motion_state.is_homed {
                                        ui.colored_label(egui::Color32::GREEN, "✅ OK");
                                    } else {
                                        ui.colored_label(egui::Color32::from_rgb(255, 165, 0), "⚠️ NO");
                                    }
                                });
                                
                                if let Some(ref pos) = motion_state.position {
                                    ui.horizontal(|ui| {
                                        ui.label("Pos:");
                                        ui.label(format!("X:{:.1} Y:{:.1}", pos.x, pos.y));
                                    });
                                }
                                
                                ui.horizontal(|ui| {
                                    ui.label("Moves:");
                                    ui.label(format!("{}", motion_state.total_moves));
                                });
                            }
                            
                            ui.add_space(5.0);
                            ui.label(egui::RichText::new("ℹ️ gRPC controlled").small().color(egui::Color32::GRAY).italics());
                        });
                    });

                    // === COLUMN 3: Controls ===
                    ui.vertical(|ui| {

                        // GPIO Control Panel Launcher
                        ui.group(|ui| {
                            ui.set_min_width(180.0);
                            ui.heading("🔧 GPIO Panel");
                            ui.add_space(5.0);

                            if ui.button("🔌 Toggle GPIO Control Panel").clicked() {
                                self.show_gpio_panel = !self.show_gpio_panel;
                            }

                            ui.add_space(5.0);
                            if self.is_demo_mode {
                                ui.label(egui::RichText::new("ℹ️ GPIO simulator").italics().small().color(egui::Color32::GRAY));
                            } else {
                                ui.label(egui::RichText::new("ℹ️ GPIO monitor").italics().small().color(egui::Color32::GRAY));
                            }
                        });

                        ui.add_space(10.0);

                        // System Controls
                        if self.is_demo_mode {
                            ui.group(|ui| {
                                ui.set_min_width(180.0);
                                ui.heading("⚙️ System");
                                ui.add_space(5.0);

                                if ui.button("🔴 E-Stop").clicked() {
                                    if let Ok(mut gpio_state) = self.gpio.state.try_write() {
                                        gpio_state.emergency_stop_ok = false;  // pressed = unsafe
                                    }
                                }

                                if ui.button("✅ Reset Interlocks").clicked() {
                                    if let Ok(mut gpio_state) = self.gpio.state.try_write() {
                                        gpio_state.emergency_stop_ok = true;   // E-stop released (safe)
                                        gpio_state.door_closed_ok = true;
                                        gpio_state.radiation_safe_input = true;
                                        gpio_state.beam_stop_output = true;
                                        gpio_state.cooling_ok = true;
                                        gpio_state.power_ok = true;
                                    }
                                }

                                ui.add_space(5.0);
                                ui.label(egui::RichText::new("⚠️ DEMO only").italics().small().color(egui::Color32::from_rgb(255, 165, 0)));
                            });
                        }
                    });
                });

                ui.add_space(10.0);
            });

            // Footer
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Omniscan Hardware Server v0.2.0").small().color(egui::Color32::GRAY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new("FDA/IEC 62304 Class B").small().color(egui::Color32::GRAY));
                });
            });

            ui.label(egui::RichText::new("ℹ️ Server continues running when this window is closed").italics().small().color(egui::Color32::GRAY));
        });

        // Embedded GPIO Control Panel (DEV only)
        if self.is_demo_mode && self.show_gpio_panel {
            let (key_switch_on, emergency_stop, door_closed, rad_safe, cooling_ok, power_ok, activation_active, activation_remaining, beam_stop_output) = {
                if let Ok(state) = self.gpio.state.try_read() {
                    let remaining = self.gpio.get_activation_button_remaining_time_sync();
                    (state.key_switch_on, state.emergency_stop_ok, state.door_closed_ok, state.radiation_safe_input, 
                     state.cooling_ok, state.power_ok, state.activation_button_active, remaining, state.beam_stop_output)
                } else {
                    (false, true, true, true, true, true, false, None, true)
                }
            };
            egui::Window::new("🔧 GPIO Control Panel (DEV)")
                .open(&mut self.show_gpio_panel)
                .default_width(650.0)
                .default_height(500.0)
                .show(ctx, |ui| {
                    ui.label(egui::RichText::new("GPIO PCIe Card Simulator").strong().color(egui::Color32::LIGHT_BLUE));
                    ui.label(egui::RichText::new("Left: Pin readings | Right: Sensor simulators").small().color(egui::Color32::GRAY).italics());
                    ui.separator();
                    
                    // 2-Column Layout
                    egui::Grid::new("gpio_dev_grid")
                        .num_columns(2)
                        .spacing([20.0, 10.0])
                        .min_col_width(280.0)
                        .show(ui, |ui| {
                            // === LEFT COLUMN: GPIO INPUT READINGS ===
                            ui.vertical(|ui| {
                                ui.heading(egui::RichText::new("📥 GPIO INPUTS").color(egui::Color32::LIGHT_BLUE));
                                ui.label(egui::RichText::new("Pin readings (read-only)").small().italics().color(egui::Color32::GRAY));
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
                                
                                input_row(ui, 1, "E-Stop Released", emergency_stop);
                                input_row(ui, 2, "Door Closed", door_closed);
                                input_row(ui, 3, "Radiation Safe", rad_safe);
                                input_row(ui, 4, "Cooling OK", cooling_ok);
                                input_row(ui, 5, "Power OK", power_ok);
                                input_row(ui, 7, "Key Switch ON", key_switch_on);
                                
                                ui.add_space(10.0);
                                ui.separator();
                                
                                // Output pin
                                ui.label(egui::RichText::new("📤 GPIO OUTPUT").color(egui::Color32::from_rgb(255, 165, 0)));
                                ui.horizontal(|ui| {
                                    ui.label("Pin 6: Beam-Stop Cmd");
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if beam_stop_output {
                                            ui.colored_label(egui::Color32::GREEN, "🔒 CLOSE");
                                        } else {
                                            ui.colored_label(egui::Color32::RED, "🔓 OPEN");
                                        }
                                    });
                                });
                                
                                ui.add_space(10.0);
                                ui.separator();
                                
                                // Overall status
                                let overall_safe = emergency_stop && door_closed && rad_safe && cooling_ok && power_ok;
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("System Status:").strong());
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if overall_safe {
                                            ui.colored_label(egui::Color32::GREEN, egui::RichText::new("✅ SAFE").strong());
                                        } else {
                                            ui.colored_label(egui::Color32::RED, egui::RichText::new("❌ UNSAFE").strong());
                                        }
                                    });
                                });
                            });
                            
                            // === RIGHT COLUMN: SENSOR SIMULATORS ===
                            ui.vertical(|ui| {
                                ui.heading(egui::RichText::new("🎮 SENSOR SIMULATORS").color(egui::Color32::from_rgb(255, 165, 0)));
                                ui.label(egui::RichText::new("Emulate physical sensors").small().italics().color(egui::Color32::GRAY));
                                ui.add_space(5.0);
                                
                                ui.label(egui::RichText::new("Key Switch:").strong());
                                ui.horizontal(|ui| {
                                    if ui.button("🔒 Turn OFF").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(7, crate::devices::gpio::GpioPinState::Low);
                                        }
                                    }
                                    if ui.button("🔓 Turn ON").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(7, crate::devices::gpio::GpioPinState::High);
                                        }
                                    }
                                });
                                
                                ui.add_space(5.0);
                                ui.label(egui::RichText::new("Emergency Stop:").strong());
                                ui.horizontal(|ui| {
                                    if ui.button("❌ Press E-Stop").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(1, crate::devices::gpio::GpioPinState::Low);
                                        }
                                    }
                                    if ui.button("✅ Release E-Stop").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(1, crate::devices::gpio::GpioPinState::High);
                                        }
                                    }
                                });
                                
                                ui.add_space(5.0);
                                ui.label(egui::RichText::new("Door Sensor:").strong());
                                ui.horizontal(|ui| {
                                    if ui.button("🚪 Open Door").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(2, crate::devices::gpio::GpioPinState::Low);
                                        }
                                    }
                                    if ui.button("🚪 Close Door").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(2, crate::devices::gpio::GpioPinState::High);
                                        }
                                    }
                                });
                                
                                ui.add_space(5.0);
                                ui.label(egui::RichText::new("Beam-Stop Sensor:").strong());
                                ui.label(egui::RichText::new("(Requires activation button)").small().italics().color(egui::Color32::GRAY));
                                ui.horizontal(|ui| {
                                    // Remove beam-stop only allowed if activation button is active
                                    ui.add_enabled_ui(activation_active, |ui| {
                                        if ui.button("🔴 Remove Beam-Stop").clicked() {
                                            if let Ok(mut state) = self.gpio.state.try_write() {
                                                state.simulated_pin_states.insert(3, crate::devices::gpio::GpioPinState::Low);
                                            }
                                        }
                                    });
                                    if !activation_active {
                                        ui.label(egui::RichText::new("🔒").color(egui::Color32::GRAY));
                                    }
                                    
                                    // Insert beam-stop always allowed (safe operation)
                                    if ui.button("🟢 Insert Beam-Stop").clicked() {
                                        if let Ok(mut state) = self.gpio.state.try_write() {
                                            state.simulated_pin_states.insert(3, crate::devices::gpio::GpioPinState::High);
                                        }
                                    }
                                });
                                
                                ui.add_space(10.0);
                                ui.separator();
                                
                                if ui.button("✅ ARM ALL SENSORS (Safe State)").clicked() {
                                    let _ = self.gpio.arm_all_interlocks_sync();
                                }
                                
                                ui.add_space(10.0);
                                ui.separator();
                                
                                ui.label(egui::RichText::new("⚡ Activation Button:").strong());
                                if activation_active {
                                    ui.colored_label(egui::Color32::GREEN, 
                                        format!("✅ ACTIVE ({} sec)", activation_remaining.unwrap_or(0)));
                                } else {
                                    ui.colored_label(egui::Color32::RED, "🔴 INACTIVE");
                                }
                                if ui.button("🔴 ACTIVATE (20 sec)").clicked() {
                                    let _ = self.gpio.activate_enable_button_sync();
                                }
                            });
                        });
                });
        }
    }
}
