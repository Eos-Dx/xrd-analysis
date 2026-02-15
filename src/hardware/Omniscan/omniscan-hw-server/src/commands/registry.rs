use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context};
use tracing::{info, warn, error};

use super::schema::CommandSchema;

/// Registry of all available equipment commands
pub struct CommandRegistry {
    commands: HashMap<String, CommandSchema>,
}

impl CommandRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
        }
    }
    
    /// Load all command schemas from a directory
    pub fn load_from_directory<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut registry = Self::new();
        let dir_path = path.as_ref();
        
        info!("Loading command schemas from: {}", dir_path.display());
        
        if !dir_path.exists() {
            warn!("Command schema directory does not exist: {}", dir_path.display());
            return Ok(registry);
        }
        
        // Read all .toml files in the directory
        for entry in std::fs::read_dir(dir_path)
            .context("Failed to read command schema directory")?
        {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                match Self::load_schema_file(&path) {
                    Ok(schema) => {
                        info!("Loaded command schema: {} from {}", schema.command_id, path.display());
                        registry.register(schema)?;
                    }
                    Err(e) => {
                        error!("Failed to load schema from {}: {}", path.display(), e);
                        // Continue loading other files
                    }
                }
            }
        }
        
        info!("Loaded {} command schemas", registry.commands.len());
        Ok(registry)
    }
    
    /// Load a single command schema from a TOML file
    fn load_schema_file<P: AsRef<Path>>(path: P) -> Result<CommandSchema> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read schema file")?;
        
        let schema: CommandSchema = toml::from_str(&content)
            .context("Failed to parse TOML schema")?;
        
        // Validate the schema
        schema.validate()
            .context("Schema validation failed")?;
        
        Ok(schema)
    }
    
    /// Register a command schema
    pub fn register(&mut self, schema: CommandSchema) -> Result<()> {
        let command_id = schema.command_id.clone();
        
        if self.commands.contains_key(&command_id) {
            warn!("Overwriting existing command schema: {}", command_id);
        }
        
        self.commands.insert(command_id, schema);
        Ok(())
    }
    
    /// Get a command schema by ID
    pub fn get(&self, command_id: &str) -> Option<&CommandSchema> {
        self.commands.get(command_id)
    }
    
    /// Get all registered command IDs
    pub fn list_commands(&self) -> Vec<String> {
        self.commands.keys().cloned().collect()
    }
    
    /// Get commands by service
    pub fn get_by_service(&self, service: &str) -> Vec<&CommandSchema> {
        self.commands
            .values()
            .filter(|schema| schema.service == service)
            .collect()
    }
    
    /// Generate documentation for all commands
    pub fn generate_documentation(&self) -> String {
        let mut doc = String::from("# Equipment Command Reference\n\n");
        
        // Group by service
        let mut services: HashMap<String, Vec<&CommandSchema>> = HashMap::new();
        for schema in self.commands.values() {
            services.entry(schema.service.clone())
                .or_insert_with(Vec::new)
                .push(schema);
        }
        
        for (service, commands) in services.iter() {
            doc.push_str(&format!("## {} Service\n\n", service));
            
            for cmd in commands {
                doc.push_str(&format!("### `{}`\n\n", cmd.command_id));
                doc.push_str(&format!("{}\n\n", cmd.description));
                
                doc.push_str("**Safety Requirements:**\n");
                doc.push_str(&format!("- Allowed States: {:?}\n", cmd.safety_requirements.allowed_states));
                doc.push_str(&format!("- Requires Enable Button: {}\n", cmd.safety_requirements.requires_enable_button));
                doc.push_str(&format!("- Requires Calibration: {}\n", cmd.safety_requirements.requires_calibration));
                doc.push_str(&format!("- Requires Key Switch: {}\n", cmd.safety_requirements.requires_key_switch));
                doc.push_str(&format!("- Risk Level: {:?}\n", cmd.risk_level));
                doc.push_str("\n");
                
                if !cmd.parameters.is_empty() {
                    doc.push_str("**Parameters:**\n");
                    for param in &cmd.parameters {
                        doc.push_str(&format!("- `{}`: {} ({})\n", 
                            param.name, 
                            param.description,
                            if param.required { "required" } else { "optional" }
                        ));
                    }
                    doc.push_str("\n");
                }
            }
        }
        
        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::schema::*;
    
    #[test]
    fn test_registry_creation() {
        let registry = CommandRegistry::new();
        assert_eq!(registry.commands.len(), 0);
    }
    
    #[test]
    fn test_register_command() {
        let mut registry = CommandRegistry::new();
        
        let schema = CommandSchema {
            command_id: "test_command".to_string(),
            description: "Test command".to_string(),
            service: "Test".to_string(),
            safety_requirements: SafetyRequirements {
                allowed_states: vec!["Idle".to_string()],
                requires_enable_button: false,
                requires_calibration: false,
                requires_key_switch: true,
                interlock_checks: vec![],
                enable_timeout_seconds: None,
                max_execution_time_seconds: None,
            },
            parameters: vec![],
            execution: ExecutionSchema {
                steps: vec![],
                rollback_on_failure: false,
                error_cleanup: vec![],
            },
            audit_level: AuditLevel::Standard,
            risk_level: RiskLevel::Low,
        };
        
        registry.register(schema).unwrap();
        assert_eq!(registry.commands.len(), 1);
        assert!(registry.get("test_command").is_some());
    }
}
