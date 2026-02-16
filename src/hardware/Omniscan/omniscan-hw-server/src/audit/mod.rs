use anyhow::Result;
use serde::{Deserialize, Serialize};
use sqlx::{SqlitePool, Row};
use tracing::info;
use uuid::Uuid;

/// Audit logging system for medical device compliance
pub struct AuditLogger {
    pool: SqlitePool,
    session_id: String,
    device_certificate_number: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandLog {
    pub id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub command_type: String,
    pub command_data: serde_json::Value,
    pub user_context: Option<String>,
    pub device_state_before: serde_json::Value,
    pub device_state_after: Option<serde_json::Value>,
    pub result: CommandResult,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandResult {
    Success,
    Failed { error: String },
    Aborted { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DeviceStateLog {
    pub id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub device_type: String, // "detector", "motion", "gpio"
    pub device_id: String,
    pub state_data: serde_json::Value,
    pub health_data: serde_json::Value,
    pub alert_level: AlertLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum AlertLevel {
    Normal,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MeasurementRecord {
    pub id: String,
    pub session_id: String,
    pub measurement_id: String,
    pub patient_id: Option<String>,
    pub study_id: Option<String>,
    pub timestamp_start: chrono::DateTime<chrono::Utc>,
    pub timestamp_end: Option<chrono::DateTime<chrono::Utc>>,
    pub exposure_time_ms: u32,
    pub detector_config: serde_json::Value,
    pub motion_position: serde_json::Value,
    pub data_files: Vec<String>,
    pub data_size_bytes: u64,
    pub checksum: Option<String>,
    pub operator: Option<String>,
    pub notes: Option<String>,
    pub status: MeasurementStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum MeasurementStatus {
    Started,
    InProgress,
    Completed,
    Aborted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct InterlockEvent {
    pub id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub interlock_type: String,
    pub previous_state: bool,
    pub new_state: bool,
    pub trigger_source: String,
    pub system_response: String,
}

/// Authentication event for audit trail (USER_EXPECTATIONS.md Section 8)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct AuthenticationEvent {
    pub id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: AuthEventType,
    pub username: String,
    pub ip_address: Option<String>,
    pub success: bool,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum AuthEventType {
    Login,
    Logout,
    FailedLogin,
    SessionExpired,
    PermissionDenied,
}

/// Configuration change event (USER_EXPECTATIONS.md Section 8)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ConfigurationChangeEvent {
    pub id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user: String,
    pub parameter_name: String,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
    pub approved_by: Option<String>,
}

/// Software update event (USER_EXPECTATIONS.md Section 8)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SoftwareUpdateEvent {
    pub id: String,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub from_version: String,
    pub to_version: String,
    pub update_type: UpdateType,
    pub performed_by: String,
    pub success: bool,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum UpdateType {
    SecurityPatch,
    BugFix,
    FeatureUpdate,
    MajorRelease,
}

impl AuditLogger {
    /// Create new audit logger with SQLite database
    pub async fn new(database_path: &str, device_certificate_number: String) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(database_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Use file:// URI with proper escaping for Windows paths
        let path = std::path::Path::new(database_path);
        let abs_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        
        // Convert Windows path to URL format
        let path_str = abs_path.to_string_lossy().replace("\\", "/");
        // Add ?mode=rwc to create the database if it doesn't exist
        let database_url = format!("sqlite:///{}?mode=rwc", path_str);
        
        info!("Connecting to audit database: {}", database_url);
        let pool = SqlitePool::connect(&database_url).await?;
        
        let session_id = Uuid::new_v4().to_string();
        
        let logger = Self {
            pool,
            session_id,
            device_certificate_number,
        };
        
        logger.initialize_database().await?;
        logger.log_session_start().await?;
        
        Ok(logger)
    }

    /// Initialize database schema
    async fn initialize_database(&self) -> Result<()> {
        // Commands table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS command_logs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                command_type TEXT NOT NULL,
                command_data TEXT NOT NULL,
                user_context TEXT,
                device_state_before TEXT NOT NULL,
                device_state_after TEXT,
                result TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                device_certificate_number TEXT
            )
        "#).execute(&self.pool).await?;

        // Device states table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS device_state_logs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                device_type TEXT NOT NULL,
                device_id TEXT NOT NULL,
                state_data TEXT NOT NULL,
                health_data TEXT NOT NULL,
                alert_level TEXT NOT NULL
            )
        "#).execute(&self.pool).await?;

        // Measurements table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS measurement_records (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                measurement_id TEXT NOT NULL,
                patient_id TEXT,
                study_id TEXT,
                timestamp_start TEXT NOT NULL,
                timestamp_end TEXT,
                exposure_time_ms INTEGER NOT NULL,
                detector_config TEXT NOT NULL,
                motion_position TEXT NOT NULL,
                data_files TEXT NOT NULL,
                data_size_bytes INTEGER NOT NULL,
                checksum TEXT,
                operator TEXT,
                notes TEXT,
                status TEXT NOT NULL,
                device_certificate_number TEXT
            )
        "#).execute(&self.pool).await?;

        // Calibration records table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS calibration_records (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                operator TEXT,
                calibrant_material TEXT NOT NULL,
                exposure_time_ms INTEGER,
                device_serial TEXT,
                status TEXT NOT NULL,
                qc_results_json TEXT NOT NULL,
                poni_file_content TEXT
            )
        "#).execute(&self.pool).await?;

        // Interlock events table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS interlock_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                interlock_type TEXT NOT NULL,
                previous_state INTEGER NOT NULL,
                new_state INTEGER NOT NULL,
                trigger_source TEXT NOT NULL,
                system_response TEXT NOT NULL
            )
        "#).execute(&self.pool).await?;

        // Sessions table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                server_version TEXT NOT NULL,
                config_version INTEGER NOT NULL,
                device_certificate_number TEXT NOT NULL,
                operator TEXT,
                notes TEXT
            )
        "#).execute(&self.pool).await?;

        // Migrations: Add device_certificate_number column if it doesn't exist
        // SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we check first
        
        // Migrate command_logs table
        let column_exists = sqlx::query(
            "SELECT COUNT(*) as count FROM pragma_table_info('command_logs') WHERE name='device_certificate_number'"
        )
        .fetch_one(&self.pool)
        .await?
        .get::<i64, _>("count") > 0;

        if !column_exists {
            info!("Migrating command_logs table: adding device_certificate_number column");
            sqlx::query(
                "ALTER TABLE command_logs ADD COLUMN device_certificate_number TEXT"
            )
            .execute(&self.pool)
            .await?;
        }

        // Migrate measurement_records table
        let column_exists = sqlx::query(
            "SELECT COUNT(*) as count FROM pragma_table_info('measurement_records') WHERE name='device_certificate_number'"
        )
        .fetch_one(&self.pool)
        .await?
        .get::<i64, _>("count") > 0;

        if !column_exists {
            info!("Migrating measurement_records table: adding device_certificate_number column");
            sqlx::query(
                "ALTER TABLE measurement_records ADD COLUMN device_certificate_number TEXT"
            )
            .execute(&self.pool)
            .await?;
        }

        // Migrate sessions table
        let column_exists = sqlx::query(
            "SELECT COUNT(*) as count FROM pragma_table_info('sessions') WHERE name='device_certificate_number'"
        )
        .fetch_one(&self.pool)
        .await?
        .get::<i64, _>("count") > 0;

        if !column_exists {
            info!("Migrating sessions table: adding device_certificate_number column");
            sqlx::query(
                "ALTER TABLE sessions ADD COLUMN device_certificate_number TEXT NOT NULL DEFAULT 'UNKNOWN'"
            )
            .execute(&self.pool)
            .await?;
        }

        info!("Audit database initialized");
        Ok(())
    }

    /// Log session start
    async fn log_session_start(&self) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO sessions (id, start_time, server_version, config_version, device_certificate_number)
            VALUES (?, ?, ?, ?, ?)
        "#)
        .bind(&self.session_id)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(env!("CARGO_PKG_VERSION"))
        .bind(1) // Config version
        .bind(&self.device_certificate_number)
        .execute(&self.pool).await?;

        info!("Audit session started: {}", self.session_id);
        Ok(())
    }

    /// Log command execution
    pub async fn log_command(&self, log: CommandLog) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO command_logs (
                id, session_id, timestamp, command_type, command_data,
                user_context, device_state_before, device_state_after,
                result, execution_time_ms, device_certificate_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&log.id)
        .bind(&log.session_id)
        .bind(log.timestamp.to_rfc3339())
        .bind(&log.command_type)
        .bind(serde_json::to_string(&log.command_data)?)
        .bind(&log.user_context)
        .bind(serde_json::to_string(&log.device_state_before)?)
        .bind(log.device_state_after.map(|d| serde_json::to_string(&d).unwrap_or_default()))
        .bind(serde_json::to_string(&log.result)?)
        .bind(log.execution_time_ms as i64)
        .bind(&self.device_certificate_number)
        .execute(&self.pool).await?;

        Ok(())
    }

    /// Log device state change
    #[allow(dead_code)]
    pub async fn log_device_state(&self, log: DeviceStateLog) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO device_state_logs (
                id, session_id, timestamp, device_type, device_id,
                state_data, health_data, alert_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&log.id)
        .bind(&log.session_id)
        .bind(log.timestamp.to_rfc3339())
        .bind(&log.device_type)
        .bind(&log.device_id)
        .bind(serde_json::to_string(&log.state_data)?)
        .bind(serde_json::to_string(&log.health_data)?)
        .bind(serde_json::to_string(&log.alert_level)?)
        .execute(&self.pool).await?;

        Ok(())
    }

    /// Log measurement record
    #[allow(dead_code)]
    pub async fn log_measurement(&self, record: MeasurementRecord) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO measurement_records (
                id, session_id, measurement_id, patient_id, study_id,
                timestamp_start, timestamp_end, exposure_time_ms,
                detector_config, motion_position, data_files,
                data_size_bytes, checksum, operator, notes, status, device_certificate_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&record.id)
        .bind(&record.session_id)
        .bind(&record.measurement_id)
        .bind(&record.patient_id)
        .bind(&record.study_id)
        .bind(record.timestamp_start.to_rfc3339())
        .bind(record.timestamp_end.map(|t| t.to_rfc3339()))
        .bind(record.exposure_time_ms as i64)
        .bind(serde_json::to_string(&record.detector_config)?)
        .bind(serde_json::to_string(&record.motion_position)?)
        .bind(serde_json::to_string(&record.data_files)?)
        .bind(record.data_size_bytes as i64)
        .bind(&record.checksum)
        .bind(&record.operator)
        .bind(&record.notes)
        .bind(serde_json::to_string(&record.status)?)
        .bind(&self.device_certificate_number)
        .execute(&self.pool).await?;

        Ok(())
    }

    /// Log interlock event
    #[allow(dead_code)]
    pub async fn log_interlock_event(&self, event: InterlockEvent) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO interlock_events (
                id, session_id, timestamp, interlock_type,
                previous_state, new_state, trigger_source, system_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&event.id)
        .bind(&event.session_id)
        .bind(event.timestamp.to_rfc3339())
        .bind(&event.interlock_type)
        .bind(event.previous_state)
        .bind(event.new_state)
        .bind(&event.trigger_source)
        .bind(&event.system_response)
        .execute(&self.pool).await?;

        Ok(())
    }

    /// Log calibration record (including QC results)
    #[allow(dead_code)]
    pub async fn log_calibration_record(
        &self,
        id: String,
        timestamp: chrono::DateTime<chrono::Utc>,
        operator: Option<String>,
        calibrant_material: String,
        exposure_time_ms: Option<u32>,
        device_serial: Option<String>,
        status: String,
        qc_results_json: String,
        poni_file_content: Option<String>,
    ) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO calibration_records (
                id, session_id, timestamp, operator, calibrant_material,
                exposure_time_ms, device_serial, status, qc_results_json, poni_file_content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&id)
        .bind(&self.session_id)
        .bind(timestamp.to_rfc3339())
        .bind(operator)
        .bind(calibrant_material)
        .bind(exposure_time_ms.map(|v| v as i64))
        .bind(device_serial)
        .bind(status)
        .bind(qc_results_json)
        .bind(poni_file_content)
        .execute(&self.pool).await?;
        Ok(())
    }
    
    /// Log authentication event
    #[allow(dead_code)]
    pub async fn log_authentication_event(&self, event: AuthenticationEvent) -> Result<()> {
        // TODO: Create authentication_events table in initialize_database
        // For now, log as command
        let command_log = CommandLog {
            id: event.id,
            session_id: event.session_id,
            timestamp: event.timestamp,
            command_type: format!("auth:{:?}", event.event_type),
            command_data: serde_json::json!({
                "username": event.username,
                "ip_address": event.ip_address,
                "success": event.success,
                "failure_reason": event.failure_reason,
            }),
            user_context: Some(event.username),
            device_state_before: serde_json::json!({}),
            device_state_after: None,
            result: if event.success { CommandResult::Success } else { 
                CommandResult::Failed { error: event.failure_reason.unwrap_or_default() }
            },
            execution_time_ms: 0,
        };
        
        self.log_command(command_log).await
    }
    
    /// Log configuration change event
    #[allow(dead_code)]
    pub async fn log_configuration_change(&self, event: ConfigurationChangeEvent) -> Result<()> {
        // TODO: Create configuration_changes table in initialize_database
        // For now, log as command
        let command_log = CommandLog {
            id: event.id,
            session_id: event.session_id,
            timestamp: event.timestamp,
            command_type: "config_change".to_string(),
            command_data: serde_json::json!({
                "parameter": event.parameter_name,
                "old_value": event.old_value,
                "new_value": event.new_value,
                "approved_by": event.approved_by,
            }),
            user_context: Some(event.user),
            device_state_before: serde_json::json!({"value": event.old_value}),
            device_state_after: Some(serde_json::json!({"value": event.new_value})),
            result: CommandResult::Success,
            execution_time_ms: 0,
        };
        
        self.log_command(command_log).await
    }
    
    /// Log software update event
    #[allow(dead_code)]
    pub async fn log_software_update(&self, event: SoftwareUpdateEvent) -> Result<()> {
        // TODO: Create software_updates table in initialize_database
        // For now, log as command
        let command_log = CommandLog {
            id: event.id,
            session_id: event.session_id,
            timestamp: event.timestamp,
            command_type: format!("software_update:{:?}", event.update_type),
            command_data: serde_json::json!({
                "from_version": event.from_version,
                "to_version": event.to_version,
                "notes": event.notes,
            }),
            user_context: Some(event.performed_by),
            device_state_before: serde_json::json!({"version": event.from_version}),
            device_state_after: Some(serde_json::json!({"version": event.to_version})),
            result: if event.success { CommandResult::Success } else {
                CommandResult::Failed { error: "Update failed".to_string() }
            },
            execution_time_ms: 0,
        };
        
        self.log_command(command_log).await
    }

    /// Get recent commands
    #[allow(dead_code)]
    pub async fn get_recent_commands(&self, limit: i64) -> Result<Vec<CommandLog>> {
        let rows = sqlx::query(r#"
            SELECT * FROM command_logs
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        "#)
        .bind(&self.session_id)
        .bind(limit)
        .fetch_all(&self.pool).await?;

        let mut commands = Vec::new();
        for row in rows {
            let command = CommandLog {
                id: row.get("id"),
                session_id: row.get("session_id"),
                timestamp: chrono::DateTime::parse_from_rfc3339(&row.get::<String, _>("timestamp"))?
                    .with_timezone(&chrono::Utc),
                command_type: row.get("command_type"),
                command_data: serde_json::from_str(&row.get::<String, _>("command_data"))?,
                user_context: row.get("user_context"),
                device_state_before: serde_json::from_str(&row.get::<String, _>("device_state_before"))?,
                device_state_after: row.get::<Option<String>, _>("device_state_after")
                    .map(|s| serde_json::from_str(&s).ok()).flatten(),
                result: serde_json::from_str(&row.get::<String, _>("result"))?,
                execution_time_ms: row.get::<i64, _>("execution_time_ms") as u64,
            };
            commands.push(command);
        }

        Ok(commands)
    }

    /// Get measurements by patient ID
    #[allow(dead_code)]
    pub async fn get_measurements_by_patient(&self, patient_id: &str) -> Result<Vec<MeasurementRecord>> {
        let rows = sqlx::query(r#"
            SELECT * FROM measurement_records
            WHERE patient_id = ?
            ORDER BY timestamp_start DESC
        "#)
        .bind(patient_id)
        .fetch_all(&self.pool).await?;

        let mut measurements = Vec::new();
        for row in rows {
            let measurement = MeasurementRecord {
                id: row.get("id"),
                session_id: row.get("session_id"),
                measurement_id: row.get("measurement_id"),
                patient_id: row.get("patient_id"),
                study_id: row.get("study_id"),
                timestamp_start: chrono::DateTime::parse_from_rfc3339(&row.get::<String, _>("timestamp_start"))?
                    .with_timezone(&chrono::Utc),
                timestamp_end: row.get::<Option<String>, _>("timestamp_end")
                    .map(|s| chrono::DateTime::parse_from_rfc3339(&s).ok()).flatten()
                    .map(|dt| dt.with_timezone(&chrono::Utc)),
                exposure_time_ms: row.get::<i64, _>("exposure_time_ms") as u32,
                detector_config: serde_json::from_str(&row.get::<String, _>("detector_config"))?,
                motion_position: serde_json::from_str(&row.get::<String, _>("motion_position"))?,
                data_files: serde_json::from_str(&row.get::<String, _>("data_files"))?,
                data_size_bytes: row.get::<i64, _>("data_size_bytes") as u64,
                checksum: row.get("checksum"),
                operator: row.get("operator"),
                notes: row.get("notes"),
                status: serde_json::from_str(&row.get::<String, _>("status"))?,
            };
            measurements.push(measurement);
        }

        Ok(measurements)
    }

    /// Close audit session
    #[allow(dead_code)]
    pub async fn close_session(&self, notes: Option<String>) -> Result<()> {
        sqlx::query(r#"
            UPDATE sessions 
            SET end_time = ?, notes = ?
            WHERE id = ?
        "#)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(notes)
        .bind(&self.session_id)
        .execute(&self.pool).await?;

        info!("Audit session closed: {}", self.session_id);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

/// Helper for creating command logs
impl CommandLog {
    #[allow(dead_code)]
    pub fn new(
        command_type: &str,
        command_data: serde_json::Value,
        user_context: Option<String>,
        device_state_before: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: String::new(), // Will be set by logger
            timestamp: chrono::Utc::now(),
            command_type: command_type.to_string(),
            command_data,
            user_context,
            device_state_before,
            device_state_after: None,
            result: CommandResult::Success,
            execution_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_audit_logger_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let logger = AuditLogger::new(
            db_path.to_str().unwrap(),
            "TEST-CERT".to_string(),
        )
        .await
        .unwrap();
        
        // Test command logging
        let mut command_log = CommandLog::new(
            "start_exposure",
            serde_json::json!({"exposure_time": 1000}),
            Some("test_user".to_string()),
            serde_json::json!({"detector": "idle"}),
        );
        command_log.session_id = logger.session_id().to_string();
        
        logger.log_command(command_log).await.unwrap();
        
        // Test measurement logging
        let measurement = MeasurementRecord {
            id: Uuid::new_v4().to_string(),
            session_id: logger.session_id().to_string(),
            measurement_id: "M001".to_string(),
            patient_id: Some("P001".to_string()),
            study_id: Some("S001".to_string()),
            timestamp_start: chrono::Utc::now(),
            timestamp_end: None,
            exposure_time_ms: 1000,
            detector_config: serde_json::json!({"type": "demo"}),
            motion_position: serde_json::json!({"x": 10.0, "y": 20.0}),
            data_files: vec!["data1.raw".to_string()],
            data_size_bytes: 1024,
            checksum: Some("abc123".to_string()),
            operator: Some("operator1".to_string()),
            notes: None,
            status: MeasurementStatus::Started,
        };
        
        logger.log_measurement(measurement).await.unwrap();
        
        // Test retrieval
        let commands = logger.get_recent_commands(10).await.unwrap();
        assert_eq!(commands.len(), 1);
        
        let measurements = logger.get_measurements_by_patient("P001").await.unwrap();
        assert_eq!(measurements.len(), 1);
    }
}
