use omniscan_hw_server::devices::detectors::{
    DemoDetector, DetectorDevice, DetectorError, DetectorStatus, TestDetector,
};

#[tokio::test]
async fn test_detector_power_on_off() {
    let detector = TestDetector::new();

    // Initially powered off
    assert!(!detector.is_powered().await);
    assert_eq!(detector.get_status().await, DetectorStatus::Off);

    // Power on
    detector.power_on().await.unwrap();
    assert!(detector.is_powered().await);
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);

    // Power off
    detector.power_off().await.unwrap();
    assert!(!detector.is_powered().await);
    assert_eq!(detector.get_status().await, DetectorStatus::Off);
}

#[tokio::test]
async fn test_detector_exposure_requires_power() {
    let detector = TestDetector::new();

    // Should fail when not powered
    let result = detector.start_exposure(1000).await;
    assert!(matches!(result, Err(DetectorError::NotPowered)));

    // Should succeed when powered
    detector.power_on().await.unwrap();
    detector.start_exposure(1000).await.unwrap();
}

#[tokio::test]
async fn test_detector_invalid_exposure_time() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Zero exposure time
    let result = detector.start_exposure(0).await;
    assert!(matches!(result, Err(DetectorError::InvalidExposureTime(0))));

    // Too long exposure time (> 5 minutes)
    let result = detector.start_exposure(301_000).await;
    assert!(matches!(
        result,
        Err(DetectorError::InvalidExposureTime(301_000))
    ));
}

#[tokio::test]
async fn test_detector_health_info() {
    let detector = TestDetector::new();

    let health = detector.get_health().await;
    assert!(!health.powered);
    assert_eq!(health.voltage, 0.0);

    detector.power_on().await.unwrap();

    let health = detector.get_health().await;
    assert!(health.powered);
    assert_eq!(health.voltage, 12.0);
    assert_eq!(health.temperature, 23.0);
}

#[tokio::test]
async fn test_detector_exposure_result() {
    let detector = TestDetector::new();
    detector.power_on().await.unwrap();

    // No result initially
    assert!(detector.get_last_result().await.is_none());

    // Start exposure
    detector.start_exposure(1000).await.unwrap();

    // Should have result now (TestDetector completes instantly)
    let result = detector.get_last_result().await;
    assert!(result.is_some());

    let result = result.unwrap();
    assert_eq!(result.exposure_time_ms, 1000);
    assert!(result.data_path.is_some());
}

#[tokio::test]
async fn test_detector_calibration_requires_power() {
    let detector = TestDetector::new();

    // Should fail when not powered
    let result = detector.calibrate().await;
    assert!(matches!(result, Err(DetectorError::NotPowered)));

    // Should succeed when powered
    detector.power_on().await.unwrap();
    detector.calibrate().await.unwrap();
}

#[tokio::test]
async fn test_detector_stop_exposure() {
    let detector = TestDetector::new();
    detector.power_on().await.unwrap();

    // Stop exposure (should not error even if no exposure running)
    detector.stop_exposure().await.unwrap();
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);
}

#[tokio::test]
async fn test_detector_exposure_count() {
    let detector = TestDetector::new();
    detector.power_on().await.unwrap();

    let health = detector.get_health().await;
    assert_eq!(health.total_exposures, 0);

    // Do multiple exposures
    detector.start_exposure(100).await.unwrap();
    detector.start_exposure(200).await.unwrap();
    detector.start_exposure(300).await.unwrap();

    let health = detector.get_health().await;
    assert_eq!(health.total_exposures, 3);
}

// ============================================================================
// DemoDetector Specific Tests
// ============================================================================

#[tokio::test]
async fn test_demo_detector_exposure_simulation() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Start exposure
    detector.start_exposure(200).await.unwrap();

    // Initially should be exposing
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Exposing);

    // Wait for exposure to complete (200ms exposure + 100ms read)
    tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;

    // Should be idle now
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);

    // Should have result
    let result = detector.get_last_result().await;
    assert!(result.is_some());
    assert_eq!(result.unwrap().exposure_time_ms, 200);
}

#[tokio::test]
async fn test_demo_detector_reading_phase() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    detector.start_exposure(100).await.unwrap();

    // Wait for exposure phase to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(110)).await;

    // Should be in reading phase
    assert_eq!(detector.get_status().await, DetectorStatus::Reading);

    // Wait for reading to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    // Should be idle
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);
}

#[tokio::test]
async fn test_demo_detector_cannot_start_during_exposure() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Start first exposure
    detector.start_exposure(500).await.unwrap();

    // Try to start second exposure immediately
    let result = detector.start_exposure(100).await;
    assert!(matches!(result, Err(DetectorError::ExposureInProgress)));
}

#[tokio::test]
async fn test_demo_detector_stop_exposure_during_run() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Start exposure
    detector.start_exposure(1000).await.unwrap();

    // Verify it's running
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Exposing);

    // Stop it
    detector.stop_exposure().await.unwrap();

    // Should be idle
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);
}

#[tokio::test]
async fn test_demo_detector_temperature_simulation() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    let initial_health = detector.get_health().await;
    let initial_temp = initial_health.temperature;

    // Do an exposure
    detector.start_exposure(100).await.unwrap();

    // Wait for completion
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    let final_health = detector.get_health().await;
    let final_temp = final_health.temperature;

    // Temperature should increase after exposure
    assert!(final_temp > initial_temp);
}

#[tokio::test]
async fn test_demo_detector_power_off_stops_exposure() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Start long exposure
    detector.start_exposure(2000).await.unwrap();

    // Verify it's running
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Exposing);

    // Power off
    detector.power_off().await.unwrap();

    // Should be off
    assert_eq!(detector.get_status().await, DetectorStatus::Off);
    assert!(!detector.is_powered().await);
}

#[tokio::test]
async fn test_demo_detector_data_size_simulation() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Shorter exposure
    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    let result1 = detector.get_last_result().await.unwrap();
    let size1 = result1.data_size;

    // Longer exposure
    detector.start_exposure(500).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(650)).await;

    let result2 = detector.get_last_result().await.unwrap();
    let size2 = result2.data_size;

    // Longer exposure should produce more data
    assert!(size2 > size1);
}

#[tokio::test]
async fn test_demo_detector_voltage_on_power() {
    let detector = DemoDetector::new();

    let health = detector.get_health().await;
    assert_eq!(health.voltage, 0.0);

    detector.power_on().await.unwrap();

    let health = detector.get_health().await;
    assert_eq!(health.voltage, 12.0);

    detector.power_off().await.unwrap();

    let health = detector.get_health().await;
    assert_eq!(health.voltage, 0.0);
}

#[tokio::test]
async fn test_demo_detector_temperature_returns_to_room_on_power_off() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Do exposure to heat up
    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    let heated_temp = detector.get_health().await.temperature;
    assert!(heated_temp > 22.5);

    // Power off
    detector.power_off().await.unwrap();

    let cooled_temp = detector.get_health().await.temperature;
    assert_eq!(cooled_temp, 22.5); // Room temperature
}

#[tokio::test]
async fn test_demo_detector_result_includes_timestamp() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    let result = detector.get_last_result().await.unwrap();
    
    // Timestamp should be recent (within last 5 seconds)
    let now = chrono::Utc::now();
    let age = now.signed_duration_since(result.timestamp);
    assert!(age.num_seconds() < 5);
}

#[tokio::test]
async fn test_demo_detector_result_includes_detector_temp() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    let result = detector.get_last_result().await.unwrap();
    
    // Temperature should be recorded
    assert!(result.detector_temp > 0.0);
}

#[tokio::test]
async fn test_demo_detector_sequential_exposures() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // First exposure
    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);

    // Second exposure
    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);

    // Third exposure
    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);

    let health = detector.get_health().await;
    assert_eq!(health.total_exposures, 3);
}

#[tokio::test]
async fn test_demo_detector_exposure_cancellation() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Start first exposure
    detector.start_exposure(500).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    assert_eq!(detector.get_status().await, DetectorStatus::Exposing);

    // Cancel by starting new exposure
    detector.stop_exposure().await.unwrap();

    // Start new exposure immediately
    detector.start_exposure(100).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    // Should complete the second one
    assert_eq!(detector.get_status().await, DetectorStatus::Idle);
    let result = detector.get_last_result().await.unwrap();
    assert_eq!(result.exposure_time_ms, 100);
}

#[tokio::test]
async fn test_demo_detector_calibration() {
    let detector = DemoDetector::new();
    detector.power_on().await.unwrap();

    // Calibration should complete (this is deterministic in tests despite random in production)
    // We just verify it doesn't panic and returns Ok or Err properly
    let result = detector.calibrate().await;
    
    // Should either succeed or fail with CalibrationError (10% chance)
    match result {
        Ok(_) => { /* Success path */ },
        Err(DetectorError::CalibrationError(_)) => { /* Expected failure path */ },
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
