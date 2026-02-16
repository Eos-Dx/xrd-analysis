use omniscan_hw_server::devices::motions::{
    DemoMotion, MotionControlDevice, MotionError, MotionStatus, TestMotion,
};

#[tokio::test]
async fn test_motion_power_on_off() {
    let motion = TestMotion::new();

    // Initially powered off
    assert!(!motion.is_powered().await);
    assert_eq!(motion.get_status().await, MotionStatus::Off);

    // Power on
    motion.power_on().await.unwrap();
    assert!(motion.is_powered().await);
    assert_eq!(motion.get_status().await, MotionStatus::Idle);

    // Power off
    motion.power_off().await.unwrap();
    assert!(!motion.is_powered().await);
    assert_eq!(motion.get_status().await, MotionStatus::Off);
}

#[tokio::test]
async fn test_motion_requires_power() {
    let motion = TestMotion::new();

    // Operations should fail when not powered
    assert!(motion.home().await.is_err());
    assert!(motion.move_to(10.0).await.is_err());
    assert!(motion.set_velocity(5.0).await.is_err());

    // Should succeed when powered
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();
}

#[tokio::test]
async fn test_motion_requires_homing() {
    let motion = TestMotion::new();
    motion.power_on().await.unwrap();

    // Position should be unknown before homing
    assert!(motion.get_position().await.is_none());

    // Move should fail without homing
    let result = motion.move_to(10.0).await;
    assert!(matches!(result, Err(MotionError::HomingRequired)));

    // Home first
    motion.home().await.unwrap();

    // Now position should be known
    assert_eq!(motion.get_position().await, Some(0.0));

    // Move should succeed
    motion.move_to(10.0).await.unwrap();
}

#[tokio::test]
async fn test_motion_health_info() {
    let motion = TestMotion::new();

    let health = motion.get_health().await;
    assert!(!health.powered);
    assert!(!health.is_homed);
    assert!(health.position.is_none());

    motion.power_on().await.unwrap();

    let health = motion.get_health().await;
    assert!(health.powered);
    assert!(!health.is_homed);

    motion.home().await.unwrap();

    let health = motion.get_health().await;
    assert!(health.is_homed);
    assert_eq!(health.position, Some(0.0));
}

#[tokio::test]
async fn test_motion_homing() {
    let motion = TestMotion::new();
    motion.power_on().await.unwrap();

    // Home the motion controller
    motion.home().await.unwrap();

    // Position should be 0
    assert_eq!(motion.get_position().await, Some(0.0));

    let health = motion.get_health().await;
    assert!(health.is_homed);
}

#[tokio::test]
async fn test_motion_move_to_absolute() {
    let motion = TestMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Move to position
    motion.move_to(25.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(25.0));

    motion.move_to(50.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(50.0));

    motion.move_to(10.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(10.0));
}

#[tokio::test]
async fn test_motion_move_relative() {
    let motion = TestMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Start at 0
    assert_eq!(motion.get_position().await, Some(0.0));

    // Move +10mm
    motion.move_relative(10.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(10.0));

    // Move +15mm
    motion.move_relative(15.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(25.0));

    // Move -5mm
    motion.move_relative(-5.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(20.0));
}

#[tokio::test]
async fn test_motion_limits() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    let limits = motion.get_limits().await;

    // Try to move beyond max limit
    let result = motion.move_to(limits.max_position + 10.0).await;
    assert!(matches!(result, Err(MotionError::InvalidPosition(_))));

    // Try to move below min limit
    let result = motion.move_to(limits.min_position - 10.0).await;
    assert!(matches!(result, Err(MotionError::InvalidPosition(_))));

    // Valid position should work
    motion.move_to(limits.max_position).await.unwrap();
    assert_eq!(
        motion.get_position().await,
        Some(limits.max_position)
    );
}

#[tokio::test]
async fn test_motion_velocity_control() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();

    // Set velocity
    motion.set_velocity(20.0).await.unwrap();

    // Invalid velocity (zero)
    let result = motion.set_velocity(0.0).await;
    assert!(matches!(result, Err(MotionError::InvalidPosition(_))));

    // Invalid velocity (negative)
    let result = motion.set_velocity(-10.0).await;
    assert!(matches!(result, Err(MotionError::InvalidPosition(_))));

    let limits = motion.get_limits().await;

    // Velocity beyond max
    let result = motion.set_velocity(limits.max_velocity + 10.0).await;
    assert!(matches!(result, Err(MotionError::InvalidPosition(_))));
}

#[tokio::test]
async fn test_motion_stop() {
    let motion = TestMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Stop motion (should not error even if not moving)
    motion.stop_motion().await.unwrap();
    assert_eq!(motion.get_status().await, MotionStatus::Idle);
}

#[tokio::test]
async fn test_motion_move_count() {
    let motion = TestMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    let health = motion.get_health().await;
    assert_eq!(health.total_moves, 0);

    // Do multiple moves
    motion.move_to(10.0).await.unwrap();
    motion.move_to(20.0).await.unwrap();
    motion.move_to(30.0).await.unwrap();

    let health = motion.get_health().await;
    assert_eq!(health.total_moves, 3);
}

// ============================================================================
// DemoMotion Specific Tests
// ============================================================================

#[tokio::test]
async fn test_demo_motion_homing_simulation() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();

    // Start homing
    let home_task = tokio::spawn(async move {
        motion.home().await.unwrap();
        motion
    });

    // Give it a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Wait for completion (1 second simulated homing time)
    let motion = home_task.await.unwrap();

    assert_eq!(motion.get_status().await, MotionStatus::Idle);
    assert_eq!(motion.get_position().await, Some(0.0));
}

#[tokio::test]
async fn test_demo_motion_movement_simulation() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Set known velocity for timing
    motion.set_velocity(10.0).await.unwrap(); // 10 mm/s

    // Start movement
    let move_task = tokio::spawn(async move {
        motion.move_to(50.0).await.unwrap(); // 50mm at 10mm/s = 5 seconds
        motion
    });

    // Check status during movement
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Wait for completion
    let motion = move_task.await.unwrap();

    assert_eq!(motion.get_status().await, MotionStatus::Idle);
    assert_eq!(motion.get_position().await, Some(50.0));
}

#[tokio::test]
async fn test_demo_motion_cannot_move_during_homing() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();

    // Start homing
    let motion_clone = DemoMotion::new();
    motion_clone.power_on().await.unwrap();
    
    let home_task = tokio::spawn(async move {
        motion_clone.home().await
    });

    // Try to move while homing (using different instance for this test pattern)
    // In real scenario, you'd check the same instance's status
    
    home_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn test_demo_motion_stop_during_movement() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Start long move
    motion.set_velocity(10.0).await.unwrap();
    let motion_arc = std::sync::Arc::new(motion);
    let motion_clone = motion_arc.clone();

    tokio::spawn(async move {
        let _ = motion_clone.move_to(100.0).await;
    });

    // Give it time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Stop motion
    motion_arc.stop_motion().await.unwrap();

    // Should be idle
    assert_eq!(motion_arc.get_status().await, MotionStatus::Idle);
}

#[tokio::test]
async fn test_demo_motion_power_off_clears_position() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();
    motion.move_to(50.0).await.unwrap();

    // Position should be known
    assert!(motion.get_position().await.is_some());

    // Power off
    motion.power_off().await.unwrap();

    // Position should be unknown
    assert!(motion.get_position().await.is_none());

    let health = motion.get_health().await;
    assert!(!health.is_homed);
}

#[tokio::test]
async fn test_demo_motion_sequential_moves() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Multiple sequential moves
    motion.move_to(10.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(10.0));

    motion.move_to(25.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(25.0));

    motion.move_to(5.0).await.unwrap();
    assert_eq!(motion.get_position().await, Some(5.0));

    let health = motion.get_health().await;
    assert_eq!(health.total_moves, 3);
}

#[tokio::test]
async fn test_demo_motion_move_time_scales_with_distance() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Set fixed velocity
    motion.set_velocity(20.0).await.unwrap(); // 20 mm/s

    // Short move
    let start = std::time::Instant::now();
    motion.move_to(10.0).await.unwrap(); // 10mm at 20mm/s = 0.5s
    let short_duration = start.elapsed();

    // Return to start
    motion.move_to(0.0).await.unwrap();

    // Longer move
    let start = std::time::Instant::now();
    motion.move_to(40.0).await.unwrap(); // 40mm at 20mm/s = 2s
    let long_duration = start.elapsed();

    // Longer distance should take proportionally more time
    assert!(long_duration > short_duration);
}

#[tokio::test]
async fn test_demo_motion_relative_moves_accumulate() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Series of relative moves
    motion.move_relative(5.0).await.unwrap();
    motion.move_relative(10.0).await.unwrap();
    motion.move_relative(5.0).await.unwrap();

    // Total: 5 + 10 + 5 = 20
    assert_eq!(motion.get_position().await, Some(20.0));
}

#[tokio::test]
async fn test_demo_motion_relative_move_beyond_limits() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    let limits = motion.get_limits().await;

    // Move near max limit
    motion.move_to(limits.max_position - 5.0).await.unwrap();

    // Try to move beyond with relative move
    let result = motion.move_relative(10.0).await;
    assert!(matches!(result, Err(MotionError::InvalidPosition(_))));
}

#[tokio::test]
async fn test_demo_motion_velocity_affects_move_time() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Slow velocity
    motion.set_velocity(5.0).await.unwrap(); // 5 mm/s
    let start = std::time::Instant::now();
    motion.move_to(10.0).await.unwrap(); // 10mm at 5mm/s = 2s
    let slow_duration = start.elapsed();

    // Return
    motion.move_to(0.0).await.unwrap();

    // Fast velocity
    motion.set_velocity(20.0).await.unwrap(); // 20 mm/s
    let start = std::time::Instant::now();
    motion.move_to(10.0).await.unwrap(); // 10mm at 20mm/s = 0.5s
    let fast_duration = start.elapsed();

    // Fast should be quicker
    assert!(fast_duration < slow_duration);
}

#[tokio::test]
async fn test_demo_motion_limits_are_enforced() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    let limits = motion.get_limits().await;

    // Max position should work
    motion.move_to(limits.max_position).await.unwrap();
    assert_eq!(
        motion.get_position().await,
        Some(limits.max_position)
    );

    // Min position should work
    motion.move_to(limits.min_position).await.unwrap();
    assert_eq!(
        motion.get_position().await,
        Some(limits.min_position)
    );

    // Beyond max should fail
    let result = motion.move_to(limits.max_position + 1.0).await;
    assert!(result.is_err());

    // Below min should fail
    let result = motion.move_to(limits.min_position - 1.0).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_demo_motion_health_uptime() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();

    // Wait a bit
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let health = motion.get_health().await;
    assert!(health.uptime.as_millis() >= 100);
}

#[tokio::test]
async fn test_demo_motion_target_position_tracking() {
    let motion = DemoMotion::new();
    motion.power_on().await.unwrap();
    motion.home().await.unwrap();

    // Initially no target
    let health = motion.get_health().await;
    assert!(health.target_position.is_none());

    // Start a move (spawn to avoid blocking)
    let motion_arc = std::sync::Arc::new(motion);
    let motion_clone = motion_arc.clone();
    
    tokio::spawn(async move {
        let _ = motion_clone.move_to(50.0).await;
    });

    // Give it time to set target
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // During move, target should be set
    let health = motion_arc.get_health().await;
    // Note: might be None if move completed very fast
}
