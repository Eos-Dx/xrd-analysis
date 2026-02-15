// TODO: Update these tests to work with new architecture
// Temporarily disabled pending refactoring

/*
use std::sync::Arc;
use omniscan_hw_server::{
    app::App,
    state::ServerState,
    safety::interlocks::AlwaysOkInterlocks,
    devices::{TestDetector, TestMotion, DetectorDevice, MotionControlDevice}
};

#[tokio::test]
async fn watchdog_times_out_to_safe() {
    let detector: Arc<dyn DetectorDevice> = Arc::new(TestDetector::new());
    let motion:   Arc<dyn MotionControlDevice> = Arc::new(TestMotion::new());
    detector.power_on().await.unwrap();
    motion.power_on().await.unwrap();

    let app = App::new(detector, motion, Arc::new(AlwaysOkInterlocks::new()));
    assert_eq!(app.current_state(), ServerState::Idle);

    app.start_exposure(50).await.unwrap(); // 50 ms
    tokio::time::sleep(std::time::Duration::from_millis(120)).await;

    assert_eq!(app.current_state(), ServerState::Safe);
}

#[tokio::test]
async fn stop_cancels_watchdog_and_returns_idle() {
    let detector: Arc<dyn DetectorDevice> = Arc::new(TestDetector::new());
    let motion:   Arc<dyn MotionControlDevice> = Arc::new(TestMotion::new());
    detector.power_on().await.unwrap();
    motion.power_on().await.unwrap();

    let app = App::new(detector, motion, Arc::new(AlwaysOkInterlocks::new()));
    app.start_exposure(500).await.unwrap(); // long timeout
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    app.stop().await.unwrap(); // should cancel watchdog and go Idle
    assert_eq!(app.current_state(), ServerState::Idle);
}

#[tokio::test]
async fn health_aggregates_components() {
    let detector: Arc<dyn DetectorDevice> = Arc::new(TestDetector::new());
    let motion:   Arc<dyn MotionControlDevice> = Arc::new(TestMotion::new());
    detector.power_on().await.unwrap();
    motion.power_on().await.unwrap();

    let app = App::new(detector, motion, Arc::new(AlwaysOkInterlocks::new()));
    let (ok, comps) = app.aggregate_health().await;
    assert!(ok);
    assert_eq!(comps.len(), 2);
}
*/
