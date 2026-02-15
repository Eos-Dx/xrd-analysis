import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { LoginScreen } from '@/components/auth/LoginScreen';
import { NotificationToast } from '@/components/common/NotificationToast';
import { Dashboard } from '@/pages/Dashboard';
import { MeasurementPage } from '@/pages/MeasurementPage';
import { CalibrationPage } from '@/pages/CalibrationPage';
import { useStore } from '@/store/useStore';
import { wsService } from '@/services/websocket';
import { api } from '@/services/api';

function App() {
  const { user, setSystemState, setCalibrationRunning, setLatestCalibration, addNotification } = useStore();

  useEffect(() => {
    if (!user) return;

    // Get session ID from API client
    const sessionId = (api as any).sessionId;

    // Connect WebSocket for real-time updates with session auth
    wsService.connect(sessionId);

    // Subscribe to system health updates from WebSocket
    const unsubscribeHealth = wsService.on('system_health', (data) => {
      console.log('[App] WebSocket health update:', data);
      // Map legacy health data to new state format if needed
      setSystemState(data as any);
    });

    // Subscribe to GPIO updates (enable button, interlocks) from WebSocket
    const unsubscribeGpio = wsService.on('gpio_update', (data: any) => {
      console.log('[App] WebSocket GPIO update:', data);
      // Update interlocks in system state
      setSystemState((current) => {
        if (!current) return current;
        return {
          ...current,
          interlocks: {
            ...current.interlocks,
            enable_button: data.enableButton ?? current.interlocks.enable_button,
            key_switch: data.keySwitch ?? current.interlocks.key_switch,
          }
        };
      });
    });

    // Subscribe to state changes (including interlock changes) from WebSocket
    const unsubscribeStateChanges = wsService.on('state_change', (data: any) => {
      console.log('[App] WebSocket state change:', data);
      // Trigger state refresh on state changes
      pollState();
    });

    // Subscribe to calibration start events
    const unsubscribeCalStart = wsService.on('calibration_start', (data: any) => {
      console.log('[App] Calibration started:', data);
      setCalibrationRunning(true, data.calibration_id);
      addNotification('Calibration started - running in background', 'info');
    });

    // Subscribe to calibration complete events
    const unsubscribeCalComplete = wsService.on('calibration_complete', (data: any) => {
      console.log('[App] Calibration completed:', data);
      setCalibrationRunning(false, null);
      
      if (data.success) {
        // Fetch full report
        api.getLatestCalibration().then(result => {
          if (result.success && result.data) {
            setLatestCalibration(result.data);
            if (data.overall_pass) {
              addNotification('Calibration completed successfully - All QC checks passed!', 'success');
            } else {
              addNotification('Calibration completed with QC failures - Review report', 'warning');
            }
          }
        });
      } else {
        addNotification(`Calibration failed: ${data.error || 'Unknown error'}`, 'error');
      }
    });

    // Poll system state initially (compact endpoint for fast polling)
    const pollState = async () => {
      console.log('[App] Polling state...');
      const result = await api.getSystemState();
      console.log('[App] State poll result:', result);
      if (result.success && result.data) {
        console.log('[App] Setting system state:', result.data);
        setSystemState(result.data);
      } else {
        console.error('[App] State poll failed:', result.error);
      }
    };
    pollState();

    // Poll every 3 seconds for responsive UI updates (interlocks, enable button, device status)
    const interval = setInterval(pollState, 3000);

    return () => {
      unsubscribeHealth();
      unsubscribeGpio();
      unsubscribeStateChanges();
      unsubscribeCalStart();
      unsubscribeCalComplete();
      wsService.disconnect();
      clearInterval(interval);
    };
  }, [user, setSystemState, setCalibrationRunning, setLatestCalibration, addNotification]);

  // Show login screen if no user
  if (!user) {
    return (
      <>
        <LoginScreen />
        <NotificationToast />
      </>
    );
  }

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/measurement" element={<MeasurementPage />} />
          <Route path="/calibration" element={<CalibrationPage />} />
          <Route path="/history" element={<div>History Page (TODO)</div>} />
          <Route path="/maintenance" element={<div>Maintenance Page (TODO)</div>} />
          <Route path="/admin" element={<div>Admin Page (TODO)</div>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
      <NotificationToast />
    </BrowserRouter>
  );
}

export default App;
