import { useEffect } from 'react';
import { SafetyInterlockPanel } from '@/components/status/SafetyInterlockPanel';
import { HardwareStatusPanel } from '@/components/hardware/HardwareStatusPanel';
import { useStore } from '@/store/useStore';
import { api } from '@/services/api';
import { wsService } from '@/services/websocket';

export function Dashboard() {
  const systemState = useStore((state) => state.systemState);
  const addNotification = useStore((state) => state.addNotification);
  const setSystemState = useStore((state) => state.setSystemState);

  // Subscribe to WebSocket hardware initialization events
  useEffect(() => {
    const unsubscribe = wsService.on('hardware_init', (data: unknown) => {
      const event = data as { device: string; status: string; detail: Record<string, unknown> };
      
      // Show success notification
      const deviceName = event.device.charAt(0).toUpperCase() + event.device.slice(1);
      addNotification(
        `${deviceName} initialized successfully`,
        'success'
      );

      // Refresh system state to show updated device status
      const refreshSystemState = async () => {
        try {
          const response = await api.getSystemState();
          if (response.success && response.data) {
            setSystemState(response.data);
          }
        } catch (error) {
          console.error('Failed to refresh system state:', error);
        }
      };
      refreshSystemState();
    });

    return unsubscribe;
  }, [addNotification, setSystemState]);


  if (!systemState) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">System Offline</h2>
        <p className="text-gray-600">Connecting to orchestrator...</p>
        <p className="text-xs text-gray-400 mt-2">Make sure the orchestrator is running on localhost:8081</p>
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <div className="mb-1.5">
        <h1 className="text-lg font-bold text-gray-900 mb-0.5">Dashboard</h1>
        <p className="text-xs text-gray-600">System overview and equipment status</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
        {/* Left column */}
        <div>
          <SafetyInterlockPanel interlocks={systemState.interlocks} />
        </div>

        {/* Right column */}
        <div>
          <HardwareStatusPanel />
        </div>
      </div>

      <div className="p-1.5 bg-yellow-50 border border-yellow-200 rounded">
        <p className="text-xs text-yellow-800 font-medium">
          ℹ️ Measurements can only be started from the Measurement tab. Ensure all hardware is initialized and calibrated first.
        </p>
      </div>
    </div>
  );
}
