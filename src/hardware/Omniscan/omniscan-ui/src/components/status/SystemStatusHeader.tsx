import { useStore } from '@/store/useStore';
import { Badge } from '@/components/common/Badge';
import type { SystemState } from '@/types';

export function SystemStatusHeader() {
  const systemState = useStore((state) => state.systemState);

  const getStateColor = (state: SystemState) => {
    switch (state) {
      case 'IDLE':
        return 'success';
      case 'RUNNING':
        return 'info';
      case 'PENDING_ARMED':
        return 'warning';
      case 'STOPPING':
        return 'warning';
      case 'SAFE':
        return 'error';
      default:
        return 'neutral';
    }
  };

  const getStateLabel = (state: SystemState) => {
    switch (state) {
      case 'IDLE':
        return 'Ready';
      case 'RUNNING':
        return 'Running';
      case 'PENDING_ARMED':
        return 'Arming';
      case 'STOPPING':
        return 'Stopping';
      case 'SAFE':
        return 'Safe Mode';
      default:
        return state;
    }
  };

  if (!systemState) {
    return (
      <header className="bg-gray-800 text-white px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold">Omniscan</h1>
          <Badge variant="neutral" size="lg">System Offline</Badge>
        </div>
      </header>
    );
  }

  const { system_state, interlocks } = systemState;
  const allInterlocksOk = interlocks?.overall_safe ?? false;

  return (
    <header className="bg-gray-800 text-white px-6 py-3">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">Omniscan Medical XRD</h1>
          
          <div className="flex items-center space-x-4">
            {/* System State */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-300">Status:</span>
              <Badge variant={getStateColor(system_state)} size="lg">
                {getStateLabel(system_state)}
              </Badge>
            </div>

            {/* Safety Interlocks */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-300">Safety:</span>
              <Badge variant={allInterlocksOk ? 'success' : 'error'} size="lg">
                {allInterlocksOk ? 'All OK' : 'Fault'}
              </Badge>
            </div>

            {/* TODO: Add calibration status when /api/health is implemented */}
          </div>
        </div>
      </div>
    </header>
  );
}
