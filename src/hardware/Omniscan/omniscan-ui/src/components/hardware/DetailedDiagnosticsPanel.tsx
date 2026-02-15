import { useState, useEffect } from 'react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { api } from '@/services/api';
import type { SystemHealthResponse } from '@/types';

export function DetailedDiagnosticsPanel() {
  const [healthData, setHealthData] = useState<SystemHealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHealthData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.getSystemHealth();
      if (result.success && result.data) {
        setHealthData(result.data);
      } else {
        setError(result.error || 'Failed to fetch health data');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHealthData();
  }, []);

  if (isLoading && !healthData) {
    return (
      <Card title="Detailed Diagnostics">
        <div className="text-center py-4 text-gray-500">Loading diagnostics...</div>
      </Card>
    );
  }

  if (error && !healthData) {
    return (
      <Card title="Detailed Diagnostics">
        <div className="text-center py-4">
          <p className="text-red-600 mb-2">{error}</p>
          <Button variant="primary" size="sm" onClick={fetchHealthData}>
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  if (!healthData) {
    return (
      <Card title="Detailed Diagnostics">
        <div className="text-center py-4">
          <Button variant="primary" size="sm" onClick={fetchHealthData}>
            Load Diagnostics
          </Button>
        </div>
      </Card>
    );
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  return (
    <Card 
      title="Detailed Diagnostics" 
      action={
        <Button variant="secondary" size="sm" onClick={fetchHealthData} disabled={isLoading}>
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </Button>
      }
    >
      <div className="space-y-4">
        {/* System Overview */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-sm mb-2">System Overview</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-600">State:</span>{' '}
              <Badge variant="info" size="sm">{healthData.system_state}</Badge>
            </div>
            <div>
              <span className="text-gray-600">Uptime:</span>{' '}
              <span className="font-mono">{formatUptime(healthData.uptime_seconds)}</span>
            </div>
            <div>
              <span className="text-gray-600">Cloud:</span>{' '}
              <Badge variant={healthData.cloud_connected ? 'success' : 'error'} size="sm">
                {healthData.cloud_connected ? 'Connected' : 'Disconnected'}
              </Badge>
            </div>
            <div>
              <span className="text-gray-600">Last Heartbeat:</span>{' '}
              <span className="font-mono text-xs">{new Date(healthData.last_heartbeat).toLocaleTimeString()}</span>
            </div>
          </div>
        </div>

        {/* PDU Details */}
        <div className="border border-gray-200 rounded-lg p-3">
          <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
            PDU (Power Distribution Unit)
            <Badge variant={healthData.pdu.status === 'Active' ? 'success' : 'error'} size="sm">
              {healthData.pdu.status}
            </Badge>
          </h3>
          <div className="space-y-2">
            <div className="text-sm">
              <span className="text-gray-600">Uptime:</span>{' '}
              <span className="font-mono">{formatUptime(healthData.pdu.uptime_seconds)}</span>
            </div>
            <div>
              <div className="text-xs font-semibold text-gray-700 mb-1">Power Outputs:</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${healthData.pdu.outputs.main_power ? 'bg-green-500' : 'bg-gray-300'}`} />
                  Main Power
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${healthData.pdu.outputs.gpio_power ? 'bg-green-500' : 'bg-gray-300'}`} />
                  GPIO Power
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${healthData.pdu.outputs.detector_power ? 'bg-green-500' : 'bg-gray-300'}`} />
                  Detector Power
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${healthData.pdu.outputs.motion_power ? 'bg-green-500' : 'bg-gray-300'}`} />
                  Motion Power
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* GPIO Details */}
        <div className="border border-gray-200 rounded-lg p-3">
          <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
            GPIO (Interlocks & Controls)
            <Badge variant={healthData.gpio.status === 'Active' ? 'success' : 'error'} size="sm">
              {healthData.gpio.status}
            </Badge>
          </h3>
          <div className="space-y-2">
            
            <div>
              <div className="text-xs font-semibold text-gray-700 mb-1">Control Inputs:</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Key Switch</span>
                  <Badge variant={healthData.gpio.key_switch_on ? 'success' : 'neutral'} size="sm">
                    {healthData.gpio.key_switch_on ? 'ON' : 'OFF'}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Activation Button</span>
                  <Badge variant={healthData.gpio.activation_button_active ? 'success' : 'neutral'} size="sm">
                    {healthData.gpio.activation_button_active ? 'ACTIVE' : 'OFF'}
                  </Badge>
                </div>
                {healthData.gpio.activation_remaining_secs !== null && healthData.gpio.activation_remaining_secs > 0 && (
                  <div className="col-span-2 flex items-center justify-between p-1 bg-amber-50 rounded border border-amber-200">
                    <span>Activation Timer</span>
                    <Badge variant="warning" size="sm">
                      {healthData.gpio.activation_remaining_secs}s remaining
                    </Badge>
                  </div>
                )}
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Emergency Stop</span>
                  <Badge variant={healthData.gpio.interlocks.emergency_stop ? 'success' : 'error'} size="sm">
                    {healthData.gpio.interlocks.emergency_stop ? 'OK' : 'PRESSED'}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Door Closed</span>
                  <Badge variant={healthData.gpio.interlocks.door_closed ? 'success' : 'error'} size="sm">
                    {healthData.gpio.interlocks.door_closed ? 'YES' : 'NO'}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Radiation Safe</span>
                  <Badge variant={healthData.gpio.interlocks.radiation_safe ? 'success' : 'error'} size="sm">
                    {healthData.gpio.interlocks.radiation_safe ? 'YES' : 'NO'}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Cooling OK</span>
                  <Badge variant={healthData.gpio.interlocks.cooling_ok ? 'success' : 'error'} size="sm">
                    {healthData.gpio.interlocks.cooling_ok ? 'YES' : 'NO'}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Power OK</span>
                  <Badge variant={healthData.gpio.interlocks.power_ok ? 'success' : 'error'} size="sm">
                    {healthData.gpio.interlocks.power_ok ? 'YES' : 'NO'}
                  </Badge>
                </div>
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-700 mb-1">LED Indicators:</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Main LED</span>
                  <span className="font-semibold" style={{ color: healthData.gpio.main_led === 'Green' ? '#22c55e' : '#ef4444' }}>
                    {healthData.gpio.main_led}
                  </span>
                </div>
                <div className="flex items-center justify-between p-1 bg-gray-50 rounded">
                  <span>Radiation LED</span>
                  <span className="font-semibold" style={{ color: healthData.gpio.radiation_led === 'Green' ? '#22c55e' : '#ef4444' }}>
                    {healthData.gpio.radiation_led}
                  </span>
                </div>
              </div>
            </div>

            <div className="p-2 rounded" style={{ backgroundColor: healthData.gpio.interlocks.overall_safe ? '#f0fdf4' : '#fef2f2' }}>
              <div className="text-xs font-semibold mb-1">Interlock Status:</div>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${healthData.gpio.interlocks.overall_safe ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm font-semibold" style={{ color: healthData.gpio.interlocks.overall_safe ? '#15803d' : '#dc2626' }}>
                  {healthData.gpio.interlocks.overall_safe ? 'All Safe' : 'Not Safe'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Detector Details */}
        <div className="border border-gray-200 rounded-lg p-3">
          <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
            Detector
            <Badge 
              variant={
                healthData.detector.status === 'IDLE' ? 'success' : 
                healthData.detector.status === 'OFF' ? 'neutral' : 
                healthData.detector.status === 'ERROR' ? 'error' : 
                'warning'
              } 
              size="sm"
            >
              {healthData.detector.status}
            </Badge>
          </h3>
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-gray-600">Powered:</span>{' '}
                <span className="font-semibold">{healthData.detector.powered ? 'Yes' : 'No'}</span>
              </div>
              <div>
                <span className="text-gray-600">Initialized:</span>{' '}
                <span className="font-semibold">{healthData.detector.initialized ? 'Yes' : 'No'}</span>
              </div>
            </div>
            {healthData.detector.powered && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-600">Temperature:</span>{' '}
                    <span className="font-mono">{healthData.detector.temperature?.toFixed(1) ?? 'N/A'}°C</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Voltage:</span>{' '}
                    <span className="font-mono">{healthData.detector.voltage?.toFixed(2) ?? 'N/A'}V</span>
                  </div>
                </div>
                {healthData.detector.uptime_seconds !== undefined && (
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-gray-600">Uptime:</span>{' '}
                      <span className="font-mono">{formatUptime(healthData.detector.uptime_seconds)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total Exposures:</span>{' '}
                      <span className="font-mono">{healthData.detector.total_exposures ?? 'N/A'}</span>
                    </div>
                  </div>
                )}
                {healthData.detector.last_exposure_time_ms !== null && (
                  <div>
                    <span className="text-gray-600">Last Exposure:</span>{' '}
                    <span className="font-mono">{healthData.detector.last_exposure_time_ms}ms</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Motion Details */}
        <div className="border border-gray-200 rounded-lg p-3">
          <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
            Motion Control
            <Badge 
              variant={
                healthData.motion.status === 'IDLE' ? 'success' : 
                healthData.motion.status === 'OFF' ? 'neutral' : 
                healthData.motion.status === 'ERROR' ? 'error' : 
                'warning'
              } 
              size="sm"
            >
              {healthData.motion.status}
            </Badge>
          </h3>
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-gray-600">Powered:</span>{' '}
                <span className="font-semibold">{healthData.motion.powered ? 'Yes' : 'No'}</span>
              </div>
              <div>
                <span className="text-gray-600">Initialized:</span>{' '}
                <span className="font-semibold">{healthData.motion.initialized ? 'Yes' : 'No'}</span>
              </div>
            </div>
            {healthData.motion.powered && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-600">Homed:</span>{' '}
                    <Badge variant={healthData.motion.is_homed ? 'success' : 'warning'} size="sm">
                      {healthData.motion.is_homed ? 'Yes' : 'No'}
                    </Badge>
                  </div>
                  {healthData.motion.uptime_seconds !== undefined && (
                    <div>
                      <span className="text-gray-600">Uptime:</span>{' '}
                      <span className="font-mono">{formatUptime(healthData.motion.uptime_seconds)}</span>
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-600">Position:</span>{' '}
                    <span className="font-mono">{healthData.motion.position?.toFixed(2) ?? 'N/A'}mm</span>
                  </div>
                  {healthData.motion.target_position !== undefined && (
                    <div>
                      <span className="text-gray-600">Target:</span>{' '}
                      <span className="font-mono">{healthData.motion.target_position?.toFixed(2) ?? 'N/A'}mm</span>
                    </div>
                  )}
                </div>
                {healthData.motion.total_moves !== undefined && (
                  <div>
                    <span className="text-gray-600">Total Moves:</span>{' '}
                    <span className="font-mono">{healthData.motion.total_moves}</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Calibration Status */}
        <div className="border border-gray-200 rounded-lg p-3">
          <h3 className="font-semibold text-sm mb-2 flex items-center gap-2">
            Calibration Status
            <Badge variant={healthData.calibration.valid ? 'success' : 'error'} size="sm">
              {healthData.calibration.valid ? 'Valid' : 'Invalid'}
            </Badge>
          </h3>
          <div className="space-y-1 text-sm">
            <div>
              <span className="text-gray-600">Timestamp:</span>{' '}
              <span className="font-mono text-xs">{new Date(healthData.calibration.timestamp).toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-600">Expires:</span>{' '}
              <span className="font-mono text-xs">{new Date(healthData.calibration.expires_at).toLocaleString()}</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="text-gray-600">Distance Check:</span>{' '}
                <Badge variant={healthData.calibration.distance_check ? 'success' : 'error'} size="sm">
                  {healthData.calibration.distance_check ? 'Pass' : 'Fail'}
                </Badge>
              </div>
              <div>
                <span className="text-gray-600">SNR Threshold:</span>{' '}
                <span className="font-mono">{healthData.calibration.snr_threshold}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
