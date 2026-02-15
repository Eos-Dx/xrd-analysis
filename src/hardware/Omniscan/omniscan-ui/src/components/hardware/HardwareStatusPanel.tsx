import { useState } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { api } from '@/services/api';
import { useStore } from '@/store/useStore';
import type { 
  PduDeviceStatus, 
  GpioDeviceStatus, 
  DetectorDeviceStatus, 
  MotionDeviceStatus 
} from '@/types';

export function HardwareStatusPanel() {
  const { systemState, addNotification } = useStore();
  const [isOperating, setIsOperating] = useState<{
    detector: boolean;
    motion: boolean;
    gpio: boolean;
  }>({ detector: false, motion: false, gpio: false });

  const handleStartDevice = async (device: 'detector' | 'motion' | 'gpio') => {
    console.log('[HardwareStatusPanel] Init button clicked for:', device);
    console.log('[HardwareStatusPanel] Enable button pressed:', enableButtonPressed);
    console.log('[HardwareStatusPanel] Can start devices:', canStartDevices);
    
    setIsOperating(prev => ({ ...prev, [device]: true }));
    console.log('[HardwareStatusPanel] Set isOperating to true for:', device);
    
    try {
      console.log('[HardwareStatusPanel] Calling api.initializeDevice for:', device);
      const result = await api.initializeDevice(device);
      console.log('[HardwareStatusPanel] API result:', result);
      
      if (result.success) {
        console.log('[HardwareStatusPanel] Success! Adding notification');
        addNotification(
          `${device.charAt(0).toUpperCase() + device.slice(1)} started successfully`,
          'success'
        );
      } else {
        console.log('[HardwareStatusPanel] Failed! Error:', result.error);
        addNotification(
          result.error || `Failed to start ${device}`,
          'error'
        );
      }
    } catch (error) {
      console.error('[HardwareStatusPanel] Exception caught:', error);
      addNotification(
        `Network error starting ${device}`,
        'error'
      );
    } finally {
      console.log('[HardwareStatusPanel] Setting isOperating back to false for:', device);
      setIsOperating(prev => ({ ...prev, [device]: false }));
    }
  };

  const handleStopDevice = async (device: 'detector' | 'motion' | 'gpio') => {
    setIsOperating(prev => ({ ...prev, [device]: true }));
    try {
      const result = await api.stopDevice(device);
      
      if (result.success) {
        addNotification(
          `${device.charAt(0).toUpperCase() + device.slice(1)} stopped successfully`,
          'success'
        );
      } else {
        addNotification(
          result.error || `Failed to stop ${device}`,
          'error'
        );
      }
    } catch (error) {
      addNotification(
        `Network error stopping ${device}`,
        'error'
      );
    } finally {
      setIsOperating(prev => ({ ...prev, [device]: false }));
    }
  };

  const getDeviceStatusBadge = (
    status: PduDeviceStatus | GpioDeviceStatus | DetectorDeviceStatus | MotionDeviceStatus,
    powered: boolean
  ) => {
    if (!powered || status === 'OFF' || status === 'Off') {
      return <Badge variant="error" size="sm">{status}</Badge>;
    }
    if (status === 'IDLE' || status === 'Active') {
      return <Badge variant="success" size="sm">{status}</Badge>;
    }
    if (status === 'HOMING') {
      return <Badge variant="warning" size="sm">{status}</Badge>;
    }
    if (status === 'EXPOSING' || status === 'READING' || status === 'MOVING') {
      return <Badge variant="info" size="sm">{status}</Badge>;
    }
    if (status === 'ERROR' || status === 'LIMIT_HIT') {
      return <Badge variant="error" size="sm">{status}</Badge>;
    }
    return <Badge variant="neutral" size="sm">{status}</Badge>;
  };


  const devices = systemState?.devices;
  const interlocks = systemState?.interlocks;
  
  // Check if enable button is pressed (required to start detector/motion)
  const enableButtonPressed = interlocks?.enable_button || false;
  const canStartDevices = enableButtonPressed; // Only requires activation button

  if (!devices || !devices.detector || !devices.motion) {
    return (
      <Card title="Hardware Status">
        <div className="text-center py-4 text-gray-500">Loading device status...</div>
      </Card>
    );
  }

  return (
    <Card title="Hardware Status">
      <div className="space-y-2">
        {/* Device Status Cards */}
        <div className="grid grid-cols-2 gap-2">
          {/* PDU Card - only show if available */}
          {devices.pdu && (
            <div className="border border-gray-200 rounded-lg p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-gray-700">PDU</span>
                <div 
                  className={`w-2.5 h-2.5 rounded-full`}
                  style={{ backgroundColor: devices.pdu.powered && devices.pdu.status === 'Active' ? '#22c55e' : '#ef4444' }}
                />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs text-gray-600">Powered: {devices.pdu.powered ? 'Yes' : 'No'}</div>
                {getDeviceStatusBadge(devices.pdu.status as PduDeviceStatus, devices.pdu.powered)}
              </div>
            </div>
          )}

          {/* GPIO Card - only show if available */}
          {devices.gpio && (
            <div className="border border-gray-200 rounded-lg p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-gray-700">GPIO</span>
                <div 
                  className={`w-2.5 h-2.5 rounded-full`}
                  style={{ backgroundColor: devices.gpio.powered && devices.gpio.status === 'Active' ? '#22c55e' : '#ef4444' }}
                />
              </div>
              <div className="space-y-0.5">
                <div className="text-xs text-gray-600">Powered: {devices.gpio.powered ? 'Yes' : 'No'}</div>
                {getDeviceStatusBadge(devices.gpio.status as GpioDeviceStatus, devices.gpio.powered)}
              </div>
            </div>
          )}

          {/* Detector Card */}
          <div className="border border-gray-200 rounded-lg p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-gray-700">Detector</span>
              <div 
                className={`w-2.5 h-2.5 rounded-full`}
                style={{ 
                  backgroundColor: devices.detector.powered && devices.detector.status === 'IDLE' 
                    ? '#22c55e' 
                    : !devices.detector.powered || devices.detector.status === 'OFF'
                    ? '#9ca3af'
                    : devices.detector.status === 'ERROR'
                    ? '#ef4444'
                    : '#f59e0b'
                }}
              />
            </div>
            <div className="space-y-0.5">
              <div className="text-xs text-gray-600">Powered: {devices.detector.powered ? 'Yes' : 'No'}</div>
              {getDeviceStatusBadge(devices.detector.status as DetectorDeviceStatus, devices.detector.powered)}
            </div>
            <div className="mt-1">
              {!devices.detector.powered || devices.detector.status === 'OFF' ? (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleStartDevice('detector')}
                  disabled={!canStartDevices || isOperating.detector}
                  title={!canStartDevices ? 'Enable button must be pressed' : ''}
                  className="w-full text-xs py-1"
                >
                  {isOperating.detector ? 'Initializing...' : 'Init'}
                </Button>
              ) : (
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => handleStopDevice('detector')}
                  disabled={isOperating.detector}
                  className="w-full text-xs py-1"
                >
                  {isOperating.detector ? 'Stopping...' : 'Stop'}
                </Button>
              )}
            </div>
          </div>

          {/* Motion Card */}
          <div className="border border-gray-200 rounded-lg p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-gray-700">Motion</span>
              <div 
                className={`w-2.5 h-2.5 rounded-full`}
                style={{ 
                  backgroundColor: devices.motion.powered && devices.motion.status === 'IDLE' 
                    ? '#22c55e' 
                    : !devices.motion.powered || devices.motion.status === 'OFF'
                    ? '#9ca3af'
                    : devices.motion.status === 'ERROR'
                    ? '#ef4444'
                    : '#f59e0b'
                }}
              />
            </div>
            <div className="space-y-0.5">
              <div className="text-xs text-gray-600">Powered: {devices.motion.powered ? 'Yes' : 'No'}</div>
              {getDeviceStatusBadge(devices.motion.status as MotionDeviceStatus, devices.motion.powered)}
            </div>
            <div className="mt-1">
              {!devices.motion.powered || devices.motion.status === 'OFF' ? (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => handleStartDevice('motion')}
                  disabled={!canStartDevices || isOperating.motion}
                  title={!canStartDevices ? 'Enable button must be pressed' : ''}
                  className="w-full text-xs py-1"
                >
                  {isOperating.motion ? 'Initializing...' : 'Init'}
                </Button>
              ) : (
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() => handleStopDevice('motion')}
                  disabled={isOperating.motion}
                  className="w-full text-xs py-1"
                >
                  {isOperating.motion ? 'Stopping...' : 'Stop'}
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Activation Button Status */}
        <div className="p-2 rounded-lg" style={{ backgroundColor: enableButtonPressed ? '#f0fdf4' : '#f9fafb' }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-2.5 h-2.5 rounded-full ${enableButtonPressed ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`} />
              <span className="text-xs font-medium" style={{ color: enableButtonPressed ? '#15803d' : '#6b7280' }}>
                Activation Button
              </span>
            </div>
            {enableButtonPressed ? (
              <span className="text-xs font-bold text-green-700">ACTIVE</span>
            ) : (
              <span className="text-xs text-gray-500">Not Pressed</span>
            )}
          </div>
          {enableButtonPressed && (
            <p className="text-xs text-green-600 mt-0.5">
              ✓ Device controls are now enabled
            </p>
          )}
        </div>

        {!enableButtonPressed && (
          <div className="p-2 bg-yellow-50 border border-yellow-200 rounded">
            <p className="text-xs text-yellow-800 font-medium">
              ⚠️ Press Activation Button to enable device controls
            </p>
          </div>
        )}
        
        <div className="p-2 bg-blue-50 border border-blue-200 rounded">
          <p className="text-xs text-blue-800 font-medium">
            ℹ️ All hardware must be started before calibration and measurements
          </p>
        </div>
      </div>
    </Card>
  );
}
