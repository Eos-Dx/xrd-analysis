import { useState } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { api } from '@/services/api';
import { useStore } from '@/store/useStore';
import type { Patient } from '@/types';

interface MeasurementPanelProps {
  patient: Patient;
}

export function MeasurementPanel({ patient }: MeasurementPanelProps) {
  const [sampleId, setSampleId] = useState('');
  const [exposureDuration, setExposureDuration] = useState(30);
  const [isStarting, setIsStarting] = useState(false);
  const { user, systemState, activeMeasurement, setActiveMeasurement, addNotification } = useStore();

  const canStartMeasurement = 
    systemState?.system_state === 'IDLE' && 
    sampleId.trim().length > 0 &&
    !activeMeasurement;

  const handleStartMeasurement = async () => {
    if (!user || !canStartMeasurement) return;

    setIsStarting(true);
    try {
      const result = await api.startMeasurement({
        patient_id: patient.patient_id,
        sample_id: sampleId.trim(),
        exposure_duration: exposureDuration,
        notes: '',
      });

      if (result.success) {
        addNotification(`Measurement started: ${result.data?.runId}`, 'success');
        // Clear form
        setSampleId('');
      } else {
        addNotification(result.error || 'Failed to start measurement', 'error');
      }
    } catch (error) {
      addNotification('Network error starting measurement', 'error');
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopMeasurement = async () => {
    if (!activeMeasurement) return;

    try {
      const result = await api.stopMeasurement(activeMeasurement.runId);
      if (result.success) {
        addNotification('Measurement stopped', 'info');
        setActiveMeasurement(null);
      }
    } catch (error) {
      addNotification('Failed to stop measurement', 'error');
    }
  };

  const handleAbortMeasurement = async () => {
    if (!activeMeasurement) return;

    try {
      const result = await api.abortMeasurement(activeMeasurement.runId);
      if (result.success) {
        addNotification('Measurement aborted', 'warning');
        setActiveMeasurement(null);
      }
    } catch (error) {
      addNotification('Failed to abort measurement', 'error');
    }
  };

  return (
    <Card title="Start Measurement">
      {activeMeasurement ? (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded p-4">
            <h4 className="font-semibold text-blue-900 mb-2">Measurement in Progress</h4>
            <p className="text-sm text-blue-800">Sample ID: {activeMeasurement.sampleId}</p>
            <p className="text-sm text-blue-800">Run ID: {activeMeasurement.runId}</p>
          </div>
          <div className="flex space-x-3">
            <Button variant="secondary" onClick={handleStopMeasurement}>
              Stop Measurement
            </Button>
            <Button variant="danger" onClick={handleAbortMeasurement}>
              Emergency Abort
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Sample ID *
            </label>
            <input
              type="text"
              value={sampleId}
              onChange={(e) => setSampleId(e.target.value)}
              placeholder="Enter sample identifier"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Exposure Duration (seconds)
            </label>
            <input
              type="number"
              value={exposureDuration}
              onChange={(e) => setExposureDuration(Number(e.target.value))}
              min={5}
              max={300}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* TODO: Re-enable calibration check when /api/health is implemented */}
          {false && (
            <div className="bg-red-50 border border-red-200 rounded p-3">
              <p className="text-sm text-red-800 font-medium">
                ⚠️ Calibration required before measurement
              </p>
            </div>
          )}

          <Button
            variant="primary"
            size="lg"
            onClick={handleStartMeasurement}
            disabled={!canStartMeasurement || isStarting}
            className="w-full"
          >
            {isStarting ? 'Starting...' : 'Start Measurement'}
          </Button>
        </div>
      )}
    </Card>
  );
}
