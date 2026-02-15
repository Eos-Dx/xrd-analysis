import { useState, useEffect } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { api } from '@/services/api';
import { useStore } from '@/store/useStore';
import { formatDistanceToNow } from 'date-fns';
import type { SystemHealthResponse, CalibrationQcCheck } from '@/types';

export function CalibrationPanel() {
  const [healthData, setHealthData] = useState<SystemHealthResponse | null>(null);
  const [showReport, setShowReport] = useState(false);
  const { 
    systemState, 
    calibrationRunning, 
    runningCalibrationId, 
    latestCalibration,
    setLatestCalibration,
    addNotification 
  } = useStore();
  
  // Fetch system health (includes calibration and device status)
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const result = await api.getSystemHealth();
        console.log('[Calibration] getSystemHealth result:', result);
        if (result.success && result.data) {
          console.log('[Calibration] Setting healthData to:', result.data);
          setHealthData(result.data);
        } else {
          console.log('[Calibration] getSystemHealth failed:', result.error);
        }
      } catch (error) {
        console.error('[Calibration] getSystemHealth error:', error);
      }
    };
    
    fetchHealth();
    const interval = setInterval(fetchHealth, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  // Load latest calibration report on mount
  useEffect(() => {
    loadLatestCalibration();
  }, []);

  const loadLatestCalibration = async () => {
    try {
      const result = await api.getLatestCalibration();
      if (result.success && result.data) {
        console.log('[Calibration] Loaded latest calibration:', result.data);
        setLatestCalibration(result.data);
        setShowReport(false); // Don't auto-show on load
        addNotification('Latest calibration loaded', 'success');
      } else if (result.error) {
        // 404 or no calibration found - this is expected on first run
        console.log('[Calibration] No previous calibration found:', result.error);
        addNotification('No previous calibration found', 'info');
      }
    } catch (error) {
      // Network error or endpoint not implemented yet - silently handle
      console.log('[Calibration] Could not load latest calibration (endpoint may not be implemented):', error);
      addNotification('Failed to load latest calibration', 'error');
    }
  };
  
  const calibration = healthData?.calibration;
  
  // Check if we have valid calibration data from either source
  const hasValidCalibration = calibration?.valid || (latestCalibration?.calibration_id && latestCalibration?.overall_pass);
  const calibrationTimestamp = latestCalibration?.timestamp || calibration?.timestamp;
  
  // Show report when calibration completes
  useEffect(() => {
    if (latestCalibration && !calibrationRunning) {
      setShowReport(true);
    }
  }, [latestCalibration, calibrationRunning]);
  
  // Server will validate readiness when calibration is started
  // No client-side checks needed

  const handleStartCalibration = async () => {
    try {
      // Calibration is now async - returns immediately
      // WebSocket will notify when complete
      const result = await api.startCalibration();
      
      if (result.success && result.data?.calibration_id) {
        console.log('[Calibration] Calibration initiated:', result.data.calibration_id);
        // UI state updated via WebSocket events in App.tsx
      } else {
        addNotification(result.error || 'Failed to start calibration', 'error');
      }
    } catch (error) {
      addNotification('Network error starting calibration', 'error');
    }
  };

  const renderQcCheck = (label: string, check: CalibrationQcCheck) => (
    <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">{label}</span>
          <Badge variant={check.passed ? 'success' : 'error'} size="sm">
            {check.passed ? 'PASS' : 'FAIL'}
          </Badge>
        </div>
        <div className="text-xs text-gray-600 mt-1">
          Measured: <span className="font-mono">{check.measured.toFixed(3)}</span> | 
          Threshold: <span className="font-mono">{check.threshold.toFixed(3)}</span>
        </div>
        {check.details && (
          <div className="text-xs text-gray-500 mt-1">{check.details}</div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <Card title="Daily Calibration">
        <div className="space-y-4">
          {hasValidCalibration && calibrationTimestamp ? (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Status</p>
                  <Badge variant={hasValidCalibration ? 'success' : 'error'} size="lg">
                    {hasValidCalibration ? 'Valid' : 'Expired'}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Last Calibration</p>
                  <p className="font-medium text-gray-900">
                    {formatDistanceToNow(new Date(calibrationTimestamp), { addSuffix: true })}
                  </p>
                </div>
                {calibration?.expires_at && (
                  <div>
                    <p className="text-sm text-gray-600">Expires</p>
                    <p className="font-medium text-gray-900">
                      {formatDistanceToNow(new Date(calibration.expires_at), { addSuffix: true })}
                    </p>
                  </div>
                )}
                {calibration?.distance_check !== undefined && (
                  <div>
                    <p className="text-sm text-gray-600">Distance Check</p>
                    <Badge variant={calibration.distance_check ? 'success' : 'error'} size="sm">
                      {calibration.distance_check ? 'PASS' : 'FAIL'}
                    </Badge>
                  </div>
                )}
                {latestCalibration?.calibration_id && latestCalibration.overall_pass !== undefined && (
                  <div>
                    <p className="text-sm text-gray-600">QC Status</p>
                    <Badge variant={latestCalibration.overall_pass ? 'success' : 'error'} size="sm">
                      {latestCalibration.overall_pass ? 'PASS' : 'FAIL'}
                    </Badge>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
              <p className="text-sm text-yellow-800 font-medium">
                No calibration found. Please perform daily calibration.
              </p>
            </div>
          )}

          {!hasValidCalibration && !calibrationRunning && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
              <p className="text-sm text-yellow-800 font-medium">
                ⚠️ Calibration required before measurements can be performed
              </p>
            </div>
          )}

          {/* Running indicator */}
          {calibrationRunning && (
            <div className="bg-blue-50 border border-blue-200 rounded p-4">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-blue-900">
                    Calibration in Progress
                  </p>
                  <p className="text-xs text-blue-700 mt-1">
                    ID: {runningCalibrationId}
                  </p>
                  <p className="text-xs text-blue-600 mt-1">
                    This may take several seconds. You can navigate away - you'll be notified when complete.
                  </p>
                </div>
              </div>
            </div>
          )}

          <Button
            variant="primary"
            size="lg"
            onClick={handleStartCalibration}
            disabled={calibrationRunning || systemState?.system_state === 'RUNNING'}
            className="w-full"
          >
            {calibrationRunning ? 'Calibration Running...' : 'Start Daily Calibration'}
          </Button>

          <div className="flex gap-2">
            {latestCalibration?.calibration_id && !showReport && (
              <Button
                variant="secondary"
                size="md"
                onClick={() => {
                  console.log('[Calibration] Showing report:', latestCalibration);
                  setShowReport(true);
                }}
                className="flex-1"
              >
                View Last Calibration Report
              </Button>
            )}

            {/* Load latest calibration button */}
            <Button
              variant="secondary"
              size="md"
              onClick={loadLatestCalibration}
              className={latestCalibration?.calibration_id && !showReport ? 'flex-1' : 'w-full'}
            >
              Load Latest Calibration
            </Button>
          </div>

          {/* Debug: Show when endpoint is not returning data */}
          {(!latestCalibration || !latestCalibration.calibration_id) && !calibrationRunning && (
            <div className="text-xs text-gray-500 text-center p-2 bg-gray-50 rounded">
              ℹ️ No previous calibration report available. 
              {process.env.NODE_ENV === 'development' && (
                <span className="block mt-1">
                  (Orchestrator endpoint /api/calibration/latest returned no calibration_id)
                </span>
              )}
            </div>
          )}

          <div className="text-sm text-gray-600">
            <p className="font-medium mb-2">Calibration Procedure:</p>
            <ol className="list-decimal list-inside space-y-1 text-xs">
              <li>Insert calibration standard sample (LaB₆)</li>
              <li>Close safety door</li>
              <li>Click "Start Daily Calibration"</li>
              <li>System will perform automatic QC validation</li>
              <li>Review results below</li>
            </ol>
          </div>
        </div>
      </Card>

      {/* QC Report */}
      {latestCalibration?.calibration_id && showReport && latestCalibration.qc_checks && (
        <Card title="Calibration QC Report">
          <div className="space-y-4">
            {/* Overall Status */}
            <div className={`p-4 rounded-lg border-2 ${
              latestCalibration.overall_pass 
                ? 'bg-green-50 border-green-300' 
                : 'bg-red-50 border-red-300'
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Overall Status</h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Calibrant: {latestCalibration.calibrant_material || 'N/A'} | 
                    Time: {latestCalibration.timestamp ? new Date(latestCalibration.timestamp).toLocaleString() : 'N/A'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1 font-mono">
                    ID: {latestCalibration.calibration_id || 'N/A'}
                  </p>
                </div>
                <Badge 
                  variant={latestCalibration.overall_pass ? 'success' : 'error'} 
                  size="lg"
                  className="text-xl px-6 py-3"
                >
                  {latestCalibration.overall_pass ? '✓ PASS' : '✗ FAIL'}
                </Badge>
              </div>
            </div>

            {/* QC Checks */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Quality Control Checks</h4>
              <div className="space-y-2">
                {renderQcCheck('Total Intensity', latestCalibration.qc_checks.total_intensity)}
                {renderQcCheck('Goodness of Fit', latestCalibration.qc_checks.goodness_of_fit)}
                {renderQcCheck('Signal-to-Noise Ratio (SNR)', latestCalibration.qc_checks.snr)}
                {renderQcCheck('Ring Quality', latestCalibration.qc_checks.ring_quality)}
              </div>
            </div>

            {/* PONI Calibration Results */}
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-gray-700">PONI Calibration Results</h4>
                {latestCalibration.qc_checks.poni.success !== undefined && (
                  <Badge variant={latestCalibration.qc_checks.poni.success ? 'success' : 'error'} size="sm">
                    {latestCalibration.qc_checks.poni.success ? 'SUCCESS' : 'FAILED'}
                  </Badge>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-gray-50 p-2 rounded">
                  <span className="text-gray-600">Sample Distance:</span>
                  <p className="font-mono font-semibold text-gray-900">
                    {latestCalibration.qc_checks.poni.distance_mm?.toFixed(2) ?? 'N/A'} mm
                  </p>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <span className="text-gray-600">Wavelength:</span>
                  <p className="font-mono font-semibold text-gray-900">
                    {latestCalibration.qc_checks.poni.wavelength_angstrom?.toFixed(4) ?? 'N/A'} Å
                  </p>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <span className="text-gray-600">Beam Center X:</span>
                  <p className="font-mono font-semibold text-gray-900">
                    {latestCalibration.qc_checks.poni.beam_center_x?.toFixed(2) ?? 'N/A'} px
                  </p>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <span className="text-gray-600">Beam Center Y:</span>
                  <p className="font-mono font-semibold text-gray-900">
                    {latestCalibration.qc_checks.poni.beam_center_y?.toFixed(2) ?? 'N/A'} px
                  </p>
                </div>
              </div>
            </div>

            {/* Formatted Report */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-700">Full Report</h4>
                <Button 
                  variant="secondary" 
                  size="sm"
                  onClick={() => {
                    const blob = new Blob([latestCalibration.formatted_report || ''], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `calibration-${latestCalibration.calibration_id || 'unknown'}.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  Download Report
                </Button>
              </div>
              <pre className="bg-gray-900 text-green-400 p-4 rounded text-xs font-mono overflow-x-auto max-h-96">
                {latestCalibration.formatted_report || 'No formatted report available'}
              </pre>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <Button 
                variant="secondary" 
                onClick={() => setShowReport(false)}
                className="flex-1"
              >
                Close Report
              </Button>
              {latestCalibration.overall_pass && (
                <Button 
                  variant="primary" 
                  onClick={() => {
                    addNotification('Calibration accepted and saved', 'success');
                    setShowReport(false);
                  }}
                  className="flex-1"
                >
                  Accept Calibration
                </Button>
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
