import { useState, useEffect } from 'react';
import { Badge } from '@/components/common/Badge';
import { Button } from '@/components/common/Button';
import { api } from '@/services/api';
import { formatDistanceToNow } from 'date-fns';
import type { CalibrationStatus } from '@/types';

export function CalibrationHistory() {
  const [history, setHistory] = useState<CalibrationStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.getCalibrationHistory(20);
      if (result.success && result.data) {
        setHistory(result.data);
      } else {
        setError(result.error || 'Failed to load calibration history');
      }
    } catch (err) {
      setError('Network error loading calibration history');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-gray-600">Loading calibration history...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-3">
        <div className="bg-red-50 border border-red-200 rounded p-3">
          <p className="text-sm text-red-800">{error}</p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={loadHistory}
          className="w-full"
        >
          Retry
        </Button>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-600 text-sm mb-3">
          No calibration history found
        </p>
        <Button
          variant="secondary"
          size="sm"
          onClick={loadHistory}
          className="w-full"
        >
          Refresh
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm text-gray-600">
          Showing {history.length} recent calibration{history.length !== 1 ? 's' : ''}
        </p>
        <Button
          variant="secondary"
          size="sm"
          onClick={loadHistory}
        >
          Refresh
        </Button>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {history.map((cal) => (
          <div
            key={cal.id}
            className="border border-gray-200 rounded-lg p-3 hover:border-gray-300 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant={cal.valid ? 'success' : 'error'} size="sm">
                    {cal.valid ? 'Valid' : 'Expired'}
                  </Badge>
                  <Badge variant={cal.distanceCheck ? 'success' : 'error'} size="sm">
                    Distance: {cal.distanceCheck ? 'PASS' : 'FAIL'}
                  </Badge>
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <p>
                    <span className="font-medium">Time:</span>{' '}
                    {formatDistanceToNow(new Date(cal.timestamp), { addSuffix: true })}
                  </p>
                  <p>
                    <span className="font-medium">Expires:</span>{' '}
                    {formatDistanceToNow(new Date(cal.expiresAt), { addSuffix: true })}
                  </p>
                  <p>
                    <span className="font-medium">SNR Threshold:</span>{' '}
                    <span className="font-mono">{cal.snrThreshold.toFixed(1)}</span>
                  </p>
                  <p className="font-mono text-gray-500">ID: {cal.id}</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
