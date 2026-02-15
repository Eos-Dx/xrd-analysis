import { useEffect, useState } from 'react';
import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import { Loading } from '@/components/common/Loading';
import { api } from '@/services/api';
import type { MeasurementResult } from '@/types';
import { format } from 'date-fns';

export function MeasurementHistory() {
  const [measurements, setMeasurements] = useState<MeasurementResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    const result = await api.getMeasurementHistory(50);
    
    if (result.success && result.data) {
      setMeasurements(result.data);
      setError(null);
    } else {
      setError(result.error || 'Failed to load history');
    }
    setLoading(false);
  };

  if (loading) return <Loading message="Loading measurement history..." />;

  if (error) {
    return (
      <Card title="Measurement History">
        <div className="text-red-600 p-4">Error: {error}</div>
      </Card>
    );
  }

  return (
    <Card title="Measurement History">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Timestamp
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Sample ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Exposure (s)
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                SNR
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                QC
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {measurements.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                  No measurements found
                </td>
              </tr>
            ) : (
              measurements.map((m) => (
                <tr key={m.runId} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {format(new Date(m.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                  </td>
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                    {m.sampleId}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <Badge
                      variant={
                        m.status === 'SUCCESS' ? 'success' :
                        m.status === 'FAILED' ? 'error' : 'warning'
                      }
                      size="sm"
                    >
                      {m.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700">
                    {m.exposureActual.toFixed(1)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700">
                    {m.snr.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <Badge
                      variant={m.qcPassed ? 'success' : 'error'}
                      size="sm"
                    >
                      {m.qcPassed ? 'PASS' : 'FAIL'}
                    </Badge>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
