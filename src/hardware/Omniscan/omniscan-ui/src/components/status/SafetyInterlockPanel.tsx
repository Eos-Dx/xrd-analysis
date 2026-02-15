import { Card } from '@/components/common/Card';
import { Badge } from '@/components/common/Badge';
import type { CompactInterlocks } from '@/types';

interface SafetyInterlockPanelProps {
  interlocks: CompactInterlocks;
}

export function SafetyInterlockPanel({ interlocks }: SafetyInterlockPanelProps) {
  const interlockItems = [
    { key: 'key_switch' as const, label: 'Key Switch', required: true },
    { key: 'enable_button' as const, label: 'Activation Button', required: false },
    { key: 'door_closed' as const, label: 'Door Closed', required: true },
    { key: 'emergency_stop' as const, label: 'Emergency Stop', required: true },
    { key: 'radiation_safe' as const, label: 'Radiation Safe', required: true },
    { key: 'cooling_ok' as const, label: 'Cooling OK', required: true },
    { key: 'power_ok' as const, label: 'Power OK', required: true },
  ] as const;

  const overallSafe = interlocks.overall_safe;

  return (
    <Card title="Safety Interlocks">
      {/* Overall Safe Status */}
      <div className="mb-2 p-2 rounded-lg" style={{ backgroundColor: overallSafe ? '#f0fdf4' : '#fef2f2' }}>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${overallSafe ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm font-semibold" style={{ color: overallSafe ? '#15803d' : '#dc2626' }}>
            {overallSafe ? 'System Safe - Ready for Operation' : 'System Not Safe - Check Interlocks'}
          </span>
        </div>
      </div>
      <div className="space-y-0.5">
        {interlockItems.map((item) => {
          const value = interlocks[item.key];
          const isOk = value === true;
          const isOptional = !item.required && value === undefined;

          return (
            <div key={item.key} className="flex items-center gap-4 py-1 border-b border-gray-100 last:border-0">
              <span className="text-xs font-medium text-gray-700 flex-1">{item.label}</span>
              {isOptional ? (
                <Badge variant="neutral" size="sm">N/A</Badge>
              ) : (
                <Badge variant={isOk ? 'success' : 'error'} size="sm">
                  {isOk ? 'OK' : 'FAULT'}
                </Badge>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
