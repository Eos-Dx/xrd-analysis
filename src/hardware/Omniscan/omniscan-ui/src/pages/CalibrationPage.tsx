import { CalibrationPanel } from '@/components/calibration/CalibrationPanel';
import { CalibrationHistory } from '@/components/calibration/CalibrationHistory';
import { Card } from '@/components/common/Card';

export function CalibrationPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Calibrant</h1>
        <p className="text-gray-600">Perform daily calibration with calibrant standards</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CalibrationPanel />
        
        <Card title="Calibration History">
          <CalibrationHistory />
        </Card>
      </div>

      <Card title="Calibration Requirements">
        <div className="prose prose-sm max-w-none">
          <h4 className="font-semibold text-gray-900 mb-2">Daily Calibration Policy</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>Calibration must be performed once every 24 hours</li>
            <li>System will automatically lock if calibration expires</li>
            <li>Use certified calibration standard samples only</li>
            <li>Distance check must pass for valid calibration</li>
            <li>All measurements are linked to calibration ID</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}
