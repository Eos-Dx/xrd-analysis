import { useState } from 'react';
import { PatientTab } from '@/components/measurement/PatientTab';
import { MeasurementPanel } from '@/components/measurement/MeasurementPanel';
import type { Patient } from '@/types';

type Tab = 'patient' | 'measurement';

export function MeasurementPage() {
  const [activeTab, setActiveTab] = useState<Tab>('patient');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  const handlePatientSelected = (patient: Patient) => {
    setSelectedPatient(patient);
    setActiveTab('measurement');
  };

  const handleChangePatient = () => {
    setActiveTab('patient');
    setSelectedPatient(null);
  };

  return (
    <div className="h-full p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Measurement</h1>
        {selectedPatient && (
          <p className="text-lg text-gray-600">
            Patient: {selectedPatient.first_name} {selectedPatient.last_name} (MRN: {selectedPatient.medical_record_number})
          </p>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 mb-6">
        <button
          onClick={() => setActiveTab('patient')}
          className={`px-6 py-3 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'patient'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
          }`}
        >
          Patient
        </button>
        <button
          onClick={() => selectedPatient && setActiveTab('measurement')}
          disabled={!selectedPatient}
          className={`px-6 py-3 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'measurement'
              ? 'border-blue-500 text-blue-600'
              : !selectedPatient
              ? 'border-transparent text-gray-300 cursor-not-allowed'
              : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
          }`}
        >
          Measurement
        </button>
      </div>

      {/* Tab Content */}
      <div className="max-w-4xl">
        {activeTab === 'patient' ? (
          <PatientTab onPatientSelected={handlePatientSelected} />
        ) : selectedPatient ? (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <div className="bg-blue-50 border border-blue-200 rounded p-4 flex-1">
                <p className="text-sm text-blue-800">
                  <strong>Patient:</strong> {selectedPatient.first_name} {selectedPatient.last_name}
                </p>
                <p className="text-sm text-blue-800">
                  <strong>MRN:</strong> {selectedPatient.medical_record_number}
                </p>
                <p className="text-sm text-blue-800">
                  <strong>DOB:</strong> {selectedPatient.date_of_birth}
                </p>
              </div>
              <button
                onClick={handleChangePatient}
                className="ml-4 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Change Patient
              </button>
            </div>
            <MeasurementPanel patient={selectedPatient} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

