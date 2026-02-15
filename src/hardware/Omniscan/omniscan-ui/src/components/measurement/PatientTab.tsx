import { useState } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { api } from '@/services/api';
import type { Patient, PatientCreate } from '@/types';

interface PatientTabProps {
  onPatientSelected: (patient: Patient) => void;
}

export function PatientTab({ onPatientSelected }: PatientTabProps) {
  const [mode, setMode] = useState<'search' | 'create'>('search');
  const [searchMRN, setSearchMRN] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  
  // Form fields for new patient
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [dateOfBirth, setDateOfBirth] = useState('');
  const [medicalRecordNumber, setMedicalRecordNumber] = useState('');

  const handleSearch = async () => {
    if (!searchMRN.trim()) {
      setError('Please enter a Medical Record Number');
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const result = await api.findPatientByMRN(searchMRN.trim());
      if (result.success && result.data) {
        setSelectedPatient(result.data);
        setError(null);
      } else {
        setError(result.error || 'Patient not found');
        setSelectedPatient(null);
      }
    } catch (err) {
      setError('Network error while searching for patient');
      setSelectedPatient(null);
    } finally {
      setIsSearching(false);
    }
  };

  const handleCreatePatient = async () => {
    if (!firstName.trim() || !lastName.trim() || !dateOfBirth || !medicalRecordNumber.trim()) {
      setError('All fields are required');
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      const patientData: PatientCreate = {
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        date_of_birth: dateOfBirth,
        medical_record_number: medicalRecordNumber.trim(),
      };

      const result = await api.createPatient(patientData);
      if (result.success && result.data) {
        setSelectedPatient(result.data);
        setError(null);
        // Clear form
        setFirstName('');
        setLastName('');
        setDateOfBirth('');
        setMedicalRecordNumber('');
      } else {
        setError(result.error || 'Failed to create patient');
      }
    } catch (err) {
      setError('Network error while creating patient');
    } finally {
      setIsCreating(false);
    }
  };

  const handleSelectPatient = () => {
    if (selectedPatient) {
      onPatientSelected(selectedPatient);
    }
  };

  const clearSelection = () => {
    setSelectedPatient(null);
    setError(null);
    setSearchMRN('');
  };

  return (
    <div className="space-y-6">
      {selectedPatient ? (
        <Card title="Patient Selected">
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded p-4">
              <h4 className="font-semibold text-green-900 mb-2">
                {selectedPatient.first_name} {selectedPatient.last_name}
              </h4>
              <p className="text-sm text-green-800">MRN: {selectedPatient.medical_record_number}</p>
              <p className="text-sm text-green-800">DOB: {selectedPatient.date_of_birth}</p>
              <p className="text-sm text-green-800">Patient ID: {selectedPatient.patient_id}</p>
            </div>
            <div className="flex space-x-3">
              <Button variant="primary" onClick={handleSelectPatient} className="flex-1">
                Continue to Measurement
              </Button>
              <Button variant="secondary" onClick={clearSelection}>
                Change Patient
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <>
          <div className="flex space-x-4 mb-4">
            <Button
              variant={mode === 'search' ? 'primary' : 'secondary'}
              onClick={() => {
                setMode('search');
                setError(null);
              }}
            >
              Find Existing Patient
            </Button>
            <Button
              variant={mode === 'create' ? 'primary' : 'secondary'}
              onClick={() => {
                setMode('create');
                setError(null);
              }}
            >
              Register New Patient
            </Button>
          </div>

          {mode === 'search' ? (
            <Card title="Find Patient">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Medical Record Number (MRN)
                  </label>
                  <input
                    type="text"
                    value={searchMRN}
                    onChange={(e) => setSearchMRN(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Enter MRN"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded p-3">
                    <p className="text-sm text-red-800">{error}</p>
                  </div>
                )}

                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleSearch}
                  disabled={isSearching || !searchMRN.trim()}
                  className="w-full"
                >
                  {isSearching ? 'Searching...' : 'Search'}
                </Button>
              </div>
            </Card>
          ) : (
            <Card title="Register New Patient">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      First Name *
                    </label>
                    <input
                      type="text"
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      placeholder="First name"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Last Name *
                    </label>
                    <input
                      type="text"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      placeholder="Last name"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Date of Birth *
                  </label>
                  <input
                    type="date"
                    value={dateOfBirth}
                    onChange={(e) => setDateOfBirth(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Medical Record Number (MRN) *
                  </label>
                  <input
                    type="text"
                    value={medicalRecordNumber}
                    onChange={(e) => setMedicalRecordNumber(e.target.value)}
                    placeholder="Enter MRN"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded p-3">
                    <p className="text-sm text-red-800">{error}</p>
                  </div>
                )}

                <Button
                  variant="primary"
                  size="lg"
                  onClick={handleCreatePatient}
                  disabled={
                    isCreating ||
                    !firstName.trim() ||
                    !lastName.trim() ||
                    !dateOfBirth ||
                    !medicalRecordNumber.trim()
                  }
                  className="w-full"
                >
                  {isCreating ? 'Creating...' : 'Register Patient'}
                </Button>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
