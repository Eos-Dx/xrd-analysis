import type { 
  ApiResponse, 
  User,
  SystemStateResponse,
  SystemHealthResponse,
  MeasurementParams, 
  MeasurementResult,
  CalibrationStatus,
  CalibrationQcReport,
  AuditEvent,
  PatientCreate,
  Patient
} from '@/types';

const API_BASE = '/api';

class ApiClient {
  private sessionId: string | null = null;

  setSessionId(sessionId: string | null) {
    this.sessionId = sessionId;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<ApiResponse<T>> {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      
      // Merge headers from options if provided
      if (options?.headers) {
        Object.entries(options.headers).forEach(([key, value]) => {
          if (typeof value === 'string') {
            headers[key] = value;
          }
        });
      }

      // Add session ID if available
      if (this.sessionId) {
        headers['x-session-id'] = this.sessionId;
      }

      console.log('[API] Fetch request:', { endpoint, method: options?.method, headers });
      const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers,
      });
      console.log('[API] Fetch response:', { status: response.status, ok: response.ok });

      if (!response.ok) {
        let errorMessage = 'Request failed';
        let errorCode;
        
        // Handle 401 Unauthorized - clear session
        if (response.status === 401) {
          console.log('[API] 401 Unauthorized - clearing session');
          this.sessionId = null;
          // Trigger a page reload to go back to login
          if (typeof window !== 'undefined') {
            window.location.reload();
          }
        }
        
        try {
          const error = await response.json();
          errorMessage = error.error || error.message || errorMessage;
          errorCode = error.code;
        } catch {
          // Server returned non-JSON (likely HTML error page)
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        
        return {
          success: false,
          error: errorMessage,
          errorCode,
        };
      }

      const data = await response.json();
      // If the response already has success/data structure, return it as-is
      if (data && typeof data === 'object' && 'success' in data) {
        return data;
      }
      return { success: true, data };
    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Authentication
  async login(username: string, password: string): Promise<ApiResponse<User>> {
    console.log('[API] Login request:', { username });
    const response = await this.request<{ sessionId: string; user: User }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });

    console.log('[API] Login response:', response);

    if (response.success) {
      // Handle both Node.js proxy format {data: {sessionId, user}} and orchestrator format {session_id, user_id, role}
      const sessionId = (response.data as any)?.sessionId || (response as any).session_id;
      const userData = (response.data as any)?.user || {
        username: (response as any).user_id,
        role: (response as any).role,
        permissions: ['read', 'write', 'admin']
      };
      
      if (sessionId) {
        console.log('[API] Setting session ID:', sessionId);
        this.setSessionId(sessionId);
        return { success: true, data: userData };
      }
    }

    console.error('[API] Login failed:', response.error);
    return { success: false, error: response.error };
  }

  async logout(): Promise<ApiResponse<void>> {
    return this.request<void>('/auth/logout', { method: 'POST' });
  }

  // System state (compact, for polling)
  async getSystemState(): Promise<ApiResponse<SystemStateResponse>> {
    const response = await this.request<any>('/state');
    
    // Transform orchestrator format to UI format
    if (response.success && response.data) {
      const data = response.data;
      const transformed: SystemStateResponse = {
        system_state: data.state || data.system_state,
        devices: data.devices || {},
        interlocks: data.interlocks || {},
        timestamp: data.timestamp || new Date().toISOString()
      };
      return { success: true, data: transformed };
    }
    
    return response;
  }

  // System health (detailed, for diagnostics)
  async getSystemHealth(): Promise<ApiResponse<SystemHealthResponse>> {
    return this.request<SystemHealthResponse>('/health');
  }

  // Measurements
  async startMeasurement(params: MeasurementParams): Promise<ApiResponse<{ measurementId: string; status: string }>> {
    const response = await this.request<any>('/measurements/start', {
      method: 'POST',
      body: JSON.stringify(params),
    });
    
    // Orchestrator returns {measurement_id, status, patient_id, exposure_ms}
    // Transform to UI format {success, data: {measurementId, status}}
    if (response.success || response.measurement_id) {
      const data = response.data || response;
      return { 
        success: true, 
        data: { 
          measurementId: data.measurement_id, 
          status: data.status || 'started'
        } 
      };
    }
    
    return response;
  }

  async stopMeasurement(runId: string): Promise<ApiResponse<void>> {
    return this.request('/measurements/stop', { 
      method: 'POST',
      body: JSON.stringify({ run_id: runId })
    });
  }

  async abortMeasurement(runId: string): Promise<ApiResponse<void>> {
    // Orchestrator may use /measurements/abort or similar endpoint
    return this.request('/measurements/abort', { 
      method: 'POST',
      body: JSON.stringify({ run_id: runId })
    });
  }

  async getMeasurementHistory(patientId?: string, limit = 50): Promise<ApiResponse<MeasurementResult[]>> {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (patientId) {
      params.append('patient_id', patientId);
    }
    
    const response = await this.request<any>(`/measurements/history?${params.toString()}`);
    
    // Orchestrator returns {measurements: []}
    // Transform to UI format {success, data: [...]}
    if (response.success && (response.measurements || response.data?.measurements)) {
      const measurements = response.measurements || response.data?.measurements || [];
      return { success: true, data: measurements };
    }
    
    return response;
  }

  // Calibration
  async startCalibration(): Promise<ApiResponse<{ calibration_id: string }>> {
    // Calibration is now ASYNC - returns immediately with calibration_id
    // WebSocket events (calibration_start, calibration_complete) notify when done
    const response = await this.request<any>('/calibration/start', { method: 'POST' });
    
    if (response.success && (response.calibration_id || response.data?.calibration_id)) {
      const calibrationId = response.calibration_id || response.data?.calibration_id;
      return { success: true, data: { calibration_id: calibrationId } };
    }
    
    return response;
  }

  async getRunningCalibration(): Promise<ApiResponse<{ running: boolean; calibration_id?: string }>> {
    return this.request('/calibration/running');
  }

  async getLatestCalibration(): Promise<ApiResponse<CalibrationQcReport>> {
    const response = await this.request<any>('/calibration/latest');
    // Orchestrator returns {success, calibration_id, ...} directly, not wrapped in data
    // Transform to {success, data: {calibration_id, ...}} format expected by UI
    if (response.success && response.calibration_id) {
      return { success: true, data: response as CalibrationQcReport };
    }
    return response;
  }

  async getCalibrationStatus(): Promise<ApiResponse<CalibrationStatus>> {
    return this.request<CalibrationStatus>('/calibration/status');
  }

  async getCalibrationHistory(limit = 20): Promise<ApiResponse<CalibrationStatus[]>> {
    const response = await this.request<any>(`/calibration/history?limit=${limit}`);
    // Orchestrator returns {success, total, records: [...]} at top level
    if (response.success) {
      const recordsArray = (response as any).records || response.data?.records || [];
      const records = recordsArray.map((r: any) => {
        // Derive distance check robustly from possible fields
        const distanceCheck = (
          r.distance_check ??
          r.qc_checks?.distance_check?.passed ??
          (typeof r.overall_pass === 'boolean' ? r.overall_pass : false)
        );
        return {
          id: r.id,
          timestamp: new Date(r.timestamp),
          valid: true, // TODO: compute from expiry if provided
          expiresAt: new Date(new Date(r.timestamp).getTime() + 24 * 60 * 60 * 1000), // +24 hours
          distanceCheck,
          snrThreshold: r.qc_checks?.snr?.threshold ?? 10.0,
          parameters: {}
        } as CalibrationStatus;
      });
      return { success: true, data: records };
    }
    return response;
  }

  // GPIO & Safety
  async getGpioState(): Promise<ApiResponse<import('@/types').GpioState>> {
    return this.request('/gpio/state');
  }

  async getEnableButtonStatus(): Promise<ApiResponse<import('@/types').EnableButtonStatus>> {
    return this.request('/gpio/enable-button');
  }

  // Hardware control
  async initializeDevice(device: 'detector' | 'motion'): Promise<ApiResponse<void>> {
    return this.request(`/hardware/${device}/init`, { method: 'POST' });
  }

  async stopDevice(device: 'detector' | 'motion'): Promise<ApiResponse<void>> {
    return this.request(`/hardware/${device}/stop`, { method: 'POST' });
  }

  // Audit logs
  async getAuditLogs(
    startDate?: Date,
    endDate?: Date,
    limit = 100
  ): Promise<ApiResponse<AuditEvent[]>> {
    const params = new URLSearchParams();
    if (startDate) params.append('start', startDate.toISOString());
    if (endDate) params.append('end', endDate.toISOString());
    params.append('limit', limit.toString());
    
    return this.request<AuditEvent[]>(`/audit?${params.toString()}`);
  }

  // Patients
  async createPatient(patient: PatientCreate): Promise<ApiResponse<Patient>> {
    return this.request<Patient>('/patients', {
      method: 'POST',
      body: JSON.stringify(patient),
    });
  }

  async findPatientByMRN(mrn: string): Promise<ApiResponse<Patient>> {
    return this.request<Patient>(`/patients/search?mrn=${encodeURIComponent(mrn)}`);
  }

  async getPatientById(patientId: string): Promise<ApiResponse<Patient>> {
    return this.request<Patient>(`/patients/${encodeURIComponent(patientId)}`);
  }
}

export const api = new ApiClient();
