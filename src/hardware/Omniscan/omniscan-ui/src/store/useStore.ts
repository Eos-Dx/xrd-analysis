import { create } from 'zustand';
import type { User, SystemStateResponse, MeasurementResult, CalibrationQcReport } from '@/types';

interface AppState {
  // User state
  user: User | null;
  setUser: (user: User | null) => void;

  // System state (compact, from /api/state)
  systemState: SystemStateResponse | null;
  setSystemState: (state: SystemStateResponse | ((current: SystemStateResponse | null) => SystemStateResponse | null)) => void;

  // Active measurement
  activeMeasurement: MeasurementResult | null;
  setActiveMeasurement: (measurement: MeasurementResult | null) => void;

  // Calibration state
  calibrationRunning: boolean;
  runningCalibrationId: string | null;
  latestCalibration: CalibrationQcReport | null;
  setCalibrationRunning: (running: boolean, id: string | null) => void;
  setLatestCalibration: (calibration: CalibrationQcReport | null) => void;

  // UI state
  notifications: Notification[];
  addNotification: (message: string, type: Notification['type']) => void;
  removeNotification: (id: string) => void;
}

interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  timestamp: Date;
}

export const useStore = create<AppState>((set) => ({
  // Initial state
  user: null,
  systemState: null,
  activeMeasurement: null,
  calibrationRunning: false,
  runningCalibrationId: null,
  latestCalibration: null,
  notifications: [],

  // Actions
  setUser: (user) => set({ user }),
  
  setSystemState: (state) => set((prev) => ({
    systemState: typeof state === 'function' ? state(prev.systemState) : state
  })),
  
  setActiveMeasurement: (measurement) => set({ activeMeasurement: measurement }),
  
  setCalibrationRunning: (running, id) => set({ 
    calibrationRunning: running, 
    runningCalibrationId: id 
  }),
  
  setLatestCalibration: (calibration) => set({ latestCalibration: calibration }),
  
  addNotification: (message, type) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        {
          id: crypto.randomUUID(),
          type,
          message,
          timestamp: new Date(),
        },
      ],
    })),
  
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
}));
