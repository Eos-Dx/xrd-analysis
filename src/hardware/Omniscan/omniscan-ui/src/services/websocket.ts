
// Event types aligned with orchestrator WebSocket API (API_REFERENCE.md lines 384-423)
type EventType = 
  | 'state_change'              // System state changed (e.g., IDLE -> RUNNING)
  | 'system_health'             // Periodic health snapshot
  | 'gpio_state_change'         // GPIO state changed (enable button, key switch, etc.)
  | 'gpio_update'               // Legacy GPIO event name (UI compatibility)
  | 'calibration_start'         // Calibration started
  | 'calibration_complete'      // Calibration finished
  | 'measurement_complete'      // Measurement finished
  | 'measurement_start'         // Measurement started
  | 'measurement_stop'          // Measurement stopped
  | 'hardware_init'             // Device initialization
  | 'hardware_stop'             // Device stopped
  | 'safety_alert'              // Safety interlock violation
  | 'connection'                // Connection status
  | 'echo';                     // Echo response

interface WebSocketEvent {
  type: EventType;
  data?: {
    // For state_change events
    state?: string;              // System state (IDLE, RUNNING, etc.)
    timestamp?: string;
    
    // For gpio_state_change events
    enable_button_active?: boolean;
    enable_button_remaining_secs?: number;
    key_switch_on?: boolean;
    
    // For calibration_complete events
    calibration_id?: string;
    overall_pass?: boolean;
    
    // For measurement_complete events
    measurement_id?: string;
    status?: string;
    
    // Generic data
    [key: string]: any;
  };
  timestamp?: string;
}

type EventHandler = (data: unknown) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private handlers: Map<EventType, Set<EventHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(sessionId?: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    // Connect to orchestrator WebSocket (port 8081 - same as REST API)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const wsUrl = `${protocol}//${host}:8081/ws`;

    console.log('[WebSocket] Connecting to:', wsUrl);
    this.ws = new WebSocket(wsUrl);

    // Send session ID after connection if provided
    if (sessionId) {
      this.ws.onopen = () => {
        console.log('[WebSocket] Connected, sending auth');
        this.ws?.send(JSON.stringify({ type: 'auth', sessionId }));
        this.reconnectAttempts = 0;
      };
    } else {
      this.ws.onopen = () => {
        console.log('[WebSocket] Connected (no auth)');
        this.reconnectAttempts = 0;
      };
    }


    this.ws.onmessage = (event) => {
      try {
        const wsEvent: WebSocketEvent = JSON.parse(event.data);
        console.log('[WebSocket] Received:', wsEvent.type, wsEvent.data);
        this.handleEvent(wsEvent);
      } catch (error) {
        console.error('[WebSocket] Failed to parse message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };

    this.ws.onclose = () => {
      console.log('[WebSocket] Disconnected, will retry...');
      this.attemptReconnect();
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  on(eventType: EventType, handler: EventHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
    };
  }

  private handleEvent(event: WebSocketEvent): void {
    const handlers = this.handlers.get(event.type);
    if (handlers) {
      handlers.forEach((handler) => handler(event.data));
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }
}

export const wsService = new WebSocketService();
