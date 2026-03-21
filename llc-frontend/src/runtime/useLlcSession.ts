import { useCallback, useEffect, useRef, useState } from 'react';

import { RuntimeEvent, WebSocketAction } from '../api/contracts';
import { llcApiClient } from '../api/client';
import {
  RuntimeState,
  applyRuntimeEvent,
  applySessionBootstrap,
  appendUserCommand,
  createInitialRuntimeState,
} from './state';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

const SESSION_STORAGE_KEY = 'llc-ui-session-id';
const RECONNECT_BASE_DELAY_MS = 1200;
const RECONNECT_MAX_DELAY_MS = 10000;

const isRuntimeEvent = (value: unknown): value is RuntimeEvent =>
  typeof value === 'object' && value !== null && typeof (value as { type?: unknown }).type === 'string';

const getStoredSessionId = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
  return raw && raw.trim() ? raw.trim() : null;
};

export interface UseLlcSessionResult {
  runtimeState: RuntimeState;
  connectionStatus: ConnectionStatus;
  lastError: string | null;
  busy: boolean;
  sendCommand: (command: string) => boolean;
  interrupt: () => boolean;
  reconnect: () => Promise<void>;
}

export function useLlcSession(): UseLlcSessionResult {
  const [runtimeState, setRuntimeState] = useState<RuntimeState>(() => createInitialRuntimeState());
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [lastError, setLastError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const connectVersionRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);

  const closeSocket = useCallback(() => {
    const socket = socketRef.current;
    if (socket) {
      socket.close();
      socketRef.current = null;
    }
  }, []);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const processEvent = useCallback((event: RuntimeEvent) => {
    setRuntimeState((prev) => applyRuntimeEvent(prev, event));
    if (event.type === 'turn_started') {
      setBusy(true);
      return;
    }
    if (event.type === 'turn_completed' || event.type === 'error') {
      setBusy(false);
      return;
    }
    if (event.type === 'command_output' && event.should_exit) {
      setBusy(false);
    }
  }, []);

  const connect = useCallback(async (options?: { preserveState?: boolean }) => {
    const preserveState = Boolean(options?.preserveState);
    const connectVersion = connectVersionRef.current + 1;
    connectVersionRef.current = connectVersion;
    setConnectionStatus('connecting');
    if (!preserveState) {
      setLastError(null);
      setBusy(false);
    }
    clearReconnectTimer();
    closeSocket();
    if (!preserveState) {
      setRuntimeState(createInitialRuntimeState());
    }

    try {
      const summary = await llcApiClient.createSession(getStoredSessionId() ?? undefined);
      if (connectVersionRef.current !== connectVersion) {
        return;
      }
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(SESSION_STORAGE_KEY, summary.session_id);
      }
      setRuntimeState((prev) => applySessionBootstrap(prev, summary));

      const socket = new WebSocket(llcApiClient.toWebSocketUrl(summary.ws_path));
      socketRef.current = socket;

      socket.addEventListener('open', () => {
        if (connectVersionRef.current !== connectVersion) {
          return;
        }
        reconnectAttemptRef.current = 0;
        clearReconnectTimer();
        setConnectionStatus('connected');
      });

      socket.addEventListener('message', (messageEvent) => {
        if (connectVersionRef.current !== connectVersion) {
          return;
        }
        try {
          const parsed = JSON.parse(String(messageEvent.data));
          if (!isRuntimeEvent(parsed)) {
            return;
          }
          processEvent(parsed);
        } catch (error) {
          setLastError(error instanceof Error ? error.message : 'Failed to decode WebSocket payload.');
        }
      });

      socket.addEventListener('close', () => {
        if (connectVersionRef.current !== connectVersion) {
          return;
        }
        setConnectionStatus('disconnected');
        setBusy(false);
      });

      socket.addEventListener('error', () => {
        if (connectVersionRef.current !== connectVersion) {
          return;
        }
        setConnectionStatus('error');
        setBusy(false);
        setLastError('WebSocket connection error.');
      });
    } catch (error) {
      if (connectVersionRef.current !== connectVersion) {
        return;
      }
      setConnectionStatus('error');
      setLastError(error instanceof Error ? error.message : 'Failed to initialize session.');
    }
  }, [clearReconnectTimer, closeSocket, processEvent]);

  useEffect(() => {
    void connect({ preserveState: false });
    return () => {
      connectVersionRef.current += 1;
      clearReconnectTimer();
      closeSocket();
    };
  }, [clearReconnectTimer, closeSocket, connect]);

  useEffect(() => {
    if (connectionStatus === 'connected' || connectionStatus === 'connecting') {
      return;
    }
    if (reconnectTimerRef.current !== null) {
      return;
    }
    const exponent = Math.max(reconnectAttemptRef.current, 0);
    const delay = Math.min(RECONNECT_MAX_DELAY_MS, RECONNECT_BASE_DELAY_MS * (2 ** exponent));
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      void connect({ preserveState: true });
    }, delay);
  }, [connectionStatus, connect]);

  const sendAction = useCallback((payload: WebSocketAction): boolean => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setLastError('WebSocket is not connected.');
      return false;
    }
    socket.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendCommand = useCallback(
    (command: string): boolean => {
      const text = command.trim();
      if (!text) {
        return false;
      }
      const sent = sendAction({ action: 'send_message', text });
      if (sent) {
        setRuntimeState((prev) => appendUserCommand(prev, text));
      }
      return sent;
    },
    [sendAction]
  );

  const interrupt = useCallback((): boolean => sendAction({ action: 'interrupt' }), [sendAction]);

  return {
    runtimeState,
    connectionStatus,
    lastError,
    busy,
    sendCommand,
    interrupt,
    reconnect: () => connect({ preserveState: true }),
  };
}
