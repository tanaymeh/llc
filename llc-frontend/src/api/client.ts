import { SessionRecordResponse, SessionSummaryResponse } from './contracts';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

const normalizeBaseUrl = (value: string) => value.replace(/\/+$/, '');

const resolveBaseUrl = () => {
  const raw = import.meta.env.VITE_LLC_API_BASE_URL as string | undefined;
  if (!raw || !raw.trim()) {
    return DEFAULT_API_BASE_URL;
  }
  return normalizeBaseUrl(raw.trim());
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Request failed (${response.status}): ${text || response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export class LlcApiClient {
  readonly baseUrl: string;

  constructor(baseUrl = resolveBaseUrl()) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  async health(): Promise<{ ok: boolean }> {
    return fetchJson<{ ok: boolean }>(`${this.baseUrl}/api/health`);
  }

  async createSession(sessionId?: string): Promise<SessionSummaryResponse> {
    const payload = sessionId ? { session_id: sessionId } : {};
    return fetchJson<SessionSummaryResponse>(`${this.baseUrl}/api/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  }

  async getSession(sessionId: string): Promise<SessionSummaryResponse> {
    return fetchJson<SessionSummaryResponse>(`${this.baseUrl}/api/sessions/${encodeURIComponent(sessionId)}`);
  }

  async listSessions(): Promise<SessionRecordResponse[]> {
    return fetchJson<SessionRecordResponse[]>(`${this.baseUrl}/api/sessions`);
  }

  toWebSocketUrl(wsPath: string): string {
    const base = new URL(this.baseUrl);
    const protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
    const normalizedPath = wsPath.startsWith('/') ? wsPath : `/${wsPath}`;
    return `${protocol}//${base.host}${normalizedPath}`;
  }
}

export const llcApiClient = new LlcApiClient();
