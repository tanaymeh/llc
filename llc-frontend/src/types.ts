export type AgentStatus = 'running' | 'waiting' | 'idle' | 'error';
export type ThemeMode = 'dark' | 'light';
export type MessageRole = 'USER' | 'AGENT' | 'SYSTEM';
export type MessageTone = 'default' | 'enabled' | 'disabled' | 'selection' | 'error';
export type LogLevel = 'info' | 'success' | 'error' | 'warning' | 'selection';
export type AgentMessageActivityKind = 'reasoning' | 'tool' | 'summary';

export interface AgentMessageActivity {
  kind: AgentMessageActivityKind;
  label: string;
  transitionKey: number;
}

export interface Agent {
  id: string;
  status: AgentStatus;
  isPast: boolean;
  currentTask: string;
  liveActivity: string;
  toolCallCount: number;
  progress: number;
}

export interface Message {
  id: string;
  timestamp: string;
  role: MessageRole;
  content: string;
  tone?: MessageTone;
  activity?: AgentMessageActivity;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  source: string;
  content: string;
  level: LogLevel;
}

export interface Team {
  id: string;
  name: string;
  agentIds: string[];
  status: 'active' | 'inactive';
}
