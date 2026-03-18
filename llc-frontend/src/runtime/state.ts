import { RuntimeEvent, SessionSummaryResponse } from '../api/contracts';
import {
  Agent,
  AgentMessageActivity,
  LogEntry,
  LogLevel,
  Message,
  MessageRole,
  MessageTone,
} from '../types';

export type RuntimePhase = 'WORKING' | 'IDLE' | 'SEARCHING' | 'TOOLS' | 'VERIFYING';

export interface RuntimeState {
  sessionId: string | null;
  selectedModel: string;
  availableModels: Array<{ value: string; description: string }>;
  subAgentModeEnabled: boolean;
  activeSubagents: number;
  maxSubAgents: number;
  phase: RuntimePhase;
  sessionInputTokens: number;
  sessionOutputTokens: number;
  sessionCost: number;
  messages: Message[];
  agents: Agent[];
  logs: LogEntry[];
  activeAgentMessageId: string | null;
  turnReasoningDurationMs: number;
  turnReasoningStartedAtMs: number | null;
  turnToolCallCount: number;
  activityTransitionKey: number;
}

const MAX_LOG_ENTRIES = 120;

const nowTimestamp = () => new Date().toLocaleTimeString('en-US', { hour12: false }).slice(0, 8);
const createId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
const nowMs = () => Date.now();

const appendMessage = (
  state: RuntimeState,
  role: MessageRole,
  content: string,
  tone?: MessageTone
): RuntimeState => ({
  ...state,
  messages: [
    ...state.messages,
    {
      id: createId(),
      timestamp: nowTimestamp(),
      role,
      content,
      tone,
    },
  ],
});

const appendLog = (
  state: RuntimeState,
  source: string,
  content: string,
  level: LogLevel = 'info'
): RuntimeState => {
  const last = state.logs[state.logs.length - 1];
  if (last && last.source === source && last.content === content && last.level === level) {
    return state;
  }
  return {
    ...state,
    logs: [
      ...state.logs,
      {
        id: createId(),
        timestamp: nowTimestamp(),
        source,
        content,
        level,
      },
    ].slice(-MAX_LOG_ENTRIES),
  };
};

const mapWorkerStatus = (statusRaw: string): Agent['status'] => {
  const status = statusRaw.toLowerCase();
  if (status === 'running' || status === 'restarting' || status === 'terminating') {
    return 'running';
  }
  if (status === 'failed' || status === 'stuck' || status === 'terminated') {
    return 'error';
  }
  if (status === 'completed') {
    return 'idle';
  }
  return 'waiting';
};

const isPastWorker = (statusRaw: string, currentActivityRaw: string): boolean => {
  const status = statusRaw.toLowerCase();
  if (status === 'completed' || status === 'failed' || status === 'stuck' || status === 'terminated') {
    return true;
  }
  const activity = currentActivityRaw.toLowerCase();
  return activity.includes('de-spawned') || activity.includes('de spawned');
};

const mapProgress = (statusRaw: string, outputChars: number, toolCalls: number): number => {
  const status = statusRaw.toLowerCase();
  if (status === 'completed' || status === 'terminated') {
    return 100;
  }
  if (status === 'failed' || status === 'stuck') {
    return 100;
  }
  if (status === 'running' || status === 'restarting' || status === 'terminating') {
    const byOutput = Math.floor(outputChars / 260);
    const byToolCalls = Math.max(Math.floor(toolCalls * 4), 0);
    return Math.min(95, Math.max(10, byOutput + byToolCalls));
  }
  if (status === 'waiting') {
    return 5;
  }
  return 0;
};

const compactText = (value: string, maxLength: number): string => {
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return '';
  }
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
};

const readStringArg = (args: Record<string, unknown>, keys: string[]): string => {
  for (const key of keys) {
    const raw = args[key];
    if (typeof raw === 'string' && raw.trim()) {
      return raw.trim();
    }
  }
  return '';
};

const compactPath = (rawPath: string): string => {
  const normalized = rawPath.replace(/\\/g, '/').trim();
  if (!normalized) {
    return '';
  }
  if (normalized.length <= 36) {
    return normalized;
  }
  const segments = normalized.split('/').filter(Boolean);
  if (segments.length >= 2) {
    const tail = segments.slice(-2).join('/');
    if (tail.length <= 30) {
      return `.../${tail}`;
    }
  }
  return compactText(normalized, 36);
};

const summarizeToolCall = (toolName: string, args: Record<string, unknown>): string => {
  const normalizedTool = toolName.toLowerCase();
  if (normalizedTool === 'read') {
    const filePath = readStringArg(args, ['file_path', 'path']);
    return filePath ? compactPath(filePath) : 'read file';
  }
  if (normalizedTool === 'write') {
    const filePath = readStringArg(args, ['file_path']);
    return filePath ? compactPath(filePath) : 'write file';
  }
  if (normalizedTool === 'ls') {
    const targetPath = readStringArg(args, ['path']);
    return targetPath ? compactPath(targetPath) : 'list directory';
  }
  if (normalizedTool === 'edit') {
    const filePath = readStringArg(args, ['file_path']);
    return filePath ? compactPath(filePath) : 'edit file';
  }
  if (normalizedTool === 'multiedit') {
    const filePath = compactPath(readStringArg(args, ['file_path'])) || 'edit file';
    const editsRaw = args.edits;
    const editCount = Array.isArray(editsRaw) ? editsRaw.length : 0;
    return editCount > 0 ? `${filePath} (${editCount} edits)` : filePath;
  }
  if (normalizedTool === 'bash') {
    const description = readStringArg(args, ['description']);
    if (description) {
      return compactText(description, 40);
    }
    const command = readStringArg(args, ['command']);
    return command ? compactText(command, 40) : 'run shell command';
  }
  if (normalizedTool === 'grep') {
    const pattern = readStringArg(args, ['pattern']);
    return pattern ? `pattern ${compactText(pattern, 30)}` : 'search text';
  }
  if (normalizedTool === 'glob') {
    const pattern = readStringArg(args, ['pattern']);
    return pattern ? compactText(pattern, 34) : 'match file paths';
  }
  if (normalizedTool === 'code_grep') {
    const pattern = readStringArg(args, ['pattern']);
    if (pattern) {
      return `ast ${compactText(pattern, 30)}`;
    }
    const rule = readStringArg(args, ['rule']);
    return rule ? `rule ${compactText(rule, 30)}` : 'structural search';
  }
  if (normalizedTool === 'websearch') {
    const query = readStringArg(args, ['query']);
    return query ? compactText(query, 36) : 'search web';
  }
  if (normalizedTool === 'webfetch') {
    const url = readStringArg(args, ['url']);
    if (!url) {
      return 'fetch page';
    }
    try {
      const parsed = new URL(url);
      return parsed.hostname || compactText(url, 36);
    } catch {
      return compactText(url, 36);
    }
  }
  if (normalizedTool === 'launchsubagent') {
    const task = readStringArg(args, ['task']);
    return task ? compactText(task, 36) : 'launch worker';
  }
  if (normalizedTool === 'getsubagentreport') {
    return 'worker status snapshot';
  }
  if (normalizedTool === 'waitsubagents') {
    return 'wait for workers';
  }
  if (normalizedTool === 'revisesubagent') {
    const targetId = readStringArg(args, ['subagent_id']);
    return targetId ? `revise ${compactText(targetId, 28)}` : 'revise worker';
  }
  if (normalizedTool === 'interruptsubagent') {
    const targetId = readStringArg(args, ['subagent_id']);
    return targetId ? `interrupt ${compactText(targetId, 28)}` : 'interrupt worker';
  }
  if (normalizedTool === 'terminatesubagent') {
    const targetId = readStringArg(args, ['subagent_id']);
    return targetId ? `terminate ${compactText(targetId, 28)}` : 'terminate worker';
  }
  if (normalizedTool === 'todowrite') {
    const todosRaw = args.todos;
    const todoCount = Array.isArray(todosRaw) ? todosRaw.length : 0;
    return todoCount > 0 ? `${todoCount} todo items` : 'update todo list';
  }
  if (normalizedTool === 'showdiff') {
    return 'render session diff';
  }
  return '';
};

const buildToolLogLine = (toolIndex: number, toolName: string, args: Record<string, unknown>): string => {
  const summary = summarizeToolCall(toolName, args);
  if (!summary) {
    return `[${toolIndex}] ${toolName}`;
  }
  return `[${toolIndex}] ${toolName} - ${summary}`;
};

const ensureActiveAgentMessage = (state: RuntimeState): [RuntimeState, string] => {
  if (state.activeAgentMessageId) {
    return [state, state.activeAgentMessageId];
  }
  const id = createId();
  const nextState: RuntimeState = {
    ...state,
    activeAgentMessageId: id,
    messages: [
      ...state.messages,
      {
        id,
        timestamp: nowTimestamp(),
        role: 'AGENT',
        content: '',
      },
    ],
  };
  return [nextState, id];
};

const applyToActiveAgentMessage = (
  state: RuntimeState,
  updater: (message: Message) => Message
): RuntimeState => {
  const [withMessage, activeId] = ensureActiveAgentMessage(state);
  return {
    ...withMessage,
    messages: withMessage.messages.map((message) => (
      message.id === activeId ? updater(message) : message
    )),
  };
};

const activeAgentActivity = (state: RuntimeState): AgentMessageActivity | undefined => {
  if (!state.activeAgentMessageId) {
    return undefined;
  }
  const activeMessage = state.messages.find((message) => message.id === state.activeAgentMessageId);
  if (!activeMessage || activeMessage.role !== 'AGENT') {
    return undefined;
  }
  return activeMessage.activity;
};

const setActiveAgentActivity = (
  state: RuntimeState,
  activity: AgentMessageActivity
): RuntimeState => applyToActiveAgentMessage(
  state,
  (message) => ({ ...message, activity })
);

const clearActiveAgentActivity = (state: RuntimeState): RuntimeState => (
  applyToActiveAgentMessage(
    state,
    (message) => ({ ...message, activity: undefined })
  )
);

const startReasoningWindow = (state: RuntimeState, startedAt: number): RuntimeState => {
  if (state.turnReasoningStartedAtMs !== null) {
    return state;
  }
  return {
    ...state,
    turnReasoningStartedAtMs: startedAt,
  };
};

const closeReasoningWindow = (state: RuntimeState, endedAt: number): RuntimeState => {
  const startedAt = state.turnReasoningStartedAtMs;
  if (startedAt === null) {
    return state;
  }
  return {
    ...state,
    turnReasoningDurationMs: state.turnReasoningDurationMs + Math.max(endedAt - startedAt, 0),
    turnReasoningStartedAtMs: null,
  };
};

const formatReasoningDuration = (durationMs: number): string => {
  const totalSeconds = Math.max(durationMs, 0) / 1000;
  if (totalSeconds < 10) {
    return `${totalSeconds.toFixed(1)}s`;
  }
  return `${Math.round(totalSeconds)}s`;
};

const formatTurnSummary = (reasoningDurationMs: number, toolCalls: number): string => {
  const normalizedReasoningMs = Math.max(reasoningDurationMs, 0);
  const normalizedToolCalls = Math.max(toolCalls, 0);
  const parts: string[] = [];
  if (normalizedReasoningMs > 0) {
    parts.push(`Reasoned for ${formatReasoningDuration(normalizedReasoningMs)}`);
  }
  if (normalizedToolCalls > 0) {
    parts.push(`${normalizedToolCalls} ${normalizedToolCalls === 1 ? 'tool' : 'tools'}`);
  }
  return parts.join(' · ');
};

const appendDeltaToActiveMessage = (state: RuntimeState, text: string): RuntimeState => {
  const [withMessage, activeId] = ensureActiveAgentMessage(state);
  return {
    ...withMessage,
    messages: withMessage.messages.map((message) =>
      message.id === activeId
        ? {
            ...message,
            content: `${message.content}${text}`,
          }
        : message
    ),
  };
};

const toneForCommandMessage = (message: string): { role: MessageRole; tone?: MessageTone; level: LogLevel } => {
  const normalized = message.toLowerCase();
  if (normalized.startsWith('now using ')) {
    return { role: 'SYSTEM', tone: 'selection', level: 'selection' };
  }
  if (normalized.includes('enabled `sub-agent-mode`') || normalized.includes('already enabled')) {
    return { role: 'SYSTEM', tone: 'enabled', level: 'success' };
  }
  if (
    normalized.startsWith('usage:')
    || normalized.includes('failed')
    || normalized.includes('disabled')
    || normalized.includes('unknown command')
    || normalized.includes('cannot')
    || normalized.includes('unavailable')
  ) {
    return { role: 'SYSTEM', tone: 'error', level: 'error' };
  }
  return { role: 'AGENT', level: 'info' };
};

const toModelSelectionOptions = (summary: SessionSummaryResponse) =>
  summary.available_models.map((model) => ({
    value: model.id,
    description: model.name,
  }));

export const createInitialRuntimeState = (): RuntimeState => ({
  sessionId: null,
  selectedModel: 'unknown',
  availableModels: [],
  subAgentModeEnabled: false,
  activeSubagents: 0,
  maxSubAgents: 5,
  phase: 'IDLE',
  sessionInputTokens: 0,
  sessionOutputTokens: 0,
  sessionCost: 0,
  messages: [],
  agents: [],
  logs: [],
  activeAgentMessageId: null,
  turnReasoningDurationMs: 0,
  turnReasoningStartedAtMs: null,
  turnToolCallCount: 0,
  activityTransitionKey: 0,
});

export const applySessionBootstrap = (
  state: RuntimeState,
  summary: SessionSummaryResponse
): RuntimeState => {
  let nextState: RuntimeState = {
    ...state,
    sessionId: summary.session_id,
    selectedModel: summary.model_name,
    availableModels: toModelSelectionOptions(summary),
    subAgentModeEnabled: summary.sub_agent_mode_enabled,
    activeSubagents: 0,
    maxSubAgents: Math.max(summary.max_sub_agents, 1),
  };
  nextState = appendLog(nextState, 'SYS', `Connected to session ${summary.session_id}`, 'success');
  return nextState;
};

export const appendUserCommand = (state: RuntimeState, command: string): RuntimeState => {
  let nextState = appendMessage(state, 'USER', command);
  nextState = appendLog(nextState, 'USER', command, 'info');
  return nextState;
};

export const applyRuntimeEvent = (state: RuntimeState, event: RuntimeEvent): RuntimeState => {
  if (event.type === 'turn_started') {
    let nextState: RuntimeState = {
      ...state,
      phase: 'WORKING',
      activeAgentMessageId: null,
      turnReasoningDurationMs: 0,
      turnReasoningStartedAtMs: null,
      turnToolCallCount: 0,
      activityTransitionKey: 0,
    };
    nextState = appendLog(nextState, 'TURN', `Turn started: ${event.turn_id}`, 'info');
    return nextState;
  }

  if (event.type === 'text_delta') {
    let nextState = closeReasoningWindow(state, nowMs());
    nextState = appendDeltaToActiveMessage(nextState, event.text);
    if (nextState.turnReasoningDurationMs > 0 || nextState.turnToolCallCount > 0) {
      const summaryLabel = formatTurnSummary(
        nextState.turnReasoningDurationMs,
        nextState.turnToolCallCount
      );
      if (!summaryLabel) {
        return {
          ...nextState,
          phase: 'WORKING',
        };
      }
      const currentActivity = activeAgentActivity(nextState);
      if (
        currentActivity?.kind !== 'summary'
        || currentActivity.label !== summaryLabel
      ) {
        const transitionKey = currentActivity?.kind === 'summary'
          ? nextState.activityTransitionKey
          : nextState.activityTransitionKey + 1;
        nextState = {
          ...nextState,
          activityTransitionKey: transitionKey,
        };
        nextState = setActiveAgentActivity(nextState, {
          kind: 'summary',
          label: summaryLabel,
          transitionKey,
        });
      }
    }
    return {
      ...nextState,
      phase: 'WORKING',
    };
  }

  if (event.type === 'reasoning_delta') {
    let nextState = startReasoningWindow(state, nowMs());
    const currentActivity = activeAgentActivity(nextState);
    const transitionKey = currentActivity?.kind === 'reasoning'
      ? nextState.activityTransitionKey
      : nextState.activityTransitionKey + 1;
    nextState = {
      ...nextState,
      activityTransitionKey: transitionKey,
    };
    nextState = setActiveAgentActivity(nextState, {
      kind: 'reasoning',
      label: 'Reasoning',
      transitionKey,
    });
    return {
      ...nextState,
      phase: 'SEARCHING',
    };
  }

  if (event.type === 'tool_call_started') {
    let nextState = closeReasoningWindow(state, nowMs());
    const nextToolCallCount = nextState.turnToolCallCount + 1;
    const transitionKey = nextState.activityTransitionKey + 1;
    nextState = {
      ...nextState,
      turnToolCallCount: nextToolCallCount,
      activityTransitionKey: transitionKey,
    };
    nextState = setActiveAgentActivity(nextState, {
      kind: 'tool',
      label: event.tool_name,
      transitionKey,
    });
    const toolLogLine = buildToolLogLine(event.tool_index, event.tool_name, event.args);
    nextState = appendLog(
      nextState,
      event.tool_name,
      toolLogLine,
      'info'
    );
    nextState = {
      ...nextState,
      phase: 'TOOLS',
    };
    return nextState;
  }

  if (event.type === 'tool_result') {
    if (!event.user_facing) {
      return state;
    }
    return appendLog(state, event.tool_name, event.content, 'info');
  }

  if (event.type === 'usage_update') {
    return {
      ...state,
      sessionInputTokens: Math.max(event.session_input_tokens, 0),
      sessionOutputTokens: Math.max(event.session_output_tokens, 0),
      sessionCost: Math.max(event.session_cost, 0),
    };
  }

  if (event.type === 'subagent_status') {
    const mappedAgents: Agent[] = event.workers.slice(0, 5).map((worker, index) => {
      const rawStatus = String(worker.status ?? 'waiting');
      const currentActivity = String(worker.current_activity ?? worker.activity_detail ?? 'idle');
      const status = mapWorkerStatus(rawStatus);
      const outputChars = Number(worker.output_chars ?? 0);
      const rawToolCalls = Number(worker.tool_calls ?? 0);
      const toolCallCount = Number.isFinite(rawToolCalls) ? Math.max(rawToolCalls, 0) : 0;
      const workerName = typeof worker.name === 'string' && worker.name.trim()
        ? worker.name.trim()
        : String(worker.id ?? `AGENT-${String(index + 1).padStart(2, '0')}`);
      return {
        id: workerName,
        status,
        isPast: isPastWorker(rawStatus, currentActivity),
        currentTask: String(worker.current_task ?? worker.goal ?? 'Awaiting assignment'),
        liveActivity: currentActivity,
        toolCallCount,
        progress: mapProgress(
          rawStatus,
          Number.isFinite(outputChars) ? outputChars : 0,
          toolCallCount
        ),
      };
    });
    return {
      ...state,
      agents: mappedAgents,
      activeSubagents: Math.max(event.active_count, 0),
      maxSubAgents: Math.max(event.max_sub_agents, 1),
    };
  }

  if (event.type === 'command_output') {
    let nextState = closeReasoningWindow(state, nowMs());
    const currentActivity = activeAgentActivity(nextState);
    if (currentActivity && currentActivity.kind !== 'summary') {
      nextState = clearActiveAgentActivity(nextState);
    }
    if (typeof event.data.model_name === 'string' && event.data.model_name.trim()) {
      nextState = {
        ...nextState,
        selectedModel: event.data.model_name,
      };
    }
    if (typeof event.data.sub_agent_mode_enabled === 'boolean') {
      nextState = {
        ...nextState,
        subAgentModeEnabled: event.data.sub_agent_mode_enabled,
      };
    }
    if (event.message) {
      const mapped = toneForCommandMessage(event.message);
      nextState = appendMessage(nextState, mapped.role, event.message, mapped.tone);
      nextState = appendLog(nextState, 'ORCHESTRATOR', event.message, mapped.level);
    }
    return {
      ...nextState,
      phase: 'IDLE',
      activeAgentMessageId: null,
    };
  }

  if (event.type === 'error') {
    let nextState = closeReasoningWindow(state, nowMs());
    const currentActivity = activeAgentActivity(nextState);
    if (currentActivity && currentActivity.kind !== 'summary') {
      nextState = clearActiveAgentActivity(nextState);
    }
    nextState = appendMessage(nextState, 'SYSTEM', event.message, 'error');
    nextState = appendLog(nextState, 'SYS', event.message, 'error');
    return {
      ...nextState,
      phase: 'IDLE',
      activeAgentMessageId: null,
    };
  }

  if (event.type === 'session_restored') {
    let nextState = appendMessage(
      state,
      'SYSTEM',
      `Session restored: ${event.session_id} (${event.message_count} messages)`,
      'selection'
    );
    nextState = appendLog(nextState, 'SYS', `Session restored: ${event.session_id}`, 'selection');
    return nextState;
  }

  if (event.type === 'turn_completed') {
    let nextState = closeReasoningWindow(state, nowMs());
    const currentActivity = activeAgentActivity(nextState);
    if (currentActivity && currentActivity.kind !== 'summary') {
      nextState = clearActiveAgentActivity(nextState);
    }
    nextState = {
      ...nextState,
      phase: 'IDLE' as RuntimePhase,
      activeAgentMessageId: null,
    };
    nextState = appendLog(
      nextState,
      'TURN',
      `Turn completed (in:${event.input_tokens} out:${event.output_tokens})`,
      'success'
    );
    if (event.compact_message) {
      nextState = appendLog(nextState, 'SYS', event.compact_message, 'warning');
    }
    return nextState;
  }

  return state;
};
