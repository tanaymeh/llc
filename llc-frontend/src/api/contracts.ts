export interface AvailableModel {
  id: string;
  name: string;
  prompt_price: number | null;
  completion_price: number | null;
  context_length: number | null;
}

export interface SessionSummaryResponse {
  session_id: string;
  model_name: string;
  sub_agent_mode_enabled: boolean;
  max_sub_agents: number;
  available_models: AvailableModel[];
  ws_path: string;
}

export interface SessionRecordResponse {
  id: string;
  title: string;
  model_name: string;
  sub_agent_mode: boolean;
  created_at: number;
  updated_at: number;
}

export interface TurnStartedEvent {
  type: 'turn_started';
  turn_id: string;
}

export interface TextDeltaEvent {
  type: 'text_delta';
  text: string;
}

export interface ReasoningDeltaEvent {
  type: 'reasoning_delta';
  text: string;
}

export interface ToolCallStartedEvent {
  type: 'tool_call_started';
  tool_call_id: string;
  tool_name: string;
  tool_index: number;
  args: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: 'tool_result';
  tool_call_id: string;
  tool_name: string;
  content: string;
  user_facing: boolean;
  render_mode: string;
}

export interface UsageUpdateEvent {
  type: 'usage_update';
  turn_input_tokens: number;
  turn_output_tokens: number;
  session_input_tokens: number;
  session_output_tokens: number;
  session_cost: number;
}

export interface SubagentStatusEvent {
  type: 'subagent_status';
  workers: Array<Record<string, unknown>>;
  active_count: number;
  max_sub_agents: number;
}

export interface TurnCompletedEvent {
  type: 'turn_completed';
  input_tokens: number;
  output_tokens: number;
  compact_message: string | null;
}

export interface CommandOutputEvent {
  type: 'command_output';
  message: string | null;
  should_exit: boolean;
  data: Record<string, unknown>;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export interface SessionRestoredEvent {
  type: 'session_restored';
  session_id: string;
  message_count: number;
}

export type RuntimeEvent =
  | TurnStartedEvent
  | TextDeltaEvent
  | ReasoningDeltaEvent
  | ToolCallStartedEvent
  | ToolResultEvent
  | UsageUpdateEvent
  | SubagentStatusEvent
  | TurnCompletedEvent
  | CommandOutputEvent
  | ErrorEvent
  | SessionRestoredEvent;

export type WebSocketAction =
  | {
      action: 'send_message';
      text: string;
    }
  | {
      action: 'interrupt';
    }
  | {
      action: 'restore_session';
      session_id: string;
    };
