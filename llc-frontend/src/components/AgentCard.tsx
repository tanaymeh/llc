import { Agent, AgentStatus } from '../types';

interface AgentCardProps {
  agent: Agent;
}

const getStatusColor = (status: AgentStatus) => {
  switch (status) {
    case 'running': return 'bg-[var(--accent-success)]';
    case 'waiting': return 'bg-[var(--accent-warning)]';
    case 'idle': return 'bg-[var(--idle-dot)]';
    case 'error': return 'bg-[var(--accent-error)]';
  }
};

const getStatusTextColor = (status: AgentStatus) => {
  switch (status) {
    case 'running': return 'text-[var(--accent-success)]';
    case 'waiting': return 'text-[var(--accent-warning)]';
    case 'error': return 'text-[var(--accent-error)]';
    case 'idle': return 'theme-text-secondary';
  }
};

const getStatusText = (status: AgentStatus) => {
  switch (status) {
    case 'running': return 'ACTIVE';
    case 'waiting': return 'WAITING';
    case 'idle': return 'IDLE';
    case 'error': return 'ERROR';
  }
};

export function AgentCard({ agent }: AgentCardProps) {
  return (
    <div className="theme-panel-strong border theme-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="theme-text-primary text-xs tracking-wider">{agent.id}</span>
        <div className={`w-2 h-2 ${getStatusColor(agent.status)}`}></div>
        <span className={`text-xs tracking-wider ${getStatusTextColor(agent.status)}`}>
          {getStatusText(agent.status)}
        </span>
        {agent.isPast && (
          <span className="theme-text-muted text-[10px] tracking-[0.2em] ml-1">PAST</span>
        )}
        <span className="theme-text-muted ml-auto text-xs">
          CALLS: {agent.toolCallCount}
        </span>
      </div>

      <div className="space-y-2">
        <div>
          <span className="theme-text-muted text-xs tracking-wider">TASK:</span>
          <p className="theme-text-primary text-xs font-mono mt-1 leading-relaxed">
            {agent.currentTask}
          </p>
        </div>

        <div>
          <span className="theme-text-muted text-xs tracking-wider">ACTIVITY:</span>
          <p className="text-[var(--accent-success)] text-xs font-mono mt-1">
            {agent.liveActivity}
          </p>
        </div>

        <div className="pt-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="theme-text-muted text-xs tracking-wider">PROGRESS:</span>
            <span className="theme-text-primary text-xs">{agent.progress}%</span>
          </div>
          <div className="h-1 bg-[var(--surface-chip)] w-full">
            <div
              className="h-full bg-[var(--accent-success)] transition-all duration-500"
              style={{ width: `${agent.progress}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
}
