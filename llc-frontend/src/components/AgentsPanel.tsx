import { useState } from 'react';

import { Agent } from '../types';
import { AgentCard } from './AgentCard';

interface AgentsPanelProps {
  agents: Agent[];
}

export function AgentsPanel({ agents }: AgentsPanelProps) {
  const [isActiveOpen, setIsActiveOpen] = useState(true);
  const [isPastOpen, setIsPastOpen] = useState(false);

  const activeAgents = agents.filter((agent) => !agent.isPast);
  const pastAgents = agents.filter((agent) => agent.isPast);

  if (agents.length === 0) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex-1 overflow-y-auto p-4">
          <div className="px-1 py-3 text-center border-b theme-border-subtle">
            <span className="theme-text-muted text-xs tracking-wider">NO AGENTS YET</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <section className="space-y-2">
          <button
            type="button"
            onClick={() => setIsActiveOpen((prev) => !prev)}
            className="w-full px-1 py-2 border-b theme-border-subtle flex items-center justify-between text-left transition-colors hover:theme-text-secondary"
          >
            <span className="theme-text-primary text-xs tracking-wider">ACTIVE AGENTS</span>
            <span className="theme-text-muted text-xs tracking-wider">
              {isActiveOpen ? '▼' : '▶'} {activeAgents.length}
            </span>
          </button>
          {isActiveOpen && (
            activeAgents.length > 0 ? (
              <div className="pt-1 space-y-3">
                {activeAgents.map((agent) => (
                  <AgentCard key={agent.id} agent={agent} />
                ))}
              </div>
            ) : (
              <div className="px-1 pt-2">
                <span className="theme-text-muted text-xs tracking-wider">NO ACTIVE AGENTS</span>
              </div>
            )
          )}
        </section>

        <section className="space-y-2">
          <button
            type="button"
            onClick={() => setIsPastOpen((prev) => !prev)}
            className="w-full px-1 py-2 border-b theme-border-subtle flex items-center justify-between text-left transition-colors hover:theme-text-secondary"
          >
            <span className="theme-text-primary text-xs tracking-wider">PAST AGENTS</span>
            <span className="theme-text-muted text-xs tracking-wider">
              {isPastOpen ? '▼' : '▶'} {pastAgents.length}
            </span>
          </button>
          {isPastOpen && (
            pastAgents.length > 0 ? (
              <div className="pt-1 space-y-3">
                {pastAgents.map((agent) => (
                  <AgentCard key={agent.id} agent={agent} />
                ))}
              </div>
            ) : (
              <div className="px-1 pt-2">
                <span className="theme-text-muted text-xs tracking-wider">NO PAST AGENTS</span>
              </div>
            )
          )}
        </section>
      </div>
    </div>
  );
}
