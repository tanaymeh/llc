import { useState } from 'react';
import { Agent, Message } from '../types';
import { ChatPanel } from './ChatPanel';
import { AgentsPanel } from './AgentsPanel';
import { TeamsPanel } from './TeamsPanel';

interface TabsPanelProps {
  messages: Message[];
  agents: Agent[];
  view: 'agents' | 'teams';
}

export function TabsPanel({ messages, agents, view }: TabsPanelProps) {
  const [activeTab, setActiveTab] = useState<'orchestrator' | 'agentControl'>('agentControl');
  const currentAgents = agents.filter((agent) => !agent.isPast);
  const activeAgentCount = currentAgents.filter((agent) => agent.status === 'running').length;
  const hasErroredAgent = currentAgents.some((agent) => agent.status === 'error');
  const hasApprovalPendingAgent = currentAgents.some((agent) => agent.status === 'waiting');

  const statusDotClass = hasErroredAgent
    ? 'bg-[var(--accent-error)]'
    : hasApprovalPendingAgent
      ? 'bg-[var(--accent-warning)]'
      : activeAgentCount > 0
        ? 'bg-[var(--accent-success)]'
        : 'bg-[var(--idle-dot)]';

  if (view === 'teams') {
    return (
      <div className="flex flex-col h-full">
        <TeamsPanel />
      </div>
    );
  }

  const tabs = [
    { id: 'orchestrator', label: 'ORCHESTRATOR' },
    { id: 'agentControl', label: 'AGENT CONTROL' },
  ] as const;

  return (
    <div className="theme-panel flex flex-col h-full border-r theme-border transition-colors duration-300">
      <div className="h-10 border-b theme-border flex items-center">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 h-full border-r theme-border text-xs tracking-wider transition-colors flex items-center gap-2 ${
              activeTab === tab.id
                ? 'theme-tab-active border-b-2'
                : 'theme-tab-inactive'
            }`}
          >
            {tab.id === 'agentControl' ? (
              <>
                <div className={`w-2 h-2 rounded-full ${statusDotClass}`}></div>
                <span>{tab.label}</span>
                <span className="theme-text-muted">ACTIVE: {activeAgentCount}/5</span>
              </>
            ) : (
              tab.label
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-hidden">
        {activeTab === 'orchestrator' && <ChatPanel messages={messages} />}
        {activeTab === 'agentControl' && <AgentsPanel agents={agents} />}
      </div>
    </div>
  );
}
