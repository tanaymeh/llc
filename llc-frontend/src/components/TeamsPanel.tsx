import { Team } from '../types';
import { useState } from 'react';

export function TeamsPanel() {
  const [teams] = useState<Team[]>([
    {
      id: 'team-alpha',
      name: 'TEAM-ALPHA',
      agentIds: ['AGENT-01', 'AGENT-02', 'AGENT-03'],
      status: 'inactive',
    },
    {
      id: 'team-beta',
      name: 'TEAM-BETA',
      agentIds: ['AGENT-04', 'AGENT-05'],
      status: 'inactive',
    },
  ]);

  return (
    <div className="flex flex-col h-full">
      <div className="h-10 border-b theme-border flex items-center px-4">
        <span className="theme-text-muted text-xs tracking-wider">TEAMS (COMING SOON)</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className="theme-panel-muted border theme-border-subtle p-4 space-y-3">
          <p className="theme-text-muted text-xs tracking-wider mb-4">
            TEAM MANAGEMENT SYSTEM
          </p>
          {teams.map((team) => (
            <div key={team.id} className="theme-panel-strong border theme-border-subtle p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="theme-text-primary text-xs tracking-wider">{team.name}</span>
                <div className="w-2 h-2 bg-[var(--idle-dot)]"></div>
                <span className="theme-text-muted text-xs tracking-wider">
                  {team.agentIds.length} AGENTS
                </span>
              </div>
              <div className="flex gap-2 flex-wrap">
                {team.agentIds.map((agentId) => (
                  <span
                    key={agentId}
                    className="theme-chip text-xs font-mono px-2 py-1"
                  >
                    {agentId}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="theme-panel-muted border theme-border-subtle p-4 theme-text-muted text-xs leading-relaxed space-y-2">
          <p>TEAMS FEATURE ROADMAP:</p>
          <p className="theme-text-faint">
            - Group agents by project/domain
          </p>
          <p className="theme-text-faint">
            - Assign shared context and dependencies
          </p>
          <p className="theme-text-faint">
            - Monitor team-wide metrics and progress
          </p>
          <p className="theme-text-faint">
            - Enable inter-agent communication
          </p>
        </div>
      </div>
    </div>
  );
}
