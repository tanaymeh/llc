import { useEffect, useState } from 'react';
import { ThemeMode } from '../types';
import { RuntimePhase } from '../runtime/state';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface HeaderProps {
  activeSection: 'agents' | 'teams';
  onSectionChange: (section: 'agents' | 'teams') => void;
  selectedModel: string;
  phase: RuntimePhase;
  sessionInputTokens: number;
  sessionOutputTokens: number;
  sessionCost: number;
  connectionStatus: ConnectionStatus;
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
}

export function Header({
  activeSection,
  onSectionChange,
  selectedModel,
  phase,
  sessionInputTokens,
  sessionOutputTokens,
  sessionCost,
  connectionStatus,
  theme,
  onThemeChange,
}: HeaderProps) {
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  const phaseTextColor = phase === 'WORKING'
    ? 'text-[var(--accent-success)]'
    : phase === 'SEARCHING'
      ? 'text-[var(--accent-warning)]'
      : phase === 'TOOLS'
        ? 'text-[var(--accent-selection)]'
      : 'theme-text-muted';
  const displayPhase = phase === 'TOOLS' ? 'SEARCHING' : phase === 'VERIFYING' ? 'WORKING' : phase;
  const phaseDotColor = phase === 'WORKING'
    ? 'bg-[var(--accent-success)]'
    : phase === 'SEARCHING'
      ? 'bg-[var(--accent-warning)]'
      : phase === 'TOOLS'
        ? 'bg-[var(--accent-selection)]'
      : 'bg-[var(--idle-dot)]';
  const connectionDotColor = connectionStatus === 'connected'
    ? 'bg-[var(--accent-success)]'
    : connectionStatus === 'connecting'
      ? 'bg-[var(--accent-warning)]'
      : 'bg-[var(--accent-error)]';
  const connectionLabel = connectionStatus.toUpperCase();
  const formatNumber = (value: number) => value.toLocaleString('en-US');

  return (
    <header className="theme-panel h-12 border-b theme-border flex items-center px-4 gap-4 transition-colors duration-300">
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 ${connectionDotColor}`} title={`WS: ${connectionLabel}`}></div>
        <span className="theme-text-primary text-xs tracking-wider">LLC</span>
      </div>

      <div className="theme-divider h-4 w-px"></div>

      <div className="flex items-center font-mono text-xs tracking-wider">
        <span className="theme-text-muted">T:</span>
        <span className="theme-text-primary">{formatTime(elapsedTime)}</span>
      </div>

      <div className="theme-divider h-4 w-px"></div>

      <div className="flex items-center gap-1">
        <span className="theme-text-muted text-xs tracking-wider">MODEL:</span>
        <span className="theme-text-primary text-xs tracking-wider">{selectedModel}</span>
      </div>

      <div className="theme-divider h-4 w-px"></div>

      <div className="flex items-center gap-2">
        <span className="theme-text-muted text-xs tracking-wider">TOKENS:</span>
        <div className="flex items-center gap-1 font-mono text-xs tracking-wider">
          <span className="theme-text-primary">{formatNumber(sessionInputTokens)}</span>
          <span className="text-[var(--accent-success)]">↑</span>
        </div>
        <span className="theme-text-muted text-xs tracking-wider">/</span>
        <div className="flex items-center gap-1 font-mono text-xs tracking-wider">
          <span className="theme-text-primary">{formatNumber(sessionOutputTokens)}</span>
          <span className="text-[var(--accent-warning)]">↓</span>
        </div>
        <span className="theme-text-muted text-xs tracking-wider">|</span>
        <div className="flex items-center gap-1 font-mono text-xs tracking-wider">
          <span className="theme-text-muted">$</span>
          <span className="theme-text-primary">{sessionCost.toFixed(4)}</span>
        </div>
      </div>

      <div className="theme-divider h-4 w-px"></div>

      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 ${phaseDotColor}`}></div>
        <span className="theme-text-secondary text-xs tracking-wider">PHASE:</span>
        <span className={`${phaseTextColor} text-xs tracking-wider`}>{displayPhase}</span>
      </div>

      <div className="ml-auto flex items-center gap-3">
        <button
          onClick={() => onSectionChange('agents')}
          className={`text-xs tracking-wider transition-colors ${
            activeSection === 'agents'
              ? 'theme-text-primary'
              : 'theme-text-muted hover:text-[var(--text-primary)]'
          }`}
        >
          AGENTS
        </button>
        <button
          onClick={() => onSectionChange('teams')}
          className={`text-xs tracking-wider transition-colors ${
            activeSection === 'teams'
              ? 'theme-text-primary'
              : 'theme-text-muted hover:text-[var(--text-primary)]'
          }`}
        >
          TEAMS
        </button>
        <div className="theme-segment flex items-center rounded-full border p-1 transition-colors duration-200">
          <button
            type="button"
            onClick={() => onThemeChange('dark')}
            className={`h-6 rounded-full px-2.5 text-[11px] tracking-[0.18em] transition-colors ${
              theme === 'dark' ? 'theme-segment-button-active' : 'theme-segment-button'
            }`}
          >
            DARK
          </button>
          <button
            type="button"
            onClick={() => onThemeChange('light')}
            className={`h-6 rounded-full px-2.5 text-[11px] tracking-[0.18em] transition-colors ${
              theme === 'light' ? 'theme-segment-button-active' : 'theme-segment-button'
            }`}
          >
            LIGHT
          </button>
        </div>
      </div>
    </header>
  );
}
