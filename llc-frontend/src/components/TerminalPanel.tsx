import { LogEntry } from '../types';
import { useEffect, useRef } from 'react';

interface TerminalPanelProps {
  logs: LogEntry[];
}

export function TerminalPanel({ logs }: TerminalPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'success': return 'text-[var(--accent-success)]';
      case 'error': return 'text-[var(--accent-error)]';
      case 'warning': return 'text-[var(--accent-warning)]';
      case 'selection': return 'text-[var(--accent-selection)]';
      default: return 'theme-text-secondary';
    }
  };

  return (
    <div className="theme-panel flex flex-col h-full border-l theme-border transition-colors duration-300">
      <div className="h-10 border-b theme-border flex items-center px-4">
        <span className="theme-text-primary text-xs tracking-wider">EXECUTION LOG</span>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-2 bg-[var(--surface-panel-muted)] transition-colors duration-200"
      >
        {logs.map((log) => (
          <div key={log.id} className="grid grid-cols-[auto_auto_minmax(0,1fr)] gap-x-2 items-start font-mono text-xs">
            <span className="theme-text-faint leading-relaxed whitespace-nowrap">{log.timestamp}</span>
            <span className="theme-text-muted leading-relaxed [overflow-wrap:anywhere]">
              [{log.source}]
            </span>
            <span className={`${getLevelColor(log.level)} leading-relaxed whitespace-pre-wrap [overflow-wrap:anywhere]`}>
              {log.content}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
