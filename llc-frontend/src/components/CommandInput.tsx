import { useState, useEffect, useRef, useMemo } from 'react';

export type SelectionCommand = 'model';

export interface SelectionOption {
  value: string;
  description: string;
}

interface CommandInputProps {
  onSubmit: (command: string) => void;
  selectionCommands: Record<SelectionCommand, SelectionOption[]>;
  onSelectCommand: (command: SelectionCommand, option: SelectionOption) => void;
}

const parseSelectionInput = (value: string) => {
  const match = value.match(/^\/(model)(?:\s+(.*))?$/i);
  if (!match) {
    return null;
  }

  return {
    command: match[1].toLowerCase() as SelectionCommand,
    query: (match[2] ?? '').trim().toLowerCase(),
  };
};

export function CommandInput({ onSubmit, selectionCommands, onSelectCommand }: CommandInputProps) {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [selectionIndex, setSelectionIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const optionRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const parsedSelection = useMemo(() => parseSelectionInput(input), [input]);
  const filteredOptions = useMemo(() => {
    if (!parsedSelection) {
      return [];
    }

    const query = parsedSelection.query;
    const options = selectionCommands[parsedSelection.command];
    if (!query) {
      return options;
    }

    return options.filter((option) => {
      const valueMatch = option.value.toLowerCase().includes(query);
      const descriptionMatch = option.description.toLowerCase().includes(query);
      return valueMatch || descriptionMatch;
    });
  }, [parsedSelection, selectionCommands]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    setSelectionIndex(0);
  }, [input]);

  useEffect(() => {
    if (!parsedSelection || filteredOptions.length === 0) {
      return;
    }

    optionRefs.current[selectionIndex]?.scrollIntoView({
      block: 'nearest',
    });
  }, [selectionIndex, filteredOptions, parsedSelection]);

  const runPlainCommand = (command: string) => {
    onSubmit(command);
    setHistory((prev) => [...prev, command]);
    setInput('');
    setHistoryIndex(-1);
    setSelectionIndex(0);
  };

  const selectOption = (command: SelectionCommand, option: SelectionOption) => {
    const fullCommand = `/${command} ${option.value}`;
    onSelectCommand(command, option);
    setHistory((prev) => [...prev, fullCommand]);
    setInput('');
    setHistoryIndex(-1);
    setSelectionIndex(0);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    if (!trimmedInput) {
      return;
    }

    if (parsedSelection && filteredOptions.length > 0) {
      const selectedOption = filteredOptions[Math.min(selectionIndex, filteredOptions.length - 1)];
      selectOption(parsedSelection.command, selectedOption);
      return;
    }

    runPlainCommand(trimmedInput);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (parsedSelection) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (filteredOptions.length > 0) {
          setSelectionIndex((prev) => (prev + 1) % filteredOptions.length);
        }
        return;
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (filteredOptions.length > 0) {
          setSelectionIndex((prev) => (prev - 1 + filteredOptions.length) % filteredOptions.length);
        }
        return;
      }

      if (e.key === 'Escape') {
        e.preventDefault();
        setInput('');
        setHistoryIndex(-1);
        setSelectionIndex(0);
        return;
      }
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (history.length > 0) {
        const newIndex = historyIndex === -1 ? history.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setInput(history[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex >= 0) {
        const newIndex = historyIndex + 1;
        if (newIndex >= history.length) {
          setHistoryIndex(-1);
          setInput('');
        } else {
          setHistoryIndex(newIndex);
          setInput(history[newIndex]);
        }
      }
    }
  };

  return (
    <div className="theme-panel h-12 border-t theme-border flex items-center px-4 gap-2 transition-colors duration-300">
      <span className="text-[var(--accent-success)] text-sm font-mono">$</span>
      <form onSubmit={handleSubmit} className="flex-1">
        <div className="relative">
          {parsedSelection && (
            <div className="theme-overlay-panel absolute bottom-full mb-2 left-0 right-0 border border-[color:var(--accent-selection-border)] shadow-lg">
              <div className="px-3 py-2 border-b theme-border-subtle flex items-center justify-between text-[11px] font-mono">
                <span className="text-[var(--accent-selection)]">/{parsedSelection.command}</span>
                <span className="theme-text-muted">Substring search: type to filter</span>
              </div>

              <div className="max-h-44 overflow-y-auto">
                {filteredOptions.length > 0 ? (
                  filteredOptions.map((option, index) => (
                    <button
                      key={`${parsedSelection.command}-${option.value}`}
                      type="button"
                      ref={(element) => {
                        optionRefs.current[index] = element;
                      }}
                      onMouseDown={(event) => {
                        event.preventDefault();
                        selectOption(parsedSelection.command, option);
                      }}
                      className={`w-full text-left px-3 py-2 border-b theme-border-subtle last:border-b-0 transition-colors ${
                        index === selectionIndex ? 'bg-[var(--accent-selection-soft)]' : 'hover:bg-[var(--surface-hover)]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3 font-mono text-xs">
                        <span className={index === selectionIndex ? 'text-[var(--accent-selection)]' : 'theme-text-primary'}>
                          {option.value}
                        </span>
                        {index === selectionIndex && <span className="text-[var(--accent-selection)]">ENTER</span>}
                      </div>
                      <p className="theme-text-muted text-[11px] font-mono mt-1">{option.description}</p>
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-4 theme-text-muted text-xs font-mono">
                    No matches for "{parsedSelection.query}".
                  </div>
                )}
              </div>

              <div className="px-3 py-2 border-t theme-border-subtle text-[11px] font-mono theme-text-muted">
                Use ↑/↓ to navigate, Enter to select, Esc to cancel
              </div>
            </div>
          )}

          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            className="theme-input w-full bg-transparent text-sm font-mono outline-none"
            placeholder="enter command (e.g., /enable debug-mode, /model, /subagent fix auth bug)"
            spellCheck={false}
          />
        </div>
      </form>
    </div>
  );
}
