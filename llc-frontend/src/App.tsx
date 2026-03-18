import { useEffect, useMemo, useState } from 'react';
import { Header } from './components/Header';
import { TabsPanel } from './components/TabsPanel';
import { TerminalPanel } from './components/TerminalPanel';
import { CommandInput, SelectionCommand, SelectionOption } from './components/CommandInput';
import { ThemeMode } from './types';
import { useLlcSession } from './runtime/useLlcSession';

type WorkspaceView = 'agents' | 'teams';
const THEME_STORAGE_KEY = 'llc-ui-theme';

const getInitialTheme = (): ThemeMode => {
  if (typeof window === 'undefined') {
    return 'dark';
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === 'dark' || storedTheme === 'light') {
    return storedTheme;
  }

  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
};

function App() {
  const { runtimeState, connectionStatus, sendCommand } = useLlcSession();
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>('agents');
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const initialTheme = getInitialTheme();
    if (typeof document !== 'undefined') {
      document.documentElement.dataset.theme = initialTheme;
    }
    return initialTheme;
  });
  const [sidebarWidth, setSidebarWidth] = useState(380);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const selectionCommandOptions = useMemo<Record<SelectionCommand, SelectionOption[]>>(
    () => ({
      model: runtimeState.availableModels,
    }),
    [runtimeState.availableModels]
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (!isResizingSidebar) {
      return;
    }

    const handleMouseMove = (event: MouseEvent) => {
      const minWidth = 260;
      const maxWidth = Math.max(320, window.innerWidth - 320);
      const nextWidth = window.innerWidth - event.clientX;
      const clampedWidth = Math.min(maxWidth, Math.max(minWidth, nextWidth));
      setSidebarWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsResizingSidebar(false);
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizingSidebar]);

  const handleSelectionCommand = (command: SelectionCommand, option: SelectionOption) => {
    sendCommand(`/${command} ${option.value}`);
  };

  const handleCommand = (rawCommand: string) => {
    const command = rawCommand.trim();
    if (!command) {
      return;
    }
    sendCommand(command);
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-transparent theme-text-primary transition-colors duration-300">
      <Header
        activeSection={workspaceView}
        onSectionChange={setWorkspaceView}
        selectedModel={runtimeState.selectedModel}
        phase={runtimeState.phase}
        sessionInputTokens={runtimeState.sessionInputTokens}
        sessionOutputTokens={runtimeState.sessionOutputTokens}
        sessionCost={runtimeState.sessionCost}
        connectionStatus={connectionStatus}
        theme={theme}
        onThemeChange={setTheme}
      />

      {workspaceView === 'teams' ? (
        <div className="flex-1 overflow-hidden">
          <TabsPanel messages={runtimeState.messages} agents={runtimeState.agents} view={workspaceView} />
        </div>
      ) : (
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 min-w-0 overflow-hidden">
            <TabsPanel messages={runtimeState.messages} agents={runtimeState.agents} view={workspaceView} />
          </div>
          <div className="relative shrink-0 h-full" style={{ width: sidebarWidth }}>
            <div
              onMouseDown={(event) => {
                event.preventDefault();
                setIsResizingSidebar(true);
              }}
              className="absolute left-0 top-0 -ml-1 w-2 h-full cursor-col-resize z-10 bg-gradient-to-r from-transparent via-[var(--border-strong)] to-transparent opacity-70"
              role="separator"
              aria-label="Resize execution log panel"
              aria-orientation="vertical"
            />
            <TerminalPanel logs={runtimeState.logs} />
          </div>
        </div>
      )}

      <CommandInput
        onSubmit={handleCommand}
        selectionCommands={selectionCommandOptions}
        onSelectCommand={handleSelectionCommand}
      />
    </div>
  );
}

export default App;
