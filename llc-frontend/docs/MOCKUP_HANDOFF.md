# LLC Frontend Mockup Handoff

## Purpose

This repository is a UI-first mockup for a coding-agent control surface. It is not wired to a real backend yet. The current implementation is intentionally local-state-driven so the layout, interaction model, colors, and panel behavior can be finalized before integration work begins.

The current mockup focuses on:

- A coding-agent dashboard shell
- A top system header
- An Orchestrator panel for high-level messages
- An Agent Control panel for agent cards
- An Execution Log panel
- A command input with slash-command behavior
- Dark and light themes with the same visual language

## Stack

- `Vite`
- `React 18`
- `TypeScript`
- `Tailwind CSS`

There is no router, no backend client in use, no global store, and no API layer currently connected to the UI flow.

## High-Level App Structure

The app entry point is:

- `src/main.tsx`

The main shell is:

- `src/App.tsx`

Component hierarchy:

1. `App`
2. `Header`
3. Main content area
4. `TabsPanel`
5. `TerminalPanel`
6. `CommandInput`

Inside `TabsPanel`:

- Orchestrator tab -> `ChatPanel`
- Agent Control tab -> `AgentsPanel` -> `AgentCard`

When the top-level workspace view is `teams`:

- `TabsPanel` renders `TeamsPanel`

## What Is Real vs Mock

Real in this repo:

- Visual structure
- Panel layout
- Theme system
- Slash-command UI behavior
- Keyboard navigation for `/model`
- Resizable execution log sidebar
- Mock command parsing and message/log generation

Mock only:

- Agents
- Progress updates
- Log stream
- All command execution
- Model selection persistence beyond local UI state
- Teams data
- Any connection to an actual coding agent runtime

## Core State in `src/App.tsx`

`App.tsx` is the current orchestration layer for the whole mockup. It owns:

- `workspaceView`
  - switches between `agents` and `teams`
- `theme`
  - `dark` or `light`
  - persisted in `localStorage` under `llc-ui-theme`
- `sidebarWidth`
  - width of the execution log panel
- `isResizingSidebar`
  - tracks drag state for the resize handle
- `selectedModel`
  - current header model label
- `featureToggles`
  - local object for `/enable`, `/disable`, `/toggle`
- `messages`
  - Orchestrator panel content
- `agents`
  - Agent Control panel cards
- `logs`
  - Execution log rows

If this project later gets a real backend, `App.tsx` is the place most likely to be split into:

- app shell state
- command dispatcher
- websocket or streaming state
- theme state

## Current Command System

All command parsing currently lives in:

- `src/App.tsx`
- `src/components/CommandInput.tsx`

### Supported commands

1. Toggle commands
   - `/enable <feature>`
   - `/disable <feature>`
   - `/toggle <feature>`

Behavior:

- Adds a `USER` message to Orchestrator
- Adds a `SYSTEM` message with tone `enabled` or `disabled`
- Adds a matching log line in Execution Log
- Updates `featureToggles` in local state only

2. Selection command
   - `/model`
   - `/model <query>`

Behavior:

- Opens an inline searchable option picker in `CommandInput`
- Filters by substring against `value` and `description`
- Supports `ArrowUp`, `ArrowDown`, `Enter`, `Escape`
- Selecting a model updates `selectedModel`
- Emits cyan selection messages/logs

3. Task command
   - `/subagent <task>`

Behavior:

- Adds a mock agent-spawn success message
- Adds a success log entry
- Does not actually create or manage a real worker

4. Unknown slash commands

Behavior:

- Adds an error-toned system message
- Adds an error log entry

### Where to modify command behavior

- Add/change command parsing:
  - `src/App.tsx`
- Add/change inline selection UI:
  - `src/components/CommandInput.tsx`
- Add/change selection options:
  - `selectionCommandOptions` in `src/App.tsx`

### Important note

The Teams page still exists visually, but `/team` is not part of the slash-command system anymore.

## Theme System

Theme implementation is centralized in:

- `src/index.css`

This file defines two token sets:

- `:root` for dark mode
- `:root[data-theme='light']` for light mode

The design is token-driven, not hardcoded per component. Components should prefer token-backed utility classes and helper classes instead of raw colors.

### Main token groups

- Background atmosphere
  - `--bg-app`
  - `--bg-glow-*`
  - `--grid-color`
- Surface layers
  - `--surface-panel-muted`
  - `--surface-panel`
  - `--surface-panel-strong`
  - `--surface-overlay`
- Borders
  - `--border-strong`
  - `--border-subtle`
- Text
  - `--text-primary`
  - `--text-secondary`
  - `--text-muted`
  - `--text-faint`
- Semantic accents
  - `--accent-success`
  - `--accent-warning`
  - `--accent-selection`
  - `--accent-error`

### Reusable theme utility classes

Defined in `src/index.css` under `@layer components`:

- `theme-panel-muted`
- `theme-panel`
- `theme-panel-strong`
- `theme-overlay-panel`
- `theme-border`
- `theme-border-subtle`
- `theme-divider`
- `theme-text-primary`
- `theme-text-secondary`
- `theme-text-muted`
- `theme-text-faint`
- `theme-input`
- `theme-tab-active`
- `theme-tab-inactive`
- `theme-chip`
- `theme-segment`
- `theme-segment-button`
- `theme-segment-button-active`

### Where to modify theme behavior

- Change palette values:
  - `src/index.css`
- Change theme toggle UI:
  - `src/components/Header.tsx`
- Change theme persistence key or initialization:
  - `src/App.tsx`

## Component Responsibilities

### `src/App.tsx`

Responsibilities:

- Owns all mock state
- Handles theme persistence
- Parses commands
- Generates fake messages/logs
- Simulates activity updates
- Manages sidebar resize behavior
- Composes the whole layout

If your friend starts integrating real functionality, this is the first file that should be refactored into smaller state domains.

### `src/components/Header.tsx`

Responsibilities:

- Top system bar
- Elapsed time counter
- Model label
- Token/cost mock display
- Phase indicator
- Workspace view switch (`AGENTS` / `TEAMS`)
- Theme switch (`DARK` / `LIGHT`)

Modify here if you want to change:

- Header layout
- Model display
- Top-bar metrics
- Theme toggle style or placement

### `src/components/TabsPanel.tsx`

Responsibilities:

- Secondary tab system for:
  - `ORCHESTRATOR`
  - `AGENT CONTROL`
- Status dot based on agent states
- Teams page routing inside the main content shell

Modify here if you want to:

- Add more tabs
- Change tab labels
- Change which tab is the default
- Change status-dot logic

### `src/components/ChatPanel.tsx`

Responsibilities:

- Renders Orchestrator messages
- Colors messages based on `role` and `tone`

Message color meaning:

- `USER` -> primary text
- `AGENT` -> success green
- `SYSTEM enabled` -> success green
- `SYSTEM disabled` -> warning amber
- `SYSTEM selection` -> selection cyan
- `SYSTEM error` -> error red

Modify here if you want richer message UI such as:

- badges
- icons
- grouped bubbles
- command chips
- expandable metadata

### `src/components/AgentsPanel.tsx`

Responsibilities:

- Creates five visible slots
- Maps each slot to an `AgentCard`
- Shows empty slots as `UNASSIGNED`

Modify here if you want:

- a dynamic number of visible slots
- grid layout instead of stacked cards
- filters or grouping

### `src/components/AgentCard.tsx`

Responsibilities:

- Displays one agent card
- Maps `AgentStatus` to status dot and text colors
- Renders task, activity, and progress

Modify here if you want:

- more agent metrics
- tool history
- per-agent controls
- richer progress bars

### `src/components/TerminalPanel.tsx`

Responsibilities:

- Renders the Execution Log
- Auto-scrolls to the newest log
- Colors log rows by `LogLevel`

Modify here if you want:

- filtering
- sticky timestamps
- grouping by source
- collapsible system noise

### `src/components/CommandInput.tsx`

Responsibilities:

- Text input and local command history
- `/model` parsing for inline picker mode
- substring filtering of model options
- keyboard navigation for picker and history
- auto-scroll of highlighted picker option

Modify here if you want:

- more slash-selection commands
- a full command palette
- autocomplete for toggles
- highlighted search matches
- mouse hover syncing with keyboard selection

Important detail:

- `parseSelectionInput()` currently only recognizes `/model`
- If future selection commands are reintroduced, update both:
  - `SelectionCommand`
  - `parseSelectionInput()`
  - `selectionCommandOptions` in `App.tsx`

### `src/components/TeamsPanel.tsx`

Responsibilities:

- Placeholder teams page
- Mock team cards
- Roadmap copy

This page is visual-only right now.

Modify here if you want:

- real team data
- creation flow
- team metrics
- expandable team detail

## Types

All shared UI data types are in:

- `src/types.ts`

Important types:

- `AgentStatus`
- `ThemeMode`
- `MessageRole`
- `MessageTone`
- `LogLevel`
- `Agent`
- `Message`
- `LogEntry`
- `Team`

If your friend changes message or log semantics, this file should be updated first.

## Mock Data Sources

There is no external data source yet. Current fake data is created in `App.tsx` via:

- initial `messages`
- initial `agents`
- initial `logs`
- a `setInterval()` that mutates running agents
- a `setInterval()` that appends random log rows

If this gets connected to a real backend, these are the main replacement points:

1. Replace the initial arrays with fetched or streamed data.
2. Remove the random interval simulation.
3. Replace `handleCommand()` with a real dispatcher.
4. Replace `applySelection()` with a real settings mutation.

## Where to Change Common Things

### Change available models in `/model`

Edit:

- `selectionCommandOptions` in `src/App.tsx`

Each option uses:

- `value`
- `description`

### Change the default selected model

Edit:

- `selectedModel` initial state in `src/App.tsx`

### Change theme colors

Edit:

- token values in `src/index.css`

Do not scatter raw color classes across components unless necessary. The current theme system is meant to keep dark and light mode aligned.

### Change the command parsing rules

Edit:

- `handleCommand()` in `src/App.tsx`

### Change how `/model` is parsed or filtered

Edit:

- `parseSelectionInput()` in `src/components/CommandInput.tsx`
- `filteredOptions` logic in `src/components/CommandInput.tsx`
- `findSelectionOption()` in `src/App.tsx`

### Change message colors

Edit:

- `getMessageLabelColor()` in `src/components/ChatPanel.tsx`
- `getMessageContentColor()` in `src/components/ChatPanel.tsx`

### Change log colors

Edit:

- `getLevelColor()` in `src/components/TerminalPanel.tsx`

### Change sidebar resize behavior

Edit:

- resize state and mouse listeners in `src/App.tsx`

## Integration Notes for a Real Backend

Recommended integration direction:

1. Keep the presentational components mostly intact.
2. Move command execution into a dedicated command service or hook.
3. Replace mock intervals with streamed updates.
4. Keep `types.ts` as the contract boundary, then evolve it as the backend contract becomes real.

Likely first refactor targets:

- Extract `useTheme()`
- Extract `useResizableSidebar()`
- Extract `useMockOrchestrator()` or a real `useOrchestratorStream()`
- Extract command parsing and dispatch from `App.tsx`

## Current UX Assumptions

- The shell is keyboard-friendly.
- Commands are typed into a terminal-like input.
- Orchestrator is for higher-level status.
- Execution Log is for lower-level technical events.
- Agent Control is a separate operational view.
- Theme identity should feel equivalent in dark and light mode, not merely inverted.

## Known Limitations

- No persistence for agent/log/message data
- No actual backend integration
- No actual team management
- No route-based navigation
- No mobile-specific layout tuning yet
- No tests

## Suggested Next Steps for Integration

1. Introduce a real command contract between frontend and backend.
2. Replace mock arrays with streaming state.
3. Decide whether command parsing stays client-side or moves server-side.
4. Convert header metrics from placeholders to live values.
5. Decide whether Teams becomes a real workspace mode or a separate route.

## Run Commands

- `npm run dev`
- `npm run build`
- `npm run typecheck`
- `npm run lint`

## Final Summary

This repo is currently a polished UI prototype with:

- a stable layout
- a stable visual language
- a stable theme system
- a clear component split
- mock command behavior that demonstrates the intended UX

Your friend should treat it as a design reference plus a partially structured shell, not as production app architecture. The visuals are ready to preserve; the state and command plumbing are the parts most likely to be swapped during real integration.
