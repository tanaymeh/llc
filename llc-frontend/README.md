# LLC Frontend

React frontend for LLC mission control, built with React + TypeScript + Vite + Tailwind.

## What this app does

This app connects to the LLC backend API and renders:
- A session header with runtime metadata (timer, model, token totals, session cost, phase).
- A split workspace with:
- Main control area (`ORCHESTRATOR`, `AGENT CONTROL`, and `TEAMS` views).
- Execution log panel (resizable with mouse in `AGENTS` view).
- Command input with terminal-like behavior and command history navigation (`ArrowUp`/`ArrowDown`).
- Live orchestrator activity badges next to `[AGENT]` for reasoning, active tool name, and final per-turn reasoning/tool totals.

## Current UX Design Decisions

### Header
- Neutral colors are intentionally normalized:
- One white tone for primary text.
- One muted gray tone for labels/separators.
- Accents are used only for semantic states (phase/status, token arrows).
- `PHASE` reflects live backend turn/event state.
- Token/cost values reflect live `usage_update` events.

### Main Views
- Top-level switch:
- `AGENTS`: shows tabs (`ORCHESTRATOR`, `AGENT CONTROL`) + Execution Log side panel.
- `TEAMS`: shows teams panel only (no Execution Log panel).
- `AGENT CONTROL` tab displays:
- `ACTIVE: x/5` in the tab title.
- A status dot with priority: `error` (red) > `waiting` (amber) > any `running` (green) > neutral.

### Execution Log Panel
- Resizable by dragging the boundary between main panel and log panel.
- Visual separator kept minimal; resize handle is invisible overlay for cleaner aesthetic.

## Tech Stack

- React 18
- TypeScript 5
- Vite 5
- Tailwind CSS 3
- ESLint 9

## Project Structure

```text
src/
  App.tsx                         # Layout composition + runtime wiring
  index.css                       # Tailwind directives + base visual reset
  components/
    Header.tsx                    # Top metadata/status bar
    TabsPanel.tsx                 # Orchestrator/Agent Control tabs + teams route
    ChatPanel.tsx                 # Orchestrator message view
    AgentsPanel.tsx               # Agent slot list
    AgentCard.tsx                 # Per-agent card and status rendering
    TeamsPanel.tsx                # Teams view (mock)
    TerminalPanel.tsx             # Execution log
    CommandInput.tsx              # Bottom command entry/history
  types.ts                        # Shared type definitions
docs/
  agent-control-status-followups.md
```

## Run Locally

### Prerequisites
- Node.js LTS recommended (`20.x` or `22.x`)
- npm (comes with Node)
- LLC backend API running (`uv run llc` from repo root)

### Install

```bash
npm ci
```

### Start Dev Server

```bash
npm run dev
```

Open `http://localhost:5173`.

If backend runs on a different address, set:

```bash
VITE_LLC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

### Full Docker Stack (Backend + Frontend)

From repo root:

```bash
make run
```

Open:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

### Build / Verify

```bash
npm run typecheck
npm run build
npm run preview
```

## Scripts

- `npm run dev` - Start Vite dev server.
- `npm run build` - Build production bundle.
- `npm run preview` - Preview production build.
- `npm run lint` - Run ESLint.
- `npm run typecheck` - TypeScript type checks without emit.

## Mock Data and Future Integration

Current `TEAMS` view is still visual/placeholder.

See [docs/agent-control-status-followups.md](./docs/agent-control-status-followups.md) for recommended next schema/UI additions around approval-specific agent states.
