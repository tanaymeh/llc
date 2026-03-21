# LLC

```text
 _      _       _____
| |    | |     / ____|
| |    | |    | |
| |    | |    | |
| |____| |____| |____
|______|______|\_____|
```

LLC is a local coding agent runtime with:

- a backend API service (default runtime),
- a React mission-control frontend (Git submodule at `llc-frontend/`),
- and a deprecated Textual TUI mode.

The backend runs a LangGraph agent loop, streams typed events, tracks token usage/cost, and persists sessions in SQLite.

## What it includes

- Backend API with WebSocket event streaming (`/api/ws/{session_id}`)
- React mission-control web UI (`llc-frontend/`)
- Markdown-rendered agent responses in mission-control UI (GFM tables, task lists, code blocks)
- Optional Mission-control TUI (`llc tui`)
- Slash commands (`/model`, `/compact`, `/enable sub-agent-mode`, `/subagent {TASK}`, `/help`)
- Built-in tools for shell, file edits, search (`rg`/glob), and web search/fetch
- Optional orchestrator + parallel worker sub-agent mode (max 5 workers)
- Optional Langfuse tracing for backend API turns, tools, and sub-agents
- Typed backend event contract (`llc/service/events.py`) consumed by the UI mapper

## Implementation overview

- `llc/main.py`: mode-aware entrypoint (`serve` default, `tui` optional)
- `llc/agent/`: LangGraph graph, nodes, tool collection, compaction, sub-agent runtime
- `llc/service/`: backend orchestration, stream adapter, API server, prompt registry
- `llc/ui/`: Textual app, panels, state models, telemetry mapping
- `llc/storage/`: SQLite schema and async persistence layer
- `llc/prompts/`: system/compact/runtime YAML prompts
- `llc-frontend/`: React frontend shell wired to live backend events

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/)
- OpenAI-compatible model endpoint and API key
- Node.js `20+` (for local frontend development)

For full tool coverage:

- `ripgrep` (`rg`) for text search
- `ast-grep` (`sg`/`ast-grep`) for structural code search (`code_grep` tool)

## Install from source

```bash
git submodule update --init --recursive
cp .env.example .env
uv sync --frozen
```

## Run backend API (default)

```bash
uv run llc
```

Override bind host/port:

```bash
uv run llc serve --host 0.0.0.0 --port 8000
```

## Run TUI mode

```bash
uv run llc tui
```

## Run local Langfuse (separate service)

Start local Langfuse in a separate stack:

```bash
make langfuse-up
```

Open `http://localhost:3000` and inspect traces while the LLC backend is running.

Useful commands:

```bash
make langfuse-logs
make langfuse-down
```

LLC backend target URL:

- host-run backend: `LANGFUSE_BASE_URL=http://127.0.0.1:3000`
- Dockerized backend: `LLC_DOCKER_LANGFUSE_BASE_URL=http://host.docker.internal:3000`

## Run frontend (separate service)

```bash
cd llc-frontend
npm ci
npm run dev
```

Frontend reads backend URL from `VITE_LLC_API_BASE_URL` (default: `http://127.0.0.1:8000`).
If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive` first.

## Run full stack in Docker (single command)

```bash
make run
```

This starts both containers:

- backend API at `http://localhost:8000`
- frontend dev server at `http://localhost:5173`

Use an external workspace mount if needed:

```bash
make run WORKSPACE=/path/to/project
```

## Install as a local app command

From the repository root:

```bash
uv tool install .
llc
```

To update after pulling new changes:

```bash
uv tool upgrade --reinstall .
```

Alternative:

```bash
python -m pip install .
llc
```

## Build distributables

```bash
uv build
```

Outputs are created in `dist/` (wheel + sdist).

## Configuration

Set values in `.env`:

- `MODEL_NAME` (default model for normal turns)
- `COMPACT_MODEL_NAME` (optional model for `/compact` and auto-compaction)
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (OpenAI-compatible endpoint; OpenRouter works here)
- `FIRECRAWL_API_KEY` (optional, for web tools)
- `LLC_LANGFUSE_ENABLED` (`true`/`false`; defaults to enabled when keys are set)
- `LANGFUSE_BASE_URL` (for host-run backend, e.g. `http://127.0.0.1:3000`)
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LLC_DOCKER_LANGFUSE_BASE_URL` (Docker backend override, default `http://host.docker.internal:3000`)
- `SUB_AGENT_MODE_ENABLED` (`true` to boot in sub-agent mode)
- `MAX_SUB_AGENTS` (hard-capped to `5`)
- `SUB_AGENT_REPORT_INTERVAL_S`
- `SUB_AGENT_MAX_RUNTIME_S`
- `SUB_AGENT_CONTEXT_MESSAGES`
- `LLC_DB_PATH` (default: `<workspace>/.llc/sessions.db`)
- `LLC_PROMPTS_DIR` (default: `llc/prompts`)
- `LLC_API_HOST` (default: `127.0.0.1`)
- `LLC_API_PORT` (default: `8000`)
- `LLC_API_ALLOWED_ORIGINS` (comma-separated list; default includes Vite localhost origins)
- `LLC_API_SUBAGENT_REPORT_INTERVAL_S` (default: `1.0`)

Local Langfuse stack variables (used by `docker-compose.langfuse.yml`) are also in `.env.example`, including:

- `LANGFUSE_INIT_*` bootstrap variables (org/project/user + API keys)
- `NEXTAUTH_SECRET`, `SALT`, `ENCRYPTION_KEY`
- `POSTGRES_*`, `CLICKHOUSE_*`, `MINIO_*`, `REDIS_AUTH`

## TUI controls

- `Enter` (or `Ctrl+Enter`) send message
- `Esc` twice quickly interrupts active turn + active workers
- `Ctrl+G` toggle agents pane
- `Ctrl+L` focus logs
- `Ctrl+A` focus agents
- `Ctrl+E` focus execution
- `Ctrl+U` overview
- `Ctrl+Q` quit

## Commands

- `/help`
- `/model <model_id>`
- `/compact`
- `/enable sub-agent-mode`
- `/subagent {TASK}`
- `exit` / `quit`

## Docker

Full stack (backend + frontend) in Docker:

```bash
make run
```

Stop stack:

```bash
make down
```

Tail logs:

```bash
make logs
```

Run local Langfuse UI stack separately:

```bash
make langfuse-up
```

Stop local Langfuse stack:

```bash
make langfuse-down
```

Tail local Langfuse logs:

```bash
make langfuse-logs
```

Backend-only helper (no frontend):

```bash
./scripts/dev-docker.sh /path/to/project
```
