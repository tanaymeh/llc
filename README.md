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
- and a React mission-control frontend (Git submodule at `llc-frontend/`).

The backend runs a LangGraph agent loop, streams typed events, tracks token usage/cost, and persists conversations in SQLite.

## What it includes

- Backend API with WebSocket event streaming (`/api/ws/{conversation_id}`)
- React mission-control web UI (`llc-frontend/`)
- Markdown-rendered agent responses in mission-control UI (GFM tables, task lists, code blocks)
- Slash commands (`/model`, `/compact`, `/enable sub-agent-mode`, `/subagent {TASK}`, `/help`)
- Built-in tools for shell, file edits, search (`rg`/glob), and web search/fetch
- Ordered post-turn hooks with explicit `blocking` opt-in (background by default)
- Async `hook_update` runtime events plus expiring mission-control hook reminders
- Optional orchestrator + parallel worker sub-agent mode (max 5 workers, process-isolated)
- Sub-agent coordination layer with inbox messaging, shared notes, team status reads, and file lock leases
- Optional Langfuse tracing with one canonical UUID4 conversation id, live sub-agent traces, and persisted orchestrator/sub-agent transcripts
- Typed backend event contract (`llc/service/events.py`) consumed by the UI mapper

## Implementation overview

- `llc/main.py`: backend API entrypoint (`serve` optional for backward compatibility)
- `llc/agent/`: LangGraph graph, nodes, tool collection, compaction, sub-agent runtime
- `llc/agent/subagents/`: runtime facade + worker runner + reporting/usage helpers
- `llc/service/engine.py`: conversation orchestration facade
- `llc/service/engine_*.py`: hook runtime, streaming runtime, and persistence helpers
- `llc/service/api.py`: API app composition root
- `llc/service/api_*.py`: engine manager, HTTP routes, websocket flow, and API models
- `llc/storage/`: SQLite schema and async persistence layer
- `llc/prompts/`: `prompt_manifest.yaml` plus Jinja prompt templates
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

If an older clone still has the submodule cached with the previous SSH URL, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

If `git pull --recurse-submodules` fails because `llc-frontend/` already exists as a normal directory, move it out of the way first, then re-run the commands above.

## Run backend API (default)

```bash
uv run llc
```

Override bind host/port:

```bash
uv run llc serve --host 0.0.0.0 --port 8000
```

## Run local Langfuse (separate service)

Start local Langfuse in a separate stack:

```bash
make langfuse-up
```

Open `http://localhost:3000` and inspect traces while the LLC backend is running. The local stack is pinned to Langfuse `3.163.0`; the Python SDK is pinned to `langfuse==4.0.6`.
If `.env` omits `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `LANGFUSE_INIT_PROJECT_SECRET_KEY`, `make langfuse-up` will automatically reuse `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` so the local bootstrap project matches the LLC backend client config.

Useful commands:

```bash
make langfuse-logs
make langfuse-down
```

If the backend logs a Langfuse `401 Invalid credentials` warning, the base URL is reachable but the configured project keys do not match the running Langfuse stack. Update `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env`, or recreate the local stack and bootstrap project with:

```bash
docker compose -f docker-compose.langfuse.yml down -v
make langfuse-up
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
If you cloned without `--recurse-submodules`, run `git submodule sync --recursive && git submodule update --init --recursive` first.

## Run full stack in Docker (single command)

```bash
make run
```

This starts both containers:

- backend API at `http://localhost:8000`
- frontend dev server at `http://localhost:5173`
- backend defaults to `SUB_AGENT_DEBUG_LOGGING=true` in compose, so sub-agent coordination/tool logs are visible in `make run` output

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
- `COMPACT_MODEL_NAME` (optional model for `/compact`, auto-compaction, and tool-output summarization)
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
- `SUB_AGENT_STALL_TIMEOUT_S` (legacy fallback; prefer the two below)
- `SUB_AGENT_LLM_STALL_TIMEOUT_S` (watchdog: max seconds waiting for LLM response before marking stuck; default 45)
- `SUB_AGENT_TOOL_STALL_TIMEOUT_S` (watchdog: max seconds waiting for tool execution before marking stuck; default 300)
- `SUB_AGENT_STOP_GRACE_S` (watchdog grace after stop request before marked stuck)
- `SUB_AGENT_MAX_TOOL_CALLS` (loop-budget cap per worker attempt; internal coordination and sub-agent control tools do not consume this budget)
- `SUB_AGENT_REQUIRE_TOOL_CALL` (default `false`; when `true`, marks worker result as failed if it tries to complete without any tool use)
- `SUB_AGENT_WAIT_TIMEOUT_MS` (default timeout for `WaitSubagents`; use `0` for unbounded)
- `SUB_AGENT_CONTEXT_MESSAGES`
- `SUB_AGENT_TEAM_STATUS_INTERVAL_CYCLES` (reserved; no runtime-forced reads by default)
- `SUB_AGENT_SHARED_NOTES_INTERVAL_CYCLES` (reserved; no runtime-forced reads by default)
- `SUB_AGENT_LOCK_DEFAULT_LEASE_S` (default: `120`; lock lease duration in seconds)
- `SUB_AGENT_LOCK_RENEW_S` (lease extension duration for `KEEP` lock-review actions)
- `SUB_AGENT_LOCK_NEAR_EXPIRY_S` (threshold for near-expiry lock review signal)
- `SUB_AGENT_SHARED_NOTES_MAX_ENTRIES` (bounded retained shared notes)
- `SUB_AGENT_INBOX_READ_MAX` (max messages per coordination inbox read)
- `SUB_AGENT_DEBUG_LOGGING` (prints detailed sub-agent runtime/coordination/tool activity logs when enabled)
- `LLC_DB_PATH` (default: `<workspace>/.llc/sessions.db`; stores canonical conversations and legacy session aliases during migration)
- `LLC_PROMPTS_DIR` (default: `llc/prompts`)
- `LLC_API_HOST` (default: `127.0.0.1`)
- `LLC_API_PORT` (default: `8000`)
- `LLC_API_ALLOWED_ORIGINS` (comma-separated list; default includes Vite localhost origins)
- `LLC_API_SUBAGENT_REPORT_INTERVAL_S` (default: `1.0`)

Custom prompt directories must include a `prompt_manifest.yaml` file and the referenced `.jinja` templates.

At startup, LLC appends a project snapshot inside the system prompt `<env>` block: current directory name, a bounded workspace tree, git-repo status, and the latest 10 commits across local and remote refs when applicable.

Local Langfuse stack variables (used by `docker-compose.langfuse.yml`) are also in `.env.example`, including:

- `LANGFUSE_INIT_*` bootstrap variables (org/project/user + API keys)
- `NEXTAUTH_SECRET`, `SALT`, `ENCRYPTION_KEY`
- `POSTGRES_*`, `CLICKHOUSE_*`, `MINIO_*`, `REDIS_AUTH`

## UI controls

Mission-control web UI:

- `Esc` twice quickly interrupts the active turn and any running workers

TUI:

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

## Sub-agent coordination behavior

- Sub-agents can coordinate with dedicated tools (`SendMessage`, `ReadInbox`, `ReadTeamStatus`, `ReadSharedNotes`, `AppendSharedNote`).
- Runtime-forced coordination is limited to unread inbox pulls (`ReadInbox`) and lock review checks (`ReviewHeldLocks`).
- Worker tool budgets count user-task tools only; internal coordination tools and sub-agent control tools are tracked separately and do not consume `SUB_AGENT_MAX_TOOL_CALLS`.
- `SendMessage` has an anti-spam guard: after 5 consecutive messages to the same teammate within 2 minutes, further sends are blocked for 120s with a system-ping response showing remaining cooldown.
- File lock leases are available via `RequestLock`, `ReleaseLock`, `ReviewHeldLocks`, and `RespondLockReview`.
- Mutating file tools (`Write`, `Edit`, `MultiEdit`, rewrite-mode `code_grep`) are lock-aware in sub-agent mode.
- Bash is still available to sub-agents, but prompt policy strictly forbids using Bash to edit or delete files.
- Launches now run a model-availability preflight. If `MODEL_NAME` is not available on the current provider, `LaunchSubagent`/`/subagent` returns a clear error instead of spawning a worker that fails immediately.
- With `SUB_AGENT_DEBUG_LOGGING=true`, backend logs include sub-agent lifecycle events, tool invocations, forced coordination reads, inter-agent messages, lock lease decisions, and successful file-mutation events (`file_mutation_applied`).

## Docker

Full stack (backend + frontend) in Docker:

```bash
make run
```

The backend Docker image installs `uv` from PyPI during the build, so it no longer depends on `ghcr.io/astral-sh/uv` being reachable.

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
