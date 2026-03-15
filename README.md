# LLC

Local LangGraph + LangChain coding agent that:
- runs in a full-screen mission-control Textual interface (not chat bubbles),
- accepts direct instructions and slash commands from a terminal-native command surface,
- calls tools when needed (shell, read/write/edit files, grep/glob, ShowDiff, web tools),
- streams model output, tool execution, worker status, and logs live,
- tracks token usage and estimated cost (including sub-agent usage),
- stores sessions locally in SQLite,
- exposes typed backend events so other frontends can consume the same runtime.

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/) installed
- OpenAI-compatible API key (any provider exposing OpenAI-style inference API)

## Setup

```bash
uv sync
cp .env.example .env
```

Set values in `.env`:
- `MODEL_NAME` (example: `gpt-4o-mini`, `deepseek/deepseek-r1`)
- `COMPACT_MODEL_NAME` (optional; defaults to `MODEL_NAME` for `/compact` and auto-compaction summaries)
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional for custom OpenAI-compatible endpoints)
- `SUB_AGENT_MODE_ENABLED` (optional; `true` to boot directly in experimental sub-agent mode)
- `MAX_SUB_AGENTS` (optional; hard-capped at `5`)
- `SUB_AGENT_REPORT_INTERVAL_S` (optional; progress heartbeat interval in seconds)
- `SUB_AGENT_MAX_RUNTIME_S` (optional; max runtime per sub-agent attempt in seconds)
- `SUB_AGENT_CONTEXT_MESSAGES` (optional; number of recent messages to include as sub-agent context)
- `LLC_DB_PATH` (optional; defaults to `<workspace>/.llc/sessions.db`)
- `LLC_PROMPTS_DIR` (optional; defaults to `llc/prompts/`)

Edit the assistant system prompt in `llc/prompts/system_prompt.yaml`.
At runtime, LLC appends current environment metadata to the end of the system prompt inside `<env>...</env>` tags.

## Run

```bash
uv run llc
```

Or:

```bash
python -m llc
```

## Mission Control Interface

The interface is organized into persistent operational regions:

- **Top bar**: model, phase, health, token/cost totals, sub-agent counts, uptime.
- **Execution domain**: tool timeline + mission lane (role-tagged `USER` / `AGENT` / `SYS` message flow).
- **Agents pane**: compact nested orchestrator/sub-agent telemetry rows.
- **Log band**: dense categorized stream (`SYS`, `TOOL`, `CODE`, `OBS`, `WARN`, `ERR`, etc.).
- **Command surface**: anchored terminal prompt for instructions and slash commands.

### Controls

- `Enter` to submit from the command surface.
- `Ctrl+Enter` (and `Ctrl+J`/`Ctrl+M`) also trigger send.
- `Ctrl+G` toggles agents pane visibility.
- `Ctrl+L` toggles focused-log mode.
- `Ctrl+A` focuses agents mode.
- `Ctrl+E` focuses execution mode.
- `Ctrl+U` returns to overview mode.
- `Esc` twice quickly interrupts the active flow (current turn + active sub-agents).
- `Ctrl+Q` quits immediately.

### Commands

- `/help` shows commands.
- `/model <model_id>` switches model directly.
- `/model` opens inline model selection in the command surface (non-modal).
- `/compact` summarizes and replaces older model-visible history.
- `/enable sub-agent-mode` enables orchestrator/worker behavior.
- `/subagent {TASK}` manually spawns one worker (strict braces required).
- `exit` or `quit` exits.

## UI Features

- Mission-control visual language (hard edges, thin separators, restrained accent colors).
- Pydantic-backed UI state and telemetry mapping (`BaseModel`, validators, computed fields).
- Live markdown/code rendering in execution lane.
- Tool call timeline and user-facing tool output rendering.
- Side-by-side diff rendering for `ShowDiff`.
- Categorized, timestamped, source-attributed logs.
- Inline model selector (no popup picker).
- Bounded log/tool buffers with incremental updates for steady runtime performance.

## Backend Service Layer

LLC separates orchestration/backend logic from the Textual frontend:

- `llc/service/engine.py` owns turn execution, command dispatch, hooks, and sub-agent runtime management.
- `llc/service/stream_adapter.py` converts LangGraph stream chunks into typed event models.
- `llc/service/events.py` defines the backend event contract (Pydantic).
- `llc/ui/app.py` consumes backend events and renders mission-control regions.
- `llc/ui/state.py` + `llc/ui/telemetry_mapper.py` normalize event data into UI state.
- `llc/storage/store.py` persists sessions/messages/token usage/events in SQLite.

This keeps frontend concerns isolated while preserving a single backend runtime contract.

## Experimental Sub-Agent Mode

Sub-agent mode is experimental and opt-in.

- Enable with `/enable sub-agent-mode` (or `SUB_AGENT_MODE_ENABLED=true` before launch).
- Manual spawn syntax is strict: `/subagent {TASK}`.
- Successful `/subagent` launches trigger orchestrator follow-up so completion does not require manual polling.
- Workers are isolated and do not communicate with each other.
- Max active workers is enforced by `MAX_SUB_AGENTS` and capped at `5`.
- Worker telemetry appears in the persistent agents pane and updates live.
- Orchestrator receives live worker snapshots each turn.
- Worker report payloads are bounded to protect context length.

Detailed design/status: [Parallel Sub-Agent Implementation Report](PARALLEL_SUBAGENT_IMPLEMENTATION_REPORT.md).

## Docker

Rebuild and launch in one command while mounting a host workspace:

```bash
make run
```

Override mounted workspace:

```bash
make run WORKSPACE=/path/to/folder
```

Direct script usage:

```bash
./scripts/dev-docker.sh /path/to/folder
```

Debug build output:

```bash
./scripts/dev-docker.sh --debug /path/to/folder
# or
LOCAL_CLAUDE_DOCKER_DEBUG=1 ./scripts/dev-docker.sh /path/to/folder
```

Notes:
- mounted folder is available as `/workspace`,
- app still reads credentials from repo `.env`,
- `TERM` and `COLORTERM` are forwarded for better terminal rendering,
- `rg` and `ast-grep` are available in the container,
- `docker build` still runs on each launch.

## Project Layout

```text
llc/                     # All source code
├── main.py              # Entrypoint (SessionEngine + SessionStore + MissionControlApp)
├── config.py            # Settings, prompt loading, env details, DB path
├── models.py            # Model list fetching + pricing from OpenRouter
├── agent/               # LangGraph runtime, nodes, tools, sub-agent runtime
├── commands/            # Slash commands (/model, /compact, /enable, /subagent, /help, exit)
├── service/
│   ├── engine.py        # Frontend-agnostic backend orchestrator
│   ├── events.py        # Typed backend event models
│   ├── stream_adapter.py# LangGraph chunk -> event translation
│   └── prompt_registry.py
├── storage/
│   ├── models.py        # Persisted entity records
│   ├── schema.py        # SQLite schema + indexes
│   └── store.py         # Async session store
├── ui/
│   ├── app.py           # Mission-control Textual app
│   ├── app.tcss         # Mission-control style system
│   ├── state.py         # Pydantic UI state models
│   ├── telemetry_mapper.py
│   ├── theme.py         # Theme registration/palettes
│   ├── rendering.py     # Diff and tool-output render helpers
│   └── panels/          # Top bar, execution, agents, logs, command surface
└── prompts/             # System/compact + mode/runtime prompt YAML files
```
