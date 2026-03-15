# LLC

```text
 _      _       _____
| |    | |     / ____|
| |    | |    | |
| |    | |    | |
| |____| |____| |____
|______|______|\_____|
```

LLC is a local coding agent application with a full-screen Textual TUI.
It runs a LangGraph-based agent loop, streams tool/model events live, tracks token usage and cost, and stores sessions in SQLite.

## What it includes

- Mission-control TUI (execution lane, logs, agents pane, command surface)
- Slash commands (`/model`, `/compact`, `/enable sub-agent-mode`, `/subagent {TASK}`, `/help`)
- Built-in tools for shell, file edits, search (`rg`/glob), web search/fetch, and diff rendering
- Optional orchestrator + parallel worker sub-agent mode (max 5 workers)
- Typed backend event contract (`llc/service/events.py`) consumed by the UI mapper

## Implementation overview

- `llc/main.py`: entrypoint wiring (`Settings` -> `SessionStore` -> `SessionEngine` -> `MissionControlApp`)
- `llc/agent/`: LangGraph graph, nodes, tool collection, compaction, sub-agent runtime
- `llc/service/`: backend orchestration, stream-to-event adapter, prompt registry
- `llc/ui/`: Textual app, panels, state models, telemetry mapping
- `llc/storage/`: SQLite schema and async persistence layer
- `llc/prompts/`: system/compact/runtime YAML prompts

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/)
- OpenAI-compatible model endpoint and API key

For full tool coverage:

- `ripgrep` (`rg`) for text search
- `ast-grep` (`sg`/`ast-grep`) for structural code search (`code_grep` tool)

## Install and run from source

```bash
cp .env.example .env
uv sync --frozen
uv run llc
```

You can also run:

```bash
python -m llc
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
- `SUB_AGENT_MODE_ENABLED` (`true` to boot in sub-agent mode)
- `MAX_SUB_AGENTS` (hard-capped to `5`)
- `SUB_AGENT_REPORT_INTERVAL_S`
- `SUB_AGENT_MAX_RUNTIME_S`
- `SUB_AGENT_CONTEXT_MESSAGES`
- `LLC_DB_PATH` (default: `<workspace>/.llc/sessions.db`)
- `LLC_PROMPTS_DIR` (default: `llc/prompts`)

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
- `/model <model_id>` (or `/model` to open inline selector)
- `/compact`
- `/enable sub-agent-mode`
- `/subagent {TASK}`
- `exit` / `quit`

## Docker

Build and run the app in Docker, mounting a target workspace into `/workspace`:

```bash
make run
```

Override mounted directory:

```bash
make run WORKSPACE=/path/to/project
```

Direct script:

```bash
./scripts/dev-docker.sh /path/to/project
```
