# Local LangGraph Coding Agent

Minimal LangGraph + LangChain skeleton for a local coding assistant that:
- runs in a loop (REPL),
- accepts user commands,
- calls tools when needed (`shell`, `read_file`, `write_file`, `list_directory`),
- streams tokens, tool activity, and final responses.

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
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional for custom OpenAI-compatible endpoints)

## Run

```bash
uv run main.py
```

Inside the CLI:
- type your request normally,
- type `exit` or `quit` to stop.

## Docker

Rebuild and launch the app in one command while mounting any host folder as the agent workspace:

```bash
./scripts/dev-docker.sh /path/to/folder
```

Notes:
- the mounted folder is available inside the container as `/workspace`,
- the app still reads credentials from the repo `.env`,
- rerun the same command after code changes to rebuild and launch again.

## Project Layout

```text
main.py            # REPL entrypoint
agent/state.py     # LangGraph state schema
agent/nodes.py     # LLM node, tool node, routing
agent/graph.py     # StateGraph builder + MemorySaver
agent/tools/       # Tool implementations
scripts/           # helper scripts, including Docker launcher
```
