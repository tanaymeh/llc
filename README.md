# Local LangGraph Coding Agent

Minimal LangGraph + LangChain skeleton for a local coding assistant that:
- runs in a loop (REPL),
- accepts user commands,
- calls tools when needed (`shell`, `read_file`, `write_file`, `list_directory`),
- streams tool activity and final responses.

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

## Project Layout

```text
main.py            # REPL entrypoint
agent/state.py     # LangGraph state schema
agent/nodes.py     # LLM node, tool node, routing
agent/graph.py     # StateGraph builder + MemorySaver
agent/tools/       # Tool implementations
```
