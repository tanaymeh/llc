# Local LangGraph Coding Agent

Minimal LangGraph + LangChain skeleton for a local coding assistant that:
- runs in a loop (REPL),
- accepts user commands,
- calls tools when needed (`shell`, `read_file`, `write_file`, `list_directory`),
- streams tool activity and final responses.

## Requirements

- Python `>=3.13`
- [`uv`](https://docs.astral.sh/uv/) installed
- One model provider API key (OpenAI or Anthropic)

## Setup

```bash
uv sync
cp .env.example .env
```

Set values in `.env`:
- `MODEL_NAME` (examples: `openai:gpt-4o-mini`, `anthropic:claude-sonnet-4-6`)
- `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (matching `MODEL_NAME`)

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
