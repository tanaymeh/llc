# LLC

Local LangGraph + LangChain coding agent that:
- runs in a REPL with a rich terminal UI (ASCII banner, adaptive colors, syntax-highlighted markdown),
- accepts user commands,
- calls tools when needed (shell, read/write files, grep, edit, ShowDiff, etc.),
- streams tokens with live markdown rendering and code syntax highlighting,
- tracks token usage and estimates cost via OpenRouter pricing.

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

Edit the assistant system prompt in `prompts/system_prompt.yaml`.
At runtime, LLC appends current environment metadata to the very end of the system prompt inside `<env>...</env>` tags.

## Run

```bash
uv run main.py
```

Inside the CLI:
- type your request normally,
- type `exit` or `quit` to stop.

## UI Features

- **ASCII art banner** with model info on startup
- **Inline markdown rendering** of assistant responses with syntax-highlighted code blocks
- **Live streaming** with real-time markdown re-rendering as tokens arrive
- **Persistent top-right usage HUD** with cumulative input/output tokens and total cost
- **Cost estimation** from OpenRouter model pricing (when available)
- **Session summary** showing total tokens and cost on exit
- **Single rolling spinner line** for tool execution status

## Docker

Rebuild and launch the app in one command while mounting any host folder as the agent workspace:

```bash
./scripts/dev-docker.sh /path/to/folder
```

The default mode keeps Docker build output quiet so you land directly in the app chat screen.
Use debug mode only when you want full Docker logs:

```bash
./scripts/dev-docker.sh --debug /path/to/folder
# or
LOCAL_CLAUDE_DOCKER_DEBUG=1 ./scripts/dev-docker.sh /path/to/folder
```

Notes:
- the mounted folder is available inside the container as `/workspace`,
- the app still reads credentials from the repo `.env`,
- `ripgrep` (`rg`) is available inside the container,
- `ast-grep` (`ast-grep`) is available inside the container for `code_grep`,
- the script still runs `docker build` on every launch (quiet mode only hides build logs),
- rerun the same command after code changes to rebuild and launch again.

## Project Layout

```text
main.py            # REPL entrypoint
agent/state.py     # LangGraph state schema
agent/nodes.py     # LLM node, tool node, routing
agent/graph.py     # StateGraph builder + MemorySaver
agent/tools/       # Tool implementations
ui/display.py      # Rich-based banner, panels, markdown rendering
ui/repl.py         # REPL loop with streaming + token tracking
ui/token_tracker.py # Token usage and cost tracking
ui/colors.py       # ANSI escape codes (for prompt_toolkit prompt)
models.py          # Model list fetching + pricing from OpenRouter
scripts/           # helper scripts, including Docker launcher
```
