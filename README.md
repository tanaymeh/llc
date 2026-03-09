# LLC

Local LangGraph + LangChain coding agent that:
- runs in a full-screen Textual TUI with chat panels and a fixed bottom composer,
- accepts user commands,
- calls tools when needed (shell, read/write files, grep, edit, ShowDiff, etc.),
- streams tokens with live markdown rendering and code block styling,
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

Inside the TUI:
- type in the bottom composer,
- press `Enter` to insert a newline,
- press `Ctrl+Enter` to submit (or `Ctrl+J` as a universal terminal fallback),
- use `Ctrl+N` as an additional newline shortcut,
- use `exit` or `quit` to stop, `Ctrl+Q` to quit immediately.

## UI Features

- **ASCII banner** with model info, token usage, and cost at the top
- **Chat transcript** with scrollable conversation history
- **Fixed bottom composer** that stays visible like a chat app input
- **Role-specific message panels** (User and Agent with model name)
- **Tool call and reasoning traces** shown inline in agent messages
- **Send lock while streaming** (typing remains enabled but sending is disabled)
- **Markdown and code rendering** tuned for readability in dark and light themes
- **Word-level editing**: `Ctrl+Backspace` delete word, `Ctrl+Left/Right` word navigation

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
- terminal color support (`TERM`, `COLORTERM`) is forwarded for better TUI rendering,
- `ripgrep` (`rg`) is available inside the container,
- `ast-grep` (`ast-grep`) is available inside the container for `code_grep`,
- the script still runs `docker build` on every launch (quiet mode only hides build logs),
- rerun the same command after code changes to rebuild and launch again.

## Project Layout

```text
main.py            # TUI entrypoint
agent/state.py     # LangGraph state schema
agent/nodes.py     # LLM node, tool node, routing
agent/graph.py     # StateGraph builder + MemorySaver
agent/tools/       # Tool implementations
ui/display.py      # Message / tool parsing helpers for streamed chunks
ui/repl.py         # Textual app, chat workflow, streaming workers
ui/repl.tcss       # Theme-aware styles for chat panels and composer
ui/token_tracker.py # Token usage and cost tracking
models.py          # Model list fetching + pricing from OpenRouter
scripts/           # helper scripts, including Docker launcher
```
