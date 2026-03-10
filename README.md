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
- `COMPACT_MODEL_NAME` (optional; defaults to `MODEL_NAME` for `/compact` and auto-compaction summaries)
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional for custom OpenAI-compatible endpoints)

Edit the assistant system prompt in `llc/prompts/system_prompt.yaml`.
At runtime, LLC appends current environment metadata to the very end of the system prompt inside `<env>...</env>` tags.

## Run

```bash
uv run llc
```

Or equivalently:

```bash
python -m llc
```

Inside the TUI:
- type in the bottom composer,
- press `Ctrl+Enter` (or `Enter` on terminals that collapse `Ctrl+Enter`) to submit,
- use `Ctrl+N` or `Ctrl+O` to insert a newline in the composer,
- use `/compact` to summarize and replace the oldest chat history once the model-visible history has more than 5 messages,
- use `exit` or `quit` to stop, `Ctrl+Q` to quit immediately.

## UI Features

- **ASCII banner** with model info, token usage, and cost at the top
- **Chat transcript** with scrollable conversation history
- **Fixed bottom composer** that stays visible like a chat app input
- **Role-specific message panels** (User and Agent with model name)
- **Tool call and reasoning traces** shown inline in agent messages
- **User-facing tool outputs** rendered inline in chat bubbles
- **ShowDiff visual rendering** with side-by-side colored old/new columns in TUI (session baseline vs current state)
- **Chat compaction** via `/compact`, plus automatic compaction when the last prompt reaches 90% of the active model context length
- **Send lock while streaming** (typing remains enabled but sending is disabled)
- **Markdown and code rendering** tuned for readability in dark and light themes
- **Word-level editing**: `Ctrl+Backspace` delete word, `Ctrl+Left/Right` word navigation

## Docker

Rebuild and launch the app in one command while mounting any host folder as the agent workspace:

```bash
make run
```

`make run` uses the quiet Docker build mode and mounts the sibling `../llm-transpiler` project by default.
Override the mounted workspace when needed:

```bash
make run WORKSPACE=/path/to/folder
```

You can still call the Docker helper directly:

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
llc/                     # All source code
├── main.py              # Entrypoint
├── config.py            # Settings, prompt loading, env details
├── models.py            # Model list fetching + pricing from OpenRouter
├── agent/
│   ├── graph.py         # StateGraph builder + MemorySaver
│   ├── llm.py           # Chat model factory (OpenAI / OpenRouter)
│   ├── nodes.py         # LLM node, tool node, routing
│   ├── state.py         # LangGraph state schema
│   ├── compact.py       # History compaction logic
│   ├── hooks.py         # Hook protocol, TokenCounterHook, AutoCompactHook
│   ├── message_utils.py # Shared message text extraction
│   └── tools/           # Tool implementations (Read, Write, Edit, Bash, Grep, etc.)
├── commands/            # REPL slash-commands (/model, /compact, /help, exit)
├── ui/
│   ├── repl.py          # Textual app, chat workflow, streaming
│   ├── widgets.py       # ChatBubble, ComposerInput, ModelPickerScreen
│   ├── rendering.py     # Diff rendering, tool output formatting
│   ├── display.py       # Message / tool parsing helpers for streamed chunks
│   └── repl.tcss        # Theme-aware styles for chat panels and composer
└── prompts/             # System and compaction prompt YAML files
```
