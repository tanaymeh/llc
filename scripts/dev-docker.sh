#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/dev-docker.sh [--debug] [--port PORT] [target-dir]

Options:
  --debug, -d   Show full Docker build output.
  --port        API port to publish (default: 8000).
  --help, -h    Show this help message.

Examples:
  ./scripts/dev-docker.sh
  ./scripts/dev-docker.sh --port 8000
  ./scripts/dev-docker.sh /path/to/workspace
  ./scripts/dev-docker.sh --debug /path/to/workspace

Notes:
  This helper runs backend API only in Docker.
  For full backend + frontend stack, use: `make run`.
  If you cloned without `--recurse-submodules`, initialize `llc-frontend/` first.
  Session history persists in SQLite at /workspace/.llc/sessions.db by default.
  Override DB location with `LLC_DB_PATH` in .env if needed.
  Local Langfuse defaults to http://host.docker.internal:3000 in Docker.
  Sub-agent watchdog/timing knobs are read from `.env` (`SUB_AGENT_*`).
  Sub-agents run as isolated worker processes.
  Tool-output summarization runs post-turn in background and is applied before next turn.
EOF
}

DEBUG=0
PORT="${LLC_API_PORT:-8000}"
if [[ "${LLC_DOCKER_DEBUG:-0}" == "1" ]]; then
  DEBUG=1
fi

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug|-d)
      DEBUG=1
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --port" >&2
        usage >&2
        exit 1
      fi
      PORT="$2"
      shift
      ;;
    --port=*)
      PORT="${1#*=}"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      ;;
  esac
  shift
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Invalid --port value: $PORT" >&2
  exit 1
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
  echo "Expected at most one target directory argument." >&2
  usage >&2
  exit 1
fi

TARGET_DIR="${POSITIONAL_ARGS[0]:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd -P)"

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "Missing $REPO_ROOT/.env. Copy .env.example to .env and set your credentials."
  exit 1
fi

if [[ "$DEBUG" -eq 1 ]]; then
  docker build -t llc-dev "$REPO_ROOT"
else
  if ! docker build -t llc-dev "$REPO_ROOT" >/dev/null 2>&1; then
    echo "Docker build failed. Re-run with --debug for full build output." >&2
    exit 1
  fi
fi

TERM_VALUE="${TERM:-xterm-256color}"
COLORTERM_VALUE="${COLORTERM:-truecolor}"
LANGFUSE_BASE_URL_VALUE="${LLC_DOCKER_LANGFUSE_BASE_URL:-http://host.docker.internal:3000}"

DOCKER_ARGS=(
  --rm
  -it
  --init
  --add-host "host.docker.internal:host-gateway"
  -w /workspace
  --env-file "$REPO_ROOT/.env"
  -e "TERM=$TERM_VALUE"
  -e "COLORTERM=$COLORTERM_VALUE"
  -e "LANGFUSE_BASE_URL=$LANGFUSE_BASE_URL_VALUE"
  --mount "type=bind,src=$TARGET_DIR,dst=/workspace"
)

DOCKER_ARGS+=(-p "${PORT}:${PORT}")
exec docker run "${DOCKER_ARGS[@]}" llc-dev serve --host 0.0.0.0 --port "$PORT"
