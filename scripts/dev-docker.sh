#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/dev-docker.sh [--debug] [target-dir]

Options:
  --debug, -d   Show full Docker build output.
  --help, -h    Show this help message.

Examples:
  ./scripts/dev-docker.sh
  ./scripts/dev-docker.sh /path/to/workspace
  ./scripts/dev-docker.sh --debug /path/to/workspace

Notes:
  Experimental sub-agent mode can be enabled at runtime via `/enable sub-agent-mode`.
  To start with it enabled by default, set `SUB_AGENT_MODE_ENABLED=true` in .env.
  Session history persists in SQLite at /workspace/.llc/sessions.db by default.
  Override DB location with `LLC_DB_PATH` in .env if needed.
  Inside the TUI, use `Ctrl+G` or the `Agents` button to toggle the sub-agent side panel.
EOF
}

DEBUG=0
if [[ "${LOCAL_CLAUDE_DOCKER_DEBUG:-0}" == "1" ]]; then
  DEBUG=1
fi

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug|-d)
      DEBUG=1
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

exec docker run --rm -it --init \
  -w /workspace \
  --env-file "$REPO_ROOT/.env" \
  -e "TERM=$TERM_VALUE" \
  -e "COLORTERM=$COLORTERM_VALUE" \
  --mount "type=bind,src=$TARGET_DIR,dst=/workspace" \
  llc-dev
