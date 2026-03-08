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

exec docker run --rm -it --init \
  -w /workspace \
  --env-file "$REPO_ROOT/.env" \
  --mount "type=bind,src=$TARGET_DIR,dst=/workspace" \
  llc-dev
