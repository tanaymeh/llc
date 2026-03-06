#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd -P)"

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "Missing $REPO_ROOT/.env. Copy .env.example to .env and set your credentials."
  exit 1
fi

docker build -t local-claude-code-dev "$REPO_ROOT"

exec docker run --rm -it --init \
  -w /workspace \
  --env-file "$REPO_ROOT/.env" \
  --mount "type=bind,src=$TARGET_DIR,dst=/workspace" \
  local-claude-code-dev
