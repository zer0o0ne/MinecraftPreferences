#!/usr/bin/env bash
# specs/engine.md (8)-(10): MineStudio's simulator runs at real time; needs scripts/setup_env.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.engine.check configs/engine.yaml
