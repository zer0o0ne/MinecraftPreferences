#!/usr/bin/env bash
# specs/env.md Tests: the real endless-timeline smoke sequence; needs scripts/setup_env.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.runtime.env.check configs/runtime/env.yaml
