#!/usr/bin/env bash
# Run gameplay configured by configs/runtime/play.yaml; see specs/runtime-play.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.runtime.play.run "$@"
