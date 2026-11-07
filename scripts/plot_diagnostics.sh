#!/usr/bin/env bash
# Plot accepted producer-owned metric records only; no producer or evaluation invocation.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.diagnostics.plot "$@"
