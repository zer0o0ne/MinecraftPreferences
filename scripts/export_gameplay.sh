#!/usr/bin/env bash
# Export saved gameplay using configs/runtime/video.yaml; specs/gameplay-video.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.runtime.video.export "$@"
