#!/usr/bin/env bash
# Interface of specs/gui-labels.md: VLM labels of the GUI frames; needs scripts/serve_vlm.sh running.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.labels configs/gui-labels.yaml
