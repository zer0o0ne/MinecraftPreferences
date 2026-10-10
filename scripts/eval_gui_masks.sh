#!/usr/bin/env bash
# Interface of specs/gui-masks.md: mask variants against the VLM labels; the VLM server need not run.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.evaluate configs/gui-masks.yaml
