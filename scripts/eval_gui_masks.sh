#!/usr/bin/env bash
# Interface of specs/gui-masks.md: mask variants against the VLM labels; the VLM server need not run.
# Then specs/gui-stack.md: the classifier over their features.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.recognition.evaluate configs/gui-masks.yaml
uv run --locked python -m mcpref.stacking.run configs/gui-stack.yaml
