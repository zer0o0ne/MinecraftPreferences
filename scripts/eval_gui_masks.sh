#!/usr/bin/env bash
# Interface of specs/gui-masks.md: mask variants against the VLM labels; the VLM server need not run.
# Then specs/gui-stack.md: the classifier over their features.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.recognition.evaluate configs/gui/recognition/masks.yaml
uv run --locked python -m mcpref.gui.stacking.run configs/gui/recognition/stacking.yaml
