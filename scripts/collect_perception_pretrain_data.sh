#!/usr/bin/env bash
# Shared streams and classical clip views; specs/continuous-data.md.
# needs a complete gui-masks run (scripts/eval_gui_masks.sh) on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.textures configs/gui/screens.yaml
uv run --locked python -m mcpref.datasets.continuous.run configs/datasets/continuous.yaml
