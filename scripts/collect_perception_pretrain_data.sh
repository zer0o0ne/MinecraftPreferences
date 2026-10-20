#!/usr/bin/env bash
# Interface of specs/perception-data.md: clips of VPT episodes, classified by specs/gui-classifier.md;
# needs a complete gui-masks run (scripts/eval_gui_masks.sh) on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.textures configs/gui-screens.yaml
uv run --locked python -m mcpref.clips.run configs/perception-data.yaml
