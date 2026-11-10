#!/usr/bin/env bash
# specs/vpt-gui-frames.md: downloads the data; first step from specs/gui-screens.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.textures configs/gui/screens.yaml
uv run --locked python -m mcpref.datasets.vpt.gui_frames verify configs/datasets/vpt/gui-frames.yaml
uv run --locked python -m mcpref.datasets.vpt.gui_frames run configs/datasets/vpt/gui-frames.yaml
