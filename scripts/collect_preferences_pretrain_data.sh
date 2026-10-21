#!/usr/bin/env bash
# Interface of specs/preference-pool.md: clips of recording sessions of VPT and their comparisons;
# needs a complete gui-masks run (scripts/eval_gui_masks.sh) on this machine.
# Then specs/vlm-labels.md: the VLM's answers to the comparisons; needs scripts/serve_vlm.sh running.
# Then specs/acyclic.md: the pair labels, contradictions removed.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.gui.textures configs/gui-screens.yaml
uv run --locked python -m mcpref.pool.run configs/preference-pool.yaml
uv run --locked python -m mcpref.preferences.labels configs/vlm-labels.yaml
uv run --locked python -m mcpref.preferences.acyclic configs/acyclic.yaml
