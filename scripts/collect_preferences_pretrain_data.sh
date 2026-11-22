#!/usr/bin/env bash
# Dataset-only preference candidates; specs/continuous-data.md.
# Requires completed shared perception data; original labels remain archived.
# Then specs/vlm-labels.md: the VLM's answers to the comparisons; needs scripts/serve_vlm.sh running.
# Then specs/acyclic.md: the pair labels, contradictions removed.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.datasets.continuous.views.pool configs/datasets/preference-pool-shared.yaml
uv run --locked python -m mcpref.datasets.preferences.labels configs/datasets/preferences/labels-shared.yaml
uv run --locked python -m mcpref.datasets.preferences.acyclic configs/datasets/preferences/acyclic-shared.yaml
uv run --locked python -m mcpref.training.pretrain.classical.data.index configs/training/pretrain/classical/index/shared-labeled.yaml
