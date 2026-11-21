#!/usr/bin/env bash
# specs/pretrain-continuous.md: build the selected index and run classical or continuous.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.training.pretrain.run configs/training/pretrain/config.yaml
