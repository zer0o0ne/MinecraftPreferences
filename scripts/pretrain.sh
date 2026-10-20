#!/usr/bin/env bash
# Interface of specs/train-index.md: the train / validation index shards of the pretraining loop;
# needs complete perception-data (and, with pairs, pair clips and pair labels) outputs on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.pretrain.index configs/train-index.yaml
