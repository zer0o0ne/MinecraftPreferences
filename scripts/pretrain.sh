#!/usr/bin/env bash
# specs/train-index.md, then specs/policy.md (13): the index shards, then the pretraining loop;
# needs complete perception-data (and, with pairs, pair clips and pair labels) outputs on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.pretrain.index configs/train-index.yaml
uv run --locked python -m mcpref.pretrain.run configs/pretrain.yaml
