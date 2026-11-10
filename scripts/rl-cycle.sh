#!/usr/bin/env bash
# specs/rl-cycle.md: one R1 -> R3 -> joint training -> R5 cycle.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.training.rl.run configs/training/rl/cycle.yaml
