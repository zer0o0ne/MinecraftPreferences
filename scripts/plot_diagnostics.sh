#!/usr/bin/env bash
# View saved diagnostics and optional pretrain/RL logs; no model evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.diagnostics.plot "$@"
