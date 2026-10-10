#!/usr/bin/env bash
# Interface of specs/vlm-client.md: stops the VLM server on the DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.vlm.server stop configs/vlm.yaml
