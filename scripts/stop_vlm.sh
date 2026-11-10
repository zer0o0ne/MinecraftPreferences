#!/usr/bin/env bash
# Interface of specs/vlm-client.md: stops the VLM server on the DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
uv run --locked python -m mcpref.integrations.vlm.server stop configs/integrations/vlm.yaml
