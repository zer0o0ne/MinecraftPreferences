#!/usr/bin/env bash
# Interface of specs/vlm-client.md: starts the VLM server on the DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
PATH="$HOME/.local/bin:$PATH"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
uv run --locked python -m mcpref.vlm.server serve configs/vlm.yaml
