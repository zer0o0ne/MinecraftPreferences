#!/usr/bin/env bash
# Behaviour of specs/spark-env.md: installs the project environment on the DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'echo "setup_env.sh: step $name failed" >&2' ERR

config=configs/spark-env.yaml
step() { name=$1; echo "== $name"; }
fail() { echo "setup_env.sh: $1" >&2; return 1; }
cfg() { sed -n "s/^$1: //p" "$config" | grep . || fail "missing $1 in $config"; }

step config
[[ -f $config ]] || fail "missing $config"
uv_version=$(cfg uv_version)
python_version=$(cfg python_version)
driver_min=$(cfg driver_min)

step arch
[[ $(uname -m) == aarch64 ]]

step driver
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
printf '%s\n' "$driver_min" "$driver" | sort -V -C

step docker
runtimes=$(docker info --format '{{json .Runtimes}}') ||
  fail 'docker info failed; run "sudo usermod -aG docker $USER", then log in again'
[[ $runtimes == *'"nvidia"'* ]] || fail "NVIDIA container runtime not found"

step uv
PATH="$HOME/.local/bin:$PATH"
if [[ $(uv --version 2>/dev/null | awk '{print $2}') != "$uv_version" ]]; then
  curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" |
    env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi

step python
uv python install "$python_version"

step sync
uv sync --locked --python "$python_version"

step vlm
uv run --locked python -m mcpref.vlm.server prepare configs/vlm.yaml
