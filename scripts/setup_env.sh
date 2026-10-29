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
ffmpeg_deb_version=$(cfg ffmpeg_deb_version)

step arch
[[ $(uname -m) == aarch64 ]]

step driver
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
printf '%s\n' "$driver_min" "$driver" | sort -V -C ||
  fail "driver $driver is older than driver_min $driver_min"

step docker
docker info >/dev/null ||
  fail 'docker info failed; run "sudo usermod -aG docker $USER", then log in again'

step uv
PATH="$HOME/.local/bin:$PATH"
if [[ $(uv --version 2>/dev/null | awk '{print $2}') != "$uv_version" ]]; then
  curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" |
    env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi

step python
uv python install "$python_version"

step engine-deps
engine_debs=(openjdk-8-jre xvfb libgl1-mesa-dri libglx-mesa0 build-essential libx11-dev)
expected=$(printf 'installed%.0s\n' "${engine_debs[@]}")
if [[ $(dpkg-query -W -f='${db:Status-Status}\n' "${engine_debs[@]}") != "$expected" ]]; then
  sudo apt-get update
  sudo apt-get install -y "${engine_debs[@]}"
fi
java_version=$(java -version 2>&1 || true)
[[ $java_version == *'version "1.8.'* ]] ||
  fail 'java on PATH is not Java 8; run "sudo update-alternatives --config java"'

step sync
uv sync --locked --python "$python_version"

step ffmpeg
ffmpeg_debs=(ffmpeg libavcodec60 libavdevice60 libavfilter9 libavformat60 libavutil58
  libpostproc57 libswresample4 libswscale7)
expected=$(printf 'installed %s\n' "${ffmpeg_debs[@]/*/$ffmpeg_deb_version}")
if [[ $(dpkg-query -W -f='${db:Status-Status} ${Version}\n' "${ffmpeg_debs[@]}") != "$expected" ]]; then
  sudo apt-get update
  sudo apt-get install -y --allow-downgrades "${ffmpeg_debs[@]/%/=$ffmpeg_deb_version}"
  sudo apt-mark hold "${ffmpeg_debs[@]}"
fi

step vlm
uv run --locked python -m mcpref.vlm.server prepare configs/vlm.yaml

step engine
uv run --locked python -m mcpref.engine.prepare configs/engine.yaml
