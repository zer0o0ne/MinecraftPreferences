# Spark environment
Status: approved (2026-10-10, amended and re-approved the same day; test tiers added with the user's approval the same day)
Stage: infrastructure for every stage run on the NVIDIA DGX Spark (`specs/00-overview.md`); implemented first: it creates the project files every other block builds on.

## Sources
- [U] the user, 2026-10-09: Python 3.10; 2026-10-10: every action on the Spark is a bash script the user runs; `setup_env.sh` installs all dependencies, `download_data.sh` downloads the data, `serve_vlm.sh` starts the VLM.
- [UV] uv 0.12.19: `install.sh` (install dir: `UV_INSTALL_DIR`, else `XDG_BIN_HOME`, …, else `~/.local/bin`; shell profiles edited unless `UV_NO_MODIFY_PATH=1`; the URL redirects); `uv sync --locked` fails when `uv.lock` is stale, `--frozen` silently installs a stale lock; plain `uv run` re-locks; a `src/` layout needs a `[build-system]` for `python -m mcpref.…` to import.
- [NV] NVIDIA DGX Spark vLLM playbook: the user must be in the `docker` group (`sudo usermod -aG docker $USER`, then log in again).

## Notation
- `configs/spark-env.yaml`, read by bash: one `key: value` per line, `key` matching `[a-z_]+`, `value` without spaces, quotes or comments; a missing key is an error. Keys: `uv_version`, `python_version`, `driver_min` (minimum NVIDIA driver version).
- Starting values: `uv_version` = `0.12.19`, `python_version` = `3.10.20` (cpython linux-aarch64-gnu build listed by uv), `driver_min` = `580.159.03`.

## Behaviour
`scripts/setup_env.sh`, bash with `set -euo pipefail`. It first changes to the repository root (`cd "$(dirname "$0")/.."`). Each step prints `== <name>` on stdout before it runs; an `ERR` trap prints `setup_env.sh: step <name> failed` to stderr. Idempotent: a second run installs nothing new and succeeds. Step names in parentheses below.
(0) `config`: read `configs/spark-env.yaml`; a missing file or key prints `setup_env.sh: missing <key> in configs/spark-env.yaml` (or `missing configs/spark-env.yaml`) to stderr and fails.
(1) `arch`: `uname -m` is `aarch64`.
(2) `driver`: `nvidia-smi --query-gpu=driver_version --format=csv,noheader` ≥ `driver_min` (compared with `sort -V`).
(3) `docker`: `docker info --format '{{json .Runtimes}}'` succeeds — else stderr contains `sudo usermod -aG docker $USER` and `log in again`; its output contains `"nvidia"` — else stderr contains `NVIDIA container runtime not found`.
(4) `uv`: `PATH="$HOME/.local/bin:$PATH"`. If the second field of `uv --version` is not `uv_version` (or uv is absent): `curl -LsSf https://astral.sh/uv/<uv_version>/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh`.
(5) `python`: `uv python install <python_version>`.
(6) `sync`: `uv sync --locked --python <python_version>`.
(7) `ffmpeg`, added by block vpt-gui-frames: ffmpeg for frame extraction over HTTPS.
(8) `vlm`, added by block vlm-client: `uv run --locked python -m mcpref.vlm.server prepare configs/vlm.yaml`.

Every other script in `scripts/` starts the same way: `set -euo pipefail`, `cd "$(dirname "$0")/.."`, `PATH="$HOME/.local/bin:$PATH"`, and calls Python only as `uv run --locked python -m mcpref.…`.

## Interface
- Files: `scripts/setup_env.sh`; `configs/spark-env.yaml`; `uv.lock`; `pyproject.toml`:
  - `[project]`: `name = "mcpref"`, `version = "0.1.0"`, `requires-python = "==3.10.*"`, `dependencies` pinned exactly; this block adds `numpy==1.26.4` (the last 1.x; MineStudio pins `opencv-python==4.8.0.74`, built against numpy 1.x) and `pyyaml==6.0.3`; later blocks add theirs.
  - `[build-system]`: `requires = ["uv_build==0.12.19"]`, `build-backend = "uv_build"`; package `src/mcpref` (with `src/mcpref/__init__.py`).
  - `[dependency-groups] dev`: `pytest==9.1.1`, `ruff==0.16.9`, `pytest-timeout==2.4.0`, `pytest-xdist==3.8.0`.
  - `[tool.pytest.ini_options]`: markers `gpu` and `slow` registered; `addopts = "-m 'not gpu and not slow' -n auto --timeout=10 --durations=10"` (AGENTS.md Tests: fast tests fail at 10 s; a slow test carries `@pytest.mark.timeout(600)`).
  - `[tool.uv] required-version = "==<uv_version>"` (this machine upgrades its uv to the same version).
- Produces: a project environment on the Spark (`uv run --locked …` works).
- Tests (this machine is x86 without Docker or GPU): `tests/scripts/test_setup_env.py` runs `setup_env.sh` with fake `uname`, `nvidia-smi`, `docker`, `uv`, `curl` executables that record their calls, with `HOME` set to a temporary directory whose `.local/bin` holds the fakes (the script prepends `$HOME/.local/bin` itself), and checks the order, the arguments, the failure messages and idempotence. A failing step prints its own message, then the trap line. The real run is manual, on the Spark.

## Library mapping
| Item | Tool | Version | Explicit args | Source read |
|---|---|---|---|---|
| (4) | uv `install.sh` | `uv_version` | `UV_INSTALL_DIR`, `UV_NO_MODIFY_PATH=1`, `curl -L` (redirect) | [UV] `install.sh` lines 1226–1315, 1400–1402 |
| (5), (6) | `uv python install`, `uv sync --locked` | `uv_version` | `--locked`, `--python` | [UV] CLI help, tested on 0.12.19 |

## Out of scope
- Installing Docker, the NVIDIA driver or the container toolkit; joining the `docker` group (a precondition, see (3)); any `sudo`.
- Java and MineStudio's simulator (block env, later).
- Running `setup_env.sh` on other machines (this machine runs `uv sync --locked` directly).

## Exceptions to AGENTS.md (approved 2026-10-10)
- `setup_env.sh` is bash logic, not a thin wrapper: Python does not exist before it runs.
- `configs/spark-env.yaml` is read by bash, not loaded into a dataclass.
- Script tests live in `tests/scripts/`, which mirrors `scripts/`, not `src/`.

