"""Interface `main` and Server steps 2, 4 of specs/vlm-client.md: the vLLM server."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from mcpref import REPO
from mcpref.vlm.client import ServerConfig, VLMConfig, load_config
from mcpref.vlm.container import check, check_ready, readable, remove, wait_ready


def prepare_commands(srv: ServerConfig) -> list[list[str]]:
    """Server step 2: pull the image; download the weights into hf_home."""
    return [
        ["docker", "pull", srv.image],
        ["env", f"HF_HOME={REPO / srv.hf_home}"]
        + ["uvx", "--from", f"huggingface_hub=={srv.hf_cli}", "hf", "download"]
        + [srv.model_repo, "--revision", srv.revision],
    ]


def serve_command(cfg: VLMConfig, srv: ServerConfig) -> list[str]:
    """Server step 4: `docker run` of the vLLM server."""
    port, rev = str(srv.port), srv.revision
    volume = f"{REPO / srv.hf_home}:/root/.cache/huggingface"
    limit_mm = json.dumps({"image": srv.max_images, "video": 0})
    pixels = json.dumps({"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels})
    return (
        ["docker", "run", "-d", "--name", srv.container, "--gpus", "all", "--ipc=host"]
        + ["--ulimit", "memlock=-1", "--ulimit", f"stack={srv.stack_ulimit}"]
        + ["-p", f"127.0.0.1:{port}:{port}", "-v", volume]
        + ["-e", "VLLM_USE_DEEP_GEMM=0", "-e", "HF_HUB_OFFLINE=1"]
        + [srv.image]
        + [srv.model_repo, "--revision", rev, "--tokenizer-revision", rev]
        + ["--served-model-name", cfg.model, "--host", "0.0.0.0", "--port", port]
        + ["--max-model-len", str(cfg.max_tokens + srv.max_prompt_tokens)]
        + ["--gpu-memory-utilization", str(srv.gpu_memory_utilization)]
        + ["--max-num-seqs", str(cfg.max_in_flight)]
        + ["--moe-backend", "triton", "--chat-template-content-format", "openai"]
        + ["--limit-mm-per-prompt", limit_mm, "--mm-processor-kwargs", pixels]
        + ["--generation-config", "vllm", "--seed", str(srv.seed)]
    )


def prepare(srv: ServerConfig) -> None:
    """Interface `prepare`: Server step 2, then GPU access inside containers."""
    for cmd in prepare_commands(srv):
        check(cmd)
    gpu = ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "nvidia-smi"]
    try:
        check([*gpu, srv.image, "-L"])
    except RuntimeError as error:
        toolkit = "no GPU access inside containers; check the NVIDIA Container Toolkit"
        raise RuntimeError(f"{toolkit}: {error}") from error


def serve(cfg: VLMConfig, srv: ServerConfig) -> None:
    """Interface `serve`: replace the container, start it, wait and check (step 5)."""
    remove(srv)
    check(serve_command(cfg, srv))
    wait_ready(srv)
    check_ready(cfg, srv)


def stop(srv: ServerConfig) -> None:
    """Interface `stop`: remove the container if it exists; report which."""
    removed = remove(srv)
    print(f"stopped {srv.container}" if removed else f"{srv.container} is not running")


def message(error: Exception) -> str:
    """Interface `main`: the failure as one line naming its cause."""
    if isinstance(error, httpx.RequestError):
        request = f"{error.request.method} {error.request.url}"
        error = RuntimeError(f"{request}: {type(error).__name__}: {error}")
    return " ".join(str(error).split())


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.vlm.server {prepare|serve|stop} <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog="python -m mcpref.vlm.server")
    parser.add_argument("command", choices=["prepare", "serve", "stop"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        with readable(f"config {args.config}"):
            cfg, srv = load_config(args.config)
        if args.command == "prepare":
            prepare(srv)
        elif args.command == "serve":
            serve(cfg, srv)
        else:
            stop(srv)
    except (RuntimeError, httpx.RequestError) as error:
        print(f"mcpref.vlm.server {args.command}: {message(error)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
