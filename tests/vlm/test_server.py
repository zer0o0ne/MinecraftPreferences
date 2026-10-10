"""src/mcpref/vlm/server.py against specs/vlm-client.md: Interface (entry, `prepare`), Server steps 2
and 4."""

import dataclasses
import json
import subprocess
import sys
from itertools import pairwise

import pytest

from mcpref.vlm.client import ServerConfig, VLMConfig, load_config
from mcpref.vlm.server import main, prepare_commands, serve_command


def expected_prepare(s: dict, root) -> list:
    """Server step 2 as (added environment, command)."""
    hf = ["hf", "download", s["model_repo"], "--revision", s["revision"]]
    uvx = ["uvx", "--from", f"huggingface_hub=={s['hf_cli']}", *hf]
    home = {"HF_HOME": str(root / s["hf_home"])}
    return [({}, ["docker", "pull", s["image"]]), (home, uvx)]


def expected_serve(c: dict, s: dict, root) -> list:
    """Server step 4 with `{…}` substituted; the two JSON arguments via json.dumps."""
    return [
        "docker", "run", "-d", "--name", s["container"], "--gpus", "all", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", f"stack={s['stack_ulimit']}",
        "-p", f"127.0.0.1:{s['port']}:{s['port']}", "-v", f"{root / s['hf_home']}:/root/.cache/huggingface",
        "-e", "VLLM_USE_DEEP_GEMM=0", "-e", "HF_HUB_OFFLINE=1",
        s["image"],
        s["model_repo"], "--revision", s["revision"], "--tokenizer-revision", s["revision"],
        "--served-model-name", c["model"], "--host", "0.0.0.0", "--port", f"{s['port']}",
        "--max-model-len", f"{c['max_tokens'] + s['max_prompt_tokens']}",
        "--gpu-memory-utilization", f"{s['gpu_memory_utilization']}", "--max-num-seqs", f"{c['max_in_flight']}",
        "--moe-backend", "triton", "--chat-template-content-format", "openai",
        "--limit-mm-per-prompt", json.dumps({"image": s["max_images"], "video": 0}),
        "--mm-processor-kwargs", json.dumps({"min_pixels": c["min_pixels"], "max_pixels": c["max_pixels"]}),
        "--generation-config", "vllm", "--seed", f"{s['seed']}",
    ]  # fmt: skip


@pytest.fixture
def docker(make_docker, monkeypatch, tmp_path):
    """Fake Docker as subprocess.run; the working directory is not the repository root."""
    monkeypatch.chdir(tmp_path)
    return make_docker(None, "", [])


def test_prepare_commands_are_step_2(fields, repo, as_executed, docker):
    commands = prepare_commands(ServerConfig(**fields["server"]))
    assert [as_executed(c) for c in commands] == expected_prepare(
        fields["server"], repo
    )
    assert docker.calls == []  # pure


def test_serve_command_is_step_4(fields, repo, docker):
    c, s = fields["client"], fields["server"]
    assert serve_command(VLMConfig(**c), ServerConfig(**s)) == expected_serve(
        c, s, repo
    )
    assert docker.calls == []  # pure


def test_serve_command_with_starting_values(repo, docker):
    cfg, srv = load_config(repo / "configs/vlm.yaml")
    argv = serve_command(cfg, srv)
    assert argv == expected_serve(
        dataclasses.asdict(cfg), dataclasses.asdict(srv), repo
    )
    after = dict(pairwise(argv))
    assert after["--max-model-len"] == "8492"  # 300 + 8192
    assert after["--limit-mm-per-prompt"] == '{"image": 32, "video": 0}'
    pixels = '{"min_pixels": 65536, "max_pixels": 230400}'
    assert after["--mm-processor-kwargs"] == pixels
    assert after["-v"] == f"{repo}/data/hf:/root/.cache/huggingface"
    assert not any("API_KEY" in a or a == "--api-key" for a in argv)  # Server step 3


def gpu_check(s: dict) -> list:
    """Interface, `prepare`: GPU access inside containers, with the vLLM image."""
    return ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "nvidia-smi", s["image"], "-L"]  # fmt: skip


def test_prepare_runs_step_2_then_gpu_check_passing_output_through(
    docker, fields, write_config, repo, as_executed
):
    assert main(["prepare", str(write_config(fields))]) == 0
    expected = expected_prepare(fields["server"], repo)
    commands = [c for _, c in expected] + [gpu_check(fields["server"])]
    assert [as_executed(c.argv)[1] for c in docker.calls] == commands
    assert docker.calls[1].env["HF_HOME"] == expected[1][0]["HF_HOME"]
    assert all(c.passthrough for c in docker.calls)


@pytest.mark.parametrize("no_gpu", ["daemon error", "nvidia-smi fails"])
def test_prepare_without_gpu_in_containers_names_the_toolkit(
    docker, fields, write_config, failed, no_gpu
):
    if no_gpu == "daemon error":
        docker.gpus = False
    else:
        docker.fail[tuple(gpu_check(fields["server"]))] = 9
    code = main(["prepare", str(write_config(fields))])
    assert "NVIDIA Container Toolkit" in failed(code, docker)
    assert len(docker.calls) == 3


@pytest.mark.parametrize(
    "prefix, status, calls",
    [(("docker",), None, 1), (("docker", "pull"), 1, 1), (("uvx",), 1, 2)],
)  # None: the executable is missing
def test_prepare_stops_at_a_failing_command(
    docker, fields, write_config, failed, prefix, status, calls
):
    docker.fail[prefix] = status
    code = main(["prepare", str(write_config(fields))])
    assert prefix[0] in failed(code, docker)
    assert len(docker.calls) == calls


@pytest.mark.parametrize("command", ["prepare", "serve", "stop"])
@pytest.mark.parametrize("content", [None, "client: [unclosed\n"])  # None: no file
def test_unreadable_config_fails(docker, failed, tmp_path, command, content):
    path = tmp_path / "unreadable.yaml"
    if content is not None:
        path.write_text(content)
    failed(main([command, str(path)]), docker)
    assert docker.calls == []


def test_runs_as_module(tmp_path):
    argv = [sys.executable, "-m", "mcpref.vlm.server", "stop", "missing.yaml"]
    kw = {"cwd": tmp_path, "capture_output": True, "text": True}
    proc = subprocess.run(argv, check=False, **kw)
    assert proc.returncode == 1 and "Traceback" not in proc.stderr
    assert len(proc.stderr.strip().splitlines()) == 1, proc.stderr
