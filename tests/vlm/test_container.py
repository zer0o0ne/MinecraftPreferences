"""src/mcpref/vlm/container.py through `main` against specs/vlm-client.md: container existence
(`serve`, `stop`) and Server step 5."""

import time
from itertools import pairwise
from types import SimpleNamespace

import pytest

from mcpref.vlm.client import ServerConfig, VLMConfig
from mcpref.vlm.server import main, serve_command

LOG = [  # log_tail = 3: the last three lines; TRITON on stderr, before the tail
    ("out", "INFO 04:00:00 [api_server.py:1] vLLM API server version 0.31.0"),
    ("err", "INFO 04:00:01 [fp8.py:415] Using TRITON Fp8 MoE backend"),
    ("out", "INFO 04:00:02 [kv_cache_utils.py:2465] GPU KV cache size: 1,234 tokens"),
    ("err", "INFO 04:00:03 [x.py:1] tail line a"),
    ("out", "INFO 04:00:04 [x.py:2] tail line b"),
    ("err", "INFO 04:00:05 [x.py:3] tail line c"),
]
DEEPGEMM = ("err", "INFO 04:00:01 [fp8.py:415] Using DEEPGEMM Fp8 MoE backend")
NAME = "vlm-test"  # fields["server"]["container"]
EXISTS = ["docker", "ps", "-a", "--filter", f"name=^/{NAME}$", "--format", "{{.Names}}"]
RM = ["docker", "rm", "-f", NAME]


class VLLM:
    """Fake vLLM: /health 503 for the first `ready_after` polls, the first held `hang` s;
    `polls` holds [start, end] of each /health request."""

    def __init__(self, served: str):
        self.ready_after, self.hang, self.polls = 0, 0.0, []
        self.models, self.models_hang = [served], 0.0

    def __call__(self, req) -> tuple:
        if (req.method, req.path) == ("GET", "/health"):
            return self.health(req)
        if (req.method, req.path) != ("GET", "/v1/models"):
            return 404, b""
        req.hold(self.models_hang)
        if isinstance(self.models, bytes):
            return 200, self.models
        return 200, {"object": "list", "data": [{"id": m} for m in self.models]}

    def health(self, req) -> tuple:
        n = len(self.polls)
        self.polls.append([req.t, None])
        try:
            req.hold(self.hang if n == 0 else 0.0)
        finally:
            self.polls[n][1] = time.monotonic()
        return (200 if n >= self.ready_after else 503), b""


@pytest.fixture
def world(http, fields, write_config, make_docker):
    """Fake Docker as subprocess.run, fake vLLM on the port once the container runs."""
    vllm = VLLM(fields["client"]["model"])
    server = http(vllm, listen=False)
    fields["server"]["port"] = server.port
    docker = make_docker(server, fields["server"]["transformers_version"], LOG)

    def run(command: str = "serve") -> int:
        return main([command, str(write_config(fields))])

    return SimpleNamespace(vllm=vllm, docker=docker, fields=fields, run=run)


@pytest.mark.parametrize(
    "existing",
    [{}, {"vlm-test-2": True, "xvlm-test": False}, {NAME: True}, {NAME: False}],
)  # name -> running; the existence check is anchored and includes stopped containers
def test_serve_replaces_the_named_container_and_runs_serve_command(
    world, capsys, existing
):
    world.docker.containers = dict(existing)
    assert world.run() == 0
    c, s = world.fields["client"], world.fields["server"]
    run = serve_command(VLMConfig(**c), ServerConfig(**s))
    head = [EXISTS, *([RM] if NAME in existing else []), run]
    calls, step_5 = world.docker.calls[: len(head)], world.docker.calls[len(head) :]
    assert [x.argv for x in calls] == head
    assert [x.passthrough for x in calls] == [False] + [True] * (len(head) - 1)
    assert step_5 and not any(x.passthrough for x in step_5)  # outputs the code reads
    assert LOG[2][1] in "".join(capsys.readouterr())  # the KV cache size line


@pytest.mark.parametrize("running", [True, False])
def test_stop_removes_an_existing_container(world, capsys, running):
    world.docker.containers = {NAME: running, "vlm-test-2": True}
    assert world.run("stop") == 0
    assert [(c.argv, c.passthrough) for c in world.docker.calls] == [
        (EXISTS, False),
        (RM, True),
    ]
    assert world.docker.containers == {"vlm-test-2": True}
    out, err = capsys.readouterr()
    assert f"stopped {NAME}" in out.splitlines() and err == ""


def test_stop_without_the_container_reports_not_running(world, capsys):
    world.docker.containers = {"vlm-test-2": True, "xvlm-test": False}
    assert world.run("stop") == 0
    assert [c.argv for c in world.docker.calls] == [EXISTS]
    assert capsys.readouterr() == (f"{NAME} is not running\n", "")


@pytest.mark.parametrize(
    "command, prefix, status, calls",
    [("serve", ("docker",), None, 1), ("serve", ("docker", "ps"), 1, 1),
     ("serve", ("docker", "rm"), 1, 2), ("serve", ("docker", "run"), 125, 3),
     ("stop", ("docker",), None, 1), ("stop", ("docker", "ps"), 1, 1),
     ("stop", ("docker", "rm"), 1, 2)],
)  # fmt: skip  # None: docker is not installed; `calls`: the failing call is the last
def test_failing_or_missing_docker_fails(world, failed, command, prefix, status, calls):
    world.docker.containers = {NAME: True}
    world.docker.fail[prefix] = status
    assert "docker" in failed(world.run(command), world.docker)
    assert len(world.docker.calls) == calls


@pytest.mark.parametrize("not_ready", ["refused", "503", "request timeout"])
def test_polls_until_health_is_200_one_interval_after_each_poll_ends(world, not_ready):
    s, vllm = world.fields["server"], world.vllm
    s["poll_interval"] = 0.2
    world.docker.listen_delay = 0.5 if not_ready == "refused" else 0.0
    vllm.ready_after = 3 if not_ready == "503" else 0
    vllm.hang = 4.0 if not_ready == "request timeout" else 0.0
    start = time.monotonic()
    assert world.run() == 0 and time.monotonic() - start < 3.0
    polls = vllm.polls
    assert len(polls) == {"refused": 1, "503": 4, "request timeout": 2}[not_ready]
    # server-side ends: a reply leaves ~ms later, a disconnect is seen <= 0.02 s later
    assert all(0.17 <= b[0] - a[1] < 0.5 for a, b in pairwise(polls)), polls
    if not_ready == "request timeout":
        assert 0.45 <= polls[0][1] - polls[0][0] < 0.8  # poll_request_timeout = 0.5


@pytest.mark.parametrize("failure", ["ready_timeout", "stopped", "inspect fails"])
def test_readiness_failure_prints_log_tail_and_keeps_container(world, failed, failure):
    world.fields["server"]["ready_timeout"] = 0.6 if failure == "ready_timeout" else 4
    world.vllm.ready_after = 10**9
    world.docker.stops = failure == "stopped"
    world.docker.inspect_fails = failure == "inspect fails"
    code = world.run()
    low, high = (0.6, 2.1) if failure == "ready_timeout" else (0.0, 1.5)
    assert low <= time.monotonic() - world.docker.run_returned < high
    failed(code, world.docker, tail=world.fields["server"]["log_tail"])
    assert NAME in world.docker.containers
    assert not any(c.argv[:2] == ["docker", "rm"] for c in world.docker.calls)


MISMATCHES = {
    "model not listed": lambda w: setattr(w.vllm, "models", ["other-model"]),
    "models unreadable": lambda w: setattr(w.vllm, "models", b"<html>no</html>"),
    "models request timeout": lambda w: setattr(w.vllm, "models_hang", 4.0),
    "transformers version": lambda w: setattr(w.docker, "version", "5.17.0"),
    "transformers check fails": lambda w: w.docker.fail.update({("docker", "exec"): 1}),
    "other MoE backend": lambda w: setattr(
        w.docker, "log", [LOG[0], DEEPGEMM, *LOG[2:]]
    ),
}


@pytest.mark.parametrize("mismatch", list(MISMATCHES))
def test_check_mismatch_after_ready_fails(world, failed, mismatch):
    MISMATCHES[mismatch](world)
    failed(world.run(), world.docker)


def test_readiness_and_checks_ignore_proxy_variables(world, refusing_proxy):
    """Library mapping: trust_env=False; the proxy from the environment refuses connections."""
    world.fields["server"]["ready_timeout"] = 1
    assert world.run() == 0
