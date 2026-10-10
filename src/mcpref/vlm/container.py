"""Server step 5 and container existence of specs/vlm-client.md: the vLLM container."""

import shlex
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import yaml

from mcpref.vlm.client import ServerConfig, VLMConfig

MOE_BACKEND_LINE = "Using TRITON Fp8 MoE backend"
KV_CACHE_LINE = "KV cache size"


@contextmanager
def readable(what: str) -> Iterator[None]:
    """Interface `main`: an unreadable config or response is a failure."""
    try:
        yield
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        cause = f"{type(error).__name__}: {error}"
        raise RuntimeError(f"unreadable {what}: {cause}") from error


def check(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Runs one command without a shell; a missing executable or non-zero exit fails."""
    try:
        proc = subprocess.run(cmd, check=False, **kwargs)
    except OSError as error:
        raise RuntimeError(f"cannot run {shlex.join(cmd)}: {error}") from error
    if proc.returncode != 0:
        raise RuntimeError(f"exit code {proc.returncode}: {shlex.join(cmd)}")
    return proc


def output(cmd: list[str], **kwargs) -> str:
    """Captured stdout of a command that must succeed."""
    proc = check(cmd, stdout=subprocess.PIPE, **kwargs)
    return proc.stdout.decode("utf-8", "replace")


def exists(srv: ServerConfig) -> bool:
    """Interface: `docker ps -a`, anchored name filter, prints exactly the name."""
    name = ["--filter", f"name=^/{srv.container}$", "--format", "{{.Names}}"]
    return output(["docker", "ps", "-a", *name]).strip() == srv.container


def remove(srv: ServerConfig) -> bool:
    """Interface `serve`, `stop`: docker rm -f if the container exists."""
    found = exists(srv)
    if found:
        check(["docker", "rm", "-f", srv.container])
    return found


def get(srv: ServerConfig, path: str) -> httpx.Response:
    """GET on the server port with the poll request timeout."""
    url, timeout = f"http://127.0.0.1:{srv.port}{path}", srv.poll_request_timeout
    return httpx.get(url, timeout=timeout, trust_env=False)


def healthy(srv: ServerConfig) -> bool:
    """Step 5: GET /health answers 200; a connection error means not ready yet."""
    try:
        return get(srv, "/health").status_code == 200
    except httpx.TransportError:
        return False


def running(srv: ServerConfig) -> bool:
    """Step 5: `.State.Running` is not `false`; a failing inspect counts as stopped."""
    inspect = ["docker", "container", "inspect", "-f", "{{.State.Running}}"]
    try:
        return output([*inspect, srv.container]).strip() != "false"
    except RuntimeError:
        return False


def not_ready(srv: ServerConfig, reason: str) -> RuntimeError:
    """Step 5: prints the last log_tail log lines; the container is kept."""
    try:
        check(["docker", "logs", "--tail", str(srv.log_tail), srv.container])
    except RuntimeError as error:
        reason += f"; log tail failed: {error}"
    return RuntimeError(f"server not ready: {reason}; container {srv.container} kept")


def wait_ready(srv: ServerConfig) -> None:
    """Step 5: polls /health until ready, the container stops or ready_timeout passes."""
    start = time.monotonic()
    while not healthy(srv):
        if not running(srv):
            raise not_ready(srv, "container stopped")
        if time.monotonic() - start >= srv.ready_timeout:
            raise not_ready(srv, f"no 200 from /health after {srv.ready_timeout} s")
        time.sleep(srv.poll_interval)


def served_models(srv: ServerConfig) -> list:
    """Step 5: the ids listed by GET /v1/models."""
    response = get(srv, "/v1/models")
    if response.status_code != 200:
        raise RuntimeError(f"GET /v1/models: HTTP {response.status_code}")
    with readable("GET /v1/models response"):
        return [m["id"] for m in response.json()["data"]]


def check_ready(cfg: VLMConfig, srv: ServerConfig) -> None:
    """Step 5 once ready: served name, transformers version, MoE backend, KV cache."""
    if cfg.model not in served_models(srv):
        raise RuntimeError(f"/v1/models does not list {cfg.model}")
    python = ["python3", "-c", "import transformers; print(transformers.__version__)"]
    version = output(["docker", "exec", srv.container, *python]).strip()
    if version != srv.transformers_version:
        expected = srv.transformers_version
        raise RuntimeError(f"transformers {version} in the container, not {expected}")
    log = output(["docker", "logs", srv.container], stderr=subprocess.STDOUT)
    if MOE_BACKEND_LINE not in log:
        raise RuntimeError(f"no '{MOE_BACKEND_LINE}' in the container log")
    for line in log.splitlines():
        if KV_CACHE_LINE in line:
            print(line)
