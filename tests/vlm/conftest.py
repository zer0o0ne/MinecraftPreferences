"""Config fields and a fake Docker CLI for specs/vlm-client.md."""

import errno
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, CompletedProcess

import pytest
import yaml

RUNNING, NAMES = "{{.State.Running}}", "{{.Names}}"
TRANSFORMERS = "import transformers; print(transformers.__version__)"


@pytest.fixture
def fields() -> dict:
    """Config fields of Notation; every value differs from its starting value."""
    client = {"base_url": "http://127.0.0.1:9", "model": "served-test", "temperature": 0.11,
              "top_p": 0.22, "top_k": 33, "min_p": 0.044, "presence_penalty": 0.55,
              "frequency_penalty": 0.66, "repetition_penalty": 1.07, "max_tokens": 77,
              "min_pixels": 1024, "max_pixels": 2048, "max_in_flight": 5, "attempt_timeout": 4.0,
              "max_resends": 2, "resend_delay": 0.05}  # fmt: skip
    server = {"image": "registry.test/vllm@sha256:" + "ab" * 32, "model_repo": "Org/Model-Test",
              "revision": "0123abcd", "hf_cli": "9.8.7", "hf_home": "data/hf-test",
              "container": "vlm-test", "port": 18123, "gpu_memory_utilization": 0.55,
              "max_images": 7, "max_prompt_tokens": 1000, "seed": 42, "stack_ulimit": 12345,
              "ready_timeout": 4, "poll_interval": 0.05, "poll_request_timeout": 0.5,
              "log_tail": 3, "transformers_version": "4.99.1"}  # fmt: skip
    return {"client": client, "server": server}


@pytest.fixture
def write_config(tmp_path):
    """Write `{"client": …, "server": …}` as a configs/vlm.yaml file; returns its path."""

    def write(sections: dict) -> Path:
        path = tmp_path / "vlm.yaml"
        path.write_text(yaml.safe_dump(sections))
        return path

    return write


def as_executed(argv: list) -> tuple[dict, list]:
    """(added environment, command) of an argv run without a shell; a leading `env` sets NAME=VALUE."""
    if Path(argv[0]).name != "env":
        return {}, list(argv)
    i = 1
    while "=" in argv[i]:
        i += 1
    return dict(a.split("=", 1) for a in argv[1:i]), list(argv[i:])


@pytest.fixture(name="as_executed")
def as_executed_fixture():
    return as_executed


@dataclass
class Call:
    argv: list
    passthrough: bool
    env: dict


def options(args: list, valued: tuple) -> tuple[dict, list]:
    """Docker CLI arguments as ({flag: value}, positionals); `--flag=value` too."""
    opts, rest, it = {}, [], iter(args)
    for a in it:
        if a.startswith("-"):
            name, eq, value = a.partition("=")
            opts[name] = value if eq else (next(it, "") if name in valued else True)
        else:
            rest.append(a)
    return opts, rest


def unknown(args: list) -> tuple:
    return 125, [("err", f"fake-docker: unsupported {args}\n")]


def finish(argv: list, rc: int, chunks: list, kw: dict) -> CompletedProcess:
    """subprocess.run semantics of capture_output, stdout, stderr (STDOUT merges), text and check."""
    so, se = (
        (PIPE, PIPE)
        if kw.get("capture_output")
        else (kw.get("stdout"), kw.get("stderr"))
    )
    if se == STDOUT:
        se, chunks = so, [("out", text) for _, text in chunks]
    got = {"out": "", "err": ""}
    for stream, text in chunks:
        mode = so if stream == "out" else se
        if mode is None:
            (sys.stdout if stream == "out" else sys.stderr).write(text)
        elif mode == PIPE:
            got[stream] += text
    as_text = kw.get("text") or kw.get("universal_newlines") or kw.get("encoding")
    out, err = (
        (got[s] if as_text else got[s].encode()) if m == PIPE else None
        for s, m in (("out", so), ("err", se))
    )
    if kw.get("check") and rc:
        raise CalledProcessError(rc, argv, out, err)
    return CompletedProcess(argv, rc, out, err)


class FakeDocker:
    """`subprocess.run` for the Docker CLI calls of the Server section; other programs succeed.

    `containers` maps a name to whether it runs; `fail` maps a command prefix to an exit
    status, or to None for a missing executable.
    """

    def __init__(self, http, version: str, log: list):
        self.http, self.version, self.log = http, version, list(log)
        self.containers, self.calls, self.fail = {}, [], {}
        self.stops = self.inspect_fails = False
        self.listen_delay, self.run_returned = 0.0, None

    def __call__(self, args, **kw) -> CompletedProcess:
        assert not kw.get("shell")
        added, argv = as_executed(list(args))
        env = dict(os.environ if kw.get("env") is None else kw["env"]) | added
        capture = kw.get("capture_output") or kw.get("stdout") is not None
        self.calls.append(
            Call(list(args), not capture and kw.get("stderr") is None, env)
        )
        status = next(
            (v for p, v in self.fail.items() if tuple(argv[: len(p)]) == p), 0
        )
        if status is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), argv[0])
        failed = (status, [("err", f"fake-docker: {argv[:2]} failed\n")])
        rc, chunks = failed if status else self.dispatch(argv)
        if argv[:2] == ["docker", "run"]:
            self.run_returned = time.monotonic()
        return finish(list(args), rc, chunks, kw)

    def dispatch(self, a: list) -> tuple:
        if a[0] != "docker" or a[1:2] == ["pull"]:
            return 0, [("out", "fake-docker: done\n")]
        commands = {"ps": self.ps, "container": self.inspect, "rm": self.rm,
                    "run": self.run, "exec": self.exec, "logs": self.logs}  # fmt: skip
        return commands[a[1]](a[2:]) if a[1] in commands else unknown(a)

    def ps(self, a: list) -> tuple:
        """`docker ps [-a] --filter name=REGEX --format {{.Names}}`: regex search on `/name` or `name`."""
        opts, rest = options(a, ("--filter", "--format"))
        every = opts.pop("-a", False) or opts.pop("--all", False)
        kind, _, regex = opts.pop("--filter", "").partition("=")
        if opts.pop("--format", None) != NAMES or opts or rest or kind != "name":
            return unknown(a)
        names = [n for n, running in self.containers.items() if (every or running)
                 and (re.search(regex, "/" + n) or re.search(regex, n))]  # fmt: skip
        return 0, [("out", "".join(n + "\n" for n in names))]

    def inspect(self, a: list) -> tuple:
        """`docker container inspect -f {{.State.Running}} NAME`."""
        opts, names = options(a[1:], ("-f", "--format"))
        fmt = opts.pop("-f", None) or opts.pop("--format", None)
        if a[:1] != ["inspect"] or opts or len(names) != 1 or fmt != RUNNING:
            return unknown(a)
        if names[0] not in self.containers or self.inspect_fails:
            return 1, [("err", f"fake-docker: Error: No such container: {names[0]}\n")]
        return 0, [("out", str(self.containers[names[0]]).lower() + "\n")]

    def rm(self, a: list) -> tuple:
        if a[:1] != ["-f"] or len(a) != 2 or a[1] not in self.containers:
            return unknown(a)
        del self.containers[a[1]]
        return 0, [("out", a[1] + "\n")]

    def run(self, a: list) -> tuple:
        name = a[a.index("--name") + 1]
        if name in self.containers:
            return 125, [
                ("err", f'fake-docker: Conflict. The name "/{name}" is in use\n')
            ]
        self.containers[name] = not self.stops
        if not self.stops:
            timer = threading.Timer(self.listen_delay, self.http.listen)
            timer.daemon = True
            timer.start()
        return 0, [("out", "fake-docker: c0ffee\n")]

    def exec(self, a: list) -> tuple:
        if a[1:] != ["python3", "-c", TRANSFORMERS] or not self.containers.get(a[0]):
            return unknown(a)
        return 0, [("out", self.version + "\n")]

    def logs(self, a: list) -> tuple:
        """`docker logs [--tail N] NAME`: the container's stdout and stderr lines, in order."""
        opts, names = options(a, ("-n", "--tail"))
        tail = opts.pop("--tail", None) or opts.pop("-n", None) or "all"
        if opts or len(names) != 1 or names[0] not in self.containers:
            return unknown(a)
        lines = (
            self.log if tail == "all" else self.log[max(len(self.log) - int(tail), 0) :]
        )
        return 0, [(stream, line + "\n") for stream, line in lines]


@pytest.fixture
def make_docker(monkeypatch):
    """FakeDocker installed as subprocess.run."""

    def make(http, version: str, log: list) -> FakeDocker:
        docker = FakeDocker(http, version, log)
        monkeypatch.setattr("subprocess.run", docker)
        return docker

    return make


@pytest.fixture
def failed(capsys):
    """Exit 1, no traceback, one own line on stderr (fake and log lines aside); with `tail`,
    exactly the last `tail` log lines are printed. Returns the own line."""

    def check(code: int, docker: FakeDocker, tail: int = 0) -> str:
        out, err = capsys.readouterr()
        log = [line for _, line in docker.log]
        own = [
            x for x in err.splitlines() if x.strip() and not x.startswith("fake-docker")
        ]
        own = [x for x in own if not any(y in x for y in log)]
        assert code == 1 and "Traceback" not in out + err and len(own) == 1, (out, err)
        if tail:
            assert all(y in out + err for y in log[-tail:]), out + err
            assert not any(y in out + err for y in log[:-tail]), out + err
        return own[0]

    return check
