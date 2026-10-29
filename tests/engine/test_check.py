"""`main` of `mcpref.engine.check` (specs/engine.md (7)-(10)) against the fake simulator of
fake_minestudio.py: the variables set before its import, the constructor's arguments, the
calls and step counts, ρ over the timed loop with a fake `perf_counter`, the stdout line,
exit 0 / 1 at `min_rate`, an episode end, `close` also on an error, the `check` bar."""

import os
import subprocess
import sys
from dataclasses import dataclass

import pytest
from progress_bars import Bar, assert_aborted, assert_progress, screen, states

from mcpref.engine.check import main

from .engine_world import JAR, TINY, VARS, write_config
from .fake_minestudio import STEP, Fake, install, uninstall

PREFIX = "mcpref.engine.check: "
W, S = 3, 8  # warm-up and timed steps; a timed step costs 1/16 s: ρ = 16.0 ticks/s
LAUNCH, CLOSED = "fake minestudio: launching\n", "fake minestudio: closed\n"


@dataclass
class Check:
    fake: Fake
    code: int
    out: str
    err: str


@pytest.fixture
def check(root, clean_env, monkeypatch, tmp_path, capsys):
    """run(fake, jar=True, **config changes) -> Check; J present unless `jar=False`."""

    def run(fake: Fake | None = None, jar: bool = True, **changes) -> Check:
        fake = fake or Fake(W)
        install(fake, monkeypatch)
        if jar:
            (root / "ms" / JAR).parent.mkdir(parents=True)
            (root / "ms" / JAR).write_bytes(b"jar")
        fields = TINY | {"minestudio_dir": "ms", "warmup_steps": W, "steps": S}
        path = write_config(
            tmp_path / "engine.yaml", fields | {"min_rate": 16} | changes
        )
        code = main([str(path)])
        out, err = capsys.readouterr()
        return Check(fake, code, out, err)

    yield run
    uninstall()


def steps(n: int) -> list:
    return [c for k in range(1, n + 1) for c in (("noop_action",), ("step", k, True))]


def test_variables_set_before_import(check, root):
    r = check()
    assert r.code == 0, r.err
    assert r.fake.imports == [
        {"MINESTUDIO_DIR": str(root / "ms"), "MINESTUDIO_GPU_RENDER": "0"}
    ]


def test_constructor_arguments(check):
    """(9): `action_type`, `obs_size` = `render_size` = (width, height), `seed`; nothing
    else."""
    kwargs = {
        "action_type": "env",
        "obs_size": (64, 36),
        "render_size": (64, 36),
        "seed": 7,
    }
    assert check().fake.calls[0] == ("init", (), kwargs)


def test_calls_and_step_counts(check):
    """reset, then W + S steps, each `step(noop_action())` with its own noop, then close."""
    assert check().fake.calls[1:] == [("reset",), *steps(W + S), ("close",)]


def test_rate_line_after_minestudio_lines(check):
    r = check()
    assert (r.code, r.out) == (0, f"{LAUNCH}{CLOSED}rate: 16.0 ticks/s (min 16.0)\n")


@pytest.mark.parametrize(
    "step_cost, rate",
    [(STEP, "16.0"), (STEP + 1 / 128, "14.2"), (STEP - 1 / 64, "21.3"), (1.0, "1.0")],
)
def test_rate_over_the_timed_loop(check, step_cost, rate):
    """ρ = steps / (t1 - t0): the S timed steps with their noop_action, check and bar
    update (1/32 s besides the step), none of the warm-up, reset or close."""
    r = check(Fake(W, step_cost=step_cost), min_rate=0.5)
    assert r.out.splitlines()[-1] == f"rate: {rate} ticks/s (min 0.5)"


@pytest.mark.parametrize("min_rate, code", [(16, 0), (15.75, 0), (16.25, 1), (100, 1)])
def test_exit_at_min_rate(check, min_rate, code):
    """ρ = 16.0 exactly: exit 0 iff ρ ≥ min_rate; the line printed either way."""
    r = check(min_rate=min_rate)
    assert r.out.splitlines()[-1] == f"rate: 16.0 ticks/s (min {float(min_rate)})"
    assert r.code == code
    if code:
        message = f"RuntimeError: rate 16.0 ticks/s below min_rate {float(min_rate)}"
        assert_aborted(r.code, r.err)
        assert r.err.splitlines()[-1] == PREFIX + message


@pytest.mark.parametrize(
    "k, flag",
    [(1, "terminated"), (W, "truncated"), (W + 1, "terminated"), (W + S, "truncated"),
     (W + S, "terminated")],
)  # fmt: skip
def test_episode_end(check, k, flag):
    """terminated or truncated at step k (counted over warm-up and timed steps): the error
    with k, no later step, close, the bar left at k - 1, no rate line."""
    r = check(Fake(W, end=(k, flag)))
    assert assert_aborted(r.code, r.err)[-1] == Bar("check", k - 1, W + S, "step")
    assert r.err.splitlines()[-1] == f"{PREFIX}RuntimeError: episode ended at step {k}"
    assert r.fake.calls[1:] == [("reset",), *steps(k), ("close",)]
    assert r.out == LAUNCH + CLOSED


@pytest.mark.parametrize("at", [2, W + S])
def test_step_raising_closes(check, at):
    r = check(Fake(W, fail=at))
    assert_aborted(r.code, r.err)
    assert r.err.splitlines()[-1] == f"{PREFIX}RuntimeError: step failed in the fake"
    assert r.fake.calls[-1] == ("close",) and r.out == LAUNCH + CLOSED


def test_reset_raising_closes(check):
    r = check(Fake(W, fail="reset"))
    assert_aborted(r.code, r.err)
    assert [c[0] for c in r.fake.calls] == ["init", "reset", "close"]
    assert not states(r.err, "check")


def test_constructor_raising(check):
    """No simulator: nothing to close."""
    r = check(Fake(W, fail="init"))
    assert_aborted(r.code, r.err)
    assert r.err.splitlines()[-1] == f"{PREFIX}RuntimeError: init failed in the fake"
    assert [c[0] for c in r.fake.calls] == ["init"]


def test_engine_missing(check, root):
    """(7) in (8): exit 1 with use_engine's error; MineStudio not imported, nothing set."""
    r = check(jar=False)
    message = f"engine missing: {root / 'ms' / JAR}; run scripts/setup_env.sh"
    assert r.err.splitlines() == [f"{PREFIX}FileNotFoundError: {message}"]
    assert (r.code, r.out, r.fake.imports, r.fake.calls) == (1, "", [], [])
    assert not any(v in os.environ for v in VARS)


def test_config_error(check):
    r = check(steps=0)
    assert r.code == 1 and r.err.startswith(f"{PREFIX}ValueError: steps 0 is not ")
    assert r.fake.imports == []


def test_check_bar(check):
    """Created after reset; total W + S; one state per step at most; closed at W + S."""
    r = check()
    assert_progress(r.err)
    assert screen(r.err) == [
        "fake minestudio: reset",
        Bar("check", W + S, W + S, "step"),
    ]
    assert states(r.err, "check")[0] == Bar("check", 0, W + S, "step")
    assert r.err.index("fake minestudio: reset") < r.err.index("check:")


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "usage: python -m mcpref.engine.check" in capsys.readouterr().err


def test_python_m(tmp_path, repo):
    """`python -m mcpref.engine.check <config>`: J missing under the repository root →
    exit 1 with the line."""
    ms = "data/__no_such_minestudio_dir__"
    assert not (repo / ms).exists()
    path = write_config(tmp_path / "c.yaml", TINY | {"minestudio_dir": ms})
    proc = subprocess.run(
        [sys.executable, "-m", "mcpref.engine.check", str(path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    message = f"engine missing: {repo / ms / JAR}; run scripts/setup_env.sh"
    assert (proc.returncode, proc.stdout) == (1, "")
    assert proc.stderr.splitlines() == [f"{PREFIX}FileNotFoundError: {message}"]
