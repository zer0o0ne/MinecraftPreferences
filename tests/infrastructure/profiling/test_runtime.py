"""Bounded real CPU traces, transparent training, and failure-safe scopes."""

import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from mcpref.infrastructure.profiling.runtime import TrainingProfile, settings
from mcpref.infrastructure.profiling.scopes import ACTIVE, phase, qwen_batch

CFG = SimpleNamespace(device="cpu", batch=4, micro_batch=2, gradient_diagnostics=False)


def enable(tmp_path, monkeypatch, skip=1, steps=1):
    path = tmp_path / "profile.yaml"
    path.write_text(f"skip_steps: {skip}\nsteps: {steps}\ntop_ops: 5\n")
    monkeypatch.setenv("MCPREF_PROFILE", str(path))


def test_no_profile_has_no_profiler_hooks_sync_or_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MCPREF_PROFILE", "")

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive profiler called expensive instrumentation")

    monkeypatch.setattr(torch.profiler, "profile", forbidden)
    monkeypatch.setattr(torch.profiler, "record_function", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    with (
        TrainingProfile(CFG, tmp_path, {}, "test") as profile,
        profile.step(1, 4),
        phase("forward"),
    ):
        qwen_batch([2, 3], 2)
        (torch.ones(2, requires_grad=True) * 3).sum().backward()
    assert not list(tmp_path.iterdir())
    assert ACTIVE.get() is None


def test_resume_relative_window_and_real_trace_do_not_change_training(
    tmp_path, monkeypatch
):
    enable(tmp_path, monkeypatch, skip=1, steps=2)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    expected = parameter.detach().clone().requires_grad_()
    opt = torch.optim.AdamW([parameter], lr=0.01)
    reference = torch.optim.AdamW([expected], lr=0.01)
    rng = torch.get_rng_state()
    with TrainingProfile(CFG, tmp_path, {}, "test") as profile:
        for step in range(42, 46):
            with profile.step(step, 4):
                with phase("forward"):
                    qwen_batch([2, 4], 3)
                    loss = parameter.square().sum()
                with phase("backward"):
                    loss.backward()
                with phase("optimizer"):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            expected.square().sum().backward()
            reference.step()
            reference.zero_grad(set_to_none=True)
    assert torch.equal(parameter, expected)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in opt.state[parameter].items():
        assert torch.equal(value, reference.state[expected][key])
    summaries = sorted(tmp_path.glob("profiles/test/*/step-*/summary.json"))
    assert len(summaries) == 2
    for path, step in zip(summaries, [43, 44], strict=True):
        row = json.loads(path.read_text())
        assert row["step"] == step
        assert row["executed_step"] == step - 41
        assert row["items"] == 4 and row["wall_s"] > 0
        assert row["cuda_activity"] is False and row["memory"] is None
        regions = {region["name"]: region for region in row["regions"]}
        assert {"step", "forward", "backward", "optimizer"} <= regions.keys()
        assert all(region["device_s"] is None for region in regions.values())
        assert row["qwen_batches"] == [
            {
                "rows": 2,
                "frames": 3,
                "token_lengths": [2, 4],
                "padded_tokens": 8,
                "padding_fraction": 0.25,
                "grad_enabled": True,
            }
        ]
        with gzip.open(path.with_name("trace.json.gz"), "rt") as stream:
            events = json.load(stream)["traceEvents"]
        assert any(e.get("name") == "aten::square" for e in events)
        assert any(e.get("name") == "mcpref::backward" for e in events)
        assert "Operators by self CPU time" in path.with_name("summary.txt").read_text()


def test_exception_removes_hooks_and_restores_context(tmp_path, monkeypatch):
    enable(tmp_path, monkeypatch, skip=0)

    class Broken(torch.nn.Module):
        def forward(self, value):
            raise ValueError("model failure")

    backbone = SimpleNamespace(visual=Broken(), language_model=torch.nn.Identity())
    modules = {"phi": SimpleNamespace(backbone=backbone)}
    with (
        pytest.raises(ValueError, match="model failure"),
        TrainingProfile(CFG, tmp_path, modules, "failure") as profile,
        profile.step(1, 4),
    ):
        backbone.visual(torch.ones(1))
    assert ACTIVE.get() is None
    for module in (backbone.visual, backbone.language_model):
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert not list(tmp_path.glob("profiles/**/summary.json"))


@pytest.mark.parametrize(
    "text",
    [
        "skip_steps: -1\nsteps: 1\ntop_ops: 5",
        "skip_steps: true\nsteps: 1\ntop_ops: 5",
        "skip_steps: 0\nsteps: 0\ntop_ops: 5",
        "skip_steps: 0\nsteps: 1\ntop_ops: 0",
        "skip_steps: 0\nsteps: 1\ntop_ops: 5\nunknown: 3",
    ],
)
def test_invalid_profile_config_is_rejected(tmp_path, monkeypatch, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    monkeypatch.setenv("MCPREF_PROFILE", str(path))
    with pytest.raises(ValueError):
        settings()
