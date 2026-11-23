"""The continuous profiler uses observation-window fields and includes frame reads."""

import json

from training.pretrain.continuous.engine_world import (
    Reader,
    config,
    context,
    modules,
    observation_step,
)
from training.pretrain.continuous.test_resume import world

from mcpref.training.pretrain.continuous import loop, steps


def test_continuous_profile_captures_actual_window(tmp_path, monkeypatch):
    path = tmp_path / "profile.yaml"
    path.write_text("skip_steps: 0\nsteps: 1\ntop_ops: 5\n")
    monkeypatch.setenv("MCPREF_PROFILE", str(path))
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg, index = config(epochs=1, tbptt_states=3, batch_size=2), world(2)
    ctx = context(tmp_path / "run", cfg, modules(), index, total=4)
    ctx.root.mkdir()
    loop.train(ctx)
    paths = list(ctx.root.glob("profiles/*/*/step-*/summary.json"))
    assert len(paths) == 1
    summary = json.loads(paths[0].read_text())
    assert summary["tbptt_states"] == 3
    assert summary["batch_size"] == summary["items"] == 2
    assert "batch" not in summary and "micro_batch" not in summary
    assert "data.stream_read" in {row["name"] for row in summary["regions"]}
    assert paths[0].with_name("trace.json.gz").exists()
