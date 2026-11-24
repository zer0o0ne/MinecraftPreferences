"""Every continuous entry point rejects another owner before expensive or mutable work."""

from types import SimpleNamespace

import pytest
from infrastructure.runs.lock_process import holding
from training.pretrain.test_modes import selection

from mcpref.training.pretrain import run as dispatch
from mcpref.training.pretrain.continuous import loop, run
from mcpref.training.pretrain.continuous.data import index


@pytest.mark.parametrize("entry", ["selector", "trainer", "loop", "index"])
def test_busy_output_rejected_before_work(tmp_path, monkeypatch, capsys, entry):
    cfg = SimpleNamespace(out_dir="run", index_dir="index")
    for module in (dispatch, run, loop, index):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(run, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        run, "Index", lambda _: pytest.fail("Index was opened before the run lock")
    )
    path = tmp_path / "config.yaml"
    chosen = selection("continuous")["continuous"]
    monkeypatch.setattr(
        dispatch,
        "load_selection",
        lambda _: SimpleNamespace(
            mode="continuous", selected=SimpleNamespace(**chosen)
        ),
    )
    index_config = SimpleNamespace(out_dir="index", training_config=chosen["config"])
    fake_index = SimpleNamespace(
        load_config=lambda _: index_config,
        run=lambda *_: pytest.fail("Index was mutated before the run lock"),
    )
    monkeypatch.setattr(
        dispatch,
        "import_module",
        lambda name: fake_index if name.endswith("index") else run,
    )
    with holding(tmp_path / "run"):
        if entry in ("selector", "trainer"):
            assert (dispatch if entry == "selector" else run).main([str(path)]) == 1
            message = capsys.readouterr().err
            assert "Another process is already using" in message
            assert "Traceback" not in message
        else:
            with pytest.raises(RuntimeError, match="already using"):
                if entry == "loop":
                    loop.pretrain(cfg, path, {})
                else:
                    index.run(cfg, path)
    assert not (tmp_path / "run").exists()
