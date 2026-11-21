"""One entry point builds only its selected index before the selected loop."""

from types import SimpleNamespace

import pytest
import yaml

from mcpref.training.pretrain import run
from training.pretrain.test_modes import selection


@pytest.mark.parametrize("mode", ["classical", "continuous"])
def test_dispatch_order_and_paths(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(run, "REPO", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(selection(mode)))
    calls = []
    index_cfg = SimpleNamespace(
        out_dir="data/index", training_config=selection(mode)[mode]["config"]
    )
    train_cfg = SimpleNamespace(index_dir="data/index")
    prefix = f"mcpref.training.pretrain.{mode}"
    modules = {
        f"{prefix}.data.index": SimpleNamespace(
            load_config=lambda p: calls.append(("index-config", p)) or index_cfg,
            run=lambda cfg, p: calls.append(("index", cfg, p)),
        ),
        f"{prefix}.run": SimpleNamespace(
            load_config=lambda p: calls.append(("train-config", p)) or train_cfg,
            run=lambda p: calls.append(("train", p)),
        ),
    }
    monkeypatch.setattr(run, "import_module", modules.__getitem__)
    assert run.main([str(path)]) == 0
    selected = selection(mode)[mode]
    assert calls == [
        ("train-config", tmp_path / selected["config"]),
        ("index-config", tmp_path / selected["index"]),
        ("index", index_cfg, tmp_path / selected["index"]),
        ("train", tmp_path / selected["config"]),
    ]


@pytest.mark.parametrize("mismatch", [False, True])
def test_no_training_after_index_failure(tmp_path, monkeypatch, mismatch, capsys):
    monkeypatch.setattr(run, "REPO", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(selection("classical")))
    calls = []

    def fail(cfg, path):
        calls.append("index")
        raise RuntimeError("source is incomplete")

    index = SimpleNamespace(
        load_config=lambda p: SimpleNamespace(out_dir="data/x"), run=fail
    )
    train = SimpleNamespace(
        load_config=lambda p: SimpleNamespace(
            index_dir="data/y" if mismatch else "data/x"
        ),
        run=lambda p: calls.append("train"),
    )
    monkeypatch.setattr(
        run, "import_module", lambda n: index if n.endswith("index") else train
    )
    assert run.main([str(path)]) == 1
    assert calls == ([] if mismatch else ["index"])
    assert "does not match" in capsys.readouterr().err if mismatch else True


def test_continuous_index_cannot_bind_a_different_training_config(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(run, "REPO", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(selection("continuous")))
    calls = []
    index = SimpleNamespace(
        load_config=lambda p: SimpleNamespace(
            out_dir="data/x", training_config="different.yaml"
        ),
        run=lambda *args: calls.append("index"),
    )
    trainer = SimpleNamespace(
        load_config=lambda p: SimpleNamespace(index_dir="data/x"),
        run=lambda p: calls.append("trainer"),
    )
    monkeypatch.setattr(
        run, "import_module", lambda name: index if name.endswith("index") else trainer
    )
    assert run.main([str(path)]) == 1
    assert calls == []
    assert "training_config does not match" in capsys.readouterr().err
