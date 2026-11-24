"""Startup costs use indexed metadata; invalid reference metrics never publish."""

from types import SimpleNamespace

import pytest
from training.pretrain.continuous.diagnostics.test_reference import world

from mcpref.training.pretrain.continuous.diagnostics import hooks, reference_store


def test_training_cost_report_uses_index_counts_without_reading_data(capsys):
    cfg = SimpleNamespace(
        batch_size=16,
        tbptt_states=4,
        precision="fp32",
        activation_checkpointing=False,
        checkpoint_steps=100,
        diagnostic_window=100,
        reference_steps=5000,
        gradient_diagnostics=False,
        publish_diagnostics=False,
    )
    context = SimpleNamespace(
        cfg=cfg,
        phase="joint",
        step=2,
        total=21,
        index=SimpleNamespace(
            run={"populations": {"train": {"observations": 100, "fragments": 7}}}
        ),
    )
    hooks.report_start(context)
    output = capsys.readouterr().err
    for text in (
        "100 training observations/epoch",
        "7 fragments",
        "7 updates/epoch",
        "2/21 total updates completed",
        "B=16, K=4",
        "precision=fp32",
        "activation_checkpointing=False",
        "checkpoint_steps=100",
        "diagnostic_window=100",
        "reference_steps=5000",
        "gradient_diagnostics=False",
        "publish_diagnostics=False",
        "Reference evaluation is independent of both diagnostic flags",
    ):
        assert text in output


def test_reference_reports_missing_reused_and_scheduled_prefixes(
    tmp_path, monkeypatch, capsys
):
    cfg, modules, index, visits = world(monkeypatch)
    cfg.reference_steps = 2
    index.run = {"populations": {"train": {"N_P": 1}}}
    context = SimpleNamespace(
        cfg=cfg, modules=modules, index=index, root=tmp_path / "run", step=0
    )
    monkeypatch.setattr(reference_store, "REPO", tmp_path)
    reference = reference_store.References(context, None, "reference")
    original = reference_store.evaluate_initial

    def evaluate(*args):
        output = capsys.readouterr().err
        assert "initial baseline missing; full prefix replay required" in output
        assert "2 selected pairs, 4 unique prefix fragments" in output
        assert "interrupted unpublished pass restarts" in output
        return original(*args)

    monkeypatch.setattr(reference_store, "evaluate_initial", evaluate)
    try:
        reference.start()
        assert len(visits) == 12
        visits.clear()
        reference = reference_store.References(context, None, "reference")
        reference.start()
        assert not visits
        assert (
            "complete initial baseline reused; no initial replay"
            in capsys.readouterr().err
        )
        context.step = 2
        reference.due()
        output = capsys.readouterr().err
        assert (
            "reference step 2: scheduled evaluation; full prefix replay required"
            in output
        )
        assert len(visits) == 12
    finally:
        index.db.close()


def test_empty_preference_population_reports_no_reference_work(tmp_path, capsys):
    context = SimpleNamespace(
        root=tmp_path, index=SimpleNamespace(run={"populations": {"train": {"N_P": 0}}})
    )
    reference_store.References(context, None, "reference").start()
    assert "disabled (no training preference pairs)" in capsys.readouterr().err


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("publisher", [False, True])
def test_nonfinite_reference_metrics_rejected_before_any_publication(
    tmp_path, monkeypatch, value, publisher
):
    monkeypatch.setattr(reference_store, "REPO", tmp_path)
    staged = []
    diagnostics = SimpleNamespace(
        cfg=None,
        stage=lambda *args: staged.append(args),
        accept=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(reference_store, "on_reference", lambda *a, **kw: {"id": "x"})
    context = SimpleNamespace(root=tmp_path / "run", phase="joint")
    reference = reference_store.References(
        context, diagnostics if publisher else None, "reference"
    )
    reference.selection = {"items": []}
    observations = {
        key: {"train": None, "validation": None}
        for key in ("joint", "P_only", "phi_only", "initial")
    }
    with pytest.raises(ValueError, match="JSON compliant"):
        reference.publish(0, {"step": 0, "nested": {"loss": value}}, [], observations)
    assert not staged
    assert not context.root.exists()
    assert reference.count == 0
