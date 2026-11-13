"""Selected-step reader uses only accepted, persisted gradient-flow scalars."""

import json

from mcpref.diagnostics.report.gradient import gradient_records


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def summary_step(step):
    return {
        "step": step,
        "roles": ["beginning"],
        "metrics": "checkpoints/000000/gradient_flow/000000/metrics.json",
        "phi_total": {
            "norms": {"phi": 3.0, "pi": 4.0, "P": 5.0},
            "dot": {"P_phi": -2.0},
            "cosine": {"P_phi": None},
            "update_dot": -0.25,
            "update_unavailable": None,
        },
        "P_total": {"norm": 6.0, "update_dot": None},
        "delta_only_phi_mean": 0.125,
        "delta_joint_mean": -0.5,
    }


def accepted_step(step):
    return {
        "version": 1,
        "step": step,
        "roles": ["middle"],
        "checkpoint_losses_index": 4,
        "micro_batches": [{"micro_batch": 0, "losses": {"phi": 123.0}}],
        "phi": {
            "total": {"norms": {"phi": 8.0}, "cosine": {"P_pi": -0.5}},
            "groups": {
                "phi.body": {
                    "update_dot": None,
                    "update_unavailable": "zero_decay_factor",
                }
            },
        },
        "P": {"total": {"norm": 7.0, "update_dot": -0.25}},
        "nonfinite": {"phi": {"P": {"phi.body": {"nan": 0, "count": 12}}}},
        "counterfactual": {
            "delta_only_phi": {"mean": 0.5, "harmful": 0.75, "quantiles": {"0.9": 2.0}},
            "delta_joint": {"mean": None, "helpful": None},
        },
        "evidence": {"kind": "real_pretrain_step"},
    }


def test_missing_gradient_artifacts_are_empty(tmp_path):
    assert list(gradient_records(None)) == []
    assert list(gradient_records(tmp_path)) == []


def test_summary_survives_missing_checkpoint_records_and_preserves_nulls(tmp_path):
    path = tmp_path / "gradient-flow/summary.json"
    write(path, {"version": 1, "steps": [summary_step(91), summary_step(7)]})
    rows = list(gradient_records(tmp_path))
    assert [row["step"] for row in rows] == [7, 91]
    assert rows[0] == {
        "group": "saved-gradient-flow",
        "title": "Pretrain / joint / selected steps",
        "source": str(tmp_path),
        "source_file": str(path),
        "x_label": "Selected optimizer step",
        "step": 7,
        "values": {
            "phi.total.norms.phi": 3.0,
            "phi.total.norms.pi": 4.0,
            "phi.total.norms.P": 5.0,
            "phi.total.dot.P_phi": -2.0,
            "phi.total.cosine.P_phi": None,
            "phi.total.update_dot": -0.25,
            "P.total.norm": 6.0,
            "P.total.update_dot": None,
            "counterfactual.delta_only_phi.mean": 0.125,
            "counterfactual.delta_joint.mean": -0.5,
        },
    }


def test_accepted_details_override_summary_without_reading_models_or_pairs(tmp_path):
    write(tmp_path / "gradient-flow/summary.json", {"steps": [summary_step(7)]})
    path = tmp_path / "checkpoints/000000/gradient_flow/000000/metrics.json"
    write(path, accepted_step(7))
    (path.parent / "pairs.json").write_text("not loaded")
    (path.parents[2] / "checkpoint.pt").write_bytes(b"not a torch checkpoint")
    (row,) = gradient_records(tmp_path)
    assert row["source_file"] == str(path)
    assert row["values"] == {
        "phi.total.norms.phi": 8.0,
        "phi.total.cosine.P_pi": -0.5,
        "phi.groups.phi.body.update_dot": None,
        "P.total.norm": 7.0,
        "P.total.update_dot": -0.25,
        "nonfinite.phi.P.phi.body.nan": 0,
        "nonfinite.phi.P.phi.body.count": 12,
        "counterfactual.delta_only_phi.mean": 0.5,
        "counterfactual.delta_only_phi.harmful": 0.75,
        "counterfactual.delta_only_phi.quantiles.0.9": 2.0,
        "counterfactual.delta_joint.mean": None,
        "counterfactual.delta_joint.helpful": None,
    }


def test_stale_summary_does_not_hide_new_accepted_step_or_mix_retrain(tmp_path):
    write(tmp_path / "gradient-flow/summary.json", {"steps": [summary_step(7)]})
    write(
        tmp_path / "checkpoints/000000/gradient_flow/000001/metrics.json",
        accepted_step(91),
    )
    write(tmp_path / "retrain/gradient-flow/summary.json", {"steps": [summary_step(7)]})
    rows = list(gradient_records(tmp_path))
    assert [row["step"] for row in rows] == [7, 91, 7]
    assert [row["source"] for row in rows] == [
        str(tmp_path),
        str(tmp_path),
        str(tmp_path / "retrain"),
    ]
    assert rows[-1]["title"] == "Pretrain / retrain / selected steps"


def test_partial_and_synthetic_records_are_ignored(tmp_path):
    paths = (
        "checkpoints/000000/gradient_flow/000000.part/metrics.json",
        "checkpoints/000001.part/gradient_flow/000000/metrics.json",
        "gradient-flow/work/000000/metrics.json",
    )
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial JSON")
    synthetic = accepted_step(4)
    synthetic["evidence"]["kind"] = "synthetic"
    write(tmp_path / "checkpoints/000000/gradient_flow/000001/metrics.json", synthetic)
    assert list(gradient_records(tmp_path)) == []
