"""Block 33 pretrain outboxes, no-backfill watermarks and run completion."""

import json
import shutil

from datasets.preferences.test_diagnostics import patch_alias

from mcpref.diagnostics.core.artifacts import write_record
from mcpref.training.pretrain.checkpoints import write_checkpoint
from training.pretrain.test_diagnostics_joins import diagnostics, records, small_loop


def fail_final_record(mp):
    def fail(out_dir, kind, index, record, _real=write_record):
        if record["id"] == "step:2":
            raise OSError("after final checkpoint, before record")
        return _real(out_dir, kind, index, record)

    patch_alias(mp, write_record, "write_record", fail)


def test_final_checkpoint_outbox_recovers_without_any_model_call(tmp_path):
    loop = small_loop(tmp_path)
    result = loop.run(patch=fail_final_record, raises=OSError)
    assert result.error is not None
    source = loop.checkpoint_dir(0) / "losses.json"
    before = source.read_bytes()
    pending = list(diagnostics(loop).glob("outbox/pretrain-step/*.json"))
    assert len(pending) == 2
    assert not (loop.out / "model.pt").exists()
    modules = loop.modules()
    resumed = loop.run(modules)
    assert resumed.error is None
    assert all(module.calls == 0 for module in modules.values())
    assert source.read_bytes() == before
    assert any(record["id"] == "step:2" for record in records(loop))
    assert not list(diagnostics(loop).glob("outbox/pretrain-step/*.json"))


def test_pending_outbox_without_source_is_recomputed(tmp_path):
    loop = small_loop(tmp_path)

    def before_source(mp):
        def fail(*args, **kwargs):
            raise OSError("after outbox, before final checkpoint")

        patch_alias(mp, write_checkpoint, "write_checkpoint", fail)

    result = loop.run(patch=before_source, raises=OSError)
    assert result.error is not None
    assert list(diagnostics(loop).glob("outbox/pretrain-step/*.json"))
    assert not loop.checkpoint_dir(0).exists()
    modules = loop.modules()
    resumed = loop.run(modules)
    assert resumed.error is None
    assert any(module.calls > 0 for module in modules.values())
    assert loop.checkpoint_dir(0).is_dir()
    assert not list(diagnostics(loop).glob("outbox/pretrain-step/*.json"))


def test_first_instrumented_pretrain_run_sets_watermarks_without_backfill(tmp_path):
    loop = small_loop(tmp_path)
    loop.run()
    shutil.rmtree(diagnostics(loop))
    modules = loop.modules()
    result = loop.run(modules)
    assert result.error is None
    assert all(module.calls == 0 for module in modules.values())
    run = json.loads((diagnostics(loop) / "run.json").read_text())
    assert run["channels"]["pretrain-step"]["watermark"] == 3
    assert run["channels"]["pretrain-step"]["next"] == 3
    assert run["channels"]["pretrain-reference"]["watermark"] == 1
    assert run["channels"]["pretrain-reference"]["next"] == 1
    assert run["counts"] == {
        "pretrain-step": {"events": 0, "records": 0},
        "pretrain-reference": {"events": 0, "records": 0},
    }
    assert not list(diagnostics(loop).glob("pretrain/*/metrics.json"))
    assert "diagnostics pretrain: nothing to do" in result.err


def test_retained_input_identity_mismatch_blocks_outbox_recovery(tmp_path):
    loop = small_loop(tmp_path)
    loop.run(patch=fail_final_record, raises=OSError)
    source = loop.checkpoint_dir(0) / "losses.json"
    source.write_text(source.read_text() + " ")
    modules = loop.modules()
    result = loop.run(modules, raises=RuntimeError)
    assert result.error is not None
    assert all(module.calls == 0 for module in modules.values())
    assert not any(record["id"] == "step:2" for record in records(loop))


def test_run_is_complete_only_with_channels_and_current_index(tmp_path):
    loop = small_loop(tmp_path)
    loop.run()
    run = json.loads((diagnostics(loop) / "run.json").read_text())
    assert all(channel["complete"] for channel in run["channels"].values())
    assert (diagnostics(loop) / "index.json").is_file()
    assert run["complete"] is True
    (diagnostics(loop) / "index.json").unlink()
    loop.run()
    assert (diagnostics(loop) / "index.json").is_file()
    assert json.loads((diagnostics(loop) / "run.json").read_text())["complete"] is True
