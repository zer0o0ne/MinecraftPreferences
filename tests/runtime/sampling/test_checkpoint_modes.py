"""Classical runtime must not silently reset a continuously trained model."""

import json

import pytest
import torch

from mcpref.runtime.sampling.checkpoints import load_weights


def test_continuous_final_model_requires_stream_adapter(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"pretrain_mode": "continuous", "phi": {"weight": torch.ones(1)}}, path)
    with pytest.raises(RuntimeError, match="continuous perception stream adapter"):
        load_weights(path)


def test_continuous_checkpoint_is_rejected_before_loading_stream_objects(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"must not deserialize this checkpoint")
    (tmp_path / "metadata.json").write_text(json.dumps({"pretrain_mode": "continuous"}))
    with pytest.raises(RuntimeError, match="continuous perception stream adapter"):
        load_weights(path)


@pytest.mark.parametrize("name", ["model.pt", "checkpoint.pt"])
def test_classical_artifacts_keep_their_existing_format(tmp_path, name):
    path = tmp_path / name
    modules = {"phi": {"weight": torch.tensor([3.0])}}
    torch.save({"modules": modules} if name == "checkpoint.pt" else modules, path)
    torch.testing.assert_close(
        load_weights(path)["phi"]["weight"], modules["phi"]["weight"]
    )
