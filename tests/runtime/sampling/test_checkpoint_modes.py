"""Classical runtime must not silently reset a continuously trained model."""

import json

import numpy as np
import pytest
import torch

from mcpref.models.perception.continuous import ContinuousState
from mcpref.models.perception.continuous.state import GDNState
from mcpref.runtime.sampling.checkpoints import load_weights
from mcpref.training.pretrain.continuous.state import Stream


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


@pytest.mark.parametrize("name", ["model.pt", "checkpoint.pt"])
def test_continuous_artifacts_load_weights_without_importing_training_memory(
    tmp_path, name
):
    path = tmp_path / name
    weights = {"phi": {"weight": torch.tensor([7.0])}}
    saved = {"pretrain_mode": "continuous"}
    if name == "checkpoint.pt":
        saved |= {
            "modules": weights,
            "stream": Stream(
                recurrent=ContinuousState(
                    (GDNState(torch.zeros(2), torch.ones(2, 2)),), {"xpos": 900}
                ),
                previous={"frame": np.zeros((2, 2, 3), dtype=np.uint8)},
            ),
            "torch_rng": torch.get_rng_state(),
        }
    else:
        saved |= weights | {"attention": "observation", "memory": "fragment"}
    torch.save(saved, path)
    (tmp_path / "metadata.json").write_text(json.dumps({"pretrain_mode": "continuous"}))
    allowed = list(torch.serialization.get_safe_globals())
    loaded = load_weights(path, "continuous")
    torch.testing.assert_close(loaded["phi"]["weight"], weights["phi"]["weight"])
    assert "stream" not in loaded
    assert set(torch.serialization.get_safe_globals()) == set(allowed)


@pytest.mark.parametrize(
    "sidecar,payload,requested",
    [
        ("classical", "continuous", "continuous"),
        ("continuous", "classical", "continuous"),
        (None, "classical", "continuous"),
        (None, "continuous", "classical"),
    ],
)
def test_config_metadata_and_payload_modes_must_agree(
    tmp_path, sidecar, payload, requested
):
    path = tmp_path / "model.pt"
    torch.save({"pretrain_mode": payload, "phi": {}}, path)
    if sidecar:
        (tmp_path / "metadata.json").write_text(json.dumps({"pretrain_mode": sidecar}))
    with pytest.raises(RuntimeError, match="mode mismatch"):
        load_weights(path, requested)
