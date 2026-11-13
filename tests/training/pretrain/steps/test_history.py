"""A shortened stored clip through the real Qwen, losses and policy targets."""

import json
from types import SimpleNamespace

import torch
from models.perception.world import (
    backbone,
    config,
    random_clip,
    runs_backbone,
    tokenizer,
)
from models.policy.policy_oracle import cfg_of as policy_config
from models.preference.preference_oracle import cfg_of as preference_config

from mcpref.models.perception.loss import PhiLoss, latent_metric
from mcpref.models.perception.phi import Phi
from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from mcpref.models.preference.loss import PreferenceLoss
from mcpref.models.preference.network import Preference
from mcpref.training.pretrain.data.items import clip_reader
from mcpref.training.pretrain.steps.micro_loss import micro_loss


@runs_backbone
def test_stored_clip_suffix_trains_all_heads_with_aligned_targets(tmp_path, masks):
    cfg = config(tmp_path, history=3)
    clip = random_clip(1, [10, 12, 17, 18, 25], [False] * 5)
    clip.info.update(
        actions=[["forward"], ["use"], ["attack"], ["jump"]],
        n=[1] * 4,
        pitch=[0.0] * 5,
        classes=[None] * 5,
    )
    stored = tmp_path / "clip"
    stored.mkdir()
    clip.frames.tofile(stored / "frames.yuv")
    (stored / "clip.json").write_text(json.dumps(clip.info))
    (stored / "lines.jsonl").write_text("\n".join(map(json.dumps, clip.lines)) + "\n")
    read = clip_reader(
        SimpleNamespace(perception=str(tmp_path / "perception.yaml")),
        {"states": 5, "width": clip.width, "height": clip.height},
    )
    tail = read(str(stored))
    assert tail.info["states"] == [17, 18, 25]
    assert tail.info["actions"] == [["attack"], ["jump"]]
    policy = policy_config(history=5, action_dropout=0.0)
    modules = {
        "phi": Phi(backbone(), tokenizer(), cfg).double(),
        "phi_loss": PhiLoss(cfg).double(),
        "pi": Policy(policy, len(cfg.alpha), masks.names).double(),
        "pi_loss": PolicyLoss(policy, masks, latent_metric(cfg)).double(),
        "P": Preference(preference_config(history=5), len(cfg.alpha)).double(),
        "P_loss": PreferenceLoss().double(),
    }
    loss, result = micro_loss(
        modules, [tail, tail], [(0, 1, {"p": 0.5, "inc": 0})], (1, 1, 1, 1)
    )
    assert result.phi["z_norm"].shape == (2, 3)
    assert result.phi["changed"].shape == (2, 2)
    target = result.policy[1]
    assert target.shape == (2, 1, len(masks.names))
    assert target[..., masks.names.index("jump")].eq(1).all()
    loss.backward()
    for name in ("phi", "pi", "P"):
        grads = [p.grad for p in modules[name].parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    assert modules["phi"].head[-1].weight.grad.abs().sum() > 0
