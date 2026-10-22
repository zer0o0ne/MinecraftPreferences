"""(6)-(8) of specs/perception.md: `Phi` on clips against a direct run of the tiny backbone
on the hand-built sequence and the head (oracle.phi_clip), in float64 and in float32 under
bfloat16 autocast (the head outside it); gradient checkpointing (use_reentrant=False, every
decoder layer and vision block, train mode only) on vs off: equal Z and gradients."""

import numpy as np
import pytest
import torch
import transformers.modeling_utils

from mcpref.perception.phi import Phi

from . import oracle
from .world import F64, backbone, config, random_clip, runs_backbone, tokenizer, values

pytestmark = runs_backbone


def clips() -> list:
    """Two clips: states with a gap, GUI states with and without a cursor on screen."""
    return [random_clip(1, [3, 4, 6], [True, False, True]),
            random_clip(2, [0, 1, 2], [False, True, False])]  # fmt: skip


def phi64(tmp_path, **changes) -> Phi:
    """Phi over the tiny float64 backbone, then all of it in float64 (the tests' dtype)."""
    return Phi(backbone(), tokenizer(), config(tmp_path, **changes)).to(F64)


def test_forward_against_the_direct_run(tmp_path):
    """Z (n, L, D) in the head's dtype; row i = phi(clip i) of the direct run."""
    phi = phi64(tmp_path)
    cs = clips()
    Z = phi(cs)
    assert Z.shape == (2, 3, 4) and Z.dtype == F64
    for i, c in enumerate(cs):
        torch.testing.assert_close(Z[i], oracle.phi_clip(phi, c, values(tmp_path)))


def test_forward_eval_mode_and_one_clip(tmp_path):
    """The same rows in eval mode, and for a clip alone (each clip its own forward)."""
    phi = phi64(tmp_path)
    cs = clips()
    Z = phi(cs)
    phi.eval()
    torch.testing.assert_close(phi(cs[1:])[0], Z[1])


@pytest.mark.parametrize(
    "changes",
    [{"marker": "##"}, {"head_layers": 0}, {"head_layers": 2, "head_width": 5},
     {"min_pixels": 8192, "image_mean": [0.1, 0.9, 0.3], "image_std": [1.0, 0.5, 2.0]},
     {"decimals": 3, "meta": "{x}{y}{z}{inventory}{hotbar}{yaw}{pitch}"}],
    ids=["multi-token marker", "no hidden layer", "two hidden layers",
         "other image arguments", "other meta"],
)  # fmt: skip
def test_forward_other_configs(tmp_path, changes):
    """The config's marker (readout at its last token), head, image arguments, meta."""
    phi = phi64(tmp_path, **changes)
    c = random_clip(4, [0, 2, 3, 5], [True, True, False, True])
    torch.testing.assert_close(
        phi([c])[0], oracle.phi_clip(phi, c, values(tmp_path, **changes))
    )


def test_forward_float32_under_autocast(tmp_path):
    """float32 parameters: the backbone under bfloat16 autocast, h_r cast to float32 and
    the head outside autocast; Z float32. The two wrong readings differ from it."""
    phi = Phi(backbone(torch.float32), tokenizer(), config(tmp_path))
    c = random_clip(3, [0, 1, 2], [True, False, False])
    v = values(tmp_path)
    Z = phi([c])[0]
    want = oracle.phi_clip(phi, c, v)
    assert Z.dtype == torch.float32
    torch.testing.assert_close(Z, want, rtol=1e-5, atol=1e-5)
    for wrong in (oracle.phi_clip(phi, c, v, autocast=False),
                  oracle.phi_clip(phi, c, v, head_inside=True)):  # fmt: skip
        assert (wrong - want).abs().max() > 1e-3


def checkpoint_calls(monkeypatch) -> list:
    """Records the keyword arguments of every torch checkpoint call `transformers` makes."""
    calls, original = [], transformers.modeling_utils.checkpoint

    def recorder(function, *args, **kwargs):
        calls.append(kwargs)
        return original(function, *args, **kwargs)

    monkeypatch.setattr(transformers.modeling_utils, "checkpoint", recorder)
    return calls


def grads(phi: Phi, Z: torch.Tensor) -> dict:
    weights = torch.tensor(np.random.default_rng(0).normal(size=Z.shape))
    (Z * weights).sum().backward()
    out = {n: p.grad.clone() for n, p in phi.named_parameters() if p.grad is not None}
    phi.zero_grad(set_to_none=True)
    return out


def test_gradient_checkpointing(tmp_path, monkeypatch):
    """Train mode: every decoder layer (2) and vision block (1) checkpointed per clip, with
    use_reentrant=False; Z and every gradient equal to a run without checkpointing; the
    gradients reach the head, the vision tower and the language model."""
    calls = checkpoint_calls(monkeypatch)
    phi = phi64(tmp_path)
    cs = clips()
    phi.train()
    Z_on = phi(cs)
    assert len(calls) == 6 and all(k == {"use_reentrant": False} for k in calls)
    g_on = grads(phi, Z_on)
    phi.backbone.gradient_checkpointing_disable()
    Z_off = phi(cs)
    assert len(calls) == 6
    g_off = grads(phi, Z_off)
    torch.testing.assert_close(Z_on, Z_off, rtol=0, atol=0)
    assert g_on.keys() == g_off.keys()
    for name in g_on:
        torch.testing.assert_close(g_on[name], g_off[name])
    for prefix in ("head.", "backbone.visual.", "backbone.language_model."):
        assert any(n.startswith(prefix) and g.abs().sum() > 0 for n, g in g_on.items())


def test_no_checkpointing_in_eval_mode(tmp_path, monkeypatch):
    calls = checkpoint_calls(monkeypatch)
    phi = phi64(tmp_path).eval()
    phi(clips())
    assert calls == []
