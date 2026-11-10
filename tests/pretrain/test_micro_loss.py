"""PT-1 of docs/runtime-optimization-plan.md: one prepared state pair per clip."""

from types import MethodType

import numpy as np
import torch
from perception import oracle
from perception.world import F64, backbone, config, random_clip, tokenizer, values

import mcpref.perception.loss as loss_module
import mcpref.perception.phi as phi_module
from mcpref.perception.loss import PhiLoss
from mcpref.perception.phi import Phi
from mcpref.pretrain.micro_loss import micro_loss

from .loop_modules import fakes

FACTORS = (0.5, 0.4, 0.7, 1.9)


def clips() -> list:
    out = [
        random_clip(1, [3, 4, 6], [True, False, True]),
        random_clip(2, [0, 1, 2], [False, True, False]),
    ]
    for j, clip in enumerate(out):
        clip.info["a"] = [j, (j + 1) % 3, (j + 2) % 3]
        clip.lines[0]["w"] = 0.5 + j
    return out


def original_loss(modules: dict, clips: list, pairs: list) -> tuple:
    """The pre-PT-1 call path, with independent preparation in phi and its loss."""
    k_factor, w_phi, w_pi, r_factor = FACTORS
    Z = modules["phi"](clips)
    phi_terms, diagnostics = modules["phi_loss"].terms(Z, clips)
    pi_out = modules["pi"](Z, clips)
    pi_terms, _, _ = modules["pi_loss"].terms(pi_out, Z, clips)
    loss = k_factor * (w_phi * phi_terms["total"] + w_pi * pi_terms["total"]).sum()
    if pairs:
        i_a, i_b, labels = (list(x) for x in zip(*pairs))
        out = modules["P"](Z[i_a], Z[i_b])
        loss = loss + r_factor * modules["P_loss"].terms(out, labels)["total"].sum()
    return loss, Z, diagnostics


def gradients(modules: dict) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": parameter.grad.detach().clone()
        for name, module in modules.items()
        for key, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def test_micro_loss_prepares_once_and_matches_original_bitwise(tmp_path, monkeypatch):
    cfg = config(tmp_path, alpha=[1.0, 2.0])
    phi = Phi(backbone(), tokenizer(), cfg).to(F64)
    phi.A = torch.nn.Parameter(
        torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.7, -0.2]], dtype=F64)
    )
    phi.c = torch.nn.Parameter(torch.tensor([0.1, -0.3], dtype=F64))
    model_inputs = []

    def frames_z(self, frames, metas):
        model_inputs.append((frames, metas))
        x = (
            torch.tensor(np.stack([frame.mean((0, 1)) for frame in frames]), dtype=F64)
            / 255
        )
        return x @ self.A.T + self.c

    phi.frames_z = MethodType(frames_z, phi)
    modules = fakes(noise=False, stateful=False)
    modules["phi"], modules["phi_loss"] = phi, PhiLoss(cfg).to(F64)
    cs, pairs = clips(), [(0, 1, {"p": 1.0, "inc": 0})]

    expected_loss, expected_Z, expected_diagnostics = original_loss(modules, cs, pairs)
    expected_loss.backward()
    expected_gradients = gradients(modules)
    for module in modules.values():
        module.zero_grad(set_to_none=True)
    model_inputs.clear()

    calls = []
    phi_states, loss_states = phi_module.clip_states, loss_module.clip_states

    def count_phi_states(*args):
        calls.append(args[0])
        return phi_states(*args)

    def count_loss_states(*args):
        calls.append(args[0])
        return loss_states(*args)

    changed_inputs = []
    real_changed = loss_module.changed

    def record_changed(frames, metas, tau):
        changed_inputs.append((frames, metas))
        return real_changed(frames, metas, tau)

    monkeypatch.setattr(phi_module, "clip_states", count_phi_states)
    monkeypatch.setattr(loss_module, "clip_states", count_loss_states)
    monkeypatch.setattr(loss_module, "changed", record_changed)
    pi_log = []
    modules["pi"].log = pi_log
    loss, result = micro_loss(modules, cs, pairs, FACTORS)
    loss.backward()

    assert calls == cs
    assert len(model_inputs) == len(changed_inputs) == len(cs)
    for clip, model, changed in zip(cs, model_inputs, changed_inputs):
        assert model[0] is changed[0] and model[1] is changed[1]
        for frame, expected in zip(model[0], oracle.frames(clip)):
            np.testing.assert_array_equal(frame, expected)
        assert model[1] == oracle.metas(clip, values(tmp_path, alpha=[1.0, 2.0]))
    assert result.phi["changed"].tolist() == [
        oracle.chi(c, values(tmp_path)) for c in cs
    ]

    assert loss.equal(expected_loss)
    torch.testing.assert_close(pi_log[0][2]["Z"], expected_Z, rtol=0, atol=0)
    for key, expected in expected_diagnostics.items():
        torch.testing.assert_close(result.phi[key], expected, rtol=0, atol=0)
    actual_gradients = gradients(modules)
    assert actual_gradients.keys() == expected_gradients.keys()
    for key, expected in expected_gradients.items():
        torch.testing.assert_close(actual_gradients[key], expected, rtol=0, atol=0)
