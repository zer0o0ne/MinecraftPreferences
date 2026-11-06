"""Gradient collection, reduction and actual-update oracles of specs/pretrain-diagnostics.md (23)-(26)."""

import json
import math

import pytest
import torch

from mcpref.pretrain.gradient_work import collect_micro, reduce_work
from mcpref.pretrain.update import update_and_align, update_stats

from .gradient_oracle import modules, weighted
from .loop_modules import fakes
from .loop_oracle import terms
from .loop_world import clip, p
from .test_update import adam, config


def test_micro_loss_weighted_scalars_preserve_original_bitwise():
    from mcpref.pretrain.micro_loss import micro_loss

    item = p(0)
    mods = fakes(noise=False, stateful=False)
    factors = (0.5, 0.7, 1.9)
    original, result = micro_loss(mods, [clip(item["a"]), clip(item["b"])],
                                  [(0, 1, item["labels"])], factors)  # fmt: skip
    training, l_phi, l_pi, l_P = terms(
        [item], fakes(noise=False, stateful=False), clip, 1, 2, 1, 0.7, 1.9
    )
    expected = 0.5 * (l_phi + 0.7 * l_pi).sum() + 1.9 * l_P.sum()
    assert original.equal(training) and original.equal(expected)
    assert result.weighted["phi"].equal(0.5 * l_phi.sum())
    assert result.weighted["pi"].equal(0.5 * 0.7 * l_pi.sum())
    assert result.weighted["P"].equal(1.9 * l_P.sum())
    _, result = micro_loss(
        fakes(noise=False, stateful=False), [clip(item["a"])], [], factors
    )
    assert result.weighted["P"] is None


def test_collect_micro_exact_contributions_manifest_zero_and_original_state(tmp_path):
    mods, groups = modules()
    before = {
        f"{name}.{key}": parameter.detach().clone()
        for name, module in mods.items()
        for key, parameter in module.named_parameters()
    }
    for module in mods.values():
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
    gradients = {
        f"{name}.{key}": parameter.grad.clone()
        for name, module in mods.items()
        for key, parameter in module.named_parameters()
    }
    rng = torch.get_rng_state()
    got = collect_micro(weighted(mods), mods, groups, tmp_path, 0)
    assert got["micro_batch"] == 0 and got["labelled"] is True
    assert got["losses"] == {"phi": 7.5, "pi": 15.0, "P": 43.0}
    assert got["norms"]["phi"]["a"] == pytest.approx(
        {"phi": math.sqrt(17), "pi": math.sqrt(50), "P": math.sqrt(98)}
    )
    assert got["norms"]["P"]["z"]["P"] == 11
    manifest = json.loads((tmp_path / "000000/manifest.json").read_text())
    assert len(manifest) == 10 and all(
        "str(id" not in entry["parameter"] for entry in manifest.values()
    )
    expected = {
        "000000": ("phi", "phi", "a", "phi.a", [2], False),
        "000001": ("phi", "phi", "z", "phi.unused", [1], True),
        "000002": ("phi", "phi", "a", "phi_loss.dual", [], False),
        "000006": ("phi", "P", "a", "phi.a", [2], False),
        "000009": ("P", "P", "z", "P.w", [1], False),
    }
    for key, (x, component, group, parameter, shape, zero) in expected.items():
        entry = manifest[key]
        assert (
            entry["x"],
            entry["component"],
            entry["group"],
            entry["parameter"],
            entry["shape"],
            entry["zero"],
        ) == (x, component, group, parameter, shape, zero)
        path = tmp_path / "000000/tensors" / f"{key}.pt"
        assert path.exists() is not zero
        if not zero:
            tensor = torch.load(path)
            assert (
                tensor.device.type == "cpu"
                and tensor.shape == torch.Size(shape)
                and not tensor.requires_grad
            )
    assert torch.load(tmp_path / "000000/tensors/000000.pt").equal(
        torch.tensor([2.0, 2.0])
    )
    assert torch.load(tmp_path / "000000/tensors/000002.pt").equal(torch.tensor(3.0))
    assert torch.load(tmp_path / "000000/tensors/000006.pt").equal(
        torch.tensor([7.0, 7.0])
    )
    assert torch.load(tmp_path / "000000/tensors/000009.pt").equal(torch.tensor([11.0]))
    for name, module in mods.items():
        for key, parameter in module.named_parameters():
            assert parameter.detach().equal(before[f"{name}.{key}"])
            assert parameter.grad.equal(gradients[f"{name}.{key}"])
    assert torch.get_rng_state().equal(rng)


def test_collect_micro_unlabelled_has_no_preference_component(tmp_path):
    mods, groups = modules()
    values = weighted(mods)
    values["P"] = None
    got = collect_micro(values, mods, groups, tmp_path, 0)
    assert got["labelled"] is False and got["losses"]["P"] is None
    assert got["norms"]["P"] is None and got["nonfinite"]["P"] == {"P": None}
    manifest = json.loads((tmp_path / "000000/manifest.json").read_text())
    assert {entry["component"] for entry in manifest.values()} == {"phi", "pi"}


def test_collect_micro_nonfinite_counts_and_reduce_totals(tmp_path):
    mods, groups = modules()
    values = weighted(mods)
    values["P"] = (mods["phi"].a[0] * math.inf + mods["phi"].a[1] * -math.inf
                   + mods["phi_loss"].dual * math.nan + 0 * mods["P"].w.sum())  # fmt: skip
    got = collect_micro(values, mods, groups, tmp_path, 0)
    assert got["nonfinite"]["phi"]["P"]["a"] == {
        "nan": 1,
        "pos_inf": 1,
        "neg_inf": 1,
        "count": 3,
    }
    assert got["nonfinite"]["phi"]["P"]["z"]["count"] == 0
    reduced = reduce_work(tmp_path, groups=groups)
    assert reduced["nonfinite"]["phi"]["P"]["a"]["count"] == 3
    assert reduced["nonfinite"]["phi"]["P"]["z"]["count"] == 0


def test_reduce_work_sums_micro_before_metrics_and_uses_group_order(tmp_path):
    mods, groups = modules()
    for j, factor in enumerate((1.0, 2.0)):
        collect_micro(weighted(mods, factor), mods, groups, tmp_path, j)
    got = reduce_work(tmp_path, groups=groups)
    assert list(got["phi"]["groups"]) == ["z", "a"] and list(got["P"]["groups"]) == [
        "z"
    ]
    assert got["phi"]["groups"]["z"]["cosine"] == {
        "P_phi": None,
        "P_pi": None,
        "P_other": None,
    }
    a = got["phi"]["groups"]["a"]
    assert a["norms"] == pytest.approx(
        {"phi": math.sqrt(153), "pi": math.sqrt(450), "P": math.sqrt(882)}
    )
    assert a["dot"] == {"P_phi": 252, "P_pi": 630, "P_other": 882, "P_joint": 1764}
    assert a["cosine"]["P_phi"] == pytest.approx(252 / math.sqrt(153 * 882))
    assert a["cosine"]["P_pi"] == pytest.approx(630 / math.sqrt(450 * 882))
    assert a["cosine"]["P_other"] == pytest.approx(math.sqrt(882 / 963))
    assert got["P"]["total"]["norm"] == 33
    assert (
        got["phi"]["total"]["dot"]["P_other"]
        == got["phi"]["total"]["dot"]["P_phi"] + got["phi"]["total"]["dot"]["P_pi"]
    )
    with pytest.raises(TypeError):
        reduce_work(tmp_path)


def test_reduce_work_reads_one_parameter_across_micro_batches(tmp_path, monkeypatch):
    mods, groups = modules()
    for j in range(2):
        collect_micro(weighted(mods), mods, groups, tmp_path, j)
    real = torch.load
    reads = []

    def read(path, *args, **kwargs):
        reads.append(Path(path) if isinstance(path, (str, bytes)) else path)
        return real(path, *args, **kwargs)

    from pathlib import Path

    monkeypatch.setattr(torch, "load", read)
    reduce_work(tmp_path, groups=groups)
    names = [f"{path.parent.parent.name}/{path.name}" for path in reads]
    assert len(names) == 10
    for tensor_id in ("000000.pt", "000003.pt", "000006.pt", "000002.pt", "000009.pt"):
        positions = [i for i, name in enumerate(names) if name.endswith(tensor_id)]
        assert len(positions) == 2 and positions[1] == positions[0] + 1


def aligned(tmp_path, factor=1.0, zero_decay=False):
    mods, groups = modules()
    collect_micro(weighted(mods, factor), mods, groups, tmp_path, 0)
    lr = {"z": 1.0 if zero_decay else 0.1, "a": 0.2}
    decay = {"z": 1.0 if zero_decay else 0.0, "a": 0.1}
    cfg = config({q: {"lr": lr[q], "weight_decay": decay[q]} for q in lr})
    optimizer = adam(groups, cfg, lr, decay)
    loss = sum(weighted(mods).values())
    loss.backward()
    before = {
        f"{name}.{key}": parameter.detach().clone()
        for name, module in mods.items()
        for key, parameter in module.named_parameters()
    }
    optimizer.step()
    after = {
        f"{name}.{key}": parameter.detach().clone()
        for name, module in mods.items()
        for key, parameter in module.named_parameters()
    }
    stats, alignment = update_and_align(cfg, optimizer, groups, lr, tmp_path)
    assert stats == update_stats(cfg, optimizer, groups, lr)
    return alignment, before, after


def test_update_and_align_matches_actual_adamw_delta_and_signs(tmp_path):
    alignment, before, after = aligned(tmp_path)
    assert alignment["phi"]["z"]["update_dot"] == 0
    expected = 7 * sum(after["phi.a"] - before["phi.a"])
    assert alignment["phi"]["a"]["update_dot"] == pytest.approx(float(expected))
    assert alignment["P"]["z"]["update_dot"] == pytest.approx(
        float(11 * (after["P.w"] - before["P.w"]))
    )
    assert (
        alignment["phi"]["a"]["update_dot"] < 0
        and alignment["P"]["z"]["update_dot"] < 0
    )


def test_update_and_align_opposite_sign_and_zero_decay_reason(tmp_path):
    alignment, _, _ = aligned(tmp_path, factor=-1.0)
    assert (
        alignment["phi"]["a"]["update_dot"] > 0
        and alignment["P"]["z"]["update_dot"] > 0
    )
    other = aligned(tmp_path / "decay", zero_decay=True)[0]
    assert other["phi"]["z"]["update_dot"] is None
    assert other["phi"]["z"]["update_unavailable"] == "zero_decay_factor"
    assert (
        other["P"]["z"]["update_dot"] is None
        and other["P"]["z"]["update_unavailable"] == "zero_decay_factor"
    )


@pytest.mark.gpu
def test_collect_micro_restores_cpu_and_configured_cuda_rng(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda:0")
    mods, groups = modules()
    for module in mods.values():
        module.to(device)
    values = weighted(mods)
    for module in mods.values():
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
    before = {
        f"{name}.{key}": parameter.grad.clone()
        for name, module in mods.items()
        for key, parameter in module.named_parameters()
    }
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device)
    collect_micro(values, mods, groups, tmp_path, 0)
    assert torch.get_rng_state().equal(cpu)
    assert torch.cuda.get_rng_state(device).equal(cuda)
    for name, module in mods.items():
        for key, parameter in module.named_parameters():
            assert parameter.grad.equal(before[f"{name}.{key}"])
    for path in (tmp_path / "000000/tensors").glob("*.pt"):
        assert torch.load(path).device.type == "cpu"
