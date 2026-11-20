"""Runtime construction and recorded factual shifts preserve FP64 search precision."""

import torch
import yaml

from mcpref.runtime.sampling.runner import Sampler
from mcpref.runtime.sampling.shards import load_step


def test_alpha_is_loaded_directly_in_fp64(site):
    path = site.root / "configs/models/perception.yaml"
    data = yaml.safe_load(path.read_text())
    data["alpha"] = [1.001, 2.001]
    path.write_text(yaml.safe_dump(data))
    site.modules.phi.bfloat16()
    sampler = Sampler(site.cfg, site.env, site.modules)
    assert sampler.alpha.dtype == torch.float64
    assert sampler.alpha.tolist() == [1.001, 2.001]


def test_bf16_factual_shift_is_subtracted_in_fp64(site, monkeypatch):
    def latent(index):
        value = 1.0 if index == 0 else -0.00390625
        return torch.tensor([value, value], dtype=torch.bfloat16)

    monkeypatch.setattr(site.modules.phi, "latent", latent)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    record = load_step(site.out, 0)
    assert record.delta.dtype == record.predicted.dtype == torch.float64
    assert record.delta.tolist() == [-1.00390625, -1.00390625]
    assert sampler.history.Y.dtype == torch.bfloat16
    assert sampler.tree.base.Y.dtype == sampler.tree.root.z.dtype == torch.float64
    sampler._changed = True
    sampler._raw = [sampler._observation]
    sampler._preceding = [record.action]
    sampler._prepare()
    assert sampler.tree.base.Y.dtype == sampler.tree.base.C.dtype == torch.float64
