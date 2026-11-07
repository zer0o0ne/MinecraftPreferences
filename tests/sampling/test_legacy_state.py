"""Strict rejection of incomplete legacy continuation state of env-sampling.md (7)."""

import pytest
import torch

from mcpref.sampling.runner import Sampler
from mcpref.sampling.shards import write_state

from .helpers import FakeEnv, FakePhi
from .test_cache import CACHE_FIELDS
from .test_shards import sampler_state

MISSING_FIELDS = [
    "env",
    "env.info",
    "env.info.isGuiOpen",
    "env.info.custom",
    "env.info.mine_block",
    "env.info.use_item",
    "env.cursor",
    "env.held_action",
    *(f"stream.cache.0.{name}" for name in sorted(CACHE_FIELDS)),
]


def remove_path(mapping, path):
    parts = path.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target[int(part) if part.isdigit() else part]
    target.pop(parts[-1])


@pytest.mark.parametrize("missing", MISSING_FIELDS)
def test_legacy_missing_required_continuation_field_rejected_before_restore(
    site, missing
):
    shard = site.out / "shards/000000"
    shard.mkdir(parents=True)
    write_state(shard, sampler_state(FakePhi()))
    env = FakeEnv()
    sampler = Sampler(site.cfg, env, site.modules)
    path = shard / "state.pt"
    raw = torch.load(path, map_location="cpu", weights_only=True)
    remove_path(raw, missing)
    torch.save(raw, path)

    with pytest.raises(RuntimeError) as error:
        sampler.reset()
    assert str(error.value) == "incompatible sampling state"
    assert (env.resets, env.restores, env.events) == (0, 0, [])
