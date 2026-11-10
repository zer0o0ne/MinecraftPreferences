"""DiagnosticsConfig constraints of specs/rl-diagnostics.md Notation."""

import dataclasses

import pytest

from diagnostics.core.world import STARTING, config, write_config
from mcpref.diagnostics.core.config import DiagnosticsConfig, load_config


def test_starting_values_types_and_frozen(tmp_path):
    cfg = load_config(write_config(tmp_path / "diagnostics.yaml"))
    assert dataclasses.asdict(cfg) == STARTING | {"quantiles": (0.05, 0.5, 0.95)}
    assert type(cfg.quantiles) is tuple
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


@pytest.mark.parametrize("key", STARTING)
def test_missing_key(tmp_path, key):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path / "c.yaml", **{key: None}))


def test_unknown_key(tmp_path):
    path = write_config(tmp_path / "c.yaml")
    path.write_text(path.read_text() + "extra: 1\n")
    with pytest.raises(ValueError):
        load_config(path)


BAD = {
    "out_dir empty": {"out_dir": ""},
    "out_dir absolute": {"out_dir": "/data/diagnostics"},
    "out_dir parent": {"out_dir": "data/../diagnostics"},
    "out_dir type": {"out_dir": 1},
    "seed negative": {"seed": -1},
    "seed high": {"seed": 2**32},
    "seed boolean": {"seed": True},
    "seed float": {"seed": 0.0},
    "ci zero": {"ci_level": 0},
    "ci one": {"ci_level": 1},
    "ci boolean": {"ci_level": True},
    "bootstrap zero": {"bootstrap_samples": 0},
    "bootstrap float": {"bootstrap_samples": 1.0},
    "clusters zero": {"min_cluster_count": 0},
    "clusters boolean": {"min_cluster_count": False},
    "quantiles empty": {"quantiles": []},
    "quantiles duplicate": {"quantiles": [0.5, 0.5]},
    "quantiles unsorted": {"quantiles": [0.9, 0.5]},
    "quantiles low": {"quantiles": [0.0, 0.5]},
    "quantiles high": {"quantiles": [0.5, 1.0]},
    "quantiles boolean": {"quantiles": [True]},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_violated_constraints(tmp_path, changes):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path / "c.yaml", **changes))


def test_valid_boundaries():
    cfg = config(
        seed=2**32 - 1,
        ci_level=1e-9,
        bootstrap_samples=1,
        min_cluster_count=1,
        quantiles=[0.5],
    )
    assert cfg.seed == 2**32 - 1 and cfg.quantiles == (0.5,)
    assert isinstance(cfg, DiagnosticsConfig)
