"""`load_config` of src/mcpref/pretrain/config.py against specs/pretrain.md (Notation: the
config; Interface: starting values): the fields, their constraints (`ValueError` at load);
`integer` of `mcpref`, moved from `clips/collect.py`."""

import dataclasses
import math

import pytest
import torch

import mcpref
import mcpref.clips.collect
from mcpref.pretrain.config import PretrainConfig, load_config

PATHS = ["index_dir", "out_dir", "perception", "action_mask", "policy", "preference"]
NAMES = [*PATHS, "seed", "epochs", "batch", "micro_batch", "checkpoint_steps",
         "keep_checkpoints", "warmup_steps", "w_pi", "w_P", "grad_clip", "adam_eps",
         "lr_floor", "betas", "groups", "device"]  # fmt: skip
VALID = {  # YAML lines of a valid config
    "index_dir": "./x/index",
    "out_dir": "o/p/",
    "perception": "c/perception.yaml",
    "action_mask": "./c/action-mask.yaml",
    "policy": "c/sub/policy.yaml",
    "preference": "preference.yaml",
    "seed": "4294967295",
    "epochs": "1",
    "batch": "3",
    "micro_batch": "2",
    "checkpoint_steps": "1",
    "keep_checkpoints": "1",
    "warmup_steps": "0",
    "w_pi": "2",
    "w_P": "0.5",
    "grad_clip": "1.0e+9",
    "adam_eps": "1.0e-8",
    "lr_floor": "0",
    "betas": "[0, 0.999]",
    "groups": "{b: {lr: 1, weight_decay: 0}, a: {weight_decay: 0.25, lr: 3.0e-4}}",
    "device": '"cpu"',
}


def load(tmp_path, **changes: str | None) -> PretrainConfig:
    """A config of the VALID lines with `changes` (None: the line left out)."""
    lines = {k: v for k, v in (VALID | changes).items() if v is not None}
    path = tmp_path / "c.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in lines.items()))
    return load_config(path)


def test_reads_every_field(tmp_path):
    """The fields, the module configs following out_dir; strings as written, integers,
    numbers; betas a tuple of floats; groups in the config's order with float lr and
    weight_decay; frozen."""
    cfg = load(tmp_path)
    fields = [f.name for f in dataclasses.fields(PretrainConfig)]
    assert set(fields) == set(NAMES) and fields.index("out_dir") + 1 == fields.index("perception")  # fmt: skip
    assert fields[fields.index("perception") :][:4] == PATHS[2:]
    assert (cfg.index_dir, cfg.out_dir) == ("./x/index", "o/p/")
    assert (cfg.perception, cfg.action_mask, cfg.policy, cfg.preference) == (
        "c/perception.yaml", "./c/action-mask.yaml", "c/sub/policy.yaml", "preference.yaml"
    )  # fmt: skip
    ints = (cfg.seed, cfg.epochs, cfg.batch, cfg.micro_batch, cfg.checkpoint_steps,
            cfg.keep_checkpoints, cfg.warmup_steps)  # fmt: skip
    assert ints == (2**32 - 1, 1, 3, 2, 1, 1, 0) and all(type(v) is int for v in ints)
    assert (cfg.w_pi, cfg.w_P, cfg.grad_clip, cfg.adam_eps, cfg.lr_floor) == (
        2, 0.5, 1e9, 1e-8, 0
    )  # fmt: skip
    assert cfg.betas == (0.0, 0.999) and type(cfg.betas) is tuple
    assert all(type(b) is float for b in cfg.betas)
    assert cfg.groups == {"b": {"lr": 1.0, "weight_decay": 0.0},
                          "a": {"weight_decay": 0.25, "lr": 3e-4}}  # fmt: skip
    assert list(cfg.groups) == ["b", "a"]
    assert all(type(v) is float for g in cfg.groups.values() for v in g.values())
    assert torch.device(cfg.device) == torch.device("cpu")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 2


ACCEPTED = {  # boundaries inside the domain: field, YAML, value
    "seed 0": ("seed", "0", 0),
    "warmup_steps large": ("warmup_steps", "1000000000000", 10**12),
    "lr_floor 1": ("lr_floor", "1", 1),
    "lr_floor 0.5": ("lr_floor", "0.5", 0.5),
    "grad_clip inf": ("grad_clip", ".inf", math.inf),
    "w_pi tiny": ("w_pi", "1.0e-300", 1e-300),
    "groups empty": ("groups", "{}", {}),
}


@pytest.mark.parametrize("name, text, value", ACCEPTED.values(), ids=ACCEPTED)
def test_accepted_values(tmp_path, name, text, value):
    got = getattr(load(tmp_path, **{name: text}), name)
    assert got == value


def test_device_of_another_machine(tmp_path):
    """Any string torch.device accepts; loading needs no GPU."""
    got = load(tmp_path, device='"cuda:1"').device
    assert torch.device(got) == torch.device("cuda:1")


def test_betas_from_integers(tmp_path):
    assert load(tmp_path, betas="[0, 0]").betas == (0.0, 0.0)


BAD = {
    "missing key": {"seed": None},
    "missing groups": {"groups": None},
    "unknown key": {"lr": "0.1"},
    "index_dir empty": {"index_dir": '""'},
    "index_dir absolute": {"index_dir": "/abs/x"},
    "index_dir with ..": {"index_dir": "x/../y"},
    "out_dir ..": {"out_dir": ".."},
    "out_dir not a string": {"out_dir": "3"},
    "out_dir null": {"out_dir": "null"},
    **{f"{k} missing": {k: None} for k in PATHS[2:]},
    **{f"{k} empty": {k: '""'} for k in PATHS[2:]},
    **{f"{k} absolute": {k: "/abs/c.yaml"} for k in PATHS[2:]},
    **{f"{k} with ..": {k: "c/../../c.yaml"} for k in PATHS[2:]},
    **{f"{k} not a string": {k: "1"} for k in PATHS[2:]},
    **{f"{k} a list": {k: "[c.yaml]"} for k in PATHS[2:]},
    "seed 2^32": {"seed": "4294967296"},
    "seed negative": {"seed": "-1"},
    "seed a boolean": {"seed": "true"},
    "seed a float": {"seed": "1.0"},
    "seed a string": {"seed": '"1"'},
    **{
        f"{k} 0": {k: "0"}
        for k in (
            "epochs",
            "batch",
            "micro_batch",
            "checkpoint_steps",
            "keep_checkpoints",
        )
    },
    **{
        f"{k} a boolean": {k: "true"}
        for k in (
            "epochs",
            "batch",
            "micro_batch",
            "checkpoint_steps",
            "keep_checkpoints",
        )
    },
    "batch a float": {"batch": "2.0"},
    "micro_batch a string": {"micro_batch": '"2"'},
    "warmup_steps negative": {"warmup_steps": "-1"},
    "warmup_steps a boolean": {"warmup_steps": "false"},
    "warmup_steps a float": {"warmup_steps": "3.0"},
    **{f"{k} 0": {k: "0"} for k in ("w_pi", "w_P", "grad_clip", "adam_eps")},
    **{f"{k} negative": {k: "-0.5"} for k in ("w_pi", "w_P", "grad_clip", "adam_eps")},
    **{f"{k} a string": {k: '"1"'} for k in ("w_pi", "w_P", "grad_clip", "adam_eps")},
    **{f"{k} a boolean": {k: "true"} for k in ("w_pi", "w_P", "grad_clip", "adam_eps")},
    "grad_clip nan": {"grad_clip": ".nan"},
    "adam_eps 1e-8 (a string to PyYAML)": {"adam_eps": "1e-8"},
    "lr_floor above 1": {"lr_floor": "1.0001"},
    "lr_floor negative": {"lr_floor": "-0.1"},
    "lr_floor a string": {"lr_floor": '"0.1"'},
    "one beta": {"betas": "[0.9]"},
    "three betas": {"betas": "[0.9, 0.99, 0.999]"},
    "betas a scalar": {"betas": "0.9"},
    "beta 1": {"betas": "[0.9, 1]"},
    "beta negative": {"betas": "[-0.1, 0.9]"},
    "betas strings": {"betas": '["0.9", "0.99"]'},
    "beta a boolean": {"betas": "[false, 0.9]"},
    "groups a list": {"groups": "[a]"},
    "groups null": {"groups": "null"},
    "group name not a string": {"groups": "{1: {lr: 1, weight_decay: 0}}"},
    "group not a mapping": {"groups": "{a: 1}"},
    "group without lr": {"groups": "{a: {weight_decay: 0}}"},
    "group without weight_decay": {"groups": "{a: {lr: 1}}"},
    "group with an extra key": {"groups": "{a: {lr: 1, weight_decay: 0, betas: 1}}"},
    "group lr 0": {"groups": "{a: {lr: 0, weight_decay: 0}}"},
    "group lr a string": {"groups": '{a: {lr: "1", weight_decay: 0}}'},
    "group weight_decay negative": {"groups": "{a: {lr: 1, weight_decay: -1.0e-9}}"},
    "group weight_decay a boolean": {"groups": "{a: {lr: 1, weight_decay: false}}"},
    "bad device": {"device": '"gpu"'},
    "device not a string": {"device": "0"},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD)
def test_violated_constraints_raise_value_error(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


def group(lr: float, weight_decay: float) -> dict:
    return {"lr": lr, "weight_decay": weight_decay}


STARTING_GROUPS = {  # specs/perception.md, specs/policy.md, specs/preference.md (Interface)
    "phi_vision": group(1e-5, 0.01), "phi_vision_no_decay": group(1e-5, 0.0),
    "phi_llm": group(1e-5, 0.01), "phi_llm_no_decay": group(1e-5, 0.0),
    "phi_head": group(1e-4, 0.01), "phi_head_no_decay": group(1e-4, 0.0),
    "phi_dual": group(1e-4, 0.0),
    "pi": group(3e-4, 0.01), "pi_no_decay": group(3e-4, 0.0),
    "P": group(3e-4, 0.01), "P_no_decay": group(3e-4, 0.0),
}  # fmt: skip


def test_starting_values(repo):
    """configs/pretrain.yaml as Interface starts it: the module configs; the groups of
    specs/perception.md, specs/policy.md and specs/preference.md."""
    cfg = load_config(repo / "configs/pretrain.yaml")
    assert torch.device(cfg.device) == torch.device("cuda")
    assert dataclasses.asdict(cfg) | {"device": "cuda"} == {
        "index_dir": "data/train_index", "out_dir": "data/pretrain",
        "perception": "configs/perception.yaml", "action_mask": "configs/action-mask.yaml",
        "policy": "configs/policy.yaml", "preference": "configs/preference.yaml", "seed": 0,
        "epochs": 1, "batch": 32, "micro_batch": 1, "checkpoint_steps": 500,
        "keep_checkpoints": 2, "warmup_steps": 100, "w_pi": 1.0, "w_P": 1.0,
        "grad_clip": 1.0, "adam_eps": 1e-8, "lr_floor": 0.1, "betas": (0.9, 0.999),
        "groups": STARTING_GROUPS, "device": "cuda",
    }  # fmt: skip
    assert list(cfg.groups) == list(STARTING_GROUPS)  # the eleven groups in this order


INTEGERS = {  # value, low, high -> result, or None for ValueError
    "inside": (3, 0, 5, 3),
    "low": (0, 0, 5, 0),
    "high": (5, 0, 5, 5),
    "unbounded": (10**20, 0, math.inf, 10**20),
    "below": (-1, 0, 5, None),
    "above": (6, 0, 5, None),
    "a boolean": (True, 0, 5, None),
    "a float": (2.0, 0, 5, None),
    "a string": ("2", 0, 5, None),
}


@pytest.mark.parametrize("value, low, high, want", INTEGERS.values(), ids=INTEGERS)
def test_integer(value, low, high, want):
    """`integer`: a YAML integer, not a boolean, in [low, high], else ValueError."""
    if want is None:
        with pytest.raises(ValueError):
            mcpref.integer("x", value, low, high)
    else:
        got = mcpref.integer("x", value, low, high)
        assert got == want and type(got) is int


def test_integer_moved_from_clips():
    """`integer` lives in mcpref; clips/collect.py keeps no copy of it."""
    assert mcpref.integer.__module__ == "mcpref"
    assert getattr(mcpref.clips.collect, "integer", mcpref.integer) is mcpref.integer
