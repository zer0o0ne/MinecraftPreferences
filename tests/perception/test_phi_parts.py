"""Parts of `Phi` of specs/perception.md that run no backbone: the attributes (Notation), the
head (7) — its layers and its initialisation under its own seed —, the groups (12) against a
transcription of the decay rule of [T], and `build_phi` (Interface) with the library's
`from_pretrained` calls replaced by fakes recording their arguments."""

import re

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3_5Model

from mcpref.perception.phi import Phi, build_phi

from . import oracle
from .world import D_H, F64, backbone, config, tokenizer

GROUPS = ["phi_vision", "phi_vision_no_decay", "phi_llm", "phi_llm_no_decay",
          "phi_head", "phi_head_no_decay"]  # fmt: skip
PATTERNS = [
    r"bias",
    r"layernorm",
    r"rmsnorm",
    r"(?:^|\.)norm(?:$|\.)",
    r"_norm(?:$|\.)",
]


def build(tmp_path, dtype=F64, **changes) -> Phi:
    return Phi(backbone(dtype), tokenizer(), config(tmp_path, **changes))


def layer_shapes(head) -> list:
    return [(type(m).__name__, getattr(m, "in_features", None), getattr(m, "out_features", None))
            for m in head]  # fmt: skip


@pytest.mark.parametrize(
    "layers, width, want",
    [
        (0, 8, [("Linear", D_H, 4)]),
        (1, 8, [("Linear", D_H, 8), ("SiLU", None, None), ("Linear", 8, 4)]),
        (2, 5, [("Linear", D_H, 5), ("SiLU", None, None), ("Linear", 5, 5),
                ("SiLU", None, None), ("Linear", 5, 4)]),
    ],
)  # fmt: skip
def test_head_layers(tmp_path, layers, width, want):
    """(7): `head_layers` blocks Linear -> SiLU of width `head_width` (the first from d_h),
    then Linear to D = len(alpha), with biases; a Sequential."""
    phi = build(tmp_path, head_layers=layers, head_width=width)
    assert isinstance(phi.head, nn.Sequential)
    assert layer_shapes(phi.head) == want
    assert all(m.bias is not None for m in phi.head if isinstance(m, nn.Linear))


def test_attributes(tmp_path):
    """`backbone` is B; the names of named_parameters() and the keys of state_dict() are
    those of the attributes `backbone` and `head`; the head applies (7)."""
    b = backbone()
    phi = Phi(b, tokenizer(), config(tmp_path))
    assert phi.backbone is b
    names = [n for n, _ in phi.named_parameters()]
    want = [f"backbone.{n}" for n, _ in b.named_parameters()]
    want += [f"head.{n}" for n, _ in phi.head.named_parameters()]
    assert sorted(names) == sorted(want)
    assert all(k.split(".")[0] in ("backbone", "head") for k in phi.state_dict())
    h = torch.randn(3, D_H, dtype=F64)
    phi.to(F64)
    torch.testing.assert_close(phi.head(h), oracle.head(phi.head, h))
    # (7): h cast to the dtype of the head's parameters
    h32 = h.to(torch.float32)
    got = phi.head(h32)
    assert got.dtype == F64
    torch.testing.assert_close(got, oracle.head(phi.head, h32.to(F64)))


def head_values(phi: Phi) -> dict:
    return {k: v.clone() for k, v in phi.head.state_dict().items()}


def test_head_initialisation(tmp_path):
    """torch.nn.Linear's default initialisation drawn after manual_seed(seed) of the CPU
    generator inside fork_rng: the layers created in order, float32 as created."""
    phi = build(tmp_path, dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        want = nn.Sequential(nn.Linear(D_H, 8), nn.SiLU(), nn.Linear(8, 4))
    assert next(phi.head.parameters()).dtype == torch.float32
    for k, v in want.state_dict().items():
        assert torch.equal(phi.head.state_dict()[k], v), k


def test_head_initialisation_seeded(tmp_path):
    """Equal for equal `seed`, whatever the global generator's state; another seed gives
    another head; the global generator's state unchanged."""
    torch.manual_seed(1)
    a = head_values(build(tmp_path / "a"))
    torch.manual_seed(2)
    state = torch.get_rng_state()
    b = head_values(build(tmp_path / "b"))
    assert torch.equal(torch.get_rng_state(), state)
    c = head_values(build(tmp_path / "c", seed=8))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not any(torch.equal(a[k], c[k]) for k in a if k.endswith("weight"))


def decayed(phi: Phi) -> set[str]:
    """[T]: a parameter is decayed iff no module on its path is an nn.LayerNorm and its
    lower-cased full name matches none of the five patterns."""
    modules = dict(phi.named_modules())
    out = set()
    for name, _ in phi.named_parameters():
        parts = name.split(".")[:-1]
        path = [".".join(parts[:k]) for k in range(len(parts) + 1)]
        if any(isinstance(modules[p], nn.LayerNorm) for p in path):
            continue
        if not any(re.search(pat, name.lower()) for pat in PATTERNS):
            out.add(name)
    return out


def test_groups(tmp_path):
    """(12): six groups in this order, partitioning the parameters (by identity) of
    B.visual, B.language_model and the head into the decayed ones and the others, each list
    in named_parameters() order."""
    phi = build(tmp_path)
    groups = phi.groups()
    assert list(groups) == GROUPS
    named = list(phi.named_parameters())
    dec = decayed(phi)
    prefixes = ["backbone.visual.", "backbone.language_model.", "head."]
    for (name, no_decay), prefix in zip(zip(GROUPS[::2], GROUPS[1::2]), prefixes):
        want = [p for n, p in named if n.startswith(prefix) and n in dec]
        want_no = [p for n, p in named if n.startswith(prefix) and n not in dec]
        assert [id(p) for p in groups[name]] == [id(p) for p in want], name
        assert [id(p) for p in groups[no_decay]] == [id(p) for p in want_no], no_decay
    ids = [id(p) for ps in groups.values() for p in ps]
    assert sorted(ids) == sorted(id(p) for p in phi.parameters())


def test_groups_bias_and_norm(tmp_path):
    """A bias and a norm weight in the no-decay groups; a Linear weight decayed."""
    phi = build(tmp_path)
    groups, params = phi.groups(), dict(phi.named_parameters())

    def where(name):
        return [q for q, ps in groups.items() if any(p is params[name] for p in ps)]

    assert where("head.0.bias") == ["phi_head_no_decay"]
    assert where("head.0.weight") == ["phi_head"]
    assert where("backbone.language_model.norm.weight") == ["phi_llm_no_decay"]
    assert where("backbone.language_model.layers.0.linear_attn.dt_bias") == ["phi_llm_no_decay"]  # fmt: skip
    assert where("backbone.visual.blocks.0.norm1.weight") == ["phi_vision_no_decay"]
    assert where("backbone.visual.blocks.0.attn.qkv.weight") == ["phi_vision"]


def test_groups_against_the_library(tmp_path):
    """The transcription agrees with `get_parameter_names` on the tiny Phi."""
    from transformers.trainer_pt_utils import get_parameter_names

    phi = build(tmp_path)
    assert set(get_parameter_names(phi, [nn.LayerNorm], PATTERNS)) == decayed(phi)


def test_build_phi(tmp_path, monkeypatch):
    """B by Qwen3_5Model.from_pretrained(model, revision=..., dtype=torch.float32,
    attn_implementation="sdpa"), the tokenizer by AutoTokenizer.from_pretrained(model,
    revision=...); Phi on the CPU in float32 over that B."""
    cfg = config(tmp_path)
    calls, b = [], backbone(torch.float32)

    def fake(result):
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        return staticmethod(from_pretrained)

    monkeypatch.setattr(Qwen3_5Model, "from_pretrained", fake(b))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake(tokenizer()))
    phi = build_phi(cfg)
    rev = cfg.revision
    want_b = (("tiny/model",), {"revision": rev, "dtype": torch.float32, "attn_implementation": "sdpa"})  # fmt: skip
    want_t = (("tiny/model",), {"revision": rev})
    assert sorted(calls, key=lambda c: len(c[1])) == [want_t, want_b]
    assert isinstance(phi, Phi) and phi.backbone is b
    assert {p.dtype for p in phi.parameters()} == {torch.float32}
    assert {p.device.type for p in phi.parameters()} == {"cpu"}
