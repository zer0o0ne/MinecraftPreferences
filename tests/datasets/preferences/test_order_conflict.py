"""Two-order contradictions, using an exhaustive hand-computed preference table."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from datasets.preferences.acyclic_world import MB, MW, SB, START, SW, B, W
from datasets.preferences.test_verdict import A, Bref, cfg
from mcpref.datasets.preferences.acyclic import load_config, shard_pairs
from mcpref.datasets.preferences.labels import load_config as load_labels
from mcpref.datasets.preferences.verdict import verdict

TOKENS = [MB, B, SB, SW, W, MW]
# Rows: order 0; columns: order 1; entries: p(A > B), or incomparable.
P = [
    [None, None, None, "0.8", "0.9", "1"],
    [None, None, None, "0.7", "0.8", "0.9"],
    [None, None, "0.5", "0.6", "0.7", "0.8"],
    ["0.2", "0.3", "0.4", "0.5", None, None],
    ["0.1", "0.2", "0.3", None, None, None],
    ["0", "0.1", "0.2", None, None, None],
]


@pytest.mark.parametrize("i", range(len(TOKENS)))
@pytest.mark.parametrize("j", range(len(TOKENS)))
@pytest.mark.parametrize("spelling", ["legacy", "explicit", "opaque"])
def test_two_order_table(i, j, spelling):
    p = P[i][j]
    if p is None:
        expected = ("incomparable", A, Bref, None)
    else:
        p = Fraction(p)
        expected = (
            ("comparable", A, Bref, p)
            if p >= Fraction(1, 2)
            else ("comparable", Bref, A, 1 - p)
        )
    config = cfg()
    aliases = {
        token: {"legacy": token, "explicit": f"A_IS_{token}_THAN_B", "opaque": f"X{k}"}[
            spelling
        ]
        for k, token in enumerate(config.weights)
    }
    config = replace(
        config,
        weights={aliases[t]: w for t, w in config.weights.items()},
        order_categories={aliases[t]: c for t, c in config.order_categories.items()},
    )
    assert (
        verdict(A, Bref, [aliases[TOKENS[i]], aliases[TOKENS[j]]], config) == expected
    )


def test_token_meaning_is_independent_of_numeric_weights():
    weights = {**START, B: 0.1, SB: 0.9}
    assert verdict(A, Bref, [B, SB], cfg(weights)) == ("incomparable", A, Bref, None)


def test_multiple_samples_keep_the_existing_mean():
    assert verdict(A, Bref, [B, B, SB, SB], cfg()) == (
        "comparable",
        A,
        Bref,
        Fraction(3, 5),
    )


def test_numeric_token_configured_as_refusal_is_not_a_conflict():
    weights = {B: None, SB: 0.6}
    assert verdict(A, Bref, [B, SB], cfg(weights)) == (
        "comparable",
        Bref,
        A,
        Fraction(3, 5),
    )


def test_categories_override_the_spelling_of_known_tokens():
    config = cfg()
    config = replace(config, order_categories={**config.order_categories, B: "worse"})
    assert verdict(A, Bref, [B, W], config) == ("incomparable", A, Bref, None)
    assert verdict(A, Bref, [B, SB], config) == ("comparable", A, Bref, Fraction(3, 5))


@pytest.mark.parametrize(
    "filename, prefix, suffix",
    [
        ("acyclic.yaml", "", ""),
        ("acyclic-shared.yaml", "A_IS_", "_THAN_B"),
    ],
)
def test_real_config_conflicts_reach_corrected_pairs(filename, prefix, suffix):
    root = Path(__file__).resolve().parents[3]
    config = load_config(root / "configs/datasets/preferences" / filename)
    tokens = [f"{prefix}{t}{suffix}" for t in (MB, B)]
    comparison = {"a": A, "b": Bref, "requests": [{"label": t} for t in tokens]}
    pairs, counts, categories = shard_pairs(0, [comparison], config)
    assert pairs[0]["labels"] == {"p": None, "inc": 1}
    assert counts == {"comparable": 0, "removed": 0, "incomparable": 1, "dropped": 0}
    assert categories == ["incomparable"]


@pytest.mark.parametrize("filename", ["labels.yaml", "labels-shared.yaml"])
def test_real_label_answers_match_both_correction_mappings(filename):
    root = Path(__file__).resolve().parents[3]
    labels = load_labels(root / "configs/datasets/preferences" / filename)
    correction = load_config(root / labels.acyclic)
    assert [answer["token"] for answer in labels.answers] == list(correction.weights)
    assert correction.order_categories.keys() == correction.weights.keys()
