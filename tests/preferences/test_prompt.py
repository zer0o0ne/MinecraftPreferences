"""src/mcpref/preferences/prompt.py against specs/vlm-labels.md: (1) the prompt of (x, y),
(2) the label of an answer; hand cases on a tiny config and on the starting one."""

import numpy as np
import pytest

from mcpref.preferences.prompt import Shown, parse, prompt

from .config_files import STARTING, TINY, TINY_THETA, load
from .labels_oracle import same_part


def frame(code: int) -> np.ndarray:
    """A distinct (2, 4, 3) uint8 RGB frame."""
    return np.full((2, 4, 3), code, dtype=np.uint8)


def item(type_: str, quantity: int) -> dict:
    return {"type": type_, "quantity": quantity}


def same_parts(actual: list, expected: list) -> None:
    """A list of parts equal to `expected` part by part (labels_oracle.same_part)."""
    assert isinstance(actual, list) and len(actual) == len(expected), actual
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert same_part(a, e), (i, a, e)


def inventory_text(tmp_path, inventory: list, **changes) -> str:
    """iota(x) of (1): the part after the frames of the first clip."""
    cfg = load(tmp_path, **changes)
    first = Shown([frame(1)], {"inventory": inventory})
    return prompt(cfg, first, Shown([frame(2)], {"inventory": []}))[3]


def test_shown_fields():
    """Interface: `Shown(NamedTuple)` with `frames`, then `line`."""
    shown = Shown([frame(1)], {"inventory": []})
    assert isinstance(shown, tuple) and Shown._fields == ("frames", "line")
    assert shown.line == {"inventory": []} and shown.frames[0] is shown[0][0]


def test_prompt_parts_in_order(tmp_path):
    """(1): [instruction, first, frames of x in order, iota(x), second, frames of y,
    iota(y), theta]; the frames as given (oldest first); other fields of the line unused."""
    cfg = load(tmp_path)
    xs, ys = [frame(5), frame(3), frame(9)], [frame(7), frame(1), frame(4)]
    x = Shown(xs, {"inventory": [item("minecraft:dirt", 2)], "isGuiOpen": True})
    y = Shown(ys, {"inventory": [], "tick": 12})
    expected = ["I\n", "A:\n", *xs, "Inv: 2xminecraft:dirt\n", "B:\n", *ys]
    same_parts(prompt(cfg, x, y), expected + ["Inv: none\n", TINY_THETA])


def test_prompt_swapped(tmp_path):
    """(1) of (y, x): y's frames and inventory first."""
    cfg = load(tmp_path)
    x = Shown([frame(1)], {"inventory": [item("a", 1)]})
    y = Shown([frame(2)], {"inventory": [item("b", 2)]})
    expected = ["I\n", "A:\n", frame(2), "Inv: 2xb\n", "B:\n", frame(1), "Inv: 1xa\n"]
    same_parts(prompt(cfg, y, x), expected + [TINY_THETA])


def test_inventory_grouped_summed_sorted(tmp_path):
    """(1): entries grouped by type, quantities summed, types in ascending Python string
    order ('B' < 'a' < 'a_b' < 'ab' < 'b'), joined by ', '."""
    inventory = [item("b", 1), item("B", 2), item("a_b", 3), item("ab", 4)]
    inventory += [item("a", 5), item("b", 6), item("B", 10)]
    text = inventory_text(tmp_path, inventory)
    assert text == "Inv: 12xB, 5xa, 3xa_b, 4xab, 7xb\n"


def test_inventory_empty(tmp_path):
    """(1): `no_items` for an empty inventory."""
    assert inventory_text(tmp_path, []) == "Inv: none\n"


def test_inventory_type_replaced_before_quantity(tmp_path):
    """(1): `{type}` replaced, then `{quantity}`: a type holding `{quantity}` gets the sum."""
    text = inventory_text(
        tmp_path, [item("q{quantity}", 3)], item_line="{type}:{quantity}"
    )
    assert text == "Inv: q3:3\n"


def test_theta_lines_and_replacements(tmp_path):
    """(1): every `{answers}` replaced (str.replace) by the answer lines in order, joined
    by '\\n'; `{token}` replaced before `{description}`; the other texts verbatim."""
    answers = [
        {"token": "{description}", "description": "D1"},
        {"token": "T2", "description": "has {token}"},
    ]
    cfg = load(
        tmp_path,
        instruction="{answers} {items}",
        first="{token}",
        second="{answers}",
        question="Q {answers} | {answers} {token}",
        answer_line="<{token}|{description}>",
        answers=answers,
    )
    x, y = Shown([frame(1)], {"inventory": []}), Shown([frame(2)], {"inventory": []})
    lines = "<D1|D1>\n<T2|has {token}>"
    expected = ["{answers} {items}", "{token}", frame(1), "Inv: none\n", "{answers}"]
    expected += [frame(2), "Inv: none\n", f"Q {lines} | {lines} {{token}}"]
    same_parts(prompt(cfg, x, y), expected)


def test_prompt_starting_config(tmp_path):
    """(1) with the starting texts of Interface."""
    cfg = load(tmp_path, STARTING)
    inventory = [item("minecraft:stick", 1), item("minecraft:dirt", 3)]
    x = Shown([frame(1), frame(2)], {"inventory": inventory})
    y = Shown([frame(3), frame(4)], {"inventory": []})
    theta = STARTING["question"].replace(
        "{answers}",
        "MUCH_BETTER: A is much better than B\nBETTER: A is better than B\n"
        "SLIGHTLY_BETTER: A is slightly better than B\n"
        "SLIGHTLY_WORSE: A is slightly worse than B\nWORSE: A is worse than B\n"
        "MUCH_WORSE: A is much worse than B\nINCOMPARABLE: A and B cannot be compared",
    )
    head = [STARTING["instruction"], "Moment A:\n", frame(1), frame(2)]
    head.append("Inventory: 3 minecraft:dirt, 1 minecraft:stick\n\n")
    tail = ["Moment B:\n", frame(3), frame(4), "Inventory: empty\n\n", theta]
    same_parts(prompt(cfg, x, y), head + tail)


PARSED = [  # (2) with the starting config: answer, label by hand
    ("Answer: BETTER", "BETTER"),
    ("**Answer: BETTER**", "BETTER"),
    ("`Answer: BETTER`", "BETTER"),
    ('"Answer: BETTER"', "BETTER"),
    ("'Answer: BETTER'", "BETTER"),
    ("Answer: BETTER.", "BETTER"),
    ("Answer: BETTER ." + " ", "BETTER"),  # stripped, '.' removed, stripped again
    ("Answer: BETTER..", "invalid"),  # one final '.' removed
    ("Answer: BET*TER", "BETTER"),  # the characters removed anywhere
    ("A is ahead.\n```\nAnswer: WORSE\n```", "WORSE"),  # a closing code fence
    ("Answer: WORSE\n\n   \n", "WORSE"),  # empty last lines skipped
    ("  Answer: WORSE\t\r\n", "WORSE"),
    ("Answer: BETTER\nA has the pickaxe.", "invalid"),  # trailing text after it
    ("Answer: BETTER, clearly", "invalid"),
    ("Answer: SLIGHTLY_BETTER", "SLIGHTLY_BETTER"),  # BETTER is its suffix
    ("Answer: BETTER", "BETTER"),
    ("Answer: MUCH_WORSE", "MUCH_WORSE"),
    ("Answer: INCOMPARABLE", "INCOMPARABLE"),
    ("answer: BETTER", "invalid"),  # wrong case
    ("Answer: better", "invalid"),
    ("Answer:BETTER", "invalid"),  # no space after the colon
    ("Answer:  BETTER", "invalid"),
    ("BETTER", "invalid"),  # no prefix
    ("Answer: \u201cBETTER\u201d", "invalid"),  # non-ASCII quotes stay
    ("Answer: invalid", "invalid"),
    ("", "invalid"),  # no non-empty line
    ("```\n\n**", "invalid"),
    (None, "no_answer"),
]


@pytest.mark.parametrize("answer, label", PARSED)
def test_parse_starting_config(tmp_path, answer, label):
    """(2) hand cases with the starting prefix and answers."""
    assert parse(answer, load(tmp_path, STARTING)) == label


@pytest.mark.parametrize(
    "prefix, answer, label",
    [
        ("", "WORSE", "WORSE"),
        ("  ", " WORSE ", "WORSE"),  # (prefix + token) stripped
        ("  Verdict - ", "Verdict - SAME", "SAME"),
        ("  Verdict - ", "  Verdict - SAME", "SAME"),
        ("Verdict - ", "Answer: SAME", "invalid"),
        ("Verdict - ", "Verdict - SAME.", "SAME"),
    ],
)
def test_parse_prefix(tmp_path, prefix, answer, label):
    """(2): the last line equals (prefix + token) stripped of whitespace."""
    assert parse(answer, load(tmp_path, TINY, prefix=prefix)) == label
