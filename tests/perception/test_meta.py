"""(2) of specs/perception.md: `inventory_text`, `meta_text` on hand cases (grouping, wrap,
`-0.0`, relative x and z, the order of the replacements); `inventory` of vlm-labels'
prompt.py through `inventory_text`; `clip_states` (v_s with the cursor where g_s, mu_s)."""

import numpy as np
import pytest
from preferences.config_files import STARTING, TINY
from preferences.config_files import load as vlm_config

from mcpref.perception.cursor import load_sprite
from mcpref.perception.meta import clip_states, inventory_text, meta_text
from mcpref.preferences import prompt

from . import oracle
from .world import H, W, clip, config, item, line, random_line, values

LINES = ("Inv: {items}\n", "{quantity} {type}", "none")


@pytest.mark.parametrize(
    "inventory, text",
    [
        ([], "Inv: none\n"),
        ([item("dirt", 3)], "Inv: 3 dirt\n"),
        # grouped by type, quantities summed, ascending Python string order
        (
            [
                item("b", 1),
                item("B", 2),
                item("a_b", 3),
                item("ab", 4),
                item("a", 5),
                item("b", 6),
                item("B", 10),
            ],
            "Inv: 12 B, 5 a, 3 a_b, 4 ab, 7 b\n",
        ),
    ],
)
def test_inventory_text(inventory, text):
    assert inventory_text({"inventory": inventory}, *LINES) == text


def test_inventory_text_type_before_quantity():
    """`{type}` replaced first: a type holding `{quantity}` gets the sum."""
    got = inventory_text({"inventory": [item("q{quantity}", 2)]}, "{items}", "{type}={quantity}", "-")  # fmt: skip
    assert got == "q2=2"


def test_vlm_labels_inventory_unchanged(tmp_path):
    """Interface: prompt.inventory(cfg, line) = inventory_text(line, cfg.inventory_line,
    cfg.item_line, cfg.no_items), on vlm-labels' starting config and a tiny one."""
    rng = np.random.default_rng(1)
    lines = [random_line(rng) for _ in range(20)] + [line()]
    for base in (STARTING, TINY):
        cfg = vlm_config(tmp_path, base)
        fields = {
            k: getattr(cfg, k) for k in ("inventory_line", "item_line", "no_items")
        }
        for x in lines:
            want = inventory_text(x, *fields.values())
            assert prompt.inventory(cfg, x) == want == oracle.inventory(x, fields)


@pytest.fixture
def cfg(tmp_path):
    return config(tmp_path, meta="{inventory}|{hotbar}|{yaw}|{pitch}|{x}|{y}|{z}")


FIRST = line(xpos=100.25, ypos=70.0, zpos=-20.5)


def meta(cfg, **fields) -> str:
    return meta_text(cfg, line(**fields), FIRST)


def test_meta_fields(cfg):
    """hotbar + 1; yaw wrapped; pitch; x, z relative to the first state, y absolute."""
    got = meta(cfg, inventory=[item("log", 2), item("log", 1)], hotbar=0, yaw=90.0,
               pitch=-12.34, xpos=103.5, ypos=64.0, zpos=-30.75)  # fmt: skip
    assert got == "Inv: 3 log\n|1|90.0|-12.3|3.2|64.0|-10.2"  # 3.25, -10.25: exact ties


@pytest.mark.parametrize(
    "yaw, text",
    [
        (180.0, "-180.0"),  # ((180 + 180) % 360) - 180
        (-180.0, "-180.0"),
        (179.94, "179.9"),
        (540.0, "-180.0"),
        (360.0, "0.0"),
        (-2148.7, "11.3"),  # -1968.7 % 360 = 191.3
        (3007.4, "127.4"),  # 3187.4 % 360 = 307.4
        (-0.04, "0.0"),  # wrap(-0.04) = -0.04: '-0.0' without its '-'
    ],
)
def test_meta_yaw_wrapped(cfg, yaw, text):
    assert meta(cfg, yaw=yaw).split("|")[2] == text


@pytest.mark.parametrize(
    "pitch, decimals, text",
    [
        (-0.04, 1, "0.0"),
        (-0.0, 1, "0.0"),
        (-0.06, 1, "-0.1"),
        (-0.4, 0, "0"),
        (-0.6, 0, "-1"),
        (-10.0, 1, "-10.0"),
        (-0.0004, 3, "0.000"),
        (-0.0005001, 3, "-0.001"),
        (90.0, 2, "90.00"),
    ],
)
def test_meta_fmt(tmp_path, pitch, decimals, text):
    """fmt: f"{v:.{decimals}f}", the leading '-' removed when every digit is 0."""
    c = config(
        tmp_path, decimals=decimals, meta="{inventory}{hotbar}{yaw}{x}{y}{z}<{pitch}>"
    )
    assert meta_text(c, line(pitch=pitch), line()).split("<")[1] == text + ">"


def test_meta_relative_to_the_first(cfg):
    """x and z of the first state itself are 0 (no '-0.0')."""
    got = meta_text(cfg, FIRST, FIRST).split("|")
    assert got[4:] == ["0.0", "70.0", "0.0"]
    neg = meta(cfg, xpos=100.21, zpos=-20.54).split("|")
    assert (neg[4], neg[6]) == ("0.0", "0.0")


def test_meta_order_of_replacements(tmp_path):
    """{inventory} first: an item type holding `{yaw}` gets the yaw; a placeholder twice
    replaced twice; `{x}` in the inventory line replaced too."""
    c = config(tmp_path, meta="{inventory}{hotbar}/{hotbar}{yaw}{pitch}{x}{y}{z}",
               inventory_line="[{items}|{x}]")  # fmt: skip
    got = meta_text(c, line(inventory=[item("{yaw}", 1)], hotbar=4, yaw=10.0,
                            pitch=1.0, xpos=2.0, ypos=3.0, zpos=4.0), line())  # fmt: skip
    assert got == "[1 10.0|2.0]5/510.01.02.03.04.0"


def test_meta_no_items(tmp_path):
    c = config(tmp_path, no_items="nothing at all")
    assert meta_text(c, line(), line()).startswith("Inv: nothing at all\nH1 ")


def test_meta_matches_the_oracle(tmp_path):
    rng = np.random.default_rng(3)
    c = config(tmp_path)
    for _ in range(50):
        a, b = random_line(rng), random_line(rng)
        assert meta_text(c, a, b) == oracle.meta(a, b, values(tmp_path))


def gui_clip() -> object:
    """L = 3 at states 4, 6, 7 (line 5 unused); g = (True, False, True)."""
    rng = np.random.default_rng(5)
    lumas = [rng.integers(16, 236, (H, W)) for _ in range(3)]
    lines = [line(xpos=10.0, zpos=1.0, mouse={"x": 240.0, "y": 120.0}),
             line(hotbar=8, mouse={"x": 0.0, "y": 0.0}),
             line(xpos=11.5, hotbar=2, mouse={"x": 600.0, "y": 360.0}),
             line(zpos=-2.0, yaw=200.0, mouse={"x": 1500.0, "y": 600.0})]  # fmt: skip
    return clip(lumas, lines, [4, 6, 7], [True, False, True], [1, 2])


def test_clip_states(tmp_path):
    """v_s: the RGB frame of row s, the cursor drawn where g_s; mu_s of line
    states[s] - states[0], x and z relative to the clip's first state."""
    c = config(tmp_path)
    k = gui_clip()
    frames, metas = clip_states(k, c, load_sprite(c))
    assert len(frames) == len(metas) == 3
    for got, want in zip(frames, oracle.frames(k)):
        assert got.dtype == np.uint8 and got.shape == (H, W, 3)
        np.testing.assert_array_equal(got, want)
    assert metas == oracle.metas(k, values(tmp_path))
    assert metas[1] == "Inv: none\nH3 Y0.0 P0.0 X1.5 64.0 -1.0\n"
    assert metas[2] == "Inv: none\nH1 Y-160.0 P0.0 X-10.0 64.0 -3.0\n"


def test_clip_states_cursor_only_where_gui(tmp_path):
    """g_s false: the frame as converted, without a cursor."""
    c = config(tmp_path)
    k = gui_clip()
    frames, _ = clip_states(k, c, load_sprite(c))
    plain = oracle.rgb_of_row(k, 1)
    np.testing.assert_array_equal(frames[1], plain)
    assert (frames[0] != oracle.rgb_of_row(k, 0)).any()
