"""Configs of specs/vlm-labels.md for its tests: the starting values of Interface, a tiny
config (keys in another order than Notation's) and a writer of YAML config files."""

from pathlib import Path

import yaml

from mcpref.preferences.labels import VlmLabelsConfig, load_config

FIELDS = (  # Notation: VlmLabelsConfig, fields in this order
    "pool_dir", "vlm", "out_dir", "instruction", "first", "second", "question",
    "answer_line", "prefix", "inventory_line", "item_line", "no_items", "answers",
    "frames", "stride", "samples", "seed",
)  # fmt: skip
INSTRUCTION = (
    "You will see two moments, A and B, from the same game of Minecraft Java Edition 1.16 "
    "in survival mode, played by one player in one world. Each moment is shown as a few "
    "screenshots taken shortly one after another, oldest first; the last screenshot is the "
    "moment itself. After the screenshots comes the player's inventory at that moment.\n"
    "Judge which moment is better for the player. A moment is better when the player has "
    "made more progress (better tools, weapons and armour; more and rarer resources; food; "
    "a shelter; crafting stations; explored places), is safer (not falling, drowning, "
    "burning, starving or being attacked) and has better prospects. Refuse to compare only "
    "when the two moments really cannot be compared.\n\n"
)
QUESTION = (
    "How good is moment A for the player compared with moment B? Explain in at most three "
    "short sentences, then write a last line of the form Answer: TOKEN, with TOKEN one "
    "of:\n{answers}"
)
STARTING_ANSWERS = [
    {"token": "MUCH_BETTER", "description": "A is much better than B"},
    {"token": "BETTER", "description": "A is better than B"},
    {"token": "SLIGHTLY_BETTER", "description": "A is slightly better than B"},
    {"token": "SLIGHTLY_WORSE", "description": "A is slightly worse than B"},
    {"token": "WORSE", "description": "A is worse than B"},
    {"token": "MUCH_WORSE", "description": "A is much worse than B"},
    {"token": "INCOMPARABLE", "description": "A and B cannot be compared"},
]
STARTING = {
    "pool_dir": "data/pair_clips",
    "vlm": "configs/vlm.yaml",
    "out_dir": "data/pair_votes",
    "instruction": INSTRUCTION,
    "first": "Moment A:\n",
    "second": "Moment B:\n",
    "question": QUESTION,
    "answer_line": "{token}: {description}",
    "prefix": "Answer: ",
    "inventory_line": "Inventory: {items}\n\n",
    "item_line": "{quantity} {type}",
    "no_items": "empty",
    "answers": STARTING_ANSWERS,
    "frames": 8,
    "stride": 2,
    "samples": 1,
    "seed": 0,
}
TINY_ANSWERS = [  # tokens; their order is the key order of `labels`
    {"token": "BETTER", "description": "A is better"},
    {"token": "WORSE", "description": "A is worse"},
    {"token": "SAME", "description": "même, été"},  # non-ASCII: json.dumps escapes it
    {"token": "INCOMPARABLE", "description": "cannot compare"},
]
TINY = {  # the shown frames j = 1, 3, 5 of L = 6 states; two samples per order
    "seed": 1000,
    "samples": 2,
    "frames": 3,
    "stride": 2,
    "out_dir": "w/votes",
    "pool_dir": "w/pair_clips",
    "vlm": "w/vlm.yaml",
    "instruction": "I\n",
    "first": "A:\n",
    "second": "B:\n",
    "question": "Pick:\n{answers}",
    "answer_line": "{token}={description}",
    "prefix": "Answer: ",
    "inventory_line": "Inv: {items}\n",
    "item_line": "{quantity}x{type}",
    "no_items": "none",
    "answers": TINY_ANSWERS,
}
# (1): theta of TINY by hand
TINY_THETA = "Pick:\nBETTER=A is better\nWORSE=A is worse\nSAME=même, été"
TINY_THETA += "\nINCOMPARABLE=cannot compare"


def write_config(path: Path, mapping: dict) -> Path:
    """The mapping as a YAML file, keys in its order."""
    path.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True))
    return path


def load(tmp_path: Path, base: dict = TINY, **changes) -> VlmLabelsConfig:
    """load_config of `base` with `changes`, written to a file in tmp_path."""
    return load_config(write_config(tmp_path / "vlm-labels.yaml", {**base, **changes}))
