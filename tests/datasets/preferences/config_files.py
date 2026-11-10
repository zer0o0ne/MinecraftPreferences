"""Configs of specs/vlm-labels.md for its tests: the starting values of Interface, a tiny
config (keys in another order than Notation's) and a writer of YAML config files."""

from pathlib import Path

import yaml

from mcpref.datasets.preferences.labels import VlmLabelsConfig, load_config

FIELDS = (  # Notation: VlmLabelsConfig, fields in this order
    "pool_dir", "vlm", "out_dir", "diagnostics", "acyclic", "instruction", "first",
    "second", "question", "answer_line", "prefix", "inventory_line", "item_line",
    "no_items", "answers", "frames", "stride", "samples", "seed",
)  # fmt: skip
INSTRUCTION = (
    "You will see two moments, A and B, from the same game of Minecraft Java Edition 1.16 "
    "in survival mode, played by one player in one world. Each moment is shown as a few "
    "screenshots taken shortly one after another, oldest first; the last screenshot is the "
    "moment itself. After the screenshots comes the player's inventory at that moment.\n"
    "Evaluate each moment on three criteria: available potential for gathering resources "
    "of every kind, safety, and current level of progress. Compare the moments only under "
    "the following rules:\n"
    "1. If one moment is Pareto-better (at least as good on every criterion and strictly "
    "better on at least one), use the six graded answers to indicate how much better or "
    "worse A is than B.\n"
    "2. Exception: if one moment is only very slightly worse on some criteria but much "
    "better on others, treat it as nearly Pareto-better and compare the moments using "
    "the same six answers.\n"
    "3. If the moments are approximately equal on all three criteria, you may choose "
    "between them on aesthetic grounds, especially the beauty or rarity of the situation. "
    "Prefer the more beautiful or rarer moment when differences in resource-gathering "
    "potential, safety, and progress are negligible.\n"
    "Otherwise, answer INCOMPARABLE.\n\n"
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
    "vlm": "configs/integrations/vlm.yaml",
    "out_dir": "data/pair_votes",
    "diagnostics": "configs/diagnostics.yaml",
    "acyclic": "configs/datasets/preferences/acyclic.yaml",
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
    "frames": 6,
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
    "diagnostics": "w/diagnostics.yaml",
    "acyclic": "w/acyclic.yaml",
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

DIAGNOSTICS = {
    "out_dir": "w/diagnostics",
    "seed": 0,
    "bootstrap_samples": 20,
    "ci_level": 0.95,
    "min_cluster_count": 2,
    "quantiles": [0.05, 0.5, 0.95],
}
ACYCLIC = {
    "votes_dir": "w/votes",
    "out_dir": "w/labels",
    "diagnostics": "w/acyclic-diagnostics.yaml",
    "weights": {"BETTER": 0.8, "WORSE": 0.2, "SAME": 0.5, "INCOMPARABLE": None},
    "refusal_share": 0.5,
}


def write_config(path: Path, mapping: dict) -> Path:
    """The mapping as a YAML file, keys in its order."""
    path.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True))
    return path


def linked_configs(directory) -> dict:
    directory = Path(directory)
    write_config(
        directory / "diagnostics.yaml",
        DIAGNOSTICS | {"out_dir": f"{directory.name}/diagnostics"},
    )
    write_config(
        directory / "acyclic.yaml",
        ACYCLIC | {"diagnostics": f"{directory.name}/diagnostics.yaml"},
    )
    return {
        "diagnostics": f"{directory.name}/diagnostics.yaml",
        "acyclic": f"{directory.name}/acyclic.yaml",
    }


def load(tmp_path: Path, base: dict = TINY, **changes) -> VlmLabelsConfig:
    """load_config of `base` with `changes`, written to a file in tmp_path."""
    return load_config(write_config(tmp_path / "vlm-labels.yaml", {**base, **changes}))
