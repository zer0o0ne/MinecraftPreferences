"""src/mcpref/gui/labels.py against specs/gui-labels.md: config, (1) prompt, (3) parse.
`main` and its files: tests/gui/test_label_files.py."""

import dataclasses

import numpy as np
import pytest
import yaml

from mcpref.gui.labels import GuiLabelsConfig, load_config, parse, prompt
from mcpref.gui.screens import load_screens

QUESTION = (
    "Which of these screens is open? Answer with the name only, exactly as written."
)
STARTING = {
    "instruction": "This is a screenshot of Minecraft 1.16 with a GUI screen open.",
    "question": QUESTION + "\n{classes}",
    "class_line": "{name}: {description}",
    "other_description": "any other screen, or no screen",
    "seed": 0,
    "batch": 256,
}
STARTING_PATHS = {
    "frames_dir": "data/gui_frames",
    "screens": "configs/gui-screens.yaml",
}
STARTING_PATHS.update(vlm="configs/vlm.yaml", out_dir="data/gui_labels")


def test_load_config_starting_values(repo, tmp_path, monkeypatch):
    """Starting values; paths relative to the repository root, whatever the working directory."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(repo / "configs/gui-labels.yaml")
    assert isinstance(cfg, GuiLabelsConfig)
    assert {f.name for f in dataclasses.fields(cfg)} == {*STARTING, *STARTING_PATHS}
    assert {k: getattr(cfg, k) for k in STARTING} == STARTING
    # The spec leaves open whether load_config or its user resolves: both pass here.
    paths = {k: (repo / getattr(cfg, k)).resolve() for k in STARTING_PATHS}
    assert paths == {k: (repo / v).resolve() for k, v in STARTING_PATHS.items()}
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


@pytest.fixture
def labels_config(repo, tmp_path):
    """Writes the starting config with `texts` replaced; returns it loaded."""

    def write(**texts: str) -> GuiLabelsConfig:
        cfg = yaml.safe_load((repo / "configs/gui-labels.yaml").read_text())
        (tmp_path / "gui-labels.yaml").write_text(yaml.safe_dump({**cfg, **texts}))
        return load_config(tmp_path / "gui-labels.yaml")

    return write


def test_prompt_order_placeholders_image(catalogue, write_config, labels_config):
    """(1): classes in the given order, then other; str.replace on question and class_line only."""
    screens = load_screens(write_config(catalogue))
    cfg = labels_config(
        instruction="Look {classes} {name}",
        question="Pick {x}:\n{classes}\nor {classes}.",
        class_line="- {name} ({description}) [{name}]",
        other_description="none",
    )
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    parts = prompt(cfg, [screens["beta"], screens["alpha"]], image)
    lines = "- beta (Beta screen.) [beta]\n- alpha (Alpha screen.) [alpha]"
    lines += "\n- other (none) [other]"
    assert isinstance(parts, list) and len(parts) == 3
    assert parts[0] == "Look {classes} {name}"
    assert parts[2] == f"Pick {{x}}:\n{lines}\nor {lines}."
    assert isinstance(parts[1], np.ndarray) and parts[1].dtype == np.uint8
    assert np.array_equal(parts[1], image)


def test_prompt_starting_config_and_catalogue(repo, classes):
    """(1) with the approved config and catalogue; the oracle reads descriptions as plain YAML."""
    cfg = load_config(repo / "configs/gui-labels.yaml")
    screens = load_screens(repo / "configs/gui-screens.yaml")
    lines = [f"{name}: {c['description']}" for name, c in classes.items()]
    text = "\n".join([QUESTION, *lines, "other: any other screen, or no screen"])
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    parts = prompt(cfg, list(screens.values()), image)
    assert parts[0] == STARTING["instruction"] and parts[2] == text


NAMES = ("inventory", "chest3", "other")  # names of K, `other` included
VALID = [
    (None, "no_answer"),
    ("chest3", "chest3"),
    ("other", "other"),
    (" \tchest3\n ", "chest3"),
    ("`chest3`", "chest3"),
    ("'chest3'", "chest3"),
    ('"chest3"', "chest3"),
    (" ` chest3 `\n", "chest3"),  # strip, drop the pair, strip again
    ("'other'", "other"),
]
INVALID = ["Chest3", "CHEST3", "chest3.", "chest", "chest3\nother", "*chest3*"]
INVALID += ["`chest3'", "``chest3``", "'`chest3`'", "‘chest3’"]  # one equal pair
INVALID += ["`", "``", "", "no_answer", "invalid"]


@pytest.mark.parametrize("answer, label", [*VALID, *[(a, "invalid") for a in INVALID]])
def test_parse_hand_cases(answer, label):
    """(3): strip; drop one pair of equal `, ', " ends (length >= 2); strip; exact name."""
    assert parse(answer, NAMES) == label
