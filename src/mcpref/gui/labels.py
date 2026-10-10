"""Interface of specs/gui-labels.md: VLM labels of the sampled GUI frames."""

import argparse
import dataclasses
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.label_files import (
    INVALID,
    NO_ANSWER,
    OTHER,
    check_rerun,
    finished,
    frame_size,
    ok_frames,
    rgb_view,
    sha256,
    shown,
    write_outputs,
)
from mcpref.gui.screens import ScreenClass, load_screens
from mcpref.vlm.client import VLMConfig, ask_many
from mcpref.vlm.client import load_config as load_vlm_config

QUOTES = "`'\""
INVENTORY = {"inventory", "inventory_book"}
VPT_PAIRS = {
    "inventory_flag_true": (True, True),
    "inventory_flag_false": (True, False),
    "other_flag_true": (False, True),
    "other_flag_false": (False, False),
}


@dataclass(frozen=True)
class GuiLabelsConfig:
    """Notation: config fields of configs/gui-labels.yaml; paths relative to REPO."""

    frames_dir: str
    screens: str
    vlm: str
    instruction: str
    question: str
    class_line: str
    other_description: str
    seed: int
    batch: int
    out_dir: str


def load_config(path: Path) -> GuiLabelsConfig:
    """Interface: the YAML file loaded into GuiLabelsConfig."""
    return GuiLabelsConfig(**yaml.safe_load(path.read_text()))


def class_lines(cfg: GuiLabelsConfig, classes: Sequence[ScreenClass]) -> str:
    """(1): one `class_line` per class of K in order (`other` last), joined by newlines."""
    pairs = [(c.name, c.description) for c in classes]
    pairs.append((OTHER, cfg.other_description))
    lines = (
        cfg.class_line.replace("{name}", name).replace("{description}", description)
        for name, description in pairs
    )
    return "\n".join(lines)


def prompt(
    cfg: GuiLabelsConfig, classes: Sequence[ScreenClass], rgb: np.ndarray
) -> list[str | np.ndarray]:
    """(1): q_f = [instruction, x_f, question with {classes} replaced]."""
    question = cfg.question.replace("{classes}", class_lines(cfg, classes))
    return [cfg.instruction, rgb, question]


def parse(answer: str | None, names: Sequence[str]) -> str:
    """(3): the label y_f of the answer a_f."""
    if answer is None:
        return NO_ANSWER
    s = answer.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in QUOTES:
        s = s[1:-1].strip()
    return s if s in names else INVALID


def vpt_check(records: list[dict], flags: dict[str, bool]) -> dict[str, int]:
    """(5): counts of (y_f in {inventory, inventory_book}, g_f) over class labels."""
    pairs = Counter(
        (r["label"] in INVENTORY, flags[r["id"]])
        for r in records
        if r["label"] not in (INVALID, NO_ANSWER)
    )
    return {key: pairs[pair] for key, pair in VPT_PAIRS.items()}


def hashes(cfg: GuiLabelsConfig, config_path: Path) -> dict[str, str]:
    """(4): SHA-256 of the config, of configs/vlm.yaml and of the catalogue file."""
    return {
        "config": sha256(config_path),
        "vlm": sha256(REPO / cfg.vlm),
        "screens": sha256(REPO / cfg.screens),
    }


def ask_batch(
    cfg: GuiLabelsConfig,
    vlm: VLMConfig,
    classes: Sequence[ScreenClass],
    size: tuple[int, int],
    batch: list[tuple[int, dict]],
) -> list[dict]:
    """(1)-(4): one `ask_many` call; records `id`, `seed`, `answer`, `label`."""
    frames_dir = REPO / cfg.frames_dir
    seeds = [cfg.seed + position for position, _ in batch]
    prompts = [prompt(cfg, classes, rgb_view(frames_dir, f, size)) for _, f in batch]
    answers = ask_many(prompts, seeds, vlm)
    names = [c.name for c in classes] + [OTHER]
    return [
        {"id": f["id"], "seed": seed, "answer": a, "label": parse(a, names)}
        for (_, f), seed, a in zip(batch, seeds, answers)
    ]


def run(cfg: GuiLabelsConfig, config_path: Path) -> None:
    """Behaviour (1)-(5): label the frames still to label, batch by batch."""
    frames_dir, out = REPO / cfg.frames_dir, REPO / cfg.out_dir
    header = {"config": dataclasses.asdict(cfg), "sha256": hashes(cfg, config_path)}
    check_rerun(out / "run.json", header["sha256"])
    classes = list(load_screens(REPO / cfg.screens).values())
    vlm, _ = load_vlm_config(REPO / cfg.vlm)
    size, ok = frame_size(frames_dir), ok_frames(frames_dir)
    if not ok:
        raise RuntimeError(f"no ok frame in {frames_dir / 'frames.jsonl'}")
    flags = {f["id"]: f["line"]["isGuiInventory"] for _, f in ok}
    first = prompt(cfg, classes, rgb_view(frames_dir, ok[0][1], size))
    header["prompt"] = shown(first)
    done = finished(out)
    labelled = {i for i, r in done.items() if r["label"] != NO_ANSWER}
    todo = [(position, f) for position, f in ok if f["id"] not in labelled]
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), cfg.batch):
        batch = todo[start : start + cfg.batch]
        done.update((r["id"], r) for r in ask_batch(cfg, vlm, classes, size, batch))
        records = [done[f["id"]] for _, f in ok if f["id"] in done]
        write_outputs(out, header, records, vpt_check(records, flags))


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.gui.labels <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog="python -m mcpref.gui.labels")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.config)
    except FAILURES as error:
        return failed("mcpref.gui.labels", error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
