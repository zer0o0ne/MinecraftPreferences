"""Interface of specs/gui-labels.md: VLM labels of the sampled GUI frames."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.label_files import (
    INVALID,
    NO_ANSWER,
    OTHER,
    frame_counts,
    frame_size,
    resume,
    rgb_view,
    sha256,
    shown,
    write_labels,
    write_run,
    write_summary,
    yuv_path,
)
from mcpref.gui.screens import ScreenClass, load_screens
from mcpref.shards.store import complete_shards
from mcpref.vlm.client import VLMConfig, ask_many
from mcpref.vlm.client import load_config as load_vlm_config
from mcpref.vpt.plan import read_shard

QUOTES = "`'\""


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


def hashes(cfg: GuiLabelsConfig, config_path: Path) -> dict[str, str]:
    """(4): SHA-256 of the config, of configs/vlm.yaml and of the catalogue file."""
    return {
        "config": sha256(config_path),
        "vlm": sha256(REPO / cfg.vlm),
        "screens": sha256(REPO / cfg.screens),
    }


def check_rerun(cfg: GuiLabelsConfig, header: dict) -> None:
    """(4): an existing {out_dir}/run.json must hold the same SHA-256 values (compared
    first, in order) and the same fingerprint of the frames."""
    path = REPO / cfg.out_dir / "run.json"
    if not path.exists():
        return
    stored = json.loads(path.read_text())
    for key, value in header["sha256"].items():
        if stored["sha256"][key] != value:
            raise RuntimeError(f"{path} holds another SHA-256 of {key}")
    if stored["frames"] != header["frames"]:
        changed = f"{cfg.frames_dir} changed since labelling started"
        raise RuntimeError(f"{changed}; delete {cfg.out_dir}")


def ask_shard(
    cfg: GuiLabelsConfig,
    vlm: VLMConfig,
    classes: Sequence[ScreenClass],
    size: tuple[int, int],
    shard: dict,
    start: int,
) -> list[dict]:
    """(1)-(4): one `ask_many` call for the `ok` frames of a plan shard whose first frame
    has position `start` (none without them); their label records."""
    frames_dir = REPO / cfg.frames_dir
    frames = enumerate(shard["frames"], start)
    ok = [(p, f) for p, f in frames if yuv_path(frames_dir, f).exists()]
    if not ok:
        return []
    seeds = [cfg.seed + position for position, _ in ok]
    prompts = [prompt(cfg, classes, rgb_view(frames_dir, f, size)) for _, f in ok]
    answers = ask_many(prompts, seeds, vlm)
    names = [c.name for c in classes] + [OTHER]
    return [
        {
            "id": f["id"],
            "seed": seed,
            "answer": a,
            "label": parse(a, names),
            "gui_inventory": f["line"]["isGuiInventory"],
        }
        for (_, f), seed, a in zip(ok, seeds, answers)
    ]


def label(
    cfg: GuiLabelsConfig,
    vlm: VLMConfig,
    classes: Sequence[ScreenClass],
    H: int,
    n_ok: int,
) -> None:
    """(4): label shards h, ..., H - 1, one per plan shard; bar `label`."""
    frames_dir, labels_dir = REPO / cfg.frames_dir, REPO / cfg.out_dir / "labels"
    h = complete_shards(labels_dir)
    if h == H:
        print("label: already done", file=sys.stderr)
        return
    initial, end = resume(labels_dir, h)
    size = frame_size(frames_dir)
    with tqdm(total=n_ok, initial=initial, desc="label", unit="frame") as bar:
        for k in range(h, H):
            shard = read_shard(frames_dir / "plan", k)
            records = ask_shard(cfg, vlm, classes, size, shard, end)
            end += len(shard["frames"])
            write_labels(labels_dir, k, records, end)
            bar.update(len(records))


def run(cfg: GuiLabelsConfig, config_path: Path) -> None:
    """Behaviour (4), (4'): the checks, run.json (frames: the fingerprint), the label
    shards, summary.json."""
    frames_dir, out = REPO / cfg.frames_dir, REPO / cfg.out_dir
    counts = frame_counts(frames_dir)
    H = complete_shards(frames_dir / "plan")
    header = {"config": asdict(cfg), "sha256": hashes(cfg, config_path)}
    header["frames"] = {"shards": H, **counts}
    check_rerun(cfg, header)
    classes = list(load_screens(REPO / cfg.screens).values())
    header["prompt"] = shown(prompt(cfg, classes, np.empty(0, np.uint8)))
    write_run(out, header)
    vlm, _ = load_vlm_config(REPO / cfg.vlm)
    label(cfg, vlm, classes, H, counts["ok"])
    write_summary(out, H)


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
