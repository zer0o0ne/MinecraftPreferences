"""Global fit and deterministic fixed-reference artifacts of pretrain diagnostics."""

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from mcpref import REPO
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.diagnostics import GlobalFit, Stratum
from mcpref.pretrain.reference_select import PARTS, select_items
from mcpref.runs.files import json_bytes, write_json
from mcpref.shards.store import complete_shards, shard_path, write_shard

GLOBAL_FIT = "global-fit.json"


@dataclass(frozen=True)
class ReferenceData:
    selection: dict
    baselines: GlobalFit
    Z0: Tensor
    P0: dict[str, Tensor]
    initial_metrics: dict


def fit_from_json(value: dict) -> GlobalFit:
    return GlobalFit(
        **value
        | {
            "strata": tuple(Stratum(**stratum) for stratum in value["strata"]),
        }
    )


def read_fit(path: Path) -> GlobalFit:
    return fit_from_json(json.loads(path.read_bytes()))


def load_global_fit(out_dir: Path) -> GlobalFit:
    """(10): complete shard 0, then the shared file, with identity checking."""
    shared = out_dir / GLOBAL_FIT
    initial = out_dir / "reference" / "000000" / "baselines.json"
    if initial.parent.is_dir():
        fit = read_fit(initial)
        if shared.exists() and read_fit(shared) != fit:
            raise RuntimeError(
                "global-fit.json differs from reference/000000/baselines.json"
            )
        return fit
    if shared.exists():
        return read_fit(shared)
    raise FileNotFoundError(shared)


def write_global_fit(out_dir: Path, fit: GlobalFit) -> None:
    """Atomically write the shared fit once, in the later shard-0 byte format."""
    path = out_dir / GLOBAL_FIT
    if path.exists():
        raise FileExistsError(path)
    write_json(path, asdict(fit))


def logit(probability: float | None) -> float | None:
    if probability is None or probability == 0 or probability == 1:
        return None
    return math.log(probability / (1 - probability))


def count_parts(cfg: PretrainConfig) -> tuple[dict, dict, dict, float]:
    """Read every train/validation pair once, retaining only counts and `sum_p`."""
    directories = {part: REPO / cfg.index_dir / part for part in PARTS}
    shards = {part: complete_shards(directories[part]) for part in PARTS}
    comparable = {part: {} for part in PARTS}
    incomparable = dict.fromkeys(PARTS, 0)
    pairs = dict.fromkeys(PARTS, 0)
    sum_p = 0.0
    with tqdm(
        total=sum(shards.values()),
        desc="pretrain reference counts",
        unit="shard",
        file=sys.stderr,
    ) as bar:
        for part in PARTS:
            for shard in range(shards[part]):
                path = shard_path(directories[part], shard) / "items.json"
                items = json.loads(path.read_bytes())["items"]
                for item in items:
                    if "labels" not in item:
                        continue
                    labels = item["labels"]
                    pairs[part] += 1
                    if labels["inc"]:
                        incomparable[part] += 1
                    else:
                        p = labels["p"]
                        comparable[part][p] = comparable[part].get(p, 0) + 1
                        if part == "train":
                            sum_p += p
                bar.update()
    return comparable, incomparable, pairs, sum_p


def catalogue(comparable: dict, incomparable: dict) -> tuple[Stratum, ...]:
    values = sorted(set(comparable["train"]) | set(comparable["validation"]))
    return tuple(
        Stratum(
            key=json.dumps(p),
            kind="comparable",
            p=p,
            train=comparable["train"].get(p, 0),
            validation=comparable["validation"].get(p, 0),
        )
        for p in values
    ) + (
        Stratum(
            key="inc",
            kind="inc",
            p=None,
            train=incomparable["train"],
            validation=incomparable["validation"],
        ),
    )


def count_fit(cfg: PretrainConfig) -> GlobalFit:
    """One shard-bounded train/validation counts pass, with no clip reads."""
    comparable, incomparable, pairs, sum_p = count_parts(cfg)
    train_pairs = pairs["train"]
    train_comparable = train_pairs - incomparable["train"]
    inc_probability = incomparable["train"] / train_pairs
    succ_probability = sum_p / train_comparable if train_comparable else None
    return GlobalFit(
        pairs=train_pairs,
        incomparable=incomparable["train"],
        comparable=train_comparable,
        sum_p=sum_p if train_comparable else None,
        inc_probability=inc_probability,
        succ_probability=succ_probability,
        inc_logit=logit(inc_probability),
        succ_logit=logit(succ_probability),
        strata=catalogue(comparable, incomparable),
    )


def select_reference(cfg: PretrainConfig, run: dict) -> tuple[dict, GlobalFit]:
    """(17): reuse shard 0/global fit or perform the shard-bounded second pass."""
    out_dir = REPO / cfg.out_dir
    initial = shard_path(out_dir / "reference", 0)
    if initial.is_dir():
        print("pretrain reference counts: already done", file=sys.stderr)
        print("pretrain reference select: already done", file=sys.stderr)
        selection = json.loads((initial / "selection.json").read_bytes())
        return selection, load_global_fit(out_dir)
    try:
        fit = load_global_fit(out_dir)
    except FileNotFoundError:
        fit = count_fit(cfg)
        write_global_fit(out_dir, fit)
    else:
        print("pretrain reference counts: already done", file=sys.stderr)
    return select_items(cfg, fit), fit


def load_initial(directory: Path) -> ReferenceData:
    """Load the complete typed shard 0 without selection or model evaluation."""
    initial = shard_path(directory, 0)
    saved = torch.load(initial / "initial.pt", map_location="cpu", weights_only=True)
    return ReferenceData(
        selection=json.loads((initial / "selection.json").read_bytes()),
        baselines=read_fit(initial / "baselines.json"),
        Z0=saved["Z0"],
        P0=saved["P0"],
        initial_metrics=json.loads((initial / "metrics.json").read_bytes()),
    )


def write_initial(
    directory: Path,
    selection: dict,
    baselines: GlobalFit,
    initial: dict,
    metrics: dict,
) -> None:
    """(18): atomically publish complete shard 0 and byte-identical baselines."""
    shared = directory.parent / GLOBAL_FIT
    baseline_bytes = (
        shared.read_bytes() if shared.exists() else json_bytes(asdict(baselines))
    )
    with write_shard(directory, 0) as staged:
        (staged / "selection.json").write_bytes(json_bytes(selection))
        (staged / "baselines.json").write_bytes(baseline_bytes)
        torch.save(initial, staged / "initial.pt")
        (staged / "metrics.json").write_bytes(json_bytes(metrics))


def write_metrics(directory: Path, k: int, metrics: dict) -> None:
    """(21): atomically publish one positive-cadence reference shard."""
    with write_shard(directory, k) as staged:
        (staged / "metrics.json").write_bytes(json_bytes(metrics))
