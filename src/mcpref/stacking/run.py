"""(R), (M), (F) model.pt, (P), (6) and `main` of specs/gui-stack.md: the classifier
trained on S from the features shards of specs/gui-masks.md, evaluated on F^te."""

import argparse
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from mcpref.cli.failure import FAILURES, failed
from mcpref.recognition.metrics import (
    CRITERIA,
    evaluation,
    matrices,
    tally,
    tally_matrices,
)
from mcpref.recognition.report import variant_preds, write_report
from mcpref.recognition.stored import (
    FEATURES,
    PARTS,
    read_arrays,
    shard_bar,
)
from mcpref.runs.files import check_fingerprints, finished_run, sha256, write_json
from mcpref.shards.store import complete_shards, shard_path, write_file, write_shard
from mcpref.stacking.config import GuiStackConfig, load_config
from mcpref.stacking.fit import Fitted, Shard, cpu_copies, fit
from mcpref.stacking.model import predicted, probabilities, standardisation, z_tensor

PROG = "mcpref.stacking.run"
PREDICTIONS = "predictions.jsonl"


@dataclass(frozen=True)
class Features:
    """Notation: Y, the columns, the shards of S with |S_q|, n_y, mu, s, the test shards
    with their frames."""

    labels: list[str]
    columns: list[str]
    s_paths: list[Path]
    sizes: list[int]
    n: dict[str, int]
    mu: np.ndarray
    s: np.ndarray
    test_paths: list[Path]
    test_sizes: list[int]


def checks(cfg: GuiStackConfig, path: Path, raw: dict) -> None:
    """(R): gui-masks complete; an existing {out_dir}/run.json with this run's
    fingerprints (config, masks); then {out_dir}/run.json written."""
    finished_run(cfg.masks_dir, raw["masks_dir"], "complete")
    masks_run, run_path = cfg.masks_dir / "run.json", cfg.out_dir / "run.json"
    hashes = {"config": sha256(path), "masks": sha256(masks_run)}
    check_fingerprints(run_path, hashes, raw["out_dir"])
    write_json(run_path, {"config": raw, "sha256": hashes})


def features_paths(masks_dir: Path, part: str) -> list[Path]:
    """Consumes: features/<part>/<name(r)>/features.npz of the complete shards."""
    directory = masks_dir / FEATURES / part
    n = complete_shards(directory)
    return [shard_path(directory, r) / "features.npz" for r in range(n)]


def read_features(masks_dir: Path) -> Features:
    """(R) sizes: `ids` (and `y` for S) of every features shard, one at a time; C = 0,
    then an empty S abort; then (M)."""
    header = json.loads((masks_dir / FEATURES / "header.json").read_text())
    labels, columns = header["labels"], header["columns"]
    n, s_paths, sizes = dict.fromkeys(labels, 0), [], []
    for part in PARTS[:2]:
        for path in features_paths(masks_dir, part):
            ids, y = read_arrays(path, ("ids", "y"))
            for k in y.tolist():
                n[k] += 1
            if len(ids):
                s_paths.append(path)
                sizes.append(len(ids))
    test_paths = features_paths(masks_dir, PARTS[2])
    test_sizes = [len(read_arrays(p, ("ids",))[0]) for p in test_paths]
    if not columns:
        raise RuntimeError("no column")
    if not s_paths:
        raise RuntimeError("no training frame")
    mu, s = standardisation(lambda r: read_arrays(s_paths[r], ("x",))[0], len(s_paths))
    return Features(labels, columns, s_paths, sizes, n, mu, s, test_paths, test_sizes)


def training_shard(cfg: GuiStackConfig, f: Features, r: int) -> Shard:
    """`read` of `fit`: z, y (indices in Y) and w_f = 1 / n_{y_f} of shard r of S."""
    x, y = read_arrays(f.s_paths[r], ("x", "y"))
    index = {k: i for i, k in enumerate(f.labels)}
    y = y.tolist()
    return (
        z_tensor(x, f.mu, f.s, cfg.device),
        torch.tensor([index[k] for k in y], dtype=torch.int64, device=cfg.device),
        torch.tensor([1 / f.n[k] for k in y], dtype=torch.float32, device=cfg.device),
    )


def trained(cfg: GuiStackConfig, f: Features) -> Fitted:
    """(F): theta_T, L(theta_0), L(theta_T) read from model.pt when it exists, else by
    `fit` and model.pt written. sum_{f in S} w_f = the number of labels present in S."""
    path = cfg.out_dir / "model.pt"
    if path.exists():
        model = torch.load(path)
        for desc in ("loss initial", "stack fit", "loss final"):
            print(f"{desc}: already done", file=sys.stderr)
        params = [p.to(cfg.device) for p in model["params"]]
        return params, model["loss_initial"], model["loss_final"]
    read, directory = partial(training_shard, cfg, f), cfg.out_dir / "checkpoints"
    c, n_labels = len(f.columns), len(f.labels)
    w_total = float(sum(n_y > 0 for n_y in f.n.values()))
    params, initial, final = fit(cfg, read, f.sizes, c, n_labels, w_total, directory)
    model = {"params": cpu_copies(params), "mu": torch.from_numpy(f.mu)}
    model |= {"s": torch.from_numpy(f.s), "columns": f.columns, "labels": f.labels}
    buffer = io.BytesIO()
    torch.save(model | {"loss_initial": initial, "loss_final": final}, buffer)
    write_file(path, buffer.getvalue())
    return params, initial, final


def predictions(f: Features, ids: np.ndarray, y: np.ndarray, p: np.ndarray) -> str:
    """Produces predictions.jsonl: id, label, predicted (5), p per frame, in order."""
    y_hat = predicted(p, f.labels)
    records = (
        {"id": i, "label": k, "predicted": yh, "p": dict(zip(f.labels, pf))}
        for i, k, yh, pf in zip(ids.tolist(), y.tolist(), y_hat, p.tolist())
    )
    return "".join(json.dumps(x) + "\n" for x in records)


def predict(cfg: GuiStackConfig, f: Features, params: Sequence[torch.Tensor]) -> None:
    """(P): prediction shard r for each test features shard r, continued after the
    complete ones; bar `predict`."""
    directory, sizes = cfg.out_dir / "predictions", f.test_sizes
    h = complete_shards(directory)
    with shard_bar("predict", len(sizes), h, sum(sizes), sum(sizes[:h]), "frame") as up:
        for r in range(h, len(sizes)):
            ids, y, x = read_arrays(f.test_paths[r], ("ids", "y", "x"))
            p = probabilities(params, z_tensor(x, f.mu, f.s, cfg.device))
            text = predictions(f, ids, y, p)
            with write_shard(directory, r) as staged:
                (staged / PREDICTIONS).write_text(text)
            up(len(ids))


def evaluated(cfg: GuiStackConfig, f: Features) -> dict:
    """(6): gui-masks (7) with y_hat, (8) with vlm, (a)-(f) (`pred`), `stack` (y_hat
    under both choices); the prediction and test shards one at a time; bar `report`."""
    acc, matrix_acc, directory = {}, {}, cfg.out_dir / "predictions"
    with shard_bar("report", len(f.test_sizes), 0, sum(f.test_sizes), 0, "frame") as up:
        for r, path in enumerate(f.test_paths):
            ids, y, g, pred = read_arrays(path, ("ids", "y", "g", "pred"))
            lines = (shard_path(directory, r) / PREDICTIONS).read_text().splitlines()
            y_hat = [json.loads(line)["predicted"] for line in lines]
            tally(acc, y.tolist(), y_hat, ids.tolist(), g.tolist())
            stack = {"stack": dict.fromkeys(CRITERIA, y_hat)}
            tally_matrices(matrix_acc, y.tolist(), variant_preds(pred) | stack)
            up(len(ids))
    test = evaluation(acc, f.labels)
    return {"test": test, "agreement_matrices": matrices(matrix_acc, f.labels[:-1])}


def run(cfg: GuiStackConfig, path: Path) -> None:
    """Behaviour: (R), (M), (F), (P), (6); report.json and report.md."""
    checks(cfg, path, yaml.safe_load(path.read_text()))
    f = read_features(cfg.masks_dir)
    params, initial, final = trained(cfg, f)
    predict(cfg, f, params)
    train = {"frames": sum(f.sizes), "per_label": f.n}
    train |= {"loss_initial": initial, "loss_final": final}
    write_report(cfg.out_dir, {"train": train} | evaluated(cfg, f), "GUI stack report")


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.stacking.run <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.config)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
