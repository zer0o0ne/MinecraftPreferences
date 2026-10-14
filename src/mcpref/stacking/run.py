"""Notation, (5) labels, (6) and `main` of specs/gui-stack.md: the classifier trained on
S from features.npz of specs/gui-masks.md, evaluated on F^te."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcpref.cli.failure import FAILURES, failed
from mcpref.recognition.metrics import CRITERIA, evaluation, matrices
from mcpref.recognition.pairs import VARIANTS
from mcpref.recognition.report import write_report
from mcpref.stacking.config import GuiStackConfig, load_config
from mcpref.stacking.model import fit, init_params, moments, probabilities, standardised

PROG = "mcpref.stacking.run"
TITLE = "GUI stack report"
PREDS = {v: {c: f"test_pred_{v}_{c}" for c in CRITERIA} for v in VARIANTS}
KEYS = ("columns", "labels", "train_x", "train_y", "validation_x", "validation_y")
KEYS += ("test_ids", "test_y", "test_g", "test_x")
KEYS += tuple(k for by_choice in PREDS.values() for k in by_choice.values())


def read_features(path: Path) -> dict[str, np.ndarray]:
    """Interface Consumes: the keys of features.npz this block reads; C = 0 or an empty
    S raise."""
    with np.load(path, allow_pickle=False) as npz:
        f = {k: npz[k] for k in KEYS}
    if not len(f["columns"]):
        raise ValueError(f"{path} has no feature columns (C = 0)")
    if not len(f["train_x"]) + len(f["validation_x"]):
        raise ValueError(f"{path} has no train or validation frames (S empty)")
    return f


def training_set(f: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Notation: S, the train frames, then the validation frames: x_f and y_f."""
    x = np.concatenate([f["train_x"], f["validation_x"]])
    return x, f["train_y"].tolist() + f["validation_y"].tolist()


def tensor(a: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Library mapping, numpy -> torch: `a` as `dtype` on `device`."""
    return torch.from_numpy(a).to(device=device, dtype=dtype)


def trained(
    cfg: GuiStackConfig, z: np.ndarray, y: list[str], w: np.ndarray, labels: list[str]
) -> tuple[list[torch.Tensor], float, float]:
    """(2)-(4) on S; bar `stack fit`; returns theta_T, L(theta_0), L(theta_T)."""
    params = init_params(z.shape[1], cfg.hidden, len(labels), cfg.seed, cfg.device)
    data = (
        tensor(z, cfg.device, torch.float32),
        tensor(np.array([labels.index(k) for k in y]), cfg.device, torch.int64),
        tensor(w, cfg.device, torch.float32),
    )
    with tqdm(total=cfg.steps, desc="stack fit", unit="step") as bar:
        initial, final = fit(params, *data, cfg, bar.update)
    return params, initial, final


def predicted(p: np.ndarray, labels: list[str]) -> list[str]:
    """(5): y_hat_f, the first label of Y with the largest p_f."""
    return [labels[i] for i in np.argmax(p, axis=1)]


def write_outputs(
    out: Path, model: dict, f: dict[str, np.ndarray], p: np.ndarray, y_hat: list[str]
) -> None:
    """Interface Produces: model.pt and predictions.jsonl (F^te in file order)."""
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model, out / "model.pt")
    labels = model["labels"]
    rows = zip(f["test_ids"].tolist(), f["test_y"].tolist(), y_hat, p.tolist())
    records = (
        {"id": i, "label": y, "predicted": yh, "p": dict(zip(labels, pf))}
        for i, y, yh, pf in rows
    )
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


def evaluated(f: dict[str, np.ndarray], y_hat: list[str]) -> dict:
    """(6): gui-masks (7) with y_hat; gui-masks (8) with VLM, (a)-(f) and `stack`."""
    labels, y = f["labels"].tolist(), f["test_y"].tolist()
    preds = {v: {c: f[k].tolist() for c, k in ks.items()} for v, ks in PREDS.items()}
    preds["stack"] = {c: y_hat for c in CRITERIA}
    return {
        "test": evaluation(
            y, y_hat, f["test_ids"].tolist(), f["test_g"].tolist(), labels
        ),
        "agreement_matrices": matrices(y, preds, labels[:-1]),
    }


def run(cfg: GuiStackConfig) -> None:
    """Notation, (1)-(6): model.pt, predictions.jsonl, report.json and report.md."""
    f = read_features(cfg.masks_dir / "features.npz")
    labels = f["labels"].tolist()
    x, y = training_set(f)
    n = {k: y.count(k) for k in labels}
    w = np.array([1 / n[k] for k in y])
    mu, s = moments(x)
    params, initial, final = trained(cfg, standardised(x, mu, s), y, w, labels)
    z_te = tensor(standardised(f["test_x"], mu, s), cfg.device, torch.float32)
    p = probabilities(params, z_te)
    y_hat = predicted(p, labels)
    model = {
        "params": [t.detach().cpu() for t in params],
        "mu": torch.from_numpy(mu),
        "s": torch.from_numpy(s),
        "columns": f["columns"].tolist(),
        "labels": labels,
    }
    write_outputs(cfg.out_dir, model, f, p, y_hat)
    train = {"frames": len(y), "per_label": n}
    train |= {"loss_initial": initial, "loss_final": final}
    write_report(cfg.out_dir, {"train": train} | evaluated(f, y_hat), TITLE)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.stacking.run <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config))
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
