"""(6)-(8) of specs/gui-masks.md: thresholds chosen on F^va, metrics on F^te, agreement
between labellers."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

from tqdm import tqdm

from mcpref.gui.recognise import Point, Scores, predictions


def share(num: int, den: int) -> Fraction | None:
    """(6)-(8): an exact ratio; null when the denominator is 0."""
    return Fraction(num, den) if den else None


def agreement(y: list[str], y_hat: list[str]) -> Fraction | None:
    """(6): share of frames with y_hat = y."""
    return share(sum(a == b for a, b in zip(y, y_hat)), len(y))


def macro_recall(y: list[str], y_hat: list[str]) -> Fraction | None:
    """(6): mean over the labels present in y of the share of their frames predicted."""
    hits = Counter(a for a, b in zip(y, y_hat) if a == b)
    recalls = [Fraction(hits[k], n) for k, n in Counter(y).items()]
    return sum(recalls, Fraction(0)) / len(recalls) if recalls else None


CRITERIA = {"agreement": agreement, "macro_recall": macro_recall}


def choose(
    grid: Sequence[Point],
    s: Scores,
    y: list[str],
    order: list[str],
    step: Callable[[], None],
) -> dict[str, Point]:
    """(6): per criterion, the first grid point (tau outermost, mu innermost) maximising
    it; step() after each grid point's predictions."""
    preds = []
    for point in grid:
        preds.append((point, predictions(s, point, order)))
        step()
    return {
        name: max(preds, key=lambda pp: criterion(y, pp[1]))[0]
        for name, criterion in CRITERIA.items()
    }


def thresholds(
    grid: Sequence[Point], s: Mapping[str, Scores], y: list[str], order: list[str]
) -> dict[str, dict[str, Point]]:
    """(6) for the variants in turn; bar `thresholds`."""
    with tqdm(total=len(s) * len(grid), desc="thresholds", unit="point") as bar:
        return {v: choose(grid, sv, y, order, bar.update) for v, sv in s.items()}


def label_metrics(y: list[str], y_hat: list[str], ids: list[str], k: str) -> dict:
    """(7): counts, precision, recall, F1 and misclassified ids of label k."""
    tp = sum(a == b == k for a, b in zip(y, y_hat))
    n_true, n_pred = y.count(k), y_hat.count(k)
    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "precision": share(tp, n_pred),
        "recall": share(tp, n_true),
        "f1": share(2 * tp, n_true + n_pred),
        "errors": [i for i, a, b in zip(ids, y, y_hat) if a == k != b],
    }


def metrics(frames: list[dict], y_hat: list[str], labels: list[str]) -> dict:
    """(7): agreement, macro recall, per-label metrics, confusion matrix (y x y_hat)."""
    y, ids = [f["label"] for f in frames], [f["id"] for f in frames]
    pairs = Counter(zip(y, y_hat))
    return {
        "agreement": agreement(y, y_hat),
        "macro_recall": macro_recall(y, y_hat),
        "per_label": {k: label_metrics(y, y_hat, ids, k) for k in labels},
        "confusion": {a: {b: pairs[a, b] for b in labels} for a in labels},
    }


def evaluation(frames: list[dict], y_hat: list[str], labels: list[str]) -> dict:
    """(7): metrics on F^te, and on its frames with g_f true / false."""
    by_flag = {}
    for name, flag in (("true", True), ("false", False)):
        keep = [i for i, f in enumerate(frames) if f["line"]["isGuiInventory"] == flag]
        by_flag[name] = metrics(
            [frames[i] for i in keep], [y_hat[i] for i in keep], labels
        )
    return metrics(frames, y_hat, labels) | {"by_flag": by_flag}


def agreement_matrix(labellers: dict[str, list[str]], k: str) -> dict:
    """(8): A_k[i, j] over F_k, the frames to which some labeller assigns k."""
    f_k = [row for row in zip(*labellers.values()) if k in row]
    n = range(len(labellers))
    matrix = [[share(sum(r[i] == r[j] for r in f_k), len(f_k)) for j in n] for i in n]
    return {
        "n": len(f_k),
        "labellers": list(labellers),
        "matrix": matrix if f_k else None,
    }


def matrices(test: list[dict], preds: Mapping[str, dict], names: list[str]) -> dict:
    """(8) per threshold choice: labellers VLM, then the variants; one matrix per class."""
    y = [f["label"] for f in test]
    return {
        c: {
            k: agreement_matrix({"vlm": y} | {v: p[c] for v, p in preds.items()}, k)
            for k in names
        }
        for c in CRITERIA
    }
