"""(6)-(8) of specs/gui-masks.md, accumulated shard by shard: the criteria of the thresholds,
the metrics on F^te, the agreement between labellers; also used by specs/gui-stack.md."""

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction


def share(num: int, den: int) -> Fraction | None:
    """(6)-(8): an exact ratio; null when the denominator is 0."""
    return Fraction(num, den) if den else None


def agreement(n: Counter) -> Fraction | None:
    """(6): share of frames with y_hat = y; n counts the frames per (y, y_hat)."""
    return share(sum(c for (a, b), c in n.items() if a == b), sum(n.values()))


def macro_recall(n: Counter) -> Fraction | None:
    """(6): mean over the labels present of the share of their frames predicted
    correctly; n counts the frames per (y, y_hat)."""
    totals, hits = Counter(), Counter()
    for (a, b), c in n.items():
        totals[a] += c
        hits[a] += c * (a == b)
    recalls = [Fraction(hits[k], t) for k, t in totals.items() if t]
    return sum(recalls, Fraction(0)) / len(recalls) if recalls else None


CRITERIA = {"agreement": agreement, "macro_recall": macro_recall}
FLAGS = (("true", True), ("false", False))


def choose(acc: Sequence[Counter]) -> dict[str, int]:
    """(6): per criterion, the index of the first grid point (tau outermost, mu innermost)
    maximising it; acc[p] counts the frames of F^va per (y, y_hat) at grid point p."""
    return {
        name: max(range(len(acc)), key=lambda p: criterion(acc[p]))
        for name, criterion in CRITERIA.items()
    }


def tally(
    acc: dict,
    y: Sequence[str],
    y_hat: Sequence[str],
    ids: Sequence[str],
    g: Sequence[bool],
) -> None:
    """(7): adds the frames of one shard: counts per (y, y_hat, g) and (id, y, g) of the
    frames with y_hat != y, in order."""
    counts, errors = acc.setdefault("counts", Counter()), acc.setdefault("errors", [])
    for i, a, b, flag in zip(ids, y, y_hat, g):
        counts[a, b, bool(flag)] += 1
        if a != b:
            errors.append((i, a, bool(flag)))


def label_metrics(n: Counter, errors: list[tuple[str, str]], k: str) -> dict:
    """(7): counts, precision, recall, F1 and misclassified ids of label k."""
    n_true = sum(c for (a, _), c in n.items() if a == k)
    n_pred = sum(c for (_, b), c in n.items() if b == k)
    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "precision": share(n[k, k], n_pred),
        "recall": share(n[k, k], n_true),
        "f1": share(2 * n[k, k], n_true + n_pred),
        "errors": [i for i, a in errors if a == k],
    }


def metrics(acc: dict, flags: set[bool], labels: Sequence[str]) -> dict:
    """(7) on the tallied frames with g_f in `flags`: agreement, macro recall, per-label
    metrics, confusion matrix (y x y_hat)."""
    n = Counter()
    for (a, b, flag), c in acc.get("counts", Counter()).items():
        if flag in flags:
            n[a, b] += c
    errors = [(i, a) for i, a, flag in acc.get("errors", []) if flag in flags]
    return {
        "agreement": agreement(n),
        "macro_recall": macro_recall(n),
        "per_label": {k: label_metrics(n, errors, k) for k in labels},
        "confusion": {a: {b: n[a, b] for b in labels} for a in labels},
    }


def evaluation(acc: dict, labels: Sequence[str]) -> dict:
    """(7) from `acc` (empty: as on an empty set), and split by g_f true / false."""
    by_flag = {name: metrics(acc, {flag}, labels) for name, flag in FLAGS}
    return metrics(acc, {True, False}, labels) | {"by_flag": by_flag}


def tally_matrices(
    acc: dict, y: Sequence[str], preds: Mapping[str, Mapping[str, Sequence[str]]]
) -> None:
    """(8): adds one shard's labels of every labeller (VLM, then `preds` in its order;
    labeller -> threshold choice -> y_hat): per choice and label k, |F_k| and the
    agreeing frames of each pair of labellers."""
    names = acc.setdefault("labellers", ["vlm", *preds])
    for c in CRITERIA:
        n, agree = acc.setdefault(c, (Counter(), {}))
        for row in zip(y, *(p[c] for p in preds.values())):
            for k in set(row):
                n[k] += 1
                a = agree.setdefault(k, [[0] * len(names) for _ in names])
                for i, ri in enumerate(row):
                    for j, rj in enumerate(row):
                        a[i][j] += ri == rj


def agreement_matrix(acc: dict, c: str, k: str) -> dict:
    """(8): A_k[i, j] over F_k, reported with |F_k|; no matrix when F_k is empty."""
    n, agree = acc.get(c, (Counter(), {}))
    names, a = acc["labellers"], agree.get(k)
    matrix = [[share(x, n[k]) for x in row] for row in a] if a else None
    return {"n": n[k], "labellers": names, "matrix": matrix}


def matrices(acc: dict, names: Sequence[str]) -> dict:
    """(8) from `acc`, per threshold choice, one matrix per class of `names`."""
    return {c: {k: agreement_matrix(acc, c, k) for k in names} for c in CRITERIA}
