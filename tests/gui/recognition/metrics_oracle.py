"""Naive references for specs/gui-masks.md (6)-(8): exact fractions over lists of labels;
nothing from src/."""

from fractions import Fraction


def criteria(y: list, yhat: list) -> dict:
    """(6), (7): agreement and macro recall (None on an empty set)."""
    if not y:
        return {"agreement": None, "macro_recall": None}
    recalls = [recall(y, yhat, k) for k in dict.fromkeys(y)]
    agree = Fraction(sum(a == b for a, b in zip(y, yhat)), len(y))
    return {"agreement": agree, "macro_recall": sum(recalls) / len(recalls)}


def recall(y: list, yhat: list, k: str) -> Fraction:
    return Fraction(sum(a == b == k for a, b in zip(y, yhat)), y.count(k))


def choose(y: list, preds: dict, grid: tuple) -> dict:
    """(6): criterion -> the first (tau, theta, mu) in grid order with the largest value;
    `preds[(tau, theta, mu)]` are the predictions on the validation frames."""
    out = {}
    for crit in ("agreement", "macro_recall"):
        best = None
        for point in ((t, th, m) for t in grid[0] for th in grid[1] for m in grid[2]):
            value = criteria(y, preds[point])[crit]
            if best is None or value > best[0]:
                best = (value, point)
        out[crit] = best[1]
    return out


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else float(Fraction(num, den))


def metrics(ids: list, y: list, yhat: list, labels: list) -> dict:
    """(7) on one set of frames: fractions as floats, errors sorted, no zero confusion cells."""
    crit = {k: None if v is None else float(v) for k, v in criteria(y, yhat).items()}
    per_label, confusion = {}, {}
    for k in labels:
        tp = sum(a == b == k for a, b in zip(y, yhat))
        n_true, n_pred = y.count(k), yhat.count(k)
        per_label[k] = {"n_true": n_true, "n_pred": n_pred}
        per_label[k].update(precision=ratio(tp, n_pred), recall=ratio(tp, n_true))
        per_label[k]["f1"] = ratio(2 * tp, n_true + n_pred)
        per_label[k]["errors"] = sorted(
            i for i, a, b in zip(ids, y, yhat) if a == k != b
        )
    for a, b in zip(y, yhat):
        confusion.setdefault(a, {}).setdefault(b, 0)
        confusion[a][b] += 1
    return {**crit, "per_label": per_label, "confusion": confusion}


def matrices(classes: list, labellers: list) -> dict:
    """(8): class -> (|F_k|, matrix as floats) for the classes some labeller assigns."""
    out = {}
    for k in classes:
        fk = [
            f for f in range(len(labellers[0])) if any(lab[f] == k for lab in labellers)
        ]
        if fk:
            agree = [
                [sum(li[f] == lj[f] for f in fk) for lj in labellers]
                for li in labellers
            ]
            out[k] = (
                len(fk),
                [[float(Fraction(n, len(fk))) for n in row] for row in agree],
            )
    return out


def num(v):
    """A fraction as a float (Fraction or float accepted); None stays None."""
    return None if v is None else float(v)


def normal(part: dict) -> dict:
    """One set of (7) in the form of `metrics`: fractions as floats, errors sorted, confusion
    without zero cells."""
    per_label = {
        k: {n: sorted(x) if n == "errors" else num(x) for n, x in v.items()}
        for k, v in part["per_label"].items()
    }
    rows = {y: {p: n for p, n in r.items() if n} for y, r in part["confusion"].items()}
    crit = {k: num(part[k]) for k in ("agreement", "macro_recall")}
    confusion = {y: r for y, r in rows.items() if r}
    return {**crit, "per_label": per_label, "confusion": confusion}
