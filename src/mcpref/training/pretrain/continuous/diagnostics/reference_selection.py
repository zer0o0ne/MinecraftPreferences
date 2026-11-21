"""Stratified original-edge references, independent of recurrent training order."""

import json
import math

from mcpref.diagnostics.preference.suites import GlobalFit, Stratum
from mcpref.training.pretrain.references.selection import (
    PARTS,
    selected_ordinals,
)


def _counts(index):
    comparable, incomparable = {part: {} for part in PARTS}, dict.fromkeys(PARTS, 0)
    for part, body in index.db.execute("SELECT part,body FROM edges ORDER BY id"):
        labels = json.loads(body)["labels"]
        if labels["inc"]:
            incomparable[part] += 1
        else:
            p = labels["p"]
            comparable[part][p] = comparable[part].get(p, 0) + 1
    return comparable, incomparable


def _logit(p):
    return math.log(p / (1 - p)) if p is not None and 0 < p < 1 else None


def global_fit(index) -> GlobalFit:
    comparable, inc = _counts(index)
    values = sorted(set(comparable["train"]) | set(comparable["validation"]))
    strata = tuple(
        Stratum(
            json.dumps(p),
            "comparable",
            p,
            *(comparable[part].get(p, 0) for part in PARTS),
        )
        for p in values
    ) + (Stratum("inc", "inc", None, inc["train"], inc["validation"]),)
    count = sum(comparable["train"].values())
    pairs = count + inc["train"]
    if not pairs:
        raise ValueError(
            "Continuous reference evaluation requires training preference edges"
        )
    total = sum(p * n for p, n in comparable["train"].items())
    p_inc, p_succ = inc["train"] / pairs, total / count if count else None
    return GlobalFit(
        pairs,
        inc["train"],
        count,
        total if count else None,
        p_inc,
        p_succ,
        _logit(p_inc),
        _logit(p_succ),
        strata,
    )


def select_reference(cfg, index) -> tuple[dict, GlobalFit]:
    fit = global_fit(index)
    chosen = selected_ordinals(cfg, fit)
    items = []
    for part in PARTS:
        ordinals = dict.fromkeys(chosen[part], 0)
        found = {key: [] for key in chosen[part]}
        for (body,) in index.db.execute(
            "SELECT body FROM edges WHERE part=? ORDER BY id", (part,)
        ):
            edge = json.loads(body)
            labels = edge["labels"]
            key = "inc" if labels["inc"] else json.dumps(labels["p"])
            ordinal = ordinals[key]
            ordinals[key] += 1
            if ordinal in chosen[part][key]:
                found[key].append(edge | {"part": part, "stratum": key})
        items.extend(edge for stratum in fit.strata for edge in found[stratum.key])
    return {
        "version": 1,
        "mode": "continuous",
        "seed": cfg.seed,
        "pairs_per_part": cfg.reference_pairs,
        "items": items,
    }, fit
