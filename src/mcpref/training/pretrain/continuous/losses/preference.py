"""Exactly-once preference edges, with the earlier history entirely detached."""

from itertools import groupby

import torch

from mcpref.training.pretrain.continuous.events import backward


def score(context, hooks, archive, batch: list, current, result, factor: float) -> None:
    P, loss = context.modules["P"], context.modules["P_loss"]
    # Histories can differ in length; equal shapes alone may share a model call.
    pairs = []
    for edge, other, old in batch:
        pair = (current, old) if edge["a"] != other else (old, current)
        pairs.append((edge, other, pair))
    for _, grouped in groupby(pairs, key=lambda row: tuple(v.shape[0] for v in row[2])):
        rows = list(grouped)
        A, B = (torch.stack([row[2][side] for row in rows]) for side in (0, 1))
        labels = [row[0]["labels"] for row in rows]
        s, t = P(A, B)
        terms = loss.terms((s, t), labels)
        weighted = factor * terms["total"].sum()
        records = [
            {"edge": edge, "s": s[i], "t": t[i], "loss": factor * terms["total"][i]}
            for i, (edge, _, _) in enumerate(rows)
        ]
        backward(context, hooks, {"P": weighted}, {"preference": records}, retain=True)
        result.weighted["P"] = result.weighted.get("P", 0) + weighted.detach()
        result.add("P", {key: value.sum() for key, value in terms.items()})
        result.preference_rows.extend(
            {"edge": row["edge"], "s": row["s"].detach(), "t": row["t"].detach()}
            for row in records
        )
        result.counts["P"] += len(rows)
        for edge, other, _ in rows:
            archive.consume(edge["id"], other)


def endpoints(context, hooks, archive, stream, observation, result, factor) -> None:
    for endpoint in sorted(observation.endpoints, key=lambda item: item["id"]):
        key = endpoint["id"]
        if key in archive.seen:
            continue
        current = torch.stack(
            [stream.recent[oid] for oid in endpoint["observation_ids"]]
        )
        pending, batch = endpoint["pending_edges"], []
        for edge in context.index.edges(key):
            if edge["id"] in archive.consumed:
                continue
            other = edge["b"] if edge["a"] == key else edge["a"]
            if other not in archive.live:
                continue
            batch.append((edge, other, archive.get(other, current)))
            pending -= 1
            if len(batch) == context.cfg.preference_micro_batch:
                score(context, hooks, archive, batch, current, result, factor)
                batch = []
        if batch:
            score(context, hooks, archive, batch, current, result, factor)
        archive.put(key, current, pending)
