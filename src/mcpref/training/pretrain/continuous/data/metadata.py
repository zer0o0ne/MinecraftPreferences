"""One metadata shard at a time, including intervals crossing storage boundaries."""

import json

from mcpref.datasets.continuous.config import digest
from mcpref.gui.mask import state_mask
from mcpref.infrastructure.shards.store import shard_path


def read_shard(root, index):
    path = shard_path(root / "shards", index)
    state = json.loads((path / "state.json").read_text())
    values = []
    for name in ("observations.jsonl", "intervals.jsonl"):
        file = path / name
        if digest(file) != state["files"][name]:
            raise ValueError(f"Continuous metadata checksum differs: {file}")
        values.append([json.loads(line) for line in file.read_text().splitlines()])
    observations, intervals = values
    if (
        len(observations) != state["observations"]
        or len(intervals) != state["intervals"]
    ):
        raise ValueError("Continuous metadata shard length differs")
    by_right = {item["right"]: item for item in intervals}
    if len(by_right) != len(intervals):
        raise ValueError("Repeated continuous interval")
    return observations, by_right


def interval(left, right, raw, names):
    if raw is None:
        return None
    if raw["left"] != left["id"] or raw["right"] != right["id"]:
        raise ValueError("Continuous interval references different observations")
    actions = raw["actions"]
    if raw["n"] != len(actions) or any(a not in names for a in actions):
        raise ValueError("Continuous interval action metadata differs")
    legal = [a for a in actions if left["mask"][names[a]]]
    return {
        **raw,
        "a": raw["left"],
        "b": raw["right"],
        "legal_actions": legal,
        "valid": bool(raw["valid"] and raw["known"] and legal),
    }


def rows(root, fragment, masks):
    previous, ordinal = None, 0
    names = {name: i for i, name in enumerate(masks.names)}
    for shard in range(fragment["first_shard"], fragment["next_shard"]):
        observations, intervals = read_shard(root, shard)
        for position, item in enumerate(observations):
            if item["fragment_id"] != fragment["id"]:
                raise ValueError("Storage shard crosses continuous fragments")
            line = item["meta"]
            if ordinal == 0 and line != fragment["origin"]:
                raise ValueError(
                    "Continuous fragment origin is not its first retained observation"
                )
            item = item | {
                "index": ordinal,
                "shard": shard,
                "position": position,
                "mask": state_mask(
                    masks,
                    item["gui"],
                    item["screen_class"],
                    line["pitch"],
                    line["hotbar"],
                ).tolist(),
            }
            raw = intervals.pop(item["id"], None)
            adjacent = (
                previous is not None
                and previous["grid_index"] + 1 == item["grid_index"]
            )
            if (raw is not None) != adjacent:
                raise ValueError("Missing or fabricated continuous grid interval")
            if previous and item["grid_index"] <= previous["grid_index"]:
                raise ValueError("Continuous observations are not chronological")
            item["incoming"] = interval(previous, item, raw, names)
            yield item
            previous, ordinal = item, ordinal + 1
        if intervals:
            raise ValueError("Continuous interval has no right observation")
    if ordinal != fragment["observations"]:
        raise ValueError("Continuous fragment observation count differs")


def population(root, fragment, masks, K):
    phi, policy, previous = 0, 0, False
    for item in rows(root, fragment, masks):
        valid = bool(item["incoming"] and item["incoming"]["valid"])
        phi += valid
        policy += previous and valid
        previous = valid
    return {"N_phi": phi, "N_pi": policy, "M": (fragment["observations"] + K - 1) // K}
