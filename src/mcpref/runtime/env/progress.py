"""Required engine action-progress observation; specs/runtime-stagnation.md."""

import math


def action_progress(info: dict) -> dict:
    """Read actual progress, never replace an absent engine signal with zeros."""
    try:
        value = info["action_progress"]
        if value["version"] != 1:
            raise ValueError("unsupported telemetry version")
        mining, use = value["mining"], value["item_use"]
        progress = float(mining["progress"])
        target = tuple(int(x) for x in mining["target"])
        elapsed, remaining = int(use["elapsed"]), int(use["remaining"])
        duration, charge = int(use["duration"]), int(use["charge_duration"])
        if len(target) != 3 or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("invalid mining progress")
        if min(elapsed, duration, charge) < 0:
            raise ValueError("invalid item-use progress")
        return {
            "version": 1,
            "mining": {
                "active": bool(mining["active"]),
                "target": target,
                "progress": progress,
            },
            "item_use": {
                "active": bool(use["active"]),
                "elapsed": elapsed,
                "remaining": remaining,
                "action": str(use["action"]),
                "duration": duration,
                "charge_duration": charge,
            },
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "engine action-progress telemetry missing or invalid; run scripts/setup_env.sh"
        ) from error


def advanced(before: dict, after: dict) -> bool:
    """Increasing destruction or active use counts; targeting and release do not."""
    old, new = before["mining"], after["mining"]
    baseline = (
        old["progress"] if old["active"] and old["target"] == new["target"] else 0
    )
    if new["active"] and new["progress"] > baseline:
        return True
    old, new = before["item_use"], after["item_use"]
    baseline = use_progress(old) if old["action"] == new["action"] else 0
    return use_progress(new) > baseline


def use_progress(use: dict) -> float:
    """Only the engine-defined finite consumption/charging phase advances."""
    if not use["active"] or not use["charge_duration"]:
        return 0.0
    return min(use["elapsed"] / use["charge_duration"], 1.0)
