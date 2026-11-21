"""Recheck source joins against validated full rows, before establishing a grid."""


def blocks(sources, rows, maximum_gap: int):
    current = []
    for source in sources:
        if source["reason"] is not None:
            if current:
                yield current
                current = []
            yield [source]
            continue
        if current:
            previous = current[-1]
            a = rows[previous["offset"] + previous["rows"] - 1]
            b = rows[source["offset"]]
            reason = None
            if b["starts_session"]:
                reason = "world_start"
            elif not 0 <= b["milli"] - a["milli"] < maximum_gap:
                reason = "recording_gap"
            elif b["serverTick"] < a["serverTick"]:
                reason = "server_clock_reset"
            if reason:
                source["boundary_before"] = reason
                yield current
                current = []
        current.append(source)
    if current:
        yield current
