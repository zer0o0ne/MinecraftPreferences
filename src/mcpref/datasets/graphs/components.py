"""Connected components: the groups of specs/train-index.md Notation; (G) 2 of
specs/preference-pool.md."""

from collections.abc import Hashable


def components(
    episodes: list[Hashable], edges: list[tuple[Hashable, Hashable]]
) -> list[int]:
    """Per vertex in `episodes` order, the j of its connected component under `edges`,
    the components ordered by their first vertex in that order."""
    linked = {e: [] for e in episodes}
    for a, b in edges:
        linked[a].append(b)
        linked[b].append(a)
    group, j = {}, 0
    for e in episodes:
        if e in group:
            continue
        group[e], stack = j, [e]
        while stack:
            for f in linked[stack.pop()]:
                if f not in group:
                    group[f] = j
                    stack.append(f)
        j += 1
    return [group[e] for e in episodes]
