"""External references and frozen records of specs/rl-preference-pairs.md."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, order=True)
class StateRef:
    """The stable id (episode, index) of a raw game situation."""

    episode: str
    index: int


Encoder = Callable[[Sequence[StateRef]], Sequence[Tensor]]
"""One batch-1 latent history Y per input reference, in input order."""


@dataclass(frozen=True)
class CurrentComparison:
    """The ordered raw result (a, b, p, inc) of one current-cycle comparison."""

    a: StateRef
    b: StateRef
    p: float
    inc: float


@dataclass(frozen=True)
class Comparison:
    """An oriented arithmetic-mean record attached to a History vertex."""

    ref: StateRef
    p: float
    inc: float
    n: int


@dataclass(frozen=True)
class Vertex:
    """A seed vertex; n_u is derived from its currently attached records."""

    ref: StateRef
    key: tuple[float, ...]
    unused: int
    comparisons: tuple[Comparison, ...]

    @property
    def n(self) -> int:
        """The stored comparison count n_u."""
        return sum(c.n for c in self.comparisons)


def ref_dict(ref: StateRef) -> dict:
    """The JSON representation of a StateRef."""
    return {"episode": ref.episode, "index": ref.index}


def state_ref(value: dict) -> StateRef:
    """A StateRef from its JSON representation."""
    return StateRef(value["episode"], value["index"])


def comparison_dict(c: CurrentComparison) -> dict:
    """The raw-work JSON representation of a CurrentComparison."""
    return {"a": ref_dict(c.a), "b": ref_dict(c.b), "p": c.p, "inc": c.inc}


def current_comparison(value: dict) -> CurrentComparison:
    """A CurrentComparison from its raw-work JSON representation."""
    return CurrentComparison(
        state_ref(value["a"]), state_ref(value["b"]), value["p"], value["inc"]
    )


def comparison_json(c: Comparison) -> dict:
    """The History JSON representation of an oriented comparison record."""
    return ref_dict(c.ref) | {"p": c.p, "inc": c.inc, "n": c.n}


def vertex_json(v: Vertex) -> dict:
    """The History JSON representation of a vertex."""
    return ref_dict(v.ref) | {
        "key": list(v.key),
        "unused": v.unused,
        "comparisons": [comparison_json(c) for c in v.comparisons],
    }


def read_vertex(value: dict) -> Vertex:
    """Read one vertex without retaining its containing shard."""
    comparisons = tuple(
        Comparison(state_ref(c), c["p"], c["inc"], c["n"]) for c in value["comparisons"]
    )
    return Vertex(state_ref(value), tuple(value["key"]), value["unused"], comparisons)


def attach(v: Vertex, record: Comparison) -> Vertex:
    """Attach a new oriented record or replace its previous generation."""
    records = tuple(record if c.ref == record.ref else c for c in v.comparisons)
    if all(c.ref != record.ref for c in v.comparisons):
        records += (record,)
    return Vertex(v.ref, v.key, v.unused, records)


def update(v: Vertex, partner: StateRef, p: float, inc: float) -> Vertex:
    """Eq. (7): one oriented running arithmetic-mean update."""
    old = next((c for c in v.comparisons if c.ref == partner), None)
    n = 1 if old is None else old.n + 1
    mean_p = p if old is None else old.p + (p - old.p) / n
    mean_inc = inc if old is None else old.inc + (inc - old.inc) / n
    return attach(v, Comparison(partner, mean_p, mean_inc, n))
