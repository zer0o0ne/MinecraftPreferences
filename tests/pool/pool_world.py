"""Fake VPT episodes for the tests of specs/preference-pool.md: the episodes of
tests/clips/collect_world.py named `data/<version>/<alias>-<hex>-<date>-<time>.mp4` ([R]),
line t with `milli` M + 50 t and `serverTick` T + t, line 0 with the head's `stats` (empty
for a session start); `chain` builds the pieces of one recording, each joining the one
before by (S); EPISODES is the default world."""

import dataclasses
import json
from dataclasses import dataclass, field

from clips.collect_world import GOOD, PLAY, Ep, jsonl, options

REST = "minecraft.custom:minecraft.time_since_rest"
DEATH = "minecraft.custom:minecraft.time_since_death"
BAD_HEAD = b'{"stats": {}, "serverTick": 1, "milli": 2\n' + jsonl("w" * 6, False)


@dataclass(frozen=True)
class PoolEp(Ep):
    """An episode of collect_world.Ep whose served `.jsonl` (when GOOD) carries `milli` +
    50 t and `tick` + t on line t and `stats` on line 0 ({} when `session`)."""

    tick: int = 0
    milli: int = 0
    stats: dict = field(default_factory=lambda: {PLAY: 1})

    def files(self) -> dict:
        out = super().files()
        if self.jsonl is GOOD:
            lines = [json.loads(x) for x in out[".jsonl"].split(b"\n")[:-1]]
            for t, line in enumerate(lines):
                line["milli"], line["serverTick"] = self.milli + 50 * t, self.tick + t
            lines[0]["stats"] = {} if self.session else self.stats
            dumps = (json.dumps(x, separators=(",", ":"), ensure_ascii=False) for x in lines)  # fmt: skip
            out[".jsonl"] = "".join(d + "\n" for d in dumps).encode()
        return out


def relpath(version: str, alias: str, hexa: str, k: int) -> str:
    return f"data/{version}/{alias}-{hexa}-{20211000 + k}-{120000 + k}.mp4"


def chain(version: str, alias: str, hexa: str, scripts, session: bool = True,
          t0: int = 1000, m0: int = 10**6, k0: int = 0) -> list[PoolEp]:  # fmt: skip
    """Pieces k0, k0 + 1, ... of one recording: piece i + 1 starts len(script_i) + 1 ticks
    and 50 ms per tick after piece i, its PLAY stat larger; piece 0 a session start iff
    `session`."""
    pieces, t, m = [], t0, m0
    for i, script in enumerate(scripts):
        rp = relpath(version, alias, hexa, k0 + i)
        start = session and i == 0
        pieces.append(
            PoolEp(rp, script, session=start, tick=t, milli=m, stats={PLAY: t})
        )
        t, m = t + len(script) + 1, m + 50 * (len(script) + 1)
    return pieces


def replace(ep: PoolEp, **changes) -> PoolEp:
    return dataclasses.replace(ep, **changes)


def default_episodes() -> dict:
    """With states 3, delta 2, per_episode 2: `w`*16 gives two clips, `w`*6 one, `w`*4
    none. Sessions: alpha (6xx; three pieces, out of index order, one listed twice), beta
    (one clip), alpha again in 7xx (its own group), zeta (missing options, missing video),
    eta (one clip and `no_clip`), theta (`options`), iota (a malformed later line), kap-pa
    (four clips; its next piece under another hex is an orphan). Not in a session: an entry
    below min_version, the heads `missing`, `empty`, `malformed`, an orphan chain, the
    continuation of 6xx alpha listed in 7xx."""
    a6 = chain("6.13", "alpha", "a1" * 6, ["w" * 16, "w" * 6, "w" * 6])
    b6 = chain("6.13", "beta", "b2" * 6, ["w" * 6], t0=50)
    old = chain("6.9", "gamma", "c3" * 6, ["w" * 6])
    a7 = chain(
        "7.1", "alpha", "a1" * 6, ["w" * 6, "w" * 16], t0=5000, m0=5 * 10**6, k0=5
    )
    # in 7xx, the next piece of alpha's 6xx session: an orphan of its own group
    a7.append(chain("7.1", "alpha", "a1" * 6, ["w" * 6], False, 1031, 1001550, 9)[0])
    d7 = chain("7.2", "delta", "d4" * 6, ["w" * 6] * 3)
    d7 = [replace(d7[0], jsonl=None), replace(d7[1], jsonl=b""), replace(d7[2], jsonl=BAD_HEAD)]  # fmt: skip
    e7 = chain("7.2", "eps", "e5" * 6, ["w" * 6] * 2, session=False)
    z8 = chain("8.1", "zeta", "f6" * 6, ["w" * 6] * 2)
    z8 = [replace(z8[0], options=None), replace(z8[1], video=None)]
    h8 = chain("8.1", "eta", "0a" * 6, ["w" * 6, "w" * 4])
    t9 = chain("9.1", "theta", "0b" * 6, ["w" * 16, "w" * 6])
    t9[1] = replace(t9[1], options=options(autoJump=True))
    i9 = chain("9.1", "iota", "0c" * 6, ["w" * 6] * 2)
    i9[1] = replace(i9[1], jsonl=i9[1].files()[".jsonl"] + b'{"isGuiOpen": tr\n')
    k10 = chain("10.0", "kap-pa", "0d" * 6, ["w" * 16, "w" * 16, "w" * 6])
    k10[2] = replace(k10[2], relpath=relpath("10.0", "kap-pa", "0e" * 6, 2))
    return {
        "6xx": [a6[0], b6[0], a6[2], old[0], a6[1], a6[0]],
        "7xx": [d7[0], a7[0], e7[0], d7[1], a7[2], a7[1], e7[1], d7[2]],
        "8xx": [z8[0], h8[0], z8[1], h8[1]],
        "9xx": [t9[0], i9[0], t9[1], i9[1]],
        "10xx": [k10[0], k10[2], k10[1]],
    }


EPISODES = default_episodes()


def small_episodes() -> dict:
    """The sessions alpha (6xx), beta, zeta, theta and kap-pa of EPISODES, nothing else:
    five draws, three of them with clips."""
    keep = ("alpha-", "beta-", "zeta-", "theta-", "kap-pa-0d")
    return {s: [e for e in es if s != "7xx" and e.stem.startswith(keep)]
            for s, es in EPISODES.items()}  # fmt: skip
