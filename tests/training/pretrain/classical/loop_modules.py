"""Tiny fake components implementing the protocol of specs/pretrain.md (Interface), in
float64, written from the spec alone: a linear $\\phi$ over the first bytes of each state
frame (drawing from torch's global generator), a loss of $\\phi$ with a parameter (a dual
variable) and a buffer, a linear $\\pi$ (called with the clips) and $P$; $\\phi$ declares two
groups. Each records its calls (name, training mode, what it was given) in `log` when one is
set."""

import numpy as np
import torch
from torch import nn

L, W, H = 3, 4, 2  # states, width, height of every clip
FRAME = W * H * 3 // 2  # bytes of a yuv420p state frame
F, D, ACTIONS = 4, 2, 3  # bytes of a frame phi reads, latent size, actions of pi
F64 = torch.float64
NAMES = ("phi", "phi_loss", "pi", "pi_loss", "P", "P_loss")  # module order


def fixed(seed: int, *shape: int) -> nn.Parameter:
    """A parameter of fixed values, the same in every instance (no torch generator)."""
    values = np.random.default_rng(seed).normal(scale=0.5, size=shape)
    return nn.Parameter(torch.tensor(values, dtype=F64))


class Fake(nn.Module):
    """A component; `cut`: its inputs detached (a stop-gradient of the component's own)."""

    def __init__(self, cut: bool = False):
        super().__init__()
        self.cut, self.log, self.calls = cut, None, 0

    def note(self, name: str, **what) -> None:
        self.calls += 1
        if self.log is not None:
            self.log.append((name, self.training, what))

    def cut_off(self, x):
        return x.detach() if self.cut else x


class Phi(Fake):
    """Z[i, t] = (A x_{i,t} + c)(1 + 0.1 u_i): x_{i,t} the first F bytes of state frame t of
    clip i over 255, u_i one `torch.rand` per clip when `noise`; groups phi_A, phi_c."""

    def __init__(self, noise: bool = True):
        super().__init__()
        self.A, self.c, self.noise = fixed(1, D, F), fixed(2, D), noise

    def prepare(self, clips: list) -> list:
        return [None] * len(clips)

    def forward(self, clips: list, states: list | None = None) -> torch.Tensor:
        self.note("phi", ids=[c.info["id"] for c in clips])
        x = torch.tensor(np.stack([c.frames[:, :F] for c in clips]), dtype=F64) / 255
        z = x @ self.A.T + self.c
        if self.noise:
            z = z * (1 + 0.1 * torch.rand(len(clips), 1, 1, dtype=F64))
        return z

    def groups(self) -> dict:
        return {"phi_A": [self.A], "phi_c": [self.c]}


class PhiLoss(Fake):
    """A scalar fake loss with the diagnostic terms and tensors of the revised protocol."""

    def __init__(self, stateful: bool = True, cut: bool = False):
        super().__init__(cut)
        self.lam, self.stateful = nn.Parameter(torch.tensor(0.7, dtype=F64)), stateful
        self.register_buffer("seen", torch.zeros((), dtype=F64))

    def terms(
        self, Z: torch.Tensor, clips: list, states: list | None = None
    ) -> tuple[dict, dict]:
        self.note("phi_loss", Z=Z.detach().clone(), ids=[c.info["id"] for c in clips])
        Z = self.cut_off(Z)
        w = torch.tensor([c.lines[0]["w"] for c in clips], dtype=F64)
        steps = ((Z[:, 1:] - Z[:, :-1]) ** 2).sum((1, 2))
        scale = 1 / (1 + 0.01 * float(self.seen))
        if self.stateful:
            self.seen += len(clips)
        last = (Z[:, -1] ** 2).sum(1)
        push = self.lam * w * steps * scale
        constraint = -0.5 * self.lam.expand(len(clips))
        dual = 0.1 * self.lam**2 * torch.ones(len(clips), dtype=F64)
        norm = 0.05 * last
        total = self.lam * (w * steps * scale - 0.5) + 0.1 * self.lam**2 + 0.05 * last
        d_alpha = torch.linalg.vector_norm(Z[:, 1:] - Z[:, :-1], ord=2, dim=-1)
        z_norm = torch.linalg.vector_norm(Z, ord=2, dim=-1)
        diagnostics = {
            "changed": torch.ones_like(d_alpha, dtype=torch.bool),
            "d_alpha": d_alpha,
            "z_norm": z_norm,
            "hinge": torch.clamp(z_norm**2 - 1, min=0),
        }
        return {
            "push": push,
            "constraint": constraint,
            "dual": dual,
            "norm": norm,
            "total": total,
        }, diagnostics

    def forward(self, Z: torch.Tensor, clips: list) -> torch.Tensor:
        return self.terms(Z, clips)[0]["total"]

    @property
    def log_lambda(self) -> nn.Parameter:
        return self.lam

    def groups(self) -> dict:
        return {"dual": [self.lam]}


class Pi(Fake):
    """A linear policy returning logits, shifts and current weights for every state."""

    def __init__(self):
        super().__init__()
        self.Wa, self.ba, self.V = (
            fixed(3, D, ACTIONS),
            fixed(4, ACTIONS),
            fixed(5, D, D),
        )

    def forward(self, Z: torch.Tensor, clips: list) -> tuple:
        self.note("pi", Z=Z.detach().clone(), ids=[c.info["id"] for c in clips])
        w = torch.tensor([c.lines[0]["w"] for c in clips], dtype=F64)
        a = torch.tensor([c.info["a"] for c in clips])
        current = torch.nn.functional.one_hot(a, ACTIONS).to(F64)
        return Z @ self.Wa + self.ba, Z @ self.V + 0.1 * w[:, None, None], current

    def groups(self) -> dict:
        return {"pi": [self.Wa, self.ba, self.V]}


def pi_terms(logits: torch.Tensor, shift: torch.Tensor, Z: torch.Tensor, clips: list):
    """sum_t CE(logits_{i,t}, a_{i,t}) + sum_{t < L} ||shift_{i,t} - (Z_{i,t+1} - Z_{i,t})||^2,
    a_{i,t} the fake's own key `a` of the clip's clip.json."""
    a = torch.tensor([c.info["a"] for c in clips])
    ce = -torch.log_softmax(logits, -1).gather(-1, a[..., None])[..., 0].sum(1)
    return ce + ((shift[:, :-1] - (Z[:, 1:] - Z[:, :-1])) ** 2).sum((1, 2))


class PiLoss(Fake):
    """l_i = pi_terms + 0 k: k a parameter whose gradient is zero (not None); group idle."""

    def __init__(self, cut: bool = False):
        super().__init__(cut)
        self.k = fixed(6, 2)

    def terms(
        self, out: tuple, Z: torch.Tensor, clips: list
    ) -> tuple[dict, dict, dict]:
        self.note("pi_loss", n=len(clips))
        logits, shift, current = out
        logits, shift, current, Z = (
            self.cut_off(x) for x in (logits, shift, current, Z)
        )
        a = torch.tensor([c.info["a"] for c in clips])
        action = -torch.log_softmax(logits, -1).gather(-1, a[..., None])[..., 0].sum(1)
        factual = Z[:, 1:] - Z[:, :-1]
        shift_term = ((shift[:, :-1] - factual) ** 2).sum((1, 2))
        total = pi_terms(logits, shift, Z, clips) + 0.0 * self.k.sum()
        valid = torch.ones(len(clips), L - 2, dtype=torch.bool)
        q = torch.nn.functional.one_hot(a[:, : L - 2], ACTIONS).to(F64)
        observations = {
            "q": q,
            "current": current[:, : L - 2],
            "zero_shift": (factual**2).sum((1, 2)),
        }
        return (
            {"action": action, "shift": shift_term, "total": total},
            {
                "action_valid": valid,
                "shift_valid": valid,
            },
            observations,
        )

    def forward(self, out: tuple, Z: torch.Tensor, clips: list) -> torch.Tensor:
        return self.terms(out, Z, clips)[0]["total"]

    def groups(self) -> dict:
        return {"idle": [self.k]}


class Pref(Fake):
    """A tied linear preference head returning the two logits; group pref."""

    def __init__(self):
        super().__init__()
        from types import SimpleNamespace

        self.u, self.beta, self.v = fixed(7, D), fixed(8, 1), fixed(9, D)
        self.pref = SimpleNamespace(weight=self.u)
        self.inc = SimpleNamespace(weight=self.v, bias=self.beta)

    def forward(self, Za: torch.Tensor, Zb: torch.Tensor) -> tuple:
        self.note("P", Za=Za.detach().clone(), Zb=Zb.detach().clone())
        a, b = Za[:, -1], Zb[:, -1]
        return (a - b) @ self.u + self.beta, (a + b) @ self.v

    def groups(self) -> dict:
        return {"pref": [self.u, self.beta, self.v]}


class PLoss(Fake):
    """A two-head fake preference loss with the revised per-pair terms."""

    def terms(self, out: tuple, labels: list) -> dict[str, torch.Tensor]:
        self.note("P_loss", labels=labels)
        s, t = (self.cut_off(x) for x in out)
        p = torch.tensor(
            [lab["p"] if lab["inc"] == 0 else 0.5 for lab in labels], dtype=F64
        )
        inc = torch.tensor([lab["inc"] for lab in labels], dtype=F64)
        succ = 2 * torch.nn.functional.binary_cross_entropy_with_logits(
            s, p, reduction="none"
        )
        succ = torch.where(inc == 0, succ, torch.zeros_like(succ))
        inc_loss = 2 * torch.nn.functional.binary_cross_entropy_with_logits(
            t, inc, reduction="none"
        )
        return {"inc": inc_loss, "succ": succ, "total": inc_loss + succ}

    def forward(self, out: tuple, labels: list) -> torch.Tensor:
        return self.terms(out, labels)["total"]

    def groups(self) -> dict:
        return {}


def fakes(pairs: bool = True, noise: bool = True, stateful: bool = True) -> dict:
    """The modules in module order, as a caller builds them; P and P_loss iff `pairs`."""
    mods = {"phi": Phi(noise), "phi_loss": PhiLoss(stateful), "pi": Pi(),
            "pi_loss": PiLoss(), "P": Pref(), "P_loss": PLoss()}  # fmt: skip
    return mods if pairs else {k: mods[k] for k in NAMES[:4]}


def group_order(mods: dict) -> list[tuple[str, list]]:
    """Groups: module order, then each module's declaration order."""
    return [(q, ps) for m in mods.values() for q, ps in m.groups().items()]


def shared_log(mods: dict) -> list:
    """One log for every module of `mods`."""
    log: list = []
    for m in mods.values():
        m.log = log
    return log
