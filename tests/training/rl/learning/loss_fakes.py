"""Tiny modules and an independent float64 oracle for rl-cycle equations (2)-(3)."""

from types import SimpleNamespace

import torch
from torch import nn

from mcpref.runtime.mcts.tree import LatentHistory
from mcpref.training.rl.ppo.config import PpoConfig
from mcpref.training.rl.ppo.metra import MetraLoss

F64 = torch.float64
ALPHA = torch.tensor([1.5, 0.5], dtype=F64)
BEHAVIOR = torch.tensor([0.2, 0.3, 0.5], dtype=F64)


def base(index, dtype=F64):
    return torch.tensor([index + 1.0, 0.5 * index - 1.0], dtype=dtype)


def history(index, phi):
    y = base(index, phi.scale.dtype) * phi.scale
    c = torch.zeros(1, 1, 4, dtype=y.dtype, device=y.device)
    c[0, 0, -1] = 1.0
    return LatentHistory(y.reshape(1, 1, 2), c)


class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.2, dtype=F64))
        self.head = nn.Sequential(nn.Linear(1, 1, dtype=F64))
        with torch.no_grad():
            self.head[0].weight.fill_(0.4)
            self.head[0].bias.fill_(-0.2)

    def frames_z(self, frames, metas):
        ids = torch.tensor([float(frame[0, 0, 0]) - 1 for frame in frames], dtype=F64)
        return torch.stack((ids + 1, 0.5 * ids - 1), dim=-1) * self.scale

    def batch_frames_z(self, states):
        return [self.frames_z(*state) for state in states]

    def groups(self):
        return {name: [self.scale, *self.head.parameters()] if name == "phi_head" else []
                for name in ["phi_vision", "phi_vision_no_decay", "phi_llm",
                             "phi_llm_no_decay", "phi_head", "phi_head_no_decay"]}  # fmt: skip


class Policy(nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.read_weight = nn.Parameter(torch.tensor(0.7, dtype=F64))
        self.logit_weight = nn.Parameter(
            torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]], dtype=F64)
        )
        self.logit_bias = nn.Parameter(torch.tensor([0.1, -0.2, 0.05], dtype=F64))
        self.shift_weight = nn.Parameter(
            torch.tensor([[0.3, 0.2], [-0.4, 0.5]], dtype=F64)
        )
        self.action_shift = nn.Parameter(
            torch.tensor([[0.1, 0.0], [0.0, 0.2], [-0.1, 0.1]], dtype=F64)
        )
        self.events = [] if events is None else events
        self.names = ["a0", "a1", "a2"]
        self.shift_calls = 0

    def readout(self, y, c):
        self.events.append(("readout", float(y[0, -1, 0].detach())))
        return y * self.read_weight

    def logits(self, readout):
        return readout @ self.logit_weight.T + self.logit_bias

    def shift(self, readout, actions):
        self.shift_calls += 1
        self.events.append(("shift", int(actions[0])))
        return readout @ self.shift_weight.T + self.action_shift[actions]

    def groups(self):
        return {"pi": list(self.parameters()), "pi_no_decay": []}


class Preference(nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25, -0.35], dtype=F64))
        self.events = [] if events is None else events

    def forward(self, a, b):
        self.events.append(
            ("P", float(a[0, -1, 0].detach()), float(b[0, -1, 0].detach()))
        )
        av, bv = a[:, -1], b[:, -1]
        return (av - bv) @ self.weight, (av + bv) @ self.weight

    def groups(self):
        return {"P": [self.weight], "P_no_decay": []}


class PreferenceLoss(nn.Module):
    def groups(self):
        return {}

    def forward(self, out, labels):
        succ, inc = out
        p = torch.tensor(
            [label["p"] if label["p"] is not None else 0.5 for label in labels],
            dtype=succ.dtype,
        )
        target = torch.tensor([label["inc"] for label in labels], dtype=inc.dtype)
        return (torch.sigmoid(succ) - p).square() + (
            torch.sigmoid(inc) - target
        ).square()


def modules(events=None):
    return {"phi": Phi(), "pi": Policy(events), "P": Preference(events),
            "P_loss": PreferenceLoss(), "metra": MetraLoss(ppo_config()).to(F64)}  # fmt: skip


def ppo_config(horizon=2):
    return PpoConfig(horizon=horizon, clip=0.2, skill_eps=1.0e-8,
                     dual_init=2.0, dual_slack=0.1)  # fmt: skip


def load_step(out_dir, index):
    return SimpleNamespace(
        action=index % 3,
        probabilities=BEHAVIOR.clone(),
        delta=torch.full((2,), torch.nan, dtype=F64),
        predicted=torch.full((2,), torch.nan, dtype=F64),
        preference_close={"succ": torch.nan, "inc": torch.nan},
        preference_start={"succ": torch.nan, "inc": torch.nan},
    )


def install(monkeypatch, events):
    def encoded(ref, phi, perception, sampling, states, action_count):
        events.append(("encode", ref.index))
        return history(ref.index, phi)

    monkeypatch.setattr(
        "mcpref.training.rl.learning.encoding.encode_refs",
        lambda refs, *args: [encoded(ref, *args) for ref in refs],
    )
    monkeypatch.setattr("mcpref.training.rl.learning.loss.load_step", load_step)


def item_losses(item, mods, discount=0.8, min_p=0.1, horizon=2):
    """Direct transcription of ppo.md (1)-(6), preference loss, and raw fake wiring."""
    kind = item["kind"]
    phi, pi, preference = mods["phi"], mods["pi"], mods["P"]
    if kind == "preference":
        ya = base(item["a"]["index"]) * phi.scale
        yb = base(item["b"]["index"]) * phi.scale
        succ = (ya - yb) @ preference.weight
        inc = (ya + yb) @ preference.weight
        p = item["labels"]["p"]
        target = item["labels"]["inc"]
        value = (torch.sigmoid(succ) - p).square() + (
            torch.sigmoid(inc) - target
        ).square()
        return {"P": value}
    t = item["index"]
    z = base(t) * phi.scale
    readout = z * pi.read_weight
    result = {}
    eligible = kind == "pending" or t == 0
    if eligible:
        logits = readout @ pi.logit_weight.T + pi.logit_bias
        action = t % 3
        advantages = torch.zeros((), dtype=F64)
        for k in range(1, horizon + 1):
            future = base(t + k) * phi.scale
            succ = (future - z) @ preference.weight
            inc = (future + z) @ preference.weight
            value = (torch.sigmoid(inc) + min_p) * (2 * torch.sigmoid(succ) - 1)
            advantages = advantages + discount**k * value
        ratio = torch.exp(torch.log_softmax(logits, dim=-1)[action]) / BEHAVIOR[action]
        clipped = torch.clamp(ratio, 0.8, 1.2)
        result["ppo"] = -torch.minimum(
            ratio * advantages.detach(), clipped * advantages.detach()
        )
    if kind == "fresh":
        action = t % 3
        predicted = readout @ pi.shift_weight.T + pi.action_shift[action]
        actual = (base(t + 1) - base(t)) * phi.scale
        result["shift"] = (
            ALPHA.square() * (predicted - actual.detach()).square()
        ).sum()
        norm = torch.linalg.vector_norm(predicted * ALPHA, ord=2, dim=-1)
        skill = (predicted / torch.clamp(norm, min=1.0e-8)).detach()
        weighted = actual * ALPHA
        q = torch.clamp(1 - weighted.square().sum(), max=0.1)
        log_lambda = mods["metra"].log_lambda.to(F64)
        result["metra"] = (
            -(weighted * (skill * ALPHA)).sum()
            - log_lambda.exp().detach() * q
            + log_lambda * q.detach()
        )
    return result


def expected(
    items, mods, m, n_a, n_delta, r_c, weights=(2.0, 3.0, 0.7, 1.4), horizon=2
):
    roles = [item_losses(item, mods, horizon=horizon) for item in items]
    sums = {role: sum((values[role] for values in roles if role in values),
                      torch.zeros((), dtype=F64)) for role in ["ppo", "shift", "metra", "P"]}  # fmt: skip
    w_ppo, w_delta, w_metra, w_pref = weights
    total = torch.zeros((), dtype=F64)
    if n_a:
        total = total + m * w_ppo / n_a * sums["ppo"]
    if n_delta:
        total = total + m / n_delta * (
            w_delta * sums["shift"] + w_metra * sums["metra"]
        )
    if r_c:
        total = total + m * w_pref / r_c * sums["P"]
    means = {role: (sums[role] / sum(role in values for values in roles)).detach().item()
             if any(role in values for values in roles) else None
             for role in sums}  # fmt: skip
    return total, means


def gradients(mods):
    return {f"{name}.{parameter_name}": parameter.grad.clone()
            for name, module in mods.items() for parameter_name, parameter in module.named_parameters()
            if parameter.grad is not None}  # fmt: skip


def zero_grad(mods):
    for module in mods.values():
        module.zero_grad(set_to_none=True)


def assert_gradient_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])
