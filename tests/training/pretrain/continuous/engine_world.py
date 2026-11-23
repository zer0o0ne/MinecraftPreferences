"""Small differentiable stream with independently specified observations and heads."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mcpref.models.preference.loss import PreferenceLoss
from mcpref.training.groups import check_groups
from mcpref.training.optim import adamw
from mcpref.training.pretrain.continuous.data.reader import Observation
from mcpref.training.pretrain.continuous.events import Context


class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.4, -0.2]))
        self.cfg, self.sprite = SimpleNamespace(), None

    def groups(self):
        return {"phi": list(self.parameters())}


def observation_step(phi, frame, line, state=None):
    z = phi.weight * line["value"] + (0 if state is None else state * 0.5)
    return z, z


class PhiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(0.1))
        self.cfg = SimpleNamespace(tau=0.1, c=3.0, beta=0.2, dual_slack=0.7)
        self.metric = lambda value: torch.linalg.vector_norm(value, dim=-1)

    def groups(self):
        return {"dual": list(self.parameters())}


class Pi(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.names, self.index, self.history = ("a", "b"), {"a": 0, "b": 1}, 3
        self.project = nn.Linear(5, 2)
        self.output = nn.Linear(2, 2)
        self.dynamics = nn.Linear(2, 2)
        self.change_head = nn.Linear(2, 1)
        self.dropout = nn.Dropout(dropout)

    def readout(self, Y, C):
        return self.dropout(self.project(torch.cat((Y, C), dim=-1)).cumsum(dim=1))

    def logits(self, value):
        return self.output(value)

    def shift(self, value, action):
        return self.dynamics(value) + action[:, None] * 0.1

    def change(self, value):
        return self.change_head(value).squeeze(-1).sigmoid()

    def groups(self):
        return {"pi": list(self.parameters())}


class PiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_weight = 0.3
        self.metric = lambda value: torch.linalg.vector_norm(value, dim=-1)

    def groups(self):
        return {}


class P(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.6, -0.3]))
        self.history = 2

    def forward(self, A, B):
        return (A.mean(1) - B.mean(1)) @ self.weight, (
            A.mean(1) + B.mean(1)
        ) @ self.weight

    def groups(self):
        return {"P": list(self.parameters())}


def modules(dropout=0.0):
    torch.manual_seed(41)
    return {
        "phi": Phi(),
        "phi_loss": PhiLoss(),
        "pi": Pi(dropout),
        "pi_loss": PiLoss(),
        "P": P(),
        "P_loss": PreferenceLoss(),
    }


def config(**changes):
    return SimpleNamespace(
        **(
            {
                "device": "cpu",
                "seed": 7,
                "epochs": 2,
                "tbptt_states": 1,
                "batch_size": 1,
                "preference_micro_batch": 2,
                "archive_shard_size": 1,
                "archive_cache_size": 1,
                "archive_keep_shards": 1,
                "checkpoint_steps": 1,
                "keep_checkpoints": 2,
                "gradient_diagnostics": False,
                "w_phi": 0.4,
                "w_pi": 0.7,
                "w_change": 1.0,
                "w_P": 1.3,
                "groups": {
                    key: {"lr": 0.01, "weight_decay": 0.03}
                    for key in ("phi", "dual", "pi", "P")
                },
                "betas": (0.8, 0.9),
                "adam_eps": 1e-7,
                "grad_clip": 10.0,
                "warmup_steps": 1,
                "lr_floor": 0.2,
            }
            | changes
        )
    )


def observations(prefix, values, *, actions=None, endpoints=None):
    actions = actions or [["a"] for _ in values[:-1]]
    endpoints = endpoints or {}
    intervals = [
        {
            "a": f"{prefix}{i}",
            "b": f"{prefix}{i + 1}",
            "actions": a,
            "legal_actions": a,
            "n": len(a),
            "known": a != ["?"],
            "valid": bool(a) and a != ["?"],
        }
        for i, a in enumerate(actions)
    ]
    intervals += [None]
    return [
        Observation(
            f"{prefix}{i}",
            i,
            i,
            np.full((2, 2, 3), value, dtype=np.uint8),
            {"value": value},
            {"value": values[0]},
            str(value),
            intervals[i],
            np.ones(2, dtype=bool),
            tuple(endpoints.get(i, ())),
            0,
            i,
            prefix,
        )
        for i, value in enumerate(values)
    ]


def endpoint(key, ids, degree):
    return {"id": key, "observation_ids": ids, "pending_edges": degree}


class Index:
    def __init__(self, streams, edges, population):
        self.streams, self.all_edges = streams, edges
        self.run = {"populations": {"train": population}}

    def fragments(self, part):
        return [
            {"id": key, "observations": len(value)}
            for key, value in self.streams.items()
        ]

    def edges(self, key):
        return [e for e in self.all_edges if key in (e["a"], e["b"])]


class Reader:
    def __init__(self, index, *args):
        self.index = index

    def observations(self, fragment, start=0):
        return iter(self.index.streams[fragment["id"]][start:])


def context(root, cfg=None, model=None, index=None, total=10):
    cfg, model = cfg or config(), model or modules()
    groups = check_groups(cfg, model)
    return Context(
        cfg, model, index, root, groups, adamw(cfg, groups), total, "joint", step=1
    )
