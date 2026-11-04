"""Small contract fakes and constructors for the black-box MCTS tests."""

import torch

from mcpref.mcts.config import MctsConfig
from mcpref.mcts.tree import LatentHistory, Node


def config(**changes) -> MctsConfig:
    values = {
        "simulations": 1,
        "min_n_sampling": 1,
        "depth": 1,
        "width": 4,
        "latency_calls": 1,
        "c_puct": 0.0,
        "discount": 0.9,
        "temperature": 1.0,
        "min_p": 0.25,
        "epsilon": 0.1,
        "latency_ms": 1.0,
        "latency_fail_ms": 2.0,
    }
    return MctsConfig(**(values | changes))


def latent_history(Y, C=None, actions: int = 4) -> LatentHistory:
    Y = torch.as_tensor(Y, dtype=torch.float64)
    if Y.ndim == 2:
        Y = Y.unsqueeze(0)
    rows = Y.shape[1]
    if C is None:
        C = torch.zeros(1, rows, actions + 1, dtype=Y.dtype)
        C[..., actions] = 1.0
    else:
        C = torch.as_tensor(C, dtype=Y.dtype).reshape(1, rows, -1)
    return LatentHistory(Y=Y, C=C)


def tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.clone()
    return torch.tensor(value, dtype=torch.float64)


def assert_values(actual, expected):
    torch.testing.assert_close(actual, expected, check_dtype=False)


def node(
    z,
    *,
    x=None,
    action=None,
    prior=None,
    shift=None,
    parent=None,
    children=None,
    n=0,
    mean=0.0,
    value=None,
) -> Node:
    z = tensor(z)
    return Node(
        z=z,
        x=z.clone() if x is None else tensor(x),
        action=action,
        prior=prior,
        shift=None if shift is None else tensor(shift),
        parent=parent,
        children=[] if children is None else children,
        n=n,
        mean=mean,
        value=value,
    )


class FakePolicy:
    def __init__(self, logits=None, shifts=None, *, logits_fn=None, shift_fn=None):
        self.logits_values = logits
        self.shift_values = shifts
        self.logits_fn = logits_fn
        self.shift_fn = shift_fn
        self.readout_calls = []
        self.logit_calls = []
        self.shift_calls = []
        self.grad_modes = []

    def readout(self, Y, C):
        self.grad_modes.append(torch.is_grad_enabled())
        self.readout_calls.append((Y.clone(), C.clone()))
        return Y[:, -1:, :]

    def logits(self, R):
        self.grad_modes.append(torch.is_grad_enabled())
        self.logit_calls.append(R.clone())
        if self.logits_fn is not None:
            return self.logits_fn(R)
        values = torch.as_tensor(self.logits_values, dtype=R.dtype, device=R.device)
        return values.expand(*R.shape[:-1], values.numel())

    def shift(self, R, a):
        self.grad_modes.append(torch.is_grad_enabled())
        self.shift_calls.append((R.clone(), a.clone()))
        if self.shift_fn is not None:
            return self.shift_fn(R, a)
        return torch.as_tensor(self.shift_values, dtype=R.dtype, device=R.device)[a]


class FakePreference:
    def __init__(self, logits=(0.0, 0.0), *, logits_fn=None):
        self.logits = logits
        self.logits_fn = logits_fn
        self.calls = []
        self.grad_modes = []

    def __call__(self, Ya, Yb):
        self.grad_modes.append(torch.is_grad_enabled())
        self.calls.append((Ya.clone(), Yb.clone()))
        values = self.logits_fn(Ya, Yb) if self.logits_fn is not None else self.logits
        return tuple(torch.as_tensor(v, dtype=Ya.dtype).reshape(1) for v in values)


class UnexpectedCalls:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        raise AssertionError("unexpected network call")

    def __getattr__(self, name):
        return self
