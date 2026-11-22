"""A scalar recurrence and closed-form replay oracle, independent of the encoder."""

from types import SimpleNamespace

import numpy as np
import torch

from mcpref.models.perception.continuous.state import ContinuousState, GDNState
from mcpref.runtime.env.state import EnvState
from mcpref.training.rl.config import ContinuousConfig
from mcpref.training.rl.continuous import replay


class Phi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False).double())
        with torch.no_grad():
            self.head[0].weight.fill_(1.25)
        self.inputs = torch.nn.Parameter(torch.arange(1, 65, dtype=torch.float64))
        self.calls = []

    @property
    def theta(self):
        return self.head[0].weight[0, 0]


def state(value, origin, *, grad=False):
    h = torch.tensor([[[[value]]]], dtype=torch.float64, requires_grad=grad)
    conv = torch.tensor([[[value]]], dtype=torch.float64, requires_grad=grad)
    return ContinuousState((GDNState(conv, h),), dict(origin))


def observed_step(phi, frame, line, previous):
    index = line["index"]
    assert int(frame[0, 0, 0]) == index
    h = (
        phi.inputs.new_zeros(())
        if previous is None
        else previous.layers[0].recurrent.reshape(())
    )
    origin = dict(line) if previous is None else dict(previous.origin)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
    ):
        value = phi.theta * h + phi.inputs[index]
    phi.calls.append(
        SimpleNamespace(
            index=index,
            training=phi.training,
            grad=torch.is_grad_enabled(),
            saved=len(saved),
            incoming_grad=h.requires_grad,
            origin=origin,
        )
    )
    updated = ContinuousState(
        (GDNState(value.reshape(1, 1, 1).clone(), value.reshape(1, 1, 1, 1)),),
        origin,
    )
    return value.reshape(1), updated


class World:
    def __init__(self, monkeypatch, tmp_path, *, k=2, stride=4, length=3):
        self.phi, self.length = Phi(), length
        self.cfg = ContinuousConfig(gradient_history=k, replay_stride=stride)
        self.sampling = SimpleNamespace(out_dir="trajectory", steps=12)
        self.root, self.anchor_reads, self.observation_reads = tmp_path, [], []
        self.origin = {"index": 0, "world": "one"}
        self.anchors = {
            a: state(self.sampled(a), self.origin, grad=True)
            for a in range(stride, 64, stride)
        }
        self.originals = {
            a: (
                s.layers[0].conv.detach().clone(),
                s.layers[0].recurrent.detach().clone(),
            )
            for a, s in self.anchors.items()
        }
        monkeypatch.setattr(replay, "REPO", tmp_path)
        monkeypatch.setattr(replay, "read_anchor", self.read_anchor)
        monkeypatch.setattr(replay, "read_observation", self.read_observation)
        monkeypatch.setattr(replay, "observation_step", observed_step)

    @staticmethod
    def sampled(a):
        """The old sampler used theta=1/2, unlike the current model's theta=5/4."""
        return sum(0.5 ** (a - 1 - j) * (j + 1) for j in range(a))

    def read_anchor(self, out_dir, index, steps, phi, device, stride):
        assert (out_dir, steps, phi, device, stride) == (
            self.root / self.sampling.out_dir,
            12,
            self.phi,
            torch.device("cpu"),
            self.cfg.replay_stride,
        )
        self.anchor_reads.append(index)
        return None if index == 0 else self.anchors[index]

    def read_observation(self, out_dir, index):
        assert out_dir == self.root / self.sampling.out_dir
        self.observation_reads.append(index)
        observation = EnvState(
            np.full((1, 1, 3), index, dtype=np.uint8),
            {"index": index, "world": "one"},
            False,
            None,
            0,
            np.ones(3, dtype=bool),
        )
        return observation, None if index == 0 else (index - 1) % 3

    def branch(self, t, **kwargs):
        return replay.branch(
            self.phi, self.sampling, self.length, 3, self.cfg, t, **kwargs
        )

    def value(self, anchor, index):
        theta = float(self.phi.theta.detach())
        return theta ** (index - anchor + 1) * self.sampled(anchor) + sum(
            theta ** (index - j) * (j + 1) for j in range(anchor, index + 1)
        )

    def expected_y(self, anchor, endpoint):
        return torch.tensor(
            [
                [
                    [self.value(anchor, j)]
                    for j in range(max(0, endpoint - self.length + 1), endpoint + 1)
                ]
            ],
            dtype=torch.float64,
        )

    def expected_c(self, endpoint):
        indices = list(range(max(0, endpoint - self.length + 1), endpoint + 1))
        rows = torch.zeros(1, len(indices), 4, dtype=torch.float64)
        for j, index in enumerate(indices):
            rows[0, j, 3 if index == 0 else (index - 1) % 3] = 1
        return rows

    def gradients(self, anchor, first, weighted_endpoints):
        theta = float(self.phi.theta.detach())
        before = (
            self.value(anchor, first - 1) if first > anchor else self.sampled(anchor)
        )
        dtheta, dx = 0.0, torch.zeros_like(self.phi.inputs)
        for endpoint, weight in weighted_endpoints:
            for j in range(max(0, endpoint - self.length + 1), endpoint + 1):
                if j < first:
                    continue
                dtheta += weight * (
                    (j - first + 1) * theta ** (j - first) * before
                    + sum(
                        (j - i) * theta ** (j - i - 1) * (i + 1)
                        for i in range(first, j)
                    )
                )
                for i in range(first, j + 1):
                    dx[i] += weight * theta ** (j - i)
        return dtheta, dx

    def unchanged(self):
        for a, saved in self.anchors.items():
            conv, recurrent = self.originals[a]
            assert torch.equal(saved.layers[0].conv, conv)
            assert torch.equal(saved.layers[0].recurrent, recurrent)
            assert saved.origin == self.origin
            assert saved.layers[0].conv.grad is None
            assert saved.layers[0].recurrent.grad is None
