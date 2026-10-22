"""`latent_metric` of src/mcpref/perception/loss.py (specs/perception.md Interface, revised by
specs/policy.md): the function $\\delta \\mapsto d_\\alpha = \\lVert \\alpha \\odot \\delta
\\rVert_2$ over the last axis, $\\alpha$ of the config in $\\delta$'s dtype; hand values, a
float64 transcription over leading axes, float32, the gradient (0 at 0)."""

import math

import numpy as np
import torch
from torch.testing import assert_close

from mcpref.perception.loss import latent_metric

from .world import F64, config, values


def test_hand_values(tmp_path):
    """alpha = (1, 2, 0.5, 3): (1, 1, 2, 0) -> sqrt(1 + 4 + 1) ; (0, 0, 0, -1) -> 3."""
    metric = latent_metric(config(tmp_path))
    delta = torch.tensor([[1.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, -1.0]], dtype=F64)
    assert_close(metric(delta), torch.tensor([math.sqrt(6.0), 3.0], dtype=F64))


def test_leading_axes_against_transcription(tmp_path):
    """delta of shape (m, L - 2, D), the policy's call: one value per row."""
    metric = latent_metric(config(tmp_path))
    delta = np.random.default_rng(1).normal(size=(2, 3, 4))
    alpha = np.asarray(values(tmp_path)["alpha"])
    want = [[math.sqrt(sum((a * x) ** 2 for a, x in zip(alpha, row))) for row in rows]
            for rows in delta]  # fmt: skip
    got = metric(torch.tensor(delta))
    assert got.shape == (2, 3)
    assert_close(got, torch.tensor(want, dtype=F64))


def test_float32(tmp_path):
    """alpha in delta's dtype: float32 in, float32 out."""
    metric = latent_metric(config(tmp_path))
    delta = torch.tensor([[1.0, 1.0, 2.0, 0.0]])
    got = metric(delta)
    assert got.dtype == torch.float32
    assert_close(got, torch.tensor([math.sqrt(6.0)]))


def test_gradient(tmp_path):
    """d d_alpha / d delta = alpha^2 delta / d_alpha; 0 at delta = 0."""
    metric = latent_metric(config(tmp_path))
    alpha = torch.tensor(values(tmp_path)["alpha"], dtype=F64)
    delta = torch.tensor([[0.5, -1.0, 2.0, 0.25], [0.0, 0.0, 0.0, 0.0]], dtype=F64,
                         requires_grad=True)  # fmt: skip
    metric(delta).sum().backward()
    d = torch.sqrt(((alpha * delta[0]) ** 2).sum())
    assert_close(delta.grad[0], (alpha**2 * delta[0] / d).detach())
    assert torch.equal(delta.grad[1], torch.zeros(4, dtype=F64))
