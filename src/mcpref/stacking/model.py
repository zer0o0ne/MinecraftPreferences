"""(M), (1)-(3'), (5) of specs/gui-stack.md: the moments over the shards of S (bar
`moments`), the standardised features, the one-hidden-layer MLP, its class-weighted
losses, the class probabilities and the predicted label."""

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def first_pass(xs: Iterable[np.ndarray]) -> tuple[np.ndarray, ...]:
    """(M) pass 1: per column the count, sum, minimum and maximum of the non-NaN values,
    within a shard by fmin/fmax.reduce, across shards by fmin/fmax."""
    count, total, low, high = 0, 0.0, np.nan, np.nan
    for x in xs:
        count = count + (~np.isnan(x)).sum(axis=0)
        total = total + np.nansum(x, axis=0)
        low = np.fmin(low, np.fmin.reduce(x, axis=0))
        high = np.fmax(high, np.fmax.reduce(x, axis=0))
    return count, total, low, high


def second_pass(xs: Iterable[np.ndarray], mu: np.ndarray) -> np.ndarray:
    """(M) pass 2: per column the sum of squared deviations from mu, NaN left out."""
    squares = 0.0
    for x in xs:
        squares = squares + np.nansum((x - mu) ** 2, axis=0)
    return squares


def moments(
    passes: Callable[[], Iterable[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """(M): mu, s over the x arrays of the shards of S (passes() iterates them afresh);
    mu = 0 and s = 0 for a column without a value; s = 0 when the minimum equals the
    maximum, else the population std."""
    count, total, low, high = first_pass(passes())
    mu = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    squares = second_pass(passes(), mu)
    flat = (count == 0) | (low == high)
    s = np.divide(squares, count, out=np.zeros_like(squares), where=~flat)
    return mu, np.sqrt(s)


def xs(
    read: Callable[[int], np.ndarray], q: int, update: Callable[[], object]
) -> Iterator[np.ndarray]:
    """(M): the x of the q shards of S, read(r) one at a time; update() after each."""
    for r in range(q):
        yield read(r)
        update()


def standardisation(
    read: Callable[[int], np.ndarray], q: int
) -> tuple[np.ndarray, np.ndarray]:
    """(M): mu, s by two passes over the q shards of S, read(r) giving the x of shard r;
    bar `moments`."""
    with tqdm(total=2 * q, desc="moments", unit="shard") as bar:
        return moments(partial(xs, read, q, bar.update))


def standardised(x: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Notation: z = (x - mu) / s where x is not NaN and s > 0, else 0."""
    keep = ~np.isnan(x) & (s > 0)
    return np.divide(x - mu, s, out=np.zeros_like(x), where=keep)


def z_tensor(
    x: np.ndarray, mu: np.ndarray, s: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Notation, library mapping numpy -> torch: z as a float32 tensor on `device`."""
    z = torch.from_numpy(standardised(x, mu, s))
    return z.to(device=device, dtype=torch.float32)


def uniform(shape: tuple[int, ...], fan_in: int, gen: torch.Generator) -> torch.Tensor:
    """(2): (2u - 1) / sqrt(fan_in), u = torch.rand of the shape (float32)."""
    u = torch.rand(shape, generator=gen, dtype=torch.float32)
    return (2 * u - 1) / math.sqrt(fan_in)


def init_params(
    c: int, h: int, n_labels: int, seed: int, device: torch.device
) -> list[torch.Tensor]:
    """(2): [W1, b1, W2, b2] drawn in this order on the CPU, then moved to `device`."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shapes = [((h, c), c), ((h,), c), ((n_labels, h), h), ((n_labels,), h)]
    drawn = [uniform(shape, fan_in, gen) for shape, fan_in in shapes]
    return [p.to(device).requires_grad_() for p in drawn]


def logits(params: Sequence[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    """(1): l = W2 max(0, W1 z + b1) + b2, frames x |Y|."""
    w1, b1, w2, b2 = params
    return torch.relu(z @ w1.T + b1) @ w2.T + b2


def weighted_nll(
    params: Sequence[torch.Tensor], z: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """(3), (3'): sum of w_f (-log p(y_f | z_f)) over the frames, log_softmax."""
    log_p = F.log_softmax(logits(params, z), dim=1)
    nll = -log_p.gather(1, y.unsqueeze(1)).squeeze(1)
    return (w * nll).sum()


def penalty(params: Sequence[torch.Tensor], l2: float) -> torch.Tensor:
    """(3), (3'): l2 (||W1||_F^2 + ||W2||_F^2)."""
    w1, _, w2, _ = params
    return l2 * (w1.square().sum() + w2.square().sum())


def loss(
    params: Sequence[torch.Tensor],
    z: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    w_total: float,
    m: int,
    l2: float,
) -> torch.Tensor:
    """(3'): M sum_{f in B} w_f (-log p(y_f | z_f)) / sum_{f in S} w_f + the L2 term."""
    return m * weighted_nll(params, z, y, w) / w_total + penalty(params, l2)


def probabilities(params: Sequence[torch.Tensor], z: torch.Tensor) -> np.ndarray:
    """(5): p_f, the softmax of the logits computed in float64, frames x |Y|."""
    ell = logits(params, z).detach().to(torch.float64)
    return torch.softmax(ell, dim=1).cpu().numpy()


def predicted(p: np.ndarray, labels: Sequence[str]) -> list[str]:
    """(5): y_hat_f, the first label of Y with the largest p_f."""
    return [labels[a] for a in np.argmax(p, axis=1).tolist()]
