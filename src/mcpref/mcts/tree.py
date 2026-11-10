"""(1), (7) and Q_epsilon of specs/mcts.md: latent histories and re-rooting."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentHistory:
    """Notation: a real or imagined latent/action history."""

    Y: Tensor
    C: Tensor


@dataclass
class Node:
    """Notation: the two coordinates and search statistics of one node."""

    z: Tensor
    x: Tensor
    action: int | None
    prior: float | None
    shift: Tensor | None
    parent: Node | None
    children: list[Node] = field(default_factory=list)
    n: int = 0
    mean: float = 0.0
    value: float | None = None
    policy: list[float] | None = None


@dataclass(init=False)
class Tree:
    """Interface: the current root and the base history from which paths extend."""

    root: Node
    base: LatentHistory

    def __init__(self, real: LatentHistory) -> None:
        z = real.Y[0, -1]
        self.root = Node(z, z, None, None, None, None, value=0.0)
        self.base = real
        self.root_policy = None


def _path(tree: Tree, node: Node) -> list[Node]:
    path = []
    while node is not tree.root:
        path.append(node)
        assert node.parent is not None
        node = node.parent
    return path[::-1]


def _append(Y: Tensor, C: Tensor, path: list[Node]) -> LatentHistory:
    Z = torch.stack([child.z for child in path], dim=0)[None]
    actions = torch.tensor([child.action for child in path]).to(
        device=C.device, non_blocking=True
    )
    rows = torch.arange(len(path), device=C.device)
    appended = C.new_zeros((1, len(path), C.shape[-1]))
    appended[0, rows, actions] = 1
    if not Y.shape[1]:
        return LatentHistory(Z, appended)
    return LatentHistory(torch.cat((Y, Z), dim=1), torch.cat((C, appended), dim=1))


def history(tree: Tree, node: Node) -> LatentHistory:
    """(1): the base history followed by the working states and actions to node."""
    path = _path(tree, node)
    if not path:
        return LatentHistory(tree.base.Y, tree.base.C)
    return _append(tree.base.Y, tree.base.C, path)


def model_history(tree: Tree, node: Node, L_pi: int, L_P: int) -> LatentHistory:
    """(1): only the tail consumed by the policy and preference history windows."""
    path = _path(tree, node)
    n = min(max(L_pi, L_P + 1), tree.base.Y.shape[1] + len(path))
    path_n = min(n, len(path))
    base_n = n - path_n
    base_start = tree.base.Y.shape[1] - base_n
    path = path[-path_n:]
    Y, C = tree.base.Y[:, base_start:], tree.base.C[:, base_start:]
    if not path:
        return LatentHistory(Y, C)
    return _append(Y, C, path)


def quantize(delta: Tensor, alpha: Tensor, epsilon: float) -> Tensor:
    """Q_epsilon: nearest-even rounding in alpha-scaled coordinates."""
    scaled = alpha * delta
    return torch.round(scaled / epsilon, decimals=0) * epsilon / alpha


def matches(node: Node, z: Tensor, alpha: Tensor, epsilon: float) -> bool:
    """Notation: exact equality of the quantised accumulated discrepancy with zero."""
    residual = quantize(node.x - z, alpha, epsilon)
    return torch.equal(residual, torch.zeros_like(residual))


def advance(
    tree: Tree, action: int, real: LatentHistory, alpha: Tensor, epsilon: float
) -> Tree:
    """(7): reuse the first exact-state match, or rebase to the real observation."""
    children = sorted(tree.root.children, key=lambda child: child.action != action)
    z = real.Y[0, -1]
    for child in children:
        if not matches(child, z, alpha, epsilon):
            continue
        base = history(tree, child)
        if child.action != action:
            child.action = action
            C = torch.cat((base.C[:, :-1], real.C[:, -1:]), dim=1)
            base = LatentHistory(base.Y, C)
        child.parent = None
        tree.root, tree.base = child, base
        tree.root_policy = child.policy
        return tree
    root = Node(z, z, action, None, None, None, value=0.0)
    tree.root, tree.base = root, real
    tree.root_policy = None
    return tree
