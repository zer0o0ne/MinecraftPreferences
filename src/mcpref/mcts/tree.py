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


@dataclass(init=False)
class Tree:
    """Interface: the current root and the base history from which paths extend."""

    root: Node
    base: LatentHistory

    def __init__(self, real: LatentHistory) -> None:
        z = real.Y[0, -1]
        self.root = Node(z, z, None, None, None, None, value=0.0)
        self.base = real


def history(tree: Tree, node: Node) -> LatentHistory:
    """(1): the base history followed by the working states and actions to node."""
    path = []
    while node is not tree.root:
        path.append(node)
        assert node.parent is not None
        node = node.parent
    Y, C = tree.base.Y, tree.base.C
    if not path:
        return LatentHistory(Y, C)
    path.reverse()
    Z = torch.stack([child.z for child in path], dim=0)[None]
    actions = torch.tensor([child.action for child in path]).to(
        device=C.device, non_blocking=True
    )
    rows = torch.arange(len(path), device=C.device)
    appended = C.new_zeros((1, len(path), C.shape[-1]))
    appended[0, rows, actions] = 1
    return LatentHistory(torch.cat((Y, Z), dim=1), torch.cat((C, appended), dim=1))


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
        return tree
    root = Node(z, z, action, None, None, None, value=0.0)
    tree.root, tree.base = root, real
    return tree
