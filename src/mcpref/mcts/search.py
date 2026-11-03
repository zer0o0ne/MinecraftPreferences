"""(2)-(6) of specs/mcts.md: PUCT simulation and the visit-count decision."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from mcpref.mcts.config import MctsConfig
from mcpref.mcts.tree import Node, Tree, history, quantize
from mcpref.policy.network import Policy
from mcpref.preferences.network import Preference


@dataclass(frozen=True)
class Decision:
    """(6): the sampled action and its probability vector over all actions."""

    action: int
    probabilities: np.ndarray


def leaf_value(tree: Tree, node: Node, P: Preference, min_p: float) -> float:
    """(2): the cached preference value against the parent; the root is worth zero."""
    if node is tree.root:
        return 0.0
    if node.value is None:
        current = history(tree, node)
        s, t = P(current.Y, current.Y[:, :-1])
        value = (torch.sigmoid(t) + min_p) * (2 * torch.sigmoid(s) - 1)
        node.value = value.item()
    return node.value


def expand(
    tree: Tree,
    node: Node,
    mask: np.ndarray,
    pi: Policy,
    alpha: Tensor,
    cfg: MctsConfig,
) -> None:
    """(3): expand the highest-probability allowed actions in probability/index order."""
    H = history(tree, node)
    R = pi.readout(H.Y, H.C)
    probabilities = torch.softmax(pi.logits(R[:, -1]), dim=-1, dtype=None)[0].tolist()
    allowed = np.flatnonzero(mask) if node is tree.root else np.arange(len(mask))
    actions = sorted(allowed, key=lambda a: (-probabilities[a], int(a)))
    actions = actions[: cfg.width]
    indices = torch.tensor(actions, dtype=torch.long).to(
        device=R.device, non_blocking=True
    )
    deltas = pi.shift(R[:, -1].expand(len(actions), -1), indices)
    shifts = quantize(deltas, alpha, cfg.epsilon)
    working, exact = node.z + shifts, node.x + deltas
    for i, action in enumerate(actions):
        child = Node(
            z=working[i],
            x=exact[i],
            action=int(action),
            prior=probabilities[action],
            shift=shifts[i],
            parent=node,
        )
        node.children.append(child)


def select(root: Node, c_puct: float) -> list[Node]:
    """(4): the root-to-leaf edges, with PUCT ties in expansion order."""
    node, path = root, []
    while node.children:
        total = sum(child.n for child in node.children)
        scores = [
            child.mean + c_puct * child.prior * total**0.5 / (1 + child.n)
            for child in node.children
        ]
        node = node.children[max(range(len(scores)), key=scores.__getitem__)]
        path.append(node)
    return path


def simulate(
    tree: Tree,
    mask: np.ndarray,
    pi: Policy,
    P: Preference,
    alpha: Tensor,
    cfg: MctsConfig,
) -> None:
    """(5): select, evaluate and optionally expand one leaf, then back up its value."""
    path = select(tree.root, cfg.c_puct)
    leaf = path[-1] if path else tree.root
    value = leaf_value(tree, leaf, P, cfg.min_p)
    if len(path) < cfg.depth:
        expand(tree, leaf, mask, pi, alpha, cfg)
    for j, node in enumerate(path):
        weight = cfg.discount ** (len(path) - j) * value
        node.n += 1
        node.mean += (weight - node.mean) / node.n


def decide(
    tree: Tree,
    mask: np.ndarray,
    pi: Policy,
    P: Preference,
    alpha: Tensor,
    cfg: MctsConfig,
    rng: np.random.Generator,
) -> Decision:
    """(1)-(6): run the simulations and sample a root child from its visit counts."""
    with torch.no_grad():
        for _ in range(cfg.simulations):
            simulate(tree, mask, pi, P, alpha, cfg)
    children = tree.root.children
    weights = np.array(
        [child.n ** (1 / cfg.temperature) for child in children], dtype=np.float64
    )
    if np.all(weights == 0):
        weights[:] = 1
    probabilities = weights / weights.sum()
    choice = rng.choice(len(children), size=None, replace=True, p=probabilities)
    full = np.zeros(len(mask), dtype=np.float64)
    for child, probability in zip(children, probabilities):
        full[child.action] = probability
    return Decision(children[choice].action, full)
