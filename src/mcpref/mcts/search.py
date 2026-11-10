"""(2)-(6) of specs/mcts.md: PUCT simulation and the visit-count decision."""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from mcpref.mcts.config import MctsConfig
from mcpref.mcts.tree import LatentHistory, Node, Tree, model_history, quantize
from mcpref.policy.network import Policy
from mcpref.preferences.network import Preference


@dataclass(frozen=True)
class Decision:
    """(6): the sampled action and its probability vector over all actions."""

    action: int
    probabilities: np.ndarray


def leaf_value(
    tree: Tree,
    node: Node,
    P: Preference,
    min_p: float,
    H: LatentHistory | None,
) -> float:
    """(2): the cached preference value against the parent; the root is worth zero."""
    if node is tree.root:
        return 0.0
    if node.value is None:
        assert H is not None
        s, t = P(H.Y, H.Y[:, :-1])
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
    H: LatentHistory,
) -> None:
    """(3): expand the highest-probability allowed actions in probability/index order."""
    R = pi.readout(H.Y, H.C)
    probabilities = torch.softmax(pi.logits(R[:, -1]), dim=-1, dtype=None)[0].tolist()
    node.policy = probabilities
    if node is tree.root:
        tree.root_policy = probabilities
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


def eligible(tree: Tree, node: Node, mask: np.ndarray) -> list[Node]:
    """Notation: existing root children allowed by the current mask, or all below it."""
    if node is not tree.root:
        return node.children
    return [child for child in node.children if mask[child.action]]


def select(tree: Tree, mask: np.ndarray, c_puct: float) -> list[Node]:
    """(4): the root-to-leaf edges, with PUCT over eligible children."""
    node, path = tree.root, []
    while children := eligible(tree, node, mask):
        total = sum(child.n for child in children)
        scores = [
            child.mean + c_puct * child.prior * total**0.5 / (1 + child.n)
            for child in children
        ]
        node = children[max(range(len(scores)), key=scores.__getitem__)]
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
    path = select(tree, mask, cfg.c_puct)
    leaf = path[-1] if path else tree.root
    expanding = len(path) < cfg.depth
    H = None
    if expanding or (leaf is not tree.root and leaf.value is None):
        H = model_history(tree, leaf, pi.history, P.history)
    value = leaf_value(tree, leaf, P, cfg.min_p, H)
    if expanding:
        assert H is not None
        expand(tree, leaf, mask, pi, alpha, cfg, H)
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
    deadline: float,
) -> Decision:
    """(1)-(6): bounded complete simulations, then a draw from root visit counts."""
    with torch.no_grad():
        for completed in range(cfg.simulations):
            if completed >= cfg.min_n_sampling and time.perf_counter() >= deadline:
                break
            simulate(tree, mask, pi, P, alpha, cfg)
    children = eligible(tree, tree.root, mask)
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
