"""Naive, independent float64 transcription of specs/mcts.md (1)–(6)."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RefNode:
    z: torch.Tensor
    x: torch.Tensor
    action: int | None = None
    prior: float | None = None
    shift: torch.Tensor | None = None
    parent: object | None = None
    children: list = field(default_factory=list)
    n: int = 0
    mean: float = 0.0
    value: float | None = None


class RefTree:
    def __init__(self, real):
        z = real.Y[0, -1].clone()
        self.root = RefNode(z, z.clone(), value=0.0)
        self.base = real


def ref_history(tree, node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    Y, C = tree.base.Y, tree.base.C
    for child in reversed(path):
        action = torch.zeros(1, 1, C.shape[-1], dtype=C.dtype)
        action[0, 0, child.action] = 1.0
        Y = torch.cat((Y, child.z.reshape(1, 1, -1)), dim=1)
        C = torch.cat((C, action), dim=1)
    return Y, C


def ref_quantize(delta, alpha, epsilon):
    result = torch.empty_like(delta)
    for i in range(delta.numel()):
        scaled = float(alpha[i] * delta[i]) / epsilon
        result[i] = epsilon * round(scaled) / alpha[i]
    return result


def ref_value(preference, tree, node, min_p):
    if node.parent is None:
        return 0.0
    if node.value is None:
        Y, _ = ref_history(tree, node)
        parent_Y, _ = ref_history(tree, node.parent)
        s, t = preference(Y, parent_Y)
        p_succ = 1.0 / (1.0 + math.exp(-float(s[0])))
        p_inc = 1.0 / (1.0 + math.exp(-float(t[0])))
        node.value = (p_inc + min_p) * (2.0 * p_succ - 1.0)
    return node.value


def ref_expand(policy, tree, node, mask, alpha, cfg):
    Y, C = ref_history(tree, node)
    R = policy.readout(Y, C)[:, -1]
    logits = [float(v) for v in policy.logits(R)[0]]
    highest = max(logits)
    exp = [math.exp(v - highest) for v in logits]
    total = sum(exp)
    probabilities = [v / total for v in exp]
    allowed = range(len(logits)) if node.parent is not None else np.flatnonzero(mask)
    ordered = sorted(allowed, key=lambda a: (-probabilities[a], int(a)))
    actions = [int(a) for a in ordered[: min(cfg.width, len(ordered))]]
    shifts = policy.shift(R.expand(len(actions), -1), torch.tensor(actions))
    for row, action in enumerate(actions):
        q = ref_quantize(shifts[row], alpha, cfg.epsilon)
        child = RefNode(
            z=node.z + q,
            x=node.x + shifts[row],
            action=action,
            prior=probabilities[action],
            shift=q,
            parent=node,
        )
        node.children.append(child)


def ref_depth(node):
    result = 0
    while node.parent is not None:
        result += 1
        node = node.parent
    return result


def ref_path(node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    return list(reversed(path))


def ref_eligible(node, mask):
    if node.parent is not None:
        return list(node.children)
    return [child for child in node.children if mask[child.action]]


def ref_decide(tree, mask, policy, preference, alpha, cfg, rng):
    for _ in range(cfg.simulations):
        leaf = tree.root
        eligible = ref_eligible(leaf, mask)
        while eligible:
            total = sum(child.n for child in eligible)
            scores = [
                child.mean + cfg.c_puct * child.prior * math.sqrt(total) / (1 + child.n)
                for child in eligible
            ]
            leaf = eligible[scores.index(max(scores))]
            eligible = ref_eligible(leaf, mask)
        value = ref_value(preference, tree, leaf, cfg.min_p)
        if ref_depth(leaf) < cfg.depth:
            ref_expand(policy, tree, leaf, mask, alpha, cfg)
        path = ref_path(leaf)
        for j, child in enumerate(path):
            weight = cfg.discount ** (len(path) - j) * value
            child.n += 1
            child.mean += (weight - child.mean) / child.n
    eligible = ref_eligible(tree.root, mask)
    weights = np.array([child.n ** (1.0 / cfg.temperature) for child in eligible])
    if np.all(weights == 0.0):
        weights = np.ones(len(eligible))
    probabilities = weights / weights.sum()
    choice = rng.choice(len(eligible), p=probabilities, replace=True, size=None)
    full = np.zeros(len(mask), dtype=np.float64)
    for child, probability in zip(eligible, probabilities, strict=True):
        full[child.action] = probability
    return eligible[choice].action, full
