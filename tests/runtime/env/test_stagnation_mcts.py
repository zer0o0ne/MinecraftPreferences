"""A width-one decision must escape a newly masked repeated action."""

import numpy as np
import torch

from mcpref.runtime.mcts.search import decide
from mcpref.runtime.mcts.tree import Tree
from runtime.mcts.helpers import FakePolicy, FakePreference, config, latent_history


def test_real_env_guard_reexpands_reused_width_one_root(ready):
    size = len(ready.env.masks.names)
    attack, forward = ready.index("attack"), ready.index("forward")
    logits = torch.full((size,), -10.0)
    logits[attack], logits[forward] = 10.0, 5.0
    policy = FakePolicy(logits=logits, shifts=torch.zeros(size, 1))
    tree = Tree(latent_history([[0.0]], actions=size))
    rng = np.random.default_rng(7)

    def decision(mask):
        return decide(
            tree,
            mask,
            policy,
            FakePreference(),
            torch.ones(1),
            config(width=1, depth=1, min_p=0.0),
            rng,
            float("inf"),
        )

    state = ready.env._state
    result = decision(state.mask)
    assert result.action == attack
    old_child = tree.root.children[0]
    for _ in range(4):
        state = ready.step("attack")
    result = decision(state.mask)
    assert result.action == forward
    assert result.probabilities[attack] == 0
    assert result.probabilities.sum() == 1
    assert tree.root.children[0] is old_child
    assert len({child.action for child in tree.root.children}) == len(
        tree.root.children
    )

    for name in ("forward", "forward", "forward", "forward"):
        state = ready.step(name)
    assert not state.mask[forward] and state.mask[attack]
    result = decision(state.mask)
    assert result.action == attack and result.probabilities[forward] == 0
    assert result.probabilities.sum() == 1
