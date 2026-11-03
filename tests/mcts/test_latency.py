"""GPU decision latency of specs/mcts.md Behaviour; run by the user on the Spark."""

import statistics
import time

import numpy as np
import pytest
import torch
import yaml

from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks, state_mask
from mcpref.mcts.config import load_config
from mcpref.mcts.search import decide
from mcpref.mcts.tree import LatentHistory, Tree
from mcpref.policy.config import load_config as load_policy_config
from mcpref.policy.network import Policy
from mcpref.preferences.config import load_config as load_preference_config
from mcpref.preferences.network import Preference


@pytest.mark.gpu
def test_decision_latency(repo):
    cfg = load_config(repo / "configs/mcts.yaml")
    alpha_values = yaml.safe_load((repo / "configs/perception.yaml").read_text())[
        "alpha"
    ]
    masks = load_masks(load_mask_config(repo / "configs/action-mask.yaml"))
    cuda = torch.device("cuda")
    alpha = torch.tensor(alpha_values, device=cuda, dtype=torch.float32)
    D = len(alpha_values)
    pi = (
        Policy(load_policy_config(repo / "configs/policy.yaml"), D, masks.names)
        .to(cuda)
        .eval()
    )
    P = (
        Preference(load_preference_config(repo / "configs/preference.yaml"), D)
        .to(cuda)
        .eval()
    )
    torch.manual_seed(0)
    Y = torch.randn(1, 12, D, device=cuda, dtype=torch.float32)
    C = torch.zeros(1, 12, len(masks.names) + 1, device=cuda, dtype=torch.float32)
    C[..., -1] = 1.0
    real = LatentHistory(Y=Y, C=C)
    mask = state_mask(masks, gui=False, cls=None, pitch=0.0, hotbar=0)

    def fresh_decision():
        tree = Tree(real)
        rng = np.random.default_rng(0)
        decide(tree, mask, pi, P, alpha, cfg, rng)

    times = []
    with torch.no_grad():
        fresh_decision()
        for _ in range(cfg.latency_calls):
            tree = Tree(real)
            rng = np.random.default_rng(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            decide(tree, mask, pi, P, alpha, cfg, rng)
            torch.cuda.synchronize()
            stop = time.perf_counter()
            times.append(stop - start)
    ms = 1000 * statistics.median(times)
    print(f"MCTS latency: median {ms:.3f} ms (target {cfg.latency_ms} ms)")
    assert ms <= cfg.latency_fail_ms, ms
