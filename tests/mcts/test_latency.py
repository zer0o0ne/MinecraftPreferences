"""GPU decision latency of specs/mcts.md Behaviour; run by the user on the Spark."""

import statistics
import time

import numpy as np
import pytest
import torch
from perception.world import STARTING as PERCEPTION_STARTING
from pretrain.index_run import repo_root
from runtime_configs import write_repository

from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks, state_mask
from mcpref.mcts.config import load_config
from mcpref.mcts.search import decide
from mcpref.mcts.tree import LatentHistory, Tree
from mcpref.perception.config import load_config as load_perception_config
from mcpref.policy.config import load_config as load_policy_config
from mcpref.policy.network import Policy
from mcpref.preferences.config import load_config as load_preference_config
from mcpref.preferences.network import Preference


@pytest.mark.gpu
def test_decision_latency(tmp_path):
    root = tmp_path / "repo"
    write_repository(root, states=12, alpha=PERCEPTION_STARTING["alpha"])
    cfg = load_config(root / "configs/mcts.yaml")
    with repo_root(root):
        alpha_values = load_perception_config(root / "configs/perception.yaml").alpha
        masks = load_masks(load_mask_config(root / "configs/action-mask.yaml"))
    cuda = torch.device("cuda")
    alpha = torch.tensor(alpha_values, device=cuda, dtype=torch.float32)
    D = len(alpha_values)
    policy = load_policy_config(root / "configs/policy.yaml")
    pi = Policy(policy, D, masks.names).to(cuda).eval()
    P = (
        Preference(load_preference_config(root / "configs/preference.yaml"), D)
        .to(cuda)
        .eval()
    )
    torch.manual_seed(0)
    Y = torch.randn(1, policy.history, D, device=cuda, dtype=torch.float32)
    C = torch.zeros(
        1, policy.history, len(masks.names) + 1, device=cuda, dtype=torch.float32
    )
    C[..., -1] = 1.0
    real = LatentHistory(Y=Y, C=C)
    mask = state_mask(masks, gui=False, cls=None, pitch=0.0, hotbar=0)

    def fresh_decision():
        tree = Tree(real)
        rng = np.random.default_rng(0)
        decide(tree, mask, pi, P, alpha, cfg, rng, deadline=float("inf"))

    times = []
    with torch.no_grad():
        fresh_decision()
        for _ in range(cfg.latency_calls):
            tree = Tree(real)
            rng = np.random.default_rng(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            decide(tree, mask, pi, P, alpha, cfg, rng, deadline=float("inf"))
            torch.cuda.synchronize()
            stop = time.perf_counter()
            times.append(stop - start)
    ms = 1000 * statistics.median(times)
    print(f"MCTS latency: median {ms:.3f} ms (target {cfg.latency_ms} ms)")
    assert ms <= cfg.latency_fail_ms, ms
