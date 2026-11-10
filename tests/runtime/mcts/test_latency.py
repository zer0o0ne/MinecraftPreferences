"""GPU decision latency of specs/mcts.md Behaviour; run by the user on the Spark."""

import statistics
import time

import numpy as np
import pytest
import torch
from models.perception.world import STARTING as PERCEPTION_STARTING
from runtime_configs import write_repository
from training.pretrain.data.index_run import repo_root

from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks, state_mask
from mcpref.models.perception.config import load_config as load_perception_config
from mcpref.models.policy.config import load_config as load_policy_config
from mcpref.models.policy.network import Policy
from mcpref.models.preference.config import load_config as load_preference_config
from mcpref.models.preference.network import Preference
from mcpref.runtime.mcts.config import load_config
from mcpref.runtime.mcts.search import decide
from mcpref.runtime.mcts.tree import LatentHistory, Tree


@pytest.mark.gpu
def test_decision_latency(tmp_path):
    root = tmp_path / "repo"
    write_repository(root, states=12, alpha=PERCEPTION_STARTING["alpha"])
    cfg = load_config(root / "configs/runtime/mcts.yaml")
    with repo_root(root):
        alpha_values = load_perception_config(
            root / "configs/models/perception.yaml"
        ).alpha
        masks = load_masks(load_mask_config(root / "configs/gui/action-mask.yaml"))
    cuda = torch.device("cuda")
    alpha = torch.tensor(alpha_values, device=cuda, dtype=torch.float32)
    D = len(alpha_values)
    policy = load_policy_config(root / "configs/models/policy.yaml")
    pi = Policy(policy, D, masks.names).to(cuda).eval()
    P = (
        Preference(load_preference_config(root / "configs/models/preference.yaml"), D)
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
