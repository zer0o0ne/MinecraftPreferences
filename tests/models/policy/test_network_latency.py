"""(12) of specs/policy.md, run by the user on the Spark: median latency with test-owned
policy, perception and action-mask configs; every scenario-material value is explicit."""

import statistics
import time

import pytest
import torch

from mcpref.models.policy.config import PolicyConfig
from mcpref.models.policy.network import Policy
from models.perception.world import config as perception_config
from models.policy.policy_world import action_masks

CFG = PolicyConfig(
    history=12,
    width=256,
    heads=4,
    hidden=768,
    layers=4,
    shift_hidden=512,
    rope_base=10000.0,
    norm_eps=1.0e-6,
    action_dropout=0.3,
    shift_weight=1.0,
    seed=0,
    latency_ms=3.0,
    latency_fail_ms=10.0,
    latency_calls=100,
)
ALPHA = [1.0] * 512


@pytest.mark.gpu
def test_latency(tmp_path):
    perception = perception_config(tmp_path / "perception", alpha=ALPHA)
    names = action_masks(tmp_path / "action-mask").names
    D, A, cuda = len(perception.alpha), len(names), torch.device("cuda")
    net = Policy(CFG, D, names).to(cuda).eval()
    torch.manual_seed(0)
    Y = torch.randn(1, CFG.history, D, device=cuda)
    C = torch.zeros(1, CFG.history, A + 1, device=cuda)
    C[..., A] = 1.0  # the one-hot of none in every row
    every = torch.arange(A, device=cuda)

    def call() -> None:
        r = net.readout(Y, C)[:, -1]
        net.logits(r)
        net.shift(r.expand(A, -1), every)

    times = []
    with torch.no_grad():
        for _ in range(CFG.latency_calls):
            call()
        for _ in range(CFG.latency_calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    ms = 1000 * statistics.median(times)
    print(f"pi latency: median {ms:.3f} ms (target {CFG.latency_ms} ms)")
    assert ms <= CFG.latency_fail_ms, ms
