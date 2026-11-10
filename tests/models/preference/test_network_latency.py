"""(13) of specs/preference.md, run by the user on the Spark: median latency with
test-owned preference and perception configs; every scenario-material value is explicit."""

import statistics
import time

import pytest
import torch

from mcpref.models.preference.config import PreferenceConfig
from mcpref.models.preference.network import Preference
from models.perception.world import config as perception_config

CFG = PreferenceConfig(
    history=12,
    width=256,
    heads=4,
    hidden=768,
    layers=("S", "X", "S", "X"),
    rope_base=10000.0,
    norm_eps=1.0e-6,
    seed=0,
    latency_ms=3.0,
    latency_fail_ms=10.0,
    latency_calls=100,
)
ALPHA = [1.0] * 512


@pytest.mark.gpu
def test_latency(tmp_path):
    perception = perception_config(tmp_path / "perception", alpha=ALPHA)
    D = len(perception.alpha)
    net = Preference(CFG, D).to("cuda").eval()
    torch.manual_seed(0)
    Za = torch.randn(1, CFG.history, D, device="cuda")
    Zb = torch.randn(1, CFG.history, D, device="cuda")
    times = []
    with torch.no_grad():
        for _ in range(CFG.latency_calls):
            net(Za, Zb)
        for _ in range(CFG.latency_calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            net(Za, Zb)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    ms = 1000 * statistics.median(times)
    print(f"P latency: median {ms:.3f} ms (target {CFG.latency_ms} ms)")
    assert ms <= CFG.latency_fail_ms, ms
