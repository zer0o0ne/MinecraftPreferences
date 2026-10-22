"""(13) of specs/preference.md, run by the user on the Spark: the median latency of one call
of $P$ (configs/preference.yaml, $D$ = `len(alpha)` of configs/perception.yaml) with $m = 1$,
$\\ell_a = \\ell_b = L_P$, eval mode, float32, `torch.no_grad`; printed; target `latency_ms`,
failing only above `latency_fail_ms`."""

import statistics
import time

import pytest
import torch
import yaml

from mcpref.preferences.config import load_config
from mcpref.preferences.network import Preference


@pytest.mark.gpu
def test_latency(repo):
    cfg = load_config(repo / "configs/preference.yaml")
    D = len(yaml.safe_load((repo / "configs/perception.yaml").read_text())["alpha"])
    net = Preference(cfg, D).to("cuda").eval()
    torch.manual_seed(0)
    Za = torch.randn(1, cfg.history, D, device="cuda")
    Zb = torch.randn(1, cfg.history, D, device="cuda")
    times = []
    with torch.no_grad():
        for _ in range(cfg.latency_calls):
            net(Za, Zb)
        for _ in range(cfg.latency_calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            net(Za, Zb)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    ms = 1000 * statistics.median(times)
    print(f"P latency: median {ms:.3f} ms (target {cfg.latency_ms} ms)")
    assert ms <= cfg.latency_fail_ms, ms
