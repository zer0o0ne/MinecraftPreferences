"""(12) of specs/policy.md, run by the user on the Spark: the median latency of one call of
$\\pi$ (configs/policy.yaml, $D$ = `len(alpha)` of configs/perception.yaml, $A$ of
configs/action-mask.yaml): `readout` of $(Y, C)$ with $m = 1$, $\\ell = L_\\pi$, `logits` of
its last row and one batched `shift` of that row for every action; eval mode, float32,
`torch.no_grad`; printed; target `latency_ms`, failing only above `latency_fail_ms`."""

import statistics
import time

import pytest
import torch
import yaml

from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.policy.config import load_config
from mcpref.policy.network import Policy


@pytest.mark.gpu
def test_latency(repo):
    cfg = load_config(repo / "configs/policy.yaml")
    D = len(yaml.safe_load((repo / "configs/perception.yaml").read_text())["alpha"])
    names = load_masks(load_mask_config(repo / "configs/action-mask.yaml")).names
    A, cuda = len(names), torch.device("cuda")
    net = Policy(cfg, D, names).to(cuda).eval()
    torch.manual_seed(0)
    Y = torch.randn(1, cfg.history, D, device=cuda)
    C = torch.zeros(1, cfg.history, A + 1, device=cuda)
    C[..., A] = 1.0  # the one-hot of none in every row
    every = torch.arange(A, device=cuda)

    def call() -> None:
        r = net.readout(Y, C)[:, -1]
        net.logits(r)
        net.shift(r.expand(A, -1), every)

    times = []
    with torch.no_grad():
        for _ in range(cfg.latency_calls):
            call()
        for _ in range(cfg.latency_calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    ms = 1000 * statistics.median(times)
    print(f"pi latency: median {ms:.3f} ms (target {cfg.latency_ms} ms)")
    assert ms <= cfg.latency_fail_ms, ms
