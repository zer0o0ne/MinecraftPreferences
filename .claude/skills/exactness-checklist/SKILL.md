---
name: exactness-checklist
description: Checklist of silent deviations between math and code (indices, reductions, masking, stop-grad, sampling, numerics, library defaults, modes, update order). Use to generate spec questions, to implement, to refactor safely and to target tests.
---
# Exactness checklist

For each item the spec gives exactly one answer, and the code matches it.

- **Indices** — inclusive/exclusive ranges; t vs t+1 alignment; first and last step; terminal vs truncated episodes; bootstrapping at the boundary.
- **Reductions** — sum vs mean; over which axes; per token / sequence / batch; masked and padded elements excluded from numerator and denominator.
- **Gradient flow** — which terms are stop-grad; which parameters each loss updates; target networks; parameters shared between losses.
- **Sampling** — which expectations are estimated by samples; how many; with or without replacement; same or fresh samples across terms; source of randomness.
- **Numerics** — eps, clamp, clip, normalization of advantages, rewards or features: present or absent.
- **Library defaults** — the value of every math-relevant argument.
- **Modes** — train/eval (dropout, normalization layers) at each call; frozen networks are in eval mode and get no gradients.
- **Update order** — optimizer steps, target updates, schedules, gradient accumulation and its scaling; what changes per iteration.
- **Data** — filtering criteria, splits, shuffling, per-sample weights.
- **Shapes** — broadcasting that silently aligns the wrong axes; in-place ops under autograd.
