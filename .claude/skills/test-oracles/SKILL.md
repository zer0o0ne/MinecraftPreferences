---
name: test-oracles
description: How to build test oracles independent of the implementation — hand-computed cases, naive float64 references, gradient structure, Monte-Carlo unbiasedness checks, fakes for external systems. Use when writing tests.
---
# Test oracles

Prefer earlier kinds.

1. **Hand case** — tiny inputs; expected value derived by hand from the equation, derivation in a one-line comment.
2. **Naive reference** — direct transcription of the equation with explicit loops in float64, inside the test module.
3. **Spec properties** — invariants the spec states (bounds, symmetries, identities between equations).
4. **Gradient structure** — which parameters get nonzero gradient from which loss; where stop-grad holds; `torch.autograd.gradcheck` in float64 for custom backward.
5. **Unbiasedness** — mean of a stochastic estimator over N seeded draws vs the exact expectation, enumerated on a tiny discrete case; tolerance 5·σ̂/√N.
6. **Library-backed code** — tiny random-init models (e.g. a transformers config with 2 layers, width 8).
7. **External systems** (Minecraft, MineStudio simulator, network) — never real; a fake implementing exactly the contract in the spec's Interface.
8. **Config files** — PyYAML reads a float without a dot (`1e9`, `1e-8`) as a string; write `1.0e+9`, `1.0e-8` for numbers.
9. **Progress bars** — read with `tests/progress_bars.py`; check only states tqdm always draws (at creation, at close, after `set_postfix_str`), never redraw timing.

Rules: float64 with `torch.testing.assert_close` defaults unless the spec fixes a dtype; fixed seeds; tiers and budgets of AGENTS.md (Monte-Carlo and multi-seed checks are usually `slow`; choose N for the budget, the tolerance follows from N); tests independent, safe under `pytest-xdist`; one behaviour per test, named after the equation or property it checks.
