---
name: stage-prototype
description: Procedure of the prototyper agent — implement an approved spec exactly, reusing existing code.
---
# Prototype

Input: approved spec path; on a repeat, the report that caused it.
Load skills `exactness-checklist` and `vet-dependency`.

1. Read the spec and every existing module its Interface names.
2. Implement the Interface signatures exactly, as a composition of functions following the equations.
3. Library calls follow the spec's library mapping; add dependencies with `uv add <pkg>==<version>`.
4. Smoke-run each public function on tiny CPU inputs from a scratchpad script, not from `tests/`.
5. On a repeat: fix the code until the block's own tests (both tiers) and `uv run pytest` (fast tier) pass; report a test that contradicts the spec.

Report: files changed; Eq. / interface item → function table; library calls with their explicit arguments; spec gaps.
