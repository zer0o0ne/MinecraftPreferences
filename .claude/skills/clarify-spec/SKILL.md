---
name: clarify-spec
description: Orchestrator procedure turning the approved work order into an approved, unambiguous specs/<block>.md. Stage 2 of run-block, before any code.
---
# Clarify a block spec

Goal: every equation and interface item admits exactly one implementation.

1. Read `specs/00-overview.md`, the specs of blocks this one consumes from or feeds, and their implemented interfaces in `src/`.
2. Every linked paper: skill `read-paper`. Every library that would compute part of the math: skill `vet-dependency`. Every rule derived from the data (a grouping, a parse of real files, a threshold): validated by an agent on a sample of the real data before it is recommended to the user.
3. Draft `specs/<block>.md` with `Status: draft` (skill `spec-format`).
4. Run agent `spec-auditor` once on the draft. Its findings and any disagreements between the user's text, the paper and its code are the open points. The author performs no self-check or self-audit.
5. Ask the user the open points, numbered, each with its concrete alternatives written as formulas or signatures. You may recommend; the user decides. Never close a point yourself.
6. Update the draft with the approved resolutions and show the final spec. Do not run a second audit or an author self-check.
7. Set `Status: approved (YYYY-MM-DD)` only on explicit approval. A new subpackage or artifact goes into the package map of `specs/00-overview.md` in the same approval.
