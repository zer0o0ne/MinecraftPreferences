---
name: clarify-spec
description: Orchestrator procedure turning the user's description of a block (text and paper links) into an approved, unambiguous specs/<block>.md. Stage 1 of run-block, before any code.
---
# Clarify a block spec

Goal: every equation and interface item admits exactly one implementation.

1. Read `specs/00-overview.md`, the specs of blocks this one consumes from or feeds, and their implemented interfaces in `src/`.
2. Every linked paper: skill `read-paper`. Every library that would compute part of the math: skill `vet-dependency`. Every rule derived from the data (a grouping, a parse of real files, a threshold): validated by an agent on a sample of the real data before it is recommended to the user.
3. Draft `specs/<block>.md` with `Status: draft` (skill `spec-format`).
4. List open points: disagreements between the user's text, the paper and its code; every `exactness-checklist` question the draft does not answer; every unfixed interface detail (arguments, results, artifacts, callers, callees); every config field without a type and domain; every loop over work items left without a bar whose run time on real inputs was not measured; every dataset built or read without its shard layout and rerun behaviour.
5. Run agent `spec-auditor` on the draft; add its findings to the open points.
6. Ask the user the open points, numbered, each with its concrete alternatives written as formulas or signatures. You may recommend; the user decides. Never close a point yourself.
7. Update the draft; repeat 4–6 until no open point remains.
8. Show the final spec; set `Status: approved (YYYY-MM-DD)` only on explicit approval. A new subpackage or artifact goes into the package map of `specs/00-overview.md` in the same approval.
