---
name: clarify-spec
description: Orchestrator procedure turning the approved work order into an approved, unambiguous specs/<block>.md. Stage 2 of run-block, before any code.
---
# Clarify a block spec

Goal: every equation and interface item admits exactly one implementation.

1. Read `specs/00-overview.md`, the specs of blocks this one consumes from or feeds, and their implemented interfaces in `src/`.
2. Every linked paper: skill `read-paper`. Every library that would compute part of the math: skill `vet-dependency`. Every rule derived from the data (a grouping, a parse of real files, a threshold): validated by an agent on a sample of the real data before it is recommended to the user.
3. Before drafting, trace the whole implementation on the real interfaces: CLI/config loading and every indirect input, producer callers and events, exact nested artifact schemas and types, accepted-source joins, plot paths, identities, streaming memory, every crash window, watermarks, resume and completion. A signature without its caller's source of every argument is incomplete. Derive routine implementation details from the approved architecture and existing code; do not turn config paths, schema plumbing or other mechanical choices into user questions.
4. Draft `specs/<block>.md` with `Status: draft` (skill `spec-format`), including the complete traces above rather than deferring them to implementation.
5. Run agent `spec-auditor` once on the draft. Give it the full implementation handoff and require a single exhaustive implementation-level audit, not only a math/interface review. Its findings and any disagreements between the user's text, the paper and its code are the open points. The author performs no self-check or self-audit.
6. Collect every known finding before asking anything. Ask genuine product, math or architecture choices once, numbered and grouped, each with concrete alternatives written as formulas or signatures. Never drip-feed technical questions or ask the user to choose details that have a determinate implementation from the approved architecture. Never close a genuine ambiguity yourself.
7. Update the draft with all approved resolutions and show the final spec. Do not run a second audit or an author self-check; the single audit in step 5 must already have covered the whole implementation. A prototype or test finding that should have been caught here is a clarification-process defect, not a routine new question.
8. Set `Status: approved (YYYY-MM-DD)` only on explicit approval. A new subpackage or artifact goes into the package map of `specs/00-overview.md` in the same approval.
