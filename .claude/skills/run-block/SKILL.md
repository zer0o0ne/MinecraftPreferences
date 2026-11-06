---
name: run-block
description: Orchestrator procedure for one block through analysis when needed, then the fixed cycle clarify → prototype → refactor → test. Use whenever the user requests logic.
---
# Run a block

Agents start without this conversation. A brief gives exactly the stage's input below, reports verbatim; it never restates the spec's math. The orchestrator only coordinates and performs no artifact checks or reviews.

1. **Analyze** an observed problem — agent `analyst`. Input: symptoms, evidence and run constraints. Output: diagnosis, discriminating probes and a proposed work order. The analyst reports the situation and order to the user; no spec drafting or prototype call starts before the user explicitly approves the order. Skip only for a request with no observed problem to diagnose.
2. **Clarify** — skill `clarify-spec` after the order's approval, until the user approves `specs/<block>.md`. Nothing below starts before that separate spec approval.
3. **Prototype** — agent `prototyper`. Input: spec path, the analyst's technical handoff; on a repeat, the report that caused it. The prototyper implements without smoke-runs or self-checks.
4. **Refactor** — agent `refactorer`. Input: spec path, prototype report. The refactorer reviews spec ↔ code and runs the code checks assigned by its skill.
5. **Test** — agent `tester`. Input: spec path; on a repeat, what changed since its last run.
6. **Route** each problem any stage reports; returning to a stage reruns every stage after it, except that a prototype change limited to `pyproject.toml` / `uv.lock` skips the refactor stage.
   | Problem | Return to |
   |---|---|
   | spec ambiguous, contradictory or incomplete | 2 |
   | code deviates from the spec | 3 |
   | redundancy or style only | 4 |
   | test contradicts the spec | 5, with the contradiction |
   | agent declined its task as too large | same stage, one brief per proposed subtask; 2 if the split changes the spec |
7. **Report** to the user in Russian: what was implemented, files, test results, open points. Before committing (after the user's OK), delegate both full-suite tiers to the tester: `uv run pytest` and `uv run pytest -m slow`.
8. **Retro**: every agent mistake a rule would have prevented becomes a rule in the smallest skill that owns it; merge or split skills when that makes them shorter.
