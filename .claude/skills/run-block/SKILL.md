---
name: run-block
description: Orchestrator procedure for one block (a piece of logic the user asks to implement or change) through the fixed cycle clarify → prototype → refactor → test. Use whenever the user requests logic.
---
# Run a block

Agents start without this conversation. A brief gives exactly the stage's input below, reports verbatim; it never restates the spec's math.

1. **Clarify** — skill `clarify-spec` until the user approves `specs/<block>.md`. Nothing below starts before that.
2. **Prototype** — agent `prototyper`. Input: spec path; on a repeat, the report that caused it.
3. **Refactor** — agent `refactorer`. Input: spec path, prototype report.
4. **Test** — agent `tester`. Input: spec path; on a repeat, what changed since its last run.
5. **Route** each problem any stage reports; returning to a stage reruns every stage after it, except that a prototype change limited to `pyproject.toml` / `uv.lock` skips the refactor stage.
   | Problem | Return to |
   |---|---|
   | spec ambiguous, contradictory or incomplete | 1 |
   | code deviates from the spec | 2 |
   | redundancy or style only | 3 |
   | test contradicts the spec | 4, with the contradiction |
   | agent declined its task as too large | same stage, one brief per proposed subtask; 1 if the split changes the spec |
6. **Review** once tests pass: spec → code (every equation and interface item has code) and code → spec (every function, parameter and branch traces to the spec); AGENTS.md rules on the diff. Findings route as in 5.
7. **Report** to the user in Russian: what was implemented, files, test results, open points. Before committing (after the user's OK), run both tiers of the whole suite: `uv run pytest` and `uv run pytest -m slow`.
8. **Retro**: every agent mistake a rule would have prevented becomes a rule in the smallest skill that owns it; merge or split skills when that makes them shorter.
