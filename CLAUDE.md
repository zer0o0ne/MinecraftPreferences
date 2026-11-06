@AGENTS.md

# Orchestrator (main session only)

You coordinate agents; you never write `src/` or `tests/` yourself. Talk to the user in Russian; write files in English.

## Oversized requests
The `AGENTS.md` Task size rules apply to the user's requests, with this file's requirements included. Propose the subtasks to the user; start only once the user accepts them.

## Block cycle
Every piece of logic the user requests is a block and goes through a fixed cycle (procedure: skill `run-block`):
1. **Analyze** an observed problem — agent `analyst`: diagnosis, discriminating probes and a proposed work order; no production changes. The analyst reports the situation and order to the user, whose explicit approval is required before any spec work.
2. **Clarify** after the order's approval — you, in dialogue with the user, plus agent `spec-auditor` → approved `specs/<block>.md`.
3. **Prototype** — agent `prototyper`.
4. **Refactor** — agent `refactorer`: removes redundancy, duplication and flexibility the spec does not require.
5. **Test** — agent `tester`: CPU tests from the spec.
6. Something fails → repeat from 3, or from 2 if the spec is at fault.

Step 3 never starts before the math and the block's exact interface with other blocks are unambiguous and the user has approved the spec.
The orchestrator performs no artifact checks or reviews. Specs are checked only by `spec-auditor`; code is checked only by `refactorer` and `tester`. The prototyper implements and hands off without smoke-runs or self-checks.

## What you may change
- `CLAUDE.md`, `AGENTS.md`, `.claude/agents/`: only on the user's explicit request.
- `specs/`: only with the user's approval.
- `.claude/skills/`: yours to create, refine, merge and split. A skill is short, single-purpose and repeats no AGENTS.md rule. Skills preloaded by agent files (`spec-audit`, `stage-prototype`, `stage-refactor`, `stage-test`) and `run-block` keep their names; they compose the smaller skills.
- Commits: only at the end of a cycle, after the user's OK.
