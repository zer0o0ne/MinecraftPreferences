---
name: stage-refactor
description: Procedure of the refactorer agent — reduce a prototype to the minimal code expressing its spec, without changing behavior or the spec's Interface.
---
# Refactor

Input: spec path, prototype report.
Load skill `exactness-checklist`: its items are invariants a refactor must not break.
Scratch files live in `<scratchpad>/<block>-<stage>/`, never in the scratchpad root: other agents run in parallel.

Over the block's code and every `src/` module it touches:
1. Review spec → code and code → spec: every equation and interface item has code; every function, parameter, option, branch, class, config field and default traces to the spec. Delete what does not trace; report math deviations instead of fixing them.
2. Find duplicated logic within the block and across `src/`; keep one function.
3. Split functions and files over the AGENTS.md soft limits along the math: one function per equation or term.
4. Keep an abstraction only if it removes more code than it adds or names a spec concept.
5. Enforce the remaining AGENTS.md Code rules: names, docstrings, comments.
6. `uv run ruff check --fix`, then `uv run ruff format` (a fix can break the layout); relevant tests for the block and every touched interface (both applicable tiers) must pass. Do not run the whole suite during development. Full-suite runs require the user's explicit command relayed by the orchestrator. When the worktree contains changes outside the block, pass the block-owned paths to both Ruff commands; a global Ruff check must be read-only and must never modify unrelated files.

Never fix math: report a suspected spec deviation instead.
Report: what was removed, merged or split and why; line count before → after; suspected deviations.
