---
name: stage-refactor
description: Procedure of the refactorer agent — reduce a prototype to the minimal code expressing its spec, without changing behavior or the spec's Interface.
---
# Refactor

Input: spec path, prototype report.
Load skill `exactness-checklist`: its items are invariants a refactor must not break.

Over the block's code and every `src/` module it touches:
1. Trace each parameter, option, branch, class, config field and default to the spec; delete what does not trace.
2. Find duplicated logic within the block and across `src/`; keep one function.
3. Split functions and files over the AGENTS.md soft limits along the math: one function per equation or term.
4. Keep an abstraction only if it removes more code than it adds or names a spec concept.
5. Enforce the remaining AGENTS.md Code rules: names, docstrings, comments.
6. `uv run ruff format`, `uv run ruff check --fix`; the block's own tests (both tiers) and `uv run pytest` (fast tier) must pass.

Never fix math: report a suspected spec deviation instead.
Report: what was removed, merged or split and why; line count before → after; suspected deviations.
