---
name: stage-test
description: Procedure of the tester agent — write CPU tests checking a block against its spec as a black box, run the block's tests and the fast tier within budget, diagnose failures.
---
# Test

Input: spec path; on a repeat, what changed.
Load skills `test-oracles` and `exactness-checklist`.
Scratch files live in `<scratchpad>/<block>-<stage>/`, never in the scratchpad root: other agents run in parallel.

1. The contract is the spec's Math and Interface; import only what the Interface names. Never read `src/`, not even through `cat`, `grep` or `git diff` output. A test the spec is not enough to write is a spec defect: report it.
2. Cover every equation (≥ 1 oracle), every boundary the spec defines, every applicable `exactness-checklist` item (e.g. gradient reaches exactly the parameters the spec says).
3. Extend existing test files and fixtures instead of duplicating them.
4. `uv run ruff check --fix`, then `uv run ruff format` on your test files (a fix can break the layout). Run the block's own tests in both tiers (`uv run pytest tests/<block paths> -m 'not gpu'`), then the whole fast tier (`uv run pytest`). Never the other blocks' slow tests.
5. Check `--durations`: a test over its AGENTS.md budget is your defect; make it cheaper or, if inherently long, mark it `slow` with `@pytest.mark.timeout(600)`.

Report per failure: test, equation, expected vs actual, minimal repro, diagnosis — code deviates from spec | test is wrong | spec is ambiguous. Plus an equation → tests table, and the run times of the block's tests and of the fast tier.
