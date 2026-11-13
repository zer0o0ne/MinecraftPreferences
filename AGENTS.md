# MinecraftPreferences

Minecraft agent training pipeline (MineStudio, VPT dataset). Stages and data flow: `specs/00-overview.md`.

## Delivery
- The main session owns the working result and may edit `src/`, `tests/`, `configs/` and `scripts/`. Talk to the user in Russian; write files in English.
- Deliver in small working increments. Do not decline or split a task merely because a rigid process makes it large; ask only when priorities or scope genuinely require a user decision.
- Reproduce an observed bug before fixing it. Implement and verify in the same loop; never hand off unrun code as completed work.
- Use subagents only for independent parallel work, investigation that benefits from isolation, or high-risk review. The main session integrates and verifies their work.
- Ask the user only about genuine product, math, architecture or irreversible choices. Decide routine implementation details directly.
- Follow `.claude/skills/deliver-change/SKILL.md` for changes to source, tests, configs and scripts.

## Contracts and specs
- An approved `specs/<block>.md` defines intended external behaviour, math, interfaces and artifacts. Code must match it.
- A bug fix that preserves the approved contract requires no spec change or approval cycle.
- New product, math or architecture decisions, and changes to an approved contract, require a concise spec approved by the user before those decisions are implemented.
- Keep specs proportional: behaviour, formulas, interfaces, artifacts and acceptance criteria. Do not duplicate routine control flow, internal module decomposition or every implementation branch in a spec.
- When a genuine ambiguity affects results, report it and ask one focused question. Do not silently choose between materially different mathematical behaviours.
- Update documentation affected by a change. Do not rewrite unrelated specs or add approval history and process scaffolding.

## Exactness and dependencies
- The expectation of every computed quantity must equal the contract. Pay particular attention to indices, reductions, masks, stop-gradients, sampling, numerical constants, library defaults and update order.
- Mathematically identical rewrites, dtype, mixed precision and device choices are free unless the contract fixes them.
- Pin new dependencies. Read the relevant source of new or math-affecting library calls and pass math-relevant arguments explicitly. Routine I/O and standard container operations do not require exhaustive source audits.

## Code
- Write minimal code for the requested behaviour: no unused parameters, options, flags, branches, classes or config fields. Search for and reuse existing logic.
- Use short functions with clear names. Soft limits: file ≤ 200 lines, function ≤ 30 lines, ≤ 10 modules per directory; do not create pointless abstractions just to satisfy them.
- Docstrings are short and link to a spec equation when useful. Do not restate code in comments.
- Code is device-agnostic: the device comes from the caller.
- Datasets live on disk in shards, written atomically as soon as complete with the state needed to resume. Readers load one shard at a time and stages pass paths, not whole in-memory datasets. Training checkpoints include weights, optimizer, RNG states and data position; reruns continue from the last complete checkpoint.
- Long loops over work items show a `tqdm` bar on stderr with a known total when available. Empty loops print one concise stderr line. Progress output does not change results.

## Layout
```
specs/                                 block contracts; 00-overview.md is the pipeline map
src/mcpref/<domain>/<block>/<module>.py  domain-grouped blocks; large blocks add one grouping level
tests/<domain>/<block>/test_<module>.py  mirrors src/
tests/scripts/test_<script>.py         mirrors scripts/
configs/<domain>/<block>.yaml          mirrors src/; shared values live in higher-level base configs
scripts/<action>.sh                    thin bash wrappers over python -m mcpref.…
data/                                  gitignored datasets, checkpoints, logs and run outputs
```
Code holds no numeric defaults; secrets come from environment variables. Shared code lives in a subpackage named after its purpose, never `utils`/`common`/`helpers`.

## Verification
- The current session runs on a local macOS arm64 machine without a CUDA GPU. GPU-only tests run on the Spark.
- The implementer runs the code. Add a regression test for every reproduced defect; tests check the contract with independent oracles rather than copying production logic.
- Run Ruff on touched files, relevant tests for the change and touched interfaces, then the original failing scenario. For long training, exercise the failing transition with real artifacts rather than rerunning the whole training.
- Run the full fast suite (`uv run pytest`) before completion. Run relevant slow/GPU tests when they cover the change; run the full slow suite on user request.
- Tests are independent, safe under `pytest-xdist`, and use test-owned config values material to the scenario. Fast tests target ≤ 5 s each and the fast tier ≤ 10 min; inherently long tests are marked `slow`.
- Do not mark work complete while tests covering the change are red or the acceptance scenario is unverified. Report unrelated pre-existing failures without fixing them unless asked.

## Commands
`uv sync` · `uv run pytest` · `uv run ruff format` · `uv run ruff check --fix`
