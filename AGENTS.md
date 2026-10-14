# MinecraftPreferences

Minecraft agent training pipeline (MineStudio, VPT dataset). Stages and data flow: `specs/00-overview.md`.

## Task size
- Before starting a task, judge whether you can complete all of it at full quality, meeting every requirement of the task and of this file.
- If not, decline it explicitly; never do it partially or at reduced quality. Propose instead subtasks that together are roughly equivalent to it, each of which you can complete at full quality, and name where they differ from the original.
- If this becomes clear only mid-task, stop: report what is done and propose subtasks for the rest.
- Judge harshly: splitting a task you could have completed whole is better than taking one you then fail to complete at full quality.

## Source of truth
- A block is defined by `specs/<block>.md` with `Status: approved`: math, interface, library mapping, out-of-scope.
- Code implements exactly the approved spec: every equation and interface item has code; every piece of code traces to the spec.
- Ambiguity, contradiction or a missing detail → stop and report it. Never pick one of several plausible readings, never approximate, never smooth over a corner case.
- Specs change only with the user's approval.

## Exactness
Correct = the expectation of every computed quantity equals the spec.
- Free: dtype, mixed precision, device; mathematically identical rewrites (e.g. `logsumexp`).
- Only if the spec says so: eps, clamps, clipping, normalization, stop-grad, reductions, sampling schemes, math-affecting library defaults.

## Libraries
Allowed (MineStudio, transformers, RL frameworks, igraph, …) if:
- every call that computes part of the math was vetted by reading its source and is listed in the spec's library mapping;
- every math-relevant argument is passed explicitly; versions are pinned.

## Code
Minimal code that expresses the spec.
- No parameter, option, flag, branch, class or config field the spec does not require. No handling of inputs outside the spec's domain.
- No duplicated logic: search `src/` before writing; reuse or extract.
- Complex logic is a composition of short functions, ideally one per equation or term.
- Soft limits (hold in ≥95% of cases): file ≤ 200 lines, function ≤ 30 lines, ≤ 10 modules per directory.
- Names follow the spec's notation and terms.
- Docstrings: one or two lines linking code to math, e.g. `"""Eq. (3) of specs/metra.md: skill reward."""`. Tensor shapes only where the spec fixes them. No comments that restate code.
- Code is device-agnostic: the device comes from the caller.
- Progress: a loop over work items (frames, episodes, files, requests, classes, …; not a wait for a condition) that can run longer than a couple of seconds on real data shows a `tqdm` progress bar on stderr, with its total when known. Nested loops get one bar, on the outermost loop, whose total sums the work items of the inner loops (items of different kinds count as steps); a parallel loop gets one bar for all its workers, advanced as items finish. A loop with nothing to do draws no bar and prints one line on stderr instead: `<desc>: already done` when an earlier run did the work, else `<desc>: nothing to do`. Bars replace per-item progress lines, write nothing to stdout or output files and change no result; an error message is the last line of stderr, below the bars. A spec names each bar of its block (skill `spec-format`).

## Layout
```
specs/                                 block specs; 00-overview.md is the pipeline map
src/mcpref/<subpackage>/<module>.py    never deeper
tests/<subpackage>/test_<module>.py    mirrors src/
tests/scripts/test_<script>.py         mirrors scripts/
configs/<block>.yaml                   every exact number of a block; loaded into the block's frozen dataclass
scripts/<action>.sh                    every action the user runs on the Spark; bash, thin wrappers over `python -m mcpref.…` (exception: `setup_env.sh` bootstraps uv and Python in bash and reads `configs/spark-env.yaml` itself)
data/                                  gitignored: datasets, checkpoints, logs, all run outputs
```
Code holds no numeric defaults; secrets come from environment variables, never from configs.
Code shared by several stages lives in a subpackage named after what it is (e.g. `rl/`), never `utils`/`common`/`helpers`.

## Tests
- No GPU on this machine: tests run on CPU. GPU-only tests are marked `@pytest.mark.gpu` and skipped here; the user runs them elsewhere.
- Tests check the spec, not the implementation: oracles never reuse `src/` logic.
- Two tiers. Fast (default `uv run pytest`): each test ≤ 5 s (fails at 10 s), the whole tier ≤ 10 min. Slow (`@pytest.mark.slow`, `uv run pytest -m slow`): statistical checks, long scenarios; each test ≤ 5 min (fails at 10 min), the whole tier ≤ 1 h.
- Tests are independent (no shared files, no order) and run in parallel (`pytest-xdist`).
- A test over its budget is a defect: make it cheaper (smaller samples within the oracle's tolerance, vectorised code, tiny models) or, if it is inherently long, mark it slow; slow block code goes to refactoring.
- Test code follows the Code rules.

## Commands
`uv sync` · `uv run pytest` · `uv run ruff format` · `uv run ruff check --fix`
