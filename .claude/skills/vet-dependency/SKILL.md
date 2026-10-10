---
name: vet-dependency
description: Verify what a library call actually computes at a pinned version by reading its source, and fix every math-relevant argument. Use before a spec maps an equation to a library call, and when implementing one.
---
# Vet a library call

1. Pin the version: not yanked (PyPI JSON `yanked` is false) and released at least two weeks ago. Locate the source before the package is in the project: `uv run --no-project --with <pkg>==<ver> python -c "import inspect, <mod>; print(inspect.getsourcefile(<mod>.<obj>))"`; after: the same with `uv run python`.
2. Read the whole path executed with our arguments: the function, config defaults, base classes, overridden hooks, callbacks. For trainers (e.g. GRPO): sampling, reward scaling, advantages, KL estimator, clipping, loss aggregation, iterations per batch, gradient accumulation.
3. List every argument or default that affects the math, with the value the spec requires.
4. Spec library-mapping row: Eq. | call | version | explicit args | `file:line` read.
5. The library computes something other than the spec → open point for the user; never patch around it silently.
