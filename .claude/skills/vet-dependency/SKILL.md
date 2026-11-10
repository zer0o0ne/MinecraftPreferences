---
name: vet-dependency
description: Verify a new or math-affecting library dependency without auditing routine standard operations.
---
# Vet a dependency

Use this for a new dependency, a library call that computes part of the math, or an API whose relevant behaviour is uncertain.

1. Pin new Python dependencies with `uv add <pkg>==<version>` and check that the release is not yanked.
2. Read the source path relevant to the arguments and inputs used. Cover overridden defaults and hooks when they affect results; do not audit unrelated execution paths.
3. Pass every math-relevant argument explicitly and record the required values in the contract when they affect externally observable results.
4. If the library computes something other than the required behaviour, report the mismatch instead of silently patching around it.
