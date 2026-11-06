---
name: stage-prototype
description: Procedure of the prototyper agent — implement an approved spec exactly, reusing existing code.
---
# Prototype

Input: approved spec path; on a repeat, the report that caused it.
Load skills `exactness-checklist` and `vet-dependency`.
Scratch files live in `<scratchpad>/<block>-<stage>/`, never in the scratchpad root: other agents run in parallel.

1. Read the spec and every existing module its Interface names.
2. Implement the Interface signatures exactly, as a composition of functions following the equations.
3. Library calls follow the spec's library mapping; add dependencies with `uv add <pkg>==<version>`.
4. On a repeat: apply the refactorer's or tester's report; report a test that contradicts the spec. Do not smoke-run, test or otherwise check your own implementation: code checks belong to the refactorer and tester.

Report: files changed; Eq. / interface item → function table; library calls with their explicit arguments; spec gaps. Make no verification claims.
