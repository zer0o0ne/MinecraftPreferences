---
name: spec-audit
description: Procedure of the spec-auditor agent — find every place in a draft spec that admits more than one implementation or leaves the block's interface open.
---
# Audit a draft spec

Input: path to a draft spec. Load skills `spec-format` and `exactness-checklist`; spot-check sources with `read-paper` and the library mapping with `vet-dependency`.

A finding is a place where two different implementations both satisfy the text. For each equation and interface item, try to construct such a pair; each success is a finding. Also check:
- every symbol is in Notation; every equation has a source;
- consumed and produced artifacts and functions match the specs of neighbouring blocks and their real implementations;
- the library mapping matches the library source at the pinned version;
- every dataset the block builds or reads, intermediate ones included, is in Interface Datasets; an interrupted run at any point loses at most the shards in work;
- every Interface callable, constructors included, has its full signature, and its arguments carry every Notation symbol its equations use (no value reached through hidden state); tensor arguments state their shape and dtype;
- no hidden choice: "e.g.", "typically", "or similar", "some", "standard".

Trace every workflow end to end before reporting. Start at CLI/config loading and prove that every caller has an explicit source for every argument, including indirect config dependencies. Follow each producer event through pending/accepted artifacts, exact source joins and plot lookup. Check the complete nested schema for mixed types, arbitrary ids versus numeric indices, multi-population alignments and unavailable values. Inspect every function that reads records for whole-dataset materialization. Enumerate crash points before/after each source and diagnostic publication, watermark initialization on an existing run, identity checks, completion and no-backfill resume. Do not dismiss these as implementation details.

Return all findings from this single exhaustive pass, not a first subset and not only issues likely to need user input. For each finding give: location (section / Eq.), the two implementations, a concrete resolution, and whether it is a genuine user decision or a determinate implementation fix. Group genuine user decisions so they can be asked in one batch. No findings → say so explicitly only after every trace above is covered.
