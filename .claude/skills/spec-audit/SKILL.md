---
name: spec-audit
description: Procedure of the spec-auditor agent — find every place in a draft spec that admits more than one implementation or leaves the block's interface open.
---
# Audit a draft spec

Input: path to a draft spec. Load skills `spec-format` and `exactness-checklist`; spot-check sources with `read-paper` and the library mapping with `vet-dependency`.

A finding is a place where two different implementations both satisfy the text. For each equation and interface item, try to construct such a pair; each success is a finding. Also check:
- every symbol is in Notation; every equation has a source;
- consumed and produced artifacts and functions match the specs of neighbouring blocks;
- the library mapping matches the library source at the pinned version;
- every dataset the block builds or reads, intermediate ones included, is in Interface Datasets; an interrupted run at any point loses at most the shards in work;
- every Interface callable, constructors included, has its full signature, and its arguments carry every Notation symbol its equations use (no value reached through hidden state); tensor arguments state their shape and dtype;
- no hidden choice: "e.g.", "typically", "or similar", "some", "standard".

Report one item per finding: location (section / Eq.), the two implementations, the question for the user. No findings → say so explicitly.
