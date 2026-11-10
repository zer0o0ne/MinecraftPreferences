---
name: spec-format
description: Concise format for a new or changed block contract. Use only when the contract itself needs specification.
---
# Spec format

Write only what is needed to make externally observable behaviour and mathematical results unambiguous.

```markdown
# <Block>
Status: draft | approved (YYYY-MM-DD)

## Contract
Purpose, behaviour and invariants. Cite user decisions and papers where relevant.

## Math
Equations, index ranges, reductions, masks, stop-gradients and sampling that affect results. Omit for non-mathematical blocks.

## Interfaces and artifacts
Public callables, config fields, consumed and produced formats, checkpoint/resume behaviour and external events. Do not prescribe routine internal control flow.

## Acceptance
Concrete scenarios and properties that demonstrate the contract, including important boundaries and integration paths.
```

Keep implementation notes elsewhere unless they are part of the contract. Do not include approval transcripts, exhaustive library source maps, internal file decomposition or process checklists.
