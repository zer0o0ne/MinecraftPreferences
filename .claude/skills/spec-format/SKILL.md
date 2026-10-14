---
name: spec-format
description: Template and rules for a block spec file specs/<block>.md. Use when writing, auditing or reading a spec.
---
# Spec format

Every symbol is defined once in Notation; every equation is numbered and has a source.

```markdown
# <Block>
Status: draft | approved (YYYY-MM-DD)
Stage: <number and name from specs/00-overview.md>

## Sources
- [S1] <paper>, arXiv:<id>; [S1-code] <repo>@<commit>
- [U] the user, <date>

## Notation
- $x$ — meaning — type; shape only if fixed

## Math
(1) $...$ — [S1] Eq. (5)
Index ranges, boundaries, reductions, stop-grad and sampling are written out, never implied.

## Interface
- Module: `src/mcpref/<subpackage>/<module>.py`
- `def f(a: T, ...) -> R` — each argument as a Notation symbol; the result as an equation number.
- Consumes: <artifact path and format | function> from <block spec>
- Produces: <artifact path and format> for <block spec>
- Progress bars (AGENTS.md Code): `<desc>` (unit) — loop; total; advanced when / by; postfix or none; line when there is nothing to do

## Library mapping
| Eq. | Call | Version | Explicit args | Source read |

## Out of scope
What is deliberately absent (no flags or options for it).
```
