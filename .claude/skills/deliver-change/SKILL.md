---
name: deliver-change
description: Own a change from reproduction or contract through implementation and verification, without mandatory agent stages.
---
# Deliver a change

1. State the observable acceptance criterion. For a bug, reproduce it with the smallest useful real scenario or existing artifacts.
2. Determine whether the approved contract changes. If it does not, proceed directly. If it does, get the user's approval of a concise spec before implementing the new decisions.
3. Implement the smallest complete change. Add or update tests in the same loop; use `exactness-checklist` and `test-oracles` only where applicable.
4. Run Ruff on touched files and relevant tests. Diagnose failures and fix them before requesting review.
5. Exercise the original acceptance scenario with real calls. For a long-running pipeline, reproduce the affected transition using real saved artifacts instead of starting the whole run.
6. Run the full fast suite and relevant slow/GPU tests within their budgets. Do not treat unrelated pre-existing failures as permission to ignore failures covering this change.
7. Update only affected documentation. Report what changed, the commands and scenarios that passed, and any remaining limitation.

Do not create a plan document, work-order approval, spec audit, separate prototype stage or mandatory refactor stage unless the user asks. Do not add a new process rule after every mistake; revise existing rules only for a recurring systemic problem.
