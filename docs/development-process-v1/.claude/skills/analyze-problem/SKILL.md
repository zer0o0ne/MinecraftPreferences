---
name: analyze-problem
description: Procedure of the analyst agent — diagnose an observed problem, discriminate causes with small checks, and specify useful metrics for the block cycle.
---
# Analyze a problem

Input: the observed symptoms, available evidence, constraints and an assigned scratchpad directory.

1. Read the relevant approved specs, their implementations and the existing artifacts. State the expected behaviour before interpreting measurements.
2. Reproduce the symptom from existing artifacts where possible. Decompose aggregate values and compare with meaningful baselines; do not treat a falling loss as evidence of useful learning.
3. Build an explicit hypothesis set covering data, targets, model outputs, gradients, optimization and logging as applicable. Mark each hypothesis confirmed, refuted or open, citing the check behind the verdict.
4. Use small, targeted probes on real data to discriminate the leading hypotheses. Respect the run constraints; never substitute a long training run for diagnosis. Keep scratch files only in the assigned directory.
5. Propose metrics that distinguish the remaining hypotheses and expose each model head. Give their exact definitions, population and reduction, and identify misleading reductions or baselines to avoid.
6. Prepare the proposed work order and the prototyper's technical handoff: prioritized findings, evidence, proposed spec changes, diagnostic experiments and open decisions. Separate established facts from hypotheses. Do not implement production changes and do not approve or change a spec.
7. Through the orchestrator, report the situation and proposed work order to the user before any spec drafting or prototype call. Stop until the user explicitly approves that order; the later approved-spec gate before implementation remains separate.
8. After writing a draft, hand it directly to the independent auditor. Do not perform or report self-checks or self-audits; verification of the author's draft is not the author's stage.

Report: symptom reproduction; hypothesis table; probe commands and numerical results; metric proposals; recommended work order; unresolved questions for the user.
