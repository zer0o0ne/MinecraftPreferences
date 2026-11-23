# Action-change diagnostics

These observations supplement `policy.md`, `pretrain-diagnostics.md` and
`rl-diagnostics.md`. They do not change losses, updates or action selection.

## Pretraining

Use the action logits and change output from the same training readout, including
its existing dropout. Let `h = [change >= 0.5]` and `a = argmax(logits)` over all
actions, with the smallest index breaking ties. The threshold is a diagnostic
decision rule; the auxiliary loss does not calibrate a change probability.

`S` is the set of factual actions in the immediately preceding interval, before
masking or input dropout. A row participates only with a known adjacent
predecessor and a nonempty current target. A known idle predecessor has `S = {}`;
clip/fragment starts, gaps and unknown predecessors are excluded. Current targets
retain multiplicities and the existing current-action mask. Detached target
frequencies `q` are normalized per row in float64 for diagnostic arithmetic.

Define `g = sum(q[b] for b not in S)`, `r = [a in S]`, and
`c = [a not in S and q[a] > 0]`. Thus choosing an unrelated new action is not
successful reproduction of the GT. Mixed intervals contribute fractional change
and repeat mass; they are neither discarded nor assigned a majority label.

| Metric under `policy.change` | Numerator | Denominator |
|---|---|---|
| `repeat_when_predicted` | sum(h r) | sum(h) |
| `head_recall` | sum(g h) | sum(g) |
| `action_recall` | sum(g c) | sum(g) |
| `joint_recall` | sum(g h c) | sum(g) |
| `false_positive` | sum((1-g) h) | sum(1-g) |

Each metric stores `numerator`, `denominator` and `value` (null at denominator
zero). `count` is the number of participating rows. Weighted masses are not
binomial sample counts and receive no Wilson interval. Micro-batches, optimizer
steps and diagnostic windows pool sums before division, never average rates.
Training covers all standalone policy-target rows in classical mode and all CE
events in continuous mode. Existing reference policy populations use the same
definitions when those evaluations are available. Legacy missing observations
remain unavailable; they are not reconstructed.

Pretraining reuses its existing readout. With the change loss disabled, the
continuous path may evaluate the small change head on a detached existing
readout for known predecessors, without autograd or random draws.

## Sampling

At each existing MCTS node expansion, compute and cache `change_probability`
from the exact readout already used for that node's action logits. No additional
policy readout, rollout or random draw is permitted. Serialize the scalar with
each node in step snapshots and continuation state; reuse it with the cached
node policy. This is a cached search prediction, potentially made on imagined
history or older weights, including when rerooting matches through another
action. It is not a fresh prediction on factual history.

`mcts.repeat_when_change_predicted` compares the action actually sampled by MCTS
with the immediately preceding executed action, conditional on the cached root
output being at least 0.5. It stores numerator, denominator (`n`) and rate using
the existing binary diagnostic schema. The first action has no predecessor and
is excluded; trajectory shard boundaries do not reset the predecessor. A legacy
root without the cached head output yields an unavailable metric without a
replacement inference. `mcts.change_prediction` records the head's positive
rate on available rows with a predecessor, providing coverage for the condition.

Reports separate rate curves from their supporting counts/masses. Diagnostics
do not alter search arithmetic or RNG; fixed-simulation decisions remain equal.
The extra scalar head computation can affect wall-clock-budget stopping at a
deadline, as other measurement overhead can.

## Acceptance

Hand-computed cases cover mixed GT, idle/unknown predecessors, current masks,
input dropout, incorrect new actions, threshold ties and empty denominators.
Unequal micro-batch/window populations must produce pooled ratios. Check both
pretrain modes and their persisted step/window outputs. Sampling checks actual
sampled actions, cached-node reuse (including another-action matches), legacy
absence, snapshot/continuation round trips and unchanged fixed-budget decisions
and RNG without additional readouts. Report tests verify rates and masses have
separate axes.
