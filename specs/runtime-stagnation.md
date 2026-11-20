# Runtime stagnation guard
Status: approved (2026-11-20): actual mining progress and limit 3.

## Contract

Prevent an uninterrupted sequence of the same action with no observed result
from continuing indefinitely. Apply to both pretrained and RL gameplay, using
the real environment rather than learned latents or image differences.

Add `stagnation_limit` to `configs/runtime/env.yaml`: a required positive integer.
Initial value: **3**. With `delta: 6`, the first blocked decision
follows four unsuccessful executions, or 24 ordinary game ticks (1.2 seconds).
The first unsuccessful execution counts as one; the threshold is strictly `>`.

For each executed macro action, track whether any ordinary or automatic-close
tick produces an observed change. Changes count even if reversed before the
macro action ends. An observed change is any of:

- A change in raw player `x`, `y`, `z`, `pitch`, or yaw modulo 360. Compare exact
  values, so small actual movement is not discarded.
- A change in inventory contents **by slot**, after normalizing empty slots;
  compare item type and quantity. A slot transfer counts even when totals agree.
- A change in selected hotbar slot or equipped item type, damage, or maxDamage.
- A change in physical GUI-open state, or the classified returned GUI screen.
- An increase in the engine's actual block-breaking progress. A target change,
  arm swing, or an attempted hit without advancing destruction is not enough.
  Observe every tick so completion/reset within a macro action is not lost.
- An increase in finite item-use progress: `min(elapsed / charge_duration, 1)`
  while active, or zero for a zero-duration phase. The engine supplies the
  actual consumption duration for food/drink, full draw time for bows, full
  charge time (including Quick Charge enchantment) for crossbows, and the
  throw threshold for tridents. Shield holding and elapsed time after full
  charge do not independently count as progress. This prevents the short
  watchdog from interrupting food before consumption.
- An increase in `mine_block`, `use_item`, `craft_item`, `pickup`, `break_item`,
  or `kill_entity` statistics; or in custom `jump`, `damage_dealt`,
  `damage_dealt_absorbed`, or `damage_dealt_resisted`. Read the actual bare-key
  MineStudio representation. Absent statistic keys mean zero.

Time counters, input presses, arm animations, lighting, RGB differences, and
latent changes do not independently count as progress. Detection adds no model
inference, extra simulator ticks, or new reward.

## Counter and mask

Let `last` and `count` describe the preceding run of unsuccessful executions.
For a successfully requested, unmasked action `a`:

- If an observed change occurred, set `count = 0`.
- Otherwise, set `count = count + 1` when `a == last`, or `count = 1` otherwise.
- Set `last = a`.

Before the next decision, compute the ordinary action mask. When `count` exceeds
`stagnation_limit`, additionally mask `last` if another action is allowed. If
`last` is already forbidden, the ordinary mask suffices. Never produce an empty
mask. Executing a different action ends the old streak and releases its ban;
there is no accumulated blacklist or detection of multi-action cycles.

A caller requesting an already masked action still gets the existing idle
behavior. It does not count as a new execution of that action. Observed changes
during those idle ticks can clear the previous unsuccessful streak.

The effective mask is returned in `EnvState.mask`, used by MCTS root expansion,
selection and final draw, and preserved in trajectory observations. Keep masked
subtrees for possible later reuse. Store the actual post-mask behavior
probabilities. Existing PPO full-softmax mathematics remains unchanged.

## Interfaces and artifacts

Extend `EnvConfig` with `stagnation_limit`. The environment owns the counter and
its observation baselines. Persist these in `EnvContinuation` and sampling
`state.pt`, preserving them across trajectories, RL cycles and weight changes.
Restoration does not reset the counter, issue commands or advance the world.
A new world starts with no previous action and zero count. Old continuation
artifacts lacking the new guard state fail clearly; completed legacy step
records remain readable for investigation and video export.

## Engine observation

Every engine tick exposes `action_progress` with `version: 1`:
- `mining`: `active`, `target: [x,y,z]`, `progress` (actual client block damage).
- `item_use`: `active`, `action`, `duration`, `charge_duration`, `elapsed`,
  `remaining`, read from the player and current item. Inactive values are zero,
  with action `NONE` and mining target `[0,0,0]`.
  `remaining` is signed: the client can report negative remaining ticks while
  awaiting the server's completion flag. Elapsed time past completion stays
  capped at full progress and does not reset the watchdog.

The pinned engine's `PlayerController.onPlayerDamageBlock` accumulates
`curBlockDamageMP` using actual relative block hardness; `currentBlock` and
`isHittingBlock` identify it. This precedes the ten-stage rendered crack
quantization. `LivingEntity.getItemInUseMaxCount` returns elapsed use ticks;
`getItemInUseCount` returns remaining ticks. Charge times are the actual engine
rules: food/drink `ItemStack.getUseDuration`, bow 20 (the plateau of
`BowItem.getArrowVelocity`), trident 10 (`TridentItem.onPlayerStoppedUsing`),
crossbow `CrossbowItem.getChargeTime` (including enchantments); other use
phases have zero charge duration. Export only, with no gameplay modification.

The setup step patches the engine observation serializer and adds a small
Java observer. Source-class hashes and method signatures must match the pinned
engine; installation is atomic and idempotent. Compilation uses Java 8 and
ASM already bundled in that engine. No installed MineStudio Python files are
changed. `EndlessHumanSurvival` registers the extra observation after
`use_engine`, keeping the existing simulator initialization order.

## Telemetry and limitations

Expose real breaking progress from the pinned engine through MineStudio. Fail
clearly if required telemetry is unavailable; never treat missing telemetry as
zero progress. Useful mining must remain allowed even when the block takes much
longer than four macro actions to break. Full GUI container/cursor/widget state
remains outside the observable interface. Exact position jitter or unrelated
listed events can delay blocking; this is an observed-progress watchdog, not a
proof that the chosen action caused a useful result.

## Acceptance

- At exactly N unchanged executions the action remains allowed; after N+1 it
  is blocked for the next choice. An alternative action releases the old ban.
- Successful mining, damage, use, inventory transfer and movement reset the
  counter, including changes that occur and reverse within one macro action.
- Timer increments and changing frames/latents alone do not reset the counter.
- Eating longer than four actions remains allowed. Fully charged bows and
  shield holding do not keep resetting the counter merely as time passes.
- Real engine: long mining advances without a ban; attacking air and an
  unbreakable block without other changes is blocked after four executions.
- A reused width-one MCTS root with its only child blocked selects another
  allowed action, with zero probability assigned to the blocked action.
- Split-trajectory and restored runs match uninterrupted counters and masks;
  missing new continuation fields fail without hidden ticks or reset.
- The last allowed action is never removed. Existing masked-request idle
  behavior and PPO probability conventions are preserved.
