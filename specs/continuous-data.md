# Continuous pretraining data
Status: approved (2026-11-21)

## Contract

Build one resumable observation dataset for classical and
[continuous pretraining](pretrain-continuous.md), with separate eligibility views
over the same fixed-grid pixels and metadata. Existing clip datasets, votes and
corrected preference labels remain immutable archives; migration writes a new root.
The standard perception collection command uses the same producer and frozen
acquisition inputs as migration, including when rebuilding into an empty root. User decisions of
2026-11-21: fetch missing source recordings; join successive episodes when their
boundary gap is less than one second; discard unsupported/death screens; permit
one shift of an old preference clip by at most three **kept-timeline steps**;
new preference clips must come from the completed dataset.

The initial acquisition scope is the union of complete available continuous
fragments containing any existing ordinary clip or preference endpoint. It is
not limited to the previously sampled episodes or clip spans. Freeze the source
indices, ordered source paths and available boundary metadata before collection.
The 2026-11-21 manifest covers 20,033 pieces in 12,184 fragments, selected from
200,000 ordinary clips and 10,030 preference clips. These are acquisition counts,
not a promise that every source passes complete decoding and data checks.

Within one `(subset, player alias, recorder id)` group, sort pieces by first
`milli`, then source path. Join consecutive pieces iff
`0 <= next.first.milli - previous.last.milli < 1000`, the next piece does not
start a new world (`first.stats` is empty), and `serverTick` does not decrease.
Overlaps, unavailable sources, invalid contents, incompatible required options
and uncheckable boundaries break continuity. Include fragments without an
observed world start. Do not synthesize frames/actions for missing time.
The boundary survey is only a candidate catalogue: collection validates whole
JSONL files and video/frame alignment before publishing observations.

One fragment is one recurrent stream. Memory resets at every fragment boundary,
including two fragments of the same world (user decision of 2026-11-21). Episode
files, storage shards and preference clips introduce no memory boundaries.
Old comparisons between fragments remain usable through the detached preference
archive; that archive is independent of recurrent state.

## Timeline and preference mapping

For a fragment, concatenate source rows in episode order. A row refers to its
source video frame with the same zero-based index, as in perception-data. The
extra terminal video frame without a JSONL row is not a state. Verify all rows,
source tick ordering and the expected decoded frame count. Preserve original
`(source path, row index)` provenance throughout.

Use the existing GUI classifier, screen catalogue, action mapping and options
checks. Remove the `other`, `pause` and `death` spans according to
[VPT GUI actions](vpt-gui-actions.md). Other existing state exclusions (debug
overlay, hidden HUD, non-first-person view, and unusable GUI state metadata) are
checked on scheduled grid observations: omit a failing observation without
moving the grid or renumbering its scheduled indices. They do not change kappa;
keep their positions and reasons in the audit. Exclusions do not reset recurrent
state. Action/evidence
rules still inspect excluded raw spans where their existing contract requires
it, for example riding or a forbidden input inside a cut.

Let `kappa[0..T-1]` exclude exactly the GUI cut lines, as in the existing clips
contract. The scheduled observation grid
is `kappa[j * delta]` for every integer `j >= 0` with `j * delta < T`, starting
at the first kept row of the fragment. `delta = 6` is inherited from
`configs/datasets/clips.yaml`. The grid phase continues across episode and shard
boundaries. Surviving scheduled indices j and j+2 never become an adjacent
training interval after j+1 is omitted. Storage limits never truncate a stream
logically. Every retained observation
is fed to phi, even if its adjacent action interval is idle or unsuitable for a
training loss. Metadata, GUI masks and action targets are recomputed at the new
positions; they are never copied from a shifted old clip. A subsecond join keeps
recurrent continuity but an action interval crossing that join is loss-valid
only when source-row completeness is verified and the boundary `serverTick`
increment is exactly one. Otherwise mask own transition/target losses across
the unobserved span; do not infer the missing action count from elapsed time.

For each old ordinary or preference clip, recover the kept ordinals `p_i` of **all** its
stored states, not just its endpoint or the suffix currently consumed by P.
It is transferable iff they lie in one fragment, form `p_i = p_0 + i * delta`,
and there is an integer shift `s` with `abs(s) <= 3` for which every
`p_i + s` is an existing grid observation in that fragment. Choose the candidate
with minimum `(abs(s), s)`: prefer the earlier grid on a tie, or the equally near
later grid if any earlier candidate observation is absent or excluded. One common shift is
applied to the whole clip; independent per-frame rounding is forbidden.

Example: kept ordinals `(6x+1, 6x+7, 6x+13, 6x+19)` become
`(6x, 6x+6, 6x+12, 6x+18)` with `s=-1`. Physical/raw displacement may exceed
three ticks around deleted screens; the limit is explicitly on the kept scale.
There are no extra off-grid observations for old preference anchors.

Keep the original endpoint ID, full original state references, shift, full new
observation-ID history and mapping status. Unmappable clips and affected pairs
remain in the original archive and are listed with reasons; they do not silently
receive another history. Reasons distinguish absent source, changed/excluded
state, nonuniform kept history and insufficient fragment extent.

Several old endpoint IDs may alias one mapped history. Preserve each original
comparison's ID, orientation and corrected label, with no re-voting, correction,
averaging or deduplication. Exclude from the continuous training view a comparison
whose two complete mapped histories coincide, with reason `collapsed_pair`;
one detached copy of an identical history must not create artificial perception
supervision. Distinct original comparisons between the same two distinct mapped
histories remain distinct examples.

New preference selection reads only the immutable observation IDs of a complete
continuous dataset. A candidate is a consecutive grid window within one fragment
with no omitted scheduled indices and satisfies the existing clip validity
rules. Its configured history length
belongs to preference selection, not to stream storage or memory. It may cross
an episode/shard boundary, but never a fragment boundary. Annotation worlds use
the recording-session identity reconstructed by preference-pool rule (S), not
the larger conservative split group below. An orphan with no established session
is its own fragment-local annotation world. Freeze these IDs in the dataset.
Future preference selection cannot fetch another episode, choose an off-grid
start or manufacture a different window. Votes/labels fingerprint the dataset
and candidate catalogue. New selection uses the existing subset/session/episode draw order and graph
rules, restricted to the completed dataset catalogue. It draws non-overlapping
configured-length suffixes of stored valid histories, with the existing
per-episode quota. Canonical identical histories are candidates only once.
No new votes are collected during migration.

Train/validation separation applies to all observations and labels together.
Use the full `(subset, alias, recorder id)` group as a conservative world group;
all its fragments share one split. Union groups connected by a retained
preference edge before splitting, so neither endpoint crosses the split.
Reuse train-index's seeded component-split rule on these groups, not its old
episode groups. Both views use this shared split, assigned before classical
filtering. Rebuild the index; do not inherit old per-episode
assignments. No training latent archive is reused during validation.
This is an operational separation of recording groups, not proof that physical
Minecraft saves never recur under another recorder ID.

## Classical eligibility view

A mapped history belongs to classical only when every adjacent interval meets
its existing full-clip validity rules, including nonempty legal actions and no
unknown/foreign span. A preference pair belongs to classical only if both mapped
histories pass. Continuous retains its otherwise valid preferences and masks its
own losses per interval. This filtering was explicitly selected by the user.

Rebuild the classical standalone population as the unique union of valid mapped
ordinary histories and endpoints of retained classical pairs. A valid orphan
endpoint of a rejected pair does not enter this union unless independently an
ordinary history or endpoint of a retained pair. Keep repeated labelled edges;
recompute counts and loss denominators for each view. Report classical exclusions
separately from continuous exclusions. Pixel payloads are never duplicated for
these views. Readers adapt reference histories to the existing classical Clip
interface and apply the configured history tail and its own coordinate origin.

## Interfaces and artifacts

- Producer: `python -m mcpref.datasets.continuous.run <config>`; a thin
  `scripts/collect_perception_pretrain_data.sh` invokes it. The config lives at
  `configs/datasets/continuous.yaml`, references the shared perception/clip
  configs, the frozen acquisition plan, existing clip roots and corrected label
  root, and sets the new output directory, bounded shard/cache size and disk
  reserve. Numeric values live in configs; shared access values remain shared.
- A source manifest records ordered pieces, fragment IDs, URLs, expected sizes
  and ETags, source/run/config/classifier fingerprints and per-source disposition.
  A changed ETag is an error requiring an explicit new manifest, not a silent
  replacement. Transient download failures remain retryable, never permanent
  missing-data decisions.
- `shards/<id>/` stores ordered `observations.jsonl`, `intervals.jsonl`, a
  losslessly compressed YUV420p frame payload and `state.json`. Observations
  include stable ID, fragment and world-group IDs, grid and kept ordinals, source
  row reference, model metadata and mask inputs. Intervals include endpoint IDs,
  recognized actions, action counts, validity/mask reasons and source span.
  Pending intervals at a shard boundary are completed exactly once by its
  continuation; no artificial terminal interval is created.
- The frame payload must decode to exactly the YUV420p bytes produced by the
  existing source decoder, in observation order. Frame count, dimensions and a
  hash of decoded bytes are verified before shard publication. FFV1 in Matroska
  is the initial lossless storage candidate; its final parameters and capacity
  estimate are recorded with the collection run after the real-data check.
- Source JSONL/options and exclusion decisions are retained compressed with
  checksums for provenance and label mapping. Whole source videos are bounded
  staging files and may be removed only after their validated output is durably
  published. Existing classical artifacts are not deleted to free space.
- `preference-map/` contains original-to-new endpoint mappings and a separate
  retained-edge view with unchanged labels. `mapping-report.json` counts shifts,
  aliases, rejected endpoints and affected/collapsed edges, with their IDs.
- `fragments/<id>/` contains source dispositions, stream summaries, original-to-new
  mappings and lightweight clip references. `classical-view/` and
  `classical-report.json` describe the filtered classical population. The standard
  train-index command accepts this root and preserves its shared split.
- `scripts/collect_preferences_pretrain_data.sh` selects references from the
  completed dataset, then runs the existing voting/correction stages into new
  `*_shared` roots. It never fetches source videos. Additional corrected labels are attached by
  `configs/training/pretrain/shared-labeled-index.yaml`; attachment checks the
  immutable dataset version and pool fingerprint, preserves the shared split and
  adds each new canonical endpoint to the classical own-loss union once.
- Collection checkpoints retain the source cursor and validated whole-fragment
  inputs needed to reconstruct the kept grid and pending action/GUI/mode state.
  An interrupted fragment is replayed against its already published shard hashes;
  completed fragments are checksum-verified and skipped.
  Readers load one shard at a time. Atomic publication and fingerprints follow
  [shards](shards.md) and [runs](runs.md). Restart ignores incomplete staging,
  reuses verified sources/shards and reproduces uninterrupted output.
- `run.json` is complete only after every planned source has a terminal,
  audited disposition, all grid observations/intervals and preference mappings
  are published, and counts/hashes reconcile. Exhausted retries or low disk
  space stop resumably with `complete=false`; they do not shrink the dataset.

## Acceptance

- A real two-episode join retains one grid phase, held actions and GUI/mode
  context. Processing the same input with different storage shard sizes gives
  identical observations, targets and preference mappings.
- Boundary gaps 999 ms and 1000 ms respectively join and split; world resets,
  overlaps and missing/corrupt pieces never create fictitious transitions.
- Raw GUI/death cuts preserve memory and use kept-scale spacing. Shift cases
  `-3..3`, an exact tie, cuts inside a clip, endpoint aliases and fragment-end
  failures match an independent hand-written oracle. Every retained new state
  is an actual dataset observation; every original edge is accounted for.
- Classical filtering removes invalid endpoints and affected pairs, rebuilds
  the own-loss union, retains repeated valid edges and shares the continuous split.
- New preference collection cannot produce an observation or clip window absent
  from the chosen completed dataset version.
- Real source decode and lossless round-trip checks pass. Interrupting downloads,
  shard publication and mapping resumes without duplicate/lost observations or
  changed labels. A changed source/config/cut fingerprint rejects resume.
- The final report states fetched bytes, observed fragments/hours, exclusions,
  mapped/retained preference counts and actual storage size. Planned acquisition
  counts are never reported as a completed continuous dataset.
