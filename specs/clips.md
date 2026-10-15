# Clips
Status: approved (2026-10-15; clarified after the prototype the same day: library mapping of the draw, `opens_container`, `gui_flag` in the Interface)
Stage: data of P1, P2 (blocks perception-data, preference-pool; `specs/00-overview.md`, block 22)

## Sources
- [U] the user, 2026-10-15 (Q G-12, G-14, ACT-20, the data-block entries): elementary states $\Delta$ ticks apart; a clip is $L$ consecutive elementary states with every interval valid — $n \ge 1$ legal actions and nothing excluded; clip starts uniform among the valid ones, at most $m$ non-overlapping clips per episode; the spans of unrecognised screens are cut out of the timeline; the grid starts at the clip's first state; a key held through a cut span is one segment; sequential sampling of non-overlapping clips as in (4).
- `specs/vpt-actions.md` (approved, with the revision of 2026-10-15): world segments, `interval_actions`, `mode_frames`, `foreign_lines`, `riding_lines`, (12).
- `specs/vpt-gui-actions.md` (draft): `GuiActions` (GUI segments, cut lines, foreign lines), `excluded_states` (8), (9).
- Audit of 2026-10-15 (scratchpad `audit_clips/`): positions among kept lines vs line numbers checked consistent.

## Notation
- Lines $\ell_0..\ell_{T-1}$ of one episode; the state of line $t$ is frame $t$ with $\ell_t$.
- $C$ — the cut lines of `GuiActions`; $\kappa_0 < \dots < \kappa_{|K|-1}$ — the kept lines $K = \{0..T-1\} \setminus C$.
- Evidence of the episode: $S$ — the world segments, computed on the timeline without the cut lines ([U]: `segments` applied to the lines $\ell_{\kappa_0}, \ell_{\kappa_1}, \dots$, each segment's first and last position mapped back to $\kappa$), and the GUI segments of `gui_actions`, merged by first line (they never share a line); $F$ — the foreign lines: `foreign_lines` ∪ `riding_lines` ∪ `GuiActions.foreign`; $\Omega_t$ — `mode_frames`; `foreign_lines`, `riding_lines` and `mode_frames` are applied to all lines $\ell_0..\ell_{T-1}$; $\rho$ — `turn_deg`.
- $\Delta$ (`delta`) — kept lines per interval; $L \ge 2$ — states per clip; $m \ge 1$ — clips per episode at most (both given by the caller).
- Interval $p$ ($0 \le p$, $p + \Delta \le |K| - 1$): from state $a = \kappa_p$ to state $b = \kappa_{p+\Delta}$, input lines $I_p = \{\kappa_p, \dots, \kappa_{p+\Delta-1}\}$.
- Clip at $p$: states $\kappa_{p + i\Delta}$, $i = 0..L-1$, intervals $p + i\Delta$, $i = 0..L-2$; it exists iff $p + (L-1)\Delta \le |K| - 1$; its span — the lines $\kappa_p..\kappa_{p+(L-1)\Delta}$ (cut lines between included).
- GUI flag of a state $t$: $\text{gui}_t$ = `isGuiOpen` of $\ell_t$, false on the first line of a container span (the first-line rule of `specs/vpt-gui-actions.md`).
- Config `configs/clips.yaml` → `@dataclass(frozen=True) class ClipsConfig`: `delta` (a YAML integer ≥ 1, not a boolean), `actions` (path of `configs/vpt-actions.yaml`, relative to the repository root, resolved against it); violated constraints → `ValueError` at load.

## Math
(1) Interval $p$ is valid iff $n(p) = $ `len(interval_actions(`$S$`, lines, `$a$`, `$b$`, `$C$`, `$\rho$`))` $\ge 1$ and $\{a, \dots, b - 1\} \cap F = \emptyset$ (cut lines included: riding goes on while a screen is cut) ([U]; `specs/vpt-actions.md` (12), `specs/vpt-gui-actions.md` (9)).
(2) A state $t$ passes the line checks iff $\neg\Omega_t$.
(3) Valid starts $V$: the $p$ whose clip exists, whose $L - 1$ intervals are valid by (1) and whose $L$ states pass (2); ascending.
(4) Sampling, with a `numpy.random.Generator` $g$ and a state check $\chi$ given by the caller (the caller classifies the frames of the clip's states with $\text{gui}_t$ true and applies `excluded_states` of `specs/vpt-gui-actions.md` (8)): $W = V$ (ascending); while fewer than $m$ clips are taken and $W \ne \emptyset$: $j$ = $g$`.integers(len(W))`, $p = W[j]$; if $\chi$ accepts the clip at $p$, it is taken and every $p' \in W$ whose span intersects its span (one common line suffices) is removed from $W$, itself included; else $p$ alone is removed; $W$ stays ascending. The clips are returned in the order taken. (Only the first clip is uniform among the accepted starts; later ones are drawn uniformly among the starts that remain.)
(5) A taken clip carries, per interval $p + i\Delta$, its actions (the list of (1)) and $n$; per state, $\text{gui}_t$ and `pitch`.

## Interface
- Subpackage `src/mcpref/clips/`:
  - `config.py`: `ClipsConfig`; `def load_config(path: Path) -> ClipsConfig`.
  - `evidence.py`: `@dataclass(frozen=True) class Evidence`: `lines: Sequence[Mapping]`, `segments: list[Segment]` ($S$), `cut: frozenset[int]`, `foreign: frozenset[int]`, `modes: np.ndarray` ($\Omega$), `turn_deg: float`; `def evidence(lines: Sequence[Mapping], gui: GuiActions, start: tuple[bool, bool, int], actions: ActionsConfig) -> Evidence` — Notation (calls `segments`, `foreign_lines`, `riding_lines`, `mode_frames` of `specs/vpt-actions.md`); `def kept(ev: Evidence) -> list[int]` — $\kappa$.
  - `validity.py`: `def interval(ev: Evidence, kappa: Sequence[int], p: int, delta: int) -> tuple[list[str], bool]` — the actions of interval $p$ and its validity (1); `def valid_starts(ev: Evidence, kappa: Sequence[int], delta: int, states: int) -> list[int]` — (3).
  - `sample.py`: `def gui_flag(lines: Sequence[Mapping], t: int) -> bool` — $\text{gui}_t$ of Notation (also used by block perception-data); `@dataclass(frozen=True) class Clip`: `p: int`, `states: list[int]` (lines), `actions: list[list[str]]`, `n: list[int]` (per interval), `gui: list[bool]`, `pitch: list[float]` (per state); `def sample(ev: Evidence, kappa: Sequence[int], starts: Sequence[int], delta: int, states: int, m: int, g: np.random.Generator, accept: Callable[[list[int]], bool]) -> list[Clip]` — (4), (5); `accept` receives the clip's state lines.
- Consumes: the functions above of `specs/vpt-actions.md` and `specs/vpt-gui-actions.md`. Produces: nothing on disk.
- Callers: block perception-data ($L$, $m$ from its config), block preference-pool.
- No progress bar here: the caller's bar counts clips (one episode's work is in memory; the run time of `valid_starts` on a 6000-line episode is measured at the prototype and reported).
- `specs/00-overview.md`: block 22 clips; block 25 perception-data; package map `clips` → P1, P2 data → this spec and `specs/perception-data.md`.
- Starting config values: `delta` = 20 (one second), `actions` = `configs/vpt-actions.yaml`.
- Tests: hand-made lines, segments, cuts, foreign lines and modes: $\kappa$ around cut spans; a world hold across a cut span (one segment); interval $p$ with $\Delta$ kept lines across a cut span; validity by $n = 0$, by a foreign line, by a riding line inside a cut span, by a mode on a state (first, middle, last); a clip that does not fit; (3) against a naive reference over all $p$; (4) with a fixed seed: draw order on an ascending $W$, overlap removal (spans touching at one line overlap), rejection by `accept`, stop at $m$ and at an empty $W$; (5); the config loader.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1) | `mcpref.vpt.actions.interval_actions` | this repo | $S$, lines, $a$, $b$, $C$, $\rho$ | `specs/vpt-actions.md` |
| $S$ | `mcpref.vpt.actions.segments` | this repo | the kept lines in order, `r_min` | `specs/vpt-actions.md` |
| (4) | `numpy.random.Generator.integers` | numpy 1.26.4 | `0, len(W)`, `dtype=np.int64`, `endpoint=False` (the defaults written out); with `len(W) == 1` it returns 0 without advancing the generator | `_generator.pyx:526, 610-612, 620-621`, `_bounded_integers.pyx.in:340-362`, `distributions.c:1275-1310, 1508-1544` (prototype of 2026-10-15) |
| GUI flag | `mcpref.gui.actions.opens_container` | this repo | lines, $t$ | `specs/vpt-gui-actions.md` |
| config | `yaml.safe_load` | PyYAML 6.0.3 | — | as in `specs/gui-screens.md` |

## Out of scope
- Frames, the classifier and storage (block perception-data); the train / validation split (training blocks).
- Seeding of $g$ (the caller's).
- Clips across episodes.

