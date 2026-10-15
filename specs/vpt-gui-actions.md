# VPT GUI actions
Status: approved (2026-10-15; clarified after the prototype the same day: `opens_container` in the Interface)
Stage: data of P1, P2 (blocks perception-data, preference-pool), later P5 (`specs/00-overview.md`, block 21)

## Sources
- [U] the user, 2026-10-15 (Q ACT-19, ACT-20 and the keys 1–9 entries): GUI actions are `close` and left, right and shift + left click on element $i = 1..n_{max}$; the screen classifier is pluggable — this block takes the class of each frame it needs as an input; a GUI span whose first frame is `other` or `pause` is cut out of the timeline; a state inside a kept span whose frame is `other` is not used; a click on no element, a drag across elements, and Q or F in a GUI exclude their interval; E and ESC in a GUI are `close`; the middle button is ignored; spans opened without an action (death screen) are kept; a digit pressed in a GUI (swap of the hovered slot with a hotbar slot) is replaced by the shortest equivalent chain of clicks, counted as three clicks; a double click is the action `double_click_`$i$ (90 more), the pair counted as one action.
- [U] the user, 2026-10-15 (after the audit): shift + click is `shift_click_`$i$ on every element, the game deciding the effect (a recipe-book recipe crafts as many as possible); the first line of a container span is a world line; a right press on a recipe (pop-up), lists scrolled by the wheel, the press position of the record after the press, presses without effect: mapped as written; a shift + double click is foreign; a detected pair is a double click whatever the cursor holds; the digit chain is (hovered, $s$, hovered); actions unsupported by a screen are masked by a later block (action-mask).
- `specs/vpt-actions.md` (approved, with the revision of 2026-10-15): lines, $W_k$, $K_k$, $N_k$, `Segment`, `interval_actions`; world actions.
- `specs/gui-screens.md` (approved): classes $\mathcal{K}$, elements with canonical indices (1–9 = hotbar slots 0–8 where the player inventory is shown), `kind`, `element_at`; VPT mouse positions map to the frame point $(x/2, y/2)$; $n_{max} = 90$.
- `specs/vpt-gui-frames.md` [R]: a container GUI is visible from frame $k+1$ when line $k$ is the first with `isGuiOpen` true, the player inventory already in frame $k$; the frame of the first line without a GUI shows none.
- [R4] Measurements of 2026-10-15 on the 10 episodes of `specs/vpt-actions.md` [R3]: 72 GUI spans (35 of the player inventory), median 50 lines, at most 223; 263 presses in GUI lines (240 left, 23 right), 64 with `left.shift` in `keys` (2 of them right); spans ending before the episode's end: 64 with `escape` (41) or `e` (23) in `newKeys` of the span's last line, none with it only in the line before, 7 otherwise; keys pressed in GUI lines besides: `enter` 5, `w` 2, `left.control` 1, `space` 1, a digit 1, `f` 1; 54 of the left presses repeat the left press of 2–5 lines before within 4 frame px (none with shift), 50 of them within 250 ms by `milli` (2–4 lines), 2 right ones likewise; 7 of the left presses start inside a run of $B$ of the previous press (released and pressed again within a record); 82 of 263 presses moved between lines $t - 1$ and $t$, 9 by ≥ 4 frame px, 2 by ≥ 9; no `dwheel` ≠ 0 on a GUI line; all 10 options files have `touchscreen` false and the default `key_key.hotbar.1` … `9`, `key_key.pickItem` (audit of 2026-10-15, scratchpad `audit21/`); 378 GUI lines with a button held while the mouse moves; press positions $x \in [370, 952]$, $y \in [228, 505]$ (window px of 1280×720).
- [G] MCP-Reborn of `specs/gui-screens.md` [G], `src/main/java/net/minecraft/client/gui/screen/inventory/ContainerScreen.java`: `:277-345` (`mouseClicked`: with an empty cursor, a press of button 0 or 1 on a slot with shift held is a quick move, outside the window a throw, else a pickup; with an item on the cursor the press starts a drag and acts on release), `:284` (a double click: the same slot and button within 250 ms) with `:423-436` (on release, for button 0 on a slot that can merge: a pickup of every matching item, or with shift a quick move of all of them; the timer is reset when the cursor is empty, `:504-506`, and after a double click, `:437`), `:532-549` (`keyPressed` consumes every key), `:552-562` (`itemStackMoved`: with an empty cursor over a slot, digit $s$ swaps the slot with hotbar slot $s - 1$, F with the off hand); `inventory/container/Container.java:231, 243` (a quick move with button 1 equals one with button 0); `client/gui/screen/Screen.java:378-381` (`hasShiftDown`: left or right shift); `client/KeyboardListener.java:259-261, 319` (`newKeys` holds OS auto-repeats; a container screen receives repeats only while the recipe book enables them, so a held digit swaps once).

## Notation
- Lines $\ell_0..\ell_{T-1}$, $W_k$, $K_k$, $N_k$ as in `specs/vpt-actions.md`; $N^B_t$ = `mouse.newButtons`, $B_t$ = `mouse.buttons`; $(x_t, y_t)$ = `mouse.x`, `mouse.y`; digits $D$ = {`1`, …, `9`}, digit $s$ ↔ element $s$; $\text{shift}_t$ — shift held: `left.shift` or `right.shift` in $K_t$.
- Span $[k, j]$ — a maximal run of lines with `isGuiOpen` true; its first frame $u = k + 1$ if $j > k$, else $k$.
- $c_t$ — the class of frame $t$, an element of $\mathcal{K} \cup \{\text{other}\}$, given by the caller's classifier for the frames it needs ((1) and the states).
- $e_t$ — `element_at(screens[`$c_t$`], `$x_t / 2$`, `$y_t / 2$`)`, none when $c_t \in$ {`other`, `pause`} ([U]: pause is never an action) or no element contains the point; $\iota(e)$ — its canonical index.
- Kept span — a span that is not cut (2).
- First-line rule ([U]): the first line $k \ge 1$ of a span with `isGuiInventory` of $\ell_k$ false (a container, visible only from frame $k+1$, [R]) counts as a world line for (3)–(8) and for the GUI flag of its state: its presses and keys are ignored (no segment, not foreign) and its state is not subject to (8).

## Math
(1) Needed frames: the first frame $u$ of every span, and every line $t$ of a span with $N^B_t \cap \{0, 1\} \ne \emptyset$ or $N_t \cap D \ne \emptyset$.

(2) Cut: a span is cut iff $c_u \in$ {`other`, `pause`}; its lines $k..j$ are cut lines (Q ACT-20 (a)).

(3) Clicks, for every line $t$ of a kept span and every $b \in N^B_t \cap \{0, 1\}$ that is not a press of a pair of (3a): if $e_t$ is none, line $t$ is foreign; else the segment $(a, t, t)$ with $a$ = `shift_click_`$\iota(e_t)$ if $\text{shift}_t$, for either button and every element kind ([U], [G]); else `left_click_`$\iota(e_t)$ for $b = 0$, `right_click_`$\iota(e_t)$ for $b = 1$.

(3a) Double clicks ([U], [G] `:284`, `:423-436`): in line order within a span, a left press at line $t$ with $e_t$ a slot and a left press at a later line $t'$ of the span with $e_{t'} = e_t$, no left press on the lines between, and `milli`$_{t'}$ − `milli`$_t$ < `double_click_ms`, form a pair unless the press at $t$ already completes a pair; the pair is the one segment (`double_click_`$\iota(e_t)$, $t$, $t'$) instead of two clicks; a pair with $\text{shift}_{t'}$ makes every line of $t..t'$ foreign instead ([U]: a shift + double click moves every matching item, no action).

(4) Drags: for a press $(t, b)$ of (3) or (3a) with $e_t$ not none, its hold is the lines $t..t'$, $t'$ the last line of the run of consecutive lines from $t$ with $b \in B$ within the span and before the next line with $b \in N^B$; if $e_r \ne e_t$ for some line $r$ of the hold, computed with the class $c_t$ (the screen does not change during a hold), every line of the hold is foreign.

(5) Close: for a kept span $[k, j]$ with $j \le T - 2$ and $N_j \cap \{\texttt{e}, \texttt{escape}\} \ne \emptyset$: the segment (`close`, $j$, $j$).

(6) Digits, for every line $t$ of a kept span and $s \in N_t \cap D$ with $t = 0$ or $s \notin K_{t-1}$ (an auto-repeat swaps nothing, [G]) ([U], [G]): if $e_t$ is a slot, $\iota(e_t) \ne s$, and $c_t$ has an element with index $s$: three segments (`left_click_`$\iota(e_t)$, $t$, $t$), (`left_click_`$s$, $t$, $t$), (`left_click_`$\iota(e_t)$, $t$, $t$); else none (no swap happens).

(7) Foreign keys: a line $t$ of a kept span with $N_t \cap \{\texttt{q}, \texttt{f}\} \ne \emptyset$ is foreign. Every other key and the middle button are ignored.

(8) States: a state $t$ with `isGuiOpen` true on a line of a kept span and $c_t \in$ {`other`, `pause`} is not used, nor is any window or history containing it (as in `specs/vpt-actions.md` (12)).

(9) Segments are listed by first line, then in the order (3) with (3a), (5), (6) within a line; (3) and (3a) in the order of $N^B_t$, a pair at the place of its first press; (6) in the order of $N_t$; together with the world segments of `specs/vpt-actions.md`, sorted by first line (they never share a line: world segments lie on world lines), they are the `segs` of `interval_actions`; cut lines are its `cut`; an interval whose lines $a..b-1$ (cut lines included, `specs/clips.md` (1)) contain a foreign line is not used.

## Interface
- Module `src/mcpref/gui/actions.py` (subpackage `gui`: the catalogue it reads lives there; `vpt/` is full):
  - `def gui_action_names(screens: Mapping[str, ScreenClass]) -> tuple[str, ...]` — `close`, then `left_click_1` … `left_click_`$n_{max}$, `right_click_…`, `shift_click_…`, `double_click_…`, $n_{max}$ = the largest element index of the catalogue (90: 361 names).
  - `def action_names(screens: Mapping[str, ScreenClass]) -> tuple[str, ...]` — the order of $A$: `WORLD_ACTIONS` of `specs/vpt-actions.md`, then `gui_action_names`.
  - `def spans(lines: Sequence[Mapping]) -> list[tuple[int, int]]` — the spans $[k, j]$ in order.
  - `def needed_frames(lines: Sequence[Mapping]) -> list[int]` — (1), ascending, without repeats.
  - `def opens_container(lines: Sequence[Mapping], k: int) -> bool` — the first-line rule for a line $k$ with `isGuiOpen` true (its domain): $k \ge 1$, $\ell_{k-1}$ has `isGuiOpen` false, and `isGuiInventory` of $\ell_k$ is false (used by block clips for the GUI flag).
  - `@dataclass(frozen=True) class GuiActions`: `segments: list[Segment]` ((3)–(6) in the order of (9)), `cut: frozenset[int]` ((2)), `foreign: frozenset[int]` ((3), (3a), (4), (7)).
  - `GuiActionsConfig`, `def load_config(path: Path) -> GuiActionsConfig` (config below).
  - `def gui_actions(lines: Sequence[Mapping], classes: Mapping[int, str], screens: Mapping[str, ScreenClass], double_click_ms: int) -> GuiActions` — `classes` = $c_t$ for at least the frames of (1) (a missing one → `KeyError`); `screens` from `load_screens`.
  - `def excluded_states(lines: Sequence[Mapping], classes: Mapping[int, str], cut: Collection[int]) -> frozenset[int]` — (8) over the frames in `classes`.
- Domain: besides the options of `specs/vpt-actions.md`, `key_key.hotbar.1` … `9` = `key.keyboard.1` … `9`, `key_key.pickItem` = `key.mouse.middle`, `touchscreen` false; the callers check them in their `required_options`.
- Callers: block perception-data (and preference-pool, which reuses it) obtains $c_t$ from the classifier for `needed_frames` and the state frames and calls `gui_actions`; block clips merges its segments with the world segments, passes `cut` to `interval_actions` and applies (9) through the foreign lines; block perception-data applies (8) through `excluded_states` in the `accept` of `specs/clips.md` (4).
- `specs/00-overview.md`: block 21; $A$ = 34 world actions and 361 GUI actions; package map `gui` → also P1, P2, P5 data → this spec.
- Config `configs/vpt-gui-actions.yaml` → `@dataclass(frozen=True) class GuiActionsConfig`: `double_click_ms` (a YAML integer, not a boolean, ≥ 1; violated constraints → `ValueError` at load); starting value 250 ([G]).
- No progress bar (one episode in memory).
- Tests: hand-made lines and classes on a tiny catalogue: spans and first frames (1-line span, an episode starting in a GUI, an episode ending in a GUI); `needed_frames` (ascending, no repeats, cut spans included); `excluded_states` with `cut` and a `pause` frame; shift + right on a slot, right shift; presses inside one run of $B$; a hold cut at the span's end; a held digit (auto-repeat counted once); `action_names` order and $n_{max}$ from the catalogue; cut on `other` and `pause`, kept on `death`; the first line of a container span (press ignored, world state) and of the player inventory (mapped); clicks (left, right, shift on a slot, shift on a button, outside every element, on an `other` frame); double clicks at `double_click_ms` − 1 and `double_click_ms`, on another element, a triple press (one pair and one click), a pair across two intervals, a shift + double click (foreign); a drag across two slots and a hold inside one slot; close by `e` and `escape` on the last line only, none at the episode's end; digits over a slot, over slot $s$, over a button, over nothing, on a class without element $s$; Q, F; (8); the order of (9).

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| $e_t$ | `mcpref.gui.screens.element_at` | this repo | the class, $x_t / 2$, $y_t / 2$ | `specs/gui-screens.md` |

## Out of scope
- The classifier (pluggable; blocks 18, 19 or their successor); frames (the caller decodes and classifies them).
- Text input (anvil name, recipe-book search, signs, chat); creative screens.
- The cursor's content: a press with an item on the cursor is mapped like any press (its drag, if it crosses elements, is foreign by (4)); a digit with an item on the cursor is counted as three clicks; a detected pair is a double click whatever the cursor holds.
- Masking of actions a screen does not support (block action-mask); presses without effect count as actions ([U]).
- Server-side closes (a container out of reach, a broken chest): no action.

