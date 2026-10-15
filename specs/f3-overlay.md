# F3 overlay
Status: approved (2026-10-15)
Stage: data of P1, P2 (blocks perception-data, preference-pool; `specs/00-overview.md`, block 24)

## Sources
- [U] the user, 2026-10-15 (`specs/vpt-actions.md`, start of (9)): an episode that does not start a session reads the debug overlay's state at its start from frame 0, by a detector of the overlay's text; a separate small block; it reads the prefix `Minecraft 1.16.5 (`, independent of the launcher's version string; the coverage of [R5] is accepted.
- [G] MCP-Reborn of `specs/gui-screens.md` [G], `src/main/java/net/minecraft/`: `client/gui/overlay/DebugOverlayGui.java:199` (first left line `"Minecraft " + version + " (" + launcher version + "/" + brand + …")"`), `:124-133` (line $i$: pen at $(2, 2 + 9i)$; background `fill` over $x \in [1, k + 3)$, $y \in [1 + 9i, 10 + 9i)$, $k$ the line's width, colour 0x90505050; text `drawString` in 0xE0E0E0, no shadow); `client/gui/FontRenderer.java:65-68, 134-143` (no shadow; opaque colour); `client/gui/AbstractGui.java:46-75` (alpha blending of the fill); `client/gui/screen/Screen.java:330-336` (screens that draw the background gradient 0xC0101010 → 0xD0101010 dim everything below, the overlay included); `client/renderer/GameRenderer.java:468-471` (the HUD with the overlay is drawn unless the HUD is hidden with no screen open), `:489-491` (then the screen, whose background gradient therefore dims the overlay); `client/gui/screen/DeathScreen.java:80` (the death screen draws its own red gradient 0x60500000 → 0xA0803030 instead).
- `specs/gui-masks.md` (9): text pixels $G_0$ of a label drawn with `font/ascii.png` (no shadow for a rectangle of height 8); `load_font`.
- `specs/vpt-gui-frames.md` (5) [BT601]: limited-range luma of the raw yuv420p frames.
- [R5] Research of 2026-10-15 (scratchpad `f3-research/`): 26 episodes of 20 players, 134,400 frames, labelled from the F3 holds of their `.jsonl` (85 of 85 toggles match the video to the frame): 4,624 frames with the overlay (5 players; versions 7.2, 7.7, 10.0; 236 of them dimmed by a screen), 129,776 without; the first line reads `Minecraft 1.16.5 (null/vanilla)` for all 5 players, at the predicted pixels (share on target 1.000 at offset (0, 0), ≤ 0.434 at ±1 px); predicted luma: text 208.38 (73.7 dimmed), box $47.83 + 0.4353\,Y_{scene} \in [54.8, 150.1]$ ([35.9, 59.4] dimmed); measured text median 206.0 (std 2.3; 99.99 % within ±24), dimmed 73.0 (99.97 % within ±8); box pixels next to glyphs up to 159 (ringing), elsewhere ≤ 150.1 but for 0.002 %; with the measure of (1)–(2) on the prefix `Minecraft 1.16.5 (`: $S \in [0.959, 1]$ with the overlay, $[0, 0.633]$ without (the full line: $[0.969, 1]$ and $[0, 0.583]$), including dark scenes and bright sky. Coverage: frames with the overlay only from recorder versions 7.2, 7.7 and 10.0 (6.13, 8.x, 9.x only without it); the 236 dimmed ones from one player (2 episodes); snow behind line 0 in 1 frame with the overlay and 53 without; 1 of the 26 frames 0 with the overlay. Audit of 2026-10-15 (scratchpad `audit-f3/verify.py`): (1)–(3) with the starting config on all 134,400 frames give $S \ge 0.9586$ with the overlay, $\le 0.6331$ without; for integer $Y$ the starting values select text luma {185..232} and {66..81}, box $Y \le 160$ and $\le 63$.

## Notation
- $Y$ — the luma plane of a raw yuv420p frame: its first $W \cdot H$ bytes as an $H \times W$ array (`uint8`).
- $\tau$ — the detected text (`prefix`); $P$ — its text pixels $G_0$ by `specs/gui-masks.md` (9) for the label with name $\tau$, rectangle $(2, 2, w_\tau, 8)$ and colour 0xE0E0E0, $w_\tau = \sum_i \text{adv}(\tau_i)$; $R$ — the box rectangle $x \in [1, 2 + w_\tau)$, $y \in [1, 10)$; $\bar P$ — the 3×3 dilation of $P$: $\{p + (d_x, d_y) : p \in P,\ d_x, d_y \in \{-1, 0, 1\}\}$; $Q = R \setminus \bar P$.
- Two templates $m \in \{\text{n}, \text{d}\}$ (normal, dimmed by a screen): text luma $\lambda_m$, tolerance $\delta_m$, box bound $\beta_m$.
- Config `configs/f3-overlay.yaml` → `@dataclass(frozen=True) class OverlayConfig`: `screens` (path of `configs/gui-screens.yaml` relative to the repository root, resolved against it; its `textures.out_dir` holds `font/ascii.png`), `prefix` (a non-empty string of characters 0x20–0x7E), `text_luma`, `tolerance`, `box_max` (each a mapping with exactly the keys `normal`, `dimmed`; values YAML integers or floats, not booleans, read with `float()`; `tolerance` ≥ 0), `threshold` ($\theta$, a YAML integer or float, not a boolean, in (0, 1]); violated constraints → `ValueError` at load.
- $W$, $H$ — the frame size of the caller's config (`width`, `height`).

## Math
(1) Shares ([R5]): $a_m = |\{p \in P : |Y(p) - \lambda_m| \le \delta_m\}| / |P|$, $g_m = |\{p \in Q : Y(p) \le \beta_m\}| / |Q|$, in float64 with $Y$ as integers.
(2) [R5]: $S_m = \min(a_m, g_m)$; $S = \max(S_{\text{n}}, S_{\text{d}})$.
(3) The overlay is shown iff $S \ge \theta$ ([R5]: $\theta$ separates the measured frames).

## Interface
- Module `src/mcpref/recognition/debug_overlay.py`:
  - `OverlayConfig`; `def load_config(path: Path) -> OverlayConfig`.
  - `def overlay_font(cfg: OverlayConfig) -> np.ndarray` — $\Phi$: `load_font(load_textures_config(cfg.screens).out_dir)`; a missing `font/ascii.png` raises `FileNotFoundError` (the caller's script downloads the textures first, as `scripts/download_data.sh` does).
  - `def overlay_masks(font: np.ndarray, prefix: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]` — $P$, $Q$ (`bool`, $H \times W$); `ValueError` when $|P| = 0$, $|Q| = 0$ or $R$ does not fit in the frame.
  - `def luma(buf: bytes, width: int, height: int) -> np.ndarray` — $Y$ of a raw yuv420p frame.
  - `def overlay_score(y: np.ndarray, masks: tuple[np.ndarray, np.ndarray], cfg: OverlayConfig) -> float` — (1), (2).
  - `def overlay_shown(y: np.ndarray, masks: tuple[np.ndarray, np.ndarray], cfg: OverlayConfig) -> bool` — (3).
- Caller: block perception-data (25; its pipeline is reused by preference-pool): for an episode that does not start a session (`starts_session` of `specs/vpt-actions.md` false), $O_0$ of `mode_frames` = `overlay_shown` on `luma` of frame 0.
- Domain: the options of `specs/vpt-gui-frames.md` `required_options` (guiScale 2, framebuffer 1280×720, not fullscreen, no Unicode font, no resource packs), enforced by the caller's `required_options`.
- Consumes: `font/ascii.png` under `textures.out_dir` of `configs/gui-screens.yaml`; frame 0 of the caller. Produces: nothing on disk.
- No progress bar (one frame).
- `specs/00-overview.md`: block 24; package map `recognition` → also P1, P2 data → this spec.
- Starting config values: `screens` = `configs/gui-screens.yaml`, `prefix` = `Minecraft 1.16.5 (`, `text_luma` = {normal: 208.38, dimmed: 73.7}, `tolerance` = {normal: 24, dimmed: 8}, `box_max` = {normal: 160.1, dimmed: 63.4}, `threshold` = 0.75 ([R5]: any value in [0.65, 0.95] separates the measured frames).
- Tests: on a hand-made font: $P$ (positions, advance, space), $R$, the dilation and $Q$ at the borders of the frame; (1) on synthetic $Y$ planes at the tolerance and box bounds (inclusive), integer arithmetic (no `uint8` wrap-around); (2), (3) at $S = \theta$; on the real `ascii.png` (skipped with the reason when absent): $w_\tau = 87$, $|P| = 169$, $|Q| = 217$ for the starting prefix; `ValueError` of `overlay_masks`; `luma`; the config loader.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| $P$, $w_\tau$ | `mcpref.recognition.drawn_masks.text_pixels`, `glyph`, `advance`, `load_font` | this repo | the label of Notation | `specs/gui-masks.md` (9) |
| font path | `mcpref.gui.textures.load_textures_config` | this repo | `cfg.screens` | `specs/gui-screens.md` |
| config | `yaml.safe_load` | PyYAML 6.0.3 | — | as in `specs/gui-screens.md` |

## Out of scope
- The other overlay lines and the right column (their texts vary between players, toasts cover them).
- Anything drawn over line 0 (tooltips, the player list); the hidden HUD (F1), with which the overlay is not drawn while no screen is open; frame 0 under the death screen's red gradient or an opaque screen (an overlay that is on reads as off; [U] 2026-10-15: accepted).
- Tracking the overlay over an episode (`specs/vpt-actions.md` (9)).

