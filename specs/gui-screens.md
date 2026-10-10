# GUI screens
Status: approved (2026-10-10; YAML forms and `load_textures_config` added after the gui-masks audit, approved the same day)
Stage: GUI-recognition experiment (Q GX), later env (block 14) and the mapping of human clicks (`specs/00-overview.md`, block 15)

## Sources
- [U] the user, 2026-10-09/10: GUI actions are left / right click on element $i = 1..n_{max}$ of the recognised screen; text input ignored; canonical numbering with the player's 36 slots at the same indices on every screen; classes = distinct element layouts (identical layouts merged), plus "other" (Q ACT-16, ACT-19, GX-2).
- [G] Minecraft 1.16.5 client code as patched by MineRL (MCP-Reborn): `github.com/karry-z/MineRL@71712776d6af5d162e81753fc68f8cf45151e1d7:src/minerl/assets/MCP-Reborn/src/main/java/net/minecraft/client/gui/` and siblings. Its GUI geometry equals vanilla MCP 1.16.5 (`github.com/ConeTin/MCP-1.16.5-Eclipse-Ready@76d553e3411737e5f31b9dc530df4bc6fd12f725`); it differs in behaviour: the death screen respawns after its first frame, enchanting uses a fixed random seed. MineStudio's own engine (`mcprec-6.13`, with OptiFine) is assumed to have the same geometry; the user's plan rebuilds templates on engine frames (Q GX).
- [J] Official 1.16.5 client jar (sha1 `37fd3c903861eeff3bc24b71eed48f828b5269c8`): GUI textures; `font/ascii.png` glyph widths.
- [R] Research of 2026-10-09/10 (checked on real VPT frames): VPT recordings (1280×720 at GUI scale 2, downsampled ×½) and the MineStudio simulator (640×360 at GUI scale 1) share one frame: 1 GUI px = 1 px of the 640×360 frame. The catalogue was generated from [G] with a source reference per number and checked by script and by an independent audit.

## Notation
- Frame coordinates: px of the 640×360 frame, origin top-left, $x$ right, $y$ down; rectangle $(x, y, w, h)$ = $[x, x+w) \times [y, y+h)$. Every coordinate in the catalogue is absolute in this frame; texture coordinates $(u, v)$ are in the texture's pixels.
- Class $k$ — a screen layout; its members — the game screens merged into it; window $(x, y, W, H)$ of $k$; `book_panel` — the recipe-book panel rectangle of the recipe-book classes.
- Element — a slot or button of $k$: canonical index $i$, hit rectangle, click point; $n_{max}$ — the largest index over all classes.
- Blit — texture path (relative to `assets/minecraft/textures/`), $u, v, w, h$, destination $(x, y)$.
- Dynamic rectangle — a region of a member whose pixels depend on game state or on widget state. Label rectangle — a text label.

## Behaviour
(1) Classes and members (screens not listed are "other"):
| Class | Members |
|---|---|
| `inventory`, `inventory_book` | player inventory, recipe book closed / open |
| `crafting`, `crafting_book` | crafting table, closed / open |
| `furnace`, `furnace_book` | furnace, blast furnace, smoker; closed / open |
| `chest3`, `chest6` | chest, trapped chest, barrel, ender chest, minecart with chest / double chest, double trapped chest |
| `shulker_box` | shulker box |
| `dispenser` | dispenser, dropper |
| `hopper` | hopper, minecart with hopper |
| `brewing_stand`, `enchanting_table`, `grindstone`, `stonecutter`, `loom`, `cartography_table`, `merchant`, `beacon`, `lectern` | one game screen each |
| `anvil` | anvil, smithing table (same slots; the anvil's name field is text input, ignored) |
| `horse` | horse, skeleton horse, zombie horse, donkey, mule, llama, trader llama inventories without a chest |
| `horse_chest3` … `horse_chest15` | llamas and trader llamas with 3, 6, 9, 12, 15 chest slots; donkey and mule with a chest (15) |
| `death`, `sleep`, `pause` | death screen, "Leave Bed" screen, pause menu |

(2) Elements: every slot the player can click and every button; not text fields, not drag-only scroll bars. Hit rectangle = the rectangle tested by the screen's click handler; for a slot at $(x_s, y_s)$: $(x_s - 1, y_s - 1, 18, 18)$ ([G] `ContainerScreen.java:512-521`). A class lists the union of its members' elements and every state-dependent element (recipe-book tabs and recipes, trade offers, beacon powers, enchanting options, disabled horse slots); clicking one that is not shown does nothing.
(3) Canonical index: 1–9 = hotbar slots 0–8, 10–36 = main inventory slots 9–35 (the numbering of the simulator's `info['inventory']`) on every class showing the player inventory; then from 37: container slots in the container's slot order; then buttons by group, groups in the order recipe-book button, recipe-book tabs (top to bottom), recipe-book filter, recipe-book recipes, previous page, next page, enchanting options, trades, stonecutter recipes, loom patterns, beacon primary powers, beacon secondary powers, beacon confirm, beacon cancel, lectern buttons, death buttons, sleep button, pause buttons; row-major within a group. Classes without the player inventory start at 37 too. $n_{max} = 90$ (`chest6`).
(4) Click point = centre of the hit rectangle, $(x + w/2, y + h/2)$.
(5) Drawing data per member, for template construction (block gui-masks): blits as drawn in the normal state (not hovered, not selected, active), in draw order; dynamic rectangles: state-dependent drawings at fixed positions (progress arrows, flames, bubbles, enchanting text, anvil cost, models, previews, recipe-book contents, payment icons), and, for a button whose non-normal drawings (selected, hovered, inactive) extend beyond its hit rectangle, the union of all its drawn rectangles; label rectangles: title and "Inventory" with en_us widths (default container names; the merchant's widest title), height 8 without a shadow, 9 with one, 18 for the death title drawn at 2×. A drawing that lies fully inside an element's hit rectangle, a dynamic rectangle or a label rectangle needs no entry. Entries may lie partly outside the window (potion panel, lectern buttons, recipe-book tabs). Slot contents — item icon, durability bar, stack count with its shadow — are drawn within $(x_s, y_s, 17, 18)$, i.e. one row below the slot's hit rectangle; consumers treat that row like the slot (no catalogue entry). Full-screen classes (`death`, `sleep`, `pause`) carry labels and elements only.

## Interface
- Data: `configs/gui-screens.yaml`, generated from [R]'s catalogue and approved with this spec. Top-level keys `classes`, `textures`. Per class: `members` (mapping member → {`blits`, `dynamic`, `labels`}), `description` (one line telling the class apart, for the VLM, Q GX-3), `window`, `book_panel` (recipe-book classes only), `elements`. YAML forms: a rectangle is `[x, y, w, h]`; `window` and `book_panel` are rectangles at class level; dynamic and label entries are `{name, rect}`; elements are `{index, name, kind, hit, click}` with `hit` a rectangle and `click` `[x, y]`; blits are `{texture, u, v, w, h, x, y}`. Entries at every level (classes included) may carry `src`, `note`, `cond` strings, which the loader ignores; an unknown key or a missing required key at any level raises `ValueError`. Unnamed rectangles get the names `window`, `book_panel`, and the element's name for `hit`. `textures`: `url`, `sha1`, `out_dir` (relative to the repository root, resolved against it whatever the working directory), `http_timeout` (s, per phase), `http_retries`, `retry_delay` (s).
- Module `src/mcpref/gui/screens.py`:
  - `@dataclass(frozen=True)` classes: `Rect(name: str, x: int, y: int, w: int, h: int)`; `Blit(texture: str, u: int, v: int, w: int, h: int, x: int, y: int)`; `Element(index: int, name: str, kind: Literal["slot", "button"], hit: Rect, click: tuple[float, float])`; `Member(blits: tuple[Blit, ...], dynamic: tuple[Rect, ...], labels: tuple[Rect, ...])`; `ScreenClass(name: str, description: str, window: Rect, book_panel: Rect | None, members: Mapping[str, Member], elements: tuple[Element, ...])`.
  - `def load_screens(path: Path) -> dict[str, ScreenClass]` — reads `classes`; checks that every `click` equals (4) for its hit rectangle (else `ValueError`).
  - `def element_at(cls: ScreenClass, x: float, y: float) -> Element | None` — the element whose hit rectangle contains the frame point $(x, y)$; consumers convert recorded mouse positions to frame coordinates first (VPT: $(x/2, y/2)$, Q ACT-16).
- Module `src/mcpref/gui/textures.py`; `@dataclass(frozen=True) class TexturesConfig` (the `textures` keys; `out_dir: Path` already resolved against the repository root, absolute) and `def load_textures_config(path: Path) -> TexturesConfig` (used by consumers to find `out_dir`); `main(argv)` takes the arguments without the program name; `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.gui.textures configs/gui-screens.yaml`: if every texture named by a blit and `font/ascii.png` already exists under `out_dir`, does nothing; otherwise downloads `url` to `out_dir/client.jar.part` (`httpx`, `timeout` per phase, retries after `retry_delay` on a transport error (any `httpx.TransportError`) or HTTP 5xx; other non-2xx statuses, redirects included, fail at once), checks its sha1 against `sha1` (mismatch → delete the file, exit 1; the body is held in memory until then), extracts every member under `assets/minecraft/textures/gui/` and `assets/minecraft/textures/font/ascii.png` to `out_dir/<path relative to assets/minecraft/textures/>`, deletes the jar; exit 0, or 1 with a one-line message on stderr.
- `scripts/download_data.sh` (block vpt-gui-frames): gains `uv run --locked python -m mcpref.gui.textures configs/gui-screens.yaml` as its first step.
- Produces: the catalogue for gui-labels, gui-masks, env; textures under `out_dir`.
- Tests: hand-checked entries (`inventory` hotbar slot 0: hit (239, 238, 18, 18), click (248, 247); `chest6` index 90 = chest slot 53; `inventory_book` index 47 = the recipe-book button; `beacon` index 42 = primary power "strength"); no two elements of a class overlap; indices unique, 1–36 the player slots where shown; every click inside its hit rectangle; unknown keys rejected, `src` / `note` / `cond` ignored; `element_at` on rectangle borders (half-open); `textures.py` with a fake HTTP server serving a small zip (sha1 match, mismatch, skip when present, retries).

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| loading | `yaml.safe_load` | PyYAML 6.0.3 | — | to vet at implementation |
| download | `httpx.Client.get` | 0.28.1 | `timeout=httpx.Timeout(http_timeout)`, `follow_redirects=False`; client defaults otherwise (`trust_env=True`: a proxy from the environment may be needed on the internet) | `httpx/_config.py` (audit of 2026-10-10) |
| checksum | `hashlib.sha1` | Python 3.10 stdlib | — | — |
| extraction | `zipfile.ZipFile.open` | Python 3.10 stdlib | members by exact name | — |

## Out of scope
- Creative-mode screens, books and quills, written books read from the hand, signs, advancements, statistics, chat, options: class "other".
- Text fields, drag-only scroll bars, the recipe-book alternative-recipe pop-up; renamed containers and name-tagged animals (longer titles).
- Cursor-dependent drawings (hover highlights, held item, tooltips), the in-game HUD, the world-dimming gradient and anything outside a class's window: no template data.
- GUI scales other than the frame of [R].
