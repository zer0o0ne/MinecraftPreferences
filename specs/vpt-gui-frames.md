# VPT GUI frames
Status: approved (2026-10-10; clarified after the prototype the same day: verify URL, run.json form and timing)
Stage: GUI-recognition experiment (Q GX; `specs/00-overview.md`, block 16)

## Sources
- [U] the user, 2026-10-10: several thousand random VPT frames with an open GUI, without downloading whole videos; episodes uniformly over the five subsets, frames uniformly among an episode's GUI frames, at most $m$ per episode (Q GX-1); `isGuiInventory` kept as an independent check (Q GX-3); `scripts/download_data.sh` downloads the data.
- [VPT] `github.com/openai/Video-Pre-Training@095519fbd4ee0e9281d19f19601e45629de9ac3f`: README L144, L162, L170, L204, L285 (index URLs), L343-346 (file naming); `data_loader.py` L80-117 (line $i$ ↔ frame $i$).
- [R] Measurements of 2026-10-10 on the real data (notes in the session scratchpad, `vpt_frames/`):
  - index `basedir` = `https://openaipublic.blob.core.windows.net/minecraft-rl/`; `relpaths` = `data/<version>/<stem>.mp4`; `<stem>.jsonl`, `<stem>-options.json` sit next to it; an episode has all its files or none (6xx: 38 % exist); zero-size `.mp4` files occur only with zero-size `.jsonl` files; 6xx lists 3 identical relpaths twice; 217 stems occur under two relpaths, at most one of which has files;
  - `.jsonl`: one JSON object per tick; `isGuiOpen`; `isGuiInventory` (true only for the player inventory) from recorder version 6.12 on; options files have `guiScale`, `fullscreen`, `framebufferWidth`, `framebufferHeight`, `lang`, `forceUnicodeFont`, `resourcePacks`;
  - videos: H.264, 640×360, yuv420p, constant 20 fps, pts of frame $i$ = $i/20$ s, frame count = line count + 1 (checked on 24 episodes);
  - a container GUI is visible from frame $k+1$ when line $k$ is the first with `isGuiOpen` true; the player inventory already in frame $k$; when line $k$ is the first with `isGuiOpen` false after a GUI, frame $k$ shows no GUI;
  - Azure serves byte ranges only with the request header `x-ms-version: 2019-12-12`; `ffmpeg -ss <t> -i <url> -frames:v 1 -f rawvideo -pix_fmt yuv420p` with that header returns frame $i$ bit-exactly when $t$ is the exact decimal of $i/20$ (28 frames of 2 videos against a full local decode, ffmpeg 7.0.2), ~1–6 MB and ~2.5 s per frame;
  - ffmpeg on Ubuntu 24.04 arm64: apt version `7:6.1.1-3ubuntu5` (gnutls, `https`); static builds cannot resolve host names; the videos carry no colour metadata.
- [BT601] ITU-R BT.601-7: $Y' \in [16, 235]$, $C_b, C_r \in [16, 240]$, $K_R = 0.299$, $K_B = 0.114$.

## Notation
- Subsets $S$: the keys of config `index_urls`, in config order (`6xx`, `7xx`, `8xx`, `9xx`, `10xx`).
- Episode $e$: a relpath of $S$'s index; version $v_e$ = the `<version>` path component, compared as a tuple of integers.
- $\ell_0, \dots, \ell_{n-1}$ — the lines of $e$'s `.jsonl`; frame $i$ is paired with $\ell_i$.
- $F_e$ — eligible frames, ascending; $r$ — frames taken from $e$.
- $N$, $m$ — total frames and frames per episode (config).
- $t_i$ — seek time of frame $i$; $Y, U, V$ — luma and the two chroma planes ($U = C_b$, $V = C_r$); $R, G, B$ — the RGB view.
- Config `configs/vpt-gui-frames.yaml` → `@dataclass(frozen=True) class GuiFramesConfig`: `index_urls` (mapping subset → URL), `x_ms_version`, `min_version`, `required_options` (mapping option → value), `width`, `height`, `fps`, `n_frames` ($N$), `per_episode` ($m$), `seed`, `workers`, `http_timeout` (s, per connect / read / write / pool phase), `http_retries`, `retry_delay` (s), `ffmpeg` (program path), `ffmpeg_rw_timeout` (s), `ffmpeg_timeout` (s), `verify_relpath`, `verify_frames` (list of $i$), `out_dir` (relative to the repository root).

## Math
(1) $F_e = \{ i : 1 \le i \le n - 1,\ \ell_i.\text{isGuiOpen} \wedge \ell_{i-1}.\text{isGuiOpen} \}$ — the GUI is drawn in frame $i$ for every screen type ([R], containers one tick late).
(2) Episode reason, checks in this order, the first failing one is the reason, `None` if all pass: `version` ($v_e <$ `min_version`; decided from the relpath, nothing downloaded); `missing` (HTTP 404 for the `-options.json` or the `.jsonl`); `empty` (either file has zero bytes); `malformed` (either file is not valid JSON / JSON Lines); `options` (some `required_options` key differs from the options file); `no_gui` ($F_e = \emptyset$). The `.mp4` is not checked ([R]: it is empty only with an empty `.jsonl`).
(3) Plan, one `numpy.random.Generator(PCG64(seed))` used only here, sequentially, until $N$ frames are planned or every pool is empty: pools = each subset's relpaths in index order with exact repeats removed; $s$ = `rng.integers(len(active))` over the subsets whose pool is non-empty, in config order; then within subset $s$, repeat: $j$ = `rng.integers(len(pool))`, `e = pool.pop(j)`, compute the reason (2), until an episode is valid or the pool is empty (an empty pool drops $s$ from the active subsets); for a valid $e$: $r = \min(m, |F_e|, N - \text{planned})$, frames = `rng.choice(F_e, size=r, replace=False)` in draw order.
(4) $t_i$ = `"%d.%02d" % (i // fps, (100 // fps) * (i % fps))` for `fps` = 20 ([R]).
(5) RGB view of a raw frame (planes $Y$: height × width, then $U$, $V$: height/2 × width/2), [BT601] limited range with chroma repeated over its 2 × 2 block, in float64:
$R = \frac{255}{219}(Y - 16) + \frac{255}{224} \cdot 1.402\,(V - 128)$,
$G = \frac{255}{219}(Y - 16) - \frac{255}{224} \cdot \frac{1.402 \cdot 0.299}{0.587}(V - 128) - \frac{255}{224} \cdot \frac{1.772 \cdot 0.114}{0.587}(U - 128)$,
$B = \frac{255}{219}(Y - 16) + \frac{255}{224} \cdot 1.772\,(U - 128)$,
each value $x \mapsto \min(255, \max(0, \lfloor x + 0.5 \rfloor))$ as `uint8`.

## Behaviour
- (a) Indices: download each index URL; any HTTP failure after the retries aborts the run.
- (b) Plan (3), with downloads of options and `.jsonl` files over `httpx` (header `x-ms-version`); 404 → reason `missing`; a timeout, connection error or HTTP 5xx is retried after `retry_delay` up to `http_retries` times, then aborts the run (so the plan never depends on network luck).
- (c) Extraction of planned frames on `workers` threads: `ffmpeg_command` writes to `<id>.yuv.part`; success = exit code 0 and exactly width × height × 3 / 2 bytes, then the file is renamed to `<id>.yuv`; otherwise retried after `retry_delay` up to `http_retries` times, then the frame's status is `failed`; a run of ffmpeg longer than `ffmpeg_timeout` counts as a failure.
- `run.json` is written (config and index ETags) right after the indices are downloaded and the rerun check of (d) passes, before planning, and completed with the counts at the end; so a run interrupted at any later point is protected by (d) (a rerun with another config then needs `run.json` deleted, even if no frame was extracted).
- (d) Rerun with the same config: if `run.json` exists and its config or index ETags differ → abort; otherwise the plan is recomputed (identical by (3)), frames whose `.yuv` exists are skipped, `failed` ones are tried again, and the record files are rewritten in plan order.
- (e) `verify` (run by `download_data.sh` before the full run, VF-2): downloads the whole `verify_relpath` video (URL: the `basedir` of the first index, in config order, that lists it; none → abort) into a temporary file, decodes it sequentially with the same ffmpeg to raw yuv420p, extracts each of `verify_frames` with the extraction of (c) (`ffmpeg_command`, retries, `ffmpeg_timeout`, `.part` and size check; a frame still failing aborts), and aborts unless every one equals the sequential decode byte for byte; the temporary files are deleted.

## Interface
- Module `src/mcpref/vpt/gui_frames.py`; `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.vpt.gui_frames {verify|run} <config path>`; exit 0 on success (failed frames counted, not fatal), 1 on abort with a one-line message on stderr; progress counts on stderr.
  - `def load_config(path: Path) -> GuiFramesConfig`.
  - `def eligible_frames(lines: Sequence[Mapping]) -> list[int]` — (1).
  - `def version_ok(version: str, min_version: str) -> bool`; `def options_ok(options: Mapping, required: Mapping) -> bool` — parts of (2).
  - `def plan(cfg: GuiFramesConfig, fetch: Callable[[str], bytes | None]) -> tuple[list[PlannedFrame], list[EpisodeRecord]]` — (3); `fetch` returns `None` for 404; `PlannedFrame` = (`id`, `subset`, `relpath`, `version`, `frame`, `lines` = $(\ell_{i-1}, \ell_i)$); `EpisodeRecord` = (`subset`, `relpath`, `reason`, `options` or `None`).
  - `def frame_time(i: int, fps: int) -> str` — (4).
  - `def ffmpeg_command(cfg: GuiFramesConfig, url: str, i: int, out: Path) -> list[str]` = `[cfg.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-headers", "x-ms-version: <x_ms_version>\r\n" (a real CR LF), "-rw_timeout", <ffmpeg_rw_timeout × 10⁶ as an integer>, "-ss", t_i, "-i", url, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", out]`.
  - `def yuv420p_to_rgb(buf: bytes, width: int, height: int) -> np.ndarray` — (5); shape (height, width, 3), `uint8`.
- Produces `{out_dir}/`: `frames/<id>.yuv`, `<id>` = `<version>_<stem>_<i>`; `frames.jsonl` (plan order): `id`, `subset`, `relpath` (with `.mp4`), `version`, `frame`, `prev_line` ($\ell_{i-1}$), `line` ($\ell_i$), `status` (`ok` | `failed`); `episodes.jsonl` (draw order): `subset`, `relpath`, `reason` (string or `null`), `options` (object or `null`); `run.json`: `{"config": <the config as loaded>, "index_etags": {subset: ETag or null}, "episodes": {reason or "null": count}, "frames": {status: count}}` (counts only for reasons and statuses that occurred; consumers read `config.width`, `config.height`).
- Script `scripts/download_data.sh` (preamble of `specs/spark-env.md`): `uv run --locked python -m mcpref.vpt.gui_frames verify configs/vpt-gui-frames.yaml`, then `… run configs/vpt-gui-frames.yaml`; block gui-screens adds the texture download.
- `scripts/setup_env.sh` step (7) `ffmpeg` (block spark-env; uses `sudo`, stated here): `configs/spark-env.yaml` gains `ffmpeg_deb_version`; if `dpkg-query -W -f='${db:Status-Status} ${Version}\n' ffmpeg libavcodec60 libavformat60 libavutil58 libswscale7 libswresample4` does not print exactly six lines `installed <ffmpeg_deb_version>`, one per package (a removed package keeps its version but not the status `installed`; only the printed lines are compared, the exit status of `dpkg-query` is ignored; second-architecture copies, which could make the step reinstall on every run or mask a missing package, do not occur on the arm64-only Spark — accepted): `sudo apt-get update`, `sudo apt-get install -y --allow-downgrades` each package `=<ffmpeg_deb_version>`, `sudo apt-mark hold` them.
- `pyproject.toml`: adds `httpx==0.28.1` if absent (shared with block vlm-client).
- Starting config values: `index_urls` = `all_6xx_Jun_29.json`, `all_7xx_Apr_6.json`, `all_8xx_Jun_29.json`, `all_9xx_Jun_29.json`, `all_10xx_Jun_29.json` under `https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/`; `x_ms_version` = `2019-12-12`; `min_version` = `6.12`; `required_options` = {`guiScale`: 2, `fullscreen`: false, `framebufferWidth`: 1280, `framebufferHeight`: 720, `lang`: `en_us`, `forceUnicodeFont`: false, `resourcePacks`: `"[]"`} (the GUI geometry and textures of `specs/gui-screens.md` hold only then); `width` = 640, `height` = 360, `fps` = 20; `n_frames` = 5000; `per_episode` = 5; `seed` = 0; `workers` = 8; `http_timeout` = 60; `http_retries` = 3; `retry_delay` = 10; `ffmpeg` = `/usr/bin/ffmpeg`; `ffmpeg_rw_timeout` = 60; `ffmpeg_timeout` = 300; `verify_relpath` = `data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4`; `verify_frames` = [0, 127, 128, 129, 1271, 5999, 6000]; `out_dir` = `data/gui_frames`.
- Tests: (1), (2) parts, (4), (5) against hand cases (GUI open / close boundaries; reasons in order; $t_i$ for $i$ = 0, 1, 19, 20, 1271; (5) on hand-picked pixels incl. clipping; no exact .5 occurs for any (Y, U, V), checked exhaustively); (3) with a fake `fetch` and a fixed seed (subset choice, pool exhaustion, the cut at $N$, reasons) and its statistics over many seeds (subset shares, per-episode uniformity); `ffmpeg_command` against the list above; `main` with a fake HTTP server and a fake `ffmpeg` (config `ffmpeg` pointing at it): retries, `.part` handling, rerun, abort cases; `setup_env.sh` step (7) and `download_data.sh` with fake `sudo`, `apt-get`, `apt-mark`, `dpkg-query`, `uv`. The real run is manual, on the Spark.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (c), (e) | `ffmpeg` CLI | apt `7:6.1.1-3ubuntu5` (all six packages held) | as in `ffmpeg_command` | [R] (checked with 7.0.2; `verify` re-checks on the Spark) |
| (a), (b) | `httpx.Client.get` | 0.28.1 | `timeout=httpx.Timeout(http_timeout)` (per phase, no total deadline), header `x-ms-version`, `follow_redirects=False` | `httpx/_config.py` (audit of 2026-10-10) |
| (3) | `numpy.random.Generator(PCG64(seed))`; `integers(k)`; `choice(a, size=r, replace=False)` | numpy 1.26.4 | as written in (3) (`choice` with its default `shuffle=True`) | to vet at implementation |

## Out of scope
- Recorder versions < 6.12, other window sizes, GUI scales, languages, resource packs; option changes in the middle of an episode (the options file is one snapshot); BASALT data.
- Whole-video downloads (except `verify`); caching `.jsonl` files.
- Checking each video's frame rate and frame count (accepted from [R]).
- Drawing the cursor (Q D-9: done where frames feed $\phi$).
