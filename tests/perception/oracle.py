"""Oracles of specs/perception.md written from the spec alone: VPT's cursor composite (1),
the meta text (2), the sequence and readouts (3), the M-RoPE rule of [V1] (5), phi(clip) by
a direct run of the backbone (6)-(8), the changed pairs (10) and the loss (11) with its
gradients in float64."""

import contextlib
import math

import numpy as np
import torch
from transformers.image_utils import PILImageResampling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)

from .world import B_ID, E_ID, MARKER_ID, P_ID, S_ALPHA, S_RGB, rgb


def composite(frame: np.ndarray, line: dict, mouse_height: int = 720) -> np.ndarray:
    """`composite_images_with_alpha` of VPT's data_loader.py on a copy, with the cursor
    position of its loader ([V2]): S_RGB, S_ALPHA of the test sprite."""
    image1 = frame.copy()
    f = image1.shape[0] / mouse_height
    x, y = int(line["mouse"]["x"] * f), int(line["mouse"]["y"] * f)
    ch = max(0, min(image1.shape[0] - y, S_RGB.shape[0]))
    cw = max(0, min(image1.shape[1] - x, S_RGB.shape[1]))
    if ch == 0 or cw == 0:
        return image1
    alpha = S_ALPHA[:ch, :cw]
    patch = image1[y : y + ch, x : x + cw, :]
    image1[y : y + ch, x : x + cw, :] = (
        patch * (1 - alpha) + S_RGB[:ch, :cw, :] * alpha
    ).astype(np.uint8)
    return image1


def inventory(line: dict, v: dict) -> str:
    """iota (1) of specs/vlm-labels.md."""
    sums: dict = {}
    for it in line["inventory"]:
        sums[it["type"]] = sums.get(it["type"], 0) + it["quantity"]
    if not sums:
        return v["inventory_line"].replace("{items}", v["no_items"])
    parts = [v["item_line"].replace("{type}", t).replace("{quantity}", str(sums[t]))
             for t in sorted(sums)]  # fmt: skip
    return v["inventory_line"].replace("{items}", ", ".join(parts))


def fmt(x: float, decimals: int) -> str:
    s = f"{x:.{decimals}f}"
    if s.startswith("-") and all(c == "0" for c in s if c.isdigit()):
        return s[1:]
    return s


def wrap(psi: float) -> float:
    return ((psi + 180) % 360) - 180


def meta(line: dict, first: dict, v: dict) -> str:
    """(2), the replacements in their order."""
    d = v["decimals"]
    pairs = [("{inventory}", inventory(line, v)), ("{hotbar}", str(line["hotbar"] + 1)),
             ("{yaw}", fmt(wrap(line["yaw"]), d)), ("{pitch}", fmt(line["pitch"], d)),
             ("{x}", fmt(line["xpos"] - first["xpos"], d)), ("{y}", fmt(line["ypos"], d)),
             ("{z}", fmt(line["zpos"] - first["zpos"], d))]  # fmt: skip
    text = v["meta"]
    for key, value in pairs:
        text = text.replace(key, value)
    return text


def state_line(clip, s: int) -> dict:
    return clip.lines[clip.info["states"][s] - clip.info["states"][0]]


def rgb_of_row(clip, s: int) -> np.ndarray:
    """The RGB frame of row s (neutral chroma), without a cursor."""
    y = clip.frames[s][: clip.width * clip.height]
    return rgb(y.reshape(clip.height, clip.width))


def frames(clip) -> list[np.ndarray]:
    """v_s: the frame of row s, the cursor composited where g_s."""
    out = []
    for s in range(len(clip.frames)):
        v = rgb_of_row(clip, s)
        out.append(composite(v, state_line(clip, s)) if clip.info["gui"][s] else v)
    return out


def metas(clip, v: dict) -> list[str]:
    first = state_line(clip, 0)
    return [
        meta(state_line(clip, s), first, v) for s in range(len(clip.info["states"]))
    ]


def processor(v: dict) -> Qwen2VLImageProcessor:
    """(4)'s processor with P_v = 16, t_p = 2, m_s = 2 of the tiny backbone."""
    return Qwen2VLImageProcessor(
        do_resize=True,
        size={"shortest_edge": v["min_pixels"], "longest_edge": v["max_pixels"]},
        resample=PILImageResampling.BICUBIC,
        do_rescale=True,
        rescale_factor=v["rescale_factor"],
        do_normalize=True,
        image_mean=v["image_mean"],
        image_std=v["image_std"],
        do_convert_rgb=True,
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
    )


def pixels(images: list[np.ndarray], v: dict):
    out = processor(v)(images, input_data_format="channels_last", return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def marker_ids(marker: str) -> list[int]:
    """tau_tok of the tiny tokenizer: the special marker one id, else the UTF-8 bytes."""
    return [MARKER_ID] if marker == "<|im_end|>" else list(marker.encode())


def sequence(texts: list[str], grid, marker: str) -> tuple[list[int], list[int]]:
    """(3): sigma_0 sigma_1 ... and the readouts r_s (the tiny tokenizer: byte k -> id k)."""
    ids, readouts = [], []
    for text, (t, h, w) in zip(texts, grid.tolist()):
        n_s = t * h * w // 4
        ids += [B_ID, *[P_ID] * n_s, E_ID, *text.encode(), *marker_ids(marker)]
        readouts.append(len(ids) - 1)
    return ids, readouts


def rope(ids: list[int], grids: list, merge: int = 2) -> torch.Tensor:
    """(5) by the rule of [V1]: a text token at (q, q, q), then q + 1; consecutive image
    tokens one image of LLM grid H' x W' at (q, q + i, q + j) row-major, then
    q + max(H', W'). Shape (3, 1, len(ids))."""
    cols, q, k, images = [], 0, 0, iter(grids)
    while k < len(ids):
        if ids[k] != P_ID:
            cols.append((q, q, q))
            q, k = q + 1, k + 1
            continue
        _, h, w = (int(x) for x in next(images))
        hh, ww = h // merge, w // merge
        cols += [(q, q + i, q + j) for i in range(hh) for j in range(ww)]
        q, k = q + max(hh, ww), k + hh * ww
    return torch.tensor(cols, dtype=torch.int64).T[:, None, :]


def head(layers, h: torch.Tensor) -> torch.Tensor:
    """(7) from the head's own weights: Linear -> SiLU blocks, then Linear."""
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            h = h @ layer.weight.T + layer.bias
        else:
            h = h * torch.sigmoid(h)
    return h


def phi_clip(phi, clip, v: dict, autocast: bool = True, head_inside: bool = False):
    """(8) of one clip by a direct run of phi.backbone on the hand-built sequence; with
    `head_inside` the head also under autocast (a wrong reading, for contrast)."""
    pv, grid = pixels(frames(clip), v)
    ids, readouts = sequence(metas(clip, v), grid, v["marker"])
    dtype = next(phi.head.parameters()).dtype
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()  # fmt: skip
    with torch.no_grad(), ctx:
        out = phi.backbone(input_ids=torch.tensor([ids]), pixel_values=pv,
                           image_grid_thw=grid, position_ids=rope(ids, grid),
                           use_cache=False)  # fmt: skip
        h = out.last_hidden_state[0, readouts].to(dtype)
        if head_inside:
            return head(phi.head, h)
    with torch.no_grad():
        return head(phi.head, h)


def chi(clip, v: dict) -> list[int]:
    """(10): 1 iff the meta texts differ or the mean |v_{s+1} - v_s| over all values, as
    integers, exceeds tau."""
    fs, ms = frames(clip), metas(clip, v)
    out = []
    for s in range(len(fs) - 1):
        diff = np.abs(fs[s + 1].astype(np.int64) - fs[s].astype(np.int64))
        out.append(int(ms[s] != ms[s + 1] or diff.mean() > v["tau"]))
    return out


def loss(Z: np.ndarray, chis: list, ns: list, w: np.ndarray, v: dict, log_lam: float):
    """(11) of each clip, and the gradients of sum_i w_i l_i by hand: on Z (lambda held
    fixed, q held fixed in the log-lambda term) and on log lambda (only its own term).
    `chis`, `ns` per clip. Returns (values, dZ, dlog)."""
    alpha, lam = np.asarray(v["alpha"], float), math.exp(log_lam)
    eps, beta, c2 = v["dual_slack"], v["beta"], v["c"] ** 2
    values, dZ, dlog = np.zeros(len(Z)), np.zeros_like(Z), 0.0
    for i, z in enumerate(Z):
        L = len(z)
        for s in range(L - 1):
            delta = alpha * (z[s + 1] - z[s])
            d = math.sqrt(float(delta @ delta))
            n2 = ns[i][s] ** 2
            q = min(eps, n2 - d * d)
            norms = [float(z[k] @ z[k]) for k in (s, s + 1)]
            term = beta * sum(max(0.0, x - c2) for x in norms)
            if chis[i][s]:
                term += -d - lam * q + log_lam * q
                g_d = -1.0 + (2 * lam * d if n2 - d * d < eps else 0.0)
                g = g_d * alpha * delta / d if d > 0 else np.zeros_like(delta)
                dZ[i, s + 1] += w[i] * g / (L - 1)
                dZ[i, s] -= w[i] * g / (L - 1)
                dlog += w[i] * q / (L - 1)
            for k, x in zip((s, s + 1), norms):
                if x > c2:
                    dZ[i, k] += w[i] * beta * 2 * z[k] / (L - 1)
            values[i] += term / (L - 1)
    return values, dZ, dlog
