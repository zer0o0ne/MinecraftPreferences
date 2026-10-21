"""Transcription of specs/vlm-labels.md for the runs of labels_run.py, written from the spec
alone: the prompts (1) of the tiny config by hand, the requests and positions of Notation,
the label shards and run.json of (I), (L); prompt comparison; the bars' final states."""

import hashlib
import json

import numpy as np
import yaml
from pretrain.index_oracle import clips_fingerprint, json_bytes, name, shard_files
from progress_bars import Bar

from .config_files import TINY_THETA
from .labels_world import ANSWERS, CLIPS, code, frame_rgb, ref

# Notation, j_q = L - 1 - (k_V - 1 - q) s_V by hand for L = 6: (k_V, s_V) -> shown frames
SHOWN = {(3, 2): [1, 3, 5], (6, 1): [0, 1, 2, 3, 4, 5], (1, 7): [5], (2, 5): [0, 5]}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Expected:
    """The expected calls and {out_dir} of a Site (config, shards, at, path)."""

    def prompt(self, x: str, y: str) -> list:
        """(1) of (x, y) with TINY's texts, the frames of SHOWN, by hand."""
        js = SHOWN[self.config["frames"], self.config["stride"]]
        fx, fy = [[frame_rgb(code(c, j)) for j in js] for c in (x, y)]
        return ["I\n", "A:\n", *fx, CLIPS[x][2], "B:\n", *fy, CLIPS[y][2], TINY_THETA]

    def requests(self) -> list[list[tuple]]:
        """Per shard: (prompt, seed, comparison, order, sample) of its requests in order,
        position 2 s m + o s + i (Notation)."""
        s, m, out = self.config["samples"], 0, []
        for shard in self.shards:
            out.append([])
            for a, b in shard.comparisons:
                for o, (x, y) in enumerate([(a, b), (b, a)]):
                    for i in range(s):
                        seed = self.config["seed"] + 2 * s * m + o * s + i
                        out[-1].append((self.prompt(x, y), seed, (a, b), o, i))
                m += 1
        return out

    def expected_calls(self) -> list[tuple[list, list]]:
        """ask_many calls: (prompts, seeds) per shard; none for a shard without
        comparisons."""
        return [([r[0] for r in q], [r[1] for r in q]) for q in self.requests() if q]

    def expected_shard(self, k: int, labels: dict, end: int) -> dict:
        """Label shard k: answers.json, counts.json (its `labels` given)."""
        answers, requests = [], self.requests()[k]
        for a, b in self.shards[k].comparisons:
            answers.append({"a": ref(a), "b": ref(b), "requests": []})
            for _, seed, _, o, i in [r for r in requests if r[2] == (a, b)]:
                answer, label = ANSWERS[(seed - self.config["seed"]) % len(ANSWERS)]
                request = {"order": o, "sample": i, "seed": seed, "answer": answer}
                answers[-1]["requests"].append(request | {"label": label})
        n = len(self.shards[k].comparisons)
        counts = {"comparisons": n, "labels": labels, "end": end}
        return {"answers.json": json_bytes(answers), "counts.json": json_bytes(counts)}

    def expected_run(self, labels: dict | None = None) -> bytes:
        """{out_dir}/run.json after (I) or, with the summed `labels`, after (L)."""
        pool = self.at("pool_dir") / "shards"
        records = shard_files(pool, "records.json")
        run = json.loads((self.at("pool_dir") / "run.json").read_text())
        fp = {
            "config": sha(self.path.read_bytes()),
            "vlm": sha(self.at("vlm").read_bytes()),
        }
        fp["pool"] = clips_fingerprint(run, records)
        fp["comparisons"] = sha(b"".join(shard_files(pool, "comparisons.json")))
        value = {"config": yaml.safe_load(self.path.read_text()), "sha256": fp}
        if labels is not None:
            n = sum(len(s.comparisons) for s in self.shards)
            value |= {"complete": True, "counts": {"comparisons": n, "labels": labels}}
        return json_bytes(value)

    def expected_tree(self, labels: list, total: dict, ends: list) -> dict:
        """{out_dir} after (L): per label shard its `labels` and `end`; `total` summed."""
        tree = {"run.json": self.expected_run(total)}
        for k in range(len(self.shards)):
            for file, data in self.expected_shard(k, labels[k], ends[k]).items():
                tree[f"shards/{name(k)}/{file}"] = data
        return tree


def same_part(a, e) -> bool:
    """A text part exactly; an image as a uint8 array of the same shape and pixels."""
    if isinstance(e, str):
        return isinstance(a, str) and a == e
    return isinstance(a, np.ndarray) and a.dtype == np.uint8 and np.array_equal(a, e)


def same_prompts(actual: list, expected: list) -> bool:
    """Prompts equal part by part."""
    return len(actual) == len(expected) and all(
        len(p) == len(q) and all(map(same_part, p, q)) for p, q in zip(actual, expected)
    )


def inputs_bar(k: int) -> Bar:
    """The bar `label inputs` at its end: two passes over the K complete shards."""
    return Bar("label inputs", 2 * k, 2 * k, "shard")


def label_bar(n: int, total: int) -> Bar:
    return Bar("label", n, total, "comparison")
