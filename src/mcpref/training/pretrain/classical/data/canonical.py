"""Canonical clip items and a fingerprint of the metadata defining them."""

import hashlib
import json

from mcpref.config import REPO


class CanonicalClips:
    """Retain compact identities and digests, reading one clip.json at a time."""

    def __init__(self) -> None:
        self.entries: list[tuple[dict, tuple[str, ...]]] = []
        self.paths: dict[str, str] = {}
        self.identities: dict[tuple, tuple[str, str]] = {}
        self.fingerprint = hashlib.sha256()

    def add(self, path: str, episode: str) -> None:
        if path in self.paths:
            if self.paths[path] != episode:
                raise RuntimeError(f"{path} has inconsistent episode references")
            return
        data = (REPO / path / "clip.json").read_bytes()
        info = json.loads(data)
        if info["relpath"] != episode:
            raise RuntimeError(f"{path} metadata disagrees with episode {episode}")
        digest = hashlib.sha256(data).hexdigest()
        self.fingerprint.update(json.dumps([path, digest]).encode())
        relevant = {
            key: value for key, value in info.items() if key not in ("id", "subset")
        }
        signature = hashlib.sha256(
            json.dumps(relevant, sort_keys=True).encode()
        ).hexdigest()
        identity = (episode, tuple(info["states"]))
        if identity in self.identities:
            first_path, first_signature = self.identities[identity]
            if first_signature != signature:
                raise RuntimeError(
                    f"{path} conflicts with clip metadata of {first_path}"
                )
        else:
            self.identities[identity] = (path, signature)
            self.entries.append(({"clip": path}, (episode,)))
        self.paths[path] = episode
