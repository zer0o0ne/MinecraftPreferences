"""Small catalogue in the YAML forms of specs/gui-screens.md (Interface); the approved one."""

from pathlib import Path

import pytest
import yaml


def blit(texture: str, u: int, v: int, w: int, h: int, x: int, y: int) -> dict:
    return {"texture": texture, "u": u, "v": v, "w": w, "h": h, "x": x, "y": y}


def element(index: int, name: str, kind: str, hit: list, click: list) -> dict:
    return {"index": index, "name": name, "kind": kind, "hit": hit, "click": click}


@pytest.fixture
def catalogue() -> dict:
    """`alpha`: book panel, abutting slots 1-2, odd-sized button 37; `beta`: no book panel."""
    first = {
        "blits": [
            blit("gui/container/inventory.png", 0, 0, 176, 166, 232, 97),
            blit("gui/widgets.png", 0, 66, 200, 20, 220, 200),
        ],
        "dynamic": [{"name": "arrow", "rect": [40, 2, 24, 17]}],
        "labels": [{"name": "Title", "rect": [8, 6, 31, 8]}],
    }
    second = {
        "blits": [],
        "dynamic": [],
        "labels": [{"name": "T", "rect": [8, 6, 5, 9]}],
    }
    alpha = {
        "members": {"first": first, "second": second},
        "description": "Alpha screen.",
        "window": [0, 0, 176, 166],
        "book_panel": [160, 97, 147, 166],
        "elements": [
            element(1, "a", "slot", [0, 0, 18, 18], [9.0, 9.0]),
            element(2, "b", "slot", [18, 0, 18, 18], [27.0, 9.0]),
            element(37, "c", "button", [5, 30, 21, 13], [15.5, 36.5]),
        ],
    }
    beta = {
        "members": {
            "only": {
                "blits": [blit("gui/book.png", 0, 0, 192, 192, 224, 2)],
                "dynamic": [],
                "labels": [],
            }
        },
        "description": "Beta screen.",
        "window": [224, 2, 192, 192],
        "elements": [
            element(37, "done", "button", [220, 200, 200, 20], [320.0, 210.0])
        ],
    }
    textures = {
        "url": "http://127.0.0.1:9/client.jar",
        "sha1": "0" * 40,
        "out_dir": "data/textures",
        "http_timeout": 2.0,
        "http_retries": 2,
        "retry_delay": 0.05,
    }
    return {"classes": {"alpha": alpha, "beta": beta}, "textures": textures}


@pytest.fixture(scope="session")
def classes(repo) -> dict:
    """`classes` of the approved configs/gui-screens.yaml as plain YAML; read-only."""
    text = (repo / "configs/gui-screens.yaml").read_text()
    return yaml.load(text, Loader=yaml.CSafeLoader)["classes"]


@pytest.fixture
def write_config(tmp_path):
    """Writes a catalogue dict as YAML; returns its path."""

    def write(data: dict) -> Path:
        path = tmp_path / "gui-screens.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return write
