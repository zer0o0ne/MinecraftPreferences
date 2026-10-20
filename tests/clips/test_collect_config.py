"""`load_config` and the starting values of specs/perception-data.md (Notation, Interface),
with the field `classifier_config` of its revision of 2026-10-20."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from mcpref.clips.collect import PerceptionDataConfig, load_config

from .collect_site import FIELDS

PATHS = {"clips": "configs/clips.yaml", "screens": "configs/gui-screens.yaml",
         "gui_actions": "configs/vpt-gui-actions.yaml", "overlay": "configs/f3-overlay.yaml",
         "classifier_config": "configs/gui-classifier.yaml", "out_dir": "data/elsewhere"}  # fmt: skip
VALID = FIELDS | PATHS | {"index_urls": {"7xx": "http://h/i7", "6xx": "http://h/i6"},
                          "ffmpeg": "/x/ffmpeg"}  # fmt: skip
INTEGERS = ("width", "height", "fetch_streams", "http_retries", "states", "per_episode",
            "n_clips", "seed", "workers", "classifier_batch")  # fmt: skip
DURATIONS = ("http_timeout", "retry_delay", "ffmpeg_timeout")
ACCESS = ("index_urls", "x_ms_version", "min_version", "width", "height", "fetch_streams",
          "http_timeout", "http_retries", "retry_delay", "ffmpeg")  # fmt: skip
BINDINGS = {  # Domain of specs/vpt-actions.md and specs/vpt-gui-actions.md
    "key_key.forward": "key.keyboard.w", "key_key.back": "key.keyboard.s",
    "key_key.left": "key.keyboard.a", "key_key.right": "key.keyboard.d",
    "key_key.jump": "key.keyboard.space", "key_key.inventory": "key.keyboard.e",
    "key_key.drop": "key.keyboard.q", "key_key.swapOffhand": "key.keyboard.f",
    "key_key.togglePerspective": "key.keyboard.f5",
    "key_key.sneak": "key.keyboard.left.shift", "key_key.attack": "key.mouse.left",
    "key_key.use": "key.mouse.right", "key_key.pickItem": "key.mouse.middle",
} | {f"key_key.hotbar.{i}": f"key.keyboard.{i}" for i in range(1, 10)}  # fmt: skip
STARTING = {
    "ffmpeg_timeout": 600, "states": 16, "per_episode": 10, "n_clips": 10000, "seed": 0,
    "workers": 4, "classifier_batch": 64,
}  # fmt: skip


def write(tmp_path: Path, **changes) -> Path:
    path = tmp_path / "perception-data.yaml"
    path.write_text(yaml.safe_dump(VALID | changes, sort_keys=False))
    return path


def test_load_config_reads_every_field(tmp_path, repo):
    cfg = load_config(write(tmp_path))
    assert isinstance(cfg, PerceptionDataConfig)
    assert {f.name for f in dataclasses.fields(cfg)} == set(VALID)
    assert {k: getattr(cfg, k) for k in FIELDS} == FIELDS
    assert cfg.index_urls == VALID["index_urls"] and cfg.ffmpeg == "/x/ffmpeg"
    assert list(cfg.index_urls) == ["7xx", "6xx"]  # subsets in config order
    for name, rel in PATHS.items():  # resolved against the repository root when used
        assert Path(repo, getattr(cfg, name)).resolve() == (repo / rel).resolve()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


@pytest.mark.parametrize("name", INTEGERS)
@pytest.mark.parametrize("value", [True, False, 3.0, "3", None])
def test_integers_must_be_yaml_integers(tmp_path, name, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, **{name: value}))


@pytest.mark.parametrize("name", DURATIONS)
@pytest.mark.parametrize("value", [True, "2", None])
def test_durations_must_be_yaml_numbers(tmp_path, name, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, **{name: value}))


@pytest.mark.parametrize("name", DURATIONS)
@pytest.mark.parametrize("value", [2, 2.5])
def test_durations_may_be_integers_or_floats(tmp_path, name, value):
    assert getattr(load_config(write(tmp_path, **{name: value})), name) == value


@pytest.mark.parametrize(
    "name, value",
    [("fetch_streams", 0), ("fetch_streams", 21), ("states", 1), ("states", 0),
     ("per_episode", 0), ("n_clips", 0), ("seed", -1), ("seed", 2**32), ("workers", 0),
     ("classifier_batch", 0)],
)  # fmt: skip
def test_out_of_range_values_are_rejected(tmp_path, name, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, **{name: value}))


@pytest.mark.parametrize(
    "name, value",
    [("fetch_streams", 1), ("fetch_streams", 20), ("states", 2), ("per_episode", 1),
     ("n_clips", 1), ("seed", 0), ("seed", 2**32 - 1), ("workers", 1),
     ("classifier_batch", 1)],
)  # fmt: skip
def test_boundary_values_are_accepted(tmp_path, name, value):
    assert getattr(load_config(write(tmp_path, **{name: value})), name) == value


def test_approved_config_has_starting_values(repo):
    cfg = load_config(repo / "configs/perception-data.yaml")
    frames = yaml.safe_load((repo / "configs/vpt-gui-frames.yaml").read_text())
    assert {k: getattr(cfg, k) for k in ACCESS} == {k: frames[k] for k in ACCESS}
    extended = {"autoJump": False, "toggleCrouch": False, "touchscreen": False}
    assert cfg.required_options == frames["required_options"] | extended | BINDINGS
    assert {k: getattr(cfg, k) for k in STARTING} == STARTING
    for name, rel in (PATHS | {"out_dir": "data/perception"}).items():
        assert Path(repo, getattr(cfg, name)).resolve() == (repo / rel).resolve()


@pytest.mark.parametrize(
    "name, value",
    [("index_urls", {}), ("index_urls", ["http://h/i6"]), ("index_urls", {"6xx": 1}),
     ("index_urls", {"6xx": None}), ("index_urls", {6: "http://h/i6"}), ("index_urls", "x"),
     ("required_options", ["guiScale"]), ("required_options", {"guiScale": [2]}),
     ("required_options", {"guiScale": {"v": 2}}), ("required_options", {2: 2}),
     ("x_ms_version", 20191212), ("min_version", 6.12), ("min_version", None),
     ("ffmpeg", None), ("clips", 1), ("screens", None), ("gui_actions", ["a"]),
     ("overlay", 2), ("out_dir", None), ("classifier_config", 1),
     ("classifier_config", None), ("classifier_config", ["configs/gui-classifier.yaml"]),
     ("classifier_config", {"path": "configs/gui-classifier.yaml"})],
)  # fmt: skip
def test_strings_and_mappings_are_checked(tmp_path, name, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, **{name: value}))


def test_required_options_take_every_json_scalar(tmp_path):
    scalars = {"a": "x", "b": 2, "c": 2.5, "d": True, "e": None}
    assert (
        load_config(write(tmp_path, required_options=scalars)).required_options
        == scalars
    )


def test_classifier_config_follows_overlay(tmp_path):
    """Notation: `classifier_config` is the field after `overlay` (the key order of run.json
    (8)); its value the string as written."""
    cfg = load_config(write(tmp_path, classifier_config="cfg/other-classifier.yaml"))
    names = [f.name for f in dataclasses.fields(cfg)]
    assert names[names.index("overlay") + 1] == "classifier_config"
    assert cfg.classifier_config == "cfg/other-classifier.yaml"


def test_classifier_config_is_required(tmp_path):
    path = tmp_path / "perception-data.yaml"
    doc = {k: v for k, v in VALID.items() if k != "classifier_config"}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError):
        load_config(path)
