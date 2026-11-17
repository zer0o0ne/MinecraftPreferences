"""Publication, encoder failures, and the public CLI."""

import dataclasses
import io

import pytest
import yaml

from mcpref.runtime.video import encoder, export


def configured(cfg, source, output, *, start=None, stop=None, last=None):
    return dataclasses.replace(
        cfg,
        sampling_dir=str(source),
        output=str(output),
        start_step=start,
        stop_step=stop,
        last_steps=last,
    )


def test_export_and_latest_frames(source, tmp_path, cfg, monkeypatch):
    out = tmp_path / "video/clip.mp4"
    observed = []

    def encode(config, path, frames, shape, total):
        assert not out.exists()
        assert shape == (24, 32, 3) and total == 5
        observed.extend(int(frame[0, 0, 0]) for frame in frames)
        path.write_bytes(b"encoded")

    monkeypatch.setattr(export, "encode", encode)
    assert export.export(configured(cfg, source, out, last=4)) == out
    assert observed == [40, 60, 80, 100, 120]
    assert out.read_bytes() == b"encoded"
    assert list(out.parent.iterdir()) == [out]


@pytest.mark.parametrize("where", ["source", "shards", "staged", "alias"])
def test_output_cannot_modify_source(source, tmp_path, cfg, where):
    if where == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias / "clip.mp4"
    else:
        parent = {
            "source": source,
            "shards": source / "shards",
            "staged": source / "shards/000002.part",
        }[where]
        output = parent / "clip.mp4"
    with pytest.raises(ValueError, match="outside"):
        export.export(configured(cfg, source, output, start=0, stop=1))
    assert not output.exists()


@pytest.mark.parametrize("late", [False, True])
def test_existing_output_never_overwritten(source, tmp_path, cfg, monkeypatch, late):
    output = tmp_path / "clip.mp4"

    def encode(*args):
        args[1].write_bytes(b"new")
        output.write_bytes(b"other writer")

    monkeypatch.setattr(export, "encode", encode)
    if not late:
        output.write_bytes(b"other writer")
    with pytest.raises(FileExistsError):
        export.export(configured(cfg, source, output, start=0, stop=1))
    assert output.read_bytes() == b"other writer"
    assert not list(tmp_path.glob("*.part"))


class Process:
    def __init__(self, *, stdin=None, status=0):
        self.stdin = stdin if stdin is not None else io.BytesIO()
        self.returncode = None
        self.status, self.killed, self.waited = status, False, False

    def wait(self):
        self.waited = True
        self.returncode = self.status
        return self.status

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.status = -9


@pytest.mark.parametrize("failure", ["start", "exit", "data", "pipe"])
def test_failure_cleans_output_and_process(source, tmp_path, cfg, monkeypatch, failure):
    process = Process(status=7 if failure == "exit" else 0)

    def popen(cmd, **kwargs):
        if failure == "start":
            raise FileNotFoundError("missing encoder")
        kwargs["stderr"].write(b"encoder test error")
        if failure == "data":
            (source / "shards/000001/steps/000001.pt").unlink()
        return process

    if failure == "pipe":

        class Broken(io.BytesIO):
            def write(self, value):
                raise BrokenPipeError()

        process.stdin = Broken()
    monkeypatch.setattr(encoder.subprocess, "Popen", popen)
    output = tmp_path / "clip.mp4"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        export.export(configured(cfg, source, output, start=3, stop=6))
    assert not output.exists() and not list(tmp_path.glob("*.part"))
    if failure != "start":
        assert process.waited and process.stdin.closed
        assert process.killed == (failure == "data")


def test_cli_failure_is_one_line(source, tmp_path, cfg, capsys):
    config = tmp_path / "cfg.yaml"
    cfg = configured(cfg, source, tmp_path / "clip.mp4", last=7)
    config.write_text(yaml.safe_dump(dataclasses.asdict(cfg) | {"fps": 10}))
    code = export.main([str(config)])
    captured = capsys.readouterr()
    assert code == 1 and captured.out == ""
    assert len(captured.err.splitlines()) == 1 and "[1, 6]" in captured.err


@pytest.mark.parametrize(
    "selection,values",
    [
        ({"start_step": 1, "stop_step": 4, "last_steps": None}, [20, 40, 60, 80]),
        ({"start_step": None, "stop_step": None, "last_steps": 2}, [80, 100, 120]),
    ],
)
def test_default_cli_uses_only_config_and_repo_relative_paths(
    source,
    tmp_path,
    cfg,
    monkeypatch,
    selection,
    values,
):
    monkeypatch.setattr(export, "REPO", tmp_path)
    config = tmp_path / "configs/runtime/video.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            dataclasses.asdict(cfg)
            | selection
            | {
                "sampling_dir": "sampling",
                "output": "videos/clip.mp4",
                "fps": 7.5,
            }
        )
    )
    seen = []

    def encode(config, path, frames, shape, total):
        assert config.fps == 7.5
        seen.extend(int(frame[0, 0, 0]) for frame in frames)
        path.write_bytes(b"video")

    monkeypatch.setattr(export, "encode", encode)
    monkeypatch.chdir(source)
    assert export.main([]) == 0
    assert seen == values
    assert (tmp_path / "videos/clip.mp4").read_bytes() == b"video"
