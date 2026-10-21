"""Runs of `pretrain` of specs/pretrain.md on the worlds of loop_world.py: the config file
and the inputs under a temporary repository root (`mcpref.REPO` patched as in index_run.py),
the working directory one from which the config's paths lead nowhere; readers of
`{out_dir}`."""

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from progress_bars import tqdm_defaults

from mcpref.pretrain.config import load_config
from mcpref.pretrain.loop import pretrain

from .index_oracle import name
from .index_run import repo_root
from .loop_modules import fakes
from .loop_world import CONFIG, NO_PREF, WORLD, World, config_text, write_world


@dataclass
class Outcome:
    err: str
    out: str
    error: BaseException | None


class Loop:
    """tmp/repo — the repository root (inputs, X = w/index, out_dir = w/pretrain);
    tmp/cwd — the working directory; tmp/pretrain.yaml — the config."""

    def __init__(self, tmp: Path, world: World = WORLD, **changes):
        self.tmp, self.world = tmp, world
        self.config = (CONFIG if world.n_r else NO_PREF) | changes
        self.repo, self.cwd = tmp / "repo", tmp / "cwd"
        self.path = tmp / "pretrain.yaml"
        self.out = self.repo / self.config["out_dir"]

    def write(self) -> "Loop":
        self.cwd.mkdir(parents=True, exist_ok=True)
        write_world(self.repo, self.world)
        self.path.write_text(config_text(self.config))
        return self

    def modules(self) -> dict:
        """Fresh components, as a caller builds them."""
        return fakes(self.world.n_r > 0)

    def run(self, modules: dict | None = None, patch=None, raises=None) -> Outcome:
        """pretrain on the config, tqdm with its defaults; `raises`: the exception expected."""
        modules = self.modules() if modules is None else modules
        err, out, error = io.StringIO(), io.StringIO(), None
        with pytest.MonkeyPatch.context() as mp, repo_root(self.repo):
            tqdm_defaults(mp)
            mp.chdir(self.cwd)
            if patch:
                patch(mp)
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                cfg = load_config(self.path)
                if raises is None:
                    pretrain(cfg, self.path, modules)
                else:
                    with pytest.raises(raises) as info:
                        pretrain(cfg, self.path, modules)
                    error = info.value
        return Outcome(err.getvalue(), out.getvalue(), error)

    def checkpoint_dir(self, j: int) -> Path:
        return self.out / "checkpoints" / name(j)

    def checkpoint(self, j: int) -> dict:
        """checkpoint.pt of checkpoint j, by torch.load with its defaults."""
        return torch.load(self.checkpoint_dir(j) / "checkpoint.pt")

    def losses(self, j: int) -> list:
        return json.loads((self.checkpoint_dir(j) / "losses.json").read_bytes())

    def model(self) -> dict:
        return torch.load(self.out / "model.pt")

    def tree(self) -> dict:
        """Every file under {out_dir}: path relative to it -> bytes."""
        files = sorted(p for p in self.out.rglob("*") if p.is_file())
        return {p.relative_to(self.out).as_posix(): p.read_bytes() for p in files}


def aborted(o: Outcome, message: str) -> None:
    """The run raised a RuntimeError with `message`; nothing on stdout."""
    assert type(o.error) is RuntimeError and str(o.error) == message, o.error
    assert o.out == ""
