"""Export a selected interval of saved agent gameplay to MP4."""

import argparse
import sys
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from pickle import UnpicklingError

from mcpref.config import REPO
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.runtime.video.config import VideoConfig, load_config
from mcpref.runtime.video.encoder import destination, encode, staged_output
from mcpref.runtime.video.source import frames, snapshot


def export(cfg: VideoConfig) -> Path:
    source = REPO / cfg.sampling_dir
    output = destination(source, REPO / cfg.output)
    recording = snapshot(source)
    start, stop = recording.select(cfg.start_step, cfg.stop_step, cfg.last_steps)
    selected = frames(recording, start, stop)
    try:
        first = next(selected)
        with staged_output(output) as staged:
            encode(
                cfg, staged, chain((first,), selected), first.shape, stop - start + 1
            )
    finally:
        selected.close()
    return output


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, nargs="?", default=REPO / "configs/runtime/video.yaml"
    )
    args = parser.parse_args(argv)
    try:
        output = export(load_config(args.config))
    except (*FAILURES, UnpicklingError, EOFError) as error:
        return failed(parser.prog, error)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
