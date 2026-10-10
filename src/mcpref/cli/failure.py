"""Exit 1 with a one-line message on stderr: `main` of specs/gui-screens.md (textures),
specs/vpt-gui-frames.md, specs/gui-labels.md, specs/gui-masks.md."""

import sys

import httpx
import yaml

FAILURES = (OSError, ValueError, KeyError, TypeError, RuntimeError)
FAILURES += (yaml.YAMLError, httpx.HTTPError)


def failed(prog: str, error: Exception) -> int:
    """`prog: <type>: <message>` on one stderr line; exit code 1."""
    cause = " ".join(f"{type(error).__name__}: {error}".split())
    print(f"{prog}: {cause}", file=sys.stderr)
    return 1
