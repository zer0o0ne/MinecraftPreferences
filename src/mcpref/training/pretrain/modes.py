"""Select a pretraining mode and its matching index and training configs."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import REPO, config_mapping, read_yaml, relative


@dataclass(frozen=True)
class ModePaths:
    config: str
    index: str


@dataclass(frozen=True)
class Selection:
    mode: str
    classical: ModePaths
    continuous: ModePaths

    @property
    def selected(self) -> ModePaths:
        return getattr(self, self.mode)


def load_selection(path: Path) -> Selection:
    raw = config_mapping(read_yaml(path), path, Selection)
    if raw["mode"] not in ("classical", "continuous"):
        raise ValueError("pretrain mode must be classical or continuous")
    modes = {}
    for mode in ("classical", "continuous"):
        fields = config_mapping(raw[mode], path, ModePaths)
        for key, value in fields.items():
            relative(f"{mode}.{key}", value)
        modes[mode] = ModePaths(**fields)
    return Selection(raw["mode"], **modes)


def classical_config_path(path: Path) -> Path:
    """Existing clip-based consumers reject continuous stream semantics."""
    raw = read_yaml(path)
    if isinstance(raw, dict) and "mode" in raw:
        selection = load_selection(path)
        if selection.mode != "classical":
            raise RuntimeError(
                "This consumer requires classical pretraining; continuous artifacts "
                "require the continuous perception stream adapter"
            )
        return REPO / selection.selected.config
    return path


def load_classical_config(path: Path):
    from mcpref.training.pretrain.classical.config import load_config

    return load_config(classical_config_path(path))
