"""Test-owned starting mappings for the standalone recognition config fixtures."""

from pathlib import Path

import yaml

OVERLAY_STARTING = {
    "screens": "configs/gui-screens.yaml",
    "prefix": "Minecraft 1.16.5 (",
    "text_luma": {"normal": 208.38, "dimmed": 73.7},
    "tolerance": {"normal": 24, "dimmed": 8},
    "box_max": {"normal": 160.1, "dimmed": 63.4},
    "threshold": 0.75,
}
MASKS_STARTING = {
    "frames_dir": "data/gui_frames",
    "labels_dir": "data/gui_labels",
    "screens": "configs/gui-screens.yaml",
    "out_dir": "data/gui_masks",
    "seed": 0,
    "min_episodes": 10,
    "spread_quantile": 0.9,
    "spread_max": 24,
    "patch": 8,
    "patches_per_class": 64,
    "eps": 1,
    "chunk_rows": 40,
    "folds": 5,
    "fold_seed": 1,
    "feature_quantile": 0.75,
    "tau_grid": [8, 16, 24, 32, 48],
    "theta_grid": ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95"],
    "mu_grid": ["0", "0.05", "0.1", "0.2"],
    "split": ["0.6", "0.2", "0.2"],
}


def write(path: Path, starting: dict, **changes) -> Path:
    path.write_text(yaml.safe_dump(starting | changes, sort_keys=False))
    return path
