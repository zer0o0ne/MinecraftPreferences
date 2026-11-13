"""Complete test-owned repositories for runtime and real-runtime scenarios."""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from classifier_world import Masks, both
from datasets.clips.collect_site import FIELDS as PERCEPTION_FIELDS
from gui.recognition.config_files import OVERLAY_STARTING
from models.perception.world import BASE as PERCEPTION_BASE
from models.perception.world import STARTING as PERCEPTION_STARTING
from models.perception.world import write_config as write_perception
from runtime.engine.engine_world import STARTING as ENGINE_STARTING
from training.pretrain.loop_configs import action_mask_catalogue
from training.pretrain.loop_starting import STARTING_GROUPS

MCTS_STARTING = {
    "simulations": 24, "min_n_sampling": 4, "depth": 8, "width": 8, "latency_calls": 10,
    "c_puct": 1.25, "discount": 0.99, "temperature": 1.0, "min_p": 0.5,
    "epsilon": 0.05, "latency_ms": 150.0, "latency_fail_ms": 300.0,
}  # fmt: skip
RL_PAIRS_STARTING = {
    "history_dir": "data/rl_preference_history", "out_dir": "data/rl_preference_pairs",
    "sample_states": 32, "temporal_per_state": 4, "nearest_per_state": 4, "branch": 4,
    "beam_width": 6, "beam_depth": 3, "capacity": 10000, "items_per_shard": 256,
    "history_per_shard": 256, "base_threshold": 0.5, "temporal_penalty": 1.0,
    "threshold_scale": 0.95,
}  # fmt: skip


def write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def write_repository(
    root: Path,
    *,
    steps=2,
    horizon=2,
    pairs=2,
    states=3,
    alpha=None,
    delta=6,
    mcts=None,
) -> None:
    """Write every config consumed by the runtime fixtures, with scenario overrides explicit."""
    alpha = [1.5, 0.5] if alpha is None else alpha
    paths = {
        "engine": "configs/runtime/engine.yaml",
        "gui-classifier": "configs/gui/recognition/classifier.yaml",
        "action-mask": "configs/gui/action-mask.yaml",
        "clips": "configs/datasets/clips.yaml",
        "perception": "configs/models/perception.yaml",
        "policy": "configs/models/policy.yaml",
        "preference": "configs/models/preference.yaml",
        "perception-data": "configs/datasets/perception-data.yaml",
        "pretrain": "configs/training/pretrain/config.yaml",
        "env": "configs/runtime/env.yaml",
        "mcts": "configs/runtime/mcts.yaml",
        "env-sampling": "configs/runtime/sampling.yaml",
        "ppo": "configs/training/rl/ppo.yaml",
        "rl-preference-pairs": "configs/training/rl/preference-pairs.yaml",
    }
    docs = {
        "action-mask": {"screens": "configs/gui/screens.yaml", "pitch_limit": 90},
        "gui-screens": action_mask_catalogue(),
        "vpt-actions": {"turn_deg": 10, "r_min": 3},
        "vpt-gui-actions": {"double_click_ms": 250},
        "f3-overlay": OVERLAY_STARTING,
        "gui-classifier": {"masks_dir": "data/gui_masks", "criterion": "agreement"},
        "engine": ENGINE_STARTING
        | {"minestudio_dir": "ms", "render_size": [640, 360], "seed": 7},
        "env": {
            "engine": paths["engine"],
            "classifier": paths["gui-classifier"],
            "action_mask": paths["action-mask"],
            "clips": paths["clips"],
            "cursor_scale": 20 / 3,
        },
        "clips": {"delta": delta, "actions": "configs/datasets/vpt/actions.yaml"},
        "mcts": MCTS_STARTING | (mcts or {}),
        "ppo": {
            "horizon": horizon,
            "clip": 0.2,
            "skill_eps": 1e-8,
            "dual_init": 30.0,
            "dual_slack": 1e-3,
        },
    }
    docs["policy"] = {
        "history": states, "width": 256, "heads": 4, "hidden": 768, "layers": 4,
        "shift_hidden": 512, "rope_base": 10000.0, "norm_eps": 1e-6,
        "action_dropout": 0.3, "shift_weight": 1.0, "seed": 0, "latency_ms": 3.0,
        "latency_fail_ms": 10.0, "latency_calls": 100,
    }  # fmt: skip
    docs["preference"] = {
        "history": min(12, states), "width": 256, "heads": 4, "hidden": 768,
        "layers": ["S", "X", "S", "X"], "rope_base": 10000.0, "norm_eps": 1e-6,
        "seed": 0, "latency_ms": 3.0, "latency_fail_ms": 10.0, "latency_calls": 100,
    }  # fmt: skip
    docs["rl-preference-pairs"] = RL_PAIRS_STARTING | {
        "sample_states": pairs, "temporal_per_state": 1, "nearest_per_state": 1,
        "branch": 1, "beam_width": 1, "beam_depth": 1, "items_per_shard": 2,
        "history_per_shard": 2,
    }  # fmt: skip
    docs["perception-data"] = PERCEPTION_FIELDS | {
        "index_urls": {"tiny": "http://127.0.0.1:9/index.json"},
        "ffmpeg": "/x/ffmpeg", "clips": "configs/datasets/clips.yaml",
        "screens": "configs/gui/screens.yaml", "gui_actions": "configs/datasets/vpt/gui-actions.yaml",
        "overlay": "configs/gui/recognition/f3-overlay.yaml", "classifier_config": "configs/gui/recognition/classifier.yaml",
        "states": states, "out_dir": "data/perception",
    }  # fmt: skip
    docs["diagnostics"] = {
        "out_dir": "data/diagnostics",
        "seed": 0,
        "bootstrap_samples": 20,
        "ci_level": 0.95,
        "min_cluster_count": 2,
        "quantiles": [0.05, 0.5, 0.95],
    }
    docs["pretrain"] = {
        "index_dir": "data/train_index", "out_dir": "data/pretrain-scaled-init",
        "perception": paths["perception"], "action_mask": paths["action-mask"],
        "policy": paths["policy"], "preference": paths["preference"],
        "diagnostics": "configs/diagnostics.yaml", "seed": 0,
        "epochs": 1, "retrain": False, "batch": 32, "micro_batch": 1, "diagnostic_window": 100,
        "gradient_diagnostics": True,
        "reference_steps": 500, "reference_pairs": 64, "calibration_bins": 10,
        "saturation_threshold": 0.01, "inc_classification_threshold": 0.5,
        "quantiles": [0.05, 0.5, 0.95], "action_copy_eps": 0.1,
        "latent_init_scale": 1.0,
        "checkpoint_steps": 100, "keep_checkpoints": 2, "warmup_steps": 100,
        "w_phi": 1.0, "w_pi": 1.0, "w_P": 1.0, "grad_clip": 1.0, "adam_eps": 1e-8,
        "lr_floor": 0.1, "betas": [0.9, 0.999], "groups": STARTING_GROUPS, "device": "cpu",
    }  # fmt: skip
    docs["env-sampling"] = {
        "pretrain": paths["pretrain"], "env": paths["env"], "mcts": paths["mcts"],
        "perception_data": paths["perception-data"], "out_dir": "data/sampling",
        "steps": steps, "seed": 7, "safety_limit": 25.0,
    }  # fmt: skip
    destinations = paths | {
        "gui-screens": "configs/gui/screens.yaml",
        "vpt-actions": "configs/datasets/vpt/actions.yaml",
        "vpt-gui-actions": "configs/datasets/vpt/gui-actions.yaml",
        "f3-overlay": "configs/gui/recognition/f3-overlay.yaml",
        "diagnostics": "configs/diagnostics.yaml",
    }
    for name, doc in docs.items():
        write(root / destinations[name], doc)
    perception = write_perception(
        root / "perception",
        base=PERCEPTION_STARTING,
        alpha=alpha,
        cursor_url=PERCEPTION_BASE["cursor_url"],
        cursor_sha256=PERCEPTION_BASE["cursor_sha256"],
    )
    doc = yaml.safe_load(perception.read_text())
    doc["cursor"] = "perception/sprite/cursor.png"
    write(perception, doc)
    write(root / paths["perception"], doc)


def write_classifier_masks(root: Path) -> None:
    """A complete synthetic classifier output matching the action-mask catalogue."""
    shape = (360, 640)
    pair = np.zeros((*shape, 3), dtype=np.uint8), np.ones(shape, dtype=bool)
    classes = {"inventory": {"only": pair}, "death": {"only": pair}}
    masks = Masks(root / "classifier", root, classes, both(2, "0.5", "0.1"), 640, 360)
    masks.write()
    text = yaml.safe_dump(action_mask_catalogue(), sort_keys=False)
    masks.catalogue.write_text(text)
    run_path = masks.out / "run.json"
    run = json.loads(run_path.read_text())
    run["sha256"]["screens"] = hashlib.sha256(text.encode()).hexdigest()
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    write(
        root / "configs/gui/recognition/classifier.yaml",
        {"masks_dir": masks.masks_dir, "criterion": "agreement"},
    )
    write(
        root / "configs/gui/action-mask.yaml",
        {"screens": masks.screens, "pitch_limit": 90},
    )
