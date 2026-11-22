"""Fragment scheduling, exact continuation and the optional frozen-phi phase."""

import shutil
import sys
from dataclasses import replace
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcpref.config import REPO, config_sha256, read_yaml
from mcpref.infrastructure.profiling.runtime import TrainingProfile
from mcpref.infrastructure.profiling.scopes import phase as profile_phase
from mcpref.infrastructure.runs.files import check_fingerprints, sha256, write_json
from mcpref.infrastructure.shards.store import part_path, publish
from mcpref.models.policy.targets import ALIGNMENT
from mcpref.training.groups import check_groups
from mcpref.training.optim import adamw
from mcpref.training.pretrain.checkpoints import check_alignment
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.checkpoints import restore, save, write_model
from mcpref.training.pretrain.continuous.data.reader import Index, StreamReader
from mcpref.training.pretrain.continuous.events import Context, notify
from mcpref.training.pretrain.continuous.state import Stream
from mcpref.training.pretrain.continuous.steps import window
from mcpref.training.pretrain.phases import head_groups, load_joint_model


def initial_position() -> dict:
    return {
        "step": 0,
        "epoch": 1,
        "order": [],
        "fragment_index": 0,
        "observation_index": 0,
        "schedule_rng": None,
        "epoch_counts": dict.fromkeys(("phi", "ce", "shift", "P"), 0),
    }


def archive_for(context, epoch):
    return Archive(
        context.root / "archive" / f"epoch-{epoch}" / "shards",
        context.cfg.archive_shard_size,
        context.cfg.archive_cache_size,
    )


def finish_epoch(position, population, archive):
    expected = {
        "phi": population["N_phi"],
        "ce": population["N_pi"],
        "shift": population["N_pi"],
        "P": population["N_P"],
    }
    if position["epoch_counts"] != expected or archive.live:
        raise RuntimeError(
            f"Continuous epoch coverage differs: {position['epoch_counts']} != {expected}"
        )
    if len(archive.consumed) != population["N_P"]:
        raise RuntimeError("Continuous epoch did not consume every preference edge")


def train(context, hooks=None):
    cfg, population = context.cfg, context.index.run["populations"]["train"]
    fragments = {item["id"]: item for item in context.index.fragments("train")}
    reader = StreamReader(
        context.index, context.modules["phi"].cfg, context.modules["phi"].sprite
    )
    recovered = restore(context)
    position = recovered["position"] if recovered else initial_position()
    stream = recovered["stream"] if recovered else Stream()
    archive = archive_for(context, position["epoch"])
    try:
        if recovered:
            archive.restore(recovered["archive"])
        else:
            archive.restore({"next_shard": 0, "live": {}, "seen": [], "consumed": []})
        generator = np.random.Generator(np.random.PCG64(cfg.seed))
        if position["schedule_rng"] is not None:
            generator.bit_generator.state = position["schedule_rng"]
        context.step, context.epoch = position["step"], position["epoch"]
        notify(hooks, "start", context)
        results = []
        with (
            TrainingProfile(
                cfg, context.root, context.modules, f"continuous-{context.phase}"
            ) as profile,
            tqdm(
                total=context.total,
                initial=context.step,
                desc=f"continuous {context.phase}",
                unit="step",
                file=sys.stderr,
            ) as bar,
        ):
            while position["epoch"] <= cfg.epochs:
                context.epoch = position["epoch"]
                if not position["order"]:
                    position["order"] = [
                        str(key) for key in generator.permutation(list(fragments))
                    ]
                    position["schedule_rng"] = generator.bit_generator.state
                while position["fragment_index"] < len(position["order"]):
                    descriptor = fragments[
                        position["order"][position["fragment_index"]]
                    ]
                    context.fragment = descriptor
                    source = reader.observations(
                        descriptor, position["observation_index"]
                    )
                    while position["observation_index"] < descriptor["observations"]:
                        size = min(
                            cfg.tbptt_states,
                            descriptor["observations"] - position["observation_index"],
                        )
                        with profile.step(context.step + 1, size):
                            with profile_phase("data.stream_read"):
                                batch = list(islice(source, size))
                            if len(batch) != size:
                                raise RuntimeError(
                                    "Continuous fragment ended before its indexed cursor"
                                )
                            context.step += 1
                            stream, result = window(
                                context, stream, archive, batch, population, hooks
                            )
                            position["step"] = context.step
                            position["observation_index"] += len(batch)
                            for key, value in result.counts.items():
                                position["epoch_counts"][key] += value
                            results.append((context.step, result))
                            if (
                                context.step % cfg.checkpoint_steps == 0
                                or context.step == context.total
                            ):
                                save(context, position, stream, archive, results, hooks)
                                results = []
                        bar.update()
                    if stream.pending is not None:
                        raise RuntimeError(
                            "Unfinished policy target at a fragment boundary"
                        )
                    stream = Stream()
                    position["observation_index"] = 0
                    position["fragment_index"] += 1
                finish_epoch(position, population, archive)
                position["epoch"] += 1
                position["order"], position["fragment_index"] = [], 0
                position["epoch_counts"] = dict.fromkeys(position["epoch_counts"], 0)
                if position["epoch"] <= cfg.epochs:
                    archive.close()
                    archive = archive_for(context, position["epoch"])
                    archive.restore(
                        {"next_shard": 0, "live": {}, "seen": [], "consumed": []}
                    )
        notify(hooks, "finish", context)
        write_model(context)
    finally:
        archive.close()


def phase(cfg, modules, index, root, name, hooks):
    if "P" not in modules:
        cfg = replace(
            cfg,
            groups={
                key: value
                for key, value in cfg.groups.items()
                if key not in ("P", "P_no_decay")
            },
        )
    groups = (
        head_groups(cfg, modules) if name == "retrain" else check_groups(cfg, modules)
    )
    for key, module in modules.items():
        module.train()
        module.requires_grad_(not (name == "retrain" and key in ("phi", "phi_loss")))
    torch.manual_seed(cfg.seed)
    root.mkdir(parents=True, exist_ok=True)
    context = Context(
        cfg,
        modules,
        index,
        root,
        groups,
        adamw(cfg, groups),
        index.run["populations"]["train"]["M"] * cfg.epochs,
        name,
    )
    if not (root / "model.pt").exists():
        train(context, hooks)
    else:
        load_joint_model(root / "model.pt", modules)
        print(f"continuous {name}: already done", file=sys.stderr)


def pretrain(cfg, path: Path, modules: dict, index=None, hooks=None):
    owned = index is None
    index = Index(REPO / cfg.index_dir) if owned else index
    try:
        index.validate()
        expected = index.run["mathematics"]
        if cfg.tbptt_states != expected["tbptt_states"]:
            raise ValueError("Continuous training K differs from its index")
        history = read_yaml(REPO / cfg.preference)["history"]
        if history != expected["preference_history"]:
            raise ValueError("Continuous preference history differs from its index")
        if config_sha256(REPO / cfg.action_mask) != config_sha256(
            REPO / expected["action_mask"]
        ):
            raise ValueError("Continuous action masks differ from its index")
        population = index.run["populations"]["train"]
        if not population["M"]:
            raise RuntimeError("Continuous index has no training observations")
        root = REPO / cfg.out_dir
        run_path = root / "run.json"
        check_alignment(run_path)
        fingerprints = {
            "config": config_sha256(path),
            "index": sha256(REPO / cfg.index_dir / "run.json"),
        }
        fingerprints |= {
            key: config_sha256(REPO / getattr(cfg, key))
            for key in (
                "perception",
                "action_mask",
                "policy",
                "preference",
                "diagnostics",
            )
        }
        check_fingerprints(root / "run.json", fingerprints, cfg.out_dir)
        header = {
            "pretrain_mode": "continuous",
            "policy_alignment": ALIGNMENT,
            "attention": "observation",
            "memory": "fragment",
            "sha256": fingerprints,
        }
        write_json(root / "run.json", header)
        joint = root / "joint-model.pt"
        if joint.exists():
            load_joint_model(joint, modules)
        else:
            phase(cfg, modules, index, root, "joint", hooks)
        if cfg.retrain:
            if not joint.exists():
                staged = part_path(joint)
                shutil.copyfile(root / "model.pt", staged)
                publish(staged, joint)
            phase(cfg, modules, index, root / "retrain", "retrain", hooks)
            staged = part_path(root / "model.pt")
            shutil.copyfile(root / "retrain" / "model.pt", staged)
            publish(staged, root / "model.pt")
        write_json(root / "run.json", header | {"complete": True})
    finally:
        if owned:
            index.close()
