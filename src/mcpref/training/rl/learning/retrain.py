"""Frozen-phi RL retraining with shard-fixed PPO advantages."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from mcpref.config import REPO
from mcpref.datasets.clips.collect import load_config as load_perception_data_config
from mcpref.datasets.vpt.plan import generator
from mcpref.infrastructure.profiling.runtime import TrainingProfile
from mcpref.infrastructure.profiling.scopes import phase
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import complete_shards
from mcpref.models.perception.config import load_config as load_perception_config
from mcpref.runtime.mcts.config import load_config as load_mcts_config
from mcpref.runtime.sampling.config import load_config as load_sampling_config
from mcpref.training.checkpoints import prune, write_model
from mcpref.training.groups import check_groups
from mcpref.training.optim import adamw, update
from mcpref.training.pretrain.config import load_config as load_pretrain_config
from mcpref.training.rl.config import RLCycleConfig
from mcpref.training.rl.learning.checkpoints import STATEFUL, restore, write_checkpoint
from mcpref.training.rl.learning.history import encode_ref
from mcpref.training.rl.learning.loop import START, shard_sizes
from mcpref.training.rl.learning.loss import (
    ROLES,
    ppo_eligible,
    state_ref,
    step_gradients,
)
from mcpref.training.rl.pairs.files import read_json_shard, write_json_shard
from mcpref.training.rl.ppo.config import load_config as load_ppo_config
from mcpref.training.rl.ppo.reward import reward
from mcpref.training.schedule import remaining


def fixed_advantages(root: Path, shard: int) -> dict[tuple[str, int], float]:
    rows = read_json_shard(root / "retrain/advantages/shards", shard, "items.json")
    return {(row["kind"], row["index"]): row["advantage"] for row in rows}


def prepare_advantages(cfg: RLCycleConfig, modules: dict, index_dir: Path) -> None:
    root = index_dir.parent
    directory = root / "retrain/advantages"
    if (directory / "run.json").exists():
        return
    sampling = load_sampling_config(REPO / cfg.sampling)
    pretrain = load_pretrain_config(REPO / sampling.pretrain)
    perception = load_perception_config(REPO / pretrain.perception)
    states = load_perception_data_config(REPO / sampling.perception_data).states
    ppo = load_ppo_config(REPO / cfg.ppo)
    mcts = load_mcts_config(REPO / sampling.mcts)
    run = json.loads((index_dir / "run.json").read_text())
    done = complete_shards(directory / "shards")
    total = run["counts"]["ppo"]
    initial = sum(
        len(read_json_shard(directory / "shards", shard, "items.json"))
        for shard in range(done)
    )
    with tqdm(
        total=total,
        initial=initial,
        desc="rl retrain advantages",
        unit="item",
        file=sys.stderr,
    ) as bar:
        for shard in range(done, complete_shards(index_dir / "shards")):
            items = read_json_shard(index_dir / "shards", shard, "items.json")["items"]
            rows = []
            for item in items:
                if item["kind"] == "preference" or not ppo_eligible(
                    item, sampling, ppo
                ):
                    continue
                t = item["index"]
                with torch.no_grad():
                    origin = encode_ref(
                        state_ref(sampling, t),
                        modules["phi"],
                        perception,
                        sampling,
                        states,
                        len(modules["pi"].names),
                    )
                    futures = [
                        encode_ref(
                            state_ref(sampling, t + k),
                            modules["phi"],
                            perception,
                            sampling,
                            states,
                            len(modules["pi"].names),
                        ).Y
                        for k in range(1, ppo.horizon + 1)
                    ]
                    advantage = reward(
                        modules["P"],
                        [origin.Y],
                        [futures],
                        ppo.horizon,
                        mcts.discount,
                        mcts.min_p,
                    )
                rows.append(
                    {
                        "kind": item["kind"],
                        "index": t,
                        "advantage": float(advantage[0]),
                    }
                )
                bar.update()
            write_json_shard(directory / "shards", shard, "items.json", rows)
    write_json(directory / "run.json", {"complete": True, "count": total})


def joint_optimizer(cfg: RLCycleConfig, modules: dict, index_dir: Path):
    root = index_dir.parent
    directory = root / "checkpoints"
    h = complete_shards(directory)
    if h == 0:
        raise RuntimeError(f"joint training has no final checkpoint: {directory}")
    state = restore(directory, h, modules)
    groups = check_groups(
        cfg, {name: modules[name] for name in ("phi", "metra", "pi", "P", "P_loss")}
    )
    optimizer = adamw(cfg, groups)
    optimizer.load_state_dict(state["optimizer"])
    return optimizer, groups


def retrain(
    cfg: RLCycleConfig,
    modules: dict,
    index_dir: Path,
    optimizer: torch.optim.AdamW | None,
) -> None:
    root = index_dir.parent
    marker = root / "retrain/run.json"
    if marker.exists():
        return
    stateful = {name: modules[name] for name in STATEFUL}
    groups = check_groups(
        cfg, {name: modules[name] for name in ("phi", "metra", "pi", "P", "P_loss")}
    )
    directory = root / "retrain/checkpoints"
    h = complete_shards(directory)
    if h:
        state = restore(directory, h, modules)
        optimizer = adamw(cfg, groups)
        optimizer.load_state_dict(state["optimizer"])
        position = {key: state[key] for key in START}
    else:
        if optimizer is None:
            optimizer, _ = joint_optimizer(cfg, modules, index_dir)
        position = dict(START)
        prepare_advantages(cfg, modules, index_dir)
        torch.manual_seed(cfg.seed)
    optimizer.zero_grad(set_to_none=True)
    modules["phi"].requires_grad_(False)
    modules["metra"].requires_grad_(False)
    for module in modules.values():
        module.train()
    run = json.loads((index_dir / "run.json").read_text())
    sizes, total = shard_sizes(index_dir), run["steps"]
    counts, losses, visit, t, j = run["counts"], [], None, position["step"], h
    sampling = load_sampling_config(REPO / cfg.sampling)
    pretrain = load_pretrain_config(REPO / sampling.pretrain)
    perception = load_perception_config(REPO / pretrain.perception)
    states = load_perception_data_config(REPO / sampling.perception_data).states
    ppo = load_ppo_config(REPO / cfg.ppo)
    mcts = load_mcts_config(REPO / sampling.mcts)
    weight = modules["phi"].head[0].weight
    alpha = torch.tensor(perception.alpha, dtype=weight.dtype, device=weight.device)
    with (
        TrainingProfile(cfg, root, modules, "rl-retrain") as profile,
        tqdm(
            total=total, initial=t, desc="rl retrain", unit="step", file=sys.stderr
        ) as bar,
    ):
        for at, shard, positions in remaining(
            generator([cfg.seed, 3], position["schedule_rng"]),
            sizes,
            cfg.batch,
            position,
            cfg.epochs,
        ):
            with profile.step(t + 1, len(positions)):
                with phase("data.index_read"):
                    if visit != (at["epoch"], shard):
                        shard_items = read_json_shard(
                            index_dir / "shards", shard, "items.json"
                        )["items"]
                        advantages = fixed_advantages(root, shard)
                        visit = at["epoch"], shard
                t += 1
                items = [shard_items[i] for i in positions]
                optimizer.zero_grad(set_to_none=True)
                entry = step_gradients(
                    items,
                    modules,
                    sampling=sampling,
                    perception=perception,
                    states=states,
                    alpha=alpha,
                    ppo=ppo,
                    discount=mcts.discount,
                    min_p=mcts.min_p,
                    m=run["batches"],
                    n_a=counts["ppo"],
                    n_delta=counts["fresh"],
                    r_c=counts["preference"],
                    w_ppo=cfg.w_ppo,
                    w_delta=cfg.w_delta,
                    w_metra=cfg.w_metra,
                    w_P=cfg.w_P,
                    micro_batch=cfg.micro_batch,
                    fixed_advantages=advantages,
                )
                lr = update(cfg, optimizer, groups, t, total)
                optimizer.zero_grad(set_to_none=True)
                losses.append({"step": t} | entry | {"lr": lr})
                if t % cfg.checkpoint_steps == 0 or t == total:
                    write_checkpoint(
                        directory,
                        j,
                        {"step": t} | at,
                        stateful,
                        optimizer,
                        torch.device(cfg.device),
                        losses,
                    )
                    prune(directory, j, cfg.keep_checkpoints)
                    j, losses = j + 1, []
                bar.set_postfix({key: entry[key] for key in (*ROLES, "loss")})
                bar.update()
    write_model(root / "model.pt", stateful)
    write_json(marker, {"complete": True, "steps": total})
