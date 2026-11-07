"""C4: one resumable joint AdamW loop over the scheduled mixed index."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from mcpref import REPO
from mcpref.clips.collect import load_config as load_perception_data_config
from mcpref.mcts.config import load_config as load_mcts_config
from mcpref.perception.config import load_config as load_perception_config
from mcpref.ppo.config import load_config as load_ppo_config
from mcpref.pretrain.checkpoints import prune, write_model
from mcpref.pretrain.checks import check_groups
from mcpref.pretrain.config import load_config as load_pretrain_config
from mcpref.pretrain.update import adamw, update
from mcpref.rl.checkpoints import STATEFUL, restore, write_checkpoint
from mcpref.rl.config import RLCycleConfig
from mcpref.rl.loss import ROLES, step_gradients
from mcpref.sampling.config import load_config as load_sampling_config
from mcpref.shards.store import complete_shards, shard_path
from mcpref.training.schedule import remaining
from mcpref.vpt.plan import generator

START = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}


def shard_sizes(index_dir: Path) -> list[int]:
    """Read one index shard at a time to obtain the schedule's fixed sizes."""
    return [
        len(
            json.loads(
                (shard_path(index_dir / "shards", k) / "items.json").read_text()
            )["items"]
        )
        for k in range(complete_shards(index_dir / "shards"))
    ]


def load_initial_optimizer(optimizer: torch.optim.AdamW, path: Path) -> None:
    """Load persistent Adam state after final-step and group compatibility checks."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    previous_root = path.parent.parent.parent
    run = json.loads((previous_root / "index" / "run.json").read_text())
    if state["step"] != run["steps"]:
        raise RuntimeError(f"previous checkpoint is not final: {path}")
    saved_groups = state["optimizer"]["param_groups"]
    current_groups = optimizer.state_dict()["param_groups"]
    if len(saved_groups) != len(current_groups):
        raise RuntimeError("initial optimizer group count does not match")
    for q, (saved, current) in enumerate(zip(saved_groups, current_groups)):
        for key in ("params", "betas", "eps", "weight_decay"):
            if saved[key] != current[key]:
                raise RuntimeError(f"initial optimizer group {q} has mismatched {key}")
    rates = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(state["optimizer"])
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate


def started(
    cfg: RLCycleConfig,
    modules: dict,
    groups: dict,
    directory: Path,
    initial_optimizer: Path | None,
):
    """Create a cycle-local AdamW, then freshly seed or restore an exact position."""
    stateful = {name: modules[name] for name in STATEFUL}
    h = complete_shards(directory)
    if h == 0:
        if initial_optimizer is None and int(directory.parent.name) > 0:
            raise RuntimeError("cycle > 0 requires an initial optimizer checkpoint")
        torch.manual_seed(cfg.seed)
        optimizer = adamw(cfg, groups)
        if initial_optimizer is not None:
            load_initial_optimizer(optimizer, initial_optimizer)
        return optimizer, dict(START), 0
    state = restore(directory, h, stateful)
    optimizer = adamw(cfg, groups)
    optimizer.load_state_dict(state["optimizer"])
    return optimizer, {key: state[key] for key in START}, h


def train(
    cfg: RLCycleConfig,
    modules: dict[str, torch.nn.Module],
    index_dir: Path,
    initial_optimizer: Path | None,
) -> None:
    """Run (1)-(6), checkpointing periodically and always at the final step T."""
    for module in modules.values():
        module.train()
    ordered = {name: modules[name] for name in ("phi", "metra", "pi", "P", "P_loss")}
    groups = check_groups(cfg, ordered)
    run = json.loads((index_dir / "run.json").read_text())
    sizes, total = shard_sizes(index_dir), run["steps"]
    directory = index_dir.parent / "checkpoints"
    optimizer, position, j = started(cfg, modules, groups, directory, initial_optimizer)
    stateful = {name: modules[name] for name in STATEFUL}
    if position["step"] == total:
        print("rl cycle: already done", file=sys.stderr)
        write_model(index_dir.parent / "model.pt", stateful)
        return
    sampling = load_sampling_config(REPO / cfg.sampling)
    pretrain = load_pretrain_config(REPO / sampling.pretrain)
    perception = load_perception_config(REPO / pretrain.perception)
    states = load_perception_data_config(REPO / sampling.perception_data).states
    ppo = load_ppo_config(REPO / cfg.ppo)
    mcts = load_mcts_config(REPO / sampling.mcts)
    weight = modules["phi"].head[0].weight
    alpha = torch.tensor(perception.alpha, dtype=weight.dtype, device=weight.device)
    rng = generator([cfg.seed, 3], position["schedule_rng"])
    counts, losses = run["counts"], []
    visit, t = None, position["step"]
    with tqdm(
        total=total, initial=t, desc="rl cycle", unit="step", file=sys.stderr
    ) as bar:
        for at, shard, positions in remaining(
            rng, sizes, cfg.batch, position, cfg.epochs
        ):
            if visit != (at["epoch"], shard):
                path = shard_path(index_dir / "shards", shard) / "items.json"
                shard_items = json.loads(path.read_text())["items"]
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
            )
            lr = update(cfg, optimizer, groups, t, total)
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
    write_model(index_dir.parent / "model.pt", stateful)
