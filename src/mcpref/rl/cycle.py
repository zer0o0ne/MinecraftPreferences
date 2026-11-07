"""C1-C6: one artifact-reconciled RL cycle on a caller-owned live Env."""

import json
from pathlib import Path

import torch

from mcpref import REPO
from mcpref.clips.collect import load_config as load_perception_data_config
from mcpref.env.wrapper import Env
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.perception.config import load_config as load_perception_config
from mcpref.policy.config import load_config as load_policy_config
from mcpref.ppo.config import load_config as load_ppo_config
from mcpref.ppo.metra import MetraLoss
from mcpref.preferences.config import load_config as load_preference_config
from mcpref.pretrain.config import load_config as load_pretrain_config
from mcpref.pretrain.run import build_modules
from mcpref.rl.checkpoints import STATEFUL
from mcpref.rl.config import RLCycleConfig
from mcpref.rl.files import (
    advance,
    fingerprints,
    initial_model,
    initial_optimizer,
    pair_rng,
    select_cycle,
    shards_digest,
    start_run,
    tree_digest,
)
from mcpref.rl.history import eval_encoder
from mcpref.rl.index import build_index
from mcpref.rl.loop import train
from mcpref.rl_pairs.config import load_config as load_pairs_config
from mcpref.rl_pairs.cycle import publish_memory, sample_pairs
from mcpref.rl_pairs.types import StateRef
from mcpref.runs.files import sha256
from mcpref.sampling.config import load_config as load_sampling_config
from mcpref.sampling.models import SamplingModules
from mcpref.sampling.runner import Sampler
from mcpref.shards.store import complete_shards, shard_path


def build_rl_modules(cfg: RLCycleConfig, sampling, ppo) -> dict[str, torch.nn.Module]:
    """Construct phi/pi/P and the stateless preference loss plus METRA's dual."""
    pretrain = load_pretrain_config(REPO / sampling.pretrain)
    perception = load_perception_config(REPO / pretrain.perception)
    masks = load_masks(load_mask_config(REPO / pretrain.action_mask))
    policy = load_policy_config(REPO / pretrain.policy)
    preference = load_preference_config(REPO / pretrain.preference)
    built = build_modules(perception, masks, policy, preference, True, cfg.device)
    modules = {name: built[name] for name in ("phi", "pi", "P", "P_loss")}
    modules["metra"] = MetraLoss(ppo).to(torch.device(cfg.device))
    return modules


def load_model(path: Path, modules: dict[str, torch.nn.Module], names) -> None:
    """Load exactly the named CPU state_dict entries into existing modules."""
    weights = torch.load(path, map_location="cpu", weights_only=True)
    for name in names:
        modules[name].load_state_dict(weights[name])


def run_cycle(cfg: RLCycleConfig, path: Path, env: Env) -> Path:
    """Run exactly one selected R1 -> R3 -> joint training -> R5 cycle."""
    sampling = load_sampling_config(REPO / cfg.sampling)
    ppo = load_ppo_config(REPO / cfg.ppo)
    pairs_cfg = load_pairs_config(REPO / cfg.rl_pairs)
    trajectories = complete_shards(REPO / sampling.out_dir / "shards")
    cycle = select_cycle(cfg, trajectories)
    if cycle > trajectories:
        raise RuntimeError("the live sampling timeline does not contain this cycle")
    u_before, u_after = cycle * sampling.steps, (cycle + 1) * sampling.steps
    root = REPO / cfg.out_dir / f"{cycle:06d}"
    model = initial_model(cfg, sampling, cycle)
    optimizer_path = initial_optimizer(cfg, cycle)
    modules = build_rl_modules(cfg, sampling, ppo)
    load_model(model, modules, ("phi", "pi", "P") if cycle == 0 else STATEFUL)
    run_path, run = start_run(
        cfg, path, cycle, u_before, fingerprints(cfg, path, sampling, model)
    )
    if trajectories == cycle:
        supplied = SamplingModules(
            modules["phi"], modules["pi"], modules["P"], sha256(model)
        )
        sampler = Sampler(sampling, env, supplied)
        sampler.reset()
        sampler.sample()
    trajectory = shard_path(REPO / sampling.out_dir / "shards", cycle)
    run = advance(run_path, run, "sampled", trajectory=tree_digest(trajectory))

    pairs_dir = REPO / pairs_cfg.out_dir / f"{cycle:06d}"
    pair_run_path = pairs_dir / "run.json"
    pair_run = json.loads(pair_run_path.read_text()) if pair_run_path.exists() else {}
    if pair_run.get("phase") not in ("correction", "history"):
        perception = load_perception_config(
            REPO / load_pretrain_config(REPO / sampling.pretrain).perception
        )
        states = load_perception_data_config(REPO / sampling.perception_data).states
        weight = modules["phi"].head[0].weight
        alpha = torch.tensor(perception.alpha, dtype=weight.dtype, device=weight.device)
        encode = eval_encoder(
            modules["phi"], perception, sampling, states, len(modules["pi"].names)
        )
        refs = [
            StateRef(f"{sampling.out_dir}#{cycle:06d}", t)
            for t in range(u_before, u_after + 1)
        ]
        sample_pairs(
            pairs_cfg, cycle, refs, encode, modules["P"], alpha, pair_rng(cfg, cycle)
        )
    incoming = (
        root.parent / f"{cycle - 1:06d}" / "tail" / "shards" if cycle else root / "none"
    )
    run = advance(
        run_path,
        run,
        "pairs",
        pairs=shards_digest(pairs_dir / "shards", "items.json"),
        incoming_tail=shards_digest(incoming, "items.json"),
    )
    build_index(cfg, cycle, u_before, pairs_dir)
    run = advance(
        run_path,
        run,
        "indexed",
        order=shards_digest(root / "order" / "shards", "indices.json"),
        index=shards_digest(root / "index" / "shards", "items.json"),
        tail=shards_digest(root / "tail" / "shards", "items.json"),
    )
    if not (root / "model.pt").exists():
        train(cfg, modules, root / "index", optimizer_path)
    load_model(root / "model.pt", modules, STATEFUL)
    run = advance(run_path, run, "trained")
    history_run = REPO / pairs_cfg.history_dir / f"{cycle:06d}" / "run.json"
    published = history_run.exists() and json.loads(history_run.read_text()).get(
        "complete"
    )
    if not published:
        pretrain = load_pretrain_config(REPO / sampling.pretrain)
        perception = load_perception_config(REPO / pretrain.perception)
        states = load_perception_data_config(REPO / sampling.perception_data).states
        encode = eval_encoder(
            modules["phi"], perception, sampling, states, len(modules["pi"].names)
        )
        publish_memory(pairs_cfg, cycle, encode)
    advance(run_path, run, "finalized", complete=True)
    return root
