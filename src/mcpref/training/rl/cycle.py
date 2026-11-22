"""C1-C6: one artifact-reconciled RL cycle on a caller-owned live Env."""

import json
from pathlib import Path

import torch

from mcpref.config import REPO
from mcpref.diagnostics.core.runtime import DiagnosticRun
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.infrastructure.runs.files import sha256
from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.models.factory import build_modules
from mcpref.models.policy.checkpoints import load_policy_weights
from mcpref.models.policy.config import load_config as load_policy_config
from mcpref.models.preference.config import load_config as load_preference_config
from mcpref.runtime.env.wrapper import Env
from mcpref.runtime.sampling.checkpoints import load_weights
from mcpref.runtime.sampling.config import load_config as load_sampling_config
from mcpref.runtime.sampling.models import SamplingModules
from mcpref.runtime.sampling.runner import Sampler
from mcpref.runtime.sampling.shards import record_cache
from mcpref.training.rl.config import RLCycleConfig
from mcpref.training.rl.context import load_context
from mcpref.training.rl.files import (
    advance,
    fingerprints,
    handover_from,
    initial_model,
    initial_optimizer,
    pair_rng,
    select_cycle,
    shards_digest,
    start_run,
    tree_digest,
    validate_anchor_manifests,
)
from mcpref.training.rl.learning.checkpoints import STATEFUL
from mcpref.training.rl.learning.history import eval_encoder
from mcpref.training.rl.learning.index import build_index
from mcpref.training.rl.learning.loop import train
from mcpref.training.rl.learning.memory_producer import (
    accept as accept_memory,
)
from mcpref.training.rl.learning.memory_producer import (
    register as register_memory,
)
from mcpref.training.rl.learning.memory_producer import (
    stage as stage_memory,
)
from mcpref.training.rl.learning.retrain import retrain as retrain_heads
from mcpref.training.rl.pairs.config import load_config as load_pairs_config
from mcpref.training.rl.pairs.cycle import publish_memory, sample_pairs
from mcpref.training.rl.pairs.types import StateRef
from mcpref.training.rl.ppo.config import load_config as load_ppo_config
from mcpref.training.rl.ppo.metra import MetraLoss


def build_rl_modules(cfg: RLCycleConfig, sampling, ppo) -> dict[str, torch.nn.Module]:
    """Construct phi/pi/P and the stateless preference loss plus METRA's dual."""
    context = load_context(cfg, sampling)
    pretrain, perception = context.pretrain, context.perception
    masks = load_masks(load_mask_config(REPO / pretrain.action_mask))
    policy = load_policy_config(REPO / pretrain.policy)
    preference = load_preference_config(REPO / pretrain.preference)
    built = build_modules(
        perception,
        masks,
        policy,
        preference,
        True,
        cfg.device,
        pretrain.latent_init_scale,
    )
    modules = {name: built[name] for name in ("phi", "pi", "P", "P_loss")}
    modules["metra"] = MetraLoss(ppo).to(device=cfg.device, dtype=torch.bfloat16)
    return modules


def load_model(
    path: Path,
    modules: dict[str, torch.nn.Module],
    names,
    *,
    mode: str = "classical",
    allow_legacy_policy: bool = False,
) -> None:
    """Load named weights strictly, allowing a new change head only for warm starts."""
    weights = load_weights(path, mode)
    for name in names:
        if name == "pi" and allow_legacy_policy:
            load_policy_weights(modules[name], weights[name])
        else:
            modules[name].load_state_dict(weights[name])


def run_cycle(cfg: RLCycleConfig, path: Path, env: Env) -> Path:
    """Run exactly one selected R1 -> R3 -> joint training -> R5 cycle."""
    sampling = load_sampling_config(REPO / cfg.sampling)
    ppo = load_ppo_config(REPO / cfg.ppo)
    context = load_context(cfg, sampling)
    capacity = context.states + ppo.horizon
    if context.replay is not None:
        capacity = (
            context.replay.replay_stride
            + max(context.states, context.replay.gradient_history)
            + ppo.horizon
        )
    with record_cache(REPO / sampling.out_dir, capacity):
        return _run_cycle(cfg, path, env, sampling, ppo, context)


def _run_cycle(cfg, path, env, sampling, ppo, context) -> Path:
    pretrain, perception, states = context.pretrain, context.perception, context.states
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
    load_model(
        model,
        modules,
        ("phi", "pi", "P") if cycle == 0 else STATEFUL,
        mode=context.mode,
        allow_legacy_policy=cycle == 0,
    )
    run_path, run = start_run(
        cfg, path, cycle, u_before, fingerprints(cfg, path, sampling, model)
    )
    if context.replay is not None:
        validate_anchor_manifests(cfg, sampling, cycle)
    if trajectories == cycle:
        supplied = SamplingModules(
            modules["phi"], modules["pi"], modules["P"], sha256(model)
        )
        options = {}
        if context.replay is not None:
            options = {
                "replay_stride": context.replay.replay_stride,
                "handover_from": handover_from(cfg, cycle, model),
            }
        sampler = Sampler(sampling, env, supplied, **options)
        sampler.reset()
        sampler.sample()
    trajectory = shard_path(REPO / sampling.out_dir / "shards", cycle)
    sampled_hashes = {"trajectory": tree_digest(trajectory)}
    if context.replay is not None:
        sampled_hashes["anchors_manifest"] = sha256(trajectory / "anchors.json")
    run = advance(run_path, run, "sampled", **sampled_hashes)

    pairs_dir = REPO / pairs_cfg.out_dir / f"{cycle:06d}"
    pair_run_path = pairs_dir / "run.json"
    pair_run = json.loads(pair_run_path.read_text()) if pair_run_path.exists() else {}
    if pair_run.get("phase") not in ("correction", "history"):
        weight = modules["phi"].head[0].weight
        alpha = torch.tensor(perception.alpha, dtype=weight.dtype, device=weight.device)
        encode = eval_encoder(
            modules["phi"],
            perception,
            sampling,
            states,
            len(modules["pi"].names),
            continuous=context.replay,
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
    pending = cfg.retrain and not (root / "retrain/run.json").exists()
    resumed = complete_shards(root / "retrain/checkpoints") > 0
    optimizer = None
    if not (root / "model.pt").exists() or (pending and not resumed):
        optimizer = train(cfg, modules, root / "index", optimizer_path)
    run = advance(run_path, run, "trained")
    if pending:
        retrain_heads(cfg, modules, root / "index", optimizer)
        run = advance(run_path, run, "retrained")
    load_model(root / "model.pt", modules, STATEFUL, mode=context.mode)
    history_run = REPO / pairs_cfg.history_dir / f"{cycle:06d}" / "run.json"
    published = history_run.exists() and json.loads(history_run.read_text()).get(
        "complete"
    )
    diagnostics = DiagnosticRun.load(REPO / pretrain.diagnostics)
    register_memory(diagnostics, cycle)
    if not published or pair_run.get("phase") != "history":
        encode = eval_encoder(
            modules["phi"],
            perception,
            sampling,
            states,
            len(modules["pi"].names),
            continuous=context.replay,
        )
        publish_memory(
            pairs_cfg,
            cycle,
            encode,
            observe=lambda history: stage_memory(
                diagnostics, pairs_cfg, cycle, sampling.steps, history
            ),
        )
    accept_memory(diagnostics, pairs_cfg, cycle)
    advance(run_path, run, "finalized", complete=True)
    return root
