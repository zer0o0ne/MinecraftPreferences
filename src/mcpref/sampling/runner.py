"""Sampler of specs/env-sampling.md (2)-(8): one endless, resumable timeline."""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcpref import REPO
from mcpref.clips.collect import load_config as load_perception_data_config
from mcpref.clips.config import load_config as load_clips_config
from mcpref.diagnostics.artifacts import iter_records
from mcpref.diagnostics.mcts import on_step_record
from mcpref.diagnostics.milestones import on_episode_end
from mcpref.diagnostics.observations import EpisodeArtifacts, EpisodeSummary
from mcpref.diagnostics.runtime import DiagnosticRun, diagnostic_bar
from mcpref.diagnostics.types import ProducerContext
from mcpref.env.config import load_config as load_env_config
from mcpref.env.wrapper import Env
from mcpref.mcts.config import load_config as load_mcts_config
from mcpref.mcts.tree import Tree
from mcpref.perception.config import load_config as load_perception_config
from mcpref.perception.stream import PhiStream
from mcpref.policy.config import load_config as load_policy_config
from mcpref.preferences.config import load_config as load_preference_config
from mcpref.pretrain.config import load_config as load_pretrain_config
from mcpref.pretrain.run import check_lengths
from mcpref.sampling.config import SamplingConfig
from mcpref.sampling.files import SamplingRun
from mcpref.sampling.history import encode
from mcpref.sampling.models import SamplingModules, load_pretrained
from mcpref.sampling.shards import load_state, read_history
from mcpref.sampling.steps import step
from mcpref.shards.store import complete_shards, name, shard_path, write_shard


class Sampler:
    """Interface: explicit reset, then one complete trajectory shard per sample."""

    def __init__(
        self, cfg: SamplingConfig, env: Env, modules: SamplingModules | None = None
    ) -> None:
        self.cfg, self.env = cfg, env
        self.modules = load_pretrained(cfg) if modules is None else modules
        for module in (self.modules.phi, self.modules.pi, self.modules.P):
            module.eval()
        pretrain = load_pretrain_config(REPO / cfg.pretrain)
        policy = load_policy_config(REPO / pretrain.policy)
        preference = load_preference_config(REPO / pretrain.preference)
        perception = load_perception_config(REPO / pretrain.perception)
        data = load_perception_data_config(REPO / cfg.perception_data)
        self.states = data.states
        check_lengths(
            self.states, policy, preference, paired=True, index_dir=cfg.perception_data
        )
        self.mcts = load_mcts_config(REPO / cfg.mcts)
        env_cfg = load_env_config(REPO / cfg.env)
        self.delta = load_clips_config(REPO / env_cfg.clips).delta
        weight = self.modules.phi.head[0].weight
        self.device = weight.device
        self.alpha = torch.tensor(
            perception.alpha, dtype=weight.dtype, device=self.device
        )
        self.actions = len(self.modules.pi.names)
        self.stream = PhiStream(self.modules.phi, self.states)
        path = getattr(cfg, "_source_path", REPO / "configs/env-sampling.yaml")
        self.run = SamplingRun(cfg, path)
        self.diagnostics = DiagnosticRun.load(REPO / pretrain.diagnostics)
        self._reset_done = False

    def reset(self) -> None:
        """(2): one reset; start afresh or restore the last complete boundary state."""
        if self._reset_done:
            raise RuntimeError("Sampler.reset may be called only once")
        self._reset_done = True
        out = REPO / self.cfg.out_dir
        complete = complete_shards(out / "shards")
        self.pending = None
        if complete == 0:
            self._reset_fresh()
        else:
            self._restore(out)
        channels = self.diagnostics.state["channels"]
        self.diagnostics.register(
            "mcts",
            watermark=channels.get("mcts", {}).get("watermark", self.next_step),
            next=self.next_step,
        )
        self.diagnostics.register(
            "milestones",
            watermark=channels.get("milestones", {}).get("watermark", complete),
            next=complete,
        )
        self._inventories = []
        self.diagnostics.recover("mcts")
        self.diagnostics.recover("milestones")

    def _reset_fresh(self) -> None:
        """(2): reset the environment, stream, decision RNG and global index."""
        self._observation = self.env.reset()
        self.stream = PhiStream(self.modules.phi, self.states)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.next_step = 0
        self.history, self.tree = None, None
        self._raw, self._preceding = [self._observation], [None]
        self._changed = False

    def _restore(self, out: Path) -> None:
        """(2): restore a complete boundary, rebuilding raw history if models changed."""
        state = load_state(out, self.modules.phi, self.states, self.device)
        self.env.restore(state.observation, state.env)
        self._observation, self.history, self.tree = (
            state.observation,
            state.history,
            state.tree,
        )
        self.next_step = state.next_step
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state.rng_state
        self._changed = state.fingerprint != self.modules.fingerprint
        if not self._changed:
            self.stream = state.stream
            self._raw, self._preceding = [], []
            return
        self.stream = PhiStream(self.modules.phi, self.states)
        raw = read_history(out, self.next_step, self.states)
        self._raw, self._preceding = raw.states, raw.actions

    def _prepare(self) -> None:
        """(3): encode through s_t, then construct the tree or replace a stale base."""
        if self._changed:
            self.history = None
        self.history = encode(
            self.stream,
            self.history,
            self._raw,
            self._preceding,
            self.actions,
            self.states,
        )
        self._raw, self._preceding = [], []
        if self.tree is None:
            self.tree = Tree(self.history)
        elif self._changed:
            self.tree.base = self.history
        self._changed = False

    def observe(self, record) -> None:
        """Stage a completed StepRecord before its source shard is published."""
        inventory = {}
        for item in record.after.line.get("inventory", []):
            inventory[item["type"]] = inventory.get(item["type"], 0) + item["quantity"]
        self._inventories.append(inventory)
        cycle, local = divmod(record.index, self.cfg.steps)
        source = f"{self.cfg.out_dir}/shards/{name(cycle)}/steps/{name(local)}.pt"
        context = ProducerContext(
            producer="mcts",
            split="train",
            condition="train",
            model=self.modules.fingerprint,
            source=self.cfg.out_dir,
            filters={
                "alpha": self.alpha.detach().cpu().tolist(),
                "epsilon": self.mcts.epsilon,
                "width": self.mcts.width,
            },
            steps=(record.index,),
            cluster_ids=(f"{self.cfg.out_dir}#{cycle:06d}",),
            artifact_paths=(source,),
        )
        observed = on_step_record(record, cfg=self.diagnostics.cfg, context=context)
        self.diagnostics.stage(
            "mcts", record.index, observed["id"], record.index + 1, [source], observed
        )

    def observe_milestones(self, cycle: int) -> None:
        episode = f"{self.cfg.out_dir}#{cycle:06d}"
        source = f"{self.cfg.out_dir}/shards/{name(cycle)}/state.pt"
        previous = [
            EpisodeSummary(
                record["joined"]["episode"],
                record["joined"]["first_times"],
                record["joined"]["censored_at"],
            )
            for record in iter_records(self.diagnostics.out_dir, "milestones")
        ]
        artifacts = EpisodeArtifacts(episode, self._inventories, previous)
        context = ProducerContext(
            "milestones",
            "train",
            "train",
            self.modules.fingerprint,
            self.cfg.out_dir,
            {},
            (self.next_step - 1,),
            (episode,),
            (source,),
        )
        record = on_episode_end(artifacts, cfg=self.diagnostics.cfg, context=context)
        record["joined"]["episode"] = episode
        self.diagnostics.stage(
            "milestones", cycle, record["id"], cycle + 1, [source], record
        )

    def sample(self) -> Path:
        """(3)-(8): sample n actions, publish their shard, then update run.json."""
        if not self._reset_done:
            raise RuntimeError("Sampler.reset must be called before sample")
        out = REPO / self.cfg.out_dir
        k = complete_shards(out / "shards")
        start = self.next_step
        with (
            write_shard(out / "shards", k) as staged,
            tqdm(
                total=self.cfg.steps,
                initial=0,
                desc="env sampling",
                unit="step",
                file=sys.stderr,
            ) as bar,
        ):
            for local in range(self.cfg.steps):
                step(self, staged, local, bar)
            self.observe_milestones(k)
        self.run.published(k, self.next_step, self.modules.fingerprint)
        self.diagnostics.recover("mcts")
        self.diagnostics.recover("milestones")
        state = self.diagnostics.state["channels"]["mcts"]
        self.diagnostics.finish(
            "mcts", source_complete=True, done=state["next"] >= self.next_step
        )
        self.diagnostics.finish(
            "milestones",
            source_complete=True,
            done=self.diagnostics.state["channels"]["milestones"]["next"] >= k + 1,
        )
        with diagnostic_bar(
            "diagnostics mcts", self.next_step - start, 0, "step_record"
        ) as bar:
            if bar is not None:
                bar.update(self.next_step - start)
        with diagnostic_bar("diagnostics milestones", 1, 0, "episode") as bar:
            if bar is not None:
                bar.update()
        return shard_path(out / "shards", k)
