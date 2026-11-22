"""Config relocation must preserve an existing classical RL run's identity."""

from mcpref.runtime.sampling.config import load_config as load_sampling
from mcpref.training.rl.config import load_config
from mcpref.training.rl.files import fingerprints, start_run
from training.pretrain.classical.data.index_run import repo_root
from training.rl.cycle_world import prepare
from training.rl.learning.helpers import write_json


def test_classical_rl_resume_after_pretrain_config_relocation(tmp_path, repo):
    root, path = prepare(tmp_path, repo)
    prefix = "configs/training/pretrain/"
    selector = root / (prefix + "config.yaml")
    old = root / (prefix + "classical.yaml")
    selector.rename(old)
    write_json(
        selector,
        {
            "mode": "classical",
            "classical": {"config": prefix + "classical.yaml", "index": "unused.yaml"},
            "continuous": {"config": "unused.yaml", "index": "unused.yaml"},
        },
    )
    with repo_root(root):
        cfg = load_config(path)
        sampling = load_sampling(root / cfg.sampling)
        model = root / sampling.checkpoint_dir / "model.pt"
        before = fingerprints(cfg, path, sampling, model)
        run_path, _ = start_run(cfg, path, 0, 0, before)
        recorded = run_path.read_bytes()
        new = root / (prefix + "classical/config.yaml")
        new.parent.mkdir()
        old.rename(new)
        selector.write_text(
            selector.read_text().replace(
                prefix + "classical.yaml", prefix + "classical/config.yaml"
            )
        )
        after = fingerprints(cfg, path, sampling, model)
        assert after == before
        start_run(cfg, path, 0, 0, after)
        assert run_path.read_bytes() == recorded
