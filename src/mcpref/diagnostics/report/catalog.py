"""Stable dashboard names, purpose and missing-source explanations."""

GROUPS = (
    (
        "losses",
        "Training losses",
        "Optimizer objectives and their components.",
        "Needs published checkpoint losses.json; normally every 100 optimizer steps.",
    ),
    (
        "preferences",
        "Preference quality",
        "Prediction errors, calibration and saturation.",
        "Needs labelled preference pairs. Succession metrics also need comparable pairs.",
    ),
    (
        "policy",
        "Action policy",
        "Action targets, confidence, baselines and masks.",
        "Needs valid action targets. Rich pretrain policy metrics use checkpoint-boundary batches.",
    ),
    (
        "representation",
        "State representation",
        "Latent geometry, transition prediction and METRA.",
        "Needs perception or factual-transition observations; frozen retrain may have none.",
    ),
    (
        "optimization",
        "Optimization health",
        "Learning rates, gradients, clipping and actual updates.",
        "Needs saved optimizer observations. RL retrain currently saves losses and learning rates only.",
    ),
    (
        "reference",
        "Fixed train / validation reference",
        "Conditions stay distinct; compare generalisation on fixed examples.",
        "Needs reference/metrics.json shards: step 0, then every 500 pretrain steps; no retrain reference.",
    ),
    (
        "ppo",
        "PPO updates",
        "Policy change, clipping, effective sample size and advantages.",
        "Needs RL training with eligible PPO observations; pretrain and retrain do not produce these.",
    ),
    (
        "mcts",
        "MCTS search",
        "Search-policy agreement and factual transition accuracy.",
        "Needs accepted MCTS records, published after each 1000-action sampling block.",
    ),
    (
        "milestones",
        "Minecraft milestones",
        "Reached milestones, first times and censored survival curves.",
        "Needs completed sampling blocks (normally 1000 actions). Unreached milestones have no first time.",
    ),
    (
        "data",
        "Labels and data coverage",
        "Votes, graph corrections and available training populations.",
        "Needs label-production records or saved training counts; old labels are not backfilled.",
    ),
    (
        "memory",
        "Agent memory",
        "Candidates, duplicates, capacity and episode coverage.",
        "Needs a completed RL cycle and its published memory event.",
    ),
    (
        "gradient_flow",
        "Selected-step gradient flow",
        "Component gradients and saved counterfactual preference changes.",
        "Needs pretrain gradient-flow records: first, middle and last labelled step, at most three in total.",
    ),
)

ACCEPTED_GROUPS = {
    "losses": 1,
    "pretrain-preference": 2,
    "policy-action-quality": 3,
    "policy-calibration": 3,
    "policy-masks": 3,
    "factual-metra": 4,
    "update-health": 5,
    "ppo": 7,
    "policy-mcts": 8,
    "mcts-outcomes": 8,
    "milestones": 9,
    "target-provenance": 10,
    "memory-efficiency": 11,
}


def definitions() -> list[dict]:
    return [
        {
            "group": slug,
            "file": f"{i:02d}_{slug}.png",
            "title": f"{i:02d}  {title}",
            "description": description,
            "empty_reason": reason,
            "missing": 0,
            "series": [],
            "accepted_series": [],
            "sources": [],
            "x": "mixed",
        }
        for i, (slug, title, description, reason) in enumerate(GROUPS, 1)
    ]
