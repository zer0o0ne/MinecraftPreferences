"""Semantic axes: compatible quantities overlay; units never share a dual axis."""

import re

from mcpref.diagnostics.report.catalog import ACCEPTED_GROUPS


def canonical(metric: str) -> str:
    metric = metric.removeprefix("losses.")
    for head, components in (
        ("phi", ("push", "constraint", "dual", "norm")),
        ("pi", ("action", "shift")),
        ("P", ("inc", "succ")),
    ):
        if metric == f"components.{head}.total":
            return head
        for component in components:
            if metric in (
                f"{head}_{component}",
                f"phi.{component}" if head == "phi" else "",
            ):
                return f"components.{head}.{component}"
    for old, new in (
        ("outputs.P_inc", "preference.inc"),
        ("outputs.P_succ", "preference.succ"),
        ("outputs.pi_action", "policy"),
        ("pi.action.", "policy."),
        ("phi_stats.", "phi."),
    ):
        metric = metric.replace(old, new)
    return metric


def is_reference(entry: dict) -> bool:
    return (
        entry["group"] == "saved-pretrain-reference"
        or entry.get("identity", ("", ""))[1] == "pretrain-reference"
    )


def group_number(entry: dict, metric: str) -> int:
    group = entry["group"]
    if is_reference(entry):
        return 6
    if group == "saved-gradient-flow":
        return 12
    if group in ACCEPTED_GROUPS:
        return ACCEPTED_GROUPS[group]
    key = canonical(metric)
    if key.startswith(("gradients.", "updates.", "lr")):
        return 5
    if key.startswith(("preference.", "baselines.")):
        return 2
    if key.startswith(("policy.", "pi.")):
        return 3
    if key.startswith("phi."):
        return 4
    if key.startswith(("counts.", "coverage.")):
        return 10
    return 1


# Only quantities with matching units are assigned the same panel key.
RULES = (
    (r"^(lr(?:\.|$)|update.learning_rate$)", "Learning rates", "Learning rate"),
    (r"clip_coefficient", "Gradient clipping", "Multiplier"),
    (r"^(gradients\.).*steps", "Steps with active gradients", "Steps"),
    (r"^update.group.*active$", "Active gradient groups", "Active flag"),
    (r"nonfinite", "Nonfinite gradients", "Count"),
    (
        r"^(gradients\.)|^update.(preclip_norm|group.*preclip_norm|head_row_norm)$",
        "Gradient norms",
        "L2 norm",
    ),
    (r"^updates.*(steps|unavailable)", "Update availability", "Steps"),
    (r"^updates.*relative", "Relative parameter updates", "Relative norm"),
    (
        r"^updates\.|^update.(group.*norm|head_update_row_norm)$",
        "Parameter update norms",
        "L2 norm",
    ),
    (r"^update.*(zero|unsupported)", "Action-head gradient support", "Fraction"),
    (r"^update.logit_norm", "Policy logit norm", "L2 norm"),
    (r"^update.schedule_position", "Optimizer schedule", "Position"),
    (
        r"^policy.(nll|uniform_allowed_ce|prior_ce|copy_ce)$",
        "Action cross entropy and baselines",
        "Nats / valid target row",
    ),
    (
        r"^pi.(uniform|copy_current)$",
        "Policy baselines: all state rows",
        "Nats / state row",
    ),
    (
        r"^pi.empirical_constant_oracle$",
        "Policy empirical oracle: valid targets",
        "Nats / target mass",
    ),
    (r"^policy.*entropy$|^mask.allowed_entropy$", "Action entropy", "Nats"),
    (r"^policy.*(top1|topk|recall|precision)", "Action target agreement", "Fraction"),
    (
        r"^policy.*(confidence|ece)$",
        "Action confidence and calibration",
        "Probability / error",
    ),
    (r"^policy.brier$", "Action Brier error", "Squared probability error"),
    (r"^policy.tv$", "Action distribution distance", "Total variation"),
    (r"^policy.*undefined$", "Undefined policy divergences", "Count"),
    (r"^policy.kl_", "Policy / target divergence", "Nats"),
    (
        r"^policy.*(singleton_teacher|zero_teacher)$",
        "Action target coverage",
        "Fraction",
    ),
    (r"^policy.*count$", "Action target support", "Rows"),
    (r"^mask.count$", "Action mask width", "Allowed actions per state"),
    (
        r"^mask.(empty|singleton|executed_violation)$",
        "Action mask validity",
        "Fraction",
    ),
    (r"^mask.probability$", "Probability outside action mask", "Probability mass"),
    (r"^preference.*brier$", "Preference Brier error", "Squared probability error"),
    (r"^preference.*ece$", "Preference calibration error", "Probability error"),
    (r"^preference.*(low|high)$", "Preference saturation", "Fraction"),
    (r"^preference.*(count|tp|tn|fp|fn)$", "Preference support counts", "Pairs"),
    (r"^preference.inc_classification", "Incomparability classification", "Fraction"),
    (r"^preference.*std$", "Preference prediction spread", "Standard deviation"),
    (
        r"^preference.(inc|succ).(mean|quantiles)",
        "Preference probabilities",
        "Probability",
    ),
    (r"^baselines\.", "Preference loss baselines", "Loss"),
    (
        r"^pretrain.progress",
        "Preference improvement over baseline",
        "Relative progress",
    ),
    (r"^phi.d_alpha", "Distances between states", "Latent distance"),
    (r"^phi.z_norm", "Latent vector norms", "L2 norm"),
    (
        r"^phi.hinge_active$|^metra.(constraint_active|zero_skill)$",
        "Representation constraints",
        "Fraction",
    ),
    (r"^phi.(changed|state_pairs)$", "Representation coverage", "State pairs"),
    (r"^phi.dual_lambda$|^metra.lambda$", "METRA dual multiplier", "Multiplier"),
    (r"^phi.(push|constraint|dual|norm)$", "Perception loss components", "Loss"),
    (r"^metra.d$", "METRA distance", "Latent distance"),
    (r"^metra.q$", "METRA constraint slack", "Constraint value"),
    (r"^shift.error_squared$", "Squared transition error", "Squared latent distance"),
    (
        r"^shift.(error|zero_baseline)$",
        "Transition prediction and baseline",
        "Latent distance",
    ),
    (r"^pi.zero_shift$", "Zero-shift policy baseline", "Weighted squared shift loss"),
    (r"^ppo.(ratio|ratio_max)$", "PPO probability ratio", "Ratio"),
    (r"^ppo.*(_undefined|_count)$", "PPO sample counts", "Count"),
    (r"^ppo.kl_", "PPO policy divergence", "Nats"),
    (
        r"^ppo.(clip_positive|clip_negative)$",
        "PPO clipping by advantage sign",
        "Fraction",
    ),
    (r"^ppo.ess$", "PPO effective sample size", "Samples"),
    (r"^ppo.ess_normalized$", "PPO effective sample fraction", "Fraction"),
    (r"^ppo.advantage", "PPO advantages", "Advantage"),
    (
        r"^ppo.(eligible|fresh|pending)$",
        "PPO sample eligibility and origin",
        "Fraction",
    ),
    (r"^ppo.logprob_drift$", "PPO log-probability drift", "Nats"),
    (r"^ppo.action_tv$", "PPO action distribution distance", "Total variation"),
    (r"^mcts.*agreement$", "Search and policy agreement", "Fraction"),
    (r"^mcts.rank_", "Executed action rank", "Rank"),
    (r"^mcts.*_undefined$", "Undefined search divergences", "Count"),
    (r"^mcts.(kl_|js_)", "Search and policy divergence", "Nats"),
    (r"^mcts.visit_entropy$", "Search exploration", "Nats"),
    (
        r"^mcts.(policy_support_mass|top_width_prior_mass|matched_mass)$",
        "Search probability coverage",
        "Probability mass",
    ),
    (r"^mcts.*hit$", "Factual successor match", "Fraction"),
    (
        r"^mcts.(d_alpha_error|zero_shift_error)$",
        "Factual transition error",
        "Latent distance",
    ),
    (r"^milestones.coverage_", "Reached milestones", "Fraction of sampling blocks"),
    (
        r"^milestones.time_to_first_|^joined.first_times",
        "Time to first milestone",
        "Real actions",
    ),
    (
        r"^memory.*per_real_state$",
        "Memory events per real state",
        "Events / real state",
    ),
    (r"^memory.*per_seed$", "Memory events per seed", "Events / seed"),
    (
        r"^memory.(seeds|temporal|nearest|beam|empty_candidates)$",
        "Memory candidate counts",
        "Count",
    ),
    (r"^memory.(evictions|unused)$", "Memory usage counts", "Count"),
    (
        r"^memory.(dedup_ratio|repeated_comparison|capacity|episode_coverage)$",
        "Memory efficiency",
        "Fraction",
    ),
    (r"^counts\.", "Training population counts", "Count"),
    (r"^coverage\.", "Steps contributing to window", "Steps"),
    (r"^target.vote_.*order0$", "VLM votes: original order", "Fraction"),
    (r"^target.vote_.*order1$", "VLM votes: reversed order", "Fraction"),
    (
        r"^target.(graph_removed_all_pairs|semantic_inc|all_inc)$",
        "Graph correction: all pairs",
        "Fraction",
    ),
    (r"^target.model_", "Model preference saturation", "Fraction"),
    (
        r"^delta_(only_phi|joint)$",
        "Reference counterfactual preference change",
        "Loss change",
    ),
)


def axis_spec(metric: str, entry: dict, number: int) -> tuple[str, str]:
    key = canonical(metric)
    axis = entry["x"]
    if axis.startswith("km:"):
        return "Time to milestone: survival", "Not yet reached (fraction)"
    if axis == "bin":
        return (
            ("Reliability bin support", "Count")
            if key == "n"
            else ("Action reliability", "Probability")
        )
    if axis == "matched_mass":
        return "Successor-match reliability", "Hit"
    if number == 12:
        if key.startswith("counterfactual.") and key.endswith(
            (".harmful", ".helpful", ".zero")
        ):
            return "Counterfactual change signs", "Fraction"
        for pattern, title, unit in (
            ("nonfinite", "Nonfinite component gradients", "Count"),
            ("cosine", "Component gradient alignment", "Cosine"),
            ("update_dot", "Preference gradient vs actual update", "Inner product"),
            (".dot.", "Component gradient inner products", "Inner product"),
            ("norm", "Component gradient norms", "L2 norm"),
            (".count", "Counterfactual pair support", "Pairs"),
            ("harmful", "Harmful counterfactual updates", "Fraction"),
            ("helpful", "Helpful counterfactual updates", "Fraction"),
            ("counterfactual", "Counterfactual preference loss change", "Loss change"),
        ):
            if pattern in key:
                return title, unit
    for pattern, title, unit in RULES:
        if re.search(pattern, key):
            return title, unit
    if key.startswith("components."):
        return f"{key.split('.')[1]} loss components", "Loss"
    if key in ("phi", "pi", "P", "loss", "ppo", "shift", "metra", "P_inc", "P_succ"):
        return "Training objectives", "Loss"
    if key.startswith(("phi_", "pi_")):
        return f"{key.split('_')[0]} loss components", "Loss"
    return readable(key), "Recorded value"


def readable(metric: str) -> str:
    for prefix in (
        "components.",
        "preference.",
        "gradients.groups.",
        "updates.groups.",
        "policy.",
    ):
        metric = metric.removeprefix(prefix)
    return metric.replace("_", " ").replace(".", " / ")
