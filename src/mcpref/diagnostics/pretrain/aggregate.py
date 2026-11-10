"""Equal-step and pooled 100-step reductions of specs/pretrain-diagnostics.md."""

from mcpref.diagnostics.preference.suites import pool_preferences
from mcpref.diagnostics.pretrain.diagnostics import (
    BASELINE_KEYS,
    StepObservations,
    Stratum,
    mean_nullable,
    phi_suite,
    policy_suite,
    preference_suite,
)

COUNT_KEYS = (
    "items",
    "clips",
    "pairs",
    "comparable",
    "incomparable",
    "action_targets",
    "shift_targets",
    "phi_changed",
    "phi_state_pairs",
)
LOSS_PATHS = {key: (key,) for key in ("phi", "pi", "P", "loss")}
COMPONENT_PATHS = {
    "phi": ("push", "constraint", "dual", "norm"),
    "pi": ("action", "shift"),
    "P": ("inc", "succ"),
}


def at(value: dict | None, path: tuple[str, ...]):
    for key in path:
        if value is None:
            return None
        value = value[key]
    return value


def policy_coverage(observations: list[StepObservations]) -> dict:
    return {
        key: mean_nullable(
            getattr(observation.policy, key) for observation in observations
        )[1]
        for key in ("uniform", "copy_current", "zero_shift")
    }


class WindowAccumulator:
    """Interface: bounded consecutive observations, full drains and one final drain."""

    def __init__(
        self,
        window: int,
        *,
        strata: tuple[Stratum, ...],
        calibration_bins: int,
        saturation_threshold: float,
        inc_classification_threshold: float,
        quantiles: tuple[float, ...],
        action_copy_eps: float,
    ) -> None:
        self.window = window
        self.strata = strata
        self.calibration_bins = calibration_bins
        self.saturation_threshold = saturation_threshold
        self.inc_classification_threshold = inc_classification_threshold
        self.quantiles = quantiles
        self.action_copy_eps = action_copy_eps
        self.items: list[tuple[dict, StepObservations]] = []
        self.last_step: int | None = None
        self.final = False

    def add(self, step_record: dict, observations: StepObservations) -> None:
        if self.final:
            raise RuntimeError("cannot add after the final drain")
        if self.last_step is not None and step_record["step"] != self.last_step + 1:
            raise ValueError("diagnostic steps must be consecutive")
        self.last_step = step_record["step"]
        self.items.append((step_record, observations))

    def mean(
        self, records: list[dict], path: tuple[str, ...]
    ) -> tuple[float | None, int]:
        return mean_nullable(at(record, path) for record in records)

    def losses(self, records: list[dict]) -> tuple[dict, dict]:
        losses, coverage = {}, {}
        for key, path in LOSS_PATHS.items():
            losses[key], coverage[key] = self.mean(records, path)
        losses["components"], coverage["components"] = {}, {}
        for name, keys in COMPONENT_PATHS.items():
            losses["components"][name], coverage["components"][name] = {}, {}
            for key in keys:
                path = ("components", name, key)
                value, count = self.mean(records, path)
                losses["components"][name][key] = value
                coverage["components"][name][key] = count
        losses["baselines"], coverage["baselines"] = {}, {}
        for key in BASELINE_KEYS:
            value, count = self.mean(records, ("baselines", key))
            losses["baselines"][key], coverage["baselines"][key] = value, count
        return losses, {"losses": coverage}

    def counts(self, records: list[dict]) -> dict:
        result = {
            key: sum(record["counts"][key] for record in records) for key in COUNT_KEYS
        }
        result["sum_p"] = (
            sum(
                record["counts"]["sum_p"]
                for record in records
                if record["counts"]["sum_p"] is not None
            )
            if result["comparable"]
            else None
        )
        return result

    def gradients(self, records: list[dict]) -> dict:
        global_norm, active_steps = self.mean(records, ("gradients", "global", "norm"))
        global_clip = self.mean(records, ("gradients", "global", "clip_coefficient"))[0]
        groups = {}
        for group in records[0]["gradients"]["groups"]:
            norm, active = self.mean(records, ("gradients", "groups", group, "norm"))
            groups[group] = {"mean_norm": norm, "active_steps": active}
        heads = None
        if records[0]["gradients"]["P_heads"] is not None:
            heads = {}
            for head in records[0]["gradients"]["P_heads"]:
                path = ("gradients", "P_heads", head, "norm")
                norm, active = self.mean(records, path)
                heads[head] = {"mean_norm": norm, "active_steps": active}
        return {
            "global": {
                "mean_norm": global_norm,
                "mean_clip_coefficient": global_clip,
                "active_steps": active_steps,
            },
            "groups": groups,
            "P_heads": heads,
        }

    def updates(self, records: list[dict]) -> dict:
        groups = {}
        for group in records[0]["updates"]["groups"]:
            base = ("updates", "groups", group)
            norm, norm_steps = self.mean(records, (*base, "norm"))
            relative, relative_steps = self.mean(records, (*base, "relative"))
            unavailable = sum(
                at(record, (*base, "unavailable")) == "zero_decay_factor"
                for record in records
            )
            groups[group] = {
                "mean_norm": norm,
                "mean_relative": relative,
                "norm_steps": norm_steps,
                "relative_steps": relative_steps,
                "unavailable": {"zero_decay_factor": unavailable},
            }
        return {"groups": groups}

    def phi(self, records: list[dict], observations: list[StepObservations]) -> dict:
        result = {
            key: self.mean(records, ("components", "phi", key))[0]
            for key in COMPONENT_PATHS["phi"]
        }
        result["dual_lambda"] = self.mean(records, ("phi_stats", "dual_lambda"))[0]
        return result | phi_suite([observation.phi for observation in observations])

    def record(self, items: list[tuple[dict, StepObservations]]) -> dict:
        records = [record for record, _ in items]
        observations = [observation for _, observation in items]
        losses, coverage = self.losses(records)
        coverage["pi"] = policy_coverage(observations)
        return {
            "version": 1,
            "first_step": records[0]["step"],
            "last_step": records[-1]["step"],
            "partial": len(items) < self.window,
            "losses": losses,
            "coverage": coverage,
            "counts": self.counts(records),
            "preference": preference_suite(
                pool_preferences(
                    observation.preference for observation in observations
                ),
                strata=self.strata,
                calibration_bins=self.calibration_bins,
                saturation_threshold=self.saturation_threshold,
                inc_classification_threshold=self.inc_classification_threshold,
                quantiles=self.quantiles,
                full=True,
            ),
            "pi": policy_suite(
                [observation.policy for observation in observations],
                calibration_bins=self.calibration_bins,
                copy_eps=self.action_copy_eps,
                full=True,
            ),
            "phi": self.phi(records, observations),
            "gradients": self.gradients(records),
            "updates": self.updates(records),
        }

    def drain(self, *, final: bool = False) -> list[dict]:
        if self.final:
            raise RuntimeError("the final diagnostic drain may be called only once")
        self.final = final
        result = []
        while len(self.items) >= self.window:
            result.append(self.record(self.items[: self.window]))
            self.items = self.items[self.window :]
        if final and self.items:
            result.append(self.record(self.items))
            self.items = []
        return result
