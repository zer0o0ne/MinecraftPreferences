"""Bounded interval/target statistics, without classical clip denominators."""

import torch

from mcpref.diagnostics.pretrain.diagnostics import (
    PhiObservations,
    PolicyObservations,
    phi_suite,
    policy_suite,
)
from mcpref.diagnostics.pretrain.observation import phi_observation, policy_observation


def scalar(value):
    return value.detach().cpu().item() if isinstance(value, torch.Tensor) else value


def policy_rows(result, cfg):
    if not result.ce_rows:
        return None
    rows = result.ce_rows
    logits = torch.cat([row["logits"] for row in rows], dim=0)[None].float().cpu()
    q = torch.cat([row["targets"] for row in rows], dim=0)[None].float().cpu()
    current = torch.cat([row["previous"] for row in rows], dim=0)[None].float().cpu()
    valid = {
        "action_valid": q.sum(dim=-1) > 0,
        "shift_valid": torch.zeros(q.shape[:2], dtype=torch.bool),
    }
    return policy_observation(
        (
            logits,
            q,
            current,
            valid,
            torch.zeros(q.shape[:2]),
            (
                torch.cat([row["change_probability"] for row in rows])[None].cpu(),
                torch.tensor([[row["previous_known"] for row in rows]]),
            ),
        ),
        cfg.calibration_bins,
        cfg.action_copy_eps,
    )[0]


class OwnSummary:
    """One fixed-size accumulator for any number of chronological windows."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.counts = dict.fromkeys(("phi", "ce", "shift", "P"), 0)
        self.sums = {}
        self.phi = {key: 0 for key in PhiObservations.__dataclass_fields__}
        self.policy = None
        self.shift_valid = 0
        self.zero_shift = 0.0

    def add(self, result):
        for key, count in result.counts.items():
            self.counts[key] += count
        for key, value in result.sums.items():
            self.sums[key] = self.sums.get(key, 0.0) + scalar(value)
        for row in result.phi_rows:
            for key, value in phi_observation(row).items():
                self.phi[key] += scalar(value)
        policy = policy_rows(result, self.cfg)
        if policy is not None:
            policy = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in policy.items()
            }
            if self.policy is None:
                self.policy = policy
            else:
                for key, value in policy.items():
                    self.policy[key] = (
                        self.policy[key] & value
                        if key == "copy_finite"
                        else self.policy[key] + value
                    )
        self.shift_valid += sum(int(row["valid"]) for row in result.shift_rows)
        self.zero_shift += sum(scalar(row["zero_shift"]) for row in result.shift_rows)

    def policy_observations(self, actions):
        if self.policy is None:
            if not self.counts["shift"]:
                return None
            bins = self.cfg.calibration_bins
            return PolicyObservations(
                torch.zeros(actions, dtype=torch.float64),
                0,
                self.shift_valid,
                0.0,
                0,
                0.0,
                torch.zeros(bins, dtype=torch.int64),
                torch.zeros(bins, dtype=torch.float64),
                torch.zeros(bins, dtype=torch.float64),
                None,
                None,
                self.zero_shift / self.counts["shift"],
            )
        raw = self.policy
        fields = {
            key: scalar(raw[key])
            for key in ("action_valid", "confidence_sum", "correct_sum", "entropy_sum")
        }
        fields |= {
            key: raw[key]
            for key in (
                "q_sum",
                "calibration_count",
                "calibration_confidence",
                "calibration_correct",
            )
        }
        fields |= {
            "shift_valid": self.shift_valid,
            "uniform": scalar(raw["uniform"]) / self.counts["ce"],
            "copy_current": scalar(raw["copy_current"]) / self.counts["ce"]
            if raw["copy_finite"]
            else None,
            "zero_shift": self.zero_shift / self.counts["shift"]
            if self.counts["shift"]
            else None,
            "change_counts": raw["change_counts"],
        }
        return PolicyObservations(**fields)

    def finish(self, modules):
        losses = {}
        for prefix, keys, population in (
            ("phi", ("total", "push", "constraint", "dual", "norm"), "phi"),
            ("pi", ("action", "change"), "ce"),
            ("pi", ("shift",), "shift"),
            ("P", ("total", "inc", "succ"), "P"),
        ):
            for key in keys:
                name = prefix if key == "total" else f"{prefix}_{key}"
                count = self.counts[population]
                losses[name] = (
                    self.sums.get(f"{prefix}_{key}", 0.0) / count if count else None
                )
        pi = [
            losses[key] for key in ("pi_action", "pi_shift") if losses[key] is not None
        ]
        losses["pi"] = sum(pi) if pi else None
        phi = phi_suite([PhiObservations(**self.phi)])
        phi["dual_lambda"] = scalar(modules["phi_loss"].log_lambda.exp())
        policy = (
            self.policy_observations(len(modules["pi"].names))
            if self.policy is not None or self.counts["shift"]
            else None
        )
        return {
            "counts": dict(self.counts),
            "losses": losses,
            "phi": phi,
            "pi": policy_suite(
                [policy],
                calibration_bins=self.cfg.calibration_bins,
                copy_eps=self.cfg.action_copy_eps,
                full=True,
            ),
        }
