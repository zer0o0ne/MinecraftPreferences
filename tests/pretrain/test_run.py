"""`check_lengths` and `build_modules` of src/mcpref/pretrain/run.py against specs/policy.md
(13): each check, its message (X as written) and order; the modules in module order with and
without pairs, built from the configs (a fake `build_phi`: no download), on the device."""

import pytest
import torch
from perception.world import config as perception_config
from policy.policy_oracle import alpha_metric
from policy.policy_oracle import cfg_of as policy_cfg
from policy.policy_oracle import loss as policy_loss
from policy.policy_world import random_clips
from preferences.preference_oracle import cfg_of as preference_cfg

import mcpref.perception.phi
import mcpref.pretrain.run
from mcpref.perception.loss import PhiLoss
from mcpref.policy.loss import PolicyLoss
from mcpref.policy.network import Policy
from mcpref.preferences.loss import PreferenceLoss
from mcpref.preferences.network import Preference
from mcpref.pretrain.run import build_modules, check_lengths

from .loop_modules import NAMES
from .loop_starting import PhiStart

X = "w/x/"
LENGTHS = {  # L, L_pi, L_P, paired -> the message, None when the checks pass
    "equal": (3, 3, 3, True, None),
    "L_P below L": (5, 5, 2, True, None),
    "L_P above L without pairs": (5, 5, 9, False, None),
    "L = 2": (2, 2, 2, True, f"{X} has clips of 2 states, the policy needs at least 3"),
    "L = 1, L_pi and L_P other": (1, 4, 9, True, f"{X} has clips of 1 states, the policy needs at least 3"),
    "L_pi above L": (4, 5, 2, False, f"the policy history 5 differs from the clip length 4 of {X}"),
    "L_pi below L": (4, 3, 2, True, f"the policy history 3 differs from the clip length 4 of {X}"),
    "L_pi before L_P": (4, 5, 9, True, f"the policy history 5 differs from the clip length 4 of {X}"),
    "L_P above L": (4, 4, 5, True, f"the preference history 5 exceeds the clip length 4 of {X}"),
}  # fmt: skip


@pytest.mark.parametrize("L, l_pi, l_p, paired, message", LENGTHS.values(), ids=LENGTHS)
def test_check_lengths(L, l_pi, l_p, paired, message):
    args = (L, policy_cfg(history=l_pi), preference_cfg(history=l_p), paired, X)
    if message is None:
        assert check_lengths(*args) is None
    else:
        with pytest.raises(RuntimeError) as info:
            check_lengths(*args)
        assert type(info.value) is RuntimeError and str(info.value) == message


POLICY = policy_cfg(history=5)
PREFERENCE = preference_cfg(history=4)
D = 2  # len(alpha)


@pytest.fixture
def built(tmp_path, monkeypatch) -> list:
    """build_phi faked where run.py may look it up: the configs it was given."""
    calls = []

    def build_phi(cfg):
        calls.append(cfg)
        return PhiStart()

    monkeypatch.setattr(mcpref.perception.phi, "build_phi", build_phi)
    monkeypatch.setattr(mcpref.pretrain.run, "build_phi", build_phi, raising=False)
    return calls


def test_modules_with_pairs(tmp_path, masks, built):
    """phi by build_phi(perception), PhiLoss(perception), Policy(policy, len(alpha), A),
    PolicyLoss(policy, masks, latent_metric(perception)), Preference(preference,
    len(alpha)), PreferenceLoss(), in module order."""
    perception = perception_config(tmp_path, alpha=[1.0, 2.0])
    mods = build_modules(perception, masks, POLICY, PREFERENCE, True, "cpu")
    assert list(mods) == list(NAMES)
    assert type(mods["phi"]) is PhiStart and built == [perception]
    assert type(mods["phi_loss"]) is PhiLoss
    kinds = {
        "pi": Policy,
        "pi_loss": PolicyLoss,
        "P": Preference,
        "P_loss": PreferenceLoss,
    }
    assert all(type(mods[k]) is v for k, v in kinds.items())
    fresh = {"pi": Policy(POLICY, D, masks.names), "P": Preference(PREFERENCE, D)}
    for k, net in fresh.items():
        state, want = mods[k].state_dict(), net.state_dict()
        assert list(state) == list(want) and all(
            torch.equal(state[n], want[n]) for n in want
        )
    assert mods["pi"].head.out_features == len(masks.names)


def test_policy_loss_of_the_configs(tmp_path, masks, built):
    """The policy's loss with the masks and d_alpha of the perception config (alpha (1, 2))
    and w_Delta of the policy config, against the oracle of tests/policy."""
    perception = perception_config(tmp_path, alpha=[1.0, 2.0])
    mods = build_modules(perception, masks, POLICY, PREFERENCE, False, "cpu")
    clips = random_clips(3, 4, 5)
    g = torch.Generator().manual_seed(3)
    lam = torch.randn(4, 5, len(masks.names), generator=g, dtype=torch.float64)
    sh, Z = (torch.randn(4, 5, D, generator=g, dtype=torch.float64) for _ in range(2))
    got = mods["pi_loss"]((lam, sh), Z, clips)
    metric = alpha_metric(torch.tensor([1.0, 2.0], dtype=torch.float64))
    want = policy_loss((lam, sh), Z, clips, masks.names, POLICY.shift_weight, metric)
    torch.testing.assert_close(got, want)


def test_modules_without_pairs(tmp_path, masks, built):
    perception = perception_config(tmp_path, alpha=[1.0, 2.0])
    mods = build_modules(perception, masks, POLICY, PREFERENCE, False, "cpu")
    assert list(mods) == list(NAMES[:4])


def test_modules_moved_to_the_device(tmp_path, masks, built):
    """Each module moved to torch.device(device) (here `meta`)."""
    perception = perception_config(tmp_path, alpha=[1.0, 2.0])
    mods = build_modules(perception, masks, POLICY, PREFERENCE, True, "meta")
    for name, module in mods.items():
        params = list(module.parameters())
        assert all(p.device == torch.device("meta") for p in params), name
