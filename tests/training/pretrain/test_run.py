"""`check_lengths` and `build_modules` of src/mcpref/training/pretrain/run.py against specs/policy.md
(13): each check, its message (X as written) and order; the modules in module order with and
without pairs, built from the configs (a fake `build_phi`: no download), latent initialization
scaled by row without changing other state, identity or RNG, then moved to the device."""

import pytest
import torch
from models.perception.world import config as perception_config
from models.policy.policy_oracle import alpha_metric
from models.policy.policy_oracle import cfg_of as policy_cfg
from models.policy.policy_oracle import loss as policy_loss
from models.policy.policy_world import random_clips
from models.preference.preference_oracle import cfg_of as preference_cfg

import mcpref.models.factory
import mcpref.models.perception.phi
from mcpref.models.factory import build_modules, check_lengths
from mcpref.models.perception.loss import PhiLoss
from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from mcpref.models.preference.loss import PreferenceLoss
from mcpref.models.preference.network import Preference
from training.pretrain.loop_modules import NAMES
from training.pretrain.loop_starting import PhiStart

X = "w/x/"
LENGTHS = {  # L, L_pi, L_P, paired -> the message, None when the checks pass
    "equal": (3, 3, 3, True, None),
    "L_P below L": (5, 5, 2, True, None),
    "L_P above L without pairs": (5, 5, 9, False, None),
    "L = 2": (2, 2, 2, True, f"{X} has clips of 2 states, the policy needs at least 3"),
    "L = 1, L_pi and L_P other": (1, 4, 9, True, f"{X} has clips of 1 states, the policy needs at least 3"),
    "L_pi above L": (4, 5, 2, False, f"the policy history 5 exceeds the clip length 4 of {X}"),
    "L_pi below L": (4, 3, 2, True, None),
    "L_pi one": (4, 1, 1, True, None),
    "L_pi before L_P": (4, 5, 9, True, f"the policy history 5 exceeds the clip length 4 of {X}"),
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


class PhiHead(PhiStart):
    def __init__(self, alpha):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(3, 4, dtype=torch.float64),
                torch.nn.Linear(4, len(alpha), dtype=torch.float64),
            )


def scale_row(row, alpha, kappa):
    factor = torch.div(
        torch.tensor(kappa, dtype=row.dtype), torch.tensor(alpha, dtype=row.dtype)
    )
    torch.mul(row, factor, out=row)


def policy_oracle(policy, names, alpha, kappa):
    net = Policy(policy, len(alpha), names)
    with torch.no_grad():
        for row, value in zip(net.dynamics.w2.weight, alpha, strict=True):
            scale_row(row, value, kappa)
    return net


@pytest.fixture
def built(tmp_path, monkeypatch) -> list:
    """build_phi faked where run.py may look it up: the configs it was given."""
    calls = []

    def build_phi(cfg):
        calls.append(cfg)
        return PhiHead(cfg.alpha)

    monkeypatch.setattr(mcpref.models.perception.phi, "build_phi", build_phi)
    monkeypatch.setattr(mcpref.models.factory, "build_phi", build_phi, raising=False)
    return calls


def test_modules_with_pairs(tmp_path, masks, built):
    """phi by build_phi(perception), PhiLoss(perception), Policy(policy, len(alpha), A),
    PolicyLoss(policy, masks, latent_metric(perception)), Preference(preference,
    len(alpha)), PreferenceLoss(), in module order."""
    alpha = [1.0, 2.0]
    perception = perception_config(tmp_path, alpha=alpha)
    mods = build_modules(perception, masks, POLICY, PREFERENCE, True, "cpu", 1.0)
    assert list(mods) == list(NAMES)
    assert type(mods["phi"]) is PhiHead and built == [perception]
    assert type(mods["phi_loss"]) is PhiLoss
    kinds = {
        "pi": Policy,
        "pi_loss": PolicyLoss,
        "P": Preference,
        "P_loss": PreferenceLoss,
    }
    assert all(type(mods[k]) is v for k, v in kinds.items())
    fresh = {
        "pi": policy_oracle(POLICY, masks.names, alpha, 1.0),
        "P": Preference(PREFERENCE, D),
    }
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
    mods = build_modules(perception, masks, POLICY, PREFERENCE, False, "cpu", 1.0)
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
    mods = build_modules(perception, masks, POLICY, PREFERENCE, False, "cpu", 1.0)
    assert list(mods) == list(NAMES[:4])


def test_modules_moved_to_the_device(tmp_path, masks, built):
    """Each module moved to torch.device(device) (here `meta`)."""
    perception = perception_config(tmp_path, alpha=[1.0, 2.0])
    mods = build_modules(perception, masks, POLICY, PREFERENCE, True, "meta", 1.0)
    for name, module in mods.items():
        params = list(module.parameters())
        assert all(p.device == torch.device("meta") for p in params), name


def test_latent_init_scale_is_required(tmp_path, masks):
    perception = perception_config(tmp_path, alpha=[1.0, 2.0])
    with pytest.raises(TypeError, match="latent_init_scale"):
        build_modules(perception, masks, POLICY, PREFERENCE, False, "cpu")


def test_latent_init_exact_rows_other_state_identity_and_rng(
    tmp_path, masks, monkeypatch
):
    alpha = (1.3, 2.7, 4.1)
    kappa = 1.75
    perception = perception_config(tmp_path, alpha=alpha)
    phi = PhiHead(alpha)
    constructed = {"phi": phi}
    identities = {"phi": {name: id(p) for name, p in phi.named_parameters()}}

    def build_phi(cfg):
        return phi

    def recorder(name, constructor):
        def build(*args, **kwargs):
            module = constructor(*args, **kwargs)
            constructed[name] = module
            identities[name] = {n: id(p) for n, p in module.named_parameters()}
            return module

        return build

    monkeypatch.setattr(mcpref.models.perception.phi, "build_phi", build_phi)
    monkeypatch.setattr(mcpref.models.factory, "build_phi", build_phi, raising=False)
    constructors = {
        "phi_loss": PhiLoss,
        "pi": Policy,
        "pi_loss": PolicyLoss,
        "P": Preference,
        "P_loss": PreferenceLoss,
    }
    for name, constructor in constructors.items():
        monkeypatch.setattr(
            mcpref.models.factory,
            constructor.__name__,
            recorder(name, constructor),
        )
    rng = torch.get_rng_state()
    mods = build_modules(perception, masks, POLICY, PREFERENCE, True, "cpu", kappa)
    assert torch.equal(torch.get_rng_state(), rng)

    want_phi = PhiHead(alpha)
    with torch.no_grad():
        last = want_phi.head[-1]
        for row, value in zip(last.weight, alpha, strict=True):
            scale_row(row, value, kappa)
        for row, value in zip(last.bias, alpha, strict=True):
            scale_row(row, value, kappa)
    metric = alpha_metric(torch.tensor(alpha, dtype=torch.float64))
    want = {
        "phi": want_phi,
        "phi_loss": PhiLoss(perception),
        "pi": policy_oracle(POLICY, masks.names, alpha, kappa),
        "pi_loss": PolicyLoss(POLICY, masks, metric),
        "P": Preference(PREFERENCE, len(alpha)),
        "P_loss": PreferenceLoss(),
    }
    assert mods["pi"].dynamics.w2.bias is None
    for name, oracle in want.items():
        assert mods[name] is constructed[name]
        assert {n: id(p) for n, p in mods[name].named_parameters()} == identities[name]
        state, expected = mods[name].state_dict(), oracle.state_dict()
        assert list(state) == list(expected)
        assert all(torch.equal(state[n], expected[n]) for n in expected), name
