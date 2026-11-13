"""Construction and compatibility checks of the agent's modules."""

import torch

from mcpref.gui.mask import Masks
from mcpref.models.perception.config import PerceptionConfig
from mcpref.models.perception.loss import PhiLoss, latent_metric
from mcpref.models.perception.phi import build_phi
from mcpref.models.perception.precision import enable_fp8_mlp
from mcpref.models.policy.config import PolicyConfig
from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from mcpref.models.preference.config import PreferenceConfig
from mcpref.models.preference.loss import PreferenceLoss
from mcpref.models.preference.network import Preference


def check_lengths(
    states: int,
    policy: PolicyConfig,
    preference: PreferenceConfig,
    paired: bool,
    index_dir: str,
) -> None:
    """(13): L >= 3, L_pi <= L, with pairs L_P <= L, in this order."""
    if states < 3:
        raise RuntimeError(
            f"{index_dir} has clips of {states} states, the policy needs at least 3"
        )
    if policy.history > states:
        raise RuntimeError(
            f"the policy history {policy.history} exceeds the clip length "
            f"{states} of {index_dir}"
        )
    if paired and preference.history > states:
        raise RuntimeError(
            f"the preference history {preference.history} exceeds the clip length "
            f"{states} of {index_dir}"
        )


def _scale_linear(
    layer: torch.nn.Linear, alphas: tuple[float, ...], latent_init_scale: float
) -> None:
    """Scale latent output rows by latent_init_scale / alpha."""
    with torch.no_grad():
        weight = layer.weight
        alpha = torch.tensor(alphas, dtype=weight.dtype, device=weight.device)
        factor = torch.div(latent_init_scale, alpha, rounding_mode=None, out=None)
        torch.mul(weight, factor[:, None], out=weight)
        if layer.bias is not None:
            bias = layer.bias
            alpha = torch.tensor(alphas, dtype=bias.dtype, device=bias.device)
            factor = torch.div(latent_init_scale, alpha, rounding_mode=None, out=None)
            torch.mul(bias, factor, out=bias)


def build_modules(
    perception: PerceptionConfig,
    masks: Masks,
    policy: PolicyConfig,
    preference: PreferenceConfig,
    paired: bool,
    device: str,
    latent_init_scale: float,
) -> dict[str, torch.nn.Module]:
    """(13): phi, phi_loss, pi, pi_loss (with pairs P, P_loss), in module order, each
    moved to `device` with BF16 parameters after seeded initialization/scaling."""
    D, target = len(perception.alpha), torch.device(device)
    phi = build_phi(perception)
    pi = Policy(policy, D, masks.names)
    _scale_linear(phi.head[-1], perception.alpha, latent_init_scale)
    _scale_linear(pi.dynamics.w2, perception.alpha, latent_init_scale)
    modules = {"phi": phi, "phi_loss": PhiLoss(perception), "pi": pi}
    metric = latent_metric(perception)
    modules["pi_loss"] = PolicyLoss(policy, masks, metric)
    if paired:
        modules["P"] = Preference(preference, D)
        modules["P_loss"] = PreferenceLoss()
    modules = {
        name: _bf16_parameters(module, target) for name, module in modules.items()
    }
    if perception.fp8_mlp:
        enable_fp8_mlp(phi.backbone)
    return modules


def _bf16_parameters(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Keep nontrainable buffers, including Qwen's FP32 RoPE constants, exact."""
    buffers = dict(module.named_buffers())
    module.to(device=device, dtype=torch.bfloat16)
    for name, buffer in buffers.items():
        parent, _, field = name.rpartition(".")
        setattr(module.get_submodule(parent), field, buffer.to(device=device))
    return module
