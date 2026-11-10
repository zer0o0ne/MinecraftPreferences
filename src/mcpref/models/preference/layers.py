"""(5), (6) of specs/preference.md: the layers S and X on the shared transformer parts."""

import torch
from torch import Tensor

from mcpref.models.transformer.layers import Layer


class LayerS(Layer):
    """(5): each argument attends to itself, the same parameters for both."""

    def forward(
        self, A: Tensor, B: Tensor, pos_a: Tensor, pos_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        A_hat, B_hat = self.attn_norm(A), self.attn_norm(B)
        if A.shape[1] == B.shape[1]:
            H, H_hat = torch.cat([A, B], dim=0), torch.cat([A_hat, B_hat], dim=0)
            return self.update(H, H_hat, H_hat, pos_a, pos_a, causal=False).chunk(
                2, dim=0
            )
        return (
            self.update(A, A_hat, A_hat, pos_a, pos_a, causal=False),
            self.update(B, B_hat, B_hat, pos_b, pos_b, causal=False),
        )


class LayerX(Layer):
    """(6): each argument attends to the other from the layer's inputs, one set of
    parameters for both directions."""

    def forward(
        self, A: Tensor, B: Tensor, pos_a: Tensor, pos_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        A_hat, B_hat = self.attn_norm(A), self.attn_norm(B)
        if A.shape[1] == B.shape[1]:
            H = torch.cat([A, B], dim=0)
            H_hat = torch.cat([A_hat, B_hat], dim=0)
            G_hat = torch.cat([B_hat, A_hat], dim=0)
            return self.update(H, H_hat, G_hat, pos_a, pos_b, causal=False).chunk(
                2, dim=0
            )
        return (
            self.update(A, A_hat, B_hat, pos_a, pos_b, causal=False),
            self.update(B, B_hat, A_hat, pos_b, pos_a, causal=False),
        )
