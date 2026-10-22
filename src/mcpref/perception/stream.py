"""(9) of specs/perception.md: phi's streaming step for R1, restarted every L states."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor
from transformers import DynamicCache

from mcpref.perception.meta import meta_text
from mcpref.perception.phi import Phi
from mcpref.perception.tokens import image_inputs, positions, segment


class PhiStream:
    """(9): the stream's sequence, grids, cache and first line; a new stream after phi
    changes."""

    def __init__(self, phi: Phi, states: int) -> None:
        self.phi, self.states = phi, states
        self.grids: list[Tensor] = []

    def restart(self, line: Mapping) -> None:
        """(9): empty cache and sequence; x_0, z_0 from `line`."""
        self.ids: list[int] = []
        self.grids = []
        self.cache = DynamicCache(config=self.phi.backbone.config)
        self.first = line

    @torch.no_grad()
    def step(self, frame: np.ndarray, line: Mapping) -> Tensor:
        """(9): z_k of the state (frame, line), shape (D,), in the head's dtype."""
        phi = self.phi
        if not self.grids or len(self.grids) == self.states:
            self.restart(line)
        pixels, grid = image_inputs(phi.processor, [frame])
        mu = meta_text(phi.cfg, line, self.first)
        sigma = segment(phi.tokenizer, phi.sp, grid[0], mu, phi.cfg.marker)
        start = len(self.ids)
        self.ids += sigma
        self.grids.append(grid)
        pos = positions(phi.backbone, self.ids, torch.cat(self.grids), phi.sp)
        h = phi.hidden(sigma, pixels, grid, pos[:, :, start:], self.cache)
        return phi.head(h[-1])
