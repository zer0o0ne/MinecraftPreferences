"""Accumulate token cotangents before materializing the dense vocabulary gradient."""

import torch


class EmbeddingGradients:
    def __init__(self, embedding):
        self.embedding = embedding
        self.pending = []

    def lookup(self, tokens):
        with torch.no_grad():
            value = self.embedding(tokens)
        value.requires_grad_(self.embedding.weight.requires_grad)
        if value.requires_grad:
            self.pending.append((tokens, value))
        return value

    def flush(self):
        active = [
            (ids, value.grad) for ids, value in self.pending if value.grad is not None
        ]
        if active:
            ids = torch.cat([ids.flatten() for ids, _ in active])
            gradients = torch.cat([g.flatten(0, -2) for _, g in active])
            self.embedding(ids).backward(gradients)
        self.pending.clear()
