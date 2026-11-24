"""Reuse observation graphs while applying a separate K-step VJP per loss."""

from dataclasses import dataclass, field

import torch

from mcpref.models.perception.continuous.observation import decode, encode, prepare
from mcpref.models.perception.continuous.state import ContinuousState, GDNState
from mcpref.models.perception.phi import Phi
from mcpref.training.pretrain.continuous.embedding_gradients import EmbeddingGradients
from mcpref.training.pretrain.continuous.replay import capture_rng, refresh, replay_rng
from mcpref.training.pretrain.continuous.state import detached


def tensors(state):
    return tuple(x for layer in state.layers for x in (layer.conv, layer.recurrent))


def state_leaf(state, trainable):
    if state is None:
        return None
    return ContinuousState(
        tuple(
            GDNState(*(x.detach().requires_grad_(trainable) for x in pair))
            for pair in ((layer.conv, layer.recurrent) for layer in state.layers)
        ),
        state.origin,
    )


@dataclass
class Graph:
    embeddings: torch.Tensor
    features: torch.Tensor
    incoming: ContinuousState | None
    state: ContinuousState
    output: torch.Tensor
    latent: torch.Tensor
    cotangents: dict[int, torch.Tensor] = field(default_factory=dict)

    def backward(self, carry, propagate):
        """Keep this event's cutoff; defer the linear parameter VJP until retirement."""
        outputs = (self.output, *tensors(self.state))
        gradients = (self.latent.grad, *carry)
        for i, gradient in enumerate(gradients):
            if gradient is not None:
                if i in self.cotangents:
                    self.cotangents[i].add_(gradient)
                else:
                    dtype = torch.promote_types(gradient.dtype, torch.float32)
                    self.cotangents[i] = gradient.detach().to(dtype=dtype, copy=True)
        active = [
            (x, g)
            for x, g in zip(outputs, gradients)
            if x.requires_grad and g is not None
        ]
        self.latent.grad = None
        incoming = () if self.incoming is None else tensors(self.incoming)
        if active and incoming and propagate:
            return torch.autograd.grad(
                [x for x, _ in active],
                incoming,
                [g for _, g in active],
                retain_graph=True,
                allow_unused=True,
            )
        return (None,) * len(incoming)

    def flush(self):
        """Apply the sum of all event cotangents once at the unchanged weights."""
        outputs = (self.output, *tensors(self.state))
        active = [
            (outputs[i], g)
            for i, g in self.cotangents.items()
            if outputs[i].requires_grad
        ]
        if active:
            torch.autograd.backward(
                [x for x, _ in active], [g.to(x) for x, g in active]
            )
        if self.features.grad is not None and self.embeddings.requires_grad:
            self.embeddings.backward(self.features.grad)


class SlidingGraphs:
    """Ephemeral graphs for one weight version; at most K encoders remain live."""

    def __init__(self, context):
        self.context = context
        self.graphs = {}
        self.active = []
        self.trainable = any(
            p.requires_grad for p in context.modules["phi"].parameters()
        )
        phi = context.modules["phi"]
        self.tokens = (
            EmbeddingGradients(phi.backbone.get_input_embeddings())
            if isinstance(phi, Phi)
            else None
        )

    def build(self, entry, state):
        phi = self.context.modules["phi"]
        prepared = prepare(
            phi,
            entry["frame"],
            entry["line"],
            entry["line"] if state is None else state.origin,
        )
        embeddings = (
            encode(phi, prepared)
            if self.tokens is None
            else encode(phi, prepared, self.tokens.lookup(prepared.tokens))
        )
        features = embeddings.detach().requires_grad_(self.trainable)
        incoming = state_leaf(state, self.trainable)
        z, updated = decode(phi, features, prepared, incoming)
        return Graph(
            embeddings,
            features,
            incoming,
            updated,
            z,
            z.detach().requires_grad_(self.trainable),
        )

    def observation(self, stream, current):
        device = torch.device(self.context.cfg.device)
        stream.replay.append(
            {
                "id": current.id,
                "frame": current.frame,
                "line": current.line,
                "rng": None,
            }
        )
        state, latents = stream.anchor, {}
        self.active = [entry["id"] for entry in stream.replay]
        for entry in stream.replay:
            key = entry["id"]
            if key not in self.graphs:
                if entry["rng"] is None:
                    entry["rng"] = capture_rng(device)
                    graph = self.build(entry, state)
                else:
                    with replay_rng(entry["rng"], device):
                        graph = self.build(entry, state)
                self.graphs[key] = graph
            graph = self.graphs[key]
            state, latents[key] = graph.state, graph.latent
        stream.recurrent = state
        refresh(stream, latents)
        if len(stream.replay) == self.context.cfg.tbptt_states:
            stream.anchor = detached(self.graphs[self.active[0]].state)
            stream.replay.pop(0)
        return latents[current.id]

    def backward(self, stream):
        carry = (None,) * len(tensors(self.graphs[self.active[-1]].state))
        for key in reversed(self.active):
            carry = self.graphs[key].backward(carry, key != self.active[0])
        keep = {entry["id"] for entry in stream.replay}
        for key in self.active:
            if key not in keep:
                self.graphs.pop(key).flush()

    def flush(self):
        for graph in self.graphs.values():
            graph.flush()
        self.graphs.clear()
        if self.tokens is not None:
            self.tokens.flush()


def reusable(context):
    phi = context.modules["phi"]
    if not isinstance(phi, Phi):
        return None
    if context.cfg.activation_checkpointing:
        phi.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        phi.backbone.gradient_checkpointing_disable()
    if not context.cfg.gradient_diagnostics:
        return SlidingGraphs(context)
    return None
