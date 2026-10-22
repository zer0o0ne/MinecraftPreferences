"""(9) of specs/perception.md: `PhiStream` fed with a clip's frames v_s (the cursor drawn)
and lines gives the rows of phi(clip) on the CPU with float64 parameters within float32's
tolerance; the restart before state k = L (empty cache, x_0 and z_0 from the new first
line); z_k of shape (D,), the head's dtype, without gradient."""

import torch

from mcpref.perception.phi import Phi
from mcpref.perception.stream import PhiStream

from . import oracle
from .world import F64, backbone, config, random_clip, runs_backbone, tokenizer

pytestmark = runs_backbone
F32 = {"rtol": 1.3e-6, "atol": 1e-5}  # torch.testing's float32 tolerance


def eval_phi(tmp_path) -> Phi:
    return Phi(backbone(), tokenizer(), config(tmp_path)).to(F64).eval()


def feed(stream: PhiStream, clip) -> torch.Tensor:
    """Steps on v_s (oracle frames, cursor drawn where g_s) and the lines of the states."""
    zs = []
    for s, v in enumerate(oracle.frames(clip)):
        z = stream.step(v, oracle.state_line(clip, s))
        assert z.shape == (4,) and z.dtype == F64 and not z.requires_grad
        zs.append(z)
    return torch.stack(zs)


def test_stream_equals_phi_of_the_clip(tmp_path):
    """L = 4 states of a clip with gaps and GUI states."""
    phi = eval_phi(tmp_path)
    c = random_clip(6, [2, 3, 5, 6], [True, False, True, True])
    want = phi([c])[0]
    torch.testing.assert_close(feed(PhiStream(phi, 4), c), want, **F32)


def test_stream_restarts_after_L(tmp_path):
    """2L steps: states L..2L-1 are a new stream (another clip, other positions): they give
    phi of that clip alone; then a third stream again."""
    phi = eval_phi(tmp_path)
    a = random_clip(7, [0, 1, 2], [False, True, False])
    b = random_clip(8, [10, 12, 13], [True, False, False])
    stream = PhiStream(phi, 3)
    torch.testing.assert_close(feed(stream, a), phi([a])[0], **F32)
    torch.testing.assert_close(feed(stream, b), phi([b])[0], **F32)
    torch.testing.assert_close(feed(stream, a), phi([a])[0], **F32)


def test_stream_without_grad_in_train_graph(tmp_path):
    """Under torch.no_grad: no parameter gets a gradient path through z."""
    phi = eval_phi(tmp_path)
    c = random_clip(9, [0, 1], [False, False])
    z = feed(PhiStream(phi, 2), c)
    assert z.grad_fn is None
