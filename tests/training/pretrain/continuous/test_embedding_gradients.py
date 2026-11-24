"""One dense vocabulary VJP preserves token sums, frozen weights and idle windows."""

import pytest
import torch

from mcpref.training.pretrain.continuous.embedding_gradients import EmbeddingGradients


@pytest.mark.parametrize("frozen", [False, True])
def test_repeated_tokens_padding_and_updates_match_row_sum_oracle(frozen):
    embedding = torch.nn.Embedding(7, 3, padding_idx=0).double()
    embedding.requires_grad_(not frozen)
    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
    ids = [torch.tensor([[1, 0, 2]]), torch.tensor([[2, 1]])]
    gradients = [
        torch.tensor([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.0]]]),
        torch.tensor([[[3.0, 2.0, 1.0], [1.0, 1.0, 1.0]]]),
    ]
    expected = torch.zeros_like(embedding.weight)
    expected[1], expected[2] = (
        torch.tensor([2.0, 3.0, 4.0]),
        torch.tensor([5.0, 5.0, 5.0]),
    )
    cache = EmbeddingGradients(embedding)
    rng = torch.get_rng_state()
    for _ in range(2):
        before = embedding.weight.detach().clone()
        for tokens, gradient in zip(ids, gradients):
            output = cache.lookup(tokens)
            torch.testing.assert_close(output, before[tokens], rtol=0, atol=0)
            assert output.requires_grad != frozen
            if not frozen:
                output.backward(gradient)
            assert embedding.weight.grad is None
        cache.lookup(torch.tensor([[6]]))  # Observation with no downstream loss.
        cache.flush()
        assert not cache.pending
        if frozen:
            assert embedding.weight.grad is None
        else:
            torch.testing.assert_close(embedding.weight.grad, expected, rtol=0, atol=0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cache.flush()
        assert embedding.weight.grad is None
    assert torch.equal(rng, torch.get_rng_state())


def test_multiple_events_and_fragment_flush_accumulate_in_existing_gradient():
    embedding = torch.nn.Embedding(4, 2).double()
    cache = EmbeddingGradients(embedding)
    embedding.weight.grad = torch.ones_like(embedding.weight)
    for tokens in (torch.tensor([[1, 2]]), torch.tensor([[2, 3]])):
        leaf = cache.lookup(tokens)
        leaf.backward(torch.ones_like(leaf), retain_graph=True)
        leaf.backward(2 * torch.ones_like(leaf))
        cache.flush()
    expected = torch.tensor([[1.0, 1.0], [4.0, 4.0], [7.0, 7.0], [4.0, 4.0]])
    torch.testing.assert_close(embedding.weight.grad, expected.double(), rtol=0, atol=0)
