"""Tree(real), quantisation, histories and re-rooting against specs/mcts.md (1), (7)."""

import dataclasses

import pytest
import torch

from mcpref.runtime.mcts.tree import (
    LatentHistory,
    Node,
    Tree,
    advance,
    history,
    model_history,
    quantize,
)
from runtime.mcts.helpers import assert_values, latent_history, node


def test_tree_real_initialises_public_state():
    real = latent_history([[1.0, 2.0], [3.0, 4.0]])
    tree = Tree(real)
    assert [field.name for field in dataclasses.fields(LatentHistory)] == ["Y", "C"]
    assert [field.name for field in dataclasses.fields(Node)] == [
        "z", "x", "action", "prior", "shift", "parent", "children", "n", "mean", "value",
        "policy",
    ]  # fmt: skip
    with pytest.raises(dataclasses.FrozenInstanceError):
        real.Y = real.C
    assert tree.base is real
    assert tree.root.parent is None
    assert tree.root.action is tree.root.prior is tree.root.shift is None
    assert tree.root.children == []
    assert (tree.root.n, tree.root.mean, tree.root.value) == (0, 0.0, 0.0)
    assert_values(tree.root.z, real.Y[0, -1])
    assert_values(tree.root.x, real.Y[0, -1])


def test_quantize_positive_negative_halves_and_metadata():
    delta = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49, -0.49])
    actual = quantize(delta, torch.ones(8, dtype=torch.float64), 1.0)
    # torch.round ties to even: .5 -> 0, 1.5 -> 2, 2.5 -> 2, likewise below zero.
    expected = torch.tensor([0.0, 2.0, 2.0, -0.0, -2.0, -2.0, 0.0, -0.0])
    assert_values(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantize_shape_dtype_and_device(dtype):
    delta = torch.tensor([[[0.06, -0.06], [0.14, -0.14]]], dtype=dtype)
    actual = quantize(delta, torch.tensor([1.0, 2.0], dtype=dtype), 0.1)
    expected = torch.tensor([[[0.1, -0.05], [0.1, -0.15]]], dtype=dtype)
    assert actual.shape == delta.shape
    assert actual.dtype == delta.dtype
    assert actual.device == delta.device
    assert_values(actual, expected)


def test_history_appends_working_latents_and_one_hot_actions():
    real = latent_history([[1.0], [2.0]])
    tree = Tree(real)
    child = node([3.0], x=[30.0], action=2, parent=tree.root)
    grandchild = node([4.0], x=[40.0], action=1, parent=child)
    tree.root.children = [child]
    child.children = [grandchild]
    result = history(tree, grandchild)
    assert_values(result.Y, torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]))
    assert result.C.shape == (1, 4, 5)
    assert_values(result.C[:, :2], real.C)
    assert_values(
        result.C[0, 2],
        torch.nn.functional.one_hot(torch.tensor(2), 5).to(result.C.dtype),
    )
    assert_values(
        result.C[0, 3],
        torch.nn.functional.one_hot(torch.tensor(1), 5).to(result.C.dtype),
    )
    assert_values(real.Y, torch.tensor([[[1.0], [2.0]]]))


def test_advance_prefers_executed_match_and_preserves_subtree():
    real = latent_history([[0.0, 0.0]])
    tree = Tree(real)
    other = node([0.2, 0.3], x=[0.24, 0.34], action=0, parent=tree.root)
    executed = node(
        [0.2, 0.3], x=[0.24, 0.34], action=1, prior=0.4, shift=[0.2, 0.3],
        parent=tree.root, n=3, mean=0.7, value=1.2,
    )  # fmt: skip
    descendant = node([0.4, 0.5], action=2, parent=executed, n=2, mean=0.1, value=0.3)
    executed.children = [descendant]
    tree.root.children = [other, executed]
    base = history(tree, executed)
    observed = latent_history([[8.0, 8.0], [0.25, 0.35]])
    result = advance(tree, 1, observed, torch.ones(2, dtype=torch.float64), 0.5)
    assert result is tree
    assert tree.root is executed
    assert executed.parent is None
    assert executed.action == 1
    assert executed.children == [descendant]
    assert (executed.n, executed.mean, executed.value) == (3, 0.7, 1.2)
    assert_values(executed.z, torch.tensor([0.2, 0.3]))
    assert_values(executed.x, torch.tensor([0.24, 0.34]))
    assert_values(executed.shift, torch.tensor([0.2, 0.3]))
    assert_values(tree.base.Y, base.Y)
    assert_values(tree.base.C, base.C)


def test_advance_fallback_restores_real_action_and_last_c_row():
    real = latent_history([[0.0], [0.0]])
    tree = Tree(real)
    missed = node([2.0], action=0, parent=tree.root)
    reused = node(
        [0.2], x=[0.2], action=1, prior=0.3, shift=[0.2], parent=tree.root,
        n=4, mean=-0.2, value=0.8,
    )  # fmt: skip
    later = node([0.2], action=2, parent=tree.root)
    tree.root.children = [missed, reused, later]
    old = history(tree, reused)
    observed_C = torch.zeros(1, 2, 5)
    observed_C[0, 0, 4] = 1.0
    observed_C[0, 1, 3] = 1.0
    observed = latent_history([[7.0], [0.2]], observed_C)
    advance(tree, 0, observed, torch.ones(1, dtype=torch.float64), 0.5)
    assert tree.root is reused
    assert reused.parent is None
    assert reused.action == 0
    assert (reused.prior, reused.n, reused.mean, reused.value) == (0.3, 4, -0.2, 0.8)
    assert_values(reused.shift, torch.tensor([0.2]))
    assert_values(tree.base.Y, old.Y)
    assert_values(tree.base.C[:, :-1], old.C[:, :-1])
    assert_values(tree.base.C[:, -1], observed.C[:, -1])


def test_advance_no_match_creates_empty_real_root():
    tree = Tree(latent_history([[0.0]]))
    tree.root.children = [node([2.0], action=0, parent=tree.root)]
    observed = latent_history([[5.0], [0.4]])
    result = advance(tree, 3, observed, torch.ones(1, dtype=torch.float64), 0.5)
    assert result is tree
    assert tree.base is observed
    assert tree.root.parent is None
    assert tree.root.action == 3
    assert tree.root.prior is tree.root.shift is None
    assert tree.root.children == []
    assert (tree.root.n, tree.root.mean) == (0, 0.0)
    assert_values(tree.root.z, observed.Y[0, -1])
    assert_values(tree.root.x, observed.Y[0, -1])


def test_advance_accumulates_exact_residuals_until_rebase():
    tree = Tree(latent_history([[0.0]]))
    first = node([0.4], x=[0.4], action=0, parent=tree.root)
    second = node([0.8], x=[0.8], action=0, parent=first)
    first.children = [second]
    tree.root.children = [first]
    observed = latent_history([[0.0]])
    advance(tree, 0, observed, torch.ones(1, dtype=torch.float64), 1.0)
    assert tree.root is first
    advance(tree, 0, observed, torch.ones(1, dtype=torch.float64), 1.0)
    assert tree.root is not second
    assert_values(tree.root.x, torch.zeros(1))
    assert_values(tree.root.z, torch.zeros(1))


def test_advance_switches_root_policy_to_the_new_root():
    tree = Tree(latent_history([[0.0]]))
    tree.root_policy = [0.1, 0.9]
    child = node([0.4], x=[0.4], action=0, parent=tree.root)
    child.policy = [0.6, 0.4]
    tree.root.children = [child]
    advance(tree, 0, latent_history([[0.0]]), torch.ones(1), 1.0)
    assert tree.root is child and tree.root_policy == [0.6, 0.4]
    advance(tree, 1, latent_history([[5.0]]), torch.ones(1), 0.1)
    assert tree.root is not child and tree.root_policy is None


def oracle_history(real, path):
    Y, C = real.Y, real.C
    for child in path:
        row = torch.zeros(1, 1, C.shape[-1], dtype=C.dtype)
        row[0, 0, child.action] = 1.0
        Y = torch.cat((Y, child.z.reshape(1, 1, -1)), dim=1)
        C = torch.cat((C, row), dim=1)
    return Y, C


def make_path(tree, length):
    path = []
    parent = tree.root
    for i in range(length):
        child = node(
            [100.0 + i, 200.0 - i],
            x=[1000.0 + i, 2000.0 - i],
            action=(2 * i + 1) % 5,
            parent=parent,
        )
        parent.children = [child]
        path.append(child)
        parent = child
    return path


@pytest.mark.parametrize(
    ("base_rows", "path_rows", "L_pi", "L_P"),
    [(9, 7, 3, 2), (2, 8, 4, 6), (8, 2, 20, 1), (5, 0, 2, 3)],
)
def test_model_history_is_exact_independent_full_history_tail(
    base_rows, path_rows, L_pi, L_P
):
    real = latent_history([[float(i), -float(i)] for i in range(base_rows)], actions=5)
    tree = Tree(real)
    path = make_path(tree, path_rows)
    leaf = path[-1] if path else tree.root
    base_before = (real.Y.clone(), real.C.clone())
    expected_Y, expected_C = oracle_history(real, path)
    n = min(max(L_pi, L_P + 1), base_rows + path_rows)

    result = model_history(tree, leaf, L_pi, L_P)

    torch.testing.assert_close(result.Y, expected_Y[:, -n:], rtol=0, atol=0)
    torch.testing.assert_close(result.C, expected_C[:, -n:], rtol=0, atol=0)
    full = history(tree, leaf)
    torch.testing.assert_close(full.Y, expected_Y, rtol=0, atol=0)
    torch.testing.assert_close(full.C, expected_C, rtol=0, atol=0)
    assert tree.base is real
    torch.testing.assert_close(real.Y, base_before[0], rtol=0, atol=0)
    torch.testing.assert_close(real.C, base_before[1], rtol=0, atol=0)


def test_advance_after_model_tail_keeps_complete_long_base():
    real = latent_history([[float(i), -float(i)] for i in range(8)], actions=5)
    tree = Tree(real)
    path = make_path(tree, 7)
    leaf_Y, leaf_C = oracle_history(real, path)
    assert model_history(tree, path[-1], 3, 2).Y.shape[1] == 3
    expected_Y, expected_C = oracle_history(real, path[:1])
    observed = latent_history([path[0].x.tolist()], actions=5)

    advance(tree, path[0].action, observed, torch.ones(2, dtype=torch.float64), 0.1)

    assert tree.root is path[0]
    assert tree.base.Y.shape[1] == real.Y.shape[1] + 1
    torch.testing.assert_close(tree.base.Y, expected_Y, rtol=0, atol=0)
    torch.testing.assert_close(tree.base.C, expected_C, rtol=0, atol=0)
    full = history(tree, path[-1])
    torch.testing.assert_close(full.Y, leaf_Y, rtol=0, atol=0)
    torch.testing.assert_close(full.C, leaf_C, rtol=0, atol=0)
