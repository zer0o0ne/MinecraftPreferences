"""Primitive tensor, observation, continuation, history and tree codecs of the spec."""

import copy

import torch
from torch import Tensor

from mcpref.runtime.env.state import EnvContinuation, EnvState
from mcpref.runtime.mcts.tree import LatentHistory, Node, Tree


def _cpu(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.detach().to(device="cpu", copy=True)


def _encode_observation(state: EnvState) -> dict:
    return {
        "frame": torch.from_numpy(state.frame).clone(),
        "line": copy.deepcopy(state.line),
        "gui": state.gui,
        "cls": state.cls,
        "hotbar": state.hotbar,
        "mask": torch.from_numpy(state.mask).clone(),
    }


def _decode_observation(value: dict) -> EnvState:
    return EnvState(
        value["frame"].numpy(),
        value["line"],
        value["gui"],
        value["cls"],
        value["hotbar"],
        value["mask"].numpy(),
    )


def _encode_continuation(continuation: EnvContinuation) -> dict:
    return {
        "info": {
            "isGuiOpen": continuation.info["isGuiOpen"],
            "custom": copy.deepcopy(continuation.info["custom"]),
            "mine_block": copy.deepcopy(continuation.info["mine_block"]),
            "use_item": copy.deepcopy(continuation.info["use_item"]),
        },
        "cursor": tuple(float(value) for value in continuation.cursor),
        "held_action": continuation.held_action,
    }


def _decode_continuation(value: dict) -> EnvContinuation:
    info = value["info"]
    return EnvContinuation(
        {
            "isGuiOpen": info["isGuiOpen"],
            "custom": copy.deepcopy(info["custom"]),
            "mine_block": copy.deepcopy(info["mine_block"]),
            "use_item": copy.deepcopy(info["use_item"]),
        },
        (float(value["cursor"][0]), float(value["cursor"][1])),
        value["held_action"],
    )


def _encode_history(history: LatentHistory) -> dict:
    return {"Y": _cpu(history.Y), "C": _cpu(history.C)}


def _decode_history(value: dict, device: torch.device) -> LatentHistory:
    return LatentHistory(value["Y"].to(device=device), value["C"].to(device=device))


def _encode_node(node: Node) -> dict:
    return {
        "z": _cpu(node.z),
        "x": _cpu(node.x),
        "action": node.action,
        "prior": node.prior,
        "shift": _cpu(node.shift),
        "children": [_encode_node(child) for child in node.children],
        "n": node.n,
        "mean": node.mean,
        "value": node.value,
        "policy": node.policy,
    }


def _decode_node(value: dict, device: torch.device, parent: Node | None) -> Node:
    node = Node(
        value["z"].to(device=device),
        value["x"].to(device=device),
        value["action"],
        value["prior"],
        _cpu(value["shift"]).to(device=device) if value["shift"] is not None else None,
        parent,
        n=value["n"],
        mean=value["mean"],
        value=value["value"],
        policy=value.get("policy"),
    )
    node.children = [_decode_node(child, device, node) for child in value["children"]]
    return node


def _encode_tree(tree: Tree) -> dict:
    return {
        "root": _encode_node(tree.root),
        "base": _encode_history(tree.base),
        "policy": tree.root_policy,
    }


def _decode_tree(value: dict, device: torch.device) -> Tree:
    tree = Tree(_decode_history(value["base"], device))
    tree.root = _decode_node(value["root"], device, None)
    tree.root_policy = value.get("policy")
    return tree
