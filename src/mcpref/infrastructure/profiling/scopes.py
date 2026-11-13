"""Named host regions and input sizes, active only inside a profiled step."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps

import torch

ACTIVE = ContextVar("training_profile", default=None)
PREFIX = "mcpref::"


def phase(name: str):
    if ACTIVE.get() is None:
        return nullcontext()
    return torch.profiler.record_function(PREFIX + name)


def measured(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with phase(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def qwen_batch(lengths: list[int], frames: int) -> None:
    batches = ACTIVE.get()
    if batches is not None:
        padded = len(lengths) * max(lengths)
        batches.append(
            {
                "rows": len(lengths),
                "frames": frames,
                "token_lengths": lengths,
                "padded_tokens": padded,
                "padding_fraction": 1 - sum(lengths) / padded,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )


@contextmanager
def module_scopes(modules: dict):
    """Hook only the two uncompiled Qwen parents, not checkpointed/compiled layers."""
    handles = []
    opened = []
    backbone = getattr(modules.get("phi"), "backbone", None)

    def install(module, name):
        stack = []

        def before(*_):
            scope = phase(name)
            scope.__enter__()
            stack.append(scope)
            opened.append(scope)

        def after(*_):
            scope = stack.pop()
            opened.remove(scope)
            scope.__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))

    try:
        for attribute, name in (
            ("visual", "qwen.vision"),
            ("language_model", "qwen.language"),
        ):
            module = getattr(backbone, attribute, None)
            if module is not None:
                install(module, name)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for scope in reversed(opened):
            scope.__exit__(None, None, None)
