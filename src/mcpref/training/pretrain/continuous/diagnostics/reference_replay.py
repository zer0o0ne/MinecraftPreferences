"""Fresh chronological prefixes with bounded recurrent and endpoint memory."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from tqdm import tqdm

from mcpref.models.perception.continuous import observation_step
from mcpref.training.pretrain.continuous.data.reader import StreamReader
from mcpref.training.pretrain.continuous.diagnostics.observations import OwnSummary
from mcpref.training.pretrain.continuous.events import WindowResult
from mcpref.training.pretrain.continuous.state import Stream
from mcpref.training.pretrain.continuous.steps import own_losses


@contextmanager
def evaluation_state(cfg, modules):
    """Restore each submodule's mode, buffers and device RNG even after failure."""
    all_modules = {
        id(child): child for module in modules.values() for child in module.modules()
    }
    modes = [(module, module.training) for module in all_modules.values()]
    buffers = [
        (buffer, buffer.detach().clone())
        for module in modules.values()
        for buffer in module.buffers()
    ]
    device = torch.device(cfg.device)
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    cpu = torch.get_rng_state()
    try:
        for module in modules.values():
            module.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, training in modes:
            module.training = training
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)


def prefix_plan(index, selection):
    selected = {edge[side] for edge in selection["items"] for side in ("a", "b")}
    plan = {}
    for key in sorted(selected):
        fragment, completion = index.db.execute(
            "SELECT fragment,completion FROM endpoints WHERE id=?", (key,)
        ).fetchone()
        plan.setdefault(fragment, {})[key] = completion
    return plan


def _fragment(context, reader, fragment, needed, summary):
    stream, histories = Stream(), {}
    for observation in reader.observations(fragment):
        z, stream.recurrent = observation_step(
            context.modules["phi"],
            observation.frame,
            observation.line,
            stream.recurrent,
        )
        result = WindowResult()
        own_losses(context, stream, observation, z, result)
        summary.add(result)
        stream.remember(observation.id, z, context.modules["P"].history)
        for endpoint in observation.endpoints:
            if endpoint["id"] in needed:
                histories[endpoint["id"]] = (
                    torch.stack(
                        [stream.recent[key] for key in endpoint["observation_ids"]]
                    )
                    .detach()
                    .to("cpu", copy=True)
                )
        stream.previous = {
            "id": observation.id,
            "z": z,
            "frame": observation.frame,
            "meta": observation.meta,
            "interval": observation.interval,
        }
        if len(histories) == len(needed):
            return histories
    raise ValueError("Selected reference endpoint was not completed by its fragment")


def replay(cfg, modules, index, selection):
    plan = prefix_plan(index, selection)
    summaries = {part: OwnSummary(cfg) for part in ("train", "validation")}
    reader = StreamReader(index, modules["phi"].cfg, modules["phi"].sprite)
    histories, context = {}, SimpleNamespace(modules=modules)
    if not plan:
        print("Continuous reference prefixes: empty", file=sys.stderr)
    for key, needed in tqdm(
        sorted(plan.items()),
        total=len(plan),
        desc="Continuous reference prefixes",
        unit="fragment",
    ):
        part, body = index.db.execute(
            "SELECT part,body FROM fragments WHERE id=?", (key,)
        ).fetchone()
        histories.update(
            _fragment(context, reader, json.loads(body), needed, summaries[part])
        )
    return histories, {
        part: summary.finish(modules) for part, summary in summaries.items()
    }
