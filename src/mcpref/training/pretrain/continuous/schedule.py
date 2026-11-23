"""New-observation batches cross fragments; memory never does."""

from mcpref.infrastructure.profiling.scopes import phase


def epoch_updates(index, cfg):
    count = sum(item["observations"] for item in index.fragments("train"))
    return (count + cfg.batch_size - 1) // cfg.batch_size


def observations(context, reader, fragments, position):
    while position["fragment_index"] < len(position["order"]):
        descriptor = fragments[position["order"][position["fragment_index"]]]
        context.fragment = descriptor
        for item in reader.observations(descriptor, position["observation_index"]):
            position["observation_index"] += 1
            yield item
        if position["observation_index"] != descriptor["observations"]:
            raise RuntimeError("Continuous fragment ended before its indexed cursor")
        position["observation_index"] = 0
        position["fragment_index"] += 1


def batch(source, size):
    for _ in range(size):
        with phase("data.stream_read"):
            item = next(source, None)
        if item is None:
            raise RuntimeError("Continuous epoch ended before its indexed cursor")
        yield item
