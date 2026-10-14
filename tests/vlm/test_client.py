"""src/mcpref/vlm/client.py against specs/vlm-client.md: Behaviour (1)-(6), Table 1, Interface."""

import base64
import dataclasses
import io
import re
import time
from itertools import pairwise

import httpx
import numpy as np
import pytest
from PIL import Image

from mcpref.vlm.client import ServerConfig, VLMConfig, ask_many, load_config

# Table 1 fields sent per request: all except min_pixels, max_pixels (3).
TABLE1 = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
          "repetition_penalty", "max_tokens")  # fmt: skip


def ask(fields, server, prompts, seeds=None, **changes) -> list:
    """ask_many against `server` with the test config and `changes`; seeds 0 by default."""
    url = {"base_url": f"http://127.0.0.1:{server.port}"}
    cfg = VLMConfig(**fields["client"] | url | changes)
    return ask_many(prompts, seeds or [0] * len(prompts), cfg)


def completion(content: str, finish: str = "stop") -> tuple:
    message = {"role": "assistant", "content": content}
    choice = {"index": 0, "message": message, "finish_reason": finish}
    return 200, {"object": "chat.completion", "choices": [choice]}


def first_text(req) -> str:
    return req.body["messages"][0]["content"][0]["text"]


def echo(req) -> tuple:
    return completion(first_text(req))


def held(seconds: float, then=echo):
    """App replying with `then` after `seconds`."""

    def app(req):
        req.hold(seconds)
        return then(req)

    return app


def image(h: int, w: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def decode(part: dict) -> np.ndarray:
    """(2): the array behind a PNG data URL in the standard base64 alphabet, padded, one line."""
    assert part.keys() == {"type", "image_url"} and part["type"] == "image_url"
    prefix, b64 = part["image_url"]["url"].split(",", 1)
    assert part["image_url"].keys() == {"url"} and prefix == "data:image/png;base64"
    assert re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", b64) and len(b64) % 4 == 0
    png = base64.b64decode(b64, validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(png)) as im:
        assert im.mode == "RGB"
        return np.asarray(im)


def carries(exc: BaseException, kind: type) -> bool:
    """(6): `exc` holds a `kind` error as an object (args, cause, context) or in its message."""
    linked = (*exc.args, exc.__cause__, exc.__context__)
    return any(isinstance(e, kind) for e in linked) or kind.__name__ in str(exc)


def test_request_body_has_exactly_the_spec_keys(http, fields):
    """(3): Table 1 except the pixel bounds, which only the server sets (no mm_processor_kwargs)."""
    server = http(echo)
    assert ask(fields, server, [["q"]], [2**63 - 1]) == ["q"]
    (req,) = server.requests
    assert (req.method, req.path) == ("POST", "/v1/chat/completions")
    assert "Authorization" not in req.headers  # Interface: no authentication
    c, body = fields["client"], dict(req.body)
    text = {"type": "text", "text": "q"}
    assert body.pop("messages") == [{"role": "user", "content": [text]}]
    spec = {"model": c["model"], "n": 1, "seed": 2**63 - 1}
    assert body == spec | {k: c[k] for k in TABLE1}


def test_one_user_message_with_parts_in_order(http, fields):
    """(1), (2): texts verbatim, images as lossless PNG; a non-contiguous view and a 1x1 image."""
    server = http(echo)
    prompt = ["  first\n", image(17, 23, 1), "second", image(9, 12, 2)[::2, ::-1]]
    prompt += [" ", image(1, 1, 3)]
    assert ask(fields, server, [prompt]) == ["first"]
    (message,) = server.requests[0].body["messages"]
    assert message.keys() == {"role", "content"} and message["role"] == "user"
    assert len(message["content"]) == len(prompt)
    for part, sent in zip(prompt, message["content"]):
        if isinstance(part, str):
            assert sent == {"type": "text", "text": part}
        else:
            np.testing.assert_array_equal(decode(sent), part)


def test_result_i_answers_prompt_i_with_seed_i(http, fields):
    texts, seeds = ["p0", "p1", "p2", "p3"], [5, 0, 2**63 - 1, 5]

    def app(req):  # later prompts are answered first
        req.hold(0.05 * (len(texts) - texts.index(first_text(req))))
        return completion(f"{first_text(req)}|{req.body['seed']}")

    server = http(app)
    result = ask(fields, server, [[t] for t in texts], seeds)
    assert result == [f"{t}|{s}" for t, s in zip(texts, seeds)]
    assert len(server.requests) == len(texts)  # (1): one request per prompt


def test_answer_is_stripped_content_only_on_stop(http, fields):
    """(4): "  yes \\n" on stop -> "yes"; on length, or blank or empty on stop -> None."""
    replies = {"a": ("  yes \n", "stop"), "b": ("partial", "length")}
    replies |= {"c": (" \n\t", "stop"), "d": ("", "stop")}
    server = http(lambda req: completion(*replies[first_text(req)]))
    assert ask(fields, server, [[k] for k in replies]) == ["yes", None, None, None]


@pytest.mark.parametrize("k", [3, 101])  # 101: above httpx's default pool of 100
def test_k_requests_in_flight_at_most(http, fields, k):
    server, n = http(held(0.3)), 2 * k + 1
    result = ask(fields, server, [["p"]] * n, list(range(n)), max_in_flight=k)
    assert result == ["p"] * n and server.max_active == k


def test_waiting_for_a_slot_is_not_part_of_the_attempt(http, fields):
    """(5): K = 1; the 4th prompt waits 0.75 s for its slot, its attempt takes 0.25 s < 0.6 s."""
    changes = {"max_in_flight": 1, "attempt_timeout": 0.6}
    assert ask(fields, http(held(0.25)), [["p"]] * 4, **changes) == ["p"] * 4


def test_attempt_over_timeout_is_none_cancelled_and_not_resent(http, fields):
    server = http(lambda req: held(4.0 if first_text(req) == "slow" else 0.0)(req))
    start = time.monotonic()
    result = ask(fields, server, [["slow"], ["fast"]], attempt_timeout=0.4)
    assert result == [None, "fast"] and time.monotonic() - start < 2.0
    slow = [r for r in server.requests if first_text(r) == "slow"]
    assert len(slow) == 1 and slow[0].cancelled()


@pytest.mark.parametrize("failure", ["drop", 500, 503])  # drop: closed without a reply
def test_transport_error_or_5xx_resent_after_delay_keeping_slot(http, fields, failure):
    def app(req):  # the first two attempts of "a" fail
        if (
            first_text(req) == "b"
            or [first_text(r) for r in server.requests].count("a") > 2
        ):
            return echo(req)
        return None if failure == "drop" else (failure, b"busy")

    server = http(app)
    changes = {"max_in_flight": 1, "max_resends": 2, "resend_delay": 0.2}
    assert ask(fields, server, [["a"], ["b"]], **changes) == ["a", "b"]
    order = [first_text(r) for r in server.requests]
    assert order in (["a", "a", "a", "b"], ["b", "a", "a", "a"])
    times = [r.t for r in server.requests if first_text(r) == "a"]
    assert all(0.2 - 1e-3 <= t1 - t0 < 0.4 for t0, t1 in pairwise(times)), times


@pytest.mark.parametrize(
    "failure, attempts", [("drop", 3), (503, 3), (400, 1), (422, 1)]
)
def test_failure_raises_and_cancels_requests_in_flight(http, fields, failure, attempts):
    """(6): after R = 2 resends, or at once on 4xx; the slow request in flight is cancelled."""

    def app(req):
        if first_text(req) == "slow":
            return held(4.0)(req)
        req.hold(0.1)
        return None if failure == "drop" else (failure, b'{"error": "reason-xyz"}')

    server, start = http(app), time.monotonic()
    with pytest.raises(RuntimeError) as info:
        ask(fields, server, [["slow"], ["bad"]], max_in_flight=2)
    assert time.monotonic() - start < 2.5
    assert [first_text(r) for r in server.requests].count("bad") == attempts
    slow = [r for r in server.requests if first_text(r) == "slow"]
    assert len(slow) == 1 and slow[0].cancelled()
    if failure == "drop":
        assert carries(info.value, httpx.RemoteProtocolError), repr(info.value)
    else:
        assert str(failure) in str(info.value) and "reason-xyz" in str(info.value)


def test_unreachable_server_raises_with_the_transport_error(http, fields):
    with pytest.raises(RuntimeError) as info:
        ask(fields, http(echo, listen=False), [["p"]], max_resends=1)
    assert carries(info.value, httpx.ConnectError), repr(info.value)


def test_proxy_variables_are_ignored(http, fields, refusing_proxy):
    """Library mapping: trust_env=False; the proxy from the environment refuses connections."""
    assert ask(fields, http(echo), [["p"]], max_resends=0) == ["p"]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_no_transport_timeout_below_attempt_timeout(http, fields):
    """Library mapping: timeout=None; httpx's default would end a 6 s reply at 5 s."""
    changes = {"attempt_timeout": 30.0, "max_resends": 0}
    assert ask(fields, http(held(6.0)), [["p"]], **changes) == ["p"]


def test_config_fields_are_those_of_notation(fields):
    for cls, section in ((VLMConfig, "client"), (ServerConfig, "server")):
        assert {f.name for f in dataclasses.fields(cls)} == set(fields[section])


def test_load_config_reads_both_sections(fields, write_config):
    expected = VLMConfig(**fields["client"]), ServerConfig(**fields["server"])
    assert load_config(write_config(fields)) == expected


def test_committed_config_starting_values(repo):
    cfg, srv = load_config(repo / "configs/vlm.yaml")
    assert cfg == VLMConfig(
        base_url="http://localhost:8100", model="qwen3vl-instruct-fp8", temperature=0.7, top_p=0.8,
        top_k=20, min_p=0.0, presence_penalty=1.5, frequency_penalty=0.0, repetition_penalty=1.0,
        max_tokens=300, min_pixels=65536, max_pixels=230400, max_in_flight=16, attempt_timeout=300,
        max_resends=3, resend_delay=10,
    )  # fmt: skip
    assert srv == ServerConfig(
        image="vllm/vllm-openai:v0.31.0-aarch64@sha256:"
        "3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971",
        model_repo="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", revision="d9748a51ae66354c4dad665aab2c71f26cf2c8cd",
        hf_cli="1.33.0", hf_home="data/hf", container="qwen3vl", port=8100, gpu_memory_utilization=0.5,
        max_images=32, max_prompt_tokens=8192, seed=0, stack_ulimit=67108864, ready_timeout=1800,
        poll_interval=5, poll_request_timeout=5, log_tail=200, transformers_version="5.17.0",
    )  # fmt: skip
    for frozen in (cfg, srv):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(frozen, dataclasses.fields(frozen)[0].name, None)
