"""Client of specs/vlm-client.md: config, request body (1)-(3), answers (4)-(6)."""

import asyncio
import base64
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import yaml
from PIL import Image

from mcpref.http.retry import Outcome, retryable


@dataclass(frozen=True)
class VLMConfig:
    """Client config fields of Notation and Table 1 (`client` in configs/vlm.yaml)."""

    base_url: str
    model: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    max_tokens: int
    min_pixels: int
    max_pixels: int
    max_in_flight: int
    attempt_timeout: float
    max_resends: int
    resend_delay: float


@dataclass(frozen=True)
class ServerConfig:
    """Server config fields of Notation (`server` in configs/vlm.yaml)."""

    image: str
    model_repo: str
    revision: str
    hf_cli: str
    hf_home: str
    container: str
    port: int
    gpu_memory_utilization: float
    max_images: int
    max_prompt_tokens: int
    seed: int
    stack_ulimit: int
    ready_timeout: float
    poll_interval: float
    poll_request_timeout: float
    log_tail: int
    transformers_version: str


def load_config(path: Path) -> tuple[VLMConfig, ServerConfig]:
    """Sections `client` and `server` of configs/vlm.yaml."""
    raw = yaml.safe_load(path.read_text())
    return VLMConfig(**raw["client"]), ServerConfig(**raw["server"])


def content_part(part: str | np.ndarray) -> dict:
    """(2): a text part, or an image part as a base64 PNG data URL."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    png = io.BytesIO()
    Image.fromarray(part).save(png, format="PNG")
    b64 = base64.b64encode(png.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def request_body(prompt: Sequence[str | np.ndarray], seed: int, cfg: VLMConfig) -> dict:
    """(1), (3): one user message with the parts of q in order; n, seed, Table 1
    except min_pixels, max_pixels (set by the server, Server step 4)."""
    return {
        "model": cfg.model,
        "messages": [{"role": "user", "content": [content_part(p) for p in prompt]}],
        "n": 1,
        "seed": seed,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "presence_penalty": cfg.presence_penalty,
        "frequency_penalty": cfg.frequency_penalty,
        "repetition_penalty": cfg.repetition_penalty,
        "max_tokens": cfg.max_tokens,
    }


def answer(result: dict) -> str | None:
    """(4): stripped content of choices[0] if it stopped and is non-empty, else None."""
    choice = result["choices"][0]
    text = (choice["message"]["content"] or "").strip()
    return text if choice["finish_reason"] == "stop" and text else None


async def attempt(client: httpx.AsyncClient, body: dict, cfg: VLMConfig) -> Outcome:
    """(5): one attempt; asyncio.TimeoutError once it runs longer than T_out."""
    send = client.post(f"{cfg.base_url}/v1/chat/completions", json=body)
    try:
        return await asyncio.wait_for(send, cfg.attempt_timeout)
    except httpx.TransportError as error:
        return error


def failure(outcome: Outcome) -> RuntimeError:
    """(6): the error with the HTTP status and body, or with the transport error."""
    if isinstance(outcome, httpx.TransportError):
        return RuntimeError(f"transport error: {outcome!r}")
    return RuntimeError(f"HTTP {outcome.status_code}: {outcome.text}")


async def ask_one(
    client: httpx.AsyncClient,
    slots: asyncio.Semaphore,
    prompt: Sequence[str | np.ndarray],
    seed: int,
    cfg: VLMConfig,
) -> str | None:
    """(4)-(6) for one prompt: in one slot, at most R resends after delta each."""
    async with slots:
        body = request_body(prompt, seed, cfg)
        try:
            outcome = await attempt(client, body, cfg)
            for _ in range(cfg.max_resends):
                if not retryable(outcome):
                    break
                await asyncio.sleep(cfg.resend_delay)
                outcome = await attempt(client, body, cfg)
        except asyncio.TimeoutError:
            return None
        if retryable(outcome) or outcome.is_client_error:
            raise failure(outcome)
        return answer(outcome.json())


async def ask_all(
    prompts: Sequence[Sequence[str | np.ndarray]],
    seeds: Sequence[int],
    cfg: VLMConfig,
) -> list[str | None]:
    """(5), (6): at most K in flight; a raised failure cancels all requests in flight."""
    slots = asyncio.Semaphore(cfg.max_in_flight)
    async with httpx.AsyncClient(
        timeout=None,
        limits=httpx.Limits(max_connections=cfg.max_in_flight),
        trust_env=False,
    ) as client:
        tasks = [
            asyncio.create_task(ask_one(client, slots, prompt, seed, cfg))
            for prompt, seed in zip(prompts, seeds)
        ]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def ask_many(
    prompts: Sequence[Sequence[str | np.ndarray]],
    seeds: Sequence[int],
    cfg: VLMConfig,
) -> list[str | None]:
    """(1)-(6): answer i to prompts[i] with seeds[i]; runs its own event loop."""
    return asyncio.run(ask_all(prompts, seeds, cfg))
