# VLM client
Status: approved (2026-10-10; clarifications after prototype and refactor the same day: transport errors, captured outputs, error messages, library arguments, starting values)
Stage: P2 VLM labels (`specs/00-overview.md`, block 0)

## Sources
- [U] the user, 2026-10-09: model `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`, served on an NVIDIA DGX Spark; a Python API sends a set of images and text to the model; answers limited to 300 tokens; every constant in a config. 2026-10-10: every action on the Spark is a bash script the user runs; `scripts/serve_vlm.sh` starts the VLM.
- [M] `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` @ `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`: `config.json`, `chat_template.json`, `generation_config.json`, `preprocessor_config.json`.
- [Q] Qwen3-VL README, section "Generation hyperparameters", Instruct models (github.com/QwenLM/Qwen3-VL).
- [V] vLLM v0.31.0 source (github.com/vllm-project/vllm/tree/v0.31.0); paths in Library mapping.
- [T] transformers v5.17.0 `Qwen2VLImageProcessor` (the version inside the image: Server step 5).
Open questions: `specs/00-questions.md`, section VLMC.

## Notation
- $q$ — prompt: a sequence of parts, each a text string or an image.
- Image — RGB `uint8` array of shape $(H, W, 3)$.
- $s$ — sampling seed of one request, $0 \le s < 2^{63}$.
- Answer — the model's reply text `message.content`, stripped of surrounding whitespace.
- Client config fields (`VLMConfig`, file `configs/vlm.yaml`, section `client`): `base_url`; `model` — the served name; the fields of Table 1; `max_in_flight` ($K$); `attempt_timeout` ($T_{out}$, s); `max_resends` ($R$); `resend_delay` ($\delta$, s).
- Server config fields (`ServerConfig`, same file, section `server`): `image` — the vLLM image with digest; `model_repo`, `revision` — the weights; `hf_cli` — the `huggingface_hub` version providing the `hf` command; `hf_home` — the Hugging Face cache directory, relative to the repository root (starting value `data/hf`); `container` — container name; `port`; `gpu_memory_utilization`; `max_images` ($N_{img}$); `max_prompt_tokens` ($P_{max}$); `seed` — vLLM's `--seed`; `stack_ulimit` — bytes; `ready_timeout`, `poll_interval`, `poll_request_timeout` (s); `log_tail` — log lines printed on failure; `transformers_version` — the version inside the image. The server's served name is the client's `model`, its `--max-num-seqs` the client's `max_in_flight`, its pixel bounds and `max_tokens` the client's Table 1 values; `client.base_url` = `http://localhost:{server.port}` when the client runs on the Spark.

## Behaviour
(1) One request per prompt: a single user message whose content is the parts of $q$ in their order; no system message (Q VLMC-1, VLMC-2). The chat template [M] renders text parts verbatim, every image as `<|vision_start|><|image_pad|><|vision_end|>` without a "Picture N:" prefix, with no separators, and ends the prompt with `<|im_start|>assistant\n`.
(2) Image part → `{"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>"}}`: `<b64>` is the PNG of the array, in the standard base64 alphabet, padded, without line breaks. Text part → `{"type": "text", "text": <part>}`.
(3) Request body: exactly the keys `model`, `messages` (from (1)), `n` (= 1), `seed` (= $s$), and the fields of Table 1. The defaults of v0.31.0 this relies on: generation stops at the eos ids `[151645, 151643]`; `skip_special_tokens = true`; `ignore_eos = false`; `min_tokens = 0`; no logprobs.
(4) Result `choices[0]`: if `finish_reason == "stop"` and `message.content` is non-empty after stripping whitespace → the answer. Otherwise (e.g. `"length"`: the reply hit `max_tokens`) → `None` (Q VLMC-6).
(5) At most $K$ requests in flight. An attempt longer than $T_{out}$ from its start (waiting for a slot excluded) is cancelled → `None`, not resent (Q VLMC-6).
(6) Transport error (any `httpx.TransportError`) or HTTP 5xx → resend after $\delta$, keeping its slot, at most $R$ times. Then, or at once on HTTP 4xx: cancel all requests in flight and raise `RuntimeError` with the HTTP status and body, or with the transport error.

Starting values of the other client fields: `base_url` = `http://localhost:8000`, `model` = `qwen3vl-instruct-fp8`, `max_in_flight` = 16, `attempt_timeout` = 300, `max_resends` = 3, `resend_delay` = 10.

Table 1 — sampling and preprocessing fields; all are config fields with the same names.
| Field | Starting value | Source / note |
|---|---|---|
| `temperature` | 0.7 | [Q]; same in [M] `generation_config.json` |
| `top_p` | 0.8 | [Q] |
| `top_k` | 20 | [Q] |
| `min_p` | 0.0 | |
| `presence_penalty` | 1.5 | [Q] |
| `frequency_penalty` | 0.0 | |
| `repetition_penalty` | 1.0 | [Q] |
| `max_tokens` | 300 | [U]; effective limit is `min(max_tokens, max_model_len − prompt_len)`, equal to `max_tokens` when prompt ≤ $P_{max}$ |
| `min_pixels` → `mm_processor_kwargs.min_pixels` | 65 536 | always sent together with `max_pixels` [T]; default 65 536 upscales smaller frames |
| `max_pixels` → `mm_processor_kwargs.max_pixels` | 230 400 (640×360) | default 16 777 216 |

## Interface
- Module: `src/mcpref/vlm/client.py`.
- `@dataclass(frozen=True) class VLMConfig`, `@dataclass(frozen=True) class ServerConfig` — the config fields of Notation; `def load_config(path: Path) -> tuple[VLMConfig, ServerConfig]` reads `configs/vlm.yaml`.
- `def ask_many(prompts: Sequence[Sequence[str | np.ndarray]], seeds: Sequence[int], cfg: VLMConfig) -> list[str | None]` — synchronous, runs its own event loop; result $i$ is the answer to `prompts[i]` with seed `seeds[i]`; (1)–(6). Request: `POST {cfg.base_url}/v1/chat/completions`; `base_url` has no trailing slash and no `/v1`, e.g. `http://localhost:8000`. No authentication ([U] 2026-10-10: the server is reachable only from the Spark itself, and any local process may use it).
- Domain: `len(seeds) == len(prompts) ≥ 1`; every prompt has ≥ 1 part, ≤ $N_{img}$ images and ≤ $P_{max}$ tokens; text parts contain no special-token strings (`<|…|>`); seeds as in Notation. Nothing outside it is checked.
- `max_in_flight` = server `--max-num-seqs`.
- Consumes: the server of section Server at `base_url`.
- Produces: nothing on disk; the caller (block gui-labels, later vlm-labels) stores answers.
- Module: `src/mcpref/vlm/server.py` (entry, Server steps 2 and 4), with `src/mcpref/vlm/container.py` (container existence, Server step 5); `def main(argv: Sequence[str]) -> int`, run as `python -m mcpref.vlm.server {prepare|serve|stop} <config path>` (argparse: one positional subcommand, one positional path).
  - `def prepare_commands(srv: ServerConfig) -> list[list[str]]` — the argument lists of Server step 2; `def serve_command(cfg: VLMConfig, srv: ServerConfig) -> list[str]` — the argument list of Server step 4. Pure functions; commands run without a shell; paths made absolute from the repository root.
  - `prepare`: runs `prepare_commands` in order. Called by `scripts/setup_env.sh` step (8) (block spark-env), which this block adds.
  - `serve`: `docker rm -f {server.container}` if the container exists; runs `serve_command`; then Server step 5.
  - Container existence (used by `serve` and `stop`): `docker ps -a --filter name=^/{server.container}$ --format {{.Names}}` prints exactly `{server.container}` (an anchored name match; `docker container inspect` would also match an ID prefix); its output is captured.
  - `stop`: if the container exists, `docker rm -f {server.container}` and print `stopped {server.container}`; else print `{server.container} is not running`; both exit 0.
  - Subprocess output passes through, except the outputs the code reads (the existence check, `docker container inspect -f`, the transformers check, `docker logs` on the success path), which are captured; the log tail printed after a readiness failure passes through. A failing or missing subprocess, an unmet readiness, a failed check, an unreadable config or response → a one-line message on stderr naming the cause and exit code 1, no traceback; success → 0.
- Produces (`prepare`): the vLLM image in Docker; the weights under `{server.hf_home}/hub`.
- Script: `scripts/serve_vlm.sh` (preamble of `specs/spark-env.md`): frees the page cache (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`); `uv run --locked python -m mcpref.vlm.server serve configs/vlm.yaml`.
- Script: `scripts/stop_vlm.sh` (preamble of `specs/spark-env.md`; [U] 2026-10-10): `uv run --locked python -m mcpref.vlm.server stop configs/vlm.yaml`.
- Tests: `prepare_commands` and `serve_command` against the Server section; `stop` with a fake `subprocess.run` (running / not running / failing `docker`); `scripts/stop_vlm.sh` with a fake `uv`; `main` with an injected fake `subprocess.run` and a fake HTTP server for readiness; `scripts/serve_vlm.sh` with fake `sudo` and `uv` on `PATH`. Real runs are manual, on the Spark.

## Server
The steps below are what `prepare`, `scripts/serve_vlm.sh` and `serve` do; listed for the record.
DGX Spark: GB10, aarch64, 128 GB unified memory. Checked against DGX OS 7.5.0 (driver 580.159.03, CUDA 13.0.2).
1. `nvidia-smi` shows driver ≥ 580.159.03 (checked by `scripts/setup_env.sh`). Avoid NGC `nvcr.io/nvidia/vllm` ≥ 26.04: it needs driver ≥ 595.58, and forward compatibility fails on Spark.
2. `prepare`: `docker pull {server.image}`; `HF_HOME={server.hf_home} uvx --from huggingface_hub=={server.hf_cli} hf download {server.model_repo} --revision {server.revision}` (32.3 GB). `serve_vlm.sh` frees the page cache before `serve`, since it shares the unified memory.
3. No API key: vLLM runs without `--api-key` / `VLLM_API_KEY`, so it installs no authentication; the port is bound to 127.0.0.1 ([U] 2026-10-10).
4. `serve` runs (`{…}` = config field; `{client.max_tokens + server.max_prompt_tokens}` computed):
```bash
docker run -d --name {server.container} --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack={server.stack_ulimit} \
  -p 127.0.0.1:{server.port}:{server.port} -v {server.hf_home}:/root/.cache/huggingface \
  -e VLLM_USE_DEEP_GEMM=0 -e HF_HUB_OFFLINE=1 \
  {server.image} \
  {server.model_repo} --revision {server.revision} --tokenizer-revision {server.revision} \
  --served-model-name {client.model} --host 0.0.0.0 --port {server.port} \
  --max-model-len {client.max_tokens + server.max_prompt_tokens} \
  --gpu-memory-utilization {server.gpu_memory_utilization} --max-num-seqs {client.max_in_flight} \
  --moe-backend triton --chat-template-content-format openai \
  --limit-mm-per-prompt '{"image": {server.max_images}, "video": 0}' \
  --mm-processor-kwargs '{"min_pixels": {client.min_pixels}, "max_pixels": {client.max_pixels}}' \
  --generation-config vllm --seed {server.seed}
```
   The two JSON arguments are single argv items built with `json.dumps`. Switches, not tunables: `memlock=-1` (unlimited), `"video": 0` (no video input), `VLLM_USE_DEEP_GEMM=0` (off), `HF_HUB_OFFLINE=1` (no downloads in the container).
   Starting values: `image` = `vllm/vllm-openai:v0.31.0-aarch64@sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971`, `model_repo` = `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`, `revision` = `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`, `hf_cli` = `1.33.0`, `hf_home` = `data/hf`, `container` = `qwen3vl`, `port` = 8000, `gpu_memory_utilization` = 0.70, `max_images` = 32, `max_prompt_tokens` = 8192, `seed` = 0, `stack_ulimit` = 67108864, `ready_timeout` = 1800, `poll_interval` = 5, `poll_request_timeout` = 5, `log_tail` = 200, `transformers_version` = `5.17.0`, `model` = `qwen3vl-instruct-fp8`, `max_in_flight` = 16.
   - `--moe-backend triton`, `VLLM_USE_DEEP_GEMM=0`: otherwise vLLM picks DeepGEMM on GB10, which rounds the checkpoint's FP32 block scales to powers of two (vllm#50332).
   - `--generation-config vllm`: otherwise `generation_config.json` silently fills every sampling field a request omits.
   - `--chat-template-content-format openai`: pins the layout of (1) instead of auto-detection.
   - `--mm-processor-kwargs`: the same bounds as Table 1; they size vLLM's startup memory profiling.
   - `-p 127.0.0.1:…`: reachable only from the Spark itself or through an SSH tunnel.
5. `serve`, after `docker run` returns:
   - readiness: `GET http://127.0.0.1:{server.port}/health` with timeout `{server.poll_request_timeout}` s; a transport error (any `httpx.TransportError`, timeouts included) or a non-200 status means not ready yet, and the next poll follows `{server.poll_interval}` s after this one ends; if `docker container inspect -f '{{.State.Running}}'` is `false`, fail at once; if `{server.ready_timeout}` s pass since `docker run` returned, fail. On failure: print the last `{server.log_tail}` log lines; the container is kept for inspection.
   - a failing `docker container inspect -f` during polling counts as a stopped container.
   - once ready: `GET /v1/models` (timeout `{server.poll_request_timeout}` s) must list `{client.model}`; `docker exec {server.container} python3 -c "import transformers; print(transformers.__version__)"` must print `{server.transformers_version}`; the container log (stdout and stderr) must contain the text `Using TRITON Fp8 MoE backend` within a line (vLLM prefixes its log lines) (`vllm/model_executor/layers/fused_moe/oracle/fp8.py:415`). Any mismatch → failure.
   - prints the log line with `KV cache size` (`vllm/v1/core/kv_cache_utils.py:2465`) for the user: capacity in tokens should be ≥ `max_in_flight` × max-model-len, else the user lowers `max_in_flight`.
6. Clients run on the Spark (`base_url` = `http://localhost:{server.port}`); from another machine: `ssh -N -L {server.port}:localhost:{server.port} <user>@<spark>`.

## Library mapping
| Item | Call | Version | Explicit args | Source read |
|---|---|---|---|---|
| (1) prompt layout | vLLM chat renderer + [M] template | v0.31.0 | server `--chat-template-content-format openai`; no `chat_template_kwargs` (so `add_vision_id` undefined) | `vllm/renderers/hf.py:403-506`; [M] template rendered with jinja2 |
| (2) base64 | decoded by `pybase64.b64decode(…, validate=True)` | v0.31.0 | standard alphabet, padded, no line breaks | `vllm/multimodal/media/image.py:104-105` |
| (2) preprocessing | `Qwen2VLImageProcessor(Fast)` via vLLM | [T] v5.17.0 (checked in step 5) | both `min_pixels`, `max_pixels` per request (one alone is ignored); resize to multiples of 32 px (`smart_resize`), BICUBIC; one token per 32×32 block | `vllm/model_executor/models/qwen3_vl.py:964-967`; [T] `image_processing_qwen2_vl.py` |
| (3) sampling | `/v1/chat/completions` | v0.31.0 | Table 1, `n`, `seed` (own generator per request; `-1` = unseeded, outside the domain); server `--generation-config vllm` | `vllm/entrypoints/openai/chat_completion/protocol.py:213-302`; `vllm/config/model.py:1781`; `vllm/v1/worker/gpu_model_runner.py:1281` |
| (3) `max_tokens` | same | v0.31.0 | effective `min(max_tokens, max_model_len − prompt_len)` | `vllm/entrypoints/serve/utils/api_utils.py:169-205` |
| (3) other fields | same | v0.31.0 | none sent; unknown keys are dropped with a DEBUG log only | `vllm/entrypoints/serve/engine/protocol.py:61-73` |
| (5), (6) transport | `httpx.AsyncClient` | 0.28.1 | no `Authorization` header; `timeout=None` (the attempt bound is $T_{out}$ via `asyncio.wait_for`; the httpx default would be 5 s per phase); `limits=httpx.Limits(max_connections=K)` (default 100 would queue inside the attempt); `trust_env=False` (no proxy variables or `.netrc`); transport `retries=0` (default) | prototype and refactor reports, 2026-10-10 |
| readiness, checks | `httpx.get` | 0.28.1 | `timeout=poll_request_timeout`, `trust_env=False` | same |
| (2) PNG | `PIL.Image.fromarray(a).save(buf, format="PNG")` | 12.3.0 | lossless | to vet at implementation |
| config loading | `yaml.safe_load` | PyYAML 6.0.3 (pinned by block spark-env) | — | to vet at implementation |

Known non-determinism: with temperature > 0 and a fixed seed, outputs still depend on batch composition (prefix caching, chunked prefill). Accepted.

## Out of scope
- Prompt wording, answer parsing, pair sampling, label storage: block vlm-labels.
- Answer-token probabilities and constrained choice (Q VLMC-8, VLM-6).
- Streaming, `n > 1`, video input, system message.
- Monitoring or restarting a server that dies after it became ready.
- The Hugging Face cache outside `{server.hf_home}`.
