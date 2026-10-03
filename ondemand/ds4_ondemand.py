#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
# ]
# ///
"""On-demand launcher/proxy for DwarfStar's ds4-server.

Listens on DS4_ONDEMAND_HOST:DS4_ONDEMAND_PORT (default 0.0.0.0:8001) and
proxies everything to a ds4-server child process on 127.0.0.1:8000, starting
it lazily on first request and stopping it after DS4_IDLE_SECONDS of no
in-flight requests. Because ds4-server (DS4 / Qwen3.8-Flash-Next Q2, ~43 GiB
resident) and oMLX's 35B-A3B (~20 GiB) cannot both be resident on this 64 GB
Mac, ds4-server startup first asks oMLX which models are loaded and unloads
them, tolerating oMLX being stopped entirely.

GET /v1/models answers instantly from a static alias list without starting
ds4-server, so model pickers work while it is cold. Every other /v1/* path is
proxied with full streaming passthrough (SSE included).
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import logging.handlers
import math
import os
import re
import secrets
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# --------------------------------------------------------------------------
# Configuration (all overridable via environment)
# --------------------------------------------------------------------------

DS4_ONDEMAND_HOST = os.environ.get("DS4_ONDEMAND_HOST", "0.0.0.0")
DS4_ONDEMAND_PORT = int(os.environ.get("DS4_ONDEMAND_PORT", "8001"))

DS4_SERVER_HOST = "127.0.0.1"  # ds4-server itself always stays loopback-only
DS4_SERVER_PORT = int(os.environ.get("DS4_SERVER_PORT", "8000"))

DS4_REPO_DIR = Path(
    os.environ.get("DS4_REPO_DIR", "/Users/beaglemoo/Homelab/dwarfstar")
).resolve()
# Optional overrides so a bundled install (Unsloth.app) can keep the binary,
# its metal/ shader dir (the child's cwd) and the logs outside the repo.
DS4_BINARY = Path(os.environ.get("DS4_BINARY") or (DS4_REPO_DIR / "ds4-server"))
DS4_WORKDIR = Path(os.environ.get("DS4_WORKDIR") or DS4_REPO_DIR).resolve()
DS4_MODEL_FILE = os.environ.get("DS4_MODEL_FILE", "ds4flash.gguf")
DS4_VISION_FILE = os.environ.get(
    "DS4_VISION_FILE", "gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
)
DS4_CTX = os.environ.get("DS4_CTX", "65536")
# Runtime-adjustable context window (POST /admin/config). ds4-server has no
# hard --ctx cap (ds4_server.c only parses an int; the model's rope_orig_ctx
# is 262144), so 262144 is the ceiling offered; 4096 is a sane floor.
CTX_MIN = 4096
CTX_MAX = 262144
CTX_STEP = 256
CTX_DEFAULT = 65536
DS4_PREFILL_CHUNK = os.environ.get("DS4_PREFILL_CHUNK", "1024")

DS4_START_TIMEOUT = float(os.environ.get("DS4_START_TIMEOUT", "120"))
DS4_IDLE_SECONDS = float(os.environ.get("DS4_IDLE_SECONDS", "300"))

OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8843")
# After asking oMLX to unload, wait this long for it to report nothing loaded
# or loading; otherwise the cold start is refused (503) instead of spawning.
OMLX_UNLOAD_WAIT = float(os.environ.get("DS4_OMLX_UNLOAD_WAIT", "10"))

# ComfyUI is asked to free its models before a cold start (only when its queue
# is empty). An empty value disables it.
DS4_COMFYUI_URL = os.environ.get("DS4_COMFYUI_URL", "http://127.0.0.1:8188").strip()
COMFYUI_TIMEOUT = 3.0

# Graceful shutdown: uvicorn waits this long for open connections, and the
# lifespan stop waits this long for state.lock before terminating the child
# without it (a cold start can hold the lock for DS4_START_TIMEOUT).
GRACEFUL_SHUTDOWN_SECONDS = 90
SHUTDOWN_LOCK_TIMEOUT = float(os.environ.get("DS4_SHUTDOWN_LOCK_TIMEOUT", "10"))

# Holds (POST /admin/hold) block cold starts until released or expired.
HOLD_TTL_MIN = 1
HOLD_TTL_MAX = 24 * 3600
HOLD_RETRY_AFTER_MAX = 15
START_RETRY_AFTER = 15

# ds4-server.log rotation.
DS4_LOG_MAX_BYTES = 10 * 1024 * 1024
DS4_LOG_BACKUPS = 3

DS4_SAMPLING_INJECT = os.environ.get("DS4_SAMPLING_INJECT", "1") != "0"
DS4_SAMPLING_DEFAULTS = os.environ.get("DS4_SAMPLING_DEFAULTS")

DS4_DEBUG = os.environ.get("DS4_DEBUG", "0") != "0"

LOGS_DIR = Path(os.environ.get("DS4_LOG_DIR") or (DS4_REPO_DIR / "logs"))
DS4_LOG_PATH = LOGS_DIR / "ds4-server.log"


def _default_state_dir() -> Path:
    explicit = os.environ.get("DS4_STATE_DIR")
    if explicit:
        return Path(os.path.expanduser(explicit))
    engines = Path.home() / ".unsloth" / "engines"
    return engines if engines.is_dir() else LOGS_DIR


DS4_STATE_PATH = _default_state_dir() / "ds4-state.json"

# ds4-server only recognises these ids for its thinking-mode aliases, so every
# accepted id is mapped onto them (LEGACY_ALIAS_BASE + suffix) before the
# request is forwarded, and the model id in the response is mapped back.
LEGACY_ALIAS_BASE = "qwen3.8-flash-next"
MODE_SUFFIXES = ("", "-chat", "-reasoner")

# Trailing quantisation tag of a GGUF file stem: -Q2, -Q4_K_M, -Q8_0, -IQ2_XXS,
# -UD-Q4_K_XL, -BF16, -F16 ...
QUANT_SUFFIX_RE = re.compile(
    r"-(?:ud-)?(?:i?q\d+(?:_[a-z0-9]+)*|bf16|f16|f32)$", re.IGNORECASE
)


def derive_alias_base(model_file: str, workdir: Path | None = None) -> str:
    """Alias base from the file that is actually served: realpath, basename,
    minus .gguf and the trailing quant tag, lowercased."""
    path = Path(os.path.expanduser(model_file))
    if not path.is_absolute() and workdir is not None:
        path = workdir / path
    stem = os.path.basename(os.path.realpath(path))
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    stem = QUANT_SUFFIX_RE.sub("", stem).strip().lower()
    return stem or LEGACY_ALIAS_BASE


def model_alias_base(model_file: str, override: str | None, workdir: Path | None = None) -> str:
    """DS4_MODEL_ALIAS (env or [ds4] alias in engines.toml) wins over the
    name derived from the model file."""
    explicit = (override or "").strip().lower()
    return explicit or derive_alias_base(model_file, workdir)


DS4_MODEL_ALIAS = model_alias_base(
    DS4_MODEL_FILE, os.environ.get("DS4_MODEL_ALIAS"), DS4_WORKDIR
)


def listed_models(base: str) -> list[str]:
    return [base + suffix for suffix in MODE_SUFFIXES]


STATIC_MODELS = listed_models(DS4_MODEL_ALIAS)


def resolve_model_alias(model, base: str | None = None) -> tuple[str, str] | None:
    """Maps a requested model id to (mode suffix, id ds4-server understands).
    Accepts the listed <base>, <base>-chat, <base>-reasoner and the legacy
    qwen3.8-flash-next* ids (accepted but not listed). None for any other id,
    which is forwarded untouched."""
    if not isinstance(model, str):
        return None
    key = model.strip().lower()
    for candidate in dict.fromkeys((base or DS4_MODEL_ALIAS, LEGACY_ALIAS_BASE)):
        for suffix in MODE_SUFFIXES:
            if key == candidate + suffix:
                return suffix, LEGACY_ALIAS_BASE + suffix
    return None

# Sampling parameters recommended by the Qwen3.8-Flash-Next model card ("Best
# Practices": non-thinking mode temperature 0.7 / top_p 0.8 / top_k 20,
# thinking mode temperature 1.0 / top_p 0.95 / top_k 20). ds4-server's own
# defaults are temperature 1.0 / top_p 1.0 / top_k 0 / min_p 0.05, which is far
# too flat for no-think answers, so the proxy fills these in when the client
# did not ask for anything specific. The card's presence_penalty 0..2 range is
# used the same way: 1.5 against repetition in no-think mode, 0 for thinking,
# where a penalty interferes with long reasoning chains.
SAMPLING_NOTHINK = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
}
SAMPLING_THINK = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
}

# Proxied paths that get throughput stats tracked, mapped to the short route
# key used by StreamTracker to pick the right SSE/JSON parsing rules.
STATS_ROUTES = {
    "v1/chat/completions": "chat",
    "v1/completions": "completions",
    "v1/responses": "responses",
    "v1/messages": "messages",
}

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

START_EPOCH = int(time.time())


def log(msg: str) -> None:
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{ts} [ds4-ondemand] {msg}", flush=True)


def log_debug(msg: str) -> None:
    """Verbose per-request logging, off unless DS4_DEBUG is set to non-zero."""
    if DS4_DEBUG:
        log(msg)


def _tail_log(n: int = 200) -> str:
    try:
        text = DS4_LOG_PATH.read_text(errors="replace")
    except FileNotFoundError:
        return "(no ds4-server log file yet)"
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _log_free_memory() -> None:
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
        page_size_match = re.search(r"page size of (\d+) bytes", out)
        page_size = int(page_size_match.group(1)) if page_size_match else 16384

        def pages(label: str) -> int:
            m = re.search(rf"{label}:\s+(\d+)", out)
            return int(m.group(1)) if m else 0

        free_p = pages("Pages free")
        inactive_p = pages("Pages inactive")
        spec_p = pages("Pages speculative")
        avail_gib = (free_p + inactive_p + spec_p) * page_size / (1024**3)
        log(
            f"approx reclaimable memory before ds4-server start: "
            f"{avail_gib:.1f} GiB (vm_stat free+inactive+speculative)"
        )
    except Exception as e:
        log(f"vm_stat check failed (non-fatal): {e}")


# --------------------------------------------------------------------------
# Sampling defaults
# --------------------------------------------------------------------------


def load_sampling_sets(raw: str | None) -> dict[str, dict]:
    """Parse DS4_SAMPLING_DEFAULTS, a JSON object of shape
    {"think": {...}, "nothink": {...}}. Anything unparseable or of the wrong
    shape is logged and ignored in favour of the model-card constants."""
    sets = {"think": dict(SAMPLING_THINK), "nothink": dict(SAMPLING_NOTHINK)}
    if not raw:
        return sets
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("top level is not an object")
        for key in ("think", "nothink"):
            if key not in parsed:
                continue
            value = parsed[key]
            if not isinstance(value, dict):
                raise ValueError(f"{key} is not an object")
            sets[key] = dict(value)
    except Exception as e:
        log(f"DS4_SAMPLING_DEFAULTS ignored, using built-in defaults: {e}")
        return {"think": dict(SAMPLING_THINK), "nothink": dict(SAMPLING_NOTHINK)}
    return sets


SAMPLING_SETS = load_sampling_sets(DS4_SAMPLING_DEFAULTS)


def sampling_defaults_for(model: str | None) -> dict:
    """ds4-server picks no-think mode from the alias suffix (-chat), and
    thinking for every other alias, so the sampling set follows the alias.
    The listed <base>-chat and the legacy -chat id behave the same."""
    resolved = resolve_model_alias(model)
    if resolved is not None:
        return SAMPLING_SETS["nothink" if resolved[0] == "-chat" else "think"]
    if isinstance(model, str) and model.strip().lower().endswith("-chat"):
        return SAMPLING_SETS["nothink"]
    return SAMPLING_SETS["think"]


def apply_sampling_defaults(data: dict, route_key: str) -> list[str]:
    """Fill sampling keys the client left out, in place. A key present in the
    body is client intent and is never overridden, not even when it is null.
    Returns the names of the keys that were injected."""
    if not DS4_SAMPLING_INJECT or not isinstance(data, dict):
        return []
    if route_key not in STATS_ROUTES.values():
        return []
    injected = []
    # All four tracked routes (OpenAI chat/completions, Responses, Anthropic
    # messages) parse temperature, top_p, top_k and presence_penalty in
    # ds4_server.c.
    for key, value in sampling_defaults_for(data.get("model")).items():
        if key not in data:
            data[key] = value
            injected.append(key)
    return injected


class ModelIdRewriter:
    """Puts the requested model id back into a proxied response. ds4-server
    echoes the id it was sent (a legacy alias); clients asked for another one.
    Works on complete lines so SSE events are never delayed."""

    def __init__(self, sent: str, requested: str) -> None:
        self.old = b'"model":' + json.dumps(sent).encode()
        self.new = b'"model":' + json.dumps(requested).encode()
        self.buf = b""

    def feed(self, chunk: bytes) -> bytes:
        self.buf += chunk
        idx = self.buf.rfind(b"\n")
        if idx < 0:
            return b""
        out, self.buf = self.buf[: idx + 1], self.buf[idx + 1 :]
        return out.replace(self.old, self.new)

    def flush(self) -> bytes:
        out, self.buf = self.buf, b""
        return out.replace(self.old, self.new)


# --------------------------------------------------------------------------
# Request throughput stats
# --------------------------------------------------------------------------


def _cached_tokens(usage: dict) -> int | None:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        return details["cached_tokens"]
    return None


class TimingsSseFilter:
    """Line-based pass-through for chat/completions SSE that feeds the tracker
    and adds a top-level `timings` object to the final usage chunk
    (`"choices": []` plus `"usage"`). Only partial lines are held back, so no
    event is delayed; every other line is forwarded byte for byte."""

    def __init__(self, tracker: StreamTracker) -> None:
        self.tracker = tracker
        self.buf = b""

    def feed(self, chunk: bytes) -> bytes:
        self.buf += chunk
        idx = self.buf.rfind(b"\n")
        if idx < 0:
            return b""
        block, self.buf = self.buf[: idx + 1], self.buf[idx + 1 :]
        return b"".join(self._line(line) for line in block.splitlines(keepends=True))

    def flush(self) -> bytes:
        out, self.buf = self.buf, b""
        return self._line(out) if out else b""

    def _line(self, line: bytes) -> bytes:
        try:
            self.tracker.feed_sse(line if line.endswith(b"\n") else line + b"\n")
        except Exception as e:
            log(f"stats: parse failed (non-fatal): {e}")
        if not line.startswith(b"data:") or b'"usage"' not in line:
            return line
        try:
            obj = json.loads(line[len(b"data:") :])
            if not isinstance(obj, dict) or obj.get("choices") != [] or not obj.get("usage"):
                return line
            timings = self.tracker.timings()
            if timings is None:
                return line
            obj["timings"] = timings
            ending = line[len(line.rstrip(b"\r\n")) :]
            return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + ending
        except Exception as e:
            log(f"timings: injection failed (non-fatal): {e}")
            return line


class StreamTracker:
    """Tracks token throughput for a single proxied request, from the moment
    it is dispatched to ds4-server until it finishes. Handles both SSE
    (streaming) and plain JSON (non-streaming) responses for the four
    tracked routes: chat, completions, responses, messages."""

    def __init__(self, route: str, model: str | None) -> None:
        self.route = route
        self.model = model
        self.start = time.monotonic()
        self.first_delta_time: float | None = None
        self.gen_tokens = 0
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.cached_tokens: int | None = None
        self._sse_buffer = b""
        self._pending_event: str | None = None

    def note_content_delta(self) -> None:
        now = time.monotonic()
        if self.first_delta_time is None:
            self.first_delta_time = now
        self.gen_tokens += 1

    def note_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None = None,
    ) -> None:
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self.completion_tokens = completion_tokens
        if cached_tokens is not None:
            self.cached_tokens = cached_tokens

    def feed_sse(self, chunk: bytes) -> None:
        """Parse SSE lines out of a raw byte chunk, purely for stats. Never
        alters or delays the chunk itself -- callers pass the same bytes
        through to the client regardless of what happens here."""
        self._sse_buffer += chunk
        while b"\n" in self._sse_buffer:
            line, self._sse_buffer = self._sse_buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip("\r")
            if not text:
                continue
            if text.startswith("event:"):
                self._pending_event = text[len("event:"):].strip()
                continue
            if not text.startswith("data:"):
                continue
            payload = text[len("data:"):].strip()
            event = self._pending_event
            self._pending_event = None
            if payload == "[DONE]":
                continue
            self._handle_event(event, json.loads(payload))

    def _handle_event(self, event: str | None, obj: dict) -> None:
        if self.route in ("chat", "completions"):
            choices = obj.get("choices") or []
            if choices:
                choice = choices[0]
                if self.route == "chat":
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content")
                else:
                    text = choice.get("text")
                if text:
                    self.note_content_delta()
            usage = obj.get("usage")
            if usage:
                self.note_usage(
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    _cached_tokens(usage),
                )
        elif self.route == "responses":
            if event == "response.output_text.delta" and obj.get("delta"):
                self.note_content_delta()
            elif event == "response.completed":
                usage = (obj.get("response") or {}).get("usage") or {}
                if usage:
                    self.note_usage(usage.get("input_tokens"), usage.get("output_tokens"))
        elif self.route == "messages":
            if event == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    self.note_content_delta()
            elif event == "message_start":
                usage = (obj.get("message") or {}).get("usage") or {}
                if usage.get("input_tokens") is not None:
                    self.prompt_tokens = usage.get("input_tokens")
            elif event == "message_delta":
                usage = obj.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    self.note_usage(None, usage.get("output_tokens"))

    def parse_full_body(self, raw: bytes) -> None:
        """For non-streaming responses: pull usage out of the complete JSON
        body once it has all arrived."""
        obj = json.loads(raw)
        if self.route in ("chat", "completions"):
            usage = obj.get("usage") or {}
            self.note_usage(
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                _cached_tokens(usage),
            )
        else:  # responses, messages
            usage = obj.get("usage") or {}
            self.note_usage(usage.get("input_tokens"), usage.get("output_tokens"))

    def live_snapshot(self) -> dict:
        now = time.monotonic()
        if self.first_delta_time is None:
            return {
                "gen_tokens": self.gen_tokens,
                "gen_tps": 0.0,
                "elapsed_s": now - self.start,
                "phase": "prefill",
            }
        decode_elapsed = now - self.first_delta_time
        gen_tps = self.gen_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
        return {
            "gen_tokens": self.gen_tokens,
            "gen_tps": gen_tps,
            "elapsed_s": now - self.start,
            "phase": "decode",
        }

    def timings(self, end: float | None = None) -> dict | None:
        """llama.cpp-style per-request timings for the final usage chunk or a
        non-streaming body. None when the usage numbers are missing. A request
        with no content delta (non-streaming) falls back to the whole request
        duration for both phases, like finalize() does."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        end = time.monotonic() if end is None else end
        first = self.first_delta_time if self.first_delta_time is not None else self.start
        cached = min(max(self.cached_tokens or 0, 0), self.prompt_tokens)
        prompt_n = self.prompt_tokens - cached
        prompt_s = (first if self.first_delta_time is not None else end) - self.start
        predicted_s = end - first
        return {
            "prompt_n": prompt_n,
            "prompt_ms": prompt_s * 1000.0,
            "prompt_per_second": prompt_n / prompt_s if prompt_s > 0 else 0.0,
            "predicted_n": self.completion_tokens,
            "predicted_ms": predicted_s * 1000.0,
            "predicted_per_second": (
                self.completion_tokens / predicted_s if predicted_s > 0 else 0.0
            ),
            "cache_n": cached,
        }

    def finalize(self) -> dict:
        end = time.monotonic()
        duration = end - self.start
        completion_tokens = (
            self.completion_tokens if self.completion_tokens is not None else self.gen_tokens
        )
        if self.first_delta_time is not None:
            ttft_ms = (self.first_delta_time - self.start) * 1000
            decode_elapsed = end - self.first_delta_time
            gen_tps = completion_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
        else:
            ttft_ms = duration * 1000
            gen_tps = completion_tokens / duration if duration > 0 else 0.0
        prefill_tps = None
        if self.prompt_tokens is not None and ttft_ms > 0:
            prefill_tps = self.prompt_tokens / (ttft_ms / 1000)
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft_ms,
            "prefill_tps": prefill_tps,
            "gen_tps": gen_tps,
            "duration_s": duration,
            "finished_at": time.time(),
        }


class Stats:
    """In-memory request throughput stats for /admin/status. Reset whenever
    the launcher restarts."""

    def __init__(self) -> None:
        self.requests = 0
        self.completion_tokens = 0
        self.errors = 0
        self.last: dict | None = None
        self._active: list[StreamTracker] = []

    def start_tracker(self, route: str, model: str | None) -> StreamTracker:
        tracker = StreamTracker(route, model)
        self._active.append(tracker)
        return tracker

    def finish_tracker(self, tracker: StreamTracker, *, error: bool = False) -> None:
        if tracker in self._active:
            self._active.remove(tracker)
        self.requests += 1
        if error:
            self.errors += 1
            return
        result = tracker.finalize()
        self.completion_tokens += result["completion_tokens"]
        self.last = result

    def snapshot(self) -> dict:
        live = self._active[-1].live_snapshot() if self._active else None
        return {
            "live": live,
            "last": self.last,
            "totals": {
                "requests": self.requests,
                "completion_tokens": self.completion_tokens,
                "errors": self.errors,
            },
        }


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


class ServerStartError(Exception):
    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ServerHeldError(Exception):
    """A cold start was refused because an unexpired hold exists."""

    def __init__(self, reason: str, retry_after: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


def _start_http_error(e: ServerStartError) -> HTTPException:
    headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
    return HTTPException(status_code=503, detail=str(e), headers=headers)


def _held_response(e: ServerHeldError) -> JSONResponse:
    return JSONResponse(
        {"error": "held", "reason": e.reason},
        status_code=503,
        headers={"Retry-After": str(e.retry_after)},
    )


def parse_env_ctx(raw) -> int:
    try:
        value = int(str(raw).strip())
        if value <= 0:
            raise ValueError("not positive")
        return value
    except (TypeError, ValueError):
        log(f"DS4_CTX={raw!r} is not a positive integer, using {CTX_DEFAULT}")
        return CTX_DEFAULT


def normalize_ctx(value) -> int:
    """Validates a requested ctx and rounds it to the nearest multiple of 256.
    Raises ValueError with a client-facing message."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("ctx must be an integer")
    if value < CTX_MIN or value > CTX_MAX:
        raise ValueError(f"ctx must be between {CTX_MIN} and {CTX_MAX}")
    return (value + CTX_STEP // 2) // CTX_STEP * CTX_STEP


def load_persisted_ctx() -> int | None:
    try:
        data = json.loads(DS4_STATE_PATH.read_text())
        return normalize_ctx(data["ctx"])
    except FileNotFoundError:
        return None
    except Exception as e:
        log(f"ignoring persisted ctx in {DS4_STATE_PATH}: {e}")
        return None


def save_persisted_ctx(ctx: int) -> None:
    try:
        data = json.loads(DS4_STATE_PATH.read_text())
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data["ctx"] = ctx
    DS4_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DS4_STATE_PATH.with_name(DS4_STATE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, DS4_STATE_PATH)


def apply_persisted_ctx() -> None:
    """Startup: the persisted value overrides DS4_CTX."""
    persisted = load_persisted_ctx()
    if persisted is not None:
        state.ctx = persisted
        log(f"ctx {persisted} loaded from {DS4_STATE_PATH}")


class State:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.ready = False
        self.lock = asyncio.Lock()
        self.in_flight = 0
        self.last_activity = time.monotonic()
        self.start_time: float | None = None
        self.http_client: httpx.AsyncClient | None = None
        self.idle_task: asyncio.Task | None = None
        self.stats = Stats()
        self.ctx = parse_env_ctx(DS4_CTX)
        self.ctx_active: int | None = None
        self.pending_restart = False
        self.pending_task: asyncio.Task | None = None
        # Cold starts that have begun (counted before their first await) and
        # not yet finished; `starting` is true while any exist.
        self.starting_count = 0
        # hold_id -> {"reason": str, "expires": monotonic deadline}
        self.holds: dict[str, dict] = {}
        # Set by the SIGTERM/SIGINT handler: streams end, no new cold starts.
        self.stopping = False
        self.log_task: asyncio.Task | None = None

    @property
    def starting(self) -> bool:
        return self.starting_count > 0


state = State()


# --------------------------------------------------------------------------
# Holds
# --------------------------------------------------------------------------


def _purge_holds() -> None:
    now = time.monotonic()
    for hold_id in [h for h, v in state.holds.items() if v["expires"] <= now]:
        del state.holds[hold_id]


def _active_holds() -> list[dict]:
    _purge_holds()
    now = time.monotonic()
    return [
        {
            "id": hold_id,
            "reason": v["reason"],
            "ttl_remaining_s": max(0.0, v["expires"] - now),
        }
        for hold_id, v in state.holds.items()
    ]


def _raise_if_held() -> None:
    holds = _active_holds()
    if not holds:
        return
    first = min(holds, key=lambda h: h["ttl_remaining_s"])
    retry = max(1, min(HOLD_RETRY_AFTER_MAX, math.ceil(first["ttl_remaining_s"])))
    raise ServerHeldError(first["reason"], retry)


def _raise_if_stopping() -> None:
    if state.stopping:
        raise ServerStartError(
            "launcher is shutting down", retry_after=START_RETRY_AFTER
        )

# --------------------------------------------------------------------------
# oMLX interplay
# --------------------------------------------------------------------------


async def _omlx_resident_ids() -> list[str] | None:
    """Ids oMLX has loaded or is loading; None when oMLX is unreachable."""
    url = f"{OMLX_BASE_URL}/v1/models/status"
    try:
        r = await state.http_client.get(url, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        return [
            m["id"]
            for m in data.get("models", [])
            if m.get("loaded") or m.get("is_loading")
        ]
    except Exception as e:
        log(f"omlx status check failed (tolerated, omlx may be stopped): {e}")
        return None


async def _unload_omlx_models() -> None:
    """Unload every oMLX model, then re-poll until nothing is loaded or
    loading. Raises ServerStartError (503) if it is not clear after
    OMLX_UNLOAD_WAIT seconds, so ds4-server is never spawned next to a
    resident oMLX model. An unreachable oMLX counts as clear."""
    ids = await _omlx_resident_ids()
    if not ids:
        log("omlx: nothing loaded to unload (or omlx unreachable)")
        return
    for mid in ids:
        url = f"{OMLX_BASE_URL}/v1/models/{quote(mid, safe='')}/unload"
        try:
            r = await state.http_client.post(url, timeout=30.0)
            log(f"omlx unload {mid}: HTTP {r.status_code}")
        except Exception as e:
            log(f"omlx unload {mid} failed (tolerated): {e}")
    deadline = time.monotonic() + OMLX_UNLOAD_WAIT
    while True:
        remaining = await _omlx_resident_ids()
        if not remaining:
            return
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    log(
        f"omlx still has {remaining} loaded or loading after "
        f"{OMLX_UNLOAD_WAIT:.0f}s; refusing cold start"
    )
    raise ServerStartError(
        f"oMLX still has models loaded or loading after {OMLX_UNLOAD_WAIT:.0f}s "
        f"({', '.join(remaining)}); refusing to start ds4-server",
        retry_after=START_RETRY_AFTER,
    )


async def _free_comfyui() -> None:
    """Ask ComfyUI to drop its models before ds4-server loads. Only when its
    queue is empty; never raises, never delays a start by more than a few
    seconds."""
    if not DS4_COMFYUI_URL:
        return
    base = DS4_COMFYUI_URL.rstrip("/")
    try:
        r = await state.http_client.get(f"{base}/queue", timeout=COMFYUI_TIMEOUT)
        r.raise_for_status()
        queue = r.json()
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        if running or pending:
            log(
                f"comfyui busy ({len(running or [])} running, "
                f"{len(pending or [])} pending); leaving its models loaded"
            )
            return
        r = await state.http_client.post(
            f"{base}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=COMFYUI_TIMEOUT,
        )
        log(f"comfyui free before ds4 start: HTTP {r.status_code}")
    except Exception as e:
        log(f"comfyui free skipped (non-fatal): {e}")


# --------------------------------------------------------------------------
# ds4-server process management
# --------------------------------------------------------------------------


_ds4_logger = logging.getLogger("ds4-ondemand.ds4-server-output")
_ds4_logger.propagate = False
_ds4_logger.setLevel(logging.INFO)
_ds4_handler: logging.handlers.RotatingFileHandler | None = None


def _ds4_output_logger() -> logging.Logger:
    """Logger that owns ds4-server.log: 10 MB x 3 files, message only. The
    handler is rebuilt if DS4_LOG_PATH changed (tests)."""
    global _ds4_handler
    path = str(DS4_LOG_PATH)
    if _ds4_handler is None or _ds4_handler.baseFilename != os.path.abspath(path):
        if _ds4_handler is not None:
            _ds4_logger.removeHandler(_ds4_handler)
            _ds4_handler.close()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _ds4_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=DS4_LOG_MAX_BYTES,
            backupCount=DS4_LOG_BACKUPS,
            encoding="utf-8",
            errors="replace",
        )
        _ds4_handler.setFormatter(logging.Formatter("%(message)s"))
        _ds4_logger.addHandler(_ds4_handler)
    return _ds4_logger


async def _pump_output(stream) -> None:
    """Copy the child's stdout/stderr into the rotating ds4-server.log until
    EOF. Never raises."""
    out = _ds4_output_logger()
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
            out.info(line.decode("utf-8", errors="replace").rstrip("\r\n"))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log(f"ds4-server output pump stopped (non-fatal): {e}")


async def _spawn() -> asyncio.subprocess.Process:
    if not DS4_BINARY.exists():
        raise ServerStartError(f"ds4-server binary not found: {DS4_BINARY}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ctx = state.ctx
    args = [
        str(DS4_BINARY),
        "-m",
        DS4_MODEL_FILE,
        "--ctx",
        str(ctx),
        "--prefill-chunk",
        str(DS4_PREFILL_CHUNK),
        "--mtp",
        "--host",
        DS4_SERVER_HOST,
        "--port",
        str(DS4_SERVER_PORT),
    ]
    # Relative paths resolve against the child's cwd (DS4_WORKDIR); absolute
    # DS4_MODEL_FILE / DS4_VISION_FILE values are used as given.
    if DS4_VISION_FILE and (DS4_WORKDIR / DS4_VISION_FILE).exists():
        args += ["--vision", DS4_VISION_FILE]
    else:
        log(f"vision encoder not found, starting without --vision: {DS4_VISION_FILE}")
    log(f"spawning: {' '.join(args)} (cwd={DS4_WORKDIR})")
    _ds4_output_logger().info(
        f"\n===== ds4-ondemand spawn {datetime.now().isoformat()} ====="
    )
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(DS4_WORKDIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout is not None:
        state.log_task = asyncio.create_task(_pump_output(proc.stdout))
    state.ctx_active = ctx
    return proc


async def _terminate_process(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None or proc.returncode is not None:
        return
    log(f"sending SIGTERM to ds4-server pid={proc.pid}")
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        log(f"ds4-server pid={proc.pid} still alive after 10s, sending SIGKILL")
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
    log(f"ds4-server pid={proc.pid} exited with code {proc.returncode}")


async def _wait_ready(timeout: float) -> tuple[bool, str | None]:
    url = f"http://{DS4_SERVER_HOST}:{DS4_SERVER_PORT}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.stopping:
            return False, "launcher is shutting down"
        if state.process is not None and state.process.returncode is not None:
            return False, f"process exited early with code {state.process.returncode}"
        try:
            r = await state.http_client.get(url, timeout=2.0)
            if r.status_code == 200:
                return True, None
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False, f"timed out after {timeout}s"


async def _stop_locked() -> None:
    if state.process is not None:
        await _terminate_process(state.process)
    state.process = None
    state.ready = False
    state.start_time = None
    state.ctx_active = None
    # The next spawn reads state.ctx, so any stop applies a pending ctx change.
    state.pending_restart = False


class BusyError(Exception):
    """stop?if_idle=1 refused: a request or a cold start is active."""


async def stop_ds4_server(if_idle: bool = False) -> None:
    """Stop ds4-server. With if_idle, raises BusyError instead when a request
    is in flight or a cold start is in progress (checked again under the lock,
    since a start that began meanwhile holds it)."""
    if if_idle and (state.in_flight > 0 or state.starting):
        raise BusyError
    async with state.lock:
        if if_idle and (state.in_flight > 0 or state.starting):
            raise BusyError
        await _stop_locked()


async def shutdown_stop_ds4() -> None:
    """Lifespan stop. A cold start can hold state.lock for minutes, so wait a
    bounded time for it and then terminate the child without the lock."""
    state.stopping = True
    try:
        await asyncio.wait_for(state.lock.acquire(), timeout=SHUTDOWN_LOCK_TIMEOUT)
    except asyncio.TimeoutError:
        log(
            f"state.lock still held after {SHUTDOWN_LOCK_TIMEOUT:.0f}s; "
            f"terminating ds4-server without it"
        )
        await _terminate_process(state.process)
        return
    try:
        await _stop_locked()
    finally:
        state.lock.release()


def _is_up() -> bool:
    return state.ready and state.process is not None and state.process.returncode is None


async def ensure_started() -> None:
    """Start ds4-server if it is not already up and ready. Single-start
    guarded by state.lock, so concurrent cold requests trigger exactly one
    spawn. `starting` is raised before the first await (the lock wait), so
    /admin/status reports it for the whole oMLX unload."""
    if _is_up():
        return
    _raise_if_stopping()
    _raise_if_held()
    state.starting_count += 1
    try:
        async with state.lock:
            if _is_up():
                return
            # A hold placed, or a shutdown begun, while we waited for the lock.
            _raise_if_stopping()
            _raise_if_held()

            if state.process is not None and state.process.returncode is not None:
                log(
                    f"previous ds4-server (pid={state.process.pid}) exited "
                    f"(code {state.process.returncode}); resetting state"
                )
                state.process = None
                state.ready = False
                state.ctx_active = None

            log("cold start requested: unloading oMLX models before starting ds4-server")
            await _unload_omlx_models()
            await _free_comfyui()
            _log_free_memory()
            _raise_if_stopping()
            _raise_if_held()

            proc = await _spawn()
            state.process = proc
            state.start_time = time.monotonic()
            # A ctx change that raced the spawn leaves the child on the old value.
            state.pending_restart = state.ctx != state.ctx_active

            ok, err = await _wait_ready(DS4_START_TIMEOUT)
            if not ok:
                log(f"ds4-server did not become ready: {err}")
                await _terminate_process(proc)
                state.process = None
                state.ready = False
                state.start_time = None
                state.ctx_active = None
                tail = _tail_log()
                raise ServerStartError(
                    f"ds4-server failed to become ready ({err}). Log tail:\n{tail}"
                )

            state.ready = True
            log(
                f"ds4-server ready after {time.monotonic() - state.start_time:.1f}s, "
                f"pid={proc.pid}"
            )
    finally:
        state.starting_count -= 1


async def idle_watcher() -> None:
    while True:
        await asyncio.sleep(5)
        try:
            _purge_holds()
            async with state.lock:
                if state.ready and state.in_flight == 0 and state.pending_restart:
                    log("applying pending ctx change; stopping ds4-server")
                    await _stop_locked()
                elif state.ready and state.in_flight == 0:
                    idle_for = time.monotonic() - state.last_activity
                    if idle_for >= DS4_IDLE_SECONDS:
                        log(
                            f"idle for {idle_for:.0f}s (limit {DS4_IDLE_SECONDS:.0f}s) "
                            f"with 0 in-flight requests; stopping ds4-server"
                        )
                        await _stop_locked()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log(f"idle watcher error (non-fatal): {e}")


def _server_alive() -> bool:
    return state.process is not None and state.process.returncode is None


async def _apply_pending_restart() -> None:
    """Deferred half of POST /admin/config: stop once the last in-flight
    request is done, so the next request respawns with the new ctx. Never
    pre-warms (the model is ~41 GiB)."""
    async with state.lock:
        if state.in_flight != 0 or not state.pending_restart:
            return
        if _server_alive():
            log(f"in-flight requests finished; stopping ds4-server to apply ctx {state.ctx}")
            await _stop_locked()
        else:
            state.pending_restart = False


def _kick_pending_restart() -> None:
    """Call right after an in_flight decrement. Runs as a task so the stop's
    SIGTERM wait never holds a client response open."""
    if state.pending_restart and state.in_flight == 0:
        state.pending_task = asyncio.create_task(_apply_pending_restart())


def _config_payload() -> dict:
    return {
        "ctx": state.ctx,
        "ctx_active": state.ctx_active if _server_alive() else None,
        "ctx_min": CTX_MIN,
        "ctx_max": CTX_MAX,
        "pending_restart": state.pending_restart,
    }


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_persisted_ctx()
    state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
    )
    state.idle_task = asyncio.create_task(idle_watcher())
    log(
        f"ds4-ondemand listening on {DS4_ONDEMAND_HOST}:{DS4_ONDEMAND_PORT}, "
        f"proxying to {DS4_SERVER_HOST}:{DS4_SERVER_PORT}, "
        f"idle_seconds={DS4_IDLE_SECONDS}, start_timeout={DS4_START_TIMEOUT}, "
        f"omlx={OMLX_BASE_URL}"
    )
    try:
        yield
    finally:
        log("shutting down: stopping ds4-server if running")
        if state.idle_task is not None:
            state.idle_task.cancel()
        await shutdown_stop_ds4()
        if state.http_client is not None:
            await state.http_client.aclose()


app = FastAPI(lifespan=lifespan)


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


class LoopbackOnlyAdminMiddleware:
    """Pure ASGI middleware (no response buffering, so proxied SSE is
    untouched): every /admin/* request must come from a loopback peer, else
    403. Uses the socket peer address, never forwarded headers."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            path = "/" + scope.get("path", "").lstrip("/")
            if path == "/admin" or path.startswith("/admin/"):
                client = scope.get("client")
                if not client or not is_loopback_host(client[0]):
                    log(
                        f"refused {scope.get('method')} {path} from "
                        f"{client[0] if client else 'unknown'} (admin is loopback-only)"
                    )
                    resp = JSONResponse(
                        {"error": "forbidden", "detail": "/admin is loopback-only"},
                        status_code=403,
                    )
                    await resp(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app.add_middleware(LoopbackOnlyAdminMiddleware)


@app.get("/v1/models")
async def list_models():
    """Answers from a static alias list WITHOUT starting ds4-server, so
    model pickers work while it is cold."""
    loaded = state.ready
    data = [
        {
            "id": model_id,
            "object": "model",
            "created": START_EPOCH,
            "owned_by": "dwarfstar",
            "loaded": loaded,
        }
        for model_id in STATIC_MODELS
    ]
    return {"object": "list", "data": data}


@app.get("/admin/status")
async def admin_status():
    now = time.monotonic()
    idle_for = now - state.last_activity
    remaining = None
    if state.ready and state.in_flight == 0:
        remaining = max(0.0, DS4_IDLE_SECONDS - idle_for)
    return {
        "loaded": state.ready,
        "starting": state.starting,
        "holds": _active_holds(),
        "pid": state.process.pid if state.process is not None else None,
        "uptime_seconds": (
            (now - state.start_time) if (state.ready and state.start_time) else None
        ),
        "last_activity_seconds_ago": idle_for,
        "in_flight": state.in_flight,
        "idle_seconds_remaining": remaining,
        "stats": state.stats.snapshot(),
        "config": {
            "ds4_ondemand_host": DS4_ONDEMAND_HOST,
            "ds4_ondemand_port": DS4_ONDEMAND_PORT,
            "ds4_server_host": DS4_SERVER_HOST,
            "ds4_server_port": DS4_SERVER_PORT,
            "ds4_repo_dir": str(DS4_REPO_DIR),
            "ds4_workdir": str(DS4_WORKDIR),
            "ds4_binary": str(DS4_BINARY),
            "ds4_model_file": DS4_MODEL_FILE,
            "model_alias": DS4_MODEL_ALIAS,
            "ds4_vision_file": DS4_VISION_FILE,
            "ds4_ctx": state.ctx,
            "ctx": state.ctx,
            "ctx_active": state.ctx_active if _server_alive() else None,
            "pending_restart": state.pending_restart,
            "ds4_prefill_chunk": DS4_PREFILL_CHUNK,
            "ds4_start_timeout": DS4_START_TIMEOUT,
            "ds4_idle_seconds": DS4_IDLE_SECONDS,
            "omlx_base_url": OMLX_BASE_URL,
            "sampling_inject": DS4_SAMPLING_INJECT,
            "sampling_defaults": SAMPLING_SETS,
        },
    }


@app.get("/admin/config")
async def admin_config_get():
    return _config_payload()


@app.post("/admin/config")
async def admin_config_set(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict) or "ctx" not in body:
        raise HTTPException(status_code=400, detail='body must be {"ctx": <int>}')
    try:
        ctx = normalize_ctx(body["ctx"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        save_persisted_ctx(ctx)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"could not persist ctx: {e}")
    state.ctx = ctx
    log(f"ctx set to {ctx} (persisted to {DS4_STATE_PATH})")

    if not _server_alive():
        state.pending_restart = False
        applied = "next_start"
    elif state.ctx_active == ctx:
        state.pending_restart = False
        applied = "unchanged"
    elif state.in_flight > 0:
        state.pending_restart = True
        applied = "after_current_requests"
    else:
        # Checked before taking the lock: a cold start holds it for up to
        # DS4_START_TIMEOUT, and that start already counts as in-flight.
        async with state.lock:
            if not _server_alive():
                state.pending_restart = False
                applied = "next_start"
            elif state.in_flight > 0:
                state.pending_restart = True
                applied = "after_current_requests"
            else:
                log(f"ctx change to {ctx} while idle; stopping ds4-server")
                await _stop_locked()
                applied = "restarted"
    return {**_config_payload(), "applied": applied}


@app.post("/admin/stop")
async def admin_stop(if_idle: str = "0"):
    """With ?if_idle=1: 409 instead of stopping while a request is in flight
    or a cold start is in progress. Without it, always stops."""
    try:
        await stop_ds4_server(if_idle=if_idle.strip().lower() in ("1", "true", "yes"))
    except BusyError:
        return JSONResponse(
            {
                "error": "busy",
                "in_flight": state.in_flight,
                "starting": state.starting,
            },
            status_code=409,
        )
    return {"status": "ok", "loaded": state.ready}


@app.post("/admin/hold")
async def admin_hold_create(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail='body must be {"reason": str, "ttl_s": int}'
        )
    reason = body.get("reason")
    ttl = body.get("ttl_s")
    if not isinstance(reason, str):
        raise HTTPException(status_code=400, detail="reason must be a string")
    if isinstance(ttl, bool) or not isinstance(ttl, int):
        raise HTTPException(status_code=400, detail="ttl_s must be an integer")
    if ttl < HOLD_TTL_MIN or ttl > HOLD_TTL_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"ttl_s must be between {HOLD_TTL_MIN} and {HOLD_TTL_MAX}",
        )
    _purge_holds()
    hold_id = secrets.token_hex(8)
    reason = reason.strip()[:200]
    state.holds[hold_id] = {"reason": reason, "expires": time.monotonic() + ttl}
    log(f"hold {hold_id} created: {reason!r} ttl={ttl}s")
    return {"hold_id": hold_id}


@app.delete("/admin/hold/{hold_id}")
async def admin_hold_delete(hold_id: str):
    _purge_holds()
    if state.holds.pop(hold_id, None) is None:
        raise HTTPException(status_code=404, detail="no such hold (released or expired)")
    log(f"hold {hold_id} released")
    return {"status": "ok", "released": hold_id}


@app.post("/admin/start")
async def admin_start():
    try:
        await ensure_started()
    except ServerHeldError as e:
        return _held_response(e)
    except ServerStartError as e:
        raise _start_http_error(e)
    return {
        "status": "ok",
        "loaded": state.ready,
        "pid": state.process.pid if state.process is not None else None,
    }


def _filtered_headers(headers) -> list[tuple[str, str]]:
    return [(k, v) for k, v in headers.items() if k.lower() not in HOP_BY_HOP]


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(full_path: str, request: Request):
    if not full_path.startswith("v1/"):
        raise HTTPException(status_code=404, detail="not found")

    state.in_flight += 1
    state.last_activity = time.monotonic()
    started_stream = False
    route_key = STATS_ROUTES.get(full_path)
    trackable = route_key is not None and request.method == "POST"
    tracker: StreamTracker | None = None
    rewriter: ModelIdRewriter | None = None
    try:
        try:
            await ensure_started()
        except ServerHeldError as e:
            return _held_response(e)
        except ServerStartError as e:
            raise _start_http_error(e)

        body = await request.body()

        if trackable:
            model_hint = None
            try:
                data = json.loads(body)
                model_hint = data.get("model")
                rewritten = False
                injected = apply_sampling_defaults(data, route_key)
                resolved = resolve_model_alias(model_hint)
                if resolved is not None and resolved[1] != model_hint:
                    data["model"] = resolved[1]
                    rewriter = ModelIdRewriter(resolved[1], model_hint)
                    rewritten = True
                if injected:
                    rewritten = True
                    shown = " ".join(f"{k}={data[k]}" for k in injected)
                    log_debug(
                        f"sampling defaults injected for {model_hint}: {shown}"
                    )
                if (
                    route_key in ("chat", "completions")
                    and data.get("stream") is True
                    and "stream_options" not in data
                ):
                    data["stream_options"] = {"include_usage": True}
                    rewritten = True
                if rewritten:
                    body = json.dumps(data).encode()
            except Exception as e:
                log(f"stats: failed to parse request body for {full_path} (non-fatal): {e}")
            tracker = state.stats.start_tracker(route_key, model_hint)

        target = f"http://{DS4_SERVER_HOST}:{DS4_SERVER_PORT}/{full_path}"
        if request.url.query:
            target += "?" + request.url.query

        req_headers = _filtered_headers(request.headers)
        upstream_req = state.http_client.build_request(
            request.method, target, headers=req_headers, content=body
        )
        upstream = await state.http_client.send(upstream_req, stream=True)
        started_stream = True
    except Exception:
        if tracker is not None:
            state.stats.finish_tracker(tracker, error=True)
        raise
    finally:
        if not started_stream:
            state.in_flight -= 1
            state.last_activity = time.monotonic()
            _kick_pending_restart()

    resp_headers = _filtered_headers(upstream.headers)
    is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")
    upstream_is_error = upstream.status_code >= 400
    non_stream_buffer = bytearray() if (tracker is not None and not is_sse) else None
    timings_route = route_key in ("chat", "completions") and not upstream_is_error
    sse_filter = (
        TimingsSseFilter(tracker)
        if (tracker is not None and is_sse and timings_route)
        else None
    )

    def add_timings(raw: bytes) -> bytes:
        """Non-streaming chat/completions body: parse usage, add timings."""
        tracker.parse_full_body(raw)
        timings = tracker.timings()
        if timings is None:
            return raw
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return raw
        obj["timings"] = timings
        return json.dumps(obj, separators=(",", ":")).encode()

    async def body_iter():
        try:
            async for chunk in upstream.aiter_bytes():
                if sse_filter is not None:
                    chunk = sse_filter.feed(chunk)
                elif tracker is not None:
                    try:
                        if is_sse:
                            tracker.feed_sse(chunk)
                        elif non_stream_buffer is not None:
                            non_stream_buffer.extend(chunk)
                    except Exception as e:
                        log(f"stats: parse failed for {full_path} (non-fatal): {e}")
                if non_stream_buffer is not None and timings_route:
                    continue  # held until the whole body can carry timings
                if chunk:
                    yield rewriter.feed(chunk) if rewriter is not None else chunk
                if is_sse and state.stopping:
                    log(f"shutting down: ending stream for {full_path}")
                    break
            if sse_filter is not None:
                tail = sse_filter.flush()
                if tail:
                    yield rewriter.feed(tail) if rewriter is not None else tail
            elif non_stream_buffer is not None and timings_route:
                raw = bytes(non_stream_buffer)
                try:
                    raw = add_timings(raw)
                    non_stream_buffer.clear()  # already parsed for stats
                except Exception as e:
                    log(f"timings: full-body injection failed for {full_path} (non-fatal): {e}")
                if raw:
                    yield rewriter.feed(raw) if rewriter is not None else raw
            if rewriter is not None:
                tail = rewriter.flush()
                if tail:
                    yield tail
        finally:
            await upstream.aclose()
            state.in_flight -= 1
            state.last_activity = time.monotonic()
            _kick_pending_restart()
            if tracker is not None:
                error = upstream_is_error
                if not error and non_stream_buffer:
                    try:
                        tracker.parse_full_body(bytes(non_stream_buffer))
                    except Exception as e:
                        log(f"stats: full-body parse failed for {full_path} (non-fatal): {e}")
                state.stats.finish_tracker(tracker, error=error)

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=dict(resp_headers),
    )


ACCESS_LOG_QUIET_PATHS = {"/admin/status", "/v1/models", "/admin/config"}


class AccessLogFilter(logging.Filter):
    """Drops uvicorn access-log lines for the polled GET endpoints (/admin/status,
    /v1/models, /admin/config); everything else is logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            method, target = args[1], str(args[2])
            if method == "GET" and target.split("?", 1)[0] in ACCESS_LOG_QUIET_PATHS:
                return False
        return True


def install_access_log_filter() -> None:
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, AccessLogFilter) for f in logger.filters):
        logger.addFilter(AccessLogFilter())


class LauncherServer(uvicorn.Server):
    """uvicorn.Server whose SIGTERM/SIGINT handler first raises state.stopping,
    so open SSE streams end promptly and no cold start begins while the
    lifespan stop runs."""

    def handle_exit(self, sig, frame) -> None:
        state.stopping = True
        super().handle_exit(sig, frame)


def build_server(host: str, port: int) -> LauncherServer:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        # Peer address only: /admin loopback check must not trust X-Forwarded-For.
        proxy_headers=False,
    )
    # After Config: its logging setup would not remove this, but be explicit.
    install_access_log_filter()
    return LauncherServer(config)


if __name__ == "__main__":
    build_server(DS4_ONDEMAND_HOST, DS4_ONDEMAND_PORT).run()
