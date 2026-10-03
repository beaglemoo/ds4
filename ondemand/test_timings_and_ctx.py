"""Tests for the llama.cpp-style `timings` object and the runtime ctx API.

No ds4-server is ever spawned: upstream traffic goes through an httpx
MockTransport and the server process is a fake. Run with:

    cd ~/Homelab/dwarfstar
    uv run --with fastapi --with uvicorn --with httpx \
        python -m unittest ondemand/test_timings_and_ctx.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds4_ondemand as od  # noqa: E402

TIMINGS_KEYS = {
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "cache_n",
}


def sse(obj: dict | str) -> bytes:
    payload = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()


def content_chunk(text: str, model: str = "qwen3.8-flash-next-chat") -> bytes:
    return sse(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


def usage_chunk(prompt: int, completion: int, cached: int, model: str = "qwen3.8-flash-next-chat") -> bytes:
    return sse(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            },
        }
    )


def parse_sse(raw: bytes) -> list:
    out = []
    for line in raw.decode().splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


class FakeProc:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def reset_state() -> None:
    s = od.state
    s.process = None
    s.ready = False
    s.lock = asyncio.Lock()
    s.in_flight = 0
    s.start_time = None
    s.ctx = 65536
    s.ctx_active = None
    s.pending_restart = False
    s.pending_task = None
    s.stats = od.Stats()


class TimingsUnitTest(unittest.TestCase):
    def test_timings_math(self) -> None:
        t = od.StreamTracker("chat", "m")
        t.start = 100.0
        t.first_delta_time = 100.5
        t.note_usage(1000, 40, 600)
        got = t.timings(end=102.5)
        self.assertEqual(
            got,
            {
                "prompt_n": 400,
                "prompt_ms": 500.0,
                "prompt_per_second": 800.0,
                "predicted_n": 40,
                "predicted_ms": 2000.0,
                "predicted_per_second": 20.0,
                "cache_n": 600,
            },
        )

    def test_missing_usage_gives_none(self) -> None:
        t = od.StreamTracker("chat", "m")
        self.assertIsNone(t.timings())
        t.note_usage(10, None)
        self.assertIsNone(t.timings())

    def test_filter_does_not_delay_and_injects_only_into_usage_chunk(self) -> None:
        t = od.StreamTracker("chat", "m")
        f = od.TimingsSseFilter(t)
        first = content_chunk("Hel")
        self.assertEqual(f.feed(first), first)  # forwarded immediately, unchanged
        second = content_chunk("lo")
        self.assertEqual(f.feed(second), second)

        usage = usage_chunk(500, 2, 128)
        cut = len(usage) // 2
        self.assertEqual(f.feed(usage[:cut]), b"")  # only a partial line: held
        out = f.feed(usage[cut:] + b"data: [DONE]\n\n")
        events = parse_sse(out)
        self.assertEqual(events[-1], "[DONE]")
        timings = events[0]["timings"]
        self.assertEqual(set(timings), TIMINGS_KEYS)
        self.assertEqual(timings["prompt_n"], 372)
        self.assertEqual(timings["cache_n"], 128)
        self.assertEqual(timings["predicted_n"], 2)
        for key in ("prompt_ms", "prompt_per_second", "predicted_ms", "predicted_per_second"):
            self.assertIsInstance(timings[key], float)
        self.assertEqual(events[0]["usage"]["prompt_tokens"], 500)
        self.assertEqual(f.flush(), b"")

    def test_filter_leaves_usage_chunk_alone_when_data_is_missing(self) -> None:
        t = od.StreamTracker("chat", "m")
        f = od.TimingsSseFilter(t)
        chunk = sse({"id": "c1", "choices": [], "usage": {"total_tokens": 3}})
        self.assertEqual(f.feed(chunk), chunk)

    def test_filter_ignores_content_that_mentions_usage(self) -> None:
        t = od.StreamTracker("chat", "m")
        f = od.TimingsSseFilter(t)
        chunk = content_chunk('say "usage" and {"choices": []}')
        self.assertEqual(f.feed(chunk), chunk)


class ProxyTimingsTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self._saved = (od.ensure_started, od.state.http_client, od.DS4_MODEL_ALIAS)

        async def no_start() -> None:
            return None

        od.ensure_started = no_start
        od.DS4_MODEL_ALIAS = "swift1.5-qwen3.8-flash-next"

    def tearDown(self) -> None:
        od.ensure_started, od.state.http_client, od.DS4_MODEL_ALIAS = self._saved

    def _client(self, handler) -> TestClient:
        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return TestClient(od.app)

    def test_streaming_usage_chunk_gets_timings(self) -> None:
        async def body():
            yield content_chunk("Hel")
            yield content_chunk("lo")
            yield b'data: {"id":"c1","object":"chat.completion.chunk","model":"qwen3.8-flash-next-chat",'
            yield b'"choices":[],"usage":{"prompt_tokens":300,"completion_tokens":2,'
            yield b'"total_tokens":302,"prompt_tokens_details":{"cached_tokens":100}}}\n\n'
            yield b"data: [DONE]\n\n"

        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200, content=body(), headers={"content-type": "text/event-stream"}
            )

        client = self._client(handler)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "swift1.5-qwen3.8-flash-next-chat", "stream": True, "messages": []},
        )
        self.assertEqual(resp.status_code, 200)
        events = parse_sse(resp.content)
        self.assertEqual(events[-1], "[DONE]")
        usage_events = [e for e in events[:-1] if e["choices"] == []]
        self.assertEqual(len(usage_events), 1)
        timings = usage_events[0]["timings"]
        self.assertEqual(set(timings), TIMINGS_KEYS)
        self.assertEqual(timings["prompt_n"], 200)
        self.assertEqual(timings["cache_n"], 100)
        self.assertEqual(timings["predicted_n"], 2)
        # content events untouched, and the model id is still mapped back
        for e in events[:-1]:
            self.assertEqual(e["model"], "swift1.5-qwen3.8-flash-next-chat")
            if e["choices"]:
                self.assertNotIn("timings", e)
        # stats.last keeps its shape
        last = od.state.stats.last
        self.assertEqual(
            set(last),
            {
                "model",
                "prompt_tokens",
                "completion_tokens",
                "ttft_ms",
                "prefill_tps",
                "gen_tps",
                "duration_s",
                "finished_at",
            },
        )
        self.assertEqual(last["prompt_tokens"], 300)

    def test_completions_route_stream_gets_timings(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raw = (
                sse(
                    {
                        "id": "c",
                        "object": "text_completion",
                        "model": "m",
                        "choices": [{"text": "hi", "index": 0, "finish_reason": None}],
                    }
                )
                + usage_chunk(10, 1, 0, "m")
                + b"data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=raw, headers={"content-type": "text/event-stream"}
            )

        client = self._client(handler)
        resp = client.post("/v1/completions", json={"model": "m", "stream": True, "prompt": "x"})
        events = parse_sse(resp.content)
        self.assertEqual(events[1]["timings"]["prompt_n"], 10)
        self.assertEqual(events[1]["timings"]["cache_n"], 0)

    def test_stream_without_usage_gets_no_timings(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raw = content_chunk("a") + b"data: [DONE]\n\n"
            return httpx.Response(
                200, content=raw, headers={"content-type": "text/event-stream"}
            )

        client = self._client(handler)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.8-flash-next-chat", "stream": True, "messages": []},
        )
        self.assertNotIn(b"timings", resp.content)

    def test_non_streaming_body_gets_top_level_timings(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "object": "chat.completion",
                "model": "qwen3.8-flash-next-chat",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 0},
                },
            }
            return httpx.Response(200, json=body)

        client = self._client(handler)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "swift1.5-qwen3.8-flash-next-chat", "messages": []},
        )
        data = resp.json()
        self.assertEqual(data["model"], "swift1.5-qwen3.8-flash-next-chat")
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")
        self.assertEqual(set(data["timings"]), TIMINGS_KEYS)
        self.assertEqual(data["timings"]["prompt_n"], 40)
        self.assertEqual(data["timings"]["cache_n"], 10)
        self.assertEqual(data["timings"]["predicted_n"], 7)
        self.assertEqual(od.state.stats.last["completion_tokens"], 7)

    def test_non_streaming_error_is_untouched(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "bad"}})

        client = self._client(handler)
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json(), {"error": {"message": "bad"}})

    def test_non_streaming_without_usage_is_untouched(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "chat.completion", "choices": []})

        client = self._client(handler)
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        self.assertNotIn("timings", resp.json())


class CtxConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (od.DS4_STATE_PATH, od.__dict__["_spawn"])
        od.DS4_STATE_PATH = Path(self._tmp.name) / "sub" / "ds4-state.json"

        async def no_spawn():
            raise AssertionError("ds4-server must not be spawned by /admin/config")

        od._spawn = no_spawn
        self.client = TestClient(od.app)

    def tearDown(self) -> None:
        od.DS4_STATE_PATH, od._spawn = self._saved
        self._tmp.cleanup()
        reset_state()

    def _running(self, ctx: int = 65536) -> FakeProc:
        proc = FakeProc()
        od.state.process = proc
        od.state.ready = True
        od.state.ctx = ctx
        od.state.ctx_active = ctx
        return proc

    def test_normalize(self) -> None:
        self.assertEqual(od.normalize_ctx(65536), 65536)
        self.assertEqual(od.normalize_ctx(4096), 4096)
        self.assertEqual(od.normalize_ctx(262144), 262144)
        self.assertEqual(od.normalize_ctx(10000), 9984)  # 39.06 steps -> 39
        self.assertEqual(od.normalize_ctx(10112), 10240)  # 39.5 steps rounds up
        for bad in (4095, 262145, 0, -1, True, 8192.0, "8192", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                od.normalize_ctx(bad)

    def test_get_when_cold(self) -> None:
        resp = self.client.get("/admin/config")
        self.assertEqual(
            resp.json(),
            {
                "ctx": 65536,
                "ctx_active": None,
                "ctx_min": 4096,
                "ctx_max": 262144,
                "pending_restart": False,
            },
        )

    def test_post_when_stopped_persists_and_applies_next_start(self) -> None:
        resp = self.client.post("/admin/config", json={"ctx": 131072})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {
                "ctx": 131072,
                "ctx_active": None,
                "ctx_min": 4096,
                "ctx_max": 262144,
                "pending_restart": False,
                "applied": "next_start",
            },
        )
        self.assertEqual(json.loads(od.DS4_STATE_PATH.read_text())["ctx"], 131072)
        self.assertEqual(self.client.get("/admin/config").json()["ctx"], 131072)

    def test_rounding_is_persisted_and_returned(self) -> None:
        resp = self.client.post("/admin/config", json={"ctx": 10000})
        self.assertEqual(resp.json()["ctx"], 9984)
        self.assertEqual(json.loads(od.DS4_STATE_PATH.read_text())["ctx"], 9984)

    def test_invalid_values_rejected_and_not_persisted(self) -> None:
        for body in ({"ctx": 100}, {"ctx": 999999}, {"ctx": "big"}, {"ctx": True}, {}, []):
            with self.subTest(body=body):
                resp = self.client.post("/admin/config", json=body)
                self.assertEqual(resp.status_code, 400)
        self.assertFalse(od.DS4_STATE_PATH.exists())
        self.assertEqual(od.state.ctx, 65536)
        resp = self.client.post(
            "/admin/config", content=b"nope", headers={"content-type": "application/json"}
        )
        self.assertEqual(resp.status_code, 400)

    def test_persisted_value_overrides_env_at_startup(self) -> None:
        od.DS4_STATE_PATH.parent.mkdir(parents=True)
        od.DS4_STATE_PATH.write_text(json.dumps({"ctx": 32768}))
        od.state.ctx = od.parse_env_ctx("131072")
        od.apply_persisted_ctx()
        self.assertEqual(od.state.ctx, 32768)

    def test_bad_or_missing_state_file_keeps_env_value(self) -> None:
        od.state.ctx = 131072
        od.apply_persisted_ctx()  # no file
        self.assertEqual(od.state.ctx, 131072)
        od.DS4_STATE_PATH.parent.mkdir(parents=True)
        for text in ("not json", json.dumps({"ctx": 5}), json.dumps({"ctx": "x"})):
            od.DS4_STATE_PATH.write_text(text)
            od.apply_persisted_ctx()
            self.assertEqual(od.state.ctx, 131072)

    def test_env_ctx_parsing(self) -> None:
        self.assertEqual(od.parse_env_ctx("131072"), 131072)
        self.assertEqual(od.parse_env_ctx("abc"), od.CTX_DEFAULT)

    def test_running_and_idle_stops_gracefully_and_does_not_respawn(self) -> None:
        proc = self._running()
        resp = self.client.post("/admin/config", json={"ctx": 32768})
        body = resp.json()
        self.assertEqual(body["applied"], "restarted")
        self.assertEqual(body["ctx"], 32768)
        self.assertFalse(body["pending_restart"])
        self.assertIsNone(body["ctx_active"])
        self.assertTrue(proc.terminated)
        self.assertIsNone(od.state.process)
        self.assertFalse(od.state.ready)
        self.assertEqual(json.loads(od.DS4_STATE_PATH.read_text())["ctx"], 32768)

    def test_running_with_in_flight_defers_until_idle(self) -> None:
        proc = self._running()
        od.state.in_flight = 1
        resp = self.client.post("/admin/config", json={"ctx": 32768})
        body = resp.json()
        self.assertEqual(body["applied"], "after_current_requests")
        self.assertTrue(body["pending_restart"])
        self.assertEqual(body["ctx"], 32768)
        self.assertEqual(body["ctx_active"], 65536)  # still the running value
        self.assertFalse(proc.terminated)
        status = self.client.get("/admin/status").json()["config"]
        self.assertEqual(status["ctx"], 32768)
        self.assertTrue(status["pending_restart"])

        # still busy: nothing happens
        asyncio.run(od._apply_pending_restart())
        self.assertFalse(proc.terminated)
        self.assertTrue(od.state.pending_restart)

        # last request finishes
        od.state.in_flight = 0
        asyncio.run(od._apply_pending_restart())
        self.assertTrue(proc.terminated)
        self.assertFalse(od.state.pending_restart)
        self.assertIsNone(od.state.process)
        self.assertEqual(self.client.get("/admin/config").json()["pending_restart"], False)

    def test_in_flight_decrement_schedules_the_deferred_stop(self) -> None:
        proc = self._running()
        od.state.in_flight = 1
        od.state.pending_restart = True
        od.state.ctx = 32768

        async def scenario() -> None:
            od.state.in_flight -= 1
            od._kick_pending_restart()
            self.assertIsNotNone(od.state.pending_task)
            await od.state.pending_task

        asyncio.run(scenario())
        self.assertTrue(proc.terminated)
        self.assertFalse(od.state.pending_restart)

    def test_same_ctx_as_running_does_not_restart(self) -> None:
        proc = self._running(65536)
        body = self.client.post("/admin/config", json={"ctx": 65536}).json()
        self.assertEqual(body["applied"], "unchanged")
        self.assertFalse(body["pending_restart"])
        self.assertFalse(proc.terminated)

    def test_dead_process_counts_as_not_running(self) -> None:
        proc = self._running()
        proc.returncode = 1
        body = self.client.post("/admin/config", json={"ctx": 32768}).json()
        self.assertEqual(body["applied"], "next_start")
        self.assertIsNone(body["ctx_active"])

    def test_spawn_uses_current_ctx_and_records_active(self) -> None:
        captured: dict = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = list(args)
            return FakeProc()

        od.state.ctx = 32768
        saved = (od.asyncio.create_subprocess_exec, od.DS4_BINARY, od.DS4_LOG_PATH, od.LOGS_DIR)
        od.asyncio.create_subprocess_exec = fake_exec
        od.DS4_BINARY = Path(self._tmp.name) / "ds4-server"
        od.DS4_BINARY.write_text("")
        od.LOGS_DIR = Path(self._tmp.name) / "logs"
        od.DS4_LOG_PATH = od.LOGS_DIR / "ds4-server.log"
        try:
            od._spawn = self._saved[1]
            asyncio.run(od._spawn())
        finally:
            (
                od.asyncio.create_subprocess_exec,
                od.DS4_BINARY,
                od.DS4_LOG_PATH,
                od.LOGS_DIR,
            ) = saved
        args = captured["args"]
        self.assertEqual(args[args.index("--ctx") + 1], "32768")
        self.assertEqual(od.state.ctx_active, 32768)


if __name__ == "__main__":
    unittest.main()
