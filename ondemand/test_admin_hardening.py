"""Tests for the 2026-10-03 hardening round: loopback-only /admin, stop?if_idle,
the `starting` flag, holds, the oMLX re-poll, the ComfyUI free, graceful
shutdown and log handling.

Fakes only: ds4-server is never spawned, oMLX and ComfyUI are httpx
MockTransport handlers, and nothing listens on a real port. Run with:

    cd ~/Homelab/dwarfstar
    uv run --with fastapi --with uvicorn --with httpx \
        python -m unittest discover -s ondemand -p 'test_*.py'
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds4_ondemand as od  # noqa: E402

LOCAL = ("127.0.0.1", 50000)
LAN = ("192.168.2.77", 50000)


class FakeProc:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.terminated = False
        self.stdout = None

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
    s.starting_count = 0
    s.holds = {}
    s.stopping = False


class Patched(unittest.TestCase):
    """Replaces the module attributes named in self.patches for one test."""

    patches: dict = {}

    def setUp(self) -> None:
        reset_state()
        # Holds are persisted: never let a test touch the live state file.
        self._state_tmp = tempfile.TemporaryDirectory()
        self._saved_state_path = od.DS4_STATE_PATH
        od.DS4_STATE_PATH = Path(self._state_tmp.name) / "ds4-state.json"
        self._saved_attrs = {k: getattr(od, k) for k in self.patches}
        for k, v in self.patches.items():
            setattr(od, k, v)
        self._saved_client = od.state.http_client

    def tearDown(self) -> None:
        for k, v in self._saved_attrs.items():
            setattr(od, k, v)
        od.state.http_client = self._saved_client
        od.DS4_STATE_PATH = self._saved_state_path
        self._state_tmp.cleanup()
        reset_state()


def install_pipeline(spawned: list, *, ready: bool = True) -> None:
    """Stub the parts of a cold start that touch the machine."""

    async def noop() -> None:
        return None

    async def spawn() -> FakeProc:
        proc = FakeProc()
        spawned.append(proc)
        od.state.ctx_active = od.state.ctx
        return proc

    async def wait_ready(timeout: float):
        return (True, None) if ready else (False, "timed out")

    od._unload_omlx_models = noop
    od._free_comfyui = noop
    od._log_free_memory = lambda: None
    od._spawn = spawn
    od._wait_ready = wait_ready


PIPELINE_ATTRS = (
    "_unload_omlx_models",
    "_free_comfyui",
    "_log_free_memory",
    "_spawn",
    "_wait_ready",
)


class LoopbackAdminTest(Patched):
    def test_admin_is_403_from_lan(self) -> None:
        client = TestClient(od.app, client=LAN)
        calls = [
            ("GET", "/admin/status", None),
            ("GET", "/admin/config", None),
            ("POST", "/admin/config", {"ctx": 8192}),
            ("POST", "/admin/stop", None),
            ("POST", "/admin/start", None),
            ("POST", "/admin/hold", {"reason": "x", "ttl_s": 5}),
            ("DELETE", "/admin/hold/abc", None),
        ]
        for method, path, body in calls:
            with self.subTest(call=f"{method} {path}"):
                resp = client.request(method, path, json=body)
                self.assertEqual(resp.status_code, 403)
        self.assertEqual(od.state.holds, {})
        self.assertIsNone(od.state.process)

    def test_admin_allowed_from_loopback(self) -> None:
        for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            with self.subTest(host=host):
                client = TestClient(od.app, client=(host, 1234))
                self.assertEqual(client.get("/admin/status").status_code, 200)

    def test_hostname_and_garbage_peers_are_not_loopback(self) -> None:
        for host in ("testclient", "localhost", "", None, "10.0.0.1", "::ffff:10.0.0.1"):
            with self.subTest(host=host):
                self.assertFalse(od.is_loopback_host(host))

    def test_v1_models_open_to_lan(self) -> None:
        client = TestClient(od.app, client=LAN)
        self.assertEqual(client.get("/v1/models").status_code, 200)

    def test_no_peer_address_is_refused(self) -> None:
        sent: list[dict] = []

        async def inner(scope, receive, send):
            raise AssertionError("must not reach the app")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        mw = od.LoopbackOnlyAdminMiddleware(inner)
        scope = {"type": "http", "path": "/admin/status", "method": "GET", "headers": []}
        asyncio.run(mw(scope, receive, send))
        self.assertEqual(sent[0]["status"], 403)

    def test_forwarded_header_is_not_trusted(self) -> None:
        client = TestClient(od.app, client=LAN)
        resp = client.get("/admin/status", headers={"X-Forwarded-For": "127.0.0.1"})
        self.assertEqual(resp.status_code, 403)


class StopIfIdleTest(Patched):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(od.app, client=LOCAL)
        self.proc = FakeProc()
        od.state.process = self.proc
        od.state.ready = True

    def test_409_when_request_in_flight(self) -> None:
        od.state.in_flight = 1
        resp = self.client.post("/admin/stop?if_idle=1")
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "busy")
        self.assertEqual(resp.json()["in_flight"], 1)
        self.assertFalse(self.proc.terminated)
        self.assertTrue(od.state.ready)

    def test_409_when_start_in_progress(self) -> None:
        od.state.starting_count = 1
        resp = self.client.post("/admin/stop?if_idle=1")
        self.assertEqual(resp.status_code, 409)
        self.assertTrue(resp.json()["starting"])
        self.assertFalse(self.proc.terminated)

    def test_idle_stops(self) -> None:
        resp = self.client.post("/admin/stop?if_idle=1")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["loaded"])
        self.assertTrue(self.proc.terminated)

    def test_without_flag_behaviour_unchanged(self) -> None:
        od.state.in_flight = 3
        resp = self.client.post("/admin/stop")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok", "loaded": False})
        self.assertTrue(self.proc.terminated)

    def test_if_idle_zero_is_the_unconditional_stop(self) -> None:
        od.state.in_flight = 1
        self.assertEqual(self.client.post("/admin/stop?if_idle=0").status_code, 200)


class StartingFlagTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_state()
        self._saved = {k: getattr(od, k) for k in PIPELINE_ATTRS}
        self.spawned: list[FakeProc] = []
        install_pipeline(self.spawned)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(od, k, v)
        reset_state()

    async def test_starting_is_set_before_the_first_await(self) -> None:
        await od.state.lock.acquire()  # a stop or another start holds the lock
        task = asyncio.create_task(od.ensure_started())
        await asyncio.sleep(0)
        status = await od.admin_status()
        self.assertTrue(status["starting"])
        self.assertFalse(status["loaded"])
        self.assertIsNone(status["pid"])
        od.state.lock.release()
        await task
        status = await od.admin_status()
        self.assertFalse(status["starting"])
        self.assertTrue(status["loaded"])
        self.assertEqual(status["pid"], 4242)

    async def test_starting_visible_during_omlx_unload(self) -> None:
        gate = asyncio.Event()

        async def slow_unload() -> None:
            await gate.wait()

        od._unload_omlx_models = slow_unload
        task = asyncio.create_task(od.ensure_started())
        await asyncio.sleep(0.01)
        status = await od.admin_status()
        self.assertTrue(status["starting"])
        self.assertIsNone(status["pid"])
        self.assertEqual(self.spawned, [])
        gate.set()
        await task
        self.assertFalse((await od.admin_status())["starting"])

    async def test_starting_cleared_when_start_fails(self) -> None:
        install_pipeline(self.spawned, ready=False)
        with self.assertRaises(od.ServerStartError):
            await od.ensure_started()
        self.assertFalse(od.state.starting)
        self.assertEqual(od.state.starting_count, 0)

    async def test_concurrent_cold_requests_spawn_once(self) -> None:
        await asyncio.gather(*(od.ensure_started() for _ in range(4)))
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(od.state.starting_count, 0)

    async def test_starting_cleared_when_refused_by_hold(self) -> None:
        od.state.holds["h"] = {"reason": "training", "expires": od.time.monotonic() + 60}
        with self.assertRaises(od.ServerHeldError):
            await od.ensure_started()
        self.assertEqual(od.state.starting_count, 0)


class HoldTest(Patched):
    patches = {k: None for k in PIPELINE_ATTRS}

    def setUp(self) -> None:
        super().setUp()
        self.spawned: list[FakeProc] = []
        install_pipeline(self.spawned)
        self.client = TestClient(od.app, client=LOCAL)

    def _hold(self, reason="training", ttl=60) -> str:
        resp = self.client.post("/admin/hold", json={"reason": reason, "ttl_s": ttl})
        self.assertEqual(resp.status_code, 200)
        return resp.json()["hold_id"]

    def test_create_returns_hold_id_and_status_lists_it(self) -> None:
        hold_id = self._hold("studio training", 120)
        holds = self.client.get("/admin/status").json()["holds"]
        self.assertEqual(len(holds), 1)
        self.assertEqual(holds[0]["id"], hold_id)
        self.assertEqual(holds[0]["reason"], "studio training")
        self.assertTrue(0 < holds[0]["ttl_remaining_s"] <= 120)

    def test_held_cold_start_is_503_with_retry_after(self) -> None:
        self._hold("studio training", 120)
        resp = self.client.post("/v1/chat/completions", json={"model": "x", "messages": []})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json(), {"error": "held", "reason": "studio training"})
        self.assertGreaterEqual(int(resp.headers["Retry-After"]), 1)
        self.assertEqual(self.spawned, [])
        self.assertEqual(od.state.in_flight, 0)

    def test_admin_start_is_refused_while_held(self) -> None:
        self._hold("local model resident", 30)
        resp = self.client.post("/admin/start")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"], "held")
        self.assertIn("Retry-After", resp.headers)
        self.assertEqual(self.spawned, [])

    def test_retry_after_is_capped(self) -> None:
        self._hold("long", 3600)
        resp = self.client.post("/admin/start")
        self.assertEqual(resp.headers["Retry-After"], str(od.HOLD_RETRY_AFTER_MAX))

    def test_delete_releases_and_start_proceeds(self) -> None:
        hold_id = self._hold()
        resp = self.client.delete(f"/admin/hold/{hold_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get("/admin/status").json()["holds"], [])
        self.assertEqual(self.client.post("/admin/start").status_code, 200)
        self.assertEqual(len(self.spawned), 1)

    def test_delete_unknown_is_404(self) -> None:
        self.assertEqual(self.client.delete("/admin/hold/nope").status_code, 404)

    def test_holds_expire_by_ttl(self) -> None:
        hold_id = self._hold(ttl=60)
        od.state.holds[hold_id]["expires"] = od.time.monotonic() - 1
        self.assertEqual(self.client.get("/admin/status").json()["holds"], [])
        self.assertEqual(self.client.post("/admin/start").status_code, 200)
        self.assertEqual(self.client.delete(f"/admin/hold/{hold_id}").status_code, 404)

    def test_any_unexpired_hold_blocks(self) -> None:
        a = self._hold("a", 60)
        self._hold("b", 60)
        self.client.delete(f"/admin/hold/{a}")
        self.assertEqual(self.client.post("/admin/start").status_code, 503)

    def test_status_carries_the_fields_oMLX_peer_evict_reads(self) -> None:
        cold = self.client.get("/admin/status").json()
        for key in ("loaded", "in_flight", "uptime_seconds", "starting"):
            self.assertIn(key, cold)
        self.assertIsNone(cold["uptime_seconds"])
        od.state.process = FakeProc()
        od.state.ready = True
        od.state.start_time = od.time.monotonic() - 42
        warm = self.client.get("/admin/status").json()
        self.assertTrue(warm["loaded"])
        self.assertGreaterEqual(warm["uptime_seconds"], 42)

    def test_hold_does_not_stop_a_running_server(self) -> None:
        proc = FakeProc()
        od.state.process = proc
        od.state.ready = True
        self._hold()
        asyncio.run(od.ensure_started())
        self.assertFalse(proc.terminated)
        self.assertEqual(self.spawned, [])

    def test_validation(self) -> None:
        bad = [
            None,
            [],
            {},
            {"reason": "x"},
            {"ttl_s": 5},
            {"reason": 5, "ttl_s": 5},
            {"reason": "x", "ttl_s": "5"},
            {"reason": "x", "ttl_s": True},
            {"reason": "x", "ttl_s": 5.5},
            {"reason": "x", "ttl_s": 0},
            {"reason": "x", "ttl_s": od.HOLD_TTL_MAX + 1},
        ]
        for body in bad:
            with self.subTest(body=body):
                resp = self.client.post("/admin/hold", json=body)
                self.assertEqual(resp.status_code, 400)
        self.assertEqual(od.state.holds, {})

    def test_non_json_body_is_400(self) -> None:
        resp = self.client.post("/admin/hold", content=b"nope")
        self.assertEqual(resp.status_code, 400)

    def test_hold_taken_while_waiting_for_the_lock_is_honoured(self) -> None:
        async def scenario() -> None:
            await od.state.lock.acquire()
            task = asyncio.create_task(od.ensure_started())
            await asyncio.sleep(0)
            od.state.holds["h"] = {"reason": "late", "expires": od.time.monotonic() + 60}
            od.state.lock.release()
            with self.assertRaises(od.ServerHeldError):
                await task

        asyncio.run(scenario())
        self.assertEqual(self.spawned, [])


class HoldPersistenceTest(Patched):
    """Holds survive a launcher restart: id, reason and an absolute expiry live
    in ds4-state.json, next to the persisted ctx."""

    patches = {k: None for k in PIPELINE_ATTRS}

    def setUp(self) -> None:
        super().setUp()
        self.spawned: list[FakeProc] = []
        install_pipeline(self.spawned)
        self.client = TestClient(od.app, client=LOCAL)

    def _hold(self, reason="studio training", ttl=120) -> str:
        resp = self.client.post("/admin/hold", json={"reason": reason, "ttl_s": ttl})
        self.assertEqual(resp.status_code, 200)
        return resp.json()["hold_id"]

    def _file(self) -> dict:
        return json.loads(od.DS4_STATE_PATH.read_text())

    def _restart(self) -> None:
        """What a launcher restart does to the hold table."""
        od.state.holds = {}
        od.apply_persisted_holds()

    def test_create_persists_id_reason_and_absolute_expiry(self) -> None:
        before = od.time.time()
        hold_id = self._hold("studio training", 120)
        after = od.time.time()
        [entry] = self._file()["holds"]
        self.assertEqual(entry["id"], hold_id)
        self.assertEqual(entry["reason"], "studio training")
        self.assertTrue(before + 120 <= entry["expires_at"] <= after + 120)
        self.assertFalse(od.DS4_STATE_PATH.with_name("ds4-state.json.tmp").exists())

    def test_delete_removes_the_hold_from_the_file(self) -> None:
        a = self._hold("a")
        b = self._hold("b")
        self.assertEqual(self.client.delete(f"/admin/hold/{a}").status_code, 200)
        self.assertEqual([h["id"] for h in self._file()["holds"]], [b])
        self.client.delete(f"/admin/hold/{b}")
        self.assertEqual(self._file()["holds"], [])
        self._restart()
        self.assertEqual(od.state.holds, {})

    def test_restart_keeps_the_barrier_and_the_id(self) -> None:
        hold_id = self._hold("studio training", 120)
        self._restart()
        self.assertIn(hold_id, od.state.holds)
        status = self.client.get("/admin/status").json()
        self.assertEqual([h["id"] for h in status["holds"]], [hold_id])
        self.assertTrue(100 < status["holds"][0]["ttl_remaining_s"] <= 120)

        async def attempt() -> None:
            with self.assertRaises(od.ServerHeldError) as ctx:
                await od.ensure_started()
            self.assertEqual(ctx.exception.reason, "studio training")

        asyncio.run(attempt())
        self.assertEqual(self.spawned, [])
        # The old owner can still release it by id after the restart.
        self.assertEqual(self.client.delete(f"/admin/hold/{hold_id}").status_code, 200)
        self.assertEqual(self._file()["holds"], [])

    def test_restart_does_not_extend_the_ttl(self) -> None:
        od.DS4_STATE_PATH.write_text(
            json.dumps(
                {
                    "holds": [
                        {"id": "live", "reason": "r", "expires_at": od.time.time() + 30},
                        {"id": "gone", "reason": "r", "expires_at": od.time.time() - 1},
                    ]
                }
            )
        )
        od.apply_persisted_holds()
        self.assertEqual(list(od.state.holds), ["live"])
        remaining = od.state.holds["live"]["expires"] - od.time.monotonic()
        self.assertTrue(25 < remaining <= 30)

    def test_downtime_counts_against_the_ttl(self) -> None:
        hold_id = self._hold(ttl=5)
        # The launcher was down past the expiry: nothing is reloaded.
        saved = self._file()
        saved["holds"][0]["expires_at"] = od.time.time() - 0.5
        od.DS4_STATE_PATH.write_text(json.dumps(saved))
        self._restart()
        self.assertNotIn(hold_id, od.state.holds)

    def test_ctx_and_holds_share_the_file(self) -> None:
        od.save_persisted_ctx(32768)
        hold_id = self._hold()
        data = self._file()
        self.assertEqual(data["ctx"], 32768)
        self.assertEqual([h["id"] for h in data["holds"]], [hold_id])
        od.save_persisted_ctx(65536)
        data = self._file()
        self.assertEqual(data["ctx"], 65536)
        self.assertEqual([h["id"] for h in data["holds"]], [hold_id])

    def test_malformed_state_is_ignored(self) -> None:
        for text in (
            "not json",
            json.dumps([1, 2]),
            json.dumps({"holds": "x"}),
            json.dumps({"holds": [None, 3, {"id": "a"}, {"id": 1, "reason": "r", "expires_at": 9e18}]}),
            json.dumps(
                {"holds": [{"id": "a", "reason": "r", "expires_at": True},
                           {"id": "b", "reason": "r", "expires_at": "soon"}]}
            ),
        ):
            with self.subTest(text=text):
                od.DS4_STATE_PATH.write_text(text)
                od.state.holds = {}
                od.apply_persisted_holds()
                self.assertEqual(od.state.holds, {})

    def test_absurd_expiry_is_capped_at_the_max_ttl(self) -> None:
        od.DS4_STATE_PATH.write_text(
            json.dumps({"holds": [{"id": "far", "reason": "r", "expires_at": od.time.time() + 10**9}]})
        )
        od.apply_persisted_holds()
        remaining = od.state.holds["far"]["expires"] - od.time.monotonic()
        self.assertTrue(remaining <= od.HOLD_TTL_MAX)

    def test_unwritable_state_does_not_fail_the_hold(self) -> None:
        od.DS4_STATE_PATH = Path(self._state_tmp.name) / "blocker" / "ds4-state.json"
        Path(self._state_tmp.name, "blocker").write_text("a file, not a directory")
        hold_id = self._hold()
        self.assertIn(hold_id, od.state.holds)

    def test_reload_does_not_replace_a_live_hold(self) -> None:
        hold_id = self._hold(ttl=120)
        deadline = od.state.holds[hold_id]["expires"]
        od.apply_persisted_holds()
        self.assertEqual(od.state.holds[hold_id]["expires"], deadline)


def omlx_transport(models_by_poll: list[list[dict]], posts: list[str]):
    """oMLX stand-in: /v1/models/status returns the next entry of
    models_by_poll (the last one repeats); unloads are recorded."""
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models/status":
            idx = min(polls["n"], len(models_by_poll) - 1)
            polls["n"] += 1
            return httpx.Response(200, json={"models": models_by_poll[idx]})
        if path.endswith("/unload"):
            posts.append(path)
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


class OmlxRepollTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_state()
        self._saved = (od.state.http_client, od.OMLX_UNLOAD_WAIT)
        od.OMLX_UNLOAD_WAIT = 0.4

    def tearDown(self) -> None:
        od.state.http_client, od.OMLX_UNLOAD_WAIT = self._saved
        reset_state()

    def _client(self, models_by_poll, posts) -> None:
        od.state.http_client = httpx.AsyncClient(
            transport=omlx_transport(models_by_poll, posts)
        )

    async def test_clean_unload_passes(self) -> None:
        posts: list[str] = []
        loaded = [{"id": "m1", "loaded": True, "is_loading": False}]
        clear = [{"id": "m1", "loaded": False, "is_loading": False}]
        self._client([loaded, clear], posts)
        await od._unload_omlx_models()
        self.assertEqual(posts, ["/v1/models/m1/unload"])

    async def test_202_and_409_unload_replies_rely_on_the_repoll(self) -> None:
        for status, body in ((202, {"status": "unloading"}), (409, {"detail": "loading"})):
            with self.subTest(status=status):
                polls = {"n": 0}

                def handler(request: httpx.Request, status=status, body=body, polls=polls):
                    if request.url.path == "/v1/models/status":
                        polls["n"] += 1
                        loaded = polls["n"] < 3
                        return httpx.Response(
                            200, json={"models": [{"id": "m1", "loaded": loaded}]}
                        )
                    return httpx.Response(status, json=body)

                od.state.http_client = httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                )
                await od._unload_omlx_models()  # drained during the re-poll
                self.assertGreaterEqual(polls["n"], 3)

    async def test_202_with_model_still_resident_refuses(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models/status":
                return httpx.Response(200, json={"models": [{"id": "m1", "loaded": True}]})
            return httpx.Response(202, json={"status": "unloading"})

        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with self.assertRaises(od.ServerStartError):
            await od._unload_omlx_models()

    async def test_nothing_loaded_does_not_post(self) -> None:
        posts: list[str] = []
        self._client([[{"id": "m1", "loaded": False}]], posts)
        await od._unload_omlx_models()
        self.assertEqual(posts, [])

    async def test_still_loaded_after_wait_refuses(self) -> None:
        posts: list[str] = []
        self._client([[{"id": "m1", "loaded": True}]], posts)
        with self.assertRaises(od.ServerStartError) as ctx:
            await od._unload_omlx_models()
        self.assertEqual(ctx.exception.retry_after, od.START_RETRY_AFTER)
        self.assertIn("m1", str(ctx.exception))

    async def test_still_loading_refuses(self) -> None:
        posts: list[str] = []
        loaded = [{"id": "m1", "loaded": True}]
        loading = [{"id": "m1", "loaded": False, "is_loading": True}]
        self._client([loaded, loading], posts)
        with self.assertRaises(od.ServerStartError):
            await od._unload_omlx_models()

    async def test_unreachable_omlx_is_tolerated(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
        await od._unload_omlx_models()

    async def test_status_lost_during_poll_counts_as_clear(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models/status":
                calls["n"] += 1
                if calls["n"] == 1:
                    return httpx.Response(200, json={"models": [{"id": "m1", "loaded": True}]})
                raise httpx.ConnectError("oMLX exited")
            return httpx.Response(200, json={})

        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await od._unload_omlx_models()

    async def test_cold_start_does_not_spawn_when_omlx_stays_loaded(self) -> None:
        posts: list[str] = []
        self._client([[{"id": "m1", "loaded": True}]], posts)
        spawned: list[FakeProc] = []
        saved = {k: getattr(od, k) for k in PIPELINE_ATTRS}
        install_pipeline(spawned)
        od._unload_omlx_models = saved["_unload_omlx_models"]
        try:
            with self.assertRaises(od.ServerStartError):
                await od.ensure_started()
        finally:
            for k, v in saved.items():
                setattr(od, k, v)
        self.assertEqual(spawned, [])
        self.assertEqual(od.state.starting_count, 0)

    def test_proxy_maps_refusal_to_503_retry_after(self) -> None:
        async def refuse() -> None:
            raise od.ServerStartError("omlx busy", retry_after=15)

        saved = od.ensure_started
        od.ensure_started = refuse
        try:
            resp = TestClient(od.app, client=LAN).post(
                "/v1/chat/completions", json={"model": "x", "messages": []}
            )
        finally:
            od.ensure_started = saved
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.headers["Retry-After"], "15")
        self.assertEqual(od.state.in_flight, 0)


class ComfyUiFreeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_state()
        self._saved = (od.state.http_client, od.DS4_COMFYUI_URL)
        od.DS4_COMFYUI_URL = "http://comfy.invalid:8188"
        self.requests: list[httpx.Request] = []

    def tearDown(self) -> None:
        od.state.http_client, od.DS4_COMFYUI_URL = self._saved
        reset_state()

    def _client(self, queue: dict | None, free_status: int = 200, queue_exc=None) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path == "/queue":
                if queue_exc is not None:
                    raise queue_exc
                return httpx.Response(200, json=queue)
            if request.url.path == "/free":
                return httpx.Response(free_status, json={})
            return httpx.Response(404)

        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def _paths(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]

    async def test_empty_queue_frees(self) -> None:
        self._client({"queue_running": [], "queue_pending": []})
        await od._free_comfyui()
        self.assertEqual(self._paths(), [("GET", "/queue"), ("POST", "/free")])
        self.assertEqual(
            json.loads(self.requests[1].content),
            {"unload_models": True, "free_memory": True},
        )

    async def test_running_job_leaves_comfyui_alone(self) -> None:
        self._client({"queue_running": [[0, "id"]], "queue_pending": []})
        await od._free_comfyui()
        self.assertEqual(self._paths(), [("GET", "/queue")])

    async def test_pending_job_leaves_comfyui_alone(self) -> None:
        self._client({"queue_running": [], "queue_pending": [[1, "id"]]})
        await od._free_comfyui()
        self.assertEqual(self._paths(), [("GET", "/queue")])

    async def test_empty_url_disables(self) -> None:
        od.DS4_COMFYUI_URL = ""
        self._client({"queue_running": [], "queue_pending": []})
        await od._free_comfyui()
        self.assertEqual(self.requests, [])

    async def test_uses_three_second_timeout(self) -> None:
        self._client({"queue_running": [], "queue_pending": []})
        await od._free_comfyui()
        for req in self.requests:
            self.assertEqual(req.extensions["timeout"]["read"], 3.0)
            self.assertEqual(req.extensions["timeout"]["connect"], 3.0)

    async def test_trailing_slash_in_url(self) -> None:
        od.DS4_COMFYUI_URL = "http://comfy.invalid:8188/"
        self._client({"queue_running": [], "queue_pending": []})
        await od._free_comfyui()
        self.assertEqual(self._paths(), [("GET", "/queue"), ("POST", "/free")])

    async def test_failures_never_raise(self) -> None:
        for name, kwargs in {
            "unreachable": {"queue": None, "queue_exc": httpx.ConnectError("down")},
            "timeout": {"queue": None, "queue_exc": httpx.ReadTimeout("slow")},
            "free 500": {"queue": {"queue_running": [], "queue_pending": []}, "free_status": 500},
            "bad queue json": {"queue": ["not", "a", "dict"]},
        }.items():
            with self.subTest(case=name):
                self._client(**kwargs)
                await od._free_comfyui()

    async def test_cold_start_survives_comfyui_down_and_orders_calls(self) -> None:
        order: list[str] = []
        self._client(None, queue_exc=httpx.ConnectError("down"))
        saved = {k: getattr(od, k) for k in PIPELINE_ATTRS}
        spawned: list[FakeProc] = []
        install_pipeline(spawned)

        async def unload() -> None:
            order.append("omlx")

        real_free = saved["_free_comfyui"]

        async def free() -> None:
            order.append("comfyui")
            await real_free()

        async def spawn() -> FakeProc:
            order.append("spawn")
            proc = FakeProc()
            spawned.append(proc)
            return proc

        od._unload_omlx_models, od._free_comfyui, od._spawn = unload, free, spawn
        try:
            await od.ensure_started()
        finally:
            for k, v in saved.items():
                setattr(od, k, v)
        self.assertEqual(order, ["omlx", "comfyui", "spawn"])
        self.assertTrue(od.state.ready)


class FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class ShutdownTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_state()
        self._saved = {k: getattr(od, k) for k in PIPELINE_ATTRS}
        self._timeout = od.SHUTDOWN_LOCK_TIMEOUT
        od.SHUTDOWN_LOCK_TIMEOUT = 0.05

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(od, k, v)
        od.SHUTDOWN_LOCK_TIMEOUT = self._timeout
        reset_state()

    async def test_stop_when_lock_is_free(self) -> None:
        proc = FakeProc()
        od.state.process, od.state.ready = proc, True
        await od.shutdown_stop_ds4()
        self.assertTrue(proc.terminated)
        self.assertIsNone(od.state.process)
        self.assertFalse(od.state.lock.locked())
        self.assertTrue(od.state.stopping)

    async def test_stop_does_not_block_on_a_lock_held_by_a_cold_start(self) -> None:
        proc = FakeProc()
        od.state.process = proc
        await od.state.lock.acquire()  # a cold start sits in _wait_ready
        await asyncio.wait_for(od.shutdown_stop_ds4(), timeout=2)
        self.assertTrue(proc.terminated)
        self.assertTrue(od.state.lock.locked())  # still the starter's

    async def test_stop_with_blocked_lock_and_no_child_yet(self) -> None:
        await od.state.lock.acquire()
        await asyncio.wait_for(od.shutdown_stop_ds4(), timeout=2)

    async def test_cold_start_in_oMLX_unload_aborts_instead_of_spawning(self) -> None:
        spawned: list[FakeProc] = []
        install_pipeline(spawned)
        gate = asyncio.Event()

        async def slow_unload() -> None:
            await gate.wait()

        od._unload_omlx_models = slow_unload
        task = asyncio.create_task(od.ensure_started())
        await asyncio.sleep(0.01)
        await asyncio.wait_for(od.shutdown_stop_ds4(), timeout=2)  # lock timeout path
        gate.set()
        with self.assertRaises(od.ServerStartError):
            await task
        self.assertEqual(spawned, [])
        self.assertEqual(od.state.starting_count, 0)

    async def test_wait_ready_gives_up_when_stopping(self) -> None:
        od.state.stopping = True
        ok, err = await od._wait_ready(30)
        self.assertFalse(ok)
        self.assertIn("shutting down", err)

    async def test_no_new_cold_start_while_stopping(self) -> None:
        spawned: list[FakeProc] = []
        install_pipeline(spawned)
        od.state.stopping = True
        with self.assertRaises(od.ServerStartError) as ctx:
            await od.ensure_started()
        self.assertEqual(ctx.exception.retry_after, od.START_RETRY_AFTER)
        self.assertEqual(spawned, [])


class StreamEndsOnStopTest(Patched):
    patches = {"ensure_started": None}

    def setUp(self) -> None:
        super().setUp()

        async def no_start() -> None:
            return None

        od.ensure_started = no_start

    def test_sse_stream_ends_promptly_when_stopping(self) -> None:
        async def upstream_body():
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            od.state.stopping = True
            yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"c"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=upstream_body(),
            )

        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        resp = TestClient(od.app, client=LAN).post(
            "/v1/chat/completions", json={"model": "x", "messages": [], "stream": True}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'"a"', resp.content)
        self.assertNotIn(b'"c"', resp.content)
        self.assertNotIn(b"[DONE]", resp.content)
        self.assertEqual(od.state.in_flight, 0)

    def test_stream_untouched_when_not_stopping(self) -> None:
        async def upstream_body():
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=upstream_body(),
            )

        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        resp = TestClient(od.app, client=LAN).post(
            "/v1/chat/completions", json={"model": "x", "messages": [], "stream": True}
        )
        self.assertIn(b"[DONE]", resp.content)


class ServerBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self._filters = list(logging.getLogger("uvicorn.access").filters)

    def tearDown(self) -> None:
        logger = logging.getLogger("uvicorn.access")
        for f in list(logger.filters):
            if f not in self._filters:
                logger.removeFilter(f)
        reset_state()

    def test_config(self) -> None:
        server = od.build_server("127.0.0.1", 0)
        self.assertEqual(server.config.timeout_graceful_shutdown, 90)
        self.assertFalse(server.config.proxy_headers)
        self.assertEqual(server.config.host, "127.0.0.1")

    def test_sigterm_sets_stop_flag_and_exit(self) -> None:
        server = od.build_server("127.0.0.1", 0)
        self.assertFalse(od.state.stopping)
        server.handle_exit(signal.SIGTERM, None)
        self.assertTrue(od.state.stopping)
        self.assertTrue(server.should_exit)

    def test_access_filter_is_installed_once(self) -> None:
        od.build_server("127.0.0.1", 0)
        od.build_server("127.0.0.1", 0)
        filters = [
            f for f in logging.getLogger("uvicorn.access").filters
            if isinstance(f, od.AccessLogFilter)
        ]
        self.assertEqual(len(filters), 1)


def access_record(method: str, path: str, status: int = 200) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", method, path, "1.1", status),
        None,
    )


class AccessLogFilterTest(unittest.TestCase):
    def test_polled_gets_are_dropped(self) -> None:
        f = od.AccessLogFilter()
        for path in ("/admin/status", "/v1/models", "/admin/config", "/v1/models?x=1"):
            with self.subTest(path=path):
                self.assertFalse(f.filter(access_record("GET", path)))

    def test_everything_else_is_kept(self) -> None:
        f = od.AccessLogFilter()
        kept = [
            ("POST", "/admin/config"),
            ("POST", "/admin/stop"),
            ("POST", "/admin/hold"),
            ("GET", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/models"),
            ("GET", "/v1/models/extra"),
        ]
        for method, path in kept:
            with self.subTest(call=f"{method} {path}"):
                self.assertTrue(f.filter(access_record(method, path)))

    def test_odd_records_are_kept(self) -> None:
        f = od.AccessLogFilter()
        rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 0, "plain", None, None)
        self.assertTrue(f.filter(rec))


class RotatingLogTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_state()
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (od.LOGS_DIR, od.DS4_LOG_PATH, od.DS4_LOG_MAX_BYTES, od._ds4_handler)
        od._ds4_handler = None
        od.LOGS_DIR = Path(self._tmp.name) / "logs"
        od.DS4_LOG_PATH = od.LOGS_DIR / "ds4-server.log"

    def tearDown(self) -> None:
        if od._ds4_handler is not None:
            od._ds4_logger.removeHandler(od._ds4_handler)
            od._ds4_handler.close()
        od.LOGS_DIR, od.DS4_LOG_PATH, od.DS4_LOG_MAX_BYTES, od._ds4_handler = self._saved
        self._tmp.cleanup()
        reset_state()

    def test_defaults_are_10mb_by_3(self) -> None:
        self.assertEqual(od.DS4_LOG_MAX_BYTES, 10 * 1024 * 1024)
        self.assertEqual(od.DS4_LOG_BACKUPS, 3)

    def test_handler_is_rotating_with_configured_limits(self) -> None:
        od._ds4_output_logger()
        h = od._ds4_handler
        self.assertEqual(h.maxBytes, 10 * 1024 * 1024)
        self.assertEqual(h.backupCount, 3)
        self.assertFalse(od._ds4_logger.propagate)

    def test_pump_writes_and_rotates_keeping_three_backups(self) -> None:
        od.DS4_LOG_MAX_BYTES = 200
        lines = [f"line {i:04d} {'x' * 40}\n".encode() for i in range(60)]
        asyncio.run(od._pump_output(FakeStream(lines)))
        od._ds4_handler.flush()
        names = sorted(p.name for p in od.LOGS_DIR.iterdir())
        self.assertEqual(
            names,
            ["ds4-server.log", "ds4-server.log.1", "ds4-server.log.2", "ds4-server.log.3"],
        )
        current = (od.LOGS_DIR / "ds4-server.log").read_text()
        self.assertIn("line 0059", current)
        self.assertNotIn("\n\n", current)
        self.assertLessEqual(len(current.encode()), 200 + 100)

    def test_pump_keeps_draining_after_an_overlong_line(self) -> None:
        class Stream(FakeStream):
            def __init__(self) -> None:
                super().__init__([b"before\n", b"after\n", b"last\n"])
                self.raised = False

            async def readline(self) -> bytes:
                if len(self._lines) == 2 and not self.raised:
                    self.raised = True
                    raise ValueError("Separator is not found, and chunk exceed the limit")
                return await super().readline()

        stream = Stream()
        asyncio.run(od._pump_output(stream))
        od._ds4_handler.flush()
        self.assertTrue(stream.raised)
        self.assertEqual(stream._lines, [])
        self.assertEqual(od._tail_log(10), "before\nafter\nlast")

    def test_pump_survives_a_failing_log_write(self) -> None:
        out = od._ds4_output_logger()
        calls = {"n": 0}
        real = out.info

        def flaky(msg):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk full")
            real(msg)

        out.info = flaky
        try:
            asyncio.run(od._pump_output(FakeStream([b"one\n", b"two\n"])))
        finally:
            del out.info
        od._ds4_handler.flush()
        self.assertEqual(od._tail_log(10), "two")

    def test_tail_log_reads_pumped_output(self) -> None:
        asyncio.run(od._pump_output(FakeStream([b"hello\n", b"world\r\n"])))
        od._ds4_handler.flush()
        self.assertEqual(od._tail_log(10), "hello\nworld")

    def test_spawn_pipes_output_into_rotating_log(self) -> None:
        captured: dict = {}

        class Proc(FakeProc):
            def __init__(self) -> None:
                super().__init__()
                self.stdout = FakeStream([b"loading model\n", b"ready\n"])

        async def fake_exec(*args, **kwargs):
            captured["kwargs"] = kwargs
            return Proc()

        saved = (od.asyncio.create_subprocess_exec, od.DS4_BINARY)
        od.asyncio.create_subprocess_exec = fake_exec
        od.DS4_BINARY = Path(self._tmp.name) / "ds4-server"
        od.DS4_BINARY.write_text("")

        async def run() -> None:
            await od._spawn()
            await od.state.log_task

        try:
            asyncio.run(run())
        finally:
            od.asyncio.create_subprocess_exec, od.DS4_BINARY = saved
        od._ds4_handler.flush()
        self.assertEqual(captured["kwargs"]["stdout"], od.asyncio.subprocess.PIPE)
        self.assertEqual(captured["kwargs"]["stderr"], od.asyncio.subprocess.STDOUT)
        self.assertEqual(captured["kwargs"]["limit"], od.DS4_PIPE_LINE_LIMIT)
        text = od.DS4_LOG_PATH.read_text()
        self.assertIn("ds4-ondemand spawn", text)
        self.assertIn("loading model", text)
        self.assertIn("ready", text)


if __name__ == "__main__":
    unittest.main()
