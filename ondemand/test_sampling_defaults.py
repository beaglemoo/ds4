"""Unit tests for the proxy's Qwen sampling-default injection.

Importing ds4_ondemand builds the FastAPI app but starts nothing; uvicorn only
runs under __main__. The launcher runs under `uv run`, so run these with the
same interpreter and dependency set:

    cd ~/Homelab/dwarfstar
    uv run --with fastapi --with uvicorn --with httpx \
        python -m unittest ondemand/test_sampling_defaults.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds4_ondemand as od  # noqa: E402


class SamplingDefaultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._inject = od.DS4_SAMPLING_INJECT
        self._sets = od.SAMPLING_SETS
        od.DS4_SAMPLING_INJECT = True
        od.SAMPLING_SETS = od.load_sampling_sets(None)

    def tearDown(self) -> None:
        od.DS4_SAMPLING_INJECT = self._inject
        od.SAMPLING_SETS = self._sets

    def test_chat_alias_gets_nothink(self) -> None:
        data = {"model": "qwen3.8-flash-next-chat", "messages": []}
        injected = od.apply_sampling_defaults(data, "chat")
        self.assertEqual(
            sorted(injected), ["presence_penalty", "temperature", "top_k", "top_p"]
        )
        self.assertEqual(data["temperature"], 0.7)
        self.assertEqual(data["top_p"], 0.8)
        self.assertEqual(data["top_k"], 20)
        self.assertEqual(data["presence_penalty"], 1.5)

    def test_chat_alias_is_case_insensitive(self) -> None:
        self.assertEqual(
            od.sampling_defaults_for("Qwen3.8-Flash-Next-CHAT"), od.SAMPLING_NOTHINK
        )

    def test_reasoner_and_bare_alias_get_think(self) -> None:
        for model in ("qwen3.8-flash-next-reasoner", "qwen3.8-flash-next", None):
            with self.subTest(model=model):
                data = {"model": model}
                od.apply_sampling_defaults(data, "chat")
                self.assertEqual(data["temperature"], 1.0)
                self.assertEqual(data["top_p"], 0.95)
                self.assertEqual(data["top_k"], 20)
                self.assertEqual(data["presence_penalty"], 0.0)

    def test_client_values_are_preserved(self) -> None:
        data = {
            "model": "qwen3.8-flash-next-chat",
            "temperature": 0,
            "top_p": None,
            "presence_penalty": 0,
        }
        injected = od.apply_sampling_defaults(data, "messages")
        self.assertEqual(injected, ["top_k"])
        self.assertEqual(data["temperature"], 0)
        self.assertIsNone(data["top_p"])
        self.assertEqual(data["top_k"], 20)
        self.assertEqual(data["presence_penalty"], 0)

    def test_all_tracked_routes_inject(self) -> None:
        for route in ("chat", "completions", "responses", "messages"):
            with self.subTest(route=route):
                data = {"model": "qwen3.8-flash-next-chat"}
                self.assertEqual(len(od.apply_sampling_defaults(data, route)), 4)

    def test_unknown_route_injects_nothing(self) -> None:
        data = {"model": "qwen3.8-flash-next-chat"}
        self.assertEqual(od.apply_sampling_defaults(data, "embeddings"), [])
        self.assertEqual(data, {"model": "qwen3.8-flash-next-chat"})

    def test_inject_disabled(self) -> None:
        od.DS4_SAMPLING_INJECT = False
        data = {"model": "qwen3.8-flash-next-chat"}
        self.assertEqual(od.apply_sampling_defaults(data, "chat"), [])
        self.assertEqual(data, {"model": "qwen3.8-flash-next-chat"})

    def test_env_override_applies(self) -> None:
        od.SAMPLING_SETS = od.load_sampling_sets(
            '{"nothink": {"temperature": 0.3, "top_k": 5}}'
        )
        data = {"model": "qwen3.8-flash-next-chat"}
        od.apply_sampling_defaults(data, "chat")
        self.assertEqual(data, {"model": "qwen3.8-flash-next-chat", "temperature": 0.3, "top_k": 5})
        self.assertEqual(od.SAMPLING_SETS["think"], od.SAMPLING_THINK)

    def test_invalid_env_override_falls_back(self) -> None:
        for raw in ("{not json", "[]", '{"think": 3}'):
            with self.subTest(raw=raw):
                sets = od.load_sampling_sets(raw)
                self.assertEqual(sets["think"], od.SAMPLING_THINK)
                self.assertEqual(sets["nothink"], od.SAMPLING_NOTHINK)

    def test_empty_env_override_uses_constants(self) -> None:
        sets = od.load_sampling_sets(None)
        self.assertEqual(sets["think"], od.SAMPLING_THINK)
        self.assertEqual(sets["nothink"], od.SAMPLING_NOTHINK)


class ModelAliasTest(unittest.TestCase):
    def test_derive_from_swift_and_plain_files(self) -> None:
        self.assertEqual(
            od.derive_alias_base("gguf/Swift1.5-Qwen3.8-Flash-Next-Q2.gguf"),
            "swift1.5-qwen3.8-flash-next",
        )
        self.assertEqual(
            od.derive_alias_base("/x/Qwen3.8-Flash-Next-Q2.gguf"),
            "qwen3.8-flash-next",
        )

    def test_quant_suffix_variants(self) -> None:
        for name in (
            "Foo-Q4_K_M.gguf",
            "Foo-Q8_0.gguf",
            "Foo-IQ2_XXS.gguf",
            "Foo-UD-Q4_K_XL.gguf",
            "Foo-BF16.gguf",
            "Foo.gguf",
        ):
            with self.subTest(name=name):
                self.assertEqual(od.derive_alias_base(name), "foo")

    def test_derivation_follows_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "gguf" / "Swift1.5-Qwen3.8-Flash-Next-Q2.gguf"
            real.parent.mkdir()
            real.write_bytes(b"")
            link = Path(tmp) / "ds4flash.gguf"
            os.symlink(real, link)
            self.assertEqual(od.derive_alias_base(str(link)), "swift1.5-qwen3.8-flash-next")
            self.assertEqual(
                od.derive_alias_base("ds4flash.gguf", Path(tmp)),
                "swift1.5-qwen3.8-flash-next",
            )

    def test_override_wins(self) -> None:
        self.assertEqual(
            od.model_alias_base("gguf/Swift1.5-Qwen3.8-Flash-Next-Q2.gguf", " My-Model "),
            "my-model",
        )
        for empty in (None, "", "  "):
            self.assertEqual(
                od.model_alias_base("Qwen3.8-Flash-Next-Q2.gguf", empty),
                "qwen3.8-flash-next",
            )

    def test_listing(self) -> None:
        self.assertEqual(
            od.listed_models("swift1.5-qwen3.8-flash-next"),
            [
                "swift1.5-qwen3.8-flash-next",
                "swift1.5-qwen3.8-flash-next-chat",
                "swift1.5-qwen3.8-flash-next-reasoner",
            ],
        )
        old = od.STATIC_MODELS
        od.STATIC_MODELS = od.listed_models("swift1.5-qwen3.8-flash-next")
        try:
            body = asyncio.run(od.list_models())
        finally:
            od.STATIC_MODELS = old
        ids = [m["id"] for m in body["data"]]
        self.assertEqual(ids, od.listed_models("swift1.5-qwen3.8-flash-next"))
        self.assertNotIn("qwen3.8-flash-next-chat", ids)

    def test_new_and_legacy_ids_map_to_server_aliases(self) -> None:
        base = "swift1.5-qwen3.8-flash-next"
        cases = {
            base: ("", "qwen3.8-flash-next"),
            base + "-chat": ("-chat", "qwen3.8-flash-next-chat"),
            base + "-reasoner": ("-reasoner", "qwen3.8-flash-next-reasoner"),
            "qwen3.8-flash-next": ("", "qwen3.8-flash-next"),
            "qwen3.8-flash-next-chat": ("-chat", "qwen3.8-flash-next-chat"),
            "qwen3.8-flash-next-reasoner": ("-reasoner", "qwen3.8-flash-next-reasoner"),
            "Qwen3.8-Flash-Next-CHAT": ("-chat", "qwen3.8-flash-next-chat"),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(od.resolve_model_alias(model, base), expected)

    def test_unknown_ids_are_not_mapped(self) -> None:
        for model in ("gpt-4", "swift1.5-qwen3.8-flash-next-bogus", "", None, 3):
            with self.subTest(model=model):
                self.assertIsNone(od.resolve_model_alias(model, "swift1.5-qwen3.8-flash-next"))

    def test_plain_file_base_equals_legacy(self) -> None:
        self.assertEqual(
            od.resolve_model_alias("qwen3.8-flash-next-chat", "qwen3.8-flash-next"),
            ("-chat", "qwen3.8-flash-next-chat"),
        )

    def test_presets_apply_to_new_and_legacy_names(self) -> None:
        old = od.DS4_MODEL_ALIAS
        od.DS4_MODEL_ALIAS = "swift1.5-qwen3.8-flash-next"
        sets = od.SAMPLING_SETS
        od.SAMPLING_SETS = od.load_sampling_sets(None)
        try:
            for model in ("swift1.5-qwen3.8-flash-next-chat", "qwen3.8-flash-next-chat"):
                with self.subTest(model=model):
                    self.assertEqual(od.sampling_defaults_for(model), od.SAMPLING_NOTHINK)
            for model in (
                "swift1.5-qwen3.8-flash-next",
                "swift1.5-qwen3.8-flash-next-reasoner",
                "qwen3.8-flash-next",
                "qwen3.8-flash-next-reasoner",
            ):
                with self.subTest(model=model):
                    self.assertEqual(od.sampling_defaults_for(model), od.SAMPLING_THINK)
        finally:
            od.DS4_MODEL_ALIAS = old
            od.SAMPLING_SETS = sets


class ModelIdRewriterTest(unittest.TestCase):
    def test_sse_events_rewritten_without_delay(self) -> None:
        rw = od.ModelIdRewriter("qwen3.8-flash-next-chat", "swift-flash-chat")
        event = b'data: {"id":"a","model":"qwen3.8-flash-next-chat","x":1}\n\n'
        self.assertEqual(
            rw.feed(event),
            b'data: {"id":"a","model":"swift-flash-chat","x":1}\n\n',
        )
        self.assertEqual(rw.flush(), b"")

    def test_split_chunks_and_plain_json(self) -> None:
        rw = od.ModelIdRewriter("m-old", "m-new")
        out = rw.feed(b'data: {"model":"m-')
        out += rw.feed(b'old"}\n\n{"model":"m-old"}')
        out += rw.flush()
        self.assertEqual(out, b'data: {"model":"m-new"}\n\n{"model":"m-new"}')


class ProxyRoutingTest(unittest.TestCase):
    """Drives the real proxy route against a mock upstream, so no ds4-server
    (and no model) is ever started."""

    def test_legacy_and_new_ids_reach_server_as_legacy_and_echo_requested(self) -> None:
        import json

        import httpx
        from fastapi.testclient import TestClient

        seen: list[dict] = []

        def upstream(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(
                200,
                content=json.dumps(
                    {"object": "chat.completion", "model": body["model"], "choices": []},
                    separators=(",", ":"),
                ).encode(),
                headers={"content-type": "application/json"},
            )

        async def no_start() -> None:
            return None

        saved = (od.ensure_started, od.state.http_client, od.DS4_MODEL_ALIAS)
        od.ensure_started = no_start
        od.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        od.DS4_MODEL_ALIAS = "swift1.5-qwen3.8-flash-next"
        try:
            client = TestClient(od.app)
            cases = {
                "swift1.5-qwen3.8-flash-next-chat": "qwen3.8-flash-next-chat",
                "swift1.5-qwen3.8-flash-next-reasoner": "qwen3.8-flash-next-reasoner",
                "swift1.5-qwen3.8-flash-next": "qwen3.8-flash-next",
                "qwen3.8-flash-next-chat": "qwen3.8-flash-next-chat",
                "qwen3.8-flash-next-reasoner": "qwen3.8-flash-next-reasoner",
                "qwen3.8-flash-next": "qwen3.8-flash-next",
            }
            for requested, sent in cases.items():
                with self.subTest(model=requested):
                    seen.clear()
                    resp = client.post(
                        "/v1/chat/completions",
                        json={"model": requested, "messages": []},
                    )
                    self.assertEqual(resp.status_code, 200)
                    self.assertEqual(seen[0]["model"], sent)
                    self.assertEqual(resp.json()["model"], requested)
                    nothink = requested.endswith("-chat")
                    self.assertEqual(seen[0]["presence_penalty"], 1.5 if nothink else 0.0)
        finally:
            od.ensure_started, od.state.http_client, od.DS4_MODEL_ALIAS = saved


if __name__ == "__main__":
    unittest.main()
