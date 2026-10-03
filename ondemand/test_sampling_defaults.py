"""Unit tests for the proxy's Qwen sampling-default injection.

Importing ds4_ondemand builds the FastAPI app but starts nothing; uvicorn only
runs under __main__. The launcher runs under `uv run`, so run these with the
same interpreter and dependency set:

    cd ~/Homelab/dwarfstar
    uv run --with fastapi --with uvicorn --with httpx \
        python -m unittest ondemand/test_sampling_defaults.py
"""

from __future__ import annotations

import sys
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


if __name__ == "__main__":
    unittest.main()
