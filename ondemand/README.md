# ds4-ondemand

On-demand launcher/proxy for DwarfStar's `ds4-server`, living next to oMLX
on this 64 GB Apple Silicon Mac.

## Why

DS4 (Qwen3.8-Flash-Next Q2, with `--mtp`) needs ~43 GiB resident. oMLX's
35B-A3B needs ~20 GiB. They cannot both be resident at once. oMLX already
idle-unloads its own models after a 600s TTL, but `ds4-server` itself has no
idle unload and no on-demand start — it is a plain HTTP server that stays
loaded forever once started. This proxy gives it both: it starts
`ds4-server` lazily on first request (after unloading whatever oMLX
currently has resident) and stops it again after a period of inactivity.

## What it does

- Listens on `DS4_ONDEMAND_HOST:DS4_ONDEMAND_PORT` (default
  `0.0.0.0:8001`, reachable from the LAN/tailnet — e.g. the RAG CT).
  `ds4-server` itself always stays bound to `127.0.0.1:8000` and is never
  exposed directly.
- `GET /v1/models` answers instantly from a static list of the three model
  aliases, without starting `ds4-server`, so model pickers work while it is
  cold. The ids are derived from the served GGUF (see `DS4_MODEL_ALIAS`):
  `<base>`, `<base>-chat` (thinking off) and `<base>-reasoner` (thinking on).
  The legacy `qwen3.8-flash-next`, `-chat` and `-reasoner` ids are still
  accepted but not listed. The launcher rewrites the request `model` to the
  legacy id `ds4-server` understands, and the response reports the id the
  client asked for.
- Every other `/v1/*` path (`/v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, `/v1/messages`) is proxied to `ds4-server`, starting it
  first if needed. Streaming (SSE) responses are passed through chunk by
  chunk, not buffered.
- Before starting `ds4-server`, the proxy asks oMLX (`GET
  /v1/models/status`) which checkpoints are currently `loaded` and issues
  `POST /v1/models/{id}/unload` for each. This tolerates oMLX being
  completely stopped (connection refused is logged and ignored).
- Concurrent cold requests are serialized behind a single `asyncio.Lock`, so
  only one `ds4-server` start ever happens no matter how many requests
  arrive at once while it's down.
- After `DS4_IDLE_SECONDS` (default 300) with zero in-flight requests,
  `ds4-server` is stopped: `SIGTERM`, then `SIGKILL` after a 10s grace
  period. The same shutdown happens if the proxy itself is stopped, so no
  orphan `ds4-server` process survives a proxy restart.

## Ports

| What | Address |
|---|---|
| Proxy (public) | `DS4_ONDEMAND_HOST:DS4_ONDEMAND_PORT`, default `0.0.0.0:8001` |
| ds4-server (internal only) | `127.0.0.1:8000`, never exposed |
| oMLX (queried, not modified) | `http://127.0.0.1:8843` |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DS4_ONDEMAND_HOST` | `0.0.0.0` | Proxy bind address |
| `DS4_ONDEMAND_PORT` | `8001` | Proxy bind port |
| `DS4_SERVER_PORT` | `8000` | `ds4-server`'s loopback port |
| `DS4_REPO_DIR` | `/Users/beaglemoo/Homelab/dwarfstar` | cwd for `ds4-server` (it loads `metal/*.metal` relative to cwd) |
| `DS4_WORKDIR` | `DS4_REPO_DIR` | cwd for `ds4-server` when it differs from the repo (bundled installs: the dir holding `metal/`) |
| `DS4_BINARY` | `$DS4_REPO_DIR/ds4-server` | Path of the `ds4-server` binary |
| `DS4_LOG_DIR` | `$DS4_REPO_DIR/logs` | Directory for `ds4-server.log` |
| `DS4_STATE_DIR` | `~/.unsloth/engines` (the log dir if that does not exist) | Directory for `ds4-state.json`, which holds the ctx set through `POST /admin/config`; at startup it overrides `DS4_CTX` |
| `DS4_MODEL_FILE` | `ds4flash.gguf` | `-m` argument, relative to `DS4_WORKDIR` (or an absolute path). The symlink points at Swift 1.5 (`gguf/Swift1.5-Qwen3.8-Flash-Next-Q2.gguf`) since 2026-09-25; see `docs-local/10-model-swift15.md` |
| `DS4_MODEL_ALIAS` | derived | Alias base for the listed model ids. Default: `realpath` of the model file, basename, minus `.gguf` and a trailing quant tag (`-Q2`, `-Q4_K_M`, `-Q8_0`), lowercased (`swift1.5-qwen3.8-flash-next` for the Swift file, `qwen3.8-flash-next` for the plain one) |
| `DS4_CTX` | `65536` (code default; this install's plist sets it to `131072`) | `--ctx` |
| `DS4_PREFILL_CHUNK` | `1024` | `--prefill-chunk` |
| `DS4_START_TIMEOUT` | `120` | Seconds to wait for `/v1/models` to return 200 before giving up |
| `DS4_IDLE_SECONDS` | `300` | Seconds with 0 in-flight requests before stopping `ds4-server` |
| `OMLX_BASE_URL` | `http://127.0.0.1:8843` | Where to query/unload oMLX models |
| `DS4_SAMPLING_INJECT` | `1` | Set to `0` to stop injecting sampling defaults entirely |
| `DS4_SAMPLING_DEFAULTS` | unset | JSON `{"think": {...}, "nothink": {...}}` replacing the built-in sets; unparseable values are logged and ignored |
| `DS4_DEBUG` | `0` | Set to `1` for per-request debug lines (the sampling-injection line) |

The sampling sets, the `-chat` alias rule, and the rule that a client-set value
always wins are documented in
[docs-local/05-ondemand-launcher.md](../docs-local/05-ondemand-launcher.md#sampling-defaults).

The exact spawn command is:

```
./ds4-server -m ds4flash.gguf --ctx 131072 --prefill-chunk 1024 --mtp --host 127.0.0.1 --port 8000
```

run with `cwd=DS4_REPO_DIR`.

**Prerequisite:** `--ctx 131072` requires the macOS Metal wired-memory ceiling
to be raised first, otherwise `ds4-server` gets killed by memory pressure:

```sh
sudo sysctl iogpu.wired_limit_mb=57344
```

(56 GiB; persist with `echo "iogpu.wired_limit_mb=57344" | sudo tee -a
/etc/sysctl.conf`.) Without it, `DS4_CTX` should stay at or below `65536`. `ds4-server`'s own stdout/stderr go to
`/Users/beaglemoo/Homelab/dwarfstar/logs/ds4-server.log`. The proxy's own
logs go to stdout (captured by launchd into
`/Users/beaglemoo/Homelab/dwarfstar/logs/ondemand.log`).

## Admin endpoints

- `GET /admin/status` — `loaded`, `pid`, `uptime_seconds`,
  `last_activity_seconds_ago`, `in_flight`, `idle_seconds_remaining`,
  `stats`, `config`.
- `POST /admin/stop` — force-stop `ds4-server` now (frees ~43 GiB
  immediately). Safe to call even if it isn't running.
- `POST /admin/start` — force-start `ds4-server` now (same cold-start path
  as a proxied request, including the oMLX unload step). Blocks until
  ready or returns `503` with the log tail on failure.

- `GET /admin/config` — `{"ctx", "ctx_active", "ctx_min": 4096, "ctx_max":
  262144, "pending_restart"}`. `ctx_active` is the ctx of the running
  `ds4-server`, or `null` when it is stopped.
- `POST /admin/config {"ctx": N}` — validates `N` (integer, 4096..262144,
  rounded to a multiple of 256), persists it to `ds4-state.json`, and answers
  with the config above plus `applied`: `next_start` (not running),
  `restarted` (running and idle: stopped gracefully, not pre-warmed, the next
  request respawns it with the new ctx), `after_current_requests` (requests
  in flight: `pending_restart` is true and it is stopped once they finish) or
  `unchanged` (already running with that ctx).

### `timings`

For chat and completions, the final SSE usage chunk (`"choices": []`) and
non-streaming JSON bodies carry a llama.cpp-style top-level `timings` object:
`prompt_n` (prompt tokens minus cached), `prompt_ms` (time to first content),
`prompt_per_second`, `predicted_n`, `predicted_ms` (first content to end),
`predicted_per_second`, `cache_n`. It is omitted when usage is missing.
Non-streaming bodies cannot split the phases, so both use the whole request
time.

### `stats`

Request throughput stats, tracked for POSTs to `/v1/chat/completions`,
`/v1/completions`, `/v1/responses`, and `/v1/messages` only. In-memory,
reset on launcher restart (a separate Swift menu bar app polls this).

```json
"stats": {
  "live": null | {"gen_tokens": int, "gen_tps": float, "elapsed_s": float, "phase": "prefill"|"decode"},
  "last": null | {"model": str, "prompt_tokens": int|null, "completion_tokens": int, "ttft_ms": float, "prefill_tps": float|null, "gen_tps": float, "duration_s": float, "finished_at": float},
  "totals": {"requests": int, "completion_tokens": int, "errors": int}
}
```

- `live` — non-null only while a tracked request is in flight (if several,
  the most recent). `phase` is `"prefill"` until the first content token
  arrives, then `"decode"`.
- `last` — the most recently completed tracked request. `finished_at` is a
  Unix timestamp.
- `totals` — cumulative counters across all tracked requests since the
  launcher started. `errors` counts upstream non-2xx responses and proxy
  exceptions.

For streaming chat/completions requests with no `stream_options` in the
body, the proxy injects `{"stream_options": {"include_usage": true}}` so
`ds4-server` emits a final usage chunk; SSE lines are parsed as they pass
through, unaltered and undelayed. Non-streaming responses are measured from
the full JSON body once it has arrived. Any parse failure is swallowed and
logged (non-fatal) — stats never break or truncate a proxied response.

## oMLX interplay and its known gap

On every cold start, the proxy unloads whatever oMLX currently reports as
`loaded` via `/v1/models/status`. This is one-directional: **oMLX does not
know to unload DS4.** If `ds4-server` is resident (~43 GiB) and a client
then asks oMLX to load its 35B-A3B (~20 GiB), oMLX may refuse or thrash
under memory pressure until DS4 idles out on its own.

If you need oMLX right away, call:

```sh
curl -X POST http://127.0.0.1:8001/admin/stop
```

first, to free DS4's memory before asking oMLX to load anything.

## Adding it to a client

### Open WebUI / any OpenAI-compatible client (pi, RAG CT, etc.)

Point the OpenAI base URL at `http://<mac-ip>:8001/v1` (or
`http://127.0.0.1:8001/v1` if calling from the Mac itself). No API key is
required.

```sh
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen3.8-flash-next-chat",
        "messages": [{"role": "user", "content": "Say hello in five words"}],
        "stream": true,
        "stream_options": {"include_usage": true}
      }'
```

The first request after a cold stop starts `ds4-server` (allow up to
`DS4_START_TIMEOUT` seconds; typically ~10s to ready plus ~20s to first
token on a genuinely cold start). Subsequent requests are immediate until
`DS4_IDLE_SECONDS` of inactivity passes.

### Anthropic-compatible clients (Claude Code, etc.)

```sh
curl -s http://127.0.0.1:8001/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
        "model": "qwen3.8-flash-next-chat",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello in five words"}]
      }'
```

## Install / uninstall

```sh
./install.sh    # copies the plist, launchctl bootstrap, prints status
./uninstall.sh  # launchctl bootout, removes the plist
```

Both are idempotent — safe to re-run.
