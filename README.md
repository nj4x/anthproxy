# anthproxy

A local HTTP proxy that translates Anthropic Messages API requests to multiple LLM backend APIs. Route between Bedrock, Codex, Anthropic, OpenRouter, and local LM Studio endpoints — all through a single Anthropic-compatible interface.

## Features

- **Backend routing** — Switch between Bedrock, Anthropic, Codex, OpenRouter, and local endpoints via simple proxy commands
- **Automatic backend selection** — Intelligently select backends based on subscription quota availability and configuration
- **Model-tier routing** — Route requests by task complexity using a lightweight classifier or deterministic rules
- **Admin UI** — Optional web interface for monitoring usage, routing status, and session management
- **Session pinning** — Override backend selection per-session (e.g., pin Claude Code to a specific backend)
- **Usage tracking** — Per-backend token counts, estimated costs, and subscription window visibility

## Quick start

Install and run:

```bash
pip install -e .
python -m anthproxy --port 8082
```

Send a message:

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -d '{"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'
```

Check status with a local proxy command:

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"proxy-status"}]}'
```

## Supported backends

| Backend | Auth | Upstream |
|---------|------|----------|
| **Bedrock** | AWS credentials in `x-api-key` header | AWS Bedrock Converse API |
| **Anthropic** | OAuth (Claude subscription) | Anthropic Messages API |
| **Codex** | OAuth (ChatGPT account) | ChatGPT Codex via OpenAI Codex endpoint |
| **OpenRouter** | API key in config | OpenRouter's Anthropic-compatible gateway |
| **Local** | None | LM Studio or local Anthropic-compatible server |
| **Peer** | `X-Anthproxy-Peer-Key` from `--peer-api-key` (optional) | Another anthproxy instance via its `/v1/messages` |

## Configuration

### Environment & flags

```bash
anthproxy \
  --host 127.0.0.1 \                           # Bind address (default)
  --port 8082 \                                 # Bind port (default)
  --backend bedrock \                           # Default backend (default)
  --backends anthropic,codex \                  # Restrict discoverable backends (default: all)
  --region us-east-1 \                          # AWS region for Bedrock
  --auto-backend \                              # Enable automatic selection (default)
  --auto-backend-mode subscription \            # Start mode: subscription | auto
  --enable-ui \                                 # Enable admin UI + session DB (default: off)
  --log-level INFO                              # Log level (default)
```

See `python -m anthproxy --help` for the full option list.

`--backends` (env: `ANTHPROXY_BACKENDS`) takes a comma-separated allowlist restricting
which backends are discoverable/selectable at all — via `--backend`, `/backend` local
commands, the admin UI, and auto-backend rotation. Omit it to leave every installed
backend available (the default) — except `peer`, which is enabled by `--peer-base-url`
rather than by installation, and must still be listed explicitly when `--backends` is
passed. An unknown name or an empty list is a startup error.
`--backend`'s own default is silently repaired to the first enabled backend when the
allowlist excludes it; an *explicit* `--backend` outside the allowlist is a startup
error naming both flags.

### Authentication

**Bedrock:** AWS access key + secret in `x-api-key` header (base64-encoded):
```
base64(access_key_id|secret_access_key|session_token)
```

**Anthropic & Codex:** OAuth (interactive browser login on first run, then automatic refresh):
```bash
python -m anthproxy --backend anthropic  # Opens browser for Claude subscription OAuth
python -m anthropproxy --backend codex   # Opens browser for ChatGPT OAuth
```

**OpenRouter:** Set `OPENROUTER_API_KEY` environment variable or `--openrouter-api-key` flag.

**Local:** No auth required.

## Endpoints

### POST /v1/messages

Standard Anthropic Messages API (streaming and non-streaming).

```bash
# Non-streaming
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "model": "sonnet",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Streaming
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "model": "sonnet",
    "max_tokens": 100,
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### POST /v1/messages/count_tokens

Count input tokens for a request.

```bash
curl -X POST http://localhost:8082/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "model": "sonnet",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Local proxy commands

Send these as the entire text of the final user message to control the proxy locally (never reach the LLM):

| Command | Effect |
|---------|--------|
| `proxy-help` | List all available commands |
| `proxy-status` | Show active backend + subscription usage |
| `proxy-get-backend` | Report active backend (shows session override if set) |
| `proxy-set-backend:bedrock` | Switch to Bedrock (global) |
| `proxy-set-backend:codex` | Switch to Codex (global) |
| `proxy-set-backend:anthropic` | Switch to Anthropic (global) |
| `proxy-set-backend:openrouter` | Switch to OpenRouter (global, requires `OPENROUTER_API_KEY`) |
| `proxy-set-backend:local` | Switch to Local (LM Studio, pinned, never auto-selected) |
| `proxy-set-backend:auto` | Resume automatic backend selection |
| `proxy-set-backend:subscription` | Auto-select subscription backends only (Anthropic/Codex/OpenRouter) |
| `proxy-set-backend:<name>:session` | Pin backend for this session only |
| `proxy-set-backend:auto:session` | Clear session backend pin |
| `proxy-stats` | Show today's token usage grouped by hour |
| `proxy-stats:-1d` | Show yesterday |
| `proxy-stats:1w` | Show this week grouped by day |
| `proxy-stats:<period>:<backend>` | Filter stats to one backend (e.g., `proxy-stats:1d:bedrock`) |
| `proxy-get-usage` | Fetch subscription windows from upstream API |

Example:

```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"proxy-status"}]}'
```

## Model tier routing

When `--auto-model-routing` is enabled, requests are routed by complexity to different model tiers (Haiku for simple tasks, Sonnet for standard, Opus for deep work).

Routing modes:
- **classifier** (default) — Lightweight LLM call classifies the request into `trivial`, `standard`, or `deep`
- **rules** — Deterministic keyword-based routing
- **tag** — Use a task name supplied by `X-Anthproxy-Override: task:<name>` header

Long-context requests (> 150k tokens by default) are always routed to Opus with a 1-minute thinking context.

Override globally with `proxy-set-model-routing:on` / `proxy-set-model-routing:off`, or per-session with `:session` suffix.

Use `X-Anthproxy-Override: no-classifier` to bypass routing for a single request.

## Chaining anthproxy instances

The `peer` backend dispatches to another anthproxy over its public `/v1/messages`, so one instance
can forward to another that holds the credentials.

### Setup

| Flag | Environment | Effect |
|------|-------------|--------|
| `--peer-base-url` | `ANTHPROXY_PEER_BASE_URL` | Target anthproxy instance. Setting it is what enables the `peer` backend. |
| `--peer-api-key` | `ANTHPROXY_PEER_API_KEY` | Credential sent to the peer as `X-Anthproxy-Peer-Key`. Optional; unset means no header. |

Enablement rules:

- `peer` is not enabled by installation. `--peer-base-url` enables it.
- With `--backends` passed, `peer` must also be listed there — the base URL alone is not enough.
- Naming `peer` in `--backends` without a base URL is a startup error.

Worked two-instance loopback example. The inner instance holds the subscription credentials; the
outer forwards to it:

```bash
# Inner instance — authenticates upstream as usual
python -m anthproxy --port 8083 --backend anthropic

# Outer instance — forwards to the inner one
python -m anthproxy --port 8082 \
  --peer-base-url http://127.0.0.1:8083 \
  --backends peer \
  --backend peer

# Clients talk to the outer instance
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'
```

A peer mounted behind a reverse proxy at a subpath works: any path component of the base URL
prefixes the outbound paths.

### Security posture

**anthproxy has no inbound authentication.** No client credential is checked for access, on any
endpoint, chained or not. A chain is only as private as the network between its
hops — the supported deployments are loopback, an SSH tunnel, or a private network you already
control.

**`--peer-api-key` sends, it never checks.** It exists so a peer can sit behind an existing
access-control layer — a reverse proxy, an identity-aware proxy, a mesh with its own authentication
— and that layer is what consumes the `X-Anthproxy-Peer-Key` header.

**The header is inert at the receiving instance, and must stay inert.** No anthproxy reads it; that
is the property that makes it safe to send. Making a receiving instance *check* it would be adding
inbound authentication — a separate decision to be taken as such, revisiting SRS-Chaining-006's
second invariant. It is not a hardening tweak to be slipped in alongside something else.

The key is never sent as `Authorization: Bearer`. A gateway that requires `Authorization`
specifically is a deployment concern its own configuration can bridge; anthproxy will not use that
header, because the receiving instance's OAuth credential path would absorb the peer key into its
own credential state.

### Operational surprises

Four behaviours that read as bugs if you have not seen them before:

- **An enabled peer displaces the bedrock fallback.** A peer always reports itself available, so it
  occupies the available pool that `bedrock` is only reached *after*. Run both and metered bedrock
  fallover stops firing. This is intended — a peer forwards to a subscription, bedrock bills per
  token — but it is invisible until the day it matters.
- **Routing config on the outer instance is inert for peer-bound traffic.** Chained requests are
  classified by the peer, against the peer's own configuration. Tuning model-tier routing on the
  outer instance has no effect on what leaves for the peer.
- **Cost totals are per-instance and must not be summed.** A chained request is recorded by both
  hops, so adding the two instances' totals double-counts every one. Session keys are identical
  across hops, which makes the records line up request-for-request — useful for diagnosis,
  dangerous for arithmetic.
- **The inner instance is not addressable through the outer.** No `X-Anthproxy-Override` header and
  no `proxy-*` command crosses the hop; both are consumed where they arrive. Configure or query the
  inner instance by talking to it directly. `prefer:peer` is the intended escape hatch — it is
  meaningful at the outer hop and honoured there.

### Loop guard and its known gaps

At startup, an instance with `peer` enabled resolves the peer target and refuses to boot if it
resolves to its own bind address, naming `--peer-base-url`. A target that does not resolve yet logs
a warning and proceeds, so an outer instance can start before the inner one exists.

Three gaps remain, knowingly:

- A multi-instance cycle (A→B→A) is **not** detected, and will amplify until something else fails.
- A self-reference formed after startup is not detected.
- A self-loop behind a name that does not resolve at boot slips through.

These are carried as SRS-Chaining-007 in `docs/FS-SRS-requirements-bootstrap.md`, recorded as
deferred and unsatisfied rather than scoped away. If chains of three or more hops become normal,
the follow-up is a per-request instance-ID marker.

### Reasoning

ADR-0021 (peer backend package), ADR-0022 (peer selection and neutral status), ADR-0023 (innermost
hop classification authority), ADR-0024 (the peer hop as a control boundary), ADR-0025 (per-hop
independent accounting), ADR-0026 (startup self-reference check) — all under `docs/adr/`.

## Admin UI

Enable the optional web UI with `--enable-ui`. It exposes:
- Session tracking (request history, model usage, latency)
- Routing status and per-backend statistics
- Backend and model-routing controls

The UI is unauthenticated and should only be bound to localhost (`127.0.0.1`) or protected by external access control.

```bash
python -m anthproxy --enable-ui --port 8082
# Then open http://127.0.0.1:8082/ui/ in your browser
```

![anthproxy admin dashboard](docs/images/admin-dashboard.png)

## Development

Install dev dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run python -m pytest tests/ -v
```

Run linter and type checks:

```bash
uv run python -m ruff check .
```

For the React admin UI:

```bash
cd anthproxy/ui
npm ci
npm run dev          # Development server (live reload)
npm run build        # Production bundle
```

The UI's production bundle (`anthproxy/ui/dist/`) is checked in; rebuild and commit it when UI source changes.

## Model aliases

Short names like `sonnet`, `opus`, `haiku` map to backend-specific model IDs automatically.

| Alias | Bedrock | Codex | Anthropic | OpenRouter |
|-------|---------|-------|-----------|------------|
| `opus` | claude-opus-4-... | gpt-5.5 | claude-opus-4-8 | claude-opus-4-8 |
| `sonnet` | claude-sonnet-4-... | gpt-5.4 | claude-sonnet-4-6 | claude-sonnet-4-6 |
| `haiku` | claude-haiku-4-... | gpt-5.4-mini | claude-haiku-4-5-... | claude-haiku-4-5-... |

You can also use full Anthropic model names (e.g., `claude-sonnet-4-6`) or native backend IDs (e.g., `anthropic.claude-sonnet-4-6` for Bedrock).

## License

MIT License. See LICENSE.txt for details.
