# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development

Use `README.md` for setup, run, test, and UI development commands. The Python package includes the checked-in `anthproxy/ui/dist/`; after UI source changes, rebuild it and commit the generated assets, including the `dist/index.html` update.

## Sources of truth

- `README.md`: user-facing overview, setup, CLI usage, local commands, session behavior, and examples
- `anthproxy/config.py`: server CLI options, environment variables, and `Config` defaults
- `anthproxy/model_config.py`: model aliases, pricing, labels, and external model configuration; do not duplicate concrete values
- `anthproxy/model_router.py`: routing modes, routing decisions, and routing behavior

## Technical constraints

- Python 3.10+
- The server is threaded; requests run concurrently.
- Outbound HTTP uses stdlib; Bedrock uses botocore and must create a client per request. Do not share Bedrock clients across threads.

## Architecture overview

`anthproxy.__main__` loads `Config`, prepares the initial backend, creates `BackendRegistry`, `StatsCollector`, and optional `SessionDB`, then starts the server. `server.py` owns backend construction, immutable dispatch snapshots, runtime switching, selector integration, and handler-class wiring. `ProxyRequestHandler` in `handlers.py` intercepts local commands, applies routing, takes one registry snapshot, dispatches to a provider backend, translates errors and streams, records usage and session data, and emits Anthropic-compatible responses.

Each provider package follows the same boundary: `backend.py` owns transport and provider runtime behavior, `mapper.py` converts between Anthropic Messages shapes and provider protocol shapes, and `auth.py` owns credentials where needed. Shared SSE builders and content normalization live under `anthproxy/mapper/`; shared OAuth refresh and persistence live under `anthproxy/_shared/`.

The admin UI is a React/Vite SPA under `anthproxy/ui/src`; its checked-in production bundle is served from `anthproxy/ui/dist`. `SessionDB` persists request, session, and trace data; `StatsCollector` maintains usage and routing economics. UI, DB, and stats consume normalized usage fields, so provider mappers must translate provider-specific token semantics into the Anthropic convention before data reaches handlers. Admin/UI endpoints are unauthenticated and may expose conversation history or routing controls; UI-enabled servers must remain loopback-bound unless protected by an external access-control layer.

## Detailed guidance

Module ownership and who-may-touch-what boundaries:
@docs/agents/architecture-boundaries.md

Auto-routing rules, classifier isolation, tier-cache behavior, and session-key semantics:
@docs/agents/model-routing.md

Local-command matching order and non-interactive backend switching:
@docs/agents/local-commands.md

Lock-scope discipline and selector behavior invariants:
@docs/agents/concurrency.md

Per-provider transport and mapping quirks (Codex, Anthropic, Local):
@docs/agents/backend-providers.md

Credential isolation/atomicity and client-facing failure shaping:
@docs/agents/credentials-and-errors.md
