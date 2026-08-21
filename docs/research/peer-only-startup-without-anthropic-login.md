# Peer-only startup without an Anthropic login prompt

## Question

In a fresh environment with no credentials on disk,

```bash
python -m anthproxy --backends anthropic,peer --peer-base-url http://127.0.0.1:28082
```

opens an interactive Anthropic OAuth login (blocked in the operator's environment). The intent
was "an OAuth-capable backend plus the peer, where the peer holds the real credentials".
How do we boot?

## Answer

Drop `anthropic` from the allowlist. The peer *is* the credential holder; the outer instance
needs none of its own.

```bash
python -m anthproxy --port 8082 \
  --backends peer \
  --peer-base-url http://127.0.0.1:28082 \
  --backend peer
```

**Verified** — run end to end in a sandboxed `HOME`/`ANTHPROXY_HOME`; boots with no credential
prompt, no network call, `backend=peer, auto-backend=on`. `--backend peer` is optional: with
`--backends peer` the default `bedrock` is repaired to `peer` (`config.py:551-559`, logged as
`Backend default repaired: bedrock -> peer`), but stating it removes the warning. This is the
form README.md:209-216 already documents as the thin-forwarding front end.

## Root cause

`anthropic` merely *appearing in the enabled set* triggers interactive login — not being the
active backend, not any request.

`anthproxy/__main__.py:105-119`:

```python
if config.auto_backend:                       # default True (config.py:172)
    if 'anthropic' in enabled_names:          # membership, not activeness
        anthropic_auth.ensure_credentials(config)
```

`enabled_names` is `frozenset(backend_names())` (`__main__.py:103`), i.e. the `--backends`
allowlist. `ensure_credentials` is the *interactive* variant: `_shared/oauth_base.py:370-373`
— `creds is None` → `provider.login_fn(home)` → `anthropic/auth.py:206-213` →
`run_pkce_login_server` → `webbrowser.open(auth_url)` (`_shared/oauth_base.py:229-235`), with a
300-second wait (`_LOGIN_TIMEOUT`, `oauth_base.py:44`).

The `try/except` at `__main__.py:110-115` does not save you: it only catches *after* login
fails or times out. The prompt has already been printed and the browser already opened.

This is exactly the distinction `docs/agents/credentials-and-errors.md` draws — the
non-interactive path (`ensure_credentials_noninteractive`, `oauth_base.py:395-410`) exists and
is used by runtime switches and the selector's `_refresh_tokens` (`selector.py:516-534`), but
startup in auto mode deliberately uses the interactive one. There is no flag to swap them.

## Is the mental model right?

Partly. `anthropic` and `peer` are unrelated things here:

- OAuth-backed backends are `anthropic` and `codex` only — the hardcoded pair in
  `selector.py:516` and the two branches in `__main__.py:105-119`. Both would prompt.
- `anthproxy/oauth/` is a **registered but internal** backend (`backends_registry.py:51`,
  `_INTERNAL_BACKENDS = {'oauth'}`). It is exempt from the allowlist filter
  (`backends_registry.py:88-94`), is not in `_DECLARED_ORDER` for user-facing lists, and is
  selected internally by the registry (`server.py:415-430`), not by `--backends`. It is the
  enterprise-OAuth delegation path, not something to name on the CLI.
- `peer` is **not** an OAuth backend and holds no credential. It is in `ROTATABLE_BACKENDS`
  but not `SUBSCRIPTION_BACKENDS` (`constants.py:9-21`) — "the selector may rotate onto it"
  but "its capacity is not a subscription".

So "only an OAuth backend + peer" is self-contradictory for this goal: any OAuth backend in
the set is precisely what prompts. What the operator wants is peer-only.

## Does peer-only actually work?

Yes, at every layer:

- **Auto mode survives.** `_disable_auto_backend_if_nothing_to_rotate` tests
  `ROTATABLE_BACKENDS`, which contains `peer` (`__main__.py:56-72`, `constants.py:21`), so a
  peer-only set keeps a live selector. Verified in the boot run: `auto-backend=on`.
- **Peer is selectable in the default `subscription` mode.** `_compute_best_unlocked` iterates
  `self._priority` (= `ROTATABLE_BACKENDS ∩ enabled`, `selector.py:49,192`) and
  `subscription_only` only suppresses STEP-2 bedrock parking (`selector.py:560-568`) — it does
  not filter the candidate pool. Pinned by `tests/test_selector.py:1233-1245`
  (`test_peer_selected_in_default_subscription_mode`). No mode flag needed.
- **No credential work happens.** Neither `anthropic` nor `codex` is in the set, so
  `__main__.py:105-119` is inert, and `_refresh_tokens` skips both via the
  `name not in self._priority` guard (`selector.py:516-518`).
- **Peer status is free.** `PeerBackend.five_hour_status` returns a constant neutral status
  with no network call (`peer/backend.py:305-320`), so the startup `selector.evaluate()`
  (`__main__.py:149`) does not reach out anywhere.

## Alternatives

| Option | Boots without login? | Tradeoff |
|---|---|---|
| **A. `--backends peer`** (recommended) | **Verified** yes | `anthropic` unavailable even for a later runtime switch. Correct if the peer is the only credential holder. |
| **B. Keep `anthropic`, run static: `--backends anthropic,peer --backend peer --no-auto-backend`** | **Verified** yes | `__main__.py:122-130` prepares credentials only when `config.backend` is `codex`/`anthropic`; with `peer` neither branch fires. Cost: no auto-rotation at all (`mode=token-refresh-only`), and a later `proxy-set-backend:anthropic` fails non-interactively (`oauth_base.py:409-410`) until credentials exist. |
| **C. Pre-populate `auth.json` out of band** | Inferred, **not tested** | Write a valid `{"access_token", "refresh_token", "expires_at", ...}` to `$ANTHPROXY_HOME/anthropic/auth.json` (`oauth_base.py:59-60`; home resolution `anthropic/auth.py:41-63` — note the `ANTHPROXY_HOME/anthropic` directory must already exist, `p.is_dir()`). `ensure_credentials` then takes the refresh branch, not login (`oauth_base.py:375-383`). **Danger:** if the refresh token is stale/bogus, `_classify_refresh_error` marks it terminal and `oauth_base.py:384-386` falls straight back to `login_fn` — a browser again. Not tested here because testing it means a real token exchange. |
| **D. A headless / `--no-login` flag** | No such flag | Grepped `config.py` for headless/non-interactive/no-login/skip-setup: nothing. The only lever over the interactive path is which backends are enabled and whether auto mode is on. |

## Sanity checks on the invocation

- **`--port`.** Outer must not bind `28082`. ADR-0026's guard returns early when
  `peer_port != config.port` (`peer/backend.py:149-150`), so the default `8082` against a peer
  on `28082` is fine. **Verified** the refusal fires for a matching port: running with
  `--port 18099 --peer-base-url http://127.0.0.1:18099` exits 2 with
  `resolves to this instance's own listening address`.
- **`--peer-api-key` is not needed** for a bare local pair. The outer *sends* it as
  `X-Anthproxy-Peer-Key`; anthproxy never checks one inbound (README.md:234-236,
  `docs/agents/backend-providers.md`). Set it only if the inner instance sits behind an
  external access-control layer that consumes it. The inner instance takes no matching flag.
- **`--backends anthropic,peer` without `--peer-base-url` would have been a hard error**
  (`config.py:96-103`) — the operator's target is set, so that gate passed. The allowlist is
  exhaustive: naming `--backends` means `peer` must be listed explicitly, which it is.
- **After startup, `anthropic`-in-set degrades gracefully** *if* you get past the login.
  `AnthropicBackend.five_hour_status` swallows credential errors and returns
  `available=None` (`anthropic/backend.py:293-299`), which keeps it out of the available pool,
  so the selector settles on `peer`. That is why option B works. It does not rescue the
  default auto-mode case, because the prompt happens before any of this.
- **Routing config on the outer instance is inert** for peer-bound traffic (ADR-0023 /
  `docs/agents/model-routing.md`). Anything set there — including the default
  `--lock-requested-model claude-sonnet-4-6` (`config.py:504-514`) — has no observable effect
  once every request goes to the peer. Configure routing on the inner instance.

## Uncertain / unverified

- Option C was reasoned from source only; no token was exchanged.
- The interactive login trigger itself was traced through code but deliberately not executed
  (executing it means opening a browser to `claude.ai`). The three boots reported as
  "verified" above are the peer-only case, the static-mode case, and the ADR-0026 refusal.
- Copying an `auth.json` between instances is not addressed anywhere in the repo's docs or
  tests; the concurrent-refresh guidance in `docs/agents/credentials-and-errors.md` suggests
  two instances sharing one credential directory would contend on the refresh lock
  (`oauth_base.py:377`). Untested.
