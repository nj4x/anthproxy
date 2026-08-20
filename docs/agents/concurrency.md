# Concurrency and selector

Lock-scope discipline and selector behavior invariants.

- `BackendRegistry._state_lock` and selector locks guard only short state transitions. Never hold them across inference, streaming, network I/O, OAuth refresh, usage lookup, backend construction, or candidate preparation. `_prepare_lock` serializes slow candidate preparation.
- Do not describe selector behavior as a fixed priority or drift-prone algorithm summary, and do not claim that token refresh runs independently.
- Preserve incumbent-hold and exhaustion-parking ordering. Unknown utilization remains neutral, and transient usage-fetch gaps do not remove incumbency. Subscription-only operation skips parking.
- Two backend tuples, differing by one member: `SUBSCRIPTION_BACKENDS` means "backends whose capacity is a subscription" (`/backend subscription`, the `subscription` stats filter, help text); `ROTATABLE_BACKENDS` means "backends the selector may rotate onto" (the selector pool, the `prefer:` gate, the auto-backend kill-switch). The peer is rotatable but not a subscription. Check which question a call site is asking before editing either.
- The peer reports a constant neutral status — always available, weekly utilization/reset/window all unknown — with no network call. It is therefore permanently in the elapsed-less ranking block and is selected only when no backend with a real capacity signal is available, including in the default `subscription` mode. Because it is always available it occupies the available pool, so an instance running both peer and bedrock never reaches the bedrock fallback: peer wins over bedrock by design. Peer exhaustion is reactive only — a 429 parks it through `on_rate_limited`; a 5xx, refused connection, or timeout leaves selector state untouched.
- `_refresh_tokens` walks a hardcoded OAuth pair, not the selector pool. Do not rewrite it to walk the pool — that would pull the peer (and openrouter) into token refresh.
