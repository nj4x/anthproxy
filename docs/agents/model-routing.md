# Model-tier routing and session identifiers

Auto-routing rules, classifier isolation, tier-cache behavior, and session-key semantics.

Refer to `config.py`, `model_config.py`, and `model_router.py` for mutable configuration, routing modes, model mappings, and operator controls.

- Auto routing applies only to message requests; all other endpoints and local commands bypass it.
- **Peer-bound requests are not routed at all** (ADR-0023). The handler snapshots the dispatch target first, then routes; when that target is the `peer` backend it skips `route_model` entirely and transmits the client's model verbatim. Authority rests with the innermost hop, which classifies against its own configuration. Suppression is total — no classifier, no long-context floor — and per-request: the next request routes normally if it resolves to a provider. **Outer-hop routing configuration is therefore inert for peer-bound traffic**; tuning it there has no observable effect. The suppression is logged and carries reason code `peer_hop_suppressed`, distinct from `disabled` and from a classifier run that chose the same model. A suppressed request also reads and writes no routing session state, so a session alternating between peer and provider accumulates tier-cache entries and context observations only from its provider turns; a long peer-served stretch leaves the cache cold and the next provider turn classifies from scratch. That is intended, not a cache bug. A retry that crosses the provider/peer boundary re-derives the decision against the fresh snapshot from the client's original model, captured once at request start.
- Any non-empty string model is eligible; missing or non-string model values pass through unchanged.
- **Model baseline lock** (`--lock-requested-model <model>` or env `ANTHPROXY_LOCK_REQUESTED_MODEL`): when set to a non-`off` value, that model becomes the baseline for routing decisions (tier eligibility, caps, classifier log context) instead of the client's arbitrary requested model. The client's original model is still echoed in responses, statistics, and the `applied` flag; the lock affects only routing logic internally. Default: `off` (no lock).
- Routing failures, invalid classifier output, unknown task tags, and malformed payloads must preserve the original requested model.
- Routed responses transparently echo the originally requested model; non-routed responses are byte-for-byte passthrough. Statistics retain the routed model.
- Classifier payloads carry `_anthproxy_internal_classifier`; `route_model()` must no-op for them, and mappers must remove the sentinel before outbound transport.
- The long-context size floor precedes classification and cache decisions. When it applies, it must bypass classifier evaluation and tier-cache reads/writes; its target and beta handling are configuration-driven.
- Context observations and routed-tier cache entries use a context key composed of the session ID and a stable first-user-message hash, not the bare session key. Backend and routing-mode session preferences use the bare session key.
- Record each response's context floor from distinct normalized usage fields. Across SSE events, track usage fields by maximum rather than summing cumulative re-statements.
- The tier cache is a last-resort fallback for missing final-user text. Cached tiers must not upgrade above the actual requested tier and must fail open for non-tier model identifiers.
- Classification input is bounded and excludes system prompts, tool schemas, thinking, effort, history, metadata, and headers. Text recovery prefers the final user turn, then transcript recovery, then bounded walk-back.
- Short direct user affirmations inherit the cached tier when available; without one, call the classifier — with prior-response context if a text-bearing assistant message exists, otherwise skip the classifier entirely — and write the result to the tier cache to serve subsequent turns. Fall back to the configured floor only when the classifier call itself fails or no prior assistant text is available.
- Classifier traffic is isolated from user-visible statistics, automatic backend switching/retry, and provider kill-switch behavior. Never hold registry or selector locks across classifier network calls.
- Planning, design, and outline requests must not be classified below the configured standard floor.

## Session identifiers

The session key is the full `metadata.user_id` JSON blob, not a bare UUID. Use its nested `session_id` only for human-facing labels; do not infer device IDs, scan for arbitrary UUIDs, or slug key names.
