---
artifact-type: spec
---

## Problem Statement

When a user sends a short affirmation ("yes", "go ahead", "proceed"), the current routing code inherits a cached session tier if available, or floors to the configured standard tier. However, when no cached tier exists, the affirmation text alone ("yes") classifies as trivial, misrouting the upcoming complex work to a low-capability model. Additionally, the session tier cache remains empty, so subsequent tool-result turns in the same session will re-classify from scratch rather than inheriting the established complexity from the prior planning turns.

This leads to two failures: (1) the affirmation turn itself routes to the wrong tier, and (2) downstream tool-result turns lose the tier context they should have inherited, degrading both latency and model selection accuracy.

## Solution

Enrich the affirmation code path to call the classifier when no cached tier exists, injecting the prior assistant response (task description) into the classifier input via the `prior_response_summary` field. The classifier then sees the context of what the user is agreeing to (e.g., "I'll implement X by…") alongside the affirmation; the amended system prompt instructs the classifier to prioritize `prior_response_summary` when present, producing an accurate tier that is written to the tier cache. This single classifier call establishes the tier for all subsequent tool-result turns in the same session.

When a cached tier is already available, the existing early-return path is preserved unchanged (no additional classifier call). When no cache exists and prior-response enrichment is unavailable (e.g., prior message contains only tool-use blocks), the classifier receives just the affirmation text and writes the result (typically trivial) to the tier cache.

## User Stories

1. As a user planning a complex refactor, I send a detailed description and ask the assistant to "plan the implementation". The assistant responds with an architecture overview. I then send "yes, proceed". Without this feature, the affirmation turn routes to haiku (trivial) and wastes the planning tier. With the fix, the affirmation turn sends both "yes" and the assistant's overview to the classifier, which accurately routes the turn to sonnet (or higher) and caches the tier for tool-result turns. (Cache seeding requires a valid context key; sessions without `metadata.user_id` receive correct per-turn routing but do not seed the tier cache for downstream turns.)

2. As a user in a session with a cached tier, I send "go ahead". The code inherits the cached tier immediately without an additional classifier call, preserving zero-overhead routing on cached sessions.

3. As a user whose prior assistant message contains only tool-use blocks (no text content), I send an affirmation. The classifier receives just the affirmation text and returns a tier (typically trivial); that tier is written to the cache, and subsequent tool-result turns inherit it from the cache instead of re-classifying.

4. As a user whose session opens with an immediate affirmation (first turn is "yes"), no prior assistant message with text exists. The affirmation path skips the classifier, uses the floor tier for this turn, and does **not** write to the tier cache. Subsequent non-affirmation turns (e.g., a tool-result turn) will attempt classification normally and establish a tier. This prevents the rejected "bare affirmation classification" scenario from activating.

5. As a system admin, I can configure `auto_model_routing_prior_response_summary_limit` (default 1000 characters) to control how much of the prior assistant response is sent to the classifier, preserving task preamble and final state while capping mid-body boilerplate.

6. As a system operator, I can observe via logging and telemetry that affirmation turns with no cached tier now invoke the classifier with enriched input (prior_response_summary field), allowing me to tune the limit or debug routing accuracy.

7. As a developer implementing the feature, I must handle both content forms in the prior assistant message: plain string and list of content blocks. Malformed content (non-dict items in the list, missing or non-string `text` fields) gracefully falls back to empty string and does not crash the request.

8. As a developer, I must ensure the classifier call succeeds before writing to the tier cache. If the classifier fails (network error, timeout, invalid response), the floor tier is used for this affirmation turn only, and the tier cache remains empty so the next non-affirmation turn can re-attempt classification.

## Implementation Decisions

### Routing Logic Restructuring

The affirmation-handling code path in `route_model()` will be restructured into two branches:

1. **Cached tier path**: check for `cached_session_tier is not None`; if found, inherit immediately and return early (preserving the model-routing.md invariant, no classifier call).
2. **No-cache path**: when `cached_session_tier is None`, extract `prior_response_summary` from `payload.get('messages')` directly inside `route_model()` at this branch (not in `build_routing_summary()`), then call the classifier with enriched input. On success, return a **non-None** `classification` field so the existing handler write at `handlers.py:726–728` fires naturally and stores the result under the composite context key `(session ID, stable first-user-message hash)`. On classifier failure, use the floor tier and return `classification=None` to suppress the cache write.

Both branches cap the result against the requested model (or baseline lock model) to enforce the no-upgrade invariant.

**`routed_model` and `cache_tier` derivation for the `affirmation_classified` path**: The classifier label is resolved to a tier model via `config.auto_model_routing_classification[classification]`. If `baseline_model` is set, apply `_cap_cached_tier()` to the resolved tier model to produce `routed_model` (the capped value used for this turn's dispatch). `cache_tier` is set to the **uncapped** resolved tier model — `config.auto_model_routing_classification[classification]` before capping — so subsequent turns apply their own cap when reading from the cache. In `handlers.py`, the tier-cache write at line 728 must check the reason code: for `affirmation_classified`, write `routing.cache_tier or routing.routed_model` (uncapped tier to cache); for all other paths where `classification is not None`, write `routing.routed_model` as before (preserving existing behavior; `cache_tier` is `None` on all non-affirmation paths). This ensures affirmation-classified tiers bypass the baseline cap before caching, allowing downstream turns to apply their own cap.

**Invariant dependency**: The cache write at `handlers.py:726–728` is gated on `classification is not None`. The no-cache affirmation path must return a non-None `classification` on success to trigger this write. Returning `classification=None` on classifier failure is correct and suppresses the write.

### Prior-Response Extraction and Classifier Input

`prior_response_summary` is NOT added to `RoutingSummary`. It is a routing-internal signal computed directly inside `route_model()` at the no-cache affirmation branch, after the cached-tier check (where `cached_session_tier` is known to be `None`).

**Critical invariant:** Do not classify bare affirmation text when no prior assistant message with text content exists. This prevents the "Classify with just the affirmation text" scenario from activating.

The extraction logic (executed inside `route_model()`, no-cache affirmation branch only):
1. Walk backward through `payload.get('messages', [])` (safe when `messages` key is absent) to find the most recent `role='assistant'` message with text content.
2. For each assistant message, check if `content` is a `str`: if so, collect it. If `content` is a list, iterate over elements (with `isinstance(element, dict)` guard), extract text using `element.get('text') or ''`, collect only text from blocks where `element.get('type') == 'text'`; skip all other types. Concatenate collected text blocks with `'\n'` separator.
3. If the assistant message has no text content (empty string, no text blocks, missing/None `content` key), continue walking backward to the next assistant message.
4. Return the first non-empty text collected. **If no text-bearing assistant message is found, do not call the classifier; use the floor tier and return `classification=None` (no cache write).**

**Truncation strategy (30/70 head/tail split):** When the extracted text exceeds the configured limit, compute `head_len = int(limit * 0.30)` and `tail_len = limit - head_len - len(_TRUNCATION_MARKER)`. For plain-string content, slice directly: `text[:head_len] + _TRUNCATION_MARKER + text[-tail_len:]`. For list content, use a rolling tail buffer (O(limit) memory) during extraction. The head (30%) preserves the task preamble; the tail (70%) preserves the final state. When the full text fits within the limit, return it whole without a separator.

### Classifier Payload Injection

`to_classifier_json()` gains a `prior_response_summary: str | None = None` parameter. When `prior_response_summary` is non-None, the field is merged into the routing-summary dict **before** `json.dumps()` — no deserialization/re-serialization required. At the call site in `route_model()`'s no-cache affirmation branch, pass `prior_response_summary=extracted_summary` when non-empty. This eliminates the need for a general `extra_fields` mechanism and its key-collision guard.

**Privacy justification**: Unlike `build_routing_summary()`, this extraction intentionally reads the prior assistant turn because it is the proxy's own output, already processed by the system, carrying no user-originating sensitive data beyond what was already classified in the prior planning turn.

### Classifier System Prompt and Parser Selection

**System prompt and parser consistency**: The affirmation-enriched classifier call uses the same system-prompt and parser logic as the normal (non-affirmation) routing path. When `config.auto_model_routing_confidence_bump` is True, use `_CLASSIFIER_SYSTEM_JSON` and parse structured JSON output (with confidence score extraction). When False, use `_CLASSIFIER_SYSTEM` and parse the one-word label. Do **not** override the `confidence_bump` flag for affirmation calls — this ensures the prompt/parser pair remains consistent and the output format matches the parser expectation.

Prior-response enrichment (the `prior_response_summary` field) is passed to the classifier regardless of which prompt is used.

**System prompt amendment**: When `confidence_bump` is True and `_CLASSIFIER_SYSTEM_JSON` is used, append this instruction:

> When prior_response_summary is provided, judge complexity from the context in prior_response_summary; it describes what the user is agreeing to proceed with.

When `confidence_bump` is False, the base `_CLASSIFIER_SYSTEM` is used unchanged; it treats `prior_response_summary` as harness context (similar to tool counts) and judges complexity from `final_user_text` alone.

### Configuration

Add a new config option:
- `auto_model_routing_prior_response_summary_limit` (int, default 1000)
- CLI flag: `--auto-model-routing-prior-response-summary-limit`
- Environment variable: `ANTHPROXY_AUTO_MODEL_ROUTING_PRIOR_RESPONSE_SUMMARY_LIMIT`
- Both follow the existing routing-knob pattern in `config.py` (int-parsed env var, CLI int argument, Config dataclass field).
- Startup validation must assert the limit is in the range `[50, 32_000]`; raise a `ValueError` (consistent with existing validation in `config.py`) if the value is below 50 or exceeds 32,000.
- The minimum of 50 ensures the truncation formula (`tail_len = limit - int(limit*0.30) - len(_TRUNCATION_MARKER)`) produces non-negative tail_len. At 50: head=15, tail=16 ✓; below 28 tail becomes negative.
- An upper bound of 32,000 characters guards against sending excessively large classifier payloads.
- The feature is disabled by setting `auto_model_routing_affirmation_inherit = false`, not by adjusting the limit.

**`affirmation_inherit=False` behavior**: When `auto_model_routing_affirmation_inherit` is `False`, the entire affirmation-enrichment path is skipped. The `affirmation_classified` and `affirmation_classifier_failed` reason codes become unreachable.

### Tier Cache Semantics and Concurrency

The composite context key (session ID + first-user-message hash) is already used elsewhere for tier caching. Affirmation turns with no cached tier will create the first cache entry for that context key, allowing subsequent tool-result turns to inherit the tier without re-classifying.

`ModelRoutingDecision` gains a `cache_tier: str | None = None` field. For the `affirmation_classified` path, `cache_tier` holds the uncapped resolved tier and `routed_model` holds the capped value. All other paths leave `cache_tier=None`. In `handlers.py`, the tier-cache write must distinguish the `affirmation_classified` reason code: write `routing.cache_tier or routing.routed_model` (uncapped) for that code; for all other paths, write `routing.routed_model` as before.

If the classifier call fails, the floor tier is used for this affirmation turn only; `classification=None` suppresses the cache write; the next non-affirmation turn will attempt classification normally.

**Concurrent affirmation mitigation:** When two simultaneous affirmations on the same session both read an empty cache (TOCTOU race), both invoke the classifier and both attempt to write. To prevent this race, use a per-ctx_key in-flight sentinel: (1) before dispatching the classifier, check if an in-flight sentinel exists at `ctx_key` (e.g., a reserved marker distinct from actual tier strings); if present, skip the classifier call and use the floor tier instead. (2) If no sentinel is present, set the sentinel and dispatch the classifier. (3) After the classifier returns, clear the sentinel and write the result. This ensures only one concurrent affirmation call per context key reaches the classifier and writes to the cache.

**`ctx_key is None` gate**: Sessions without `metadata.user_id` produce a `None` context key, so the tier cache write is skipped even when classification succeeds. Such sessions receive correct per-turn routing but the tier is not persisted for downstream turns.

### Latency and Concurrency Implications

- **Cached tier (affirmation or non-affirmation)**: no additional latency (early return, classification=None).
- **Affirmation with no cached tier**: one classifier round-trip (with enriched input when prior text is available).
- **All other turn types (non-affirmation)**: no additional latency from this feature.
- Concurrent affirmation misses are mitigated via in-flight sentinel: only one classifier call per context key reaches the classifier; others use the floor tier.

### Logging and Telemetry

Affirmation turns will report in telemetry with:
- `classifier_mode: 'affirmation'` (existing field, already in use).
- Reason codes for affirmation paths:
  - `affirmation_inherited` (cached tier, no classifier call) — already in use, unchanged.
  - `affirmation_classified` (no cache, classifier called, result written to tier cache) — **new**.
  - `affirmation_classifier_failed` (classifier error, fallback to floor, cache remains empty) — **new**; must be added to the `ReasonCode` Literal in `model_router.py`.
  - `affirmation_floored_standard` is retired from the affirmation code path. Existing uses in non-affirmation floor scenarios are not affected.
- `admin.py`'s `affirmation_count` KPI set must be updated to include `affirmation_classified` and `affirmation_classifier_failed`. `affirmation_floored_standard` is retained in the set to handle old stats rows across the migration boundary.

## Testing Decisions

### Seams and Test Boundaries

Tests should target the boundary between `route_model()` and the classifier transport layer. Existing seams:
1. The `snapshot` parameter to `route_model()` — allows injecting a mock config.
2. The `cached_session_tier` parameter — allows injecting `None` to force the no-cache path.
3. The `ctx_key` parameter — allows specifying the context key for logging.
4. The payload `messages` field — can be constructed to test text extraction.

No new transport seams are required; the classifier transport is already mocked in existing tests.

### Test Coverage

1. **Cached tier (affirmation)**: verify early return without calling the classifier.
2. **No-cache affirmation with prior response**: classifier called with `prior_response_summary` injected; result written to `cache_tier` (uncapped); `routed_model` is capped.
3. **No-cache affirmation, no prior text in last assistant message**: backward walk continues to find a text-bearing assistant message. If found, classifier called with enriched input. If not found, floor tier used; `classification=None`; no cache write.
4. **Session-opening affirmation (no prior message)**: no prior assistant message with text exists; floor tier used; `classification=None`; no cache write.
5. **Text extraction (string content)**: plain-string `content` extracted directly, 30/70 split applied.
6. **Text extraction (list content)**: list of content blocks iterated; text blocks collected (skipping tool_use, thinking, etc.); concatenated with `'\n'`; 30/70 split applied.
7. **Malformed content (non-dict items)**: non-dict items in the content list gracefully skipped.
8. **Classifier failure (network error)**: floor tier used; `classification=None`; tier cache not written.
9. **Config validation**: `limit < 50` raises `ValueError`; `limit > 32_000` raises `ValueError`.
10. **Baseline lock**: `cache_tier` is uncapped; `routed_model` is capped.
11. **No upgrade cap**: no-upgrade cap applied correctly when `baseline_model` is not set.
12. **`routed_model` derivation**: `routing.routed_model` equals capped tier; `routing.cache_tier` equals `config.auto_model_routing_classification[label]` (uncapped).
13. **`ctx_key=None` gate**: classification returns non-None and routes correctly; cache write not called.
14. **JSON mode forcing**: affirmation-enriched calls use `_CLASSIFIER_SYSTEM_JSON` regardless of `confidence_bump` setting.
15. **`to_classifier_json` prior_response_summary**: when non-None, `prior_response_summary` is merged into the dict before `json.dumps()`.

Existing test `test_affirmation_floors_to_standard_when_uncached` must be replaced with tests #2, #3, and #8 above.

## Out of Scope

- Changes to the main classifier system prompt (only `_CLASSIFIER_SYSTEM_JSON` is amended).
- New classifier models or endpoints.
- Database writes or persistence beyond the in-process tier cache.
- Changes to the session context-key composition or hash function.
- Interactive CLI or UI controls for affirmation routing.
- Fallback to DB-persisted response text when `payload['messages']` is unavailable.
- ADR 0010 weighted-classifier interaction: deferred until ADR 0010 is implemented in code.

## Further Notes

### Model-routing.md Update

The invariant in `docs/agents/model-routing.md` must be updated from:

> Short direct user affirmations inherit the cached tier when available; without one they use the configured floor. They must not overwrite the tier cache.

To:

> Short direct user affirmations inherit the cached tier when available; without one, call the classifier — with prior-response context if available, otherwise with the bare affirmation text — and write the result to the tier cache to serve subsequent turns. Fall back to the configured floor only when the classifier call itself fails.

### Naming and Terminology

- "prior_response_summary" is the name used in ADR 0013 and should be used consistently in code, config, and logging.
- "affirmation enrichment" is used informally to describe the feature.

### Open Questions / Future Work

The ADR considers a DB-fallback option ("Read last response from DB (`response_text`)") for cases where `payload['messages']` is unavailable. Deferred in favour of the in-process extraction approach.
