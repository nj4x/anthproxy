# Lock requested model to a configurable baseline before routing

When auto-routing is enabled, the client's requested model is arbitrary (haiku, opus, a full Bedrock ARN, whatever the client was configured with). The routing tier map assumes a sonnet-class baseline: trivial→haiku is a *downgrade*, deep→opus is an *upgrade*. When the client sends haiku, a "standard" classification routes to haiku (no rewrite), but the operator intended standard work to land on sonnet. The tier map was designed with a mid-tier baseline in mind.

We added `lock_requested_model` (default `off`): when set to a non-`off` model string (e.g. `claude-sonnet-4-6`), `prepare_routing()` in `handlers.py` passes it to `route_model()` as a `baseline_model` argument. Inside `route_model()`, the baseline replaces the client's model for all routing math (tier eligibility, classifier input, cap comparisons) while `payload['model']` and `ModelRoutingDecision.requested_model` remain the true client-sent value. At the end of routing, `payload['model']` is overwritten with the final routed model as normal. The lock is applied only when routing is active; if routing is off the field has no effect.

## Considered Options

**Overwrite `payload['model']` before calling `_route_model()`:** This is simpler but breaks the response-echo invariant. `ModelRoutingDecision.requested_model` is the field used by handlers to echo the model back to the client in every response. Overwriting the payload first makes `requested_model` equal the locked value, so clients would receive the lock target in their response regardless of what they sent — violating the documented invariant: *"Routed responses transparently echo the originally requested model."* It also corrupts the `no_classifier` bypass path and the `applied` flag.

**Skip routing when locked (set model and return):** Rejected. The whole reason to lock the baseline is so routing still fires — the classifier judges complexity and can upgrade or downgrade relative to the baseline.

**Consequences:** `ModelRoutingDecision.requested_model` continues to reflect the true client-sent model for response echoing and stats. The `applied` flag correctly reflects whether the final routed model differs from what the client originally sent. The lock baseline is a routing-internal input; it does not surface in the response.
