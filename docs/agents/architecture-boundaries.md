# Architecture boundaries

Ownership and responsibility split across modules; who may touch what.

**Handlers** (`handlers.py`): HTTP routing, local-command interception, model-tier routing orchestration, retries, and response emission.

**Model router** (`model_router.py`): bounded routing-summary extraction, classifier-payload construction, strict label parsing, fail-closed routing decisions, and `ModelRoutingDecision`. Never touches registry, selector, or HTTP transport.

**Mappers**: Protocol shaping and model-alias normalization. Must strip `_anthproxy_internal_classifier` before building outbound bodies, just as they strip `_anthropic_beta`.

**Request text** (`request_text.py`): shared reminder/transcript stripping and last-user-turn recovery for handlers and the model router.

**Backend packages**: Transport and runtime behavior. Preserve the one-way `backend.py` → `mapper.py` dependency.

**Authentication**: Provider `auth.py` modules own provider-specific policy; `_shared/oauth_base.py` owns shared refresh, persistence, locking, and atomic writes.

**Registry and snapshots**: One cached backend instance per backend. Each dispatch uses one immutable snapshot; a retry takes a fresh snapshot. Do not document stronger consistency guarantees than are enforced.
