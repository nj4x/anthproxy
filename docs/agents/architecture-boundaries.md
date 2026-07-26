# Architecture boundaries

Ownership and responsibility split across modules; who may touch what.

**Handlers** (`handlers.py`): HTTP routing, local-command interception, model-tier routing orchestration, retries, and response emission.

**Model router** (`model_router.py`): bounded routing-summary extraction, classifier-payload construction, strict label parsing, fail-closed routing decisions, and `ModelRoutingDecision`. Never touches registry, selector, or HTTP transport.

**Mappers**: Protocol shaping and model-alias normalization. Must strip `_anthproxy_internal_classifier` before building outbound bodies, just as they strip `_anthropic_beta`.

**Request text** (`request_text.py`): shared reminder/transcript stripping and last-user-turn recovery for handlers and the model router.

**Backend packages**: A backend is a direct child package of `anthproxy/` containing both `__init__.py` and `backend.py` as regular files. Its `__init__.py` calls `register_backend('<name>', BackendClass)` at import time (self-registration). `build_backend()` constructs every backend via the `from_config(cls, config)` classmethod hook — never by name-dispatch. Optional hooks `model_aliases()` and `summary_credentials()` provide plugin configuration without any public module knowing the backend name. Preserve the one-way `backend.py` → `mapper.py` dependency. **`backend.py` is a reserved filename** — never create it inside `anthproxy/_shared/` or `anthproxy/mapper/`; doing so gives those packages both discovery markers and aborts startup.

**Authentication**: Provider `auth.py` modules own provider-specific policy; `_shared/oauth_base.py` owns shared refresh, persistence, locking, and atomic writes.

**Registry and snapshots**: `backends_registry.py` owns `_BACKENDS`, `_DECLARED_ORDER`, `_assert_registered`, `discover_backends()`, and `BackendDiscoveryError`. No other module may import its underscore-prefixed symbols. One cached backend instance per backend. Each dispatch uses one immutable snapshot; a retry takes a fresh snapshot. Do not document stronger consistency guarantees than are enforced.

**Disabling a discovered plugin**: Move the package outside `anthproxy/` (`mv anthproxy/plugin ~/plugins/plugin`) or break the two-file pair by renaming `backend.py` (`mv backend.py backend.py.off`). Renaming the directory in place is not a supported disable — a valid identifier with both files still qualifies under its new name.
