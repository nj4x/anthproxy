---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Chaining-001, SRS-Chaining-003, SRS-Chaining-006)
---

# ADR-0021: A distinct `peer` backend package for anthproxy-to-anthproxy dispatch

**Date**: 2026-08-20

**Status**: Proposed

**Source SRS**: SRS-Chaining-001, SRS-Chaining-003, SRS-Chaining-006

## Context

Reaching another anthproxy's capacity requires a backend that speaks the Anthropic Messages wire format to an arbitrary host. `local` already does exactly that — it relays native Messages to a configured base URL and passes SSE through verbatim (`local/backend.py:40-59`, `:159-172`) — so the obvious move is to widen `local` with an auth header and model pass-through and call it done.

There is a second, sharper reason to keep the peer distinct, and it concerns the credential. anthproxy already assigns a meaning to inbound `Authorization: Bearer <token>`: `_oauth_credential` (`handlers.py:661-665`) treats any such header as an enterprise OAuth credential the client is delegating, and hands it to `registry.observe_oauth_credential()`. A peer credential sent in that header would therefore not merely be ignored by the receiving instance — it would be absorbed into that instance's OAuth credential state on every single request, and from there into token-derived routing decisions (SRS-Routing-003 engages specifically on the presence of a bearer token). The credential a chain uses to satisfy a network's access-control layer must not be a credential the far end reads as a delegation.

That aside, the `local`-widening reading is also wrong in the one dimension the selector cares about. `local` models an always-available, unmetered endpoint on the operator's desk: it deliberately has no `five_hour_status` (`local/backend.py:122-129`), which means the selector skips it entirely (`selector.py:581-583`), and it collapses every requested model to a single configured alias (`local/mapper.py:23-32`). A peer is the opposite on both counts — it is metered, it is exhaustible, it returns 429, and it must honour the model it is handed. Merging the two makes one backend name that cannot be reasoned about for capacity, because half its instances are free and half are not.

## Decision

Add a new backend package `anthproxy/peer/` following the standard two-file convention (`__init__.py` + `backend.py`, self-registering under the name `peer`, constructed via `from_config`). It reuses `local`'s transport shape and diverges where the semantics differ.

1. **Name is `peer`.** It becomes the directory name, the registered name, and the flag prefix. `peer` is not in `RESERVED_NAMES` (`backends_registry.py:45`) and does not collide with `auto`, `subscription`, or `oauth`. It must be added to `_DECLARED_ORDER` (`backends_registry.py:48`), which carries a completeness assertion at `:320`.

2. **Configuration mirrors the `local` precedent.** `--peer-base-url` / `ANTHPROXY_PEER_BASE_URL`, defaulting to unset. `--peer-api-key` / `ANTHPROXY_PEER_API_KEY`, defaulting to unset. Plain HTTP is preserved for loopback and private-network targets, per the existing rule in `docs/agents/backend-providers.md`.

3. **Setting `--peer-base-url` is what enables the backend.** When it is unset, `peer` is absent from the enabled set and behaves as if not installed. When it is set and `--backends` was not passed, `peer` joins the default enabled set without the operator listing it; when `--backends` *was* passed, it must be listed explicitly, because an allowlist is an exhaustive statement of intent (ADR-0020 §5).

   The converse — `--backends` naming `peer` with no `--peer-base-url` configured — is a hard startup error naming both flags. It is not repaired by dropping `peer` from the allowlist, and it is not repaired by leaving `peer` enabled-but-unreachable. ADR-0020 §5 draws the line at intent: an explicitly named backend is a request, honoured or refused, never rewritten, and naming a backend that has no target is the same class of error as `--backend` naming a backend outside the allowlist. Silently narrowing the allowlist here would also be observably wrong, because the narrowed set feeds ADR-0020 §6's "first enabled backend" repair and could change which backend the instance defaults to.

   `peer` is appended **last** in `_DECLARED_ORDER` (`backends_registry.py:48`). That position is load-bearing rather than cosmetic: ADR-0020 §6 and §8 resolve "first enabled backend" from that order, and a peer is the last thing that should become an instance's repaired default or its fallback pick.

4. **`--peer-api-key` sends, it never checks, and it does not use `Authorization`.** When set, outbound requests carry `X-Anthproxy-Peer-Key: <value>`; when unset, no credential header is sent. The dedicated header is not a style preference — `Authorization: Bearer` is already claimed inbound by `_oauth_credential` (`handlers.py:661-665`), so reusing it would feed the peer key into the receiving instance's `observe_oauth_credential()` on every request (see Context). A header anthproxy does not itself interpret inbound is the only shape that satisfies "sends but never checks" in both directions.

   This exists so a peer can sit behind an existing access-control layer — a reverse proxy, an identity-aware proxy, a mesh with its own authentication — which is the layer expected to consume the header. anthproxy itself grows no inbound authentication (SRS-Chaining-006); see Consequences.

5. **The model is transmitted as received.** No alias collapse, no default substitution. The requested model is the peer's to resolve, and if the peer rejects it, that error is propagated (ADR-0024).

6. **`count_tokens` is proxied upstream** rather than locally estimated. Unlike `local`, a peer implements the real endpoint, so estimating locally would discard an available correct answer — and SRS-Chaining-001 contracts that an operation the peer publicly exposes is served by the peer rather than locally approximated.

## Considered Options

- **Widen `local` with auth, model pass-through, and a `five_hour_status`.** Rejected per Context: it would make one backend name mean two incompatible capacity stories, and the selector's treatment of a backend is derived entirely from that name.
- **Give the `anthropic` backend a base-URL override.** Rejected. Its host is a module constant (`mapper/anthropic_protocol.py:10`) threaded through OAuth-specific headers, `?beta=true` paths, and a 401-triggered token refresh (`anthropic/backend.py:47`, `:104`). A peer needs none of that, and parameterising the host would put "am I talking to Anthropic or to something else?" branching inside the provider backend that is least able to absorb it.
- **Support a list of peers behind the single `peer` name.** Rejected for now. One directory yields one registered name (`backends_registry.py:312`), so N peers behind one name means the backend load-balances internally — which hides each peer's exhaustion state from the selector, the component whose entire job is to reason about exhaustion. The config value is parsed leniently enough that a list form could be added later without a flag rename.
- **Send the peer key as `Authorization: Bearer`,** the header every HTTP client and every fronting proxy already understands. Rejected on the strength of the receiving side: `handlers.py:661-665` classifies that header as an enterprise OAuth credential unconditionally, with no issuer check and no opt-out, so the convenience is paid for by corrupting the far end's credential state on every request. A gateway that requires `Authorization` specifically is a deployment concern its own configuration can bridge; a bearer token silently entering `observe_oauth_credential()` is not something the operator can bridge at all.
- **Peer configuration under `ANTHPROXY_HOME`** as a discovered file. Rejected as premature for a single target; the flag precedent (`--local-base-url`) is the smaller surface.

## Consequences

- **The transport is duplicated, not shared.** `peer` and `local` will contain near-identical connection and SSE pass-through code. Accepted: the duplication is small and it is what keeps the two capacity semantics from contaminating each other. Extracting a shared relay helper later is a mechanical change that does not disturb this decision.
- **anthproxy still has no inbound authentication.** Nothing in `handlers.py` checks a client credential, and this ADR does not add one. A chain is therefore only as private as the network between its hops; the supported deployments are loopback, an SSH tunnel, or a private network. §4 sends a credential but never demands one, which is exactly the asymmetric posture SRS-Chaining-006 contracts for, and it commits the project to no security surface of its own — deliberately, because a checker is a much larger commitment than a sender.
- **`X-Anthproxy-Peer-Key` is inert at the receiving instance, and must stay inert.** Nothing reads it; that is the property that makes it safe. A future decision to *check* it would be adding inbound authentication, which is a different decision from this one and must be taken as such — the danger is a half-step where the header acquires meaning on the inbound side without the surrounding contract (SRS-Chaining-006's second invariant) being revisited.
- **`peer` is a backend that can be enabled by a flag that is not `--backends`.** §3 introduces an implicit membership rule the allowlist does not express. It is confined to the case where `--backends` is absent, which ADR-0020 §6 already establishes as the "operator expressed no intent" branch.
