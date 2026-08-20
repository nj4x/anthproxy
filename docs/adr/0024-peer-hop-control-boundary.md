---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Chaining-003)
---

# ADR-0024: The peer hop is a control boundary; content passes through, directives do not

**Date**: 2026-08-20

**Status**: Proposed

**Source SRS**: SRS-Chaining-003

## Context

Two kinds of thing arrive at an instance: request content and control directives. Content — messages, system prompt, the requested model — describes the work. Directives describe how *this instance* should handle it: `X-Anthproxy-Override` carries `prefer:<backend>`, `route:<mode>`, `task:<name>`, `no-classifier` (`handlers.py:250`, `:908`), and the `proxy-*` local commands carry runtime control in the message text itself (`handlers.py:156`, `:990`).

Directives name things that exist in one instance's configuration. `prefer:codex` is a statement about the receiving instance's backend set; forwarded to a peer with a different backend set, it is at best meaningless and at worst selects something the client never intended. The two kinds must be treated differently at the hop.

## Decision

1. **Control directives are consumed at the hop that receives them and are not forwarded.** The entire `X-Anthproxy-Override` header is consumed by the receiving instance; a peer dispatch carries none of it. Local commands are already intercepted and short-circuited before dispatch (`handlers.py:990`), so they never reach a peer by construction — this ADR records that as intended, not incidental.

2. **Request content passes through unchanged.** The model the client asked for is the model the peer receives — not a tier the outer hop selected for it, and not a substitute the outer hop's context floor imposed. The rest of the payload is forwarded as-is, minus the internal keys already stripped for every backend (`_anthropic_beta`, `_anthproxy_internal_classifier`). Nothing is added: no marker identifies the request as chained.

   This is stated here in its own terms rather than by reference, because the two ADRs have to be readable in either order. ADR-0023 arrives at the same place from the routing side — it suppresses all model-tier routing for peer-bound requests, floor included — and is the fuller argument for *why* the model is untouched. If the two ever disagree, ADR-0023 governs the routing behavior and this clause follows it.

3. **Peer failures propagate as errors.** If the peer rejects the model, has no capacity, or fails for any other reason, that failure is surfaced to the client in an Anthropic error envelope. There is no local fallback to a different model and no substitution of a peer default.

## Consequences

- **A client cannot address the inner instance at all.** No override reaches it, and no `proxy-*` command reaches it — the outer instance intercepts every one. Configuring or querying the inner instance means talking to it directly. This is a real capability gap for anyone expecting to drive a chain from one endpoint, and it is the deliberate price of each hop being independently controlled (FS-Chaining-003).
- **`task:` and `route:` intent is lost at the boundary**, even though those two arguably *could* carry meaning to the peer's own classifier. Forwarding a subset would mean maintaining a per-directive forwarding policy and would reopen the double-routing ambiguity ADR-0023 closes; consuming the whole header is one rule with one meaning.
- **Model errors are honest, without qualification.** The outer hop forwards the client's original model and nothing else — no routed tier, and no floor-substituted model, since ADR-0023 §4 suppresses the floor too for peer-bound requests. So a model rejection from the peer is unambiguously a rejection of what the client actually asked for, and propagating it verbatim is correct rather than merely convenient. Had the outer hop retained even one rewriting rule, this consequence would have had a carve-out: some rejections would be of a model the client never named, and the client would receive an error mentioning a model it had never heard of. If a peer needs alias translation, it is configured at the peer, where its own model configuration already lives.
- **`prefer:peer` still works, and is the intended escape hatch.** A directive naming the peer is meaningful at the outer hop and is honoured there; it is only forwarding that is prohibited.

## Considered Options

- **Forward the override header verbatim** and let the peer's own parser decide relevance. Rejected: `prefer:<backend>` would be validated against the peer's backend set, so the same header means different things at different hops, silently.
- **Strip `prefer:` only, forward `task:` and `route:`.** Rejected: a per-directive policy is a second thing to keep correct as directives are added, and it reintroduces cross-hop routing influence immediately after ADR-0023 removed it.
- **Fall back to a peer default model on rejection.** Rejected. It converts an explicit upstream refusal into a silent substitution, and the client would be billed for a model it did not ask for with no signal that substitution occurred.
