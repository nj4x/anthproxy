---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Chaining-002)
---

# ADR-0023: The innermost hop holds model-tier routing authority

**Date**: 2026-08-20

**Status**: Proposed

**Source SRS**: SRS-Chaining-002

## Context

Both hops of a chain run the full pipeline. Left alone, a single user request is classified twice: the outer instance classifies the user text, rewrites `sonnet` to `haiku`, and dispatches; the peer receives `haiku`, classifies the *same* text again against its own configuration, and routes it wherever its own thresholds point. Two billed classifier calls per request, and two decisions that can disagree.

The model echo compounds it. Routed responses transparently echo the originally-requested model, so the peer echoes `haiku` — its "original" — and the outer instance echoes that back as the answer to a `sonnet` request. The client's model is preserved by luck of the outer's own echo, but the peer's stats record `haiku` as a client-requested model that no client ever requested.

One hop must own the decision. The hop that holds the credentials is the hop that knows the real economics of its own backends, and it is the hop whose routing configuration the operator tuned against those backends.

## Decision

1. **Model-tier routing authority rests with the innermost hop** — the one dispatching to a provider rather than to a peer. When an instance resolves its dispatch target to a peer backend, it performs no model-tier routing for that request and transmits the model exactly as its client requested it.

2. **Suppression is per-request and target-conditioned**, not a mode. The same instance routes normally for the next request if that one resolves to a provider backend.

3. **No dispatch reordering is required; the routing call becomes conditional.** An earlier reading of this design assumed the pipeline ran routing-then-snapshot, following `CLAUDE.md`'s architecture prose ("applies routing, takes one registry snapshot"). The code does the opposite: the snapshot is taken at `handlers.py:672`, and `_route_model` is not called until `handlers.py:745`. `route_model` is even handed that snapshot as a parameter already (`model_router.py:1493-1496`), so the resolved target is in scope at the moment routing runs.

   The change is therefore local and small: guard the `_route_model` call at `handlers.py:745` on `snapshot.name`. Nothing about the order of operations moves, and the prose in `CLAUDE.md` should be corrected to describe the order the code actually has.

4. **Suppression is total: no classifier and no long-context floor.** For a peer-bound request the outer hop applies no model-tier rule at all. The peer applies its own floor to the request it receives, against its own configuration and its own backends' context limits.

   Two reasons, one mechanical and one substantive. Mechanically, the long-context size floor lives *inside* `route_model` (`model_router.py:1583-1614`) and rewrites `payload['model']` there, so a decision to not call `route_model` is necessarily a decision to not apply the floor; "suppress classification but keep the floor" would mean calling `route_model` in a mode that does half its job, which is a new mode, not a suppression. Substantively, that mode should not exist here anyway: a floor is a judgement about what the *serving* backend can hold, and the outer hop is by construction not serving this request. Its floor target may name a model the peer cannot reach; its context limits are its own backends'. Deferring to the peer is deferring to the hop that knows.

   This also makes §1's promise literal. "The model exactly as the client requested it" is true without qualification, which is what ADR-0024 §2 needs in order to say request content passes through unchanged.

5. **A retry that crosses the provider/peer boundary re-derives the routing decision.** The routing decision belongs to the snapshot that will actually serve the request, so an attempt that lands on a different target class than the previous one derives it afresh rather than inheriting it. Concretely, in both directions:

   - **Peer → provider.** The first attempt was not routed, so the retry routes it: classifier and floor both run against the client's original model, as they would for any directly-served request. Carrying the first attempt's non-decision forward would silently disable routing for a request the instance is now serving itself.
   - **Provider → peer.** The first attempt produced a routed model. The retry discards it and forwards the client's original model to the peer. Carrying the routed model forward would send the peer a model the client never asked for, reintroducing exactly the misattribution described in Context — the peer would record and echo a model of the outer hop's invention.

   Deciding once on the first attempt would leave one of these two failure modes live whichever way it was decided. Deriving per attempt against the serving snapshot closes both, and matches the existing rule that a retry takes a fresh snapshot.

## Consequences

- **The change lives entirely in `handlers.py`, which is where the boundary docs already put it.** `model_router.py` is specified as never touching the registry or selector (`docs/agents/architecture-boundaries.md`), so it could not make this decision itself even in principle; the orchestration seam is the right place, and no new coupling is introduced. Non-peer traffic reaches `_route_model` on an unchanged path with unchanged arguments.
- **A peer-bound request loses the outer hop's long-context protection, and the operator has to trust the peer to have its own.** If the peer is configured with no floor, or with a floor whose target cannot hold the request, an oversized request that the outer hop would have caught now fails at the peer instead. That failure is at least honest — it comes from the hop that would have had to serve the request — but it is a real behavior change for anyone who was relying on the outer floor as a global safety net. It is the direct cost of §4, accepted because a floor imposed by a non-serving hop is a guess.
- **The peer's routing configuration is inert for chained traffic in one direction only.** Chained traffic reaching the peer is classified by the peer, as normal. It is the *outer* instance's routing configuration that goes unused for peer-bound requests. An operator who tunes routing on the outer instance and watches it have no effect on the traffic that leaves for the peer will read this as a bug; it must be documented, and ideally logged.
- **Exactly one classifier call per user request, at the hop with the real cost model.** This is the payoff, and it holds for chains longer than two.
- **Suppression is invisible in the request the peer receives.** No sentinel or header marks a request as chained — the peer simply receives an unrouted request and treats it as one, which is exactly what it is. Nothing new needs to be stripped outbound.

## Considered Options

- **Outer holds authority; send `X-Anthproxy-Override: no-classifier` to the peer.** Rejected, though it was nearly free — the header exists and is already parsed (`handlers.py:250`). It puts the routing decision at the hop that does not know which backend will ultimately serve the request, and it requires the outer to forward a control directive across the hop, which ADR-0024 establishes as a boundary. It was also argued for on the grounds that it avoided a dispatch reordering; §3 establishes there was no reordering to avoid, so it has no advantage left to weigh.
- **Let both hops classify.** Rejected. Doubling a billed model call per request is the cost the chain would impose on every user, and the disagreement between the two decisions is silent.
- **Suppress the classifier but keep the outer hop's long-context floor.** Rejected per §4. It requires `route_model` to grow a partial mode, since the floor is inside it (`model_router.py:1583-1614`), and the mode would exist to let a hop that is not serving the request make a capability judgement about a backend it does not own.
