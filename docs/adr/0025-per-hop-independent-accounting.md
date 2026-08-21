---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Chaining-004)
---

# ADR-0025: Each hop records its own traffic; chain records are not additive

**Date**: 2026-08-20

**Status**: Proposed

**Source SRS**: SRS-Chaining-004

## Context

A request traversing a chain is handled by every hop. Each hop's `SessionDB` writes a request row and a trace; each hop's `StatsCollector` writes a usage line and a cost estimate. Two hops therefore produce two records describing one user request, and because `metadata.user_id` passes through untouched, both records carry the *same* session key — the rows line up perfectly, which is exactly what makes naive summing so easy.

The alternative is to designate one hop authoritative and have the other stay silent, which requires a suppression signal crossing the hop — the shape ADR-0024 just established as a boundary.

## Decision

1. **Every instance records the traffic it dispatched**, including traffic dispatched to a peer. No suppression signal is sent, and none is honoured.

2. **Records from different instances are not aggregated.** No instance is authoritative for chain-wide cost. Each instance's figures are scoped to itself and are locally truthful: the outer instance really did serve that request, and really did dispatch it to `peer`.

3. **The `peer` backend's recorded cost is the outer hop's own estimate**, derived from normalized usage in the response as for any backend. It is not a report of what the peer actually spent.

4. **A failed peer dispatch is still recorded.** A dispatch that returns an error, is refused, or times out produces a request row and a trace at the outer hop like any other dispatch, attributed to `peer`, with no usage fields populated and a null cost estimate. Absent usage is recorded as absent, not as zero: a zero would be summable and would read as a request that genuinely cost nothing, whereas a null says the outer hop never learned. The alternative — recording nothing — would make the most diagnostically interesting traffic in a chain the one kind that leaves no trace, which is backwards. This is the same treatment a failed provider dispatch receives; the peer is not special here.

## Consequences

- **Chain-wide cost is not available from any single instance.** An operator adding the outer's total to the inner's double-counts every chained request. This is the accepted cost of the decision and it must be documented wherever cost is presented — a total labelled "cost" in the outer instance's UI is the cost of traffic *that instance* handled, and chained requests appear in both UIs.
- **The outer's cost figure for `peer` traffic may be wrong in a specific direction.** The outer estimates cost from the response's usage fields against its own pricing table, but the peer may have routed to a different model or a different provider entirely — the outer attributes the spend to `peer` and prices it against whatever model came back. The inner instance's record is the accurate one.
- **Identical session keys across hops make the duplication legible rather than confusing.** Because `metadata.user_id` is unchanged, the same session can be inspected at both hops and lined up request-for-request. What is a hazard for summing is an asset for diagnosis.
- **Each instance remains independently diagnosable.** This is the point: an operator with access to only the inner machine can still see and explain everything it served, with no dependency on the outer's records.
- **Storage and write cost are duplicated per hop.** Accepted; each hop already pays it for its own directly-submitted traffic.

## Considered Options

- **Outer records, inner suppresses** (via an `X-Anthproxy-Record: no` sentinel). Rejected. The outer's record is the less accurate of the two — it does not know the model the inner actually ran or the backend it used — so this designates the worse record as the only one. It also requires a control directive to cross the hop boundary.
- **Inner records, outer suppresses.** Rejected. It makes the outer instance unobservable for the portion of its traffic that leaves for a peer, and its own selector decisions — which are real and which the operator tuned — become invisible.
- **Reconcile after the fact by matching session keys.** Rejected as out of scope. The keys do align (see Consequences), so a reporting tool could be built later; nothing in this decision prevents it, and building it now presumes a chain-wide reporting need that has not been established.
