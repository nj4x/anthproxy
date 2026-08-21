---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Chaining-005, SRS-Chaining-007)
---

# ADR-0026: Startup self-reference rejection is the only loop guard

**Date**: 2026-08-20

**Status**: Proposed

**Source SRS**: SRS-Chaining-005 (satisfied), SRS-Chaining-007 (knowingly not satisfied)

## Context

A peer target pointing at the configured instance's own listening address makes every request recurse. Each hop sees a well-formed request and cannot tell it apart from legitimate traffic, so nothing stops it: one client request becomes an unbounded amplification, with a billed classifier call at whichever hop holds authority and a database write at every hop (ADR-0025).

The general defence is a per-request cycle marker — each instance stamps an ID into a `Via`-style header and refuses when it sees its own. It catches cycles of any length, at any time, including ones formed after startup.

The specific defence is a startup check: refuse to boot when the peer target resolves to our own bind address.

## Decision

**Validate at startup only.** When `--peer-base-url` is set, the instance rejects a target resolving to its own listening address and refuses to start, naming the flag. No per-request marker is sent, and no inbound header is inspected for cycle detection.

**Only a positive match is fatal.** The check resolves the configured host in order to compare it against the bind address, and that resolution can fail — the peer's DNS name may not exist yet, the host may be down, the resolver may be unreachable. A resolution failure is **not** a boot failure: it is logged as a warning naming `--peer-base-url` and startup proceeds. Startup is refused only when the target successfully resolves to an address this instance is bound to.

This is the asymmetry the check needs. A failed resolution is an absence of evidence, and the hazard being guarded against is an unbounded self-loop — refusing to boot on "I could not tell" would convert a peer that is merely not up yet into an outage of the instance that depends on it, which is a strictly worse failure than the one being prevented. It also preserves the ordering property operators rely on when bringing a chain up: the outer instance can start before the inner one exists.

Multi-instance cycles (A→B→A) are explicitly out of scope, as is a self-reference that becomes true after startup. Both are within the scope of SRS-Chaining-007, which this ADR does not satisfy; that requirement is recorded as deferred rather than scoped away, so the gap remains visible in the requirements corpus.

## Consequences

- **The overwhelmingly common failure is caught, at the cheapest possible moment.** A typo'd port or a copy-pasted URL is a boot failure with a message naming the flag, not a runaway bill.
- **A real cycle is undetected and unbounded.** A→B→A is a supported configuration as far as this ADR is concerned, and it will amplify until something else fails. This is a knowingly accepted gap, and it is carried in the requirements corpus as SRS-Chaining-007 rather than as a footnote here, so that a reader auditing what the system contracts for finds an unmet requirement rather than an unmentioned one. It is narrow because it requires two instances deliberately configured to point at each other, and it is the kind of gap that surfaces immediately and loudly in testing rather than subtly in production — but it is real, and if chains longer than two hops become normal, the per-request marker is the follow-up.
- **Late-forming self-reference is undetected.** If the peer is restarted and its listener moves onto our address, the check has already passed. Same rationale: narrow, and an environment where that can happen has larger problems.
- **No new header surface.** The alternative would have required the peer backend to send a header and `handlers.py` to read and act on one, which is the first inbound header anthproxy would treat as a control input from a *peer* rather than from a client. Not adding it keeps the inbound surface exactly as it is.
- **`--peer-base-url` gains a startup-time resolution attempt, but not a startup-time resolution requirement.** The check resolves the configured host to compare against the bind address; a peer that is simply not up yet does not prevent boot, because only a target that resolves *to us* is fatal. Relative to the lazier `--local-base-url` this adds a boot-time DNS lookup and a warning path, not a new precondition.
- **A self-loop behind an unresolvable name at boot slips through.** If the target does not resolve during the check but resolves to our own address later, the check has passed on incomplete information and nothing catches it afterwards. This is the price of the non-fatal branch, and it is the same shape as the late-forming case above: narrow, and preferable to making every dependent instance's boot contingent on its peer's DNS.

## Considered Options

- **Per-request instance-ID marker, in addition to the startup check.** Rejected for the first cut, not on merit — it is the correct general mechanism and it catches everything this decision misses. Deferred because it commits to an inbound control header, and because the failure it uniquely catches (a genuine multi-instance cycle) requires deliberate misconfiguration of two machines. Revisit if chains of three or more become common.
- **Treat an unresolvable peer target as a boot failure too.** Rejected. It is superficially the stricter, safer reading — "refuse to start on anything you cannot verify" — but it makes the outer instance's availability depend on the inner instance's name resolving, which turns an ordinary bring-up ordering into an outage and gives operators a reason to remove the check. The check exists to prevent an unbounded self-loop; it should fire on evidence of one, not on the absence of evidence.
- **Hop-count header with a maximum.** Rejected. It bounds a cycle rather than preventing it, so a 2-cycle still churns until the counter trips, and the bound has to be guessed.
- **No guard at all.** Rejected. The self-loop is a single-character configuration error with an unbounded blast radius; that combination is what a startup check is for.
