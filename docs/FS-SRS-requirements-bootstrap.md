---
artifact-type: requirements-bootstrap
lineage-rules:
  - "Each SRS item must reference at least one Source FS via a 'Source FS' field"
  - "Each ADR that realizes routing contracts must reference at least one Source SRS via a 'Source SRS' field"
  - "SRS items state contracts only; realization details (formulas, thresholds, module names) live in the implementing ADR"
---

# Bootstrap FS and SRS for anthproxy routing decisions

This document bootstraps the requirements corpus for anthproxy's backend-selection and routing system. It captures product outcomes (FS) and system contracts (SRS) that govern operator control of routing behavior.

---

## FS-Routing-001: Backend utilization should be window-agnostic

**Product outcome**: Operators select between backends based on remaining headroom and runway, independent of the window length (5-hour, weekly, monthly, unlimited). A backend at 50% utilization has the same headroom semantics whether the window resets in 5 hours, 7 days, or 30 days.

**Constraint**: The routing decision must reflect time-until-exhaustion, not raw consumption percentage across heterogeneous window lengths.

---

## SRS-Routing-001: Backend selection shall normalize utilization across window lengths using pace delta

**Source FS**: FS-Routing-001

**System contract**: When the auto-backend selector ranks or gates subscription backends for dispatch, it shall compare them by a window-agnostic normalization of utilization such that:

- A backend whose utilization is **ahead of linear pace** (consumed a larger fraction of its quota than the fraction of its window that has elapsed) is **deprioritized** relative to a backend whose utilization is **behind linear pace**.
- A switch from the currently-preferred backend to a challenger requires a **stability margin** so routing does not flap when the two backends are near parity.
- The comparison is well-defined for every backend with a time-based reset window, regardless of window length (5-hour, weekly, monthly).

**Scope and Applicability**:
- Applies to the OAuth-vs-personal backend selection gate.
- Applies to personal-vs-personal candidate ranking within the auto-selector.
- Does **not** apply to backends with no time-based reset (e.g., hard credit caps); those are compared on raw utilization only.
- Hard eligibility gates (cooldown, spend-cap, health, staleness) remain absolute vetoes; the pace normalization never resurrects an ineligible backend.

**Invariants**:
- Unknown or stale reset data shall yield a neutral comparison value, preserving incumbent-hold semantics.
- Incumbent-hold and hysteresis behavior shall not be weakened by the normalization.
- The stability margin shall be operator-tunable via configuration.

---

## FS-Routing-002: Enterprise OAuth token quota should align with UTC calendar boundaries

**Product outcome**: The enterprise spend cap enforces monthly resets at UTC midnight on the 1st of each month. Routing decisions that reference the spend-cap should operate in UTC time to avoid inconsistency with the actual reset boundary.

**Constraint**: Local time derivations (e.g., client-side UI pace heads) may differ from server routing decisions when the request crosses a UTC day boundary.

---

## SRS-Routing-002: Monthly quota elapsed shall be derived in UTC

**Source FS**: FS-Routing-002

**System contract**: When computing the elapsed fraction of the enterprise monthly quota for routing decisions, the server shall derive it in UTC so that the elapsed computation aligns with the spend-cap reset boundary (UTC midnight on the 1st of each month). Local-time derivations are permitted only in client-side UI surfaces that explicitly document their divergence from the routing-time computation.

---

## FS-Routing-003: Expiring enterprise capacity should be consumed rather than forfeited at monthly reset

**Product outcome**: Enterprise OAuth monthly capacity is a prepaid allotment that resets at UTC month boundary. Unspent capacity is forfeited and represents a sunk cost; routing should actively consume the monthly allowance rather than protect an underutilized subscription backend at the expense of enterprise quota expiration.

**Constraint**: When an enterprise token is eligible and behind its own monthly pace, it should receive traffic priority over a flat-fee subscription backend that is also behind its own pace but whose underutilization has no cost consequence.

---

## SRS-Routing-003: Expiring-quota precedence gate for enterprise backends

**Source FS**: FS-Routing-003

**System contract**: When the auto-backend selector evaluates an enterprise OAuth token backend that carries a time-bounded expiring quota (monthly reset), it shall apply a precedence gate: if the token's quota utilization is behind its linear monthly pace, the token shall take priority over other backends for dispatch, subject to hard eligibility gates (cooldown, spend-cap, health, staleness).

- The gate threshold is operator-tunable via `auto_backend_oauth_pace_deadband_pp` (the same stability margin used in pace-delta ranking).
- The gate is exempt from the stability-margin hysteresis invariant of SRS-Routing-001 ("A switch from the currently-preferred backend to a challenger requires a stability margin"), because the gate operates on a single-backend absolute test rather than a multi-backend comparison. Flapping near the gate threshold is mitigated by the usage-probe cadence (typically 300s between utilization updates); real-world crossing frequency is bounded by probe refresh, not per-request.
- The gate never resurrects a backend that fails hard eligibility gates.

**Applicability**: Applies only when the inbound request carries an `Authorization: Bearer <token>` header, which signals explicit delegation to the enterprise token. Requests without a bearer token do not engage this gate.

---

---

## FS-Chaining-001: An anthproxy instance should be usable as a backend of another anthproxy

**Product outcome**: An operator with credentials or capacity on one machine can reach that capacity from another machine by pointing a second anthproxy at the first. The remote instance is not a special deployment topology — it appears to the local instance as one more backend, competing for dispatch on the same terms as a provider backend.

**Constraint**: The local operator must not have to replicate the remote instance's credentials, provider accounts, or model configuration locally.

---

## SRS-Chaining-001: A peer anthproxy shall be dispatchable as a named backend

**Source FS**: FS-Chaining-001

**System contract**: The system shall provide a backend that dispatches requests to another anthproxy instance over its public Messages interface, such that:

- The peer is a named member of the backend set, subject to the same enablement, selection, pinning, and override mechanisms as any provider backend.
- The peer participates in automatic backend selection rather than requiring an explicit pin.
- Dispatch shall use the peer's public request interface; no private or administrative interface of the peer shall be required.
- An operation the peer publicly exposes shall be served by the peer rather than locally approximated, so that the answer the client receives is the peer's own.
- The peer's capacity state is discovered reactively from its dispatch responses, not by interrogating the peer's internal state.

**Invariants**:
- A backend that reports no time-based reset window shall receive a neutral comparison value and shall not be vetoed for the absence of that window. (SRS-Routing-001 states a comparable rule for stale reset data, but its Scope excludes backends with no time-based reset; this contract is stated here in its own right and does not depend on that one.)
- Exhaustion signalled by a peer shall engage the same cooldown and re-selection path as exhaustion signalled by a provider backend.

---

## FS-Chaining-002: A user request should be classified exactly once, however many proxies it traverses

**Product outcome**: Chaining proxies must not multiply the cost or latency of model-tier classification, and must not produce a routing decision that a later hop silently overrides.

**Constraint**: Classification is a billed model call. Two hops each classifying the same user text doubles that cost and creates two decisions that can disagree.

---

## SRS-Chaining-002: Model-tier routing authority shall rest with the innermost hop

**Source FS**: FS-Chaining-002

**System contract**: When a request is served by dispatch to a peer backend, the dispatching instance shall apply no model-tier routing to that request — neither classification nor any size- or configuration-derived model substitution — and shall transmit the model as originally requested by its client. The receiving instance applies its own routing configuration to that request as it would to any directly-submitted request.

**Invariants**:
- Suppression is determined per request, from the dispatch target actually used to serve that request; it is not a global mode and does not persist across requests.
- A hop that serves a request from a non-peer backend routes normally, including for a request whose earlier attempt was served differently.
- Every model-tier rule the dispatching hop would otherwise enforce is suppressed together; no partial suppression is defined, because a hop that will not serve the request cannot evaluate a rule against the serving backend's capabilities.

---

## FS-Chaining-003: Each proxy in a chain should be controlled independently

**Product outcome**: Operator controls aimed at one proxy — backend preference, routing mode, runtime commands — take effect at that proxy and do not silently reconfigure a downstream one.

**Constraint**: Control directives name backends and modes that are meaningful only in the configuration of the instance receiving them; a downstream instance has a different backend set.

---

## SRS-Chaining-003: The peer hop shall be a control boundary

**Source FS**: FS-Chaining-003

**System contract**: Per-request control directives and runtime commands received by an instance shall be consumed by that instance and shall not be forwarded across a peer dispatch. Request content that is not a control directive — including the requested model and the message payload — is transmitted unchanged.

**Invariants**:
- A failure returned by a peer is surfaced to the client as an error, not converted into a local fallback or model substitution.
- The absence of a capability at the peer is an upstream error condition, not a trigger for local retry against a different model.

---

## FS-Chaining-004: Each proxy should account for the traffic it actually handled

**Product outcome**: Every instance in a chain records the requests it served, so each is independently observable and diagnosable without access to the others.

**Constraint**: The same user request is genuinely handled by every hop it traverses; each hop's record is locally truthful.

---

## SRS-Chaining-004: Usage accounting shall be per-instance and shall not be aggregated across hops

**Source FS**: FS-Chaining-004

**System contract**: Each instance shall record request, session, trace, and usage data for the traffic it dispatches, including traffic dispatched to a peer. Records from separate instances describe overlapping traffic and shall not be summed to obtain a chain-wide total.

**Invariant**: No instance is designated authoritative for chain-wide cost; cost figures are scoped to the recording instance.

---

## FS-Chaining-005: A misconfigured chain should fail at startup, not under load

**Product outcome**: An operator who points a proxy directly at itself learns of it when the process starts rather than discovering it as runaway traffic.

**Constraint**: A self-referential chain is indistinguishable from legitimate traffic at request time; each hop sees a well-formed request.

---

## SRS-Chaining-005: Peer configuration shall be validated for self-reference before service begins

**Source FS**: FS-Chaining-005

**System contract**: When a peer target is configured, the system shall reject at startup a target that resolves to the instance's own listening address, and shall not begin serving requests with such a configuration.

**Scope**: This contract covers direct self-reference. Multi-instance cycles are not detected by this contract; they are the subject of SRS-Chaining-007.

---

## FS-Chaining-006: A chain should be deployable over a network the operator already trusts

**Product outcome**: An operator links two instances across loopback, an SSH tunnel, or a private network they already control, and the chain works without them first standing up an authentication system. Where the path between hops is fronted by an existing access-control layer — a reverse proxy, an identity-aware proxy, a service mesh — the operator can satisfy that layer's credential requirement from the dispatching instance's configuration.

**Constraint**: The confidentiality and integrity of a chain are properties of the network between its hops, not of the product. An operator must be able to tell, from the documented posture, that the product supplies no protection of its own.

---

## SRS-Chaining-006: A dispatching instance may present a credential; a receiving instance demands none

**Source FS**: FS-Chaining-006

**System contract**: A dispatching instance shall be able to transmit an operator-configured credential with each peer dispatch, and shall transmit none when no credential is configured. A receiving instance shall perform no authentication of inbound requests; possession of a credential is neither required nor checked.

**Invariants**:
- A credential presented for peer dispatch shall not be indistinguishable, at the receiving instance, from a credential the client delegates for provider use. The receiving instance shall not admit a peer credential into any provider-credential or token-observation path.
- Adding or removing a peer credential shall change what the dispatching instance sends and nothing about what any instance accepts.

---

## FS-Chaining-007: A cycle of any length should be stopped before it amplifies

**Status**: Deferred — not yet satisfied.

**Product outcome**: An operator who forms a loop across two or more instances, at any time including after those instances have started, sees the loop refused rather than a bill. The failure names the loop instead of appearing as unexplained load.

**Constraint**: A request traversing a cycle is well-formed at every hop; no hop can recognise the cycle from the request's content alone.

---

## SRS-Chaining-007: Cyclic chains shall be detected at request time, independent of hop count

**Source FS**: FS-Chaining-007

**Status**: Deferred — not yet satisfied. SRS-Chaining-005 covers direct self-reference detected at startup; the residual gap described here is knowingly open and is recorded so it remains visible rather than scoped away.

**System contract**: The system shall refuse to serve a request that has already been served by the same instance earlier in its traversal of a chain, for cycles of any length and for cycles formed after any instance has started serving.

**Invariants**:
- Detection shall not depend on a configured maximum chain length, so a cycle is prevented rather than bounded.
- Refusal shall be surfaced as an error to the client, distinguishable from an upstream capacity failure.

---

These SRS items anchor ADR-0015 (pace-delta backend selection), ADR-0016 (OAuth self-pace gate), and ADR-0021 through ADR-0026 (proxy chaining), and can be extended with additional contracts as the system evolves.
