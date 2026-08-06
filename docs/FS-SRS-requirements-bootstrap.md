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

These SRS items anchor ADR-0015 (pace-delta backend selection) and can be extended with additional routing contracts as the system evolves.
