---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Routing-001, SRS-Routing-002, SRS-Routing-003)
---

# ADR-0017: Paced OAuth precedence with personal fallback

**Date**: 2026-08-13

**Status**: Accepted

**Source SRS**: SRS-Routing-001, SRS-Routing-002, SRS-Routing-003

**Supersedes**: [ADR-0016](0016-oauth-self-pace-gate.md) for OAuth-vs-personal precedence. ADR-0016's continuous UTC month-elapsed calculation remains in force.

## Context

ADR-0016 gives OAuth absolute precedence only when its monthly utilization trails elapsed month time by more than the configured margin. Once that gate closes, routing falls back to ADR-0015's relative pace comparison.

That behavior does not express the intended allocation policy. In operation, personal subscription capacity remained underused while OAuth capacity was overused. The intention is neither to maximize OAuth consumption nor to compare which capacity pool is further behind. It is to keep OAuth near a monthly pacing allowance, then consume available personal subscription capacity.

The routing decision needs one direct boundary:

- OAuth is underused while its monthly utilization is below UTC month elapsed plus a configurable margin.
- Once OAuth reaches that allowance, available personal subscription capacity takes precedence.
- OAuth remains the fallback when personal subscription capacity is confirmed unavailable.

The `+ margin` is deliberate headroom, not a deadband below pace. With the default 3 percentage points, OAuth may run up to three points ahead of elapsed month time before personal capacity takes precedence.

## Decision

For an otherwise eligible bearer-token request, choose OAuth exactly when OAuth passes all existing hard eligibility gates and either:

1. OAuth is underused:

   ```text
   oauth_monthly_usage < utc_month_elapsed_pct + margin_pp
   ```

2. No personal subscription candidate is confirmed healthy and available.

Otherwise choose the personal path.

The default `margin_pp` remains 3 percentage points and remains operator-configurable through the existing OAuth pace-margin setting. The comparison is strict: equality chooses personal when personal is available.

This decision replaces both ADR-0016's `oauth_delta < -margin_pp` self-pace threshold and ADR-0015's relative OAuth-vs-personal pace comparison in this decision path. Personal utilization does not affect whether OAuth is underused.

### Personal availability

Personal is available when the existing selector has at least one confirmed healthy, non-cooldown personal subscription candidate whose short-window availability is not exhausted. Existing personal candidate construction and selection remain unchanged.

Unknown or stale personal status does not prove unavailability. It therefore does not activate the personal-unavailable OAuth fallback. This preserves the fail-open convention and prevents an observation gap from granting OAuth precedence.

### Invariants

- No bearer token means OAuth is not a candidate.
- Session overrides and explicit backend preferences retain their existing precedence.
- Existing OAuth hard vetoes remain absolute: missing or stale usage, prior-month usage, unhealthy token, cooldown, spend-cap parking, or any other failed eligibility check cannot be overridden by personal unavailability.
- Continuous UTC month elapsed from ADR-0016 remains the pacing clock.
- The existing personal selector continues to choose among personal backends without modification.
- OAuth's usage probe cadence continues to bound how quickly routing observes a threshold crossing.
- When pace-based routing is disabled, existing kill-switch behavior remains unchanged unless separately amended.

## Considered options

- **Retain ADR-0016 unchanged** — rejected. Its threshold trails elapsed time by the margin and then reopens a relative comparison, which does not encode “paced OAuth, otherwise personal.”
- **Choose whichever pool is further behind pace** — rejected. Relative underuse can starve either pool and obscures the desired precedence boundary.
- **Always choose OAuth whenever it is below 100%** — rejected. This overuses OAuth and leaves paid personal subscription capacity idle.
- **Choose personal until OAuth falls behind `elapsed − margin`** — rejected. This treats the margin as lag tolerance; the accepted policy treats it as OAuth pacing headroom.
- **Treat unknown or stale personal status as unavailable** — rejected. Missing observations must not authorize additional OAuth use.
- **Introduce proportional request splitting** — rejected. A deterministic threshold expresses the policy without new state or traffic-allocation machinery.

## Consequences

- OAuth usage should track approximately `UTC month elapsed + margin`, subject to request volume and usage-probe lag.
- Personal subscription routing becomes the default whenever OAuth has consumed its paced allowance and personal capacity is confirmed available.
- Personal underuse no longer causes OAuth to exceed its allowance through a relative pace comparison.
- Confirmed personal exhaustion or cooldown allows OAuth regardless of whether OAuth is currently above its pacing allowance, provided OAuth passes every hard gate.
- Unknown personal health may leave neither the personal-unavailable fallback nor an OAuth hard fallback available; existing non-OAuth fallback behavior handles that state.
- The existing setting name may still describe a deadband even though this decision uses it as positive pacing headroom. Renaming configuration is deferred unless implementation shows the current name is misleading to operators.
- One usage-probe interval of OAuth overshoot remains possible because routing acts on cached provider utilization.

## Rationale from grilling

The repository owner confirmed the intended precedence on 2026-08-13:

- always choose OAuth while it is underused;
- also choose OAuth when personal capacity is unavailable;
- define underused as `oauth_monthly_usage < UTC month elapsed + margin`;
- otherwise choose personal;
- retain hard OAuth vetoes and existing personal selection;
- do not infer personal unavailability from unknown or stale status.
