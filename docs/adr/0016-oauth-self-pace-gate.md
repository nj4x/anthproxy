---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Routing-001, SRS-Routing-002, SRS-Routing-003)
---

# ADR-0016: Self-pace gate for OAuth-token backend selection

**Date**: 2026-08-10

**Status**: Accepted

**Source SRS**: SRS-Routing-001, SRS-Routing-002, SRS-Routing-003

**Amends**: [ADR-0015](0015-pace-delta-backend-selection.md) — retains the pace-delta metric unchanged, but inserts a precedence gate ahead of the OAuth-vs-personal comparison and refines the month-elapsed formula.

## Context

ADR-0015 replaced raw-% comparison with pace delta and made OAuth-vs-personal a single comparison at `server.py:429-432`:

```
oauth_wins = oauth_delta < personal_delta - deadband
```

This is a *relative* test, and it starves the OAuth backend whenever the personal subscription is deeply underused. Observed on 2026-08-10:

| Backend | Burn | Elapsed | Delta |
|---|---|---|---|
| OAuth token | 18% of $466 monthly | 29.0% (day 10 of 31) | **−11.0** |
| Anthropic personal | 23% weekly | ~50% of window | **−27.0** |

OAuth is behind its own pace by 11pp — it *should* be consuming — but personal is behind by 27pp, so `−11 < −27 − 3` is false and personal holds indefinitely. The personal subscription is a flat-fee sunk cost with a weekly reset; the OAuth monthly cap is **use-it-or-lose-it**. Every dollar the relative comparison protects is a dollar destroyed at UTC month rollover.

The pathology is structural, not a tuning problem: a backend can be behind its own schedule and still lose, because the comparison only asks *who is further behind*, never *is this one behind at all*.

Two further constraints shape the fix:

- OAuth is only a candidate when the inbound request carries `Authorization: Bearer <token>` (`handlers.py:661-666`). This precondition is retained deliberately — it is the only point at which use-it-or-lose-it consumption is authorized.
- The enterprise OAuth capacity is a prepaid monthly allotment that is forfeited at UTC month rollover. **This is an operator-asserted premise**, confirmed by the repository owner during the 2026-08-10 grilling session; it is not independently verified against a billing contract or invoice in this repository. It is load-bearing: if the cap turns out to be a metered spend ceiling rather than a prepaid allotment, this ADR inverts — the gate would convert unbilled headroom into real charges, and it should be reverted via `auto_backend_pace_delta: off`. The provider's own field vocabulary (`monthly_limit`, `used_credits`, `spend_limit_reached`) is ambiguous between the two readings, which is why the premise is called out explicitly rather than treated as self-evident.
- The gate consumes `oauth.burn`, which `_usage_burn` populates from the provider's `utilization` field (`oauth_registry.py:336-342`) — not from `used_credits` directly. The design assumes `utilization` tracks prepaid-credit consumption linearly. `used_credits` is cumulative month-to-date, in-memory only, and lost on restart (`oauth_registry.py:41-51`); any design requiring durable per-day spend accounting would need new persistence.

## Decision

Insert an **absolute self-pace gate** ahead of the existing relative comparison. When OAuth is behind its own monthly pace by a tunable stability margin, it wins outright; the personal comparison is never consulted.

```
margin_pp = auto_backend_oauth_pace_deadband_pp  # default 3.0 pp, operator-tunable
if oauth_valid and oauth_delta is not None and oauth_delta < -margin_pp:
    oauth_wins = True          # self-pace gate
else:
    oauth_wins = <ADR-0015 comparison, unchanged>
```

Once OAuth's delta approaches pace (within the margin, `oauth_delta >= -margin_pp`), the ADR-0015 comparison runs verbatim. Strict `<`: at exactly `oauth_delta == -margin_pp` the gate does **not** fire and control falls through to the ADR-0015 comparison, mirroring ADR-0015's tie rule.

SRS-Routing-003 **exempts** this gate from the SRS-Routing-001:30 stability-margin invariant, because that invariant governs multi-backend comparisons and this is a single-backend absolute test. The margin is therefore not required for conformance — it is retained because it is free (an existing knob) and it keeps the gate's trigger point away from the exact pace line, where measurement noise in `utilization` would otherwise decide routing. Per-request oscillation is prevented not by the margin but by the probe cadence: `oauth.burn` only changes when a usage probe lands (~300s, `oauth_registry.py:234-258`), and a reading older than the TTL makes the token ineligible outright (`oauth_registry.py:143-148`).

### Steady-state behavior

The gate is a servo, not a burst. With `delta = burn − elapsed`:

- OAuth consuming raises `burn` → `delta` rises → gate approaches its threshold and stops firing.
- Time passing raises `elapsed` → `delta` falls → gate re-arms.

Because `elapsed` advances continuously (this is what the intra-day term buys), the gate re-arms as soon as `elapsed` ticks past the crossing point — not after a fixed calendar interval. The equilibrium is `burn ≈ elapsed − margin_pp`: OAuth consumption tracks the monthly pace line, trailing it by a constant offset of `margin_pp` (3.0pp of $466 ≈ $14). The control granularity is the probe TTL, since that is the only thing that updates `burn`.

### Month-elapsed refinement (Data Realization)

The `month_elapsed_pct` currently implemented at `oauth_registry.py:305-313` is a step function that jumps at UTC midnight:

```
month_elapsed_pct = (day − 1) / days_in_month × 100
```

This is what makes the step function unusable for the gate: `elapsed` is frozen for 24 hours at a time, so once OAuth's consumption lifts `delta` back above `-margin_pp`, the gate cannot re-arm until the next UTC midnight — exactly one burst per day, then dormancy. Add the intra-day term:

```
month_elapsed_pct = (day − 1 + seconds_into_utc_day / 86400) / days_in_month × 100
```

Range is now `[0, 100)` (ceiling widens from ~96.8 to just under 100). Because `elapsed` now advances every second, the gate re-arms as soon as consumption falls behind the drifting line, and the effective release granularity becomes the usage-probe cadence (~300s) rather than a per-day step. This is the mechanism that produces the steady state described above — no daily budget concept is introduced.

The refinement also applies to the ADR-0015 comparison (the same value feeds both paths). This is accepted: it makes the existing comparison more accurate and removes the step-function artifact from both selection paths. The `_month_elapsed_pct` docstring at `oauth_registry.py:305-313` needs the same update, including the widened range.

### Invariants

- **Precondition unchanged**: no bearer token on the request → OAuth is not a candidate → the gate never runs.
- **Bypass unchanged**: `prefer_backend` and session overrides bypass the gate entirely (`server.py:374-380`), exactly as they bypass the ADR-0015 comparison.
- **Hard gates remain absolute vetoes**: cooldown, monthly spend-cap, health, `eligible`, and staleness veto OAuth before the gate is reached. The self-pace gate never resurrects a blocked backend.
- **Fail open on unknown**: if `oauth_valid` is false or `oauth.burn is None`, the gate does not fire and control falls through to the ADR-0015 comparison. Unknown utilization stays neutral, consistent with `docs/agents/concurrency.md`.
- **Kill-switch inherited**: when `auto_backend_pace_delta` is `off`, no pace deltas exist; the gate is inert and the raw-% comparison applies.
- **Margin offsets the trigger; probe cadence bounds flapping**: the margin (`auto_backend_oauth_pace_deadband_pp`, default 3.0pp) is a single fixed offset, not a dual-threshold hysteresis — it relocates the crossing point from `0` to `-3.0pp` and holds the equilibrium clear of the pace line. Per-request oscillation is prevented by `oauth.burn` being constant between ~300s probes and by the staleness veto at `oauth_registry.py:143-148`, not by the margin.
- **Tie rule**: at exactly `oauth_delta == -margin_pp` the gate does not fire (strict `<`); control falls through to the ADR-0015 comparison.
- **Substitutability is assumed, not guaranteed**: during a gate-on window OAuth takes 100% of bearer-token traffic. The ADR reasons about budget only; model-tier availability, latency, and 429 behavior are not considered. An OAuth failure mid-window is absorbed by the existing hard vetoes (cooldown on 429, health check), which end the burst by making the token ineligible — the gate then falls through to the ADR-0015 comparison on the next snapshot. No gate-specific failure path is added.

## Considered Options

- **A tunable bias added to `oauth_delta` before the ADR-0015 comparison** (e.g. `oauth_delta - bias_pp < personal_delta - deadband`) — rejected. It leaves the decision relative, so it only moves the starvation threshold rather than removing it: a sufficiently underused personal backend still wins at any finite bias. The gate removes the relative dependency outright.
- **A separate `auto_backend_oauth_self_pace_margin_pp` knob** — rejected in favour of reusing `auto_backend_oauth_pace_deadband_pp`. Both quantities mean the same thing — percentage points of pace slack before a routing decision changes — so a second knob would be two names for one concept. The reuse is deliberate and has a real coupling cost, recorded in Consequences: raising the knob to damp OAuth-vs-personal switching also makes the self-pace gate fire later. If operators ever need to tune the two independently, splitting the knob is the escape hatch.
- **Fixed share split (route N% of sessions to OAuth)** — rejected. Discards the pace logic entirely and will overrun the monthly cap under sustained load.
- **Explicit daily budget tracked from SQLite** (`SUM(cost_estimate) WHERE backend='oauth' AND request_ts >= UTC midnight`) — rejected. `cost_estimate` is our list-price estimate from token counts (`db.py:368-389`), not the provider's credit accounting; the two bases drift. The self-pace gate obtains the same outcome from the provider's own authoritative figure with zero new state.
- **Persisting a `used_credits` baseline at UTC midnight** — rejected. Requires a new migration and a restart-durability story, to compute a quantity the cumulative figure already implies.
- **Blending the stale probe with in-flight local cost** — rejected. Mixes two incompatible accounting bases.
- **Shortening the 300s usage TTL** — rejected. Hammers the provider usage endpoint to solve a bounded, self-correcting problem.
- **A gate-specific dual-threshold hysteresis** (distinct arm/disarm thresholds, or a minimum dwell time, on top of the reused margin) — rejected. The probe cadence already bounds transitions to one per ~300s, so a second stickiness mechanism would add state and configuration for a problem the TTL already solves. Session overrides and `prefer_backend` remain available for callers who need hard stickiness.

## Consequences

- **OAuth consumption becomes permitted up to monthly pace rather than contingent on personal headroom.** The personal backend's utilization no longer blocks OAuth dispatch while OAuth is behind its own pace by more than the margin.
- **Overshoot within one probe TTL is accepted and bounded.** `used_credits` refreshes on a ~300s cadence (`oauth_registry.py:234-258`), so sustained load can consume up to one TTL's worth of budget before the gate notices the approach to pace. Maximum overshoot: the spend achievable in 300s of unrestricted OAuth routing. This is bounded and self-correcting: overshoot raises `oauth_delta`, eventually handing traffic back to personal as the margin is re-entered.
- **The gate holds a steady state, not a burst.** Equilibrium is `burn ≈ elapsed − margin_pp`: OAuth consumption trails the monthly pace line by a constant ~3.0pp (≈$14 of $466). Gate state can transition at most once per probe TTL, because `oauth.burn` is constant between probes.
- **The deadband knob now has two coupled effects.** `auto_backend_oauth_pace_deadband_pp` sets both the OAuth-vs-personal switch band (ADR-0015) and the self-pace gate threshold. Raising it to damp switching also delays gate firing and widens the steady-state shortfall against the monthly cap. This coupling is accepted; splitting the knob is the remedy if it ever bites.
- **The personal subscription will be used less**, by design. Its weekly window is a flat-fee sunk cost, so leaving it idle costs nothing; leaving the OAuth cap idle destroys prepaid value.
- **Month-boundary behavior**: at UTC month rollover `month_elapsed_pct` resets to ~0 while `used_credits` resets to 0 on the provider side. Until the first post-rollover probe lands (≤300s), a stale `usage_month` makes the token ineligible (`oauth_registry.py:148`), so the gate correctly does not fire on stale cross-month data.
- **ADR-0015's step-function month elapsed is superseded** by the intra-day formula. SRS-Routing-002 (UTC derivation) is unaffected — the boundary is still UTC.
- **UI StatusPanel divergence widens**: the UI's local-time step-function pace head (StatusPanel.tsx:221-222) now diverges from the server's UTC intra-day computation by up to ~3.2pp plus timezone offset, and no longer explains gate decisions. This divergence was documented in ADR-0015; aligning the UI is deferred to a follow-up.

---

## Rationale for decisions made during grilling

- **Absolute gate over relative comparison**: The relative test cannot express "this backend is behind its own schedule and should run." Adding an absolute precedence gate is the minimal change that expresses the use-it-or-lose-it constraint, and it leaves the ADR-0015 metric and its comparison entirely intact.
- **`oauth_delta < -margin_pp` as the trigger with a reused margin**: Zero is the natural boundary ("exactly on schedule"), but triggering at exactly zero would let measurement noise in `utilization` decide routing. SRS-Routing-003 exempts this gate from the SRS-Routing-001:30 stability invariant — the margin is not a conformance requirement — but it costs nothing (existing knob) and keeps the equilibrium clear of the pace line, so it is retained. Its operator-tunability is inherited from ADR-0015's precedent.
- **Intra-day elapsed term**: With the step function, `elapsed` is frozen for 24h, so a gate that disarms mid-day cannot re-arm until the next UTC midnight — one burst per day. The continuous term is what makes the gate a servo instead of a daily pulse; its granularity is then bounded only by the usage-probe refresh rate. A separate daily-budget concept would add persistence and restart-durability work for no gain.
- **Rejected finding — "3pp margin causes ~22h dormancy"**: raised in review and checked, not adopted. It assumed the gate must wait for `elapsed` to advance a further 3pp before re-arming. The sign is the other way: `delta = burn − elapsed`, so rising `elapsed` *lowers* `delta` and re-arms the gate. At the boundary an infinitesimal advance in `elapsed` re-fires it. Dormancy would only occur if `elapsed` were frozen — which is precisely the step-function behavior this ADR removes.
- **No daily-spend accounting**: The cumulative monthly figure plus a continuous elapsed fraction already encodes the daily target. Introducing tracked daily state would add persistence, restart-durability, and a UTC/local-day mismatch (`stats.py` buckets local, `db.py` buckets UTC) to compute something already implied by the existing values.
- **Accepting probe-lag overshoot**: The gate's self-correction makes overshoot transient. Tightening it (e.g., shortening the 300s TTL) would trade a bounded, automatically-resolved error for a permanent load increase on the provider endpoint.
- **Fail open, not fail toward OAuth**: Treating unknown utilization as "behind pace" would spend against a cap we cannot observe. Neutral-on-unknown is already the codebase-wide convention (concurrency.md).
- **Retaining the bearer-token precondition**: It is the authorization point for use-it-or-lose-it consumption. Making OAuth selectable from server-held credentials alone would broaden spend authority beyond what any caller opted into.
- **The prepaid premise is operator-asserted, not verified**: an earlier draft of this ADR cited a specific Terms of Service section and quoted it verbatim. That citation was fabricated and has been removed. The premise stands on the repository owner's confirmation during the 2026-08-10 session, and is recorded as a named assumption with a stated falsifier and rollback path rather than as an established fact.
- **Correct citations**: eligibility term is at `oauth_registry.py:148` (not 170 per earlier draft); the gate consumes `oauth.burn`, which is the provider's `utilization` field (`oauth_registry.py:336-342`), not `used_credits`.
