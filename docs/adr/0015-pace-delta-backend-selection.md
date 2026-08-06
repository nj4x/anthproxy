---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Routing-001, SRS-Routing-002)
---

# ADR-0015: Pace-delta metric for OAuth vs personal backend selection

**Date**: 2026-08-05

**Status**: Decided (under grilling)

**Source SRS**: SRS-Routing-001, SRS-Routing-002

## Context

The current auto-backend selector compares enterprise OAuth token burn (monthly quota %) directly against personal subscription backend burn (weekly quota %), using the same raw-% scale: `oauth_wins = oauth.burn < personal.burn`. This conflates two different windows:

- **Enterprise**: monthly quota with no API-provided reset timestamp; burn reflects day-of-month progression.
- **Personal (Anthropic, Codex)**: weekly quota with API-provided reset timestamp; burn reflects time-to-window-reset.

**Pathology** (reported in grilling): Enterprise at 17% monthly (but ahead of calendar pace → red head) beats personal at 86% weekly (but behind time-to-reset pace → green head), because 17 < 86. The decision is counterintuitive: the backend with *surplus* headroom loses to the one burning *ahead* of schedule.

The root cause: raw-% comparison across windows of different lengths measures consumption rate unevenly. A backend at 50% of its window is equally constrained whether the window is 5 hours, 7 days, or 30 days — but a raw-% comparison treats them differently.

## Decision

Replace raw-% comparison with **pace delta**: `burn% − elapsed%`, where:

- `elapsed%` = fraction of the window already consumed by time, in the 0–100% scale.
- Lower delta = more headroom relative to pace → better.
- Apply to both **oauth-vs-personal gate** (phase 1) and **personal-vs-personal ranking** (phase 2), ensuring consistent semantics.

### Elapsed formulas (realization)

**Month elapsed** (enterprise OAuth, UTC):
```
month_elapsed_pct = (now_utc.day − 1) / days_in_month(now_utc.year, now_utc.month) × 100
```
Computed in UTC (matches the real spend-cap reset boundary). `days_in_month` derived via `calendar.monthrange`. Range is `[0, ~96.8]` by construction; no clamping needed because `now_utc.day ∈ [1, days_in_month]`.

**Weekly elapsed** (personal, from API reset timestamp):
```
if weekly_resets_at ≤ now:      # stale reset — see below
    weekly_elapsed_pct = 0.0
else:
    weekly_elapsed_pct = max(0.0, min(100.0, (1 − (weekly_resets_at − now) / (weekly_window_hours × 3600)) × 100))
```
- `weekly_resets_at > now`: normal in-window case; elapsed in `[0, 100)`.
- `weekly_resets_at ≤ now` (stale reset, window should have rolled but the API has not refreshed the timestamp): treat elapsed as `0.0` — the window is about to roll (or just did), so its reported burn is about to drop. With `delta = burn − elapsed`, `elapsed = 0` yields the *highest* delta (`= burn`), which under ascending-delta ranking makes the backend the *least* preferred. This deliberately **deprioritizes** a stale-reset backend: its burn reading is on the verge of going stale, so we do not want to route new work onto it based on that reading. (A naive upper clamp to `100.0` would instead produce the lowest delta and *reward* the stale backend — the opposite of the intended safety behavior.)
- `weekly_window_hours` is `None` or `≤ 0` (provider did not supply a window size, or Codex weekly window absent): `weekly_elapsed_pct = None` → treated as neutral `50.0` (`_UNKNOWN_WEEKLY`) in delta computation.

### Comparisons

**OAuth vs personal** (`server.py:snapshot_for_request`):
- `oauth_delta = oauth.burn − month_elapsed_pct`.
- `personal_delta = personal.burn − (personal.weekly_elapsed_pct if not None else 50.0)`.
- `oauth_wins` iff `oauth_delta < personal_delta − deadband_pp` where `deadband_pp = auto_backend_oauth_pace_deadband_pp` (default `3.0`, percentage points). Strict `<`: equality means personal wins (incumbent-friendly).
- The **personal representative** is chosen by minimum pace delta across the personal candidate pool (not by raw burn). A lower-delta personal candidate must not be eliminated before the OAuth comparison runs.

**Personal vs personal** (`selector._compute_best_unlocked` and `selector.personal_candidates`):
- **Split-branch sort** (single authoritative description; applies identically in both `personal_candidates()` and `_compute_best_unlocked`):
  - Candidates with a known `weekly_elapsed_pct` form the **pace-delta block**, sorted ascending by `burn − weekly_elapsed_pct`.
  - Candidates with `weekly_elapsed_pct is None` (OpenRouter, unknown windows) form the **raw-only block**, sorted ascending by raw `burn` (`None → _UNKNOWN_WEEKLY`). They are **never** forced through a fake `burn − 50` mapping.
  - Final order is the concatenation `pace-delta block + raw-only block`, each block preserving its own internal order. Pace-delta candidates therefore rank ahead of raw-only candidates as a group.
  - This preserves the prior behavior for OpenRouter (raw credit% only) while introducing pace only where it is meaningful.
- Incumbent-hold and parking logic (STEP-1, STEP-2) **continue to compare raw weekly**; pace delta affects only the initial ordering of the candidate list and the OAuth-vs-personal gate.

### Data plumbing

**Extend `FiveHourStatus`** (`_shared/__init__.py`):
- `weekly_resets_at: float | None` — POSIX timestamp for the reset of *the weekly window that supplied the reported `weekly_utilization`*.
- `weekly_window_hours: float | None` — duration of that weekly window (168.0 for Anthropic/Codex seven-day windows).

**Extend `PersonalCandidate`** (`selector.py`):
- `weekly_elapsed_pct: float | None` — fraction of the weekly window elapsed, computed in `personal_candidates()` from the cached `(weekly_resets_at, weekly_window_hours)` pair. Gated behind the existing `fresh` cache-TTL check so a stale reading does not silently produce a contemporary elapsed.

**Extend `OAuthTokenSnapshot`** (`oauth_registry.py`):
- `month_elapsed_pct: float` — populated by the registry at `snapshot()` time using its injected `_utcnow` and `calendar.monthrange`. Keeps UTC/spend-cap arithmetic in one place next to the existing monthly-reset logic.

**Reset-cache pairing invariant**: For Anthropic, `_max_weekly_utilization` returns MAX across `seven_day` / `seven_day_sonnet` / `seven_day_opus`. The `weekly_resets_at` reported in `FiveHourStatus` shall be read from the **same window that supplied the max utilization** (so the reset timestamp always pairs with the burn it paces). If that window lacks a `resets_at`, fall back to `None` (→ neutral 50 elapsed).

### Fallback and unknowns

- **Unknown weekly reset data** (transient fetch gap, fresh startup, Codex weekly window absent, Anthropic max-window lacks reset, OpenRouter): `weekly_elapsed_pct = None` → treated as 50% neutral in delta computation, and the candidate is ranked on raw utilization alongside other raw-only candidates. Preserves incumbent-hold semantics from `concurrency.md`.
- **OpenRouter** (no time-based reset, hard USD cap): never carries a weekly elapsed; participates in ranking on raw credit% only.
- **Stale weekly reset** (`weekly_resets_at ≤ now`): `weekly_elapsed_pct = 0.0` — the window is treated as just-started, giving the highest delta (`= burn`) so the backend is *deprioritized* while its burn reading is about to go stale.
- **Pace-delta kill-switch**: `auto_backend_pace_delta: on|off` (default `on`). When `off`, the OAuth-vs-personal gate and personal-vs-personal sort fall back to the prior raw-% comparison (`oauth.burn < personal.burn`, `min(burn)`). Operator escape hatch.

### Invariants

- Hard gates (cooldown, monthly spend-cap, health, staleness) remain **absolute vetoes**; pace delta never resurrects a blocked backend.
- Incumbency guard from `concurrency.md` preserved: active backend with unknown reading held as-is.
- The selector's `_compute_best_unlocked` STEP-1 and STEP-2 logic (hysteresis, parking) **operates on raw weekly**, not delta; pace applies only to the ranking sort key.
- The dead-band boundary is strict `<`: `oauth_delta == personal_delta − deadband_pp` does **not** switch; personal wins on ties.

## Consequences

**Behavioral**:
- Backends ahead of schedule lose to backends behind schedule, independent of window length.
- A backend at 50% of its window-to-reset is ranked higher than one at 60%, even if the latter has absolute days remaining.
- Early-window transient surges (e.g., hour 1 of a 168h window, usage spikes to 10%) no longer dominate personal candidate ranking.
- Stale weekly resets (backend should have rolled but hasn't) transiently penalize that backend until the API catches up: elapsed is forced to `0.0`, yielding the highest delta and the lowest ranking priority so new work is steered away while the burn reading is about to go stale.

**Correctness**:
- Month-elapsed computed in UTC (matches the real spend-cap boundary) rather than local time (UI client-side).
- Enterprise-vs-personal selection becomes independent of accident of when in the UTC month the request arrives.
- Anthropic reset/burn pairing is consistent: the elapsed is always derived from the same weekly window that produced the reported max utilization.

**Data consistency**:
- Both windows' elapsed% are client-side observable in principle. The checked-in UI currently renders month pace heads in **browser local time**, which diverges from the server's UTC computation around UTC day boundaries. This PR **accepts that divergence** (documented in FS-Routing-002); aligning the UI is deferred to a follow-up.

## Source SRS

**SRS-Routing-001** (pace-delta normalization across windows) and **SRS-Routing-002** (UTC month-elapsed derivation).

Both requirements are defined in `docs/FS-SRS-requirements-bootstrap.md` alongside their product outcomes (FS-Routing-001, FS-Routing-002).

---

## Rationale for decisions made during grilling

- **Pace delta over raw %**: Comparable across windows of different lengths.
- **Signed delta (not projection)**: Direct interpretation ("burnrate relative to linear time"), avoids early-window blow-up (e.g., 10% in hour 1 of 168h → 1680% projected).
- **Weekly window only (not 5-hour binding)**: Personal selection already uses weekly as the tiebreaker; 5-hour is for reactive 429 recovery. Consistency with existing selector semantics.
- **3pp dead-band** (operator-tunable via `auto_backend_oauth_pace_deadband_pp`): Prevents flapping when deltas cross; matches hysteresis philosophy in `auto_backend_weekly_margin`.
- **Strict `<` boundary**: Ties favor the incumbent personal backend; avoids dithering when both sides sit at exactly the dead-band edge.
- **OpenRouter excluded**: No time-based reset; comparing it on raw credit% avoids forcing a fake elapsed value. Ranking preserves prior raw-utilization semantics for OpenRouter and other unknown-window candidates.
- **UTC month-elapsed**: Matches real spend-cap reset boundary; prevents routing decisions drifting across UTC day boundary.
- **Reset/burn pairing on Anthropic**: The `weekly_resets_at` is read from the same `seven_day*` window that supplied the max utilization, so a Sonnet-capped burn is never paired with the overall seven-day reset.
- **Stale-reset deprioritization**: Stale resets (reset timestamp in the past) force elapsed to `0.0`, giving the highest delta (`= burn`) and the lowest ranking priority. This deprioritizes a backend whose burn reading is about to go stale, rather than rewarding it. Clamping elapsed *up* to `100.0` was rejected because under ascending-delta ranking it would make the stale backend the *most* preferred — the opposite of the intended safety behavior.
- **Month-elapsed in `oauth_registry`**: The registry already owns UTC month-boundary logic (`_next_month`, `usage_month`); co-locating `month_elapsed_pct` keeps UTC/spend-cap arithmetic in one place and gives tests access to the injected `_utcnow` clock.
- **Kill-switch `auto_backend_pace_delta`**: Operators can revert to raw-% comparison without a redeploy if pace semantics misbehave against a specific provider.
