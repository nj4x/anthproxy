---
artifact-type: adr
lineage-rules:
  - "ADR must reference at least one source SRS item"
source-srs: docs/FS-SRS-requirements-bootstrap.md (SRS-Chaining-001, SRS-Routing-001)
---

# ADR-0022: The peer joins global selection reporting neutral status, and is exhausted reactively

**Date**: 2026-08-20

**Status**: Proposed

**Source SRS**: SRS-Chaining-001, SRS-Routing-001

## Context

Registering `peer` makes it a member of `backend_names()`, but that alone leaves it unreachable by automatic selection. Three independent mechanisms exclude it:

- `AutoSelector` iterates `_PRIORITY = SUBSCRIPTION_BACKENDS` (`selector.py:41`, `constants.py:8`), a hardcoded tuple. A registered name outside that tuple is never a candidate; it is reachable only by explicit pin or `prefer:`.
- A backend with no `five_hour_status` attribute is skipped before any comparison (`selector.py:581-583`).
- A backend whose `available` is `None` is never switched to — `# Don't switch to an unconfirmed backend.` (`selector.py:621-624`). Unknown *availability* is a veto, even though unknown *weekly utilization* is neutral.

So a peer that reports nothing is invisible. The motivating scenario — burn local OAuth capacity, then fall over to a peer holding a subscription — does not happen by itself.

The tempting fix for the second and third points is to have the peer report its *real* state by calling the peer's `/admin/backends`, which already exposes cached availability flags. That endpoint requires the peer to run with `--enable-ui`, and it belongs to a surface documented as potentially exposing conversation history and routing controls.

## Decision

1. **`peer` joins the global candidate pool through a new tuple, not through `SUBSCRIPTION_BACKENDS`.** `constants.py` gains `ROTATABLE_BACKENDS = SUBSCRIPTION_BACKENDS + ("peer",)`, and `_PRIORITY` (`selector.py:41`) is pointed at it. `SUBSCRIPTION_BACKENDS` itself is untouched.

   Candidate-pool membership is the only property the peer actually needs; `_PRIORITY` is the only thing that grants it. Widening `SUBSCRIPTION_BACKENDS` would have granted it as a side effect of also changing every other consumer of that tuple — the `record_backend in SUBSCRIPTION_BACKENDS` membership test (`stats.py:520`), the `/backend subscription` command's meaning, the help text, the `server.py` membership tests — none of which want a peer in them. Splitting the tuple gives the selector the pool it needs and leaves "is this a subscription?" answering the question it has always answered.

   Per ADR-0020 §8 the pool is intersected with the enabled set per selector instance, so a peer with no configured base URL never appears. That intersection is per-instance selector state and rebinds nothing at module level, so it applies to `ROTATABLE_BACKENDS` exactly as §8 specifies it for the old `_PRIORITY`.

2. **`five_hour_status` returns a constant Neutral Status**: `available=True`, `weekly_utilization=None`, `weekly_resets_at=None`, `weekly_window_hours=None`. It performs no network call and no interrogation of the peer.

3. **Exhaustion is discovered reactively, through the existing 429 path.** A peer that returns 429 is handled by `selector.on_rate_limited` exactly as a provider backend is (`selector.py:266-351`): parked for `Retry-After` or the 300s default recheck (`selector.py:732-750`), with one retry against a freshly-taken snapshot. No peer-specific exhaustion machinery is added.

4. **With pace-delta disabled, a backend that reports no reset window ranks last rather than at the midpoint.** The "peer is permanently block-1" argument below is a property of `_pace_rank_key` (`selector.py:84-96`), and `--auto-backend-pace-delta off` takes that function out of the path. Ranking then falls back to raw weekly utilization, where the peer's `weekly_utilization=None` resolves to `_UNKNOWN_WEEKLY = 50.0` (`selector.py:45`) — and a peer sitting at a synthetic 50% would outrank every real backend above 50% burn. That is the same fabricated-number outcome this ADR rejects under Considered Options, arrived at through a different door.

   So the window-less-last ordering is a property of ranking as such, not of the pace mode: with pace off, candidates reporting no time-based reset window sort behind all candidates that report one, and are compared among themselves as before. `_UNKNOWN_WEEKLY` keeps its existing role for a backend that *has* a window but whose utilization is momentarily unknown; it stops standing in for "there is no window here."

## Consequences

- **Neutral Status ranks the peer last among available backends, which is the behavior we want.** `_pace_rank_key` (`selector.py:84-96`) sorts into two blocks: every candidate with a known elapsed fraction outranks every candidate without one, regardless of burn. A peer has no window, so it is permanently block-1. It therefore wins only when no provider backend is available — which is precisely "local capacity is spent, use the remote." The fallover the motivating scenario needs is a consequence of the ranking rule, not of a special case written for it.
- **The peer is protected from being dropped as much as it is capped from winning.** Unknown weekly utilization resolves to `_UNKNOWN_WEEKLY = 50.0` (`selector.py:45`), a midpoint. Under incumbent-hold (`selector.py:683`, `active_val - best_avail_val >= margin`) a healthy incumbent is displaced by the peer only at ≥55% weekly burn — and, per the block ordering, only when no block-0 candidate is available at all. Once the peer *is* the incumbent, the same neutrality keeps it from being displaced on noise.
- **The local selector is structurally blind to the peer's real capacity.** It learns the peer is exhausted only by being told 429, one request too late, and it cannot distinguish "peer is idle" from "peer is nearly spent." Accepted: this is the same information position the selector holds for any backend between usage probes, and the reactive path is already built to absorb it.
- **A peer whose own backends are all exhausted parks for 300s by default.** If the peer returns 429 without `Retry-After` — which it will, unless it propagates its own upstream's header — the local instance parks it for the default recheck window rather than the true reset time. Correcting this would require the peer to translate its innermost `Retry-After` outward; not attempted here.
- **A peer failure that is not a 429 does not touch selector state.** A 5xx, a refused connection, or a timeout propagates to the client as an error (ADR-0024 §3) and leaves the peer exactly as available as it was; it is not parked, not demoted, and it is a candidate again on the next selection cycle. This is deliberate — 429 is the one signal that carries information about capacity, and the rest carry information about reachability, which the selector has no model for. The visible cost is that a peer whose host is down stays in the pool and is retried each cycle, so an operator sees repeated connection errors rather than one park-and-move-on. Treating transport failure as exhaustion would be worse: it would park a healthy peer for 300s on a single dropped connection.
- **There are now two tuples where there was one, and the difference between them has to be understood.** `SUBSCRIPTION_BACKENDS` means "backends whose capacity is a subscription"; `ROTATABLE_BACKENDS` means "backends the selector may rotate onto." Today they differ by exactly one member, which is precisely the condition under which someone edits the wrong one. Accepted, because the alternative was worse in a way that could not be contained: widening `SUBSCRIPTION_BACKENDS` would have silently changed `stats.py:520`'s classification of peer traffic and the meaning of `/backend subscription`, and ADR-0020 §9 explicitly defers `stats.py` rather than correcting it, so nothing downstream would have caught the drift.
- **Nothing ADR-0006 or ADR-0007 asserted is disturbed.** ADR-0006 §4 fixes `SUBSCRIPTION_BACKENDS` as the static literal `("anthropic", "codex", "openrouter")`, and ADR-0007 restates that membership stays static and enumerates its consumers. Adding a separate tuple leaves both statements true as written; widening the original would have contradicted both and required them to be amended in place.
- **OAuth token refresh needs no change to exclude the peer.** `_refresh_tokens` does not iterate `_PRIORITY` at all — it walks a hardcoded `for name in ("anthropic", "codex")` (`selector.py:485-494`), so `peer` is structurally outside it regardless of pool membership. ADR-0020 §7's requirement to filter that tuple against the enabled set stands unchanged; this ADR adds nothing to it.

## Considered Options

- **Query the peer's `/admin/backends` for real capacity.** Rejected. It buys accuracy the neutral-50 handling was designed to tolerate, in exchange for a hard dependency on the peer running `--enable-ui` and on an endpoint that exposes more than capacity. It also adds a network round-trip to every selector tick and puts network I/O in a code path where lock discipline is already the delicate part (`docs/agents/concurrency.md`).
- **Keep `peer` out of `_PRIORITY` and reach it only through the OAuth-vs-personal fallback.** Rejected. It expresses the motivating scenario and nothing else; the peer would be invisible in `auto` mode for any deployment that is not OAuth-first. The block-1 ranking already delivers "last resort" without narrowing where the peer can be used.
- **Report `available=True` with a synthetic utilization figure** (e.g. 0%, "peer is fresh"). Rejected. It would move the peer to block 0 and let it outrank real backends with real headroom, on the strength of a number that is fabricated.
- **Leave the pace-off path alone and let the peer be compared at `_UNKNOWN_WEEKLY`.** Rejected per §4. It is the "synthetic utilization figure" option above wearing a different hat: an operator who turns pace normalization off would get a peer that outranks every real backend past 50% burn, and would have no way to connect that behavior to the flag they set. A ranking property this load-bearing should not be contingent on a normalization mode.
- **Explicit pinning only — never select the peer automatically.** Rejected. It makes the operator the fallover mechanism.
