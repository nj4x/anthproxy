"""Auto-backend selector for anthproxy.

When ``--auto-backend`` is enabled, ``AutoSelector`` runs a background poll
thread and reacts to upstream 429 errors to keep requests routed at the
backend with the most available weekly quota.

The selector compares weekly utilization across backends with an open 5-hour
window.  It holds the current active backend unless a challenger's weekly
utilization is at least ``auto_backend_weekly_margin`` percentage points lower
(hysteresis band, default 5 pp), preventing flapping between similarly-consumed
backends.  A healthy available incumbent is never displaced by bedrock-parking;
parking applies only when the active backend is not itself a healthy available
one.  Bedrock is the unconditional fallback in normal (auto) mode when no
subscription backend has an open 5-hour window.  In subscription mode the
fallback is the lowest-weekly subscription backend instead — bedrock is never
used.

Selector modes
--------------
``'auto'``         Normal auto-selection; bedrock is the unconditional fallback.
``'pinned'``       Auto-selection is paused; a specific backend is locked.
``'subscription'`` Auto-selection continues but only among subscription
                   backends; bedrock is never used.
"""

import dataclasses
import logging
import threading
import time

from .constants import SUBSCRIPTION_BACKENDS

logger = logging.getLogger(__name__)

# How long (seconds) to keep a backend in the "cooling down" state after a 429
# when we have no resets_at from the usage endpoint.
_DEFAULT_RECHECK_SECS = 300.0

# Ordered list of subscription backends for initial iteration.
# bedrock is the unconditional fallback (no 5-hour window, always available).
_PRIORITY = SUBSCRIPTION_BACKENDS
_FALLBACK = "bedrock"

# When weekly utilization is unknown, use this neutral value for comparisons.
_UNKNOWN_WEEKLY = 50.0
_PERSONAL_CACHE_TTL = 300.0


@dataclasses.dataclass(frozen=True)
class PersonalCandidate:
    name: str
    burn: float
    weekly_elapsed_pct: float | None = None


def _pace_enabled(config) -> bool:
    """True unless the pace-delta kill-switch is set to 'off'."""
    return getattr(config, 'auto_backend_pace_delta', 'on') != 'off'


def _weekly_elapsed_pct(
    weekly_resets_at: float | None,
    weekly_window_hours: float | None,
    now: float,
) -> float | None:
    """Fraction of the weekly window elapsed by time, 0–100, or None.

    Returns None when the window size or reset timestamp is unknown (→ ranked on
    raw burn behind every candidate that has an elapsed figure). A stale reset
    (``weekly_resets_at <= now``) yields ``0.0``: the window is about to roll, so
    under ascending pace-delta ranking (``burn - elapsed``) the backend gets the
    highest delta and is deprioritized while its burn reading is going stale.
    """
    if weekly_window_hours is None or weekly_window_hours <= 0:
        return None
    if weekly_resets_at is None:
        return None
    if weekly_resets_at <= now:
        return 0.0
    remaining = weekly_resets_at - now
    return max(0.0, min(100.0, (1.0 - remaining / (weekly_window_hours * 3600.0)) * 100.0))


def _pace_rank_key(burn: float, elapsed: float | None, pace_on: bool):
    """Sort key implementing the ADR-0015 split-branch ranking.

    Returns a ``(block, value)`` tuple so candidates with a known elapsed
    (``block=0``) rank ahead of the rest (``block=1``) as a group, each block
    preserving its own ascending order. The block depends only on ``elapsed``,
    never on ``pace_on``; only the *value* varies with the mode (``burn -
    elapsed`` with pace on, raw ``burn`` with pace off).

    ``block=1`` means "no usable elapsed signal right now", not "no weekly
    window exists": ``elapsed`` is None whenever *either* the reset timestamp or
    the window size is missing, including for a backend that does have a weekly
    window whose ``resets_at`` is momentarily unparseable. Without an elapsed
    figure there is no pace delta to compute and the candidate cannot be
    compared on the same axis as one that has it.

    Callers that need a single scalar key (e.g. for ``sorted``) can use
    ``key=lambda c: _pace_rank_key(...)`` and Python's tuple comparison does the
    block-then-value ordering.
    """
    if elapsed is None:
        return (1, burn)
    return (0, burn - elapsed if pace_on else burn)


def _sort_personal_candidates(
    candidates, pace_on: bool,
) -> tuple[PersonalCandidate, ...]:
    """Order personal candidates: pace-delta block then raw-only block.

    Candidates that carry a known ``weekly_elapsed_pct`` rank ahead, as a group,
    of candidates with no elapsed (OpenRouter, unknown windows, stale cache
    readings) in both pace modes. Within the leading block they sort ascending by
    ``burn - weekly_elapsed_pct`` with pace on and by raw ``burn`` with pace off;
    the trailing block always sorts ascending by raw ``burn``.
    """
    return tuple(
        sorted(
            candidates,
            key=lambda c: _pace_rank_key(c.burn, c.weekly_elapsed_pct, pace_on),
        )
    )


class AutoSelector:
    """Manages automatic backend selection by weekly-consumption comparison.

    When the active backend is itself a healthy available backend, the selector
    holds it unless a challenger's weekly utilization is at least
    ``auto_backend_weekly_margin`` percentage points lower (hysteresis band).
    An active backend whose weekly is momentarily unknown is held at the
    neutral comparison value so a transient fetch gap does not strip incumbency.
    A healthy available incumbent always outranks bedrock-parking; parking
    applies only when the active backend is not a healthy available one and an
    exhausted backend is decisively below the best available by the same margin.
    Bedrock is the unconditional fallback in normal (auto) mode when no
    subscription backend has an open 5-hour window.  Subscription-only mode
    never uses bedrock.

    Thread-safety:
        ``evaluate()``, ``on_rate_limited()``, ``pin()``, and ``resume()`` are
        all safe to call from any thread.  Internal state is guarded by
        ``_lock``; ``registry.switch()`` is called outside that lock (as
        required by BackendRegistry's own locking model).
    """

    def __init__(self, registry, config, interval: float | None = None,
                 oauth_registry=None):
        self._registry = registry
        self._config = config
        self._oauth_registry = oauth_registry
        self._interval = (
            interval if interval is not None else config.auto_backend_interval
        )
        # epoch timestamp before which each subscription backend is considered
        # exhausted (set by 429 signal or confirmed-empty usage window).
        self._exhausted_until: dict[str, float] = {}
        # Last known weekly utilization per backend (0–100).  Updated whenever
        # five_hour_status() succeeds; used for exhausted backends where we
        # don't want to make a live network call.
        self._last_weekly: dict[str, float] = {}
        # Last known weekly reset timestamp and window size, paired with
        # _last_weekly, used to compute the pace-delta elapsed for cached
        # (non-probed) candidates in personal_candidates().
        self._last_weekly_resets_at: dict[str, float | None] = {}
        self._last_weekly_window_hours: dict[str, float | None] = {}
        self._last_status_at: dict[str, float] = {}
        self._last_available: dict[str, bool | None] = {}
        # Selector mode: 'auto' | 'pinned' | 'subscription'.
        # 'pinned'      — auto-evaluation is suspended; _pinned holds the name.
        # 'subscription' — auto continues but only among subscription backends.
        # Initial mode comes from config (auto_backend_mode); runtime
        # proxy-set-backend commands override it thereafter.
        self._mode: str = config.auto_backend_mode
        # The pinned backend name; meaningful only when _mode == 'pinned'.
        self._pinned: str | None = None
        # ADR-0020 §8: intersect the module-level candidate pools with the
        # enabled backend set, once, at construction. Never rebind _PRIORITY/
        # _FALLBACK themselves — those stay shared, unfiltered module state.
        enabled = frozenset(registry.list_backends())
        self._priority: tuple[str, ...] = tuple(n for n in _PRIORITY if n in enabled)
        self._fallback: str | None = _FALLBACK if _FALLBACK in enabled else None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        if self._oauth_registry is not None:
            self._oauth_registry.set_wake(self.wake)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background poll thread (daemon)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="auto-selector", daemon=True
        )
        self._thread.start()
        mode = (
            "auto-select+refresh" if self._config.auto_backend else "token-refresh-only"
        )
        logger.info(
            "Auto-selector started (mode=%s, interval=%.0fs)", mode, self._interval
        )

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def wake(self) -> None:
        self._wake.set()

    def evaluate(self) -> str:
        """Pick and switch to the best backend right now.

        Returns the name of the backend that is (or becomes) active.
        A no-op (returns pinned name) if auto is paused via ``pin()``.
        In subscription mode, selection is restricted to subscription backends.
        """
        with self._lock:
            if self._mode == "pinned":
                return self._pinned
            subscription_only = self._mode == "subscription"
            exhausted_snapshot = dict(self._exhausted_until)
            last_weekly_snapshot = dict(self._last_weekly)

        best, statuses, new_exhausted, new_weekly = self._compute_best_unlocked(
            exhausted_snapshot, last_weekly_snapshot, subscription_only
        )

        if new_exhausted:
            with self._lock:
                self._exhausted_until.update(new_exhausted)
        if new_weekly:
            with self._lock:
                self._last_weekly.update(new_weekly)
        if statuses:
            observed_at = time.time()
            with self._lock:
                for name, status in statuses.items():
                    self._last_status_at[name] = observed_at
                    self._last_available[name] = status.available
                    self._last_weekly_resets_at[name] = status.weekly_resets_at
                    self._last_weekly_window_hours[name] = status.weekly_window_hours

        active = self._registry.active_name()
        if best is None:
            logger.debug(
                "Auto-selector: no enabled backend available to switch to; staying on %s",
                active,
            )
        elif best != active:
            reason = _format_switch_reason(
                "periodic evaluation", active, best, statuses
            )
            result = self._registry.switch(best, reason=reason)
            if result.kind == "failed":
                logger.warning(
                    "Auto-selector: switch to %s failed (%s), staying on %s",
                    best,
                    result.error,
                    active,
                )
        return self._registry.active_name()

    def on_rate_limited(self, name: str, retry_after: float | None = None) -> str:
        """Called when backend ``name`` returns a 429.

        Marks ``name`` as exhausted, re-evaluates, and returns the new active
        backend name so the caller can retry on it.
        ``retry_after`` is the value of the upstream Retry-After header (seconds
        from now), if present.
        """
        with self._lock:
            if self._mode == "pinned":
                self._mark_exhausted_locked(name, retry_after)
                return self._pinned

            backend_to_query = self._mark_exhausted_locked(name, retry_after)
            subscription_only = self._mode == "subscription"
            exhausted_snapshot = dict(self._exhausted_until)
            last_weekly_snapshot = dict(self._last_weekly)

        # Network calls happen outside the lock.
        source_status = None
        if backend_to_query is not None:
            five_hour_status_fn = getattr(backend_to_query, "five_hour_status", None)
            if five_hour_status_fn is not None:
                try:
                    source_status = five_hour_status_fn(self._config)
                    # Only honor resets_at as the cooldown deadline when the
                    # backend is genuinely exhausted (available=False).  A 429
                    # from a backend that still reports available quota is
                    # request-specific (e.g. long-context credit policy), not
                    # window exhaustion.  Keep the short _DEFAULT_RECHECK_SECS
                    # placeholder so the next poll can switch back instead of
                    # locking the backend out until the 5-hour window resets.
                    if (
                        source_status.available is False
                        and source_status.resets_at is not None
                    ):
                        with self._lock:
                            self._exhausted_until[name] = source_status.resets_at
                        exhausted_snapshot[name] = source_status.resets_at
                    if source_status.weekly_utilization is not None:
                        with self._lock:
                            self._last_weekly[name] = source_status.weekly_utilization
                            self._last_weekly_resets_at[name] = source_status.weekly_resets_at
                            self._last_weekly_window_hours[name] = source_status.weekly_window_hours
                        last_weekly_snapshot[name] = source_status.weekly_utilization
                except Exception:
                    pass

        best, statuses, new_exhausted, new_weekly = self._compute_best_unlocked(
            exhausted_snapshot, last_weekly_snapshot, subscription_only
        )

        if new_exhausted:
            with self._lock:
                self._exhausted_until.update(new_exhausted)
        if new_weekly:
            with self._lock:
                self._last_weekly.update(new_weekly)
        if statuses:
            observed_at = time.time()
            with self._lock:
                for backend_name, status in statuses.items():
                    self._last_status_at[backend_name] = observed_at
                    self._last_available[backend_name] = status.available
                    self._last_weekly_resets_at[backend_name] = status.weekly_resets_at
                    self._last_weekly_window_hours[backend_name] = status.weekly_window_hours

        if source_status is not None:
            statuses.setdefault(name, source_status)
        active = self._registry.active_name()
        if best is None:
            logger.debug(
                "Auto-selector: 429 on %s, no enabled backend available to switch to",
                name,
            )
        elif best != active:
            reason = _format_switch_reason(f"429 on {name}", name, best, statuses)
            result = self._registry.switch(best, reason=reason)
            if result.kind == "failed":
                logger.warning(
                    "Auto-selector: 429 on %s, switch to %s failed (%s)",
                    name,
                    best,
                    result.error,
                )
        return self._registry.active_name()

    def pin(self, name: str) -> None:
        """Pause auto-selection and lock the active backend to ``name``."""
        with self._lock:
            self._mode = "pinned"
            self._pinned = name
        logger.info("Auto-selector paused; pinned to %s", name)

    def resume(self) -> str:
        """Resume auto-selection (clear any pin/restriction) and immediately re-evaluate."""
        with self._lock:
            self._mode = "auto"
            self._pinned = None
        logger.info("Auto-selector resumed")
        return self.evaluate()

    def restrict_subscription(self) -> str:
        """Switch to subscription-only mode and immediately re-evaluate.

        Auto-selection continues but only among subscription backends
        (see ``SUBSCRIPTION_BACKENDS``); bedrock is never used as a fallback.
        Returns the name of the backend that becomes active.
        """
        with self._lock:
            self._mode = "subscription"
            self._pinned = None
        logger.info("Auto-selector restricted to subscription backends")
        return self.evaluate()

    def is_paused(self) -> bool:
        """True only in 'pinned' mode.

        Subscription mode is NOT paused — the reactive-429 retry path remains
        active so rotation between subscription backends still works.
        """
        with self._lock:
            return self._mode == "pinned"

    def pinned_name(self) -> str | None:
        with self._lock:
            return self._pinned

    def status_line(self) -> str:
        """One-line status string for proxy-status output."""
        with self._lock:
            if self._mode == "pinned":
                return f"auto-selection: paused (manual `{self._pinned}`)"
            if self._mode == "subscription":
                return "auto-selection: subscription-only"
        return "auto-selection: on"

    def personal_candidates(self) -> tuple[PersonalCandidate, ...]:
        now = time.time()
        with self._lock:
            exhausted = dict(self._exhausted_until)
            weekly = dict(self._last_weekly)
            status_at = dict(self._last_status_at)
            available = dict(self._last_available)
            resets_at = dict(self._last_weekly_resets_at)
            window_hours = dict(self._last_weekly_window_hours)
        pace_on = _pace_enabled(self._config)
        candidates = []
        for name in self._priority:
            if now < exhausted.get(name, 0) or available.get(name) is False:
                continue
            fresh = now - status_at.get(name, 0.0) <= _PERSONAL_CACHE_TTL
            burn = weekly.get(name, _UNKNOWN_WEEKLY) if fresh else _UNKNOWN_WEEKLY
            # Gate elapsed behind the freshness check: a stale reading must not
            # silently produce a contemporary elapsed% paired with a neutral burn.
            elapsed = (
                _weekly_elapsed_pct(resets_at.get(name), window_hours.get(name), now)
                if fresh else None
            )
            candidates.append(
                PersonalCandidate(name=name, burn=burn, weekly_elapsed_pct=elapsed)
            )
        return _sort_personal_candidates(candidates, pace_on)

    def current_subscription_backend(self) -> str | None:
        """Return a subscription backend name from cached state only.

        Reads ``_exhausted_until`` / ``_last_weekly`` under ``_lock``; does
        NO network I/O and does NOT call ``registry.*``.  Safe to call while
        the registry holds ``_state_lock`` (leaf lock ordering).

        Prefers a not-currently-exhausted sub with lowest known weekly
        utilization; falls back to the lowest-weekly subscription backend
        overall.  Can only return a member of the enabled subset of
        ``_PRIORITY`` or ``None`` — never bedrock or any other non-subscription
        or disabled backend.
        """
        now = time.time()
        with self._lock:
            exhausted = dict(self._exhausted_until)
            weekly = dict(self._last_weekly)
        available = [n for n in self._priority if now >= exhausted.get(n, 0)]
        pool = available if available else list(self._priority)
        pool.sort(key=lambda n: weekly.get(n, _UNKNOWN_WEEKLY))
        return pool[0] if pool else None

    def note_exhausted(self, name: str, retry_after: float | None = None) -> None:
        """Record ``name`` as exhausted WITHOUT switching the global backend.

        Used by the session-subscription 429 path: we return the 429 to the
        client but record the exhaustion so the next request's session
        resolver picks the other subscription backend.
        """
        with self._lock:
            self._mark_exhausted_locked(name, retry_after)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background poll loop: refresh tokens then (if auto-enabled) evaluate."""
        while not self._stop.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            if not self._stop.is_set():
                self._tick()

    def _tick(self) -> None:
        """One tick: refresh OAuth tokens and (when auto is active) re-evaluate."""
        if self._oauth_registry is not None:
            self._oauth_registry.tick(background=True)
        self._refresh_tokens()
        if self._config.auto_backend:
            try:
                self.evaluate()
            except Exception:
                logger.exception("Auto-selector background evaluation error")

    def _refresh_tokens(self) -> None:
        """Proactively refresh OAuth tokens for OAuth-backed subscription backends.

        OpenRouter is intentionally excluded: it authenticates with a static
        API key, not OAuth, so there is nothing to refresh. The OAuth-backed
        pair is filtered against the enabled backend set (ADR-0020 §7) —
        `self._priority` is the OAuth pool's superset (includes openrouter),
        so membership is checked directly rather than reused wholesale.
        """
        for name in ("anthropic", "codex"):
            if name not in self._priority:
                continue
            try:
                if name == "anthropic":
                    from .anthropic import auth as anthropic_auth

                    logger.debug("Auto-selector: checking anthropic credentials")
                    anthropic_auth.ensure_credentials_noninteractive(self._config)
                elif name == "codex":
                    from .codex import auth as codex_auth

                    logger.debug("Auto-selector: checking codex credentials")
                    codex_auth.ensure_credentials_noninteractive(self._config)
            except RuntimeError as exc:
                msg = str(exc)
                if "No " in msg and "credentials" in msg.lower():
                    logger.debug("Auto-selector: %s credentials absent, skipping", name)
                else:
                    logger.warning("Auto-selector: %s refresh failed: %s", name, exc)
            except Exception:
                logger.exception("Auto-selector: unexpected error refreshing %s", name)

    def _compute_best_unlocked(
        self,
        exhausted_snapshot: dict,
        last_weekly_snapshot: dict,
        subscription_only: bool = False,
    ) -> tuple:
        """Return (best_name, statuses_dict, new_exhausted_dict, new_weekly_dict).

        Must be called WITHOUT ``_lock`` held — this method makes usage
        endpoint network calls via ``five_hour_status()``.

        Decision flow (after building available_backends and best_exhausted):

        STEP 1 — Incumbent hold (applies when the active backend is itself a
        healthy available backend):
          - If the active backend is the lowest-weekly available, hold it.
          - Otherwise switch to the challenger only when the challenger's weekly
            is at least ``auto_backend_weekly_margin`` pp below the active's.
          - An unknown active weekly is treated as the neutral value
            ``_UNKNOWN_WEEKLY`` so a transient fetch gap does not force a switch.
          - When STEP 1 applies it returns unconditionally; parking is skipped.

        STEP 2 — Bedrock-parking (applies only when the active backend is NOT
        a healthy available one, e.g. just exhausted or currently bedrock):
          - Park on bedrock when an exhausted backend's weekly is decisively
            lower than the best available (gap >= margin), so weekly quota is
            preserved while waiting for the low-consumption backend to reset.
          - Skipped in subscription-only mode (bedrock is never used there).

        Fallback: return the best available backend.

        In subscription-only mode (``subscription_only=True``):
        - Same iteration; STEP 2 parking is skipped entirely.
        - When no subscription backend has a 5-hour window open, return the
          lowest-weekly subscription backend (it will 429; the reactive path
          cycles until one opens).
        """
        now = time.time()
        active = self._registry.active_name()
        statuses: dict = {}
        new_exhausted: dict = {}
        new_weekly: dict = {}

        # available_backends: [(name, weekly_utilization)] — 5-hour open
        available_backends: list = []
        # best_exhausted: (name, weekly_utilization) — exhausted, with known weekly
        best_exhausted: tuple | None = None

        for name in self._priority:
            in_cooldown = now < exhausted_snapshot.get(name, 0)

            if in_cooldown:
                # Don't probe; use last-known weekly for comparison.
                weekly = last_weekly_snapshot.get(name)
                if weekly is not None:
                    if best_exhausted is None or weekly < best_exhausted[1]:
                        best_exhausted = (name, weekly)
                continue

            try:
                backend = self._registry.instance(name)
            except Exception:
                continue

            five_hour_fn = getattr(backend, "five_hour_status", None)
            if five_hour_fn is None:
                continue

            try:
                status = five_hour_fn(self._config)
            except Exception:
                logger.debug(
                    "Auto-selector: five_hour_status(%s) raised", name, exc_info=True
                )
                if name == active:
                    return name, statuses, new_exhausted, new_weekly
                continue

            statuses[name] = status

            if status.weekly_utilization is not None:
                new_weekly[name] = status.weekly_utilization

            effective_weekly = (
                status.weekly_utilization
                if status.weekly_utilization is not None
                else last_weekly_snapshot.get(name)
            )

            if status.available is True:
                elapsed = _weekly_elapsed_pct(
                    status.weekly_resets_at, status.weekly_window_hours, now
                )
                available_backends.append((name, effective_weekly, elapsed))

            elif status.available is False:
                if status.resets_at is not None:
                    new_exhausted[name] = status.resets_at
                else:
                    new_exhausted[name] = now + _DEFAULT_RECHECK_SECS
                if effective_weekly is not None:
                    if best_exhausted is None or effective_weekly < best_exhausted[1]:
                        best_exhausted = (name, effective_weekly)

            else:  # available is None — transient error
                if name == active:
                    return name, statuses, new_exhausted, new_weekly
                # Don't switch to an unconfirmed backend.

        if not available_backends:
            if subscription_only:
                # Never fall back to bedrock in subscription mode.  Route to the
                # lowest-weekly subscription backend; it will 429, and the reactive
                # path keeps cycling until one opens.
                if best_exhausted is not None:
                    return best_exhausted[0], statuses, new_exhausted, new_weekly
                if self._priority:
                    return self._priority[0], statuses, new_exhausted, new_weekly
                # No subscription backend is enabled at all (ADR-0020 Known
                # Limitations). Nothing to route to; caller no-ops on active.
                return None, statuses, new_exhausted, new_weekly
            if self._fallback is not None:
                return self._fallback, statuses, new_exhausted, new_weekly
            # bedrock is excluded and no subscription backend is available.
            # Prefer a known-exhausted subscription backend over returning
            # nothing, since it will at least surface a 429 to retry against.
            if best_exhausted is not None:
                return best_exhausted[0], statuses, new_exhausted, new_weekly
            return None, statuses, new_exhausted, new_weekly

        # Order available backends for the ranking sort key only.  The same
        # _pace_rank_key used by _sort_personal_candidates drives the split-branch
        # ordering here so the two sites cannot drift.  Unknown weekly →
        # _UNKNOWN_WEEKLY (50%) neutral value.  STEP 1/STEP 2 below continue to
        # compare raw weekly, never the delta.
        pace_on = _pace_enabled(self._config)
        available_backends.sort(
            key=lambda e: _pace_rank_key(
                e[1] if e[1] is not None else _UNKNOWN_WEEKLY, e[2], pace_on,
            )
        )
        best_avail_name, best_avail_weekly = available_backends[0][0], available_backends[0][1]
        best_avail_val = (
            best_avail_weekly if best_avail_weekly is not None else _UNKNOWN_WEEKLY
        )

        margin = getattr(self._config, "auto_backend_weekly_margin", _UNKNOWN_WEEKLY)

        # STEP 1: If the active backend is itself a healthy available backend,
        # apply incumbent stickiness.  Return unconditionally here so STEP 2
        # (bedrock-parking) is unreachable for a healthy incumbent.
        active_entry = next((e for e in available_backends if e[0] == active), None)
        if active_entry is not None:
            # Unknown incumbent weekly → neutral value (hold unless challenger
            # is decisively lower than the neutral midpoint).
            active_val = (
                active_entry[1] if active_entry[1] is not None else _UNKNOWN_WEEKLY
            )
            if best_avail_name == active:
                logger.debug(
                    "Auto-selector: holding active %s (already lowest-weekly available, "
                    "weekly=%.1f%%)",
                    active,
                    active_val,
                )
                return active, statuses, new_exhausted, new_weekly
            if active_val - best_avail_val >= margin:
                logger.debug(
                    "Auto-selector: switching %s (weekly=%.1f%%) -> %s (weekly=%.1f%%); "
                    "delta %.1f pp >= margin %.1f pp",
                    active,
                    active_val,
                    best_avail_name,
                    best_avail_val,
                    active_val - best_avail_val,
                    margin,
                )
                return best_avail_name, statuses, new_exhausted, new_weekly
            logger.debug(
                "Auto-selector: holding active %s (weekly=%.1f%%) over %s (weekly=%.1f%%); "
                "delta %.1f pp < margin %.1f pp",
                active,
                active_val,
                best_avail_name,
                best_avail_val,
                active_val - best_avail_val,
                margin,
            )
            return active, statuses, new_exhausted, new_weekly

        # STEP 2: Active backend is not a healthy available one (just exhausted,
        # currently bedrock, or otherwise absent from the available pool).
        # Park on bedrock when an exhausted backend is decisively lower than
        # the best available so weekly quota is preserved while we wait.
        # Skip in subscription-only mode — bedrock is never used there.
        if (
            not subscription_only
            and self._fallback is not None
            and best_exhausted is not None
            and best_avail_val - best_exhausted[1] >= margin
        ):
            logger.debug(
                "Auto-selector: %s exhausted at weekly %.1f%%, decisively below best-available "
                "%s %.1f%% (delta %.1f pp >= margin %.1f pp); waiting on bedrock",
                best_exhausted[0],
                best_exhausted[1],
                best_avail_name,
                best_avail_val,
                best_avail_val - best_exhausted[1],
                margin,
            )
            return self._fallback, statuses, new_exhausted, new_weekly

        return best_avail_name, statuses, new_exhausted, new_weekly

    def _mark_exhausted_locked(self, name: str, retry_after: float | None):
        """Record ``name`` as exhausted (called with ``_lock`` held).

        Returns the backend instance that should have ``five_hour_status``
        called on it (outside the lock) if ``retry_after`` is None, or None if
        the retry_after header already gave us a precise reset time.
        """
        now = time.time()
        if retry_after is not None and retry_after > 0:
            self._exhausted_until[name] = now + retry_after
            return None

        self._exhausted_until[name] = now + _DEFAULT_RECHECK_SECS
        try:
            backend = self._registry.instance(name)
            _invalidate_usage_cache(backend)
            return backend
        except Exception:
            return None


def _invalidate_usage_cache(backend) -> None:
    """Force-expire cached successful usage on a backend."""
    from ._shared import SubscriptionBackend

    if not isinstance(backend, SubscriptionBackend):
        return
    backend.invalidate_usage_cache()


def _format_switch_reason(trigger: str, source: str, dest: str, statuses: dict) -> str:
    """Build a log reason string including weekly utilization when available."""
    parts = [trigger]
    for name in (source, dest):
        status = statuses.get(name)
        if status is None:
            continue
        weekly = getattr(status, "weekly_utilization", None)
        utilization = getattr(status, "utilization", None)
        if weekly is not None:
            parts.append(f"{name}: {weekly:.1f}% weekly")
        elif utilization is not None:
            parts.append(f"{name}: {utilization:.1f}% used")
    return "; ".join(parts)
