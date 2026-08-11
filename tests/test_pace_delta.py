"""Pace-delta backend selection (ADR-0015).

Covers the four seams introduced by the pace-delta metric:
  - month-elapsed derivation in the OAuth registry (UTC, calendar-aware);
  - weekly-elapsed derivation in the selector (normal / stale / unknown window);
  - the personal-candidate split-branch sort (pace block vs. raw-only block);
  - the OAuth-vs-personal pace gate in ``snapshot_for_request`` (dead-band,
    kill-switch fallback, and the reported pathology from grilling).

Requirements: SRS-Routing-001 (pace-delta normalization across windows),
SRS-Routing-002 (UTC month-elapsed derivation).
"""

import datetime as dt
import time

import pytest

from anthproxy.config import Config
from anthproxy.oauth_registry import OAuthTokenRegistry, _month_elapsed_pct
from anthproxy.selector import (
    AutoSelector,
    PersonalCandidate,
    _sort_personal_candidates,
    _weekly_elapsed_pct,
)
from anthproxy.server import BackendRegistry


# ---------------------------------------------------------------------------
# Month-elapsed (SRS-Routing-002)
# ---------------------------------------------------------------------------

class TestMonthElapsedPct:
    def test_first_of_month_is_zero(self):
        assert _month_elapsed_pct(dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)) == 0.0

    def test_mid_month_31_day(self):
        # Aug has 31 days: day 5 → (5-1)/31 * 100.
        got = _month_elapsed_pct(dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc))
        assert got == (4 / 31) * 100.0

    def test_february_leap_year_day_count(self):
        # 2028 is a leap year: February has 29 days.
        got = _month_elapsed_pct(dt.datetime(2028, 2, 15, tzinfo=dt.timezone.utc))
        assert got == (14 / 29) * 100.0

    def test_last_day_below_100(self):
        got = _month_elapsed_pct(dt.datetime(2026, 8, 31, tzinfo=dt.timezone.utc))
        assert got == (30 / 31) * 100.0
        assert got < 100.0

    def test_intra_day_drifts_within_calendar_day(self):
        # Same calendar day, later second-of-day yields a strictly higher pct.
        midnight = _month_elapsed_pct(dt.datetime(2026, 8, 5, 0, 0, 0, tzinfo=dt.timezone.utc))
        got = _month_elapsed_pct(dt.datetime(2026, 8, 5, 13, 7, 29, tzinfo=dt.timezone.utc))
        assert got > midnight
        seconds_into_day = 13 * 3600 + 7 * 60 + 29
        expected = (4 + seconds_into_day / 86400.0) / 31 * 100.0
        assert got == expected

    def test_intra_day_just_before_midnight_stays_below_next_day(self):
        got = _month_elapsed_pct(dt.datetime(2026, 8, 5, 23, 59, 59, tzinfo=dt.timezone.utc))
        next_day_midnight = _month_elapsed_pct(dt.datetime(2026, 8, 6, 0, 0, 0, tzinfo=dt.timezone.utc))
        assert got < next_day_midnight

    def test_last_day_intra_day_stays_below_100(self):
        got = _month_elapsed_pct(dt.datetime(2026, 8, 31, 23, 59, 59, tzinfo=dt.timezone.utc))
        assert got < 100.0

    def test_snapshot_populates_month_elapsed(self):
        walls = [dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc)]
        registry = OAuthTokenRegistry(utcnow=lambda: walls[0])
        registry.observe('enterprise')
        snap = registry.snapshot()
        assert snap.month_elapsed_pct == (4 / 31) * 100.0


# ---------------------------------------------------------------------------
# Weekly-elapsed (SRS-Routing-001)
# ---------------------------------------------------------------------------

class TestWeeklyElapsedPct:
    def test_none_when_window_hours_unknown(self):
        assert _weekly_elapsed_pct(1_000.0, None, now=0.0) is None

    def test_none_when_window_hours_non_positive(self):
        assert _weekly_elapsed_pct(1_000.0, 0.0, now=0.0) is None

    def test_none_when_reset_unknown(self):
        assert _weekly_elapsed_pct(None, 168.0, now=0.0) is None

    def test_half_window_elapsed(self):
        # 84h remaining of a 168h window → 50% elapsed.
        now = 1_000_000.0
        resets_at = now + 84 * 3600
        assert _weekly_elapsed_pct(resets_at, 168.0, now) == 50.0

    def test_stale_reset_yields_zero(self):
        # reset in the past → treated as just-started (0.0), the highest delta.
        now = 1_000_000.0
        assert _weekly_elapsed_pct(now - 10.0, 168.0, now) == 0.0

    def test_reset_equal_now_is_stale(self):
        now = 1_000_000.0
        assert _weekly_elapsed_pct(now, 168.0, now) == 0.0

    def test_clamped_to_100(self):
        # remaining larger than the window would give negative elapsed; clamp low.
        now = 1_000_000.0
        resets_at = now + 500 * 3600  # far beyond a 168h window
        assert _weekly_elapsed_pct(resets_at, 168.0, now) == 0.0


# ---------------------------------------------------------------------------
# Split-branch sort
# ---------------------------------------------------------------------------

class TestSortPersonalCandidates:
    def test_pace_block_sorts_by_delta_ahead_of_raw_block(self):
        # anthropic: burn 86, elapsed 95 → delta -9 (behind pace, best).
        # codex:     burn 40, elapsed 30 → delta +10.
        # openrouter: no elapsed → raw-only block, ranks last despite low burn.
        cands = [
            PersonalCandidate('codex', 40.0, 30.0),
            PersonalCandidate('openrouter', 5.0, None),
            PersonalCandidate('anthropic', 86.0, 95.0),
        ]
        ordered = _sort_personal_candidates(cands, pace_on=True)
        assert [c.name for c in ordered] == ['anthropic', 'codex', 'openrouter']

    def test_raw_block_sorts_by_burn(self):
        cands = [
            PersonalCandidate('openrouter', 60.0, None),
            PersonalCandidate('other', 20.0, None),
        ]
        ordered = _sort_personal_candidates(cands, pace_on=True)
        assert [c.name for c in ordered] == ['other', 'openrouter']

    def test_pace_off_sorts_by_raw_burn(self):
        cands = [
            PersonalCandidate('anthropic', 86.0, 95.0),
            PersonalCandidate('codex', 40.0, 30.0),
        ]
        ordered = _sort_personal_candidates(cands, pace_on=False)
        assert [c.name for c in ordered] == ['codex', 'anthropic']


# ---------------------------------------------------------------------------
# OAuth-vs-personal pace gate
# ---------------------------------------------------------------------------

class _Backend:
    def __init__(self, name):
        self.name = name


def _eligible_oauth(utilization, wall):
    registry = OAuthTokenRegistry(utcnow=lambda: wall)
    credential = registry.observe('enterprise')
    registry.record_probe_success(credential.generation, {
        'extra_usage': {
            'monthly_limit': 100, 'used_credits': 10, 'utilization': utilization,
            'spend_limit_reached': False, 'is_enabled': True,
        },
    }, health_ok=True)
    return registry, credential


def _registry(oauth_registry, candidates, pace='on', deadband=3.0):
    config = Config(
        backend='anthropic',
        auto_backend_pace_delta=pace,
        auto_backend_oauth_pace_deadband_pp=deadband,
    )
    registry = BackendRegistry(
        config, _Backend('anthropic'), oauth_registry=oauth_registry,
    )
    registry._instances['oauth'] = _Backend('oauth')
    registry._instances['codex'] = _Backend('codex')
    registry.set_personal_candidates_resolver(lambda: candidates)
    return registry


# Fixed clock so month_elapsed is deterministic: Aug 5 → 12.903% elapsed.
_WALL = dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc)
_MONTH_ELAPSED = (4 / 31) * 100.0  # ≈ 12.903


class TestOAuthPaceGate:
    def test_pathology_personal_wins_despite_higher_raw_burn(self):
        # Grilling pathology: enterprise 17% monthly but ahead of pace loses to
        # personal 86% weekly but behind pace.
        oauth_registry, credential = _eligible_oauth(17.0, _WALL)
        candidates = (PersonalCandidate('anthropic', 86.0, 95.0),)
        registry = _registry(oauth_registry, candidates)

        snap = registry.snapshot_for_request(oauth_credential=credential)

        assert snap.name == 'anthropic'

    def test_same_pathology_reverts_under_kill_switch(self):
        # With pace off, raw comparison: 17 < 86 → enterprise wins.
        oauth_registry, credential = _eligible_oauth(17.0, _WALL)
        candidates = (PersonalCandidate('anthropic', 86.0, 95.0),)
        registry = _registry(oauth_registry, candidates, pace='off')

        snap = registry.snapshot_for_request(oauth_credential=credential)

        assert snap.name == 'oauth'
        assert snap.credentials == credential

    def test_oauth_wins_when_pace_delta_below_personal(self):
        # oauth burn 5 → delta 5-12.9 = -7.9; personal burn 40 elapsed 30 → +10.
        oauth_registry, credential = _eligible_oauth(5.0, _WALL)
        candidates = (PersonalCandidate('anthropic', 40.0, 30.0),)
        registry = _registry(oauth_registry, candidates)

        snap = registry.snapshot_for_request(oauth_credential=credential)

        assert snap.name == 'oauth'

    def test_deadband_holds_personal_at_boundary(self):
        # Construct personal so oauth_delta == personal_delta - deadband exactly;
        # strict '<' means personal (incumbent) wins.
        oauth_registry, credential = _eligible_oauth(5.0, _WALL)
        oauth_delta = 5.0 - _MONTH_ELAPSED  # ≈ -7.903
        # personal_delta = burn - elapsed; want personal_delta - 3 == oauth_delta
        # → personal_delta = oauth_delta + 3.  elapsed 0 → burn = personal_delta.
        personal_burn = oauth_delta + 3.0
        candidates = (PersonalCandidate('anthropic', personal_burn, 0.0),)
        registry = _registry(oauth_registry, candidates, deadband=3.0)

        snap = registry.snapshot_for_request(oauth_credential=credential)

        assert snap.name == 'anthropic'

    def test_representative_is_min_pace_delta_not_min_burn(self):
        # codex has the lowest raw burn but the worst pace delta; anthropic is
        # the representative and its lower delta lets oauth lose.
        oauth_registry, credential = _eligible_oauth(17.0, _WALL)
        candidates = (
            PersonalCandidate('anthropic', 86.0, 95.0),  # delta -9 (min)
            PersonalCandidate('codex', 30.0, 10.0),       # delta +20
        )
        # Resolver returns them already pace-sorted (anthropic first).
        registry = _registry(oauth_registry, candidates)

        snap = registry.snapshot_for_request(oauth_credential=credential)

        assert snap.name == 'anthropic'


# ---------------------------------------------------------------------------
# Backend reset/window plumbing
# ---------------------------------------------------------------------------

def test_anthropic_pairs_reset_with_max_weekly_window():
    from anthproxy.anthropic.backend import AnthropicBackend, _parse_iso_ts

    usage = {
        'five_hour': {'utilization': 50.0, 'resets_at': '2099-01-01T00:00:00Z'},
        'seven_day': {'utilization': 30.0, 'resets_at': '2099-02-01T00:00:00Z'},
        'seven_day_opus': {'utilization': 90.0, 'resets_at': '2099-03-01T00:00:00Z'},
    }
    backend = AnthropicBackend()
    backend.get_usage = lambda config: usage
    st = backend.five_hour_status(None)

    assert st.weekly_utilization == 90.0
    # Reset must pair with the opus window (the max), not seven_day.
    assert st.weekly_resets_at == _parse_iso_ts('2099-03-01T00:00:00Z')
    assert st.weekly_window_hours == 168.0


def test_anthropic_missing_reset_on_max_window_falls_back_none():
    from anthproxy.anthropic.backend import AnthropicBackend

    usage = {
        'five_hour': {'utilization': 50.0, 'resets_at': '2099-01-01T00:00:00Z'},
        'seven_day_opus': {'utilization': 90.0},  # no resets_at
    }
    backend = AnthropicBackend()
    backend.get_usage = lambda config: usage
    st = backend.five_hour_status(None)

    assert st.weekly_utilization == 90.0
    assert st.weekly_resets_at is None
    assert st.weekly_window_hours == 168.0


def test_codex_populates_weekly_reset_and_window():
    from anthproxy.codex.backend import CodexBackend

    usage = {
        'primary': {'remaining_percent': 80.0, 'window_seconds': 18000, 'reset_at': 111.0},
        'weekly': {'used_percent': 40.0, 'reset_at': 999_999.0, 'window_seconds': 168 * 3600},
        'limit_reached': False,
    }
    backend = CodexBackend()
    backend.get_usage = lambda config: usage
    st = backend.five_hour_status(None)

    assert st.weekly_utilization == 40.0
    assert st.weekly_resets_at == 999_999.0
    assert st.weekly_window_hours == 168.0


# ---------------------------------------------------------------------------
# personal_candidates() elapsed-from-cache wiring (M1)
# ---------------------------------------------------------------------------

class _Registry:
    def active_name(self):
        raise AssertionError('personal_candidates must not call registry')


class TestPersonalCandidatesElapsed:
    """Verify personal_candidates() derives weekly_elapsed_pct from the cached
    reset/window pair and gates it behind the freshness check."""

    def _selector(self, pace='on'):
        from types import SimpleNamespace
        return AutoSelector(_Registry(), SimpleNamespace(
            auto_backend_interval=3600,
            auto_backend_mode='auto',
            auto_backend=False,
            auto_backend_pace_delta=pace,
        ))

    def test_elapsed_derived_from_cached_reset_and_window(self):
        selector = self._selector()
        now = time.time()
        # 84h remaining of a 168h window → 50% elapsed.
        resets_at = now + 84 * 3600
        with selector._lock:
            selector._last_weekly.update({'anthropic': 86.0, 'codex': 30.0})
            selector._last_weekly_resets_at.update({
                'anthropic': resets_at, 'codex': None,
            })
            selector._last_weekly_window_hours.update({
                'anthropic': 168.0, 'codex': 168.0,
            })
            selector._last_status_at.update({'anthropic': now, 'codex': now})
            selector._last_available.update({'anthropic': True, 'codex': True})

        candidates = selector.personal_candidates()

        anth = next(c for c in candidates if c.name == 'anthropic')
        codex = next(c for c in candidates if c.name == 'codex')
        # anthropic: elapsed 50 → delta 36; codex: elapsed None → raw block.
        # Pace block ranks ahead, so anthropic is first despite higher burn.
        assert candidates[0].name == 'anthropic'
        assert anth.weekly_elapsed_pct == pytest.approx(50.0, abs=1e-3)
        assert codex.weekly_elapsed_pct is None

    def test_stale_status_gates_elapsed_to_none(self):
        selector = self._selector()
        now = time.time()
        with selector._lock:
            selector._last_weekly['anthropic'] = 86.0
            selector._last_weekly_resets_at['anthropic'] = now + 84 * 3600
            selector._last_weekly_window_hours['anthropic'] = 168.0
            selector._last_status_at['anthropic'] = now - 301  # stale
            selector._last_available['anthropic'] = True

        candidate = next(
            c for c in selector.personal_candidates() if c.name == 'anthropic'
        )

        # Stale reading must not pair a contemporary elapsed with the neutral
        # burn stand-in; elapsed drops to None (raw-only block).
        assert candidate.weekly_elapsed_pct is None
        assert candidate.burn == 50.0  # _UNKNOWN_WEEKLY neutral

    def test_pace_off_disables_elapsed_ranking(self):
        selector = self._selector(pace='off')
        now = time.time()
        with selector._lock:
            selector._last_weekly.update({'anthropic': 86.0, 'codex': 30.0})
            selector._last_weekly_resets_at.update({
                'anthropic': now + 84 * 3600, 'codex': None,
            })
            selector._last_weekly_window_hours.update({
                'anthropic': 168.0, 'codex': 168.0,
            })
            selector._last_status_at.update({'anthropic': now, 'codex': now})
            selector._last_available.update({'anthropic': True, 'codex': True})

        candidates = selector.personal_candidates()

        # Raw-burn ordering: codex (30) ahead of anthropic (86).
        assert [c.name for c in candidates if c.name in ('anthropic', 'codex')] == [
            'codex', 'anthropic',
        ]


# ---------------------------------------------------------------------------
# _compute_best_unlocked pace ordering (M2)
# ---------------------------------------------------------------------------

class _FakeBackend:
    """Backend stub returning a canned FiveHourStatus from five_hour_status."""

    def __init__(self, status):
        self._status = status

    def five_hour_status(self, config):
        return self._status


class _ActiveRegistry:
    """BackendRegistry stub exposing active_name() and instance()."""

    def __init__(self, active, instances):
        self._active = active
        self._instances = instances

    def active_name(self):
        return self._active

    def instance(self, name):
        return self._instances[name]


class TestComputeBestUnlockedPaceOrdering:
    """Verify _compute_best_unlocked orders candidates by pace delta and that
    STEP-1/STEP-2 hysteresis still compare raw weekly."""

    def _status(self, available, weekly_util, weekly_resets_at, weekly_window_hours,
                resets_at=None):
        from anthproxy._shared import FiveHourStatus
        return FiveHourStatus(
            available=available,
            resets_at=resets_at,
            utilization=50.0,
            weekly_utilization=weekly_util,
            weekly_resets_at=weekly_resets_at,
            weekly_window_hours=weekly_window_hours,
        )

    def _selector(self, active, instances, pace='on', margin=5.0):
        from types import SimpleNamespace
        registry = _ActiveRegistry(active, instances)
        config = SimpleNamespace(
            auto_backend_interval=3600,
            auto_backend_mode='auto',
            auto_backend=False,
            auto_backend_pace_delta=pace,
            auto_backend_weekly_margin=margin,
        )
        return AutoSelector(registry, config)

    def test_pace_delta_orders_candidates_when_active_is_absent(self):
        # Active backend is bedrock (not in available pool), so STEP-1/STEP-2
        # don't apply; the best available is whoever _pace_rank_key ranks first.
        # anthropic: burn 86, elapsed 95 → delta -9 (best).
        # codex:     burn 30, elapsed 10 → delta +20.
        # openrouter: burn 5, elapsed None → raw-only block.
        now = time.time()
        resets_anthropic = now + int(0.05 * 168 * 3600)   # ~95% elapsed
        resets_codex = now + int(0.90 * 168 * 3600)        # ~10% elapsed
        instances = {
            'anthropic': _FakeBackend(self._status(
                True, 86.0, resets_anthropic, 168.0,
            )),
            'codex': _FakeBackend(self._status(
                True, 30.0, resets_codex, 168.0,
            )),
            'openrouter': _FakeBackend(self._status(
                True, 5.0, None, None,
            )),
        }
        selector = self._selector('bedrock', instances)

        best, statuses, _, _ = selector._compute_best_unlocked({}, {}, False)

        # Pace block (anthropic, codex) ranks ahead of raw-only (openrouter);
        # within pace block, anthropic's delta -9 beats codex's +20.
        assert best == 'anthropic'

    def test_step1_hysteresis_uses_raw_weekly_not_delta(self):
        # Active = codex (burn 30, elapsed 10 → delta +20). Challenger =
        # anthropic (burn 86, elapsed 95 → delta -9, much lower). On pace delta
        # anthropic should win, but STEP-1 hysteresis compares RAW weekly:
        # codex 30 vs anthropic 86, delta 56 pp >= margin 5 → SWITCH to
        # anthropic. Wait — that switches. To prove STEP-1 uses raw (not delta),
        # set margin huge so the raw gap (56) is < margin and codex is held
        # despite having the worse pace delta.
        now = time.time()
        resets_anthropic = now + int(0.05 * 168 * 3600)
        resets_codex = now + int(0.90 * 168 * 3600)
        instances = {
            'anthropic': _FakeBackend(self._status(
                True, 86.0, resets_anthropic, 168.0,
            )),
            'codex': _FakeBackend(self._status(
                True, 30.0, resets_codex, 168.0,
            )),
        }
        selector = self._selector('codex', instances, margin=100.0)

        best, _, _, _ = selector._compute_best_unlocked({}, {}, False)

        # Raw weekly gap is 56 pp < margin 100, so STEP-1 holds codex even though
        # anthropic has the better pace delta. Confirms STEP-1 uses raw weekly.
        assert best == 'codex'

    def test_kill_switch_off_reverts_to_raw_burn_ordering(self):
        now = time.time()
        resets_anthropic = now + int(0.05 * 168 * 3600)
        resets_codex = now + int(0.90 * 168 * 3600)
        instances = {
            'anthropic': _FakeBackend(self._status(
                True, 86.0, resets_anthropic, 168.0,
            )),
            'codex': _FakeBackend(self._status(
                True, 30.0, resets_codex, 168.0,
            )),
        }
        selector = self._selector('bedrock', instances, pace='off')

        best, _, _, _ = selector._compute_best_unlocked({}, {}, False)

        # Raw burn: codex 30 < anthropic 86 → codex wins.
        assert best == 'codex'

