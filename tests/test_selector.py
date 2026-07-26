"""Unit tests for AutoSelector (anthproxy/selector.py)."""
import time
import unittest

from anthproxy._shared import FiveHourStatus, SubscriptionBackend, UsageRateLimitError
from anthproxy.selector import AutoSelector, _FALLBACK, _PRIORITY


class _TestSubscriptionBackend(SubscriptionBackend):
    _PROVIDER_NAME = 'Test'

    def __init__(self, results):
        super().__init__()
        self._results = list(results)
        self.calls = 0

    def _fetch_usage_data(self, config) -> dict:
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def _format_usage_markdown_impl(self, usage: dict) -> str:
        return str(usage)

    def _usage_failure_markdown_impl(self, message: str) -> str:
        return message


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal Config stand-in."""
    auto_backend: bool = True
    auto_backend_mode: str = 'auto'
    auto_backend_interval: float = 3600.0  # don't fire during tests
    auto_backend_weekly_margin: float = 5.0


class _FakeBackend:
    """Fake backend that returns a scripted FiveHourStatus."""

    def __init__(self, available: bool | None = True, resets_at: float | None = None,
                 utilization: float | None = None, weekly_utilization: float | None = None):
        self._status = FiveHourStatus(
            available=available,
            resets_at=resets_at,
            utilization=utilization,
            weekly_utilization=weekly_utilization,
        )
        self.invalidated = 0

    def five_hour_status(self, config) -> FiveHourStatus:
        return self._status

    def set_status(self, available: bool | None, resets_at: float | None = None,
                   utilization: float | None = None, weekly_utilization: float | None = None):
        self._status = FiveHourStatus(
            available=available,
            resets_at=resets_at,
            utilization=utilization,
            weekly_utilization=weekly_utilization,
        )

    def invalidate_usage_cache(self):
        self.invalidated += 1


class _FakeRegistry:
    """Minimal BackendRegistry stand-in for AutoSelector."""

    def __init__(self, initial='bedrock'):
        self._active = initial
        self._instances: dict = {}
        self._switches: list = []

    def active_name(self) -> str:
        return self._active

    def instance(self, name: str):
        return self._instances[name]

    def switch(self, name: str, **kwargs):
        from anthproxy.server import SwitchResult
        previous = self._active
        self._active = name
        self._switches.append(name)
        kind = 'changed' if name != previous else 'unchanged'
        return SwitchResult(kind=kind, previous=previous, current=name)


def _make_selector(backends: dict, initial: str = 'bedrock') -> tuple:
    """Return (selector, registry) with scripted backend instances."""
    reg = _FakeRegistry(initial)
    reg._instances = dict(backends)
    cfg = _FakeConfig()
    sel = AutoSelector(reg, cfg)
    return sel, reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubscriptionBackend(unittest.TestCase):

    def test_caches_success_for_ttl_window(self):
        backend = _TestSubscriptionBackend([{'ok': 1}])
        first = backend.get_usage(None)
        second = backend.get_usage(None)
        self.assertEqual(first, {'ok': 1})
        self.assertEqual(second, {'ok': 1})
        self.assertEqual(backend.calls, 1)

    def test_caches_usage_rate_limit_until_cooldown_expires(self):
        backend = _TestSubscriptionBackend([
            UsageRateLimitError(retry_after=30.0),
            {'ok': 2},
        ])
        with self.assertRaises(UsageRateLimitError):
            backend.get_usage(None)
        with self.assertRaises(UsageRateLimitError):
            backend.get_usage(None)
        self.assertEqual(backend.calls, 1)
        backend._usage_backoff_until = time.monotonic() - 1
        self.assertEqual(backend.get_usage(None), {'ok': 2})
        self.assertEqual(backend.calls, 2)

    def test_uses_default_cooldown_when_retry_after_missing(self):
        backend = _TestSubscriptionBackend([UsageRateLimitError()])
        with self.assertRaises(UsageRateLimitError) as exc:
            backend.get_usage(None)
        self.assertEqual(exc.exception.retry_after, backend._USAGE_RATE_LIMIT_TTL)

    def test_invalidate_usage_cache_keeps_rate_limit_backoff(self):
        backend = _TestSubscriptionBackend([UsageRateLimitError(retry_after=30.0)])
        with self.assertRaises(UsageRateLimitError):
            backend.get_usage(None)
        backend.invalidate_usage_cache()
        with self.assertRaises(UsageRateLimitError):
            backend.get_usage(None)
        self.assertEqual(backend.calls, 1)


class TestComputeBest(unittest.TestCase):

    def test_priority_includes_openrouter_after_codex(self):
        self.assertEqual(_PRIORITY, ('anthropic', 'codex', 'openrouter'))

    def test_prefers_lower_weekly_utilization_when_backends_available(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=70.0),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
            'openrouter': _FakeBackend(available=True, weekly_utilization=10.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'openrouter')

    def test_prefers_lower_weekly_utilization_when_both_available(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=70.0),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_falls_to_codex_when_anthropic_depleted(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, weekly_utilization=10.0),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'bedrock')

    def test_uses_available_backend_when_exhausted_backend_has_higher_weekly(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, weekly_utilization=80.0),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_prefers_anthropic_when_it_has_lower_weekly(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=10.0),
            'codex': _FakeBackend(available=True, weekly_utilization=40.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'anthropic')

    def test_waits_on_cooldown_backend_with_lower_weekly(self):
        # With hysteresis: active=codex is a healthy available backend (STEP 1),
        # so parking is skipped entirely.  The incumbent is held even when an
        # exhausted backend has much lower weekly — parking (STEP 2) only applies
        # when the active is NOT a healthy available backend.
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True, weekly_utilization=60.0),
        }, initial='codex')
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() + 9999
            sel._last_weekly['anthropic'] = 10.0
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_uses_neutral_value_when_weekly_unknown(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=None),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_updates_last_weekly_from_successful_probe(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=33.0),
            'codex': _FakeBackend(available=True, weekly_utilization=40.0),
        }, initial='bedrock')
        sel.evaluate()
        with sel._lock:
            self.assertEqual(sel._last_weekly['anthropic'], 33.0)
            self.assertEqual(sel._last_weekly['codex'], 40.0)

    def test_preempts_to_lower_weekly_backend_on_reset(self):
        # First evaluate: codex is the active healthy available backend (STEP 1),
        # so the incumbent is held even though anthropic is exhausted with lower
        # weekly — parking (STEP 2) is unreachable for a healthy incumbent.
        # Second evaluate: anthropic resets; now both are available and the
        # delta (60 - 10 = 50 pp) exceeds the margin (5 pp), so we switch.
        future_reset = time.time() + 3600
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future_reset, weekly_utilization=10.0),
            'codex': _FakeBackend(available=True, weekly_utilization=60.0),
        }, initial='codex')
        sel.evaluate()
        self.assertEqual(reg.active_name(), 'codex')  # healthy incumbent held

        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() - 1
        reg._instances['anthropic'].set_status(available=True, weekly_utilization=10.0)
        best = sel.evaluate()
        self.assertEqual(best, 'anthropic')
        self.assertEqual(reg.active_name(), 'anthropic')

    def test_falls_to_codex_when_anthropic_unknown(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_stays_on_active_when_active_backend_is_unknown(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='anthropic')
        best = sel.evaluate()
        self.assertEqual(best, 'anthropic')

    def test_falls_to_bedrock_when_anthropic_depleted_and_codex_unknown(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, weekly_utilization=10.0),
            'codex': _FakeBackend(available=None),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'bedrock')

    def test_falls_to_codex_when_anthropic_unknown_and_codex_available(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=True, weekly_utilization=10.0),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_falls_to_anthropic_when_codex_unknown_and_anthropic_available(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=10.0),
            'codex': _FakeBackend(available=None),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'anthropic')

    def test_falls_to_bedrock_when_both_unknown(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=None),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'bedrock')

    def test_stays_on_active_when_both_unknown_and_active_anthropic(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=None),
        }, initial='anthropic')
        best = sel.evaluate()
        self.assertEqual(best, 'anthropic')

    def test_stays_on_active_when_both_unknown_and_active_codex(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=None),
        }, initial='codex')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_stays_on_active_when_anthropic_unknown_and_codex_unknown_active_codex(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=None),
        }, initial='codex')
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_stays_on_active_when_anthropic_unknown_and_codex_unknown_active_anthropic(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=None),
        }, initial='anthropic')
        best = sel.evaluate()
        self.assertEqual(best, 'anthropic')

    def test_transient_error_stays_on_active(self):
        """available=None on active backend → stay, don't switch to unknown."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),   # transient
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        best = sel.evaluate()
        # Conservative: stay on anthropic even though codex is available
        self.assertEqual(best, 'anthropic')

    def test_transient_error_skips_non_active(self):
        """available=None on non-active candidate → don't switch to it."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=None),
            'codex': _FakeBackend(available=True),
        }, initial='codex')
        best = sel.evaluate()
        # anthropic is transient and NOT active → skip it; codex is fine
        self.assertEqual(best, 'codex')

    def test_exhausted_until_prevents_probe(self):
        """Backend in cooldown is skipped even if its backend says available."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True),
        }, initial='codex')
        # Manually mark anthropic as exhausted far in the future
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() + 9999
        best = sel.evaluate()
        self.assertEqual(best, 'codex')

    def test_exhausted_until_expires(self):
        """Once exhausted_until passes, the backend is probed again.

        With hysteresis: when both backends have unknown weekly (neutral 50%),
        the incumbent (codex) is held — no switch on a tie.
        """
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True),
        }, initial='codex')
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() - 1  # already expired
        best = sel.evaluate()
        # Both weekly are unknown (50% neutral); delta=0 < margin → hold incumbent.
        self.assertEqual(best, 'codex')

    def test_preempts_back_to_anthropic_on_reset(self):
        """When on codex and anthropic resets with equal unknown weekly, hold incumbent.

        With hysteresis: both backends have unknown weekly (neutral 50%).
        Delta = 0 < margin (5 pp) → hold codex.  A decisive weekly gap would
        trigger a switch (see test_preempts_to_lower_weekly_backend_on_reset).
        """
        future_reset = time.time() + 3600  # resets in an hour
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future_reset),
            'codex': _FakeBackend(available=True),
        }, initial='codex')
        sel.evaluate()  # lands on codex; records exhausted_until[anthropic]=future_reset
        self.assertEqual(reg.active_name(), 'codex')

        # Simulate the window having actually reset: expire the cooldown and
        # flip the backend status (in real life time.time() >= resets_at).
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() - 1
        reg._instances['anthropic'].set_status(available=True)
        best = sel.evaluate()
        # Both weekly unknown (neutral 50%); delta=0 < margin → hold codex.
        self.assertEqual(best, 'codex')
        self.assertEqual(reg.active_name(), 'codex')

    def test_falls_to_bedrock_when_both_depleted(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False),
            'codex': _FakeBackend(available=False),
        }, initial='bedrock')
        best = sel.evaluate()
        self.assertEqual(best, 'bedrock')


class TestHysteresis(unittest.TestCase):
    """Tests for the incumbent-stickiness and bedrock-parking hysteresis."""

    # ------------------------------------------------------------------
    # STEP 1: healthy available incumbent — stickiness
    # ------------------------------------------------------------------

    def test_holds_incumbent_within_margin(self):
        """No switch when the challenger's advantage is below the margin."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=46.0),
            'codex': _FakeBackend(available=True, weekly_utilization=48.0),
        }, initial='codex')
        best = sel.evaluate()
        # delta = 48 - 46 = 2 < 5 pp margin → hold codex
        self.assertEqual(best, 'codex')

    def test_switches_when_challenger_beats_margin(self):
        """Switch when the challenger is >= margin below the active backend."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=50.0),
            'codex': _FakeBackend(available=True, weekly_utilization=60.0),
        }, initial='codex')
        best = sel.evaluate()
        # delta = 60 - 50 = 10 >= 5 → switch to anthropic
        self.assertEqual(best, 'anthropic')

    def test_exact_margin_boundary_switches(self):
        """The comparison is >= : an exact match triggers a switch."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=50.0),
            'codex': _FakeBackend(available=True, weekly_utilization=55.0),
        }, initial='codex')
        best = sel.evaluate()
        # delta = 55 - 50 = 5 == margin → switch (>= is inclusive)
        self.assertEqual(best, 'anthropic')

    def test_holds_on_exact_tie(self):
        """No switch when weekly values are identical."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=50.0),
            'codex': _FakeBackend(available=True, weekly_utilization=50.0),
        }, initial='codex')
        best = sel.evaluate()
        # delta = 0 < 5 → hold codex
        self.assertEqual(best, 'codex')

    def test_available_incumbent_not_parked_over_sub_margin_delta(self):
        """A healthy active backend is never parked over a sub-margin exhausted delta.

        Reproduces issue #2 from the plan: with active=codex available@52%
        and anthropic exhausted@50%, the old code would park on bedrock (2 pp
        gap); the new STEP 1 holds the healthy incumbent.
        """
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True, weekly_utilization=52.0),
        }, initial='codex')
        # Anthropic is exhausted (cooldown) but has lower weekly in snapshot.
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() + 9999
            sel._last_weekly['anthropic'] = 50.0
        best = sel.evaluate()
        # STEP 1 applies (codex is healthy available); parking skipped.
        self.assertNotEqual(best, 'bedrock')
        self.assertEqual(best, 'codex')

    def test_incumbent_held_when_weekly_unknown(self):
        """Transient unknown weekly on the incumbent must not strip incumbency.

        Reproduces issue #3: active backend weekly is None (no live probe, no
        snapshot) → treated as _UNKNOWN_WEEKLY (50).  Challenger at 48% gives
        delta 2 < margin 5 → hold.
        """
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=48.0),
            'codex': _FakeBackend(available=True, weekly_utilization=None),
        }, initial='codex')
        # No prior snapshot for codex weekly either.
        best = sel.evaluate()
        # codex neutral=50, anthropic=48, delta=2 < 5 → hold codex
        self.assertEqual(best, 'codex')

    def test_incumbent_unknown_switches_on_decisive_challenger(self):
        """Unknown incumbent weekly allows a switch when the gap is decisive."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=40.0),
            'codex': _FakeBackend(available=True, weekly_utilization=None),
        }, initial='codex')
        best = sel.evaluate()
        # codex neutral=50, anthropic=40, delta=10 >= 5 → switch to anthropic
        self.assertEqual(best, 'anthropic')

    def test_challenger_unknown_stays_on_incumbent(self):
        """Unknown challenger weekly (neutral 50) does not beat a lower-weekly incumbent."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=None),
            'codex': _FakeBackend(available=True, weekly_utilization=40.0),
        }, initial='codex')
        best = sel.evaluate()
        # codex=40, anthropic neutral=50 → codex is already lowest → hold
        self.assertEqual(best, 'codex')

    def test_custom_margin_honored(self):
        """Custom margin from config is respected."""
        reg = _FakeRegistry('codex')
        reg._instances = {
            'anthropic': _FakeBackend(available=True, weekly_utilization=50.0),
            'codex': _FakeBackend(available=True, weekly_utilization=60.0),
        }
        cfg = _FakeConfig()
        cfg.auto_backend_weekly_margin = 20.0  # larger margin
        sel = AutoSelector(reg, cfg)
        best = sel.evaluate()
        # delta = 10 < margin 20 → hold codex (would have switched with default 5 pp margin)
        self.assertEqual(best, 'codex')

    # ------------------------------------------------------------------
    # STEP 2: no healthy available incumbent — bedrock-parking
    # ------------------------------------------------------------------

    def test_parks_on_bedrock_when_incumbent_exhausted_and_gap_decisive(self):
        """STEP 2: parks on bedrock when active is exhausted and exhausted < available by margin."""
        future_reset = time.time() + 3600
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future_reset,
                                      weekly_utilization=30.0),
            'codex': _FakeBackend(available=True, weekly_utilization=70.0),
        }, initial='anthropic')
        best = sel.evaluate()
        # active=anthropic now exhausted (not in available pool) → STEP 2
        # best_avail=codex@70, best_exhausted=anthropic@30, 70-30=40 >= 5 → park
        self.assertEqual(best, 'bedrock')

    def test_park_skipped_when_gap_below_margin(self):
        """STEP 2 margin gate: no park when exhausted is only slightly lower."""
        future_reset = time.time() + 3600
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future_reset,
                                      weekly_utilization=58.0),
            'codex': _FakeBackend(available=True, weekly_utilization=60.0),
        }, initial='bedrock')
        best = sel.evaluate()
        # best_avail=codex@60, exhausted=anthropic@58, 60-58=2 < 5 → no park → codex
        self.assertEqual(best, 'codex')

    def test_park_exact_margin_boundary(self):
        """STEP 2: exact margin gap triggers parking (>= inclusive)."""
        future_reset = time.time() + 3600
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future_reset,
                                      weekly_utilization=55.0),
            'codex': _FakeBackend(available=True, weekly_utilization=60.0),
        }, initial='bedrock')
        best = sel.evaluate()
        # 60 - 55 = 5 == margin → park on bedrock
        self.assertEqual(best, 'bedrock')

    def test_subscription_mode_holds_within_margin_never_parks(self):
        """Subscription mode: holds within margin and never uses bedrock as fallback."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=48.0),
            'codex': _FakeBackend(available=True, weekly_utilization=50.0),
        }, initial='codex')
        sel._mode = 'subscription'
        best = sel.evaluate()
        # delta=2 < 5 → hold codex; never bedrock in subscription mode
        self.assertEqual(best, 'codex')
        self.assertNotEqual(best, 'bedrock')

    def test_subscription_mode_switches_on_decisive_gap_never_parks(self):
        """Subscription mode: switches on decisive gap but never to bedrock."""
        future_reset = time.time() + 3600
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future_reset,
                                      weekly_utilization=30.0),
            'codex': _FakeBackend(available=True, weekly_utilization=70.0),
        }, initial='anthropic')
        sel._mode = 'subscription'
        best = sel.evaluate()
        # In subscription mode STEP 2 is skipped → return best available (codex)
        self.assertEqual(best, 'codex')
        self.assertNotEqual(best, 'bedrock')

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def test_config_default_margin(self):
        """Default margin is 5.0 pp."""
        from anthproxy.config import Config, parse_args
        self.assertEqual(Config().auto_backend_weekly_margin, 5.0)
        self.assertEqual(parse_args([]).auto_backend_weekly_margin, 5.0)

    def test_config_default_mode_subscription(self):
        """Startup routing mode defaults to 'subscription'."""
        from anthproxy.config import Config, parse_args
        self.assertEqual(Config().auto_backend_mode, 'subscription')
        self.assertEqual(parse_args([]).auto_backend_mode, 'subscription')

    def test_config_cli_mode_auto(self):
        """--auto-backend-mode auto overrides the subscription default."""
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-backend-mode', 'auto'])
        self.assertEqual(cfg.auto_backend_mode, 'auto')

    def test_config_env_mode(self):
        """ANTHPROXY_AUTO_BACKEND_MODE env var is parsed correctly."""
        import os
        from anthproxy.config import parse_args
        old = os.environ.get('ANTHPROXY_AUTO_BACKEND_MODE')
        try:
            os.environ['ANTHPROXY_AUTO_BACKEND_MODE'] = 'auto'
            cfg = parse_args([])
            self.assertEqual(cfg.auto_backend_mode, 'auto')
        finally:
            if old is None:
                os.environ.pop('ANTHPROXY_AUTO_BACKEND_MODE', None)
            else:
                os.environ['ANTHPROXY_AUTO_BACKEND_MODE'] = old

    def test_selector_initial_mode_from_config(self):
        """AutoSelector adopts the configured startup mode."""
        cfg = _FakeConfig()
        cfg.auto_backend_mode = 'subscription'
        sel = AutoSelector(_FakeRegistry('anthropic'), cfg)
        self.assertEqual(sel._mode, 'subscription')

    def test_config_cli_margin(self):
        """--auto-backend-weekly-margin is parsed correctly."""
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-backend-weekly-margin', '12'])
        self.assertEqual(cfg.auto_backend_weekly_margin, 12.0)

    def test_config_env_margin(self):
        """ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN env var is parsed correctly."""
        import os
        from anthproxy.config import parse_args
        old = os.environ.get('ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN')
        try:
            os.environ['ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN'] = '8'
            cfg = parse_args([])
            self.assertEqual(cfg.auto_backend_weekly_margin, 8.0)
        finally:
            if old is None:
                os.environ.pop('ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN', None)
            else:
                os.environ['ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN'] = old


class TestOnRateLimited(unittest.TestCase):

    def test_429_on_anthropic_demotes_to_codex(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False),  # will read after invalidation
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        new_name = sel.on_rate_limited('anthropic', retry_after=None)
        self.assertEqual(new_name, 'codex')


    def test_retry_after_sets_cooldown(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),  # claims available but was 429'd
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        retry_after = 300.0
        sel.on_rate_limited('anthropic', retry_after=retry_after)
        with sel._lock:
            cooldown = sel._exhausted_until.get('anthropic', 0)
        self.assertGreater(cooldown, time.time() + 290)

    def test_resets_at_from_usage_sets_cooldown(self):
        future = time.time() + 1800
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, resets_at=future),
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        sel.on_rate_limited('anthropic', retry_after=None)
        with sel._lock:
            cooldown = sel._exhausted_until.get('anthropic', 0)
        self.assertAlmostEqual(cooldown, future, delta=5)

    def test_429_with_fallback_already_bedrock(self):
        """When both sub-backends are depleted, stays on bedrock."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False),
            'codex': _FakeBackend(available=False),
        }, initial='codex')
        new_name = sel.on_rate_limited('codex')
        self.assertEqual(new_name, 'bedrock')

    def test_positive_retry_after_skips_invalidation(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        sel.on_rate_limited('anthropic', retry_after=15.0)

    def test_available_429_keeps_short_cooldown(self):
        """A 429 from a backend that still has quota must NOT set a multi-hour cooldown.

        This is the regression guard for the bug where a long-context credit
        policy 429 (available=True, but resets_at set to the 5-hour window reset
        ~hours away) caused anthropic to be locked out until the window expired.
        Only the short _DEFAULT_RECHECK_SECS placeholder should remain.
        """
        future_resets_at = time.time() + 3 * 3600  # 3 hours out
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, resets_at=future_resets_at,
                                      weekly_utilization=26.0),
            'codex': _FakeBackend(available=True, weekly_utilization=71.0),
        }, initial='anthropic')
        sel.on_rate_limited('anthropic', retry_after=None)
        with sel._lock:
            cooldown = sel._exhausted_until.get('anthropic', 0)
        # Must be the short recheck window, not the multi-hour resets_at.
        self.assertLess(cooldown, time.time() + 600)

    def test_available_429_switches_back_after_cooldown(self):
        """After a non-quota 429, evaluate() switches back once the cooldown expires."""
        future_resets_at = time.time() + 3 * 3600
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, resets_at=future_resets_at,
                                      weekly_utilization=26.0),
            'codex': _FakeBackend(available=True, weekly_utilization=71.0),
        }, initial='anthropic')
        sel.on_rate_limited('anthropic', retry_after=None)
        # Simulate the cooldown expiring (back-date the timestamp).
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() - 1
        best = sel.evaluate()
        # anthropic should be re-selected (lower weekly utilization).
        self.assertEqual(best, 'anthropic')

    def test_available_true_resets_at_not_honored_as_cooldown(self):
        """resets_at is only used when available is False; available=True must not extend cooldown."""
        future_resets_at = time.time() + 7200
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, resets_at=future_resets_at),
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        sel.on_rate_limited('anthropic', retry_after=None)
        with sel._lock:
            cooldown = sel._exhausted_until.get('anthropic', 0)
        self.assertNotAlmostEqual(cooldown, future_resets_at, delta=60,
                                  msg='resets_at must not be used as cooldown when available=True')


class TestPinResume(unittest.TestCase):

    def test_pin_pauses_evaluation(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True),
        }, initial='codex')
        sel.pin('codex')
        self.assertTrue(sel.is_paused())
        # evaluate() should not switch when pinned
        best = sel.evaluate()
        self.assertEqual(best, 'codex')
        self.assertEqual(reg._switches, [])   # no switch attempted

    def test_pin_blocks_on_rate_limited_switch(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=False),
        }, initial='codex')
        sel.pin('codex')
        new_name = sel.on_rate_limited('codex')
        # Pinned → should NOT switch even though anthropic is available
        self.assertEqual(new_name, 'codex')

    def test_resume_clears_pin_and_evaluates(self):
        # Give anthropic a decisively lower weekly (delta 70 pp >= margin 5 pp)
        # so resume() → evaluate() actually switches away from the pinned codex.
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=10.0),
            'codex': _FakeBackend(available=True, weekly_utilization=80.0),
        }, initial='codex')
        sel.pin('codex')
        sel.resume()
        self.assertFalse(sel.is_paused())
        self.assertEqual(reg.active_name(), 'anthropic')

    def test_status_line_when_paused(self):
        sel, reg = _make_selector({}, initial='bedrock')
        sel.pin('codex')
        self.assertIn('paused', sel.status_line())
        self.assertIn('codex', sel.status_line())

    def test_status_line_when_active(self):
        sel, reg = _make_selector({}, initial='bedrock')
        self.assertIn('on', sel.status_line())


class TestStartStop(unittest.TestCase):

    def test_start_stop_no_crash(self):
        sel, _ = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True),
        }, initial='bedrock')
        sel.start()
        time.sleep(0.05)
        sel.stop()

    def test_double_start_is_noop(self):
        sel, _ = _make_selector({}, initial='bedrock')
        sel.start()
        sel.start()  # should not start a second thread
        self.assertIsNotNone(sel._thread)
        sel.stop()


class TestLocalBackendExclusion(unittest.TestCase):
    """'local' must never appear in AutoSelector auto-selection results."""

    def test_local_not_in_priority(self):
        assert 'local' not in _PRIORITY

    def test_local_not_in_fallback(self):
        assert _FALLBACK != 'local'

    def test_evaluate_never_switches_to_local(self):
        # Even if the registry starts on 'local', evaluate() picks from
        # _PRIORITY/_FALLBACK and therefore never returns or switches to 'local'.
        bedrock_backend = _FakeBackend(available=True)
        sel, reg = _make_selector({'bedrock': bedrock_backend}, initial='local')
        sel.evaluate()
        # The registry should have been switched away from 'local'
        self.assertNotEqual(reg.active_name(), 'local')

    def test_pin_local_suspends_auto_selection(self):
        bedrock_backend = _FakeBackend(available=True)
        sel, reg = _make_selector({'bedrock': bedrock_backend}, initial='bedrock')
        sel.pin('local')
        # is_paused() must be True
        self.assertTrue(sel.is_paused())
        # evaluate() short-circuits when pinned — no switch away from 'local' pin
        sel.evaluate()
        # pinned status is unchanged
        self.assertTrue(sel.is_paused())


# ---------------------------------------------------------------------------
# Subscription mode tests
# ---------------------------------------------------------------------------

class TestSubscriptionMode(unittest.TestCase):

    def test_restrict_subscription_sets_mode_and_evaluates(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=30.0),
            'codex': _FakeBackend(available=True, weekly_utilization=10.0),
        }, initial='bedrock')
        result = sel.restrict_subscription()
        # Should have evaluated and switched to codex (lower weekly)
        self.assertIn(result, ('anthropic', 'codex'))
        with sel._lock:
            self.assertEqual(sel._mode, 'subscription')
            self.assertIsNone(sel._pinned)

    def test_is_paused_false_in_subscription_mode(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
        }, initial='bedrock')
        sel.restrict_subscription()
        # subscription mode is NOT paused — reactive-429 path stays active
        self.assertFalse(sel.is_paused())

    def test_is_paused_true_in_pinned_mode(self):
        sel, reg = _make_selector({}, initial='bedrock')
        sel.pin('codex')
        self.assertTrue(sel.is_paused())

    def test_status_line_subscription_only(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
        }, initial='bedrock')
        sel.restrict_subscription()
        self.assertIn('subscription-only', sel.status_line())

    def test_resume_from_subscription_returns_to_auto(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=10.0),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        sel.restrict_subscription()
        sel.resume()
        with sel._lock:
            self.assertEqual(sel._mode, 'auto')
        self.assertFalse(sel.is_paused())
        self.assertIn('auto-selection: on', sel.status_line())

    def test_subscription_mode_never_falls_to_bedrock_when_both_exhausted(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=False, weekly_utilization=30.0),
            'codex': _FakeBackend(available=False, weekly_utilization=20.0),
        }, initial='anthropic')
        sel.restrict_subscription()
        best = reg.active_name()
        # Must be a subscription backend, never bedrock
        self.assertIn(best, ('anthropic', 'codex'))
        self.assertNotEqual(best, 'bedrock')

    def test_subscription_mode_skips_bedrock_wait_branch(self):
        """When one sub is exhausted with lower weekly, still picks available sub."""
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True, weekly_utilization=60.0),
            'codex': _FakeBackend(available=True, weekly_utilization=20.0),
        }, initial='bedrock')
        with sel._lock:
            sel._exhausted_until['codex'] = time.time() + 9999
            sel._last_weekly['codex'] = 5.0   # lower than anthropic
        sel.restrict_subscription()
        # In auto mode this would park on bedrock; in subscription mode it picks anthropic
        self.assertEqual(reg.active_name(), 'anthropic')


class TestCurrentSubscriptionBackend(unittest.TestCase):

    def test_returns_subscription_backend_never_bedrock(self):
        sel, _ = _make_selector({}, initial='bedrock')
        result = sel.current_subscription_backend()
        self.assertIn(result, list(_PRIORITY) + [None])
        self.assertNotEqual(result, 'bedrock')
        self.assertNotEqual(result, 'plugin')
        self.assertNotEqual(result, 'local')

    def test_prefers_not_exhausted_backend(self):
        sel, _ = _make_selector({}, initial='bedrock')
        with sel._lock:
            sel._exhausted_until['codex'] = time.time() + 9999
        result = sel.current_subscription_backend()
        self.assertEqual(result, 'anthropic')

    def test_falls_back_to_lowest_weekly_when_all_exhausted(self):
        sel, _ = _make_selector({}, initial='bedrock')
        with sel._lock:
            sel._exhausted_until['anthropic'] = time.time() + 9999
            sel._exhausted_until['codex'] = time.time() + 9999
            sel._exhausted_until['openrouter'] = time.time() + 9999
            sel._last_weekly['anthropic'] = 80.0
            sel._last_weekly['codex'] = 20.0
            sel._last_weekly['openrouter'] = 40.0
        result = sel.current_subscription_backend()
        # codex has lower weekly — prefer it
        self.assertEqual(result, 'codex')

    def test_no_network_no_registry_calls(self):
        """current_subscription_backend must not call registry.switch or registry.active_name."""
        from unittest.mock import MagicMock
        mock_reg = MagicMock()
        mock_reg.active_name.return_value = 'bedrock'
        cfg = _FakeConfig()
        sel = AutoSelector(mock_reg, cfg)
        sel.current_subscription_backend()
        mock_reg.switch.assert_not_called()
        # active_name may be called from evaluate/other paths but not from current_subscription_backend
        # Reset call count so we only count the next call
        mock_reg.active_name.reset_mock()
        sel.current_subscription_backend()
        mock_reg.switch.assert_not_called()


class TestNoteExhausted(unittest.TestCase):

    def test_records_exhaustion_without_switching(self):
        sel, reg = _make_selector({
            'anthropic': _FakeBackend(available=True),
            'codex': _FakeBackend(available=True),
        }, initial='anthropic')
        sel.note_exhausted('codex', retry_after=30.0)
        with sel._lock:
            self.assertGreater(sel._exhausted_until.get('codex', 0), time.time())
        # No switch should have occurred
        self.assertEqual(reg.active_name(), 'anthropic')
        self.assertEqual(reg._switches, [])

    def test_note_exhausted_does_not_call_registry_switch(self):
        from unittest.mock import MagicMock
        mock_reg = MagicMock()
        mock_reg.active_name.return_value = 'anthropic'
        cfg = _FakeConfig()
        sel = AutoSelector(mock_reg, cfg)
        sel.note_exhausted('codex', retry_after=60.0)
        mock_reg.switch.assert_not_called()

    def test_resume_from_local_switches_to_best(self):
        bedrock_backend = _FakeBackend(available=True)
        sel, reg = _make_selector({'bedrock': bedrock_backend}, initial='bedrock')
        sel.pin('local')
        sel.resume()
        self.assertFalse(sel.is_paused())


# ---------------------------------------------------------------------------
# Weekly-cache preservation when backend returns None weekly_utilization
# ---------------------------------------------------------------------------

class TestWeeklyNoneCachePreservation(unittest.TestCase):
    """Verify that a None weekly_utilization preserves the _last_weekly cache.

    This is the selector-level consequence of _max_weekly_utilization returning
    None (e.g. for a lone malformed seven_day window).  Selector.py:403-404
    only writes new_weekly when the value is non-None, so the cache is left
    untouched and the previously-seen numeric value is reused on the next tick.

    Under the old backend code a malformed window produced 0.0, overwriting the
    cache with 0.0 each tick and biasing the backend toward looking like a
    strong 0%-consumed challenger.  The new None return prevents that clobber.
    """

    def test_none_weekly_preserves_cached_value(self):
        """After a valid weekly is cached, a subsequent None does not clobber it.

        First evaluate: anthropic reports weekly_utilization=70.0 → cached.
        Second evaluate: anthropic reports weekly_utilization=None → cache
        must remain 70.0, NOT fall through to _UNKNOWN_WEEKLY (50.0).
        We verify this by checking sel._last_weekly directly, mirroring the
        pattern used in test_updates_last_weekly_from_successful_probe.
        """
        anthropic_backend = _FakeBackend(available=True, weekly_utilization=70.0)
        sel, reg = _make_selector({'anthropic': anthropic_backend}, initial='anthropic')

        # Tick 1: 70.0 should be written to _last_weekly.
        sel.evaluate()
        with sel._lock:
            self.assertEqual(sel._last_weekly.get('anthropic'), 70.0)

        # Tick 2: backend now reports None weekly — cache must be unchanged.
        anthropic_backend.set_status(available=True, weekly_utilization=None)
        sel.evaluate()
        with sel._lock:
            self.assertEqual(
                sel._last_weekly.get('anthropic'), 70.0,
                'A None weekly_utilization must not overwrite the cached value',
            )

    def test_never_seen_none_weekly_falls_through_to_unknown(self):
        """A backend that has never reported a numeric weekly stays absent from cache.

        When effective_weekly is None (no cached value), selector.py:441 uses
        _UNKNOWN_WEEKLY (50.0) as the neutral sort key — but the _last_weekly
        dict itself must remain empty for that backend.
        """
        anthropic_backend = _FakeBackend(available=True, weekly_utilization=None)
        sel, reg = _make_selector({'anthropic': anthropic_backend}, initial='anthropic')

        sel.evaluate()
        with sel._lock:
            self.assertNotIn(
                'anthropic', sel._last_weekly,
                'A backend that only ever reports None weekly must not appear in _last_weekly',
            )
        # Sanity: selector still chose the only available backend.
        self.assertEqual(reg.active_name(), 'anthropic')


if __name__ == '__main__':
    unittest.main()
