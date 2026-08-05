import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from anthproxy.config import Config
from anthproxy.oauth_registry import OAuthTokenRegistry, OAuthTokenSnapshot
from anthproxy.selector import AutoSelector
from anthproxy.server import BackendRegistry, _oauth_decision_reason


class _Backend:
    def __init__(self, name):
        self.name = name


class _Registry:
    def active_name(self):
        raise AssertionError('personal_candidates must not call registry')


def test_personal_candidates_exclude_cooldown_and_use_cached_burn_only():
    selector = AutoSelector(_Registry(), SimpleNamespace(
        auto_backend_interval=3600,
        auto_backend_mode='auto',
        auto_backend=False,
    ))
    now = time.time()
    with selector._lock:
        selector._last_weekly.update({'anthropic': 20.0, 'codex': 30.0})
        selector._last_status_at.update({'anthropic': now, 'codex': now})
        selector._last_available.update({'anthropic': True, 'codex': True})
        selector._exhausted_until['anthropic'] = now + 100

    candidates = selector.personal_candidates()

    assert [candidate.name for candidate in candidates] == ['codex', 'openrouter']
    assert candidates[0].burn == 30.0
    assert candidates[1].burn == 50.0


def test_personal_candidates_treat_stale_usage_as_neutral_burn():
    selector = AutoSelector(_Registry(), SimpleNamespace(
        auto_backend_interval=3600,
        auto_backend_mode='auto',
        auto_backend=False,
    ))
    with selector._lock:
        selector._last_weekly['anthropic'] = 5.0
        selector._last_status_at['anthropic'] = time.time() - 301
        selector._last_available['anthropic'] = True

    candidate = next(c for c in selector.personal_candidates() if c.name == 'anthropic')

    assert candidate.burn == 50.0


def _registry(oauth_registry):
    config = Config(backend='anthropic')
    registry = BackendRegistry(config, _Backend('anthropic'), oauth_registry=oauth_registry)
    registry._instances['oauth'] = _Backend('oauth')
    registry.set_personal_candidates_resolver(
        lambda: (SimpleNamespace(name='anthropic', burn=40.0),)
    )
    return registry


def _eligible_oauth(utilization=10.0):
    registry = OAuthTokenRegistry()
    credential = registry.observe('enterprise')
    registry.record_probe_success(credential.generation, {
        'extra_usage': {
            'monthly_limit': 100,
            'used_credits': 10,
            'utilization': utilization,
            'spend_limit_reached': False,
            'is_enabled': True,
        },
    }, health_ok=True)
    return registry, credential


def test_snapshot_for_request_chooses_lower_burn_oauth():
    oauth_registry, credential = _eligible_oauth(1.0)
    registry = _registry(oauth_registry)

    snapshot = registry.snapshot_for_request(oauth_credential=credential)

    assert snapshot.name == 'oauth'
    assert snapshot.credentials == credential


def test_snapshot_for_request_tie_favors_personal():
    oauth_registry, credential = _eligible_oauth(20.0)
    registry = _registry(oauth_registry)
    oauth_burn = oauth_registry.snapshot().burn
    registry.set_personal_candidates_resolver(
        lambda: (SimpleNamespace(name='anthropic', burn=oauth_burn),)
    )

    snapshot = registry.snapshot_for_request(oauth_credential=credential)

    assert snapshot.name == 'anthropic'
    assert snapshot.credentials is None


def test_session_pin_bypasses_oauth_economics(monkeypatch):
    oauth_registry, credential = _eligible_oauth(0.001)
    registry = _registry(oauth_registry)
    registry._instances['codex'] = _Backend('codex')
    registry._session_overrides['session'] = 'codex'

    snapshot = registry.snapshot_for_request('session', oauth_credential=credential)

    assert snapshot.name == 'codex'
    assert snapshot.session_pinned is True


def test_invalid_oauth_preference_falls_back_to_personal():
    oauth_registry = OAuthTokenRegistry()
    credential = oauth_registry.observe('enterprise')
    registry = _registry(oauth_registry)

    snapshot = registry.snapshot_for_request(
        prefer_backend='oauth', oauth_credential=credential,
    )

    assert snapshot.name == 'anthropic'


def test_snapshot_logs_enterprise_choice_at_info(caplog):
    oauth_registry, credential = _eligible_oauth(1.0)
    registry = _registry(oauth_registry)

    with caplog.at_level(logging.DEBUG, logger='anthproxy.server'):
        registry.snapshot_for_request(oauth_credential=credential)

    record = next(
        r for r in caplog.records if 'OAuth backend selection' in r.message
    )
    assert record.levelno == logging.INFO
    assert 'chosen=oauth' in record.getMessage()
    assert 'enterprise_token=yes' in record.getMessage()
    assert 'below personal' in record.getMessage()


def test_snapshot_logs_personal_choice_at_debug_with_tie_reason(caplog):
    oauth_registry, credential = _eligible_oauth(20.0)
    registry = _registry(oauth_registry)
    oauth_burn = oauth_registry.snapshot().burn
    registry.set_personal_candidates_resolver(
        lambda: (SimpleNamespace(name='anthropic', burn=oauth_burn),)
    )

    with caplog.at_level(logging.DEBUG, logger='anthproxy.server'):
        registry.snapshot_for_request(oauth_credential=credential)

    record = next(
        r for r in caplog.records if 'OAuth backend selection' in r.message
    )
    assert record.levelno == logging.DEBUG
    assert 'chosen=anthropic' in record.getMessage()
    assert 'at or below enterprise' in record.getMessage()


def test_snapshot_logs_no_enterprise_token_reason(caplog):
    registry = _registry(OAuthTokenRegistry())

    with caplog.at_level(logging.DEBUG, logger='anthproxy.server'):
        registry.snapshot_for_request()

    record = next(
        r for r in caplog.records if 'OAuth backend selection' in r.message
    )
    assert 'enterprise_token=no' in record.getMessage()
    assert 'reason=no enterprise token on request' in record.getMessage()


def test_snapshot_logs_cooldown_reason(caplog):
    oauth_registry, credential = _eligible_oauth(1.0)
    oauth_registry.mark_cooldown(credential.generation, retry_after=120.0)
    registry = _registry(oauth_registry)

    with caplog.at_level(logging.DEBUG, logger='anthproxy.server'):
        registry.snapshot_for_request(oauth_credential=credential)

    record = next(
        r for r in caplog.records if 'OAuth backend selection' in r.message
    )
    assert record.levelno == logging.DEBUG
    assert 'chosen=anthropic' in record.getMessage()
    assert 'enterprise in cooldown' in record.getMessage()


def test_mark_oauth_cooldown_logs_info(caplog):
    oauth_registry, credential = _eligible_oauth(1.0)
    registry = _registry(oauth_registry)

    with caplog.at_level(logging.INFO, logger='anthproxy.server'):
        registry.mark_oauth_cooldown(credential, retry_after=90.0)

    assert any(
        'enterprise token 429: cooldown 90s' in r.getMessage()
        for r in caplog.records
    )


def test_mark_oauth_cap_exhausted_logs_info(caplog):
    oauth_registry, credential = _eligible_oauth(1.0)
    registry = _registry(oauth_registry)

    with caplog.at_level(logging.INFO, logger='anthproxy.server'):
        registry.mark_oauth_cap_exhausted(credential)

    assert any(
        'spend-cap exhausted' in r.getMessage() for r in caplog.records
    )


def test_oauth_token_status_none_without_registry():
    config = Config(backend='anthropic')
    registry = BackendRegistry(config, _Backend('anthropic'))

    assert registry.oauth_token_status() is None


def test_oauth_token_status_absent_before_observation():
    registry = _registry(OAuthTokenRegistry())

    status = registry.oauth_token_status()

    assert status['present'] is False
    assert status['eligible'] is False
    assert status['burn_pct'] is None


def test_oauth_token_status_reports_eligible_fields():
    oauth_registry, _ = _eligible_oauth(12.5)
    registry = _registry(oauth_registry)

    status = registry.oauth_token_status()

    assert status['present'] is True
    assert status['eligible'] is True
    assert status['burn_pct'] == 12.5
    assert status['monthly_blocked'] is False
    assert status['usage_stale'] is False
    assert status['cooldown_remaining_seconds'] == 0.0


def _ineligible(**overrides):
    """Return an OAuthTokenSnapshot with eligible=False and specified fields."""
    defaults = dict(
        eligible=False,
        cooldown_remaining_seconds=0.0,
        monthly_blocked=False,
        health_ok=True,
        burn=50.0,
        usage_age_seconds=0.0,
        usage_stale=False,
    )
    defaults.update(overrides)
    return OAuthTokenSnapshot(**defaults)


def test_reason_no_credential():
    assert _oauth_decision_reason(None, None, False, 40.0) == 'no enterprise token on request'


def test_reason_oauth_not_tracked():
    cred = object()
    assert _oauth_decision_reason(cred, None, False, 40.0) == 'enterprise token not tracked'


def test_reason_oauth_wins():
    snap = OAuthTokenSnapshot(eligible=True, burn=10.0)
    cred = object()
    reason = _oauth_decision_reason(cred, snap, True, 40.0)
    assert 'enterprise weekly 10.0% below personal 40.0%' == reason


def test_reason_cooldown():
    snap = _ineligible(cooldown_remaining_seconds=90.0)
    assert 'enterprise in cooldown (90s remaining)' == _oauth_decision_reason(
        object(), snap, False, 40.0
    )


def test_reason_monthly_blocked():
    snap = _ineligible(monthly_blocked=True)
    assert _oauth_decision_reason(object(), snap, False, 40.0) == (
        'enterprise spend-cap parked for the month'
    )


def test_reason_health_not_confirmed():
    snap = _ineligible(health_ok=None)
    assert _oauth_decision_reason(object(), snap, False, 40.0) == (
        'enterprise health check not confirmed'
    )


def test_reason_burn_none_ineligible():
    snap = _ineligible(burn=None)
    assert _oauth_decision_reason(object(), snap, False, 40.0) == (
        'enterprise usage reading unavailable'
    )


def test_reason_usage_age_none():
    snap = _ineligible(usage_age_seconds=None)
    assert _oauth_decision_reason(object(), snap, False, 40.0) == (
        'enterprise usage not yet probed'
    )


def test_reason_usage_stale():
    snap = _ineligible(usage_age_seconds=400.0, usage_stale=True)
    reason = _oauth_decision_reason(object(), snap, False, 40.0)
    assert reason.startswith('enterprise usage cache expired')
    assert '400s old' in reason


def test_reason_prior_month():
    snap = _ineligible()
    assert _oauth_decision_reason(object(), snap, False, 40.0) == (
        'enterprise usage from a prior month'
    )


def test_reason_eligible_lost():
    snap = OAuthTokenSnapshot(eligible=True, burn=40.0)
    reason = _oauth_decision_reason(object(), snap, False, 40.0)
    assert 'personal weekly 40.0% at or below enterprise 40.0%' == reason


def test_selector_wake_runs_registry_tick_promptly():
    oauth_registry = MagicMock()
    selector = AutoSelector(_Registry(), SimpleNamespace(
        auto_backend_interval=3600,
        auto_backend_mode='auto',
        auto_backend=False,
    ), oauth_registry=oauth_registry)
    selector.start()
    try:
        selector.wake()
        deadline = time.time() + 1
        while oauth_registry.tick.call_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        oauth_registry.tick.assert_called()
    finally:
        selector.stop()


def test_request_snapshot_path_does_not_call_health_or_usage_network():
    oauth_registry, credential = _eligible_oauth(1.0)
    registry = _registry(oauth_registry)
    backend = registry._instances['anthropic']
    backend.five_hour_status = MagicMock(side_effect=AssertionError('network call'))

    registry.snapshot_for_request(oauth_credential=credential)

    backend.five_hour_status.assert_not_called()
