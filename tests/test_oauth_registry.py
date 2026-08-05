import datetime as dt
import threading

import pytest

from anthproxy.oauth_registry import OAuthTokenRegistry


class _Clock:
    def __init__(self, wall: dt.datetime):
        self.wall = wall
        self.monotonic = 100.0

    def now(self):
        return self.wall

    def mono(self):
        return self.monotonic


def _usage(utilization=25.0, **overrides):
    extra = {
        'monthly_limit': 10000,
        'used_credits': 2500,
        'utilization': utilization,
        'spend_limit_reached': False,
        'is_enabled': True,
    }
    extra.update(overrides)
    return {'extra_usage': extra}


def test_new_token_is_redacted_and_ineligible_until_probed():
    wakeups = []
    registry = OAuthTokenRegistry(wake=lambda: wakeups.append(True))

    credential = registry.observe('secret-token')
    snapshot = registry.snapshot()

    assert credential.access_token == 'secret-token'
    assert snapshot.generation == credential.generation
    assert snapshot.fingerprint
    assert 'secret-token' not in repr(snapshot)
    assert snapshot.eligible is False
    assert wakeups == [True]


def test_tick_records_usage_health_and_calendar_month_burn():
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(
        usage_probe=lambda token: _usage(25.0),
        health_probe=lambda token: None,
        monotonic=clock.mono,
        utcnow=clock.now,
    )
    credential = registry.observe('secret-token')

    registry.tick()
    snapshot = registry.snapshot()

    elapsed = 15 / 31
    assert snapshot.eligible is True
    assert snapshot.burn == pytest.approx(25.0 / elapsed)
    assert registry.credentials(credential.generation) == credential


def test_tokens_keep_isolated_probe_state():
    registry = OAuthTokenRegistry()
    first = registry.observe('first')
    second = registry.observe('second')

    assert registry.record_probe_success(first.generation, _usage(), health_ok=True) is True
    assert registry.snapshot(first.generation).eligible is True
    assert registry.snapshot(second.generation).eligible is False


def test_token_observed_during_probe_is_probed_independently():
    started = threading.Event()
    release = threading.Event()

    def probe(token):
        if token == 'first':
            started.set()
            release.wait(timeout=1)
        return _usage()

    registry = OAuthTokenRegistry(usage_probe=probe)
    first = registry.observe('first')
    thread = threading.Thread(target=registry.tick)
    thread.start()
    assert started.wait(timeout=1)
    second = registry.observe('second')
    registry.tick()
    release.set()
    thread.join(timeout=1)

    assert registry.snapshot(first.generation).eligible is True
    assert registry.snapshot(second.generation).eligible is True


def test_usage_expiry_and_cooldown_exclude_token():
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
    credential = registry.observe('secret')
    registry.record_probe_success(credential.generation, _usage(), health_ok=True)
    assert registry.snapshot().eligible is True

    clock.monotonic += 301
    assert registry.snapshot().eligible is False

    registry.record_probe_success(credential.generation, _usage(), health_ok=True)
    registry.mark_cooldown(credential.generation, 60)
    assert registry.snapshot().eligible is False
    clock.monotonic += 61
    assert registry.snapshot().eligible is True


def test_missing_or_exhausted_usage_is_ineligible():
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    for usage in (
        _usage(monthly_limit=None),
        _usage(used_credits=None),
        _usage(utilization=None),
        _usage(spend_limit_reached=True),
        _usage(is_enabled=False),
        _usage(utilization=-1),
    ):
        registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
        credential = registry.observe('secret')
        registry.record_probe_success(credential.generation, usage, health_ok=True)
        assert registry.snapshot().eligible is False


def test_monthly_cap_stays_blocked_until_next_utc_month():
    clock = _Clock(dt.datetime(2026, 8, 31, 23, 59, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
    credential = registry.observe('secret')
    registry.record_probe_success(
        credential.generation,
        _usage(utilization=100.0, spend_limit_reached=True),
        health_ok=True,
    )
    assert registry.snapshot().eligible is False

    clock.wall = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    assert registry.snapshot().eligible is False
    assert registry.snapshot().monthly_blocked is False


def test_refresh_does_not_repeat_successful_health_probe():
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    health_calls = []
    registry = OAuthTokenRegistry(
        usage_probe=lambda token: _usage(),
        health_probe=lambda token: health_calls.append(token),
        monotonic=clock.mono,
        utcnow=clock.now,
    )
    registry.observe('secret')

    registry.tick()
    clock.monotonic += 301
    registry.tick()

    assert health_calls == ['secret']


def test_probe_failure_is_generation_safe_and_requests_health_retry():
    registry = OAuthTokenRegistry()
    credential = registry.observe('secret')

    assert registry.record_probe_failure(credential.generation, 'health') is True
    snapshot = registry.snapshot()
    assert snapshot.health_ok is False
    assert snapshot.eligible is False
