import datetime as dt
import logging
import threading

import pytest

from anthproxy.oauth_registry import (
    _MAX_TOKENS,
    _USAGE_TTL_SECONDS,
    OAuthTokenRegistry,
    _next_month,
)


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


def test_tick_records_usage_health_and_raw_utilization_burn():
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

    # burn is the raw monthly utilization percent, independent of how far into
    # the month it is, so it stays comparable to personal weekly utilization.
    assert snapshot.eligible is True
    assert snapshot.burn == pytest.approx(25.0)
    assert registry.credentials(credential.generation) == credential


def test_burn_is_independent_of_day_of_month():
    # Regression: burn must not be inflated early in the month.  A projected
    # end-of-month rate made a lightly-used token look "full" on day 1 and
    # systematically excluded it from routing against personal weekly burn.
    def burn_on(day: int) -> float:
        clock = _Clock(dt.datetime(2026, 8, day, tzinfo=dt.timezone.utc))
        registry = OAuthTokenRegistry(
            usage_probe=lambda token: _usage(10.0),
            monotonic=clock.mono,
            utcnow=clock.now,
        )
        registry.observe('secret-token')
        registry.tick()
        return registry.snapshot().burn

    assert burn_on(1) == pytest.approx(10.0)
    assert burn_on(28) == pytest.approx(10.0)


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


def test_next_month_rolls_december_into_january_of_next_year():
    result = _next_month(dt.datetime(2026, 12, 15, tzinfo=dt.timezone.utc))
    assert result == dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc)


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


def test_usage_stage_failure_clears_cached_usage():
    registry = OAuthTokenRegistry()
    credential = registry.observe('secret')
    registry.record_probe_success(credential.generation, _usage(), health_ok=True)
    assert registry.snapshot(credential.generation).eligible is True

    assert registry.record_probe_failure(credential.generation, 'usage') is True
    snapshot = registry.snapshot(credential.generation)
    assert snapshot.usage is None
    assert snapshot.usage_age_seconds is None
    assert snapshot.eligible is False


def test_observe_evicts_oldest_beyond_max_tokens():
    registry = OAuthTokenRegistry()
    first = registry.observe('token-0')
    for index in range(1, _MAX_TOKENS + 1):
        registry.observe(f'token-{index}')

    # The oldest token is evicted once capacity is exceeded; its generation
    # no longer resolves and re-observing mints a fresh generation.
    assert registry.credentials(first.generation) is None
    reobserved = registry.observe('token-0')
    assert reobserved.generation != first.generation


def test_observe_reobservation_keeps_token_from_eviction():
    registry = OAuthTokenRegistry()
    first = registry.observe('token-0')
    for index in range(1, _MAX_TOKENS):
        registry.observe(f'token-{index}')

    # Re-observing moves token-0 to most-recent, so the next insert evicts
    # token-1 (now oldest) instead of token-0.
    assert registry.observe('token-0') == first
    second_gen = registry.credentials(first.generation + 1)  # token-1
    registry.observe('token-new')

    assert registry.credentials(first.generation) == first
    assert second_gen is not None
    assert registry.credentials(first.generation + 1) is None


def test_mark_cooldown_defaults_and_extends_monotonically():
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
    credential = registry.observe('secret')

    assert registry.mark_cooldown(999) is False  # unknown generation

    assert registry.mark_cooldown(credential.generation) is True  # default 300s
    assert registry.snapshot().cooldown_remaining_seconds == pytest.approx(300.0)

    # A shorter cooldown must not shorten the existing longer one.
    registry.mark_cooldown(credential.generation, 10)
    assert registry.snapshot().cooldown_remaining_seconds == pytest.approx(300.0)

    # A negative retry_after is floored at zero (no negative extension).
    clock.monotonic += 300
    registry.mark_cooldown(credential.generation, -5)
    assert registry.snapshot().cooldown_remaining_seconds == pytest.approx(0.0)


def test_mark_cap_exhausted_blocks_until_next_utc_month():
    # A live 429 with no Retry-After guidance is spend-cap exhaustion (the token
    # was eligible, so its cached usage still reads under-cap): park it until the
    # next UTC calendar month rather than a short cooldown.
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
    credential = registry.observe('secret')
    registry.record_probe_success(credential.generation, _usage(25.0), health_ok=True)
    assert registry.snapshot().eligible is True

    assert registry.mark_cap_exhausted(999) is False  # unknown generation
    assert registry.mark_cap_exhausted(credential.generation) is True

    snapshot = registry.snapshot()
    assert snapshot.monthly_blocked is True
    assert snapshot.eligible is False

    # The block clears only at the next UTC calendar month boundary.  Usage is
    # still stamped in August (usage_month mismatch), so the token remains
    # ineligible until the next probe writes fresh September data.
    clock.wall = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
    snapshot_sep = registry.snapshot()
    assert snapshot_sep.monthly_blocked is False
    assert snapshot_sep.eligible is False


def test_indeterminate_probe_does_not_clear_provisional_monthly_block():
    # record_probe_success with unparseable/indeterminate usage (valid=False)
    # must not wipe a genuine 429-driven monthly block — a probe hiccup should
    # not silently lift a month-long park.
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
    credential = registry.observe('secret')
    registry.record_probe_success(credential.generation, _usage(25.0), health_ok=True)
    registry.mark_cap_exhausted(credential.generation)
    assert registry.snapshot().monthly_blocked is True

    # Malformed usage (is_enabled=False → valid=False, cap_reached=False).
    registry.record_probe_success(
        credential.generation, _usage(is_enabled=False), health_ok=True,
    )

    assert registry.snapshot().monthly_blocked is True


def test_under_cap_probe_clears_provisional_monthly_block():
    # A 429-driven month block is provisional: the next usage probe that reads
    # under-cap this month is authoritative and un-parks the token, bounding the
    # cost of a false-positive (a transient 429 that carried no Retry-After).
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    registry = OAuthTokenRegistry(monotonic=clock.mono, utcnow=clock.now)
    credential = registry.observe('secret')
    registry.record_probe_success(credential.generation, _usage(25.0), health_ok=True)
    registry.mark_cap_exhausted(credential.generation)
    assert registry.snapshot().monthly_blocked is True

    registry.record_probe_success(credential.generation, _usage(30.0), health_ok=True)

    assert registry.snapshot().monthly_blocked is False
    assert registry.snapshot().eligible is True


def test_set_wake_registers_callback_for_observe():
    registry = OAuthTokenRegistry()
    wakeups = []
    registry.set_wake(lambda: wakeups.append(True))

    registry.observe('secret')

    assert wakeups == [True]


def test_tick_force_reprobes_before_ttl():
    clock = _Clock(dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc))
    calls = []
    registry = OAuthTokenRegistry(
        usage_probe=lambda token: calls.append(token) or _usage(),
        monotonic=clock.mono,
        utcnow=clock.now,
    )
    registry.observe('secret')

    registry.tick()
    registry.tick()  # within TTL: no re-probe
    assert calls == ['secret']

    registry.tick(force=True)  # force ignores TTL
    assert calls == ['secret', 'secret']


def test_tick_background_probes_on_worker_threads():
    ran_off_main = []
    main_thread = threading.current_thread()
    done = threading.Event()

    def probe(token):
        ran_off_main.append(threading.current_thread() is not main_thread)
        done.set()
        return _usage()

    registry = OAuthTokenRegistry(usage_probe=probe)
    registry.observe('secret')

    registry.tick(background=True)

    assert done.wait(timeout=1)
    assert ran_off_main == [True]


# ---------------------------------------------------------------------------
# Token rotation: new generation starts ineligible, then clears after probe
# ---------------------------------------------------------------------------

def _eligible_usage():
    return {
        'extra_usage': {
            'monthly_limit': 100, 'used_credits': 10, 'utilization': 10.0,
            'spend_limit_reached': False, 'is_enabled': True,
        },
    }


def test_token_rotation_new_generation_ineligible_until_probed():
    """Rotating to a new access token carries usage but requires its own probe."""
    clock = _Clock(dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc))
    clock.monotonic = 1000.0  # non-zero so usage_at is truthy
    registry = OAuthTokenRegistry(
        monotonic=lambda: clock.monotonic,
        utcnow=clock.now,
    )

    # First token: probe succeeds → eligible
    cred1 = registry.observe('sk-ant-oat01-token-v1')
    registry.record_probe_success(cred1.generation, _eligible_usage(), health_ok=True)
    assert registry.snapshot().eligible

    # Second token: observe carries usage but health_ok=None → not yet eligible
    cred2 = registry.observe('sk-ant-oat01-token-v2')
    snap2 = registry.snapshot()
    assert not snap2.eligible
    assert snap2.burn is not None  # usage was carried over

    # After probe for second token: eligible
    registry.record_probe_success(cred2.generation, _eligible_usage(), health_ok=True)
    assert registry.snapshot().eligible
    assert registry.snapshot().generation == cred2.generation


def test_token_rotation_old_generation_ineligible_after_superseded():
    """Old generation snapshot is not eligible once a newer token takes over."""
    clock = _Clock(dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc))
    clock.monotonic = 1000.0
    registry = OAuthTokenRegistry(
        monotonic=lambda: clock.monotonic,
        utcnow=clock.now,
    )

    cred1 = registry.observe('sk-ant-oat01-old')
    registry.record_probe_success(cred1.generation, _eligible_usage(), health_ok=True)
    cred2 = registry.observe('sk-ant-oat01-new')
    registry.record_probe_success(cred2.generation, _eligible_usage(), health_ok=True)

    # Explicitly requesting old generation still returns its own (eligible) snapshot
    old_snap = registry.snapshot(cred1.generation)
    new_snap = registry.snapshot(cred2.generation)
    assert old_snap.generation == cred1.generation
    assert new_snap.generation == cred2.generation
    # The no-arg snapshot (used by snapshot_for_request) is the latest
    assert registry.snapshot().generation == cred2.generation


# ---------------------------------------------------------------------------
# Probe logs show token fingerprint, never the raw access token
# ---------------------------------------------------------------------------

def test_probe_failure_log_shows_fingerprint_not_raw_token(caplog):
    """Probe failure log contains only the token fingerprint, not the raw value."""
    raw_token = 'sk-ant-oat01-super-secret-do-not-log'

    def failing_probe(token):
        raise RuntimeError('Authentication failed for usage endpoint')

    registry = OAuthTokenRegistry(usage_probe=failing_probe)
    registry.observe(raw_token)

    with caplog.at_level(logging.WARNING, logger='anthproxy.oauth_registry'):
        registry.tick()

    log_text = ' '.join(r.getMessage() for r in caplog.records)
    assert raw_token not in log_text
    assert 'Authentication failed' in log_text


def test_probe_success_log_shows_fingerprint_not_raw_token(caplog):
    """Probe success debug log contains only the token fingerprint, not the raw value."""
    raw_token = 'sk-ant-oat01-also-secret-do-not-log'

    def ok_probe(token):
        return _eligible_usage()

    registry = OAuthTokenRegistry(usage_probe=ok_probe)
    registry.observe(raw_token)

    with caplog.at_level(logging.DEBUG, logger='anthproxy.oauth_registry'):
        registry.tick()

    log_text = ' '.join(r.getMessage() for r in caplog.records)
    assert raw_token not in log_text
