import threading

import pytest

from anthproxy.config import Config
from anthproxy.constants import SESSION_SUBSCRIPTION_SENTINEL
from anthproxy.server import (
    BackendError,
    BackendRegistry,
    _format_usage_snapshot,
    build_backend,
    create_server,
    make_handler_class,
)


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        self._available = True
        self._status_exc = None

    def five_hour_status(self, config):
        if self._status_exc is not None:
            raise self._status_exc
        from anthproxy._shared import FiveHourStatus
        return FiveHourStatus(available=self._available, resets_at=None)


def _registry(initial='bedrock', monkeypatch=None):
    config = Config(backend=initial)
    backend = _FakeBackend(initial)
    registry = BackendRegistry(config, backend)
    return config, registry


def _patch_build(monkeypatch, fail_for=None):
    created = {}

    def fake_build(name, config):
        if fail_for and name in fail_for:
            raise BackendError(f'cannot build {name}')
        created.setdefault(name, _FakeBackend(name))
        return created[name]

    monkeypatch.setattr('anthproxy.server.build_backend', fake_build)
    return created


def _patch_build_with_auth(monkeypatch, fail_for=None):
    """Like _patch_build but also stubs out auth for subscription backends."""
    created = _patch_build(monkeypatch, fail_for=fail_for)
    monkeypatch.setattr(
        'anthproxy.codex.auth.ensure_credentials_noninteractive', lambda cfg: None)
    monkeypatch.setattr(
        'anthproxy.anthropic.auth.ensure_credentials_noninteractive', lambda cfg: None)
    return created
    with pytest.raises(BackendError):
        build_backend('nope', Config())


def test_codex_weekly_only_usage_snapshot_preserves_168_hour_window():
    cache = {
        'primary': {
            'used_percent': 62,
            'window_seconds': 168 * 60 * 60,
            'reset_at': 1781200000,
        },
    }
    cache['weekly'] = cache['primary']

    usage = _format_usage_snapshot('codex', cache)

    assert usage['five_hour'] is None
    assert usage['weekly']['pct'] == 62
    assert usage['weekly']['window_hours'] == 168


def test_anthropic_usage_snapshot_sets_hourly_meter_durations():
    usage = _format_usage_snapshot('anthropic', {
        'five_hour': {'utilization': 25, 'resets_at': None},
        'seven_day': {'utilization': 70, 'resets_at': None},
    })

    assert usage['five_hour']['window_hours'] == 5
    assert usage['weekly']['window_hours'] == 168


def test_initial_snapshot_matches_startup(monkeypatch):
    config, registry = _registry('anthropic')
    snap = registry.snapshot()
    assert snap.name == 'anthropic'
    assert snap.config.backend == 'anthropic'
    assert snap.backend.name == 'anthropic'


def test_switch_changes_active_and_config(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)
    result = registry.switch('codex')
    assert result.kind == 'changed'
    assert result.previous == 'bedrock'
    assert result.current == 'codex'
    assert registry.active_name() == 'codex'
    assert config.backend == 'codex'
    assert registry.snapshot().config.backend == 'codex'


def test_switch_unchanged_is_noop(monkeypatch):
    config, registry = _registry('bedrock')
    result = registry.switch('bedrock')
    assert result.kind == 'unchanged'


def test_switch_invalid_keeps_state(monkeypatch):
    config, registry = _registry('bedrock')
    result = registry.switch('nope')
    assert result.kind == 'invalid'
    assert registry.active_name() == 'bedrock'
    assert config.backend == 'bedrock'


def test_switch_failed_keeps_state(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch, fail_for={'anthropic'})
    result = registry.switch('anthropic')
    assert result.kind == 'failed'
    assert result.current == 'bedrock'
    assert registry.active_name() == 'bedrock'
    assert config.backend == 'bedrock'


def test_instances_cached_and_state_preserved(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)
    registry.switch('anthropic')
    backend_first = registry.snapshot().backend
    registry.switch('bedrock')
    registry.switch('anthropic')
    backend_second = registry.snapshot().backend
    assert backend_first is backend_second  # same instance reused across round trip


def test_codex_switch_runs_noninteractive_readiness(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)
    calls = []
    monkeypatch.setattr(
        'anthproxy.codex.auth.ensure_credentials_noninteractive',
        lambda cfg: calls.append(cfg),
    )
    result = registry.switch('codex')
    assert result.kind == 'changed'
    assert len(calls) == 1


def test_openrouter_switch_requires_api_key(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)
    result = registry.switch('openrouter')
    assert result.kind == 'failed'
    assert result.current == 'bedrock'
    assert 'OPENROUTER_API_KEY' in result.error


def test_openrouter_switch_succeeds_with_api_key(monkeypatch):
    config, registry = _registry('bedrock')
    config.openrouter_api_key = 'sk-or-test'
    _patch_build(monkeypatch)
    result = registry.switch('openrouter')
    assert result.kind == 'changed'
    assert result.current == 'openrouter'


def test_codex_switch_failed_readiness_keeps_state(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)

    def fail(cfg):
        raise RuntimeError('No Codex credentials found.')

    monkeypatch.setattr(
        'anthproxy.codex.auth.ensure_credentials_noninteractive', fail)
    result = registry.switch('codex')
    assert result.kind == 'failed'
    assert 'No Codex credentials' in result.error
    assert registry.active_name() == 'bedrock'


def test_inflight_snapshot_unaffected_by_switch(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)
    snap = registry.snapshot()
    registry.switch('anthropic')
    assert snap.name == 'bedrock'
    assert snap.backend.name == 'bedrock'
    assert snap.config.backend == 'bedrock'


def test_concurrent_switches_never_mismatch(monkeypatch):
    config, registry = _registry('bedrock')
    _patch_build(monkeypatch)
    errors = []

    def worker(target):
        for _ in range(50):
            registry.switch(target)
            snap = registry.snapshot()
            if snap.name != snap.backend.name or snap.config.backend != snap.name:
                errors.append((snap.name, snap.backend.name, snap.config.backend))

    threads = [threading.Thread(target=worker, args=(n,)) for n in ('anthropic', 'bedrock', 'anthropic')]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_make_handler_class_wires_stats_collector():
    config, registry = _registry('bedrock')
    selector = object()
    stats_collector = object()

    handler_class = make_handler_class(registry, config, selector, stats_collector)

    assert handler_class.registry is registry
    assert handler_class.config is config
    assert handler_class.selector is selector
    assert handler_class.stats_collector is stats_collector


# ---------------------------------------------------------------------------
# Per-session backend override tests
# ---------------------------------------------------------------------------

class TestSessionBackend:
    def test_set_session_backend_records_override(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        result = registry.set_session_backend('sess-a', 'anthropic')
        assert result.kind == 'changed'
        assert result.current == 'anthropic'
        snap = registry.snapshot('sess-a')
        assert snap.name == 'anthropic'
        assert snap.session_pinned is True
        # Global active is unaffected
        assert registry.active_name() == 'bedrock'
        assert config.backend == 'bedrock'

    def test_snapshot_without_session_key_uses_global(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        registry.set_session_backend('sess-a', 'anthropic')
        snap = registry.snapshot()
        assert snap.name == 'bedrock'
        assert snap.session_pinned is False

    def test_different_session_key_gets_global(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        registry.set_session_backend('sess-a', 'anthropic')
        snap = registry.snapshot('sess-b')
        assert snap.name == 'bedrock'
        assert snap.session_pinned is False

    def test_set_session_backend_unchanged_if_same(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        registry.set_session_backend('sess-a', 'anthropic')
        result = registry.set_session_backend('sess-a', 'anthropic')
        assert result.kind == 'unchanged'
        assert result.current == 'anthropic'

    def test_set_session_backend_invalid_name(self, monkeypatch):
        config, registry = _registry('bedrock')
        result = registry.set_session_backend('sess-a', 'nope')
        assert result.kind == 'invalid'
        assert registry.session_backend('sess-a') is None

    def test_set_session_backend_failed_prep_no_override(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch, fail_for={'anthropic'})
        result = registry.set_session_backend('sess-a', 'anthropic')
        assert result.kind == 'failed'
        assert registry.session_backend('sess-a') is None
        assert registry.active_name() == 'bedrock'

    def test_clear_session_backend_reverts(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        registry.set_session_backend('sess-a', 'anthropic')
        existed = registry.clear_session_backend('sess-a')
        assert existed is True
        snap = registry.snapshot('sess-a')
        assert snap.name == 'bedrock'
        assert snap.session_pinned is False

    def test_clear_session_backend_not_set_returns_false(self, monkeypatch):
        config, registry = _registry('bedrock')
        existed = registry.clear_session_backend('sess-x')
        assert existed is False

    def test_session_backend_query(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        assert registry.session_backend('sess-a') is None
        registry.set_session_backend('sess-a', 'anthropic')
        assert registry.session_backend('sess-a') == 'anthropic'

    def test_session_override_survives_global_switch(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        registry.set_session_backend('sess-a', 'anthropic')
        registry.switch('anthropic')
        snap = registry.snapshot('sess-a')
        assert snap.name == 'anthropic'
        assert snap.session_pinned is True
        # Another session sees the new global
        snap_b = registry.snapshot('sess-b')
        assert snap_b.name == 'anthropic'

    def test_session_overrides_overflow_eviction(self, monkeypatch):
        from anthproxy.server import BackendRegistry
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        # Lower the cap to something testable
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 3
        try:
            for i in range(4):
                registry.set_session_backend(f'sess-{i}', 'anthropic')
            # Only 3 entries should remain (oldest evicted)
            assert len(registry._session_overrides) == 3
            # 'sess-0' should have been evicted
            assert registry.session_backend('sess-0') is None
            # The last three should still be set
            for i in range(1, 4):
                assert registry.session_backend(f'sess-{i}') == 'anthropic'
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap


# ---------------------------------------------------------------------------
# set_session_subscription + sentinel resolution
# ---------------------------------------------------------------------------

class TestSessionSubscription:

    def test_set_session_subscription_records_sentinel(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        result = registry.set_session_subscription('sess-a')
        assert result.kind == 'changed'
        assert result.current == SESSION_SUBSCRIPTION_SENTINEL
        assert registry.session_backend('sess-a') == SESSION_SUBSCRIPTION_SENTINEL

    def test_snapshot_with_subscription_sentinel_flags_set(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        snap = registry.snapshot('sess-a')
        assert snap.session_pinned is True
        assert snap.session_subscription is True

    def test_snapshot_resolves_subscription_via_resolver(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        registry.set_subscription_resolver(lambda: 'codex')
        snap = registry.snapshot('sess-a')
        assert snap.name == 'codex'
        assert snap.session_subscription is True

    def test_snapshot_subscription_resolver_none_falls_back_to_anthropic(self, monkeypatch):
        config, registry = _registry('bedrock')  # active is bedrock (non-subscription)
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        # No resolver → active is bedrock → fallback to anthropic
        snap = registry.snapshot('sess-a')
        assert snap.name == 'anthropic'

    def test_snapshot_subscription_resolver_none_uses_active_when_subscription(self, monkeypatch):
        config, registry = _registry('codex')
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        # No resolver → active is codex (a subscription backend) → use codex
        snap = registry.snapshot('sess-a')
        assert snap.name == 'codex'

    def test_snapshot_subscription_resolver_returns_non_sub_falls_back(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        registry.set_subscription_resolver(lambda: 'bedrock')  # invalid resolver result
        snap = registry.snapshot('sess-a')
        # bedrock is not a subscription backend → fallback to active-or-anthropic
        assert snap.name == 'anthropic'

    def test_subscription_lock_unchanged_when_already_set(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        result = registry.set_session_subscription('sess-a')
        assert result.kind == 'unchanged'

    def test_subscription_lock_survives_global_switch(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        registry.set_session_subscription('sess-a')
        registry.switch('anthropic')  # global switches away
        snap = registry.snapshot('sess-a')
        assert snap.session_subscription is True
        # Another session sees the new global
        snap_b = registry.snapshot('sess-b')
        assert snap_b.name == 'anthropic'

    def test_subscription_lru_eviction(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 2
        try:
            registry.set_session_subscription('sess-0')
            registry.set_session_subscription('sess-1')
            registry.set_session_subscription('sess-2')  # evicts sess-0
            assert registry.session_backend('sess-0') is None
            assert registry.session_backend('sess-1') == SESSION_SUBSCRIPTION_SENTINEL
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap

    def test_set_session_subscription_fails_when_all_sub_backends_fail(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch, fail_for={'anthropic', 'codex', 'openrouter'})
        # auth stubs still needed to reach prepare
        monkeypatch.setattr(
            'anthproxy.codex.auth.ensure_credentials_noninteractive', lambda cfg: None)
        monkeypatch.setattr(
            'anthproxy.anthropic.auth.ensure_credentials_noninteractive', lambda cfg: None)
        result = registry.set_session_subscription('sess-a')
        assert result.kind == 'failed'
        assert registry.session_backend('sess-a') is None


class TestCreateServerWiresResolver:

    def test_create_server_wires_subscription_resolver(self, monkeypatch):
        from unittest.mock import MagicMock
        config, registry = _registry('bedrock')
        selector = MagicMock()
        selector.current_subscription_backend.return_value = 'anthropic'
        # Patch ThreadingHTTPServer to avoid actually binding a port
        monkeypatch.setattr('anthproxy.server.ThreadingHTTPServer', MagicMock())
        create_server(config, registry, selector=selector)
        assert registry._subscription_resolver is selector.current_subscription_backend

    def test_create_server_no_resolver_when_selector_none(self, monkeypatch):
        from unittest.mock import MagicMock
        config, registry = _registry('bedrock')
        monkeypatch.setattr('anthproxy.server.ThreadingHTTPServer', MagicMock())
        create_server(config, registry, selector=None)
        assert registry._subscription_resolver is None




# ---------------------------------------------------------------------------
# BackendRegistry.set_model_routing
# ---------------------------------------------------------------------------

class TestSetModelRouting:
    def test_set_model_routing_on_mutates_config(self):
        config, registry = _registry('bedrock')
        assert config.auto_model_routing is False
        registry.set_model_routing(True)
        assert config.auto_model_routing is True

    def test_set_model_routing_off_mutates_config(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = True  # start enabled
        registry.set_model_routing(False)
        assert config.auto_model_routing is False

    def test_set_model_routing_propagates_to_snapshot_config(self):
        """toggle is visible on the next snapshot()'s config."""
        config, registry = _registry('bedrock')
        registry.set_model_routing(True)
        snap = registry.snapshot()
        assert snap.config.auto_model_routing is True

    def test_set_model_routing_off_propagates_to_snapshot_config(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = True
        registry.set_model_routing(False)
        snap = registry.snapshot()
        assert snap.config.auto_model_routing is False

    def test_set_model_routing_does_not_change_active_backend(self):
        config, registry = _registry('bedrock')
        registry.set_model_routing(True)
        assert registry.active_name() == 'bedrock'
        assert config.backend == 'bedrock'


# ---------------------------------------------------------------------------
# BackendRegistry.set_session_model_routing / session_model_routing
# ---------------------------------------------------------------------------

class TestSetSessionModelRouting:
    def test_set_session_routing_on_stored(self):
        config, registry = _registry('bedrock')
        registry.set_session_model_routing('sess-1', True)
        assert registry.session_model_routing('sess-1') is True

    def test_set_session_routing_off_stored(self):
        config, registry = _registry('bedrock')
        registry.set_session_model_routing('sess-1', False)
        assert registry.session_model_routing('sess-1') is False

    def test_clear_session_routing_returns_none(self):
        config, registry = _registry('bedrock')
        registry.set_session_model_routing('sess-1', True)
        registry.set_session_model_routing('sess-1', None)
        assert registry.session_model_routing('sess-1') is None

    def test_clear_nonexistent_session_is_noop(self):
        config, registry = _registry('bedrock')
        registry.set_session_model_routing('no-session', None)
        assert registry.session_model_routing('no-session') is None

    def test_session_routing_on_propagates_to_snapshot(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = False  # global off
        registry.set_session_model_routing('sess-1', True)
        snap = registry.snapshot('sess-1')
        assert snap.config.auto_model_routing is True

    def test_session_routing_off_propagates_to_snapshot(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = True  # global on
        registry.set_session_model_routing('sess-1', False)
        snap = registry.snapshot('sess-1')
        assert snap.config.auto_model_routing is False

    def test_session_routing_override_does_not_affect_other_sessions(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = False
        registry.set_session_model_routing('sess-1', True)
        snap2 = registry.snapshot('sess-2')
        assert snap2.config.auto_model_routing is False

    def test_session_routing_override_does_not_affect_no_session(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = False
        registry.set_session_model_routing('sess-1', True)
        snap = registry.snapshot()  # no session key
        assert snap.config.auto_model_routing is False

    def test_session_routing_cleared_follows_global(self):
        config, registry = _registry('bedrock')
        config.auto_model_routing = True
        registry.set_session_model_routing('sess-1', False)
        registry.set_session_model_routing('sess-1', None)  # clear
        snap = registry.snapshot('sess-1')
        assert snap.config.auto_model_routing is True

    def test_session_routing_does_not_change_active_backend(self):
        config, registry = _registry('bedrock')
        registry.set_session_model_routing('sess-1', True)
        assert registry.active_name() == 'bedrock'


# ---------------------------------------------------------------------------
# BackendRegistry.set_session_routed_tier / session_routed_tier
# ---------------------------------------------------------------------------

class TestSessionRoutedTier:
    def test_round_trip(self):
        config, registry = _registry('bedrock')
        registry.set_session_routed_tier('sess-1', 'sonnet')
        assert registry.session_routed_tier('sess-1') == 'sonnet'

    def test_unknown_key_returns_none(self):
        config, registry = _registry('bedrock')
        assert registry.session_routed_tier('sess-x') is None

    def test_overwrite_updates_value(self):
        config, registry = _registry('bedrock')
        registry.set_session_routed_tier('sess-1', 'haiku')
        registry.set_session_routed_tier('sess-1', 'opus')
        assert registry.session_routed_tier('sess-1') == 'opus'

    def test_different_sessions_independent(self):
        config, registry = _registry('bedrock')
        registry.set_session_routed_tier('sess-a', 'haiku')
        registry.set_session_routed_tier('sess-b', 'opus')
        assert registry.session_routed_tier('sess-a') == 'haiku'
        assert registry.session_routed_tier('sess-b') == 'opus'

    def test_oldest_entry_evicted_on_overflow(self):
        config, registry = _registry('bedrock')
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 3
        try:
            for i in range(4):
                registry.set_session_routed_tier(f'sess-{i}', 'sonnet')
            # Only 3 entries remain; the oldest (sess-0) is evicted
            assert len(registry._session_routed_tier) == 3
            assert registry.session_routed_tier('sess-0') is None
            for i in range(1, 4):
                assert registry.session_routed_tier(f'sess-{i}') == 'sonnet'
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap

    def test_write_refreshes_lru_recency(self):
        """Re-writing an entry bumps it to most-recent so it survives eviction."""
        config, registry = _registry('bedrock')
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 2
        try:
            registry.set_session_routed_tier('sess-a', 'haiku')
            registry.set_session_routed_tier('sess-b', 'opus')
            # Re-write sess-a → it becomes MRU; next write evicts sess-b
            registry.set_session_routed_tier('sess-a', 'sonnet')
            registry.set_session_routed_tier('sess-c', 'opus')
            # sess-b should be evicted; sess-a and sess-c survive
            assert registry.session_routed_tier('sess-b') is None
            assert registry.session_routed_tier('sess-a') == 'sonnet'
            assert registry.session_routed_tier('sess-c') == 'opus'
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap

    def test_shared_cap_with_routing_overrides(self):
        """_session_routed_tier uses _MAX_SESSION_OVERRIDES (shared constant) independently."""
        config, registry = _registry('bedrock')
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 2
        try:
            registry.set_session_routed_tier('sess-1', 'haiku')
            registry.set_session_model_routing('sess-1', True)
            registry.set_session_routed_tier('sess-2', 'sonnet')
            registry.set_session_routed_tier('sess-3', 'opus')  # evicts sess-1 from tier cache
            assert registry.session_routed_tier('sess-1') is None
            # But the routing override is a separate dict — unaffected
            assert registry.session_model_routing('sess-1') is True
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap


class TestSessionContext:
    def test_unknown_key_returns_identity_default(self):
        config, registry = _registry('bedrock')
        assert registry.session_context('sess-x') == (0, 1.0)

    def test_round_trip(self):
        config, registry = _registry('bedrock')
        registry.record_session_context('sess-1', 150_000, 1.7)
        assert registry.session_context('sess-1') == (150_000, 1.7)

    def test_overwrite_replaces_not_sums(self):
        config, registry = _registry('bedrock')
        registry.record_session_context('sess-1', 150_000, 1.7)
        registry.record_session_context('sess-1', 90_000, 1.2)  # e.g. after /compact
        assert registry.session_context('sess-1') == (90_000, 1.2)

    def test_different_sessions_independent(self):
        config, registry = _registry('bedrock')
        registry.record_session_context('sess-a', 10_000, 1.1)
        registry.record_session_context('sess-b', 20_000, 1.5)
        assert registry.session_context('sess-a') == (10_000, 1.1)
        assert registry.session_context('sess-b') == (20_000, 1.5)

    def test_oldest_entry_evicted_on_overflow(self):
        config, registry = _registry('bedrock')
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 3
        try:
            for i in range(4):
                registry.record_session_context(f'sess-{i}', 1000 * (i + 1), 1.0)
            assert len(registry._session_context_obs) == 3
            assert registry.session_context('sess-0') == (0, 1.0)  # evicted → default
            for i in range(1, 4):
                assert registry.session_context(f'sess-{i}') == (1000 * (i + 1), 1.0)
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap

    def test_write_refreshes_lru_recency(self):
        config, registry = _registry('bedrock')
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 2
        try:
            registry.record_session_context('sess-a', 10_000, 1.0)
            registry.record_session_context('sess-b', 20_000, 1.0)
            registry.record_session_context('sess-a', 30_000, 1.0)  # sess-a → MRU
            registry.record_session_context('sess-c', 40_000, 1.0)  # evicts sess-b
            assert registry.session_context('sess-b') == (0, 1.0)
            assert registry.session_context('sess-a') == (30_000, 1.0)
            assert registry.session_context('sess-c') == (40_000, 1.0)
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap

    def test_shared_cap_independent_of_other_session_maps(self):
        config, registry = _registry('bedrock')
        original_cap = BackendRegistry._MAX_SESSION_OVERRIDES
        BackendRegistry._MAX_SESSION_OVERRIDES = 2
        try:
            registry.record_session_context('sess-1', 10_000, 1.0)
            registry.set_session_routed_tier('sess-1', 'opus')
            registry.record_session_context('sess-2', 20_000, 1.0)
            registry.record_session_context('sess-3', 30_000, 1.0)  # evicts sess-1 here only
            assert registry.session_context('sess-1') == (0, 1.0)
            # The tier cache is a separate dict — unaffected by context eviction.
            assert registry.session_routed_tier('sess-1') == 'opus'
        finally:
            BackendRegistry._MAX_SESSION_OVERRIDES = original_cap


# ---------------------------------------------------------------------------
# Per-request backend preference (X-Anthproxy-Override: prefer:<name>)
# ---------------------------------------------------------------------------

class TestSnapshotPreferBackend:
    def test_prefer_cached_healthy(self, monkeypatch):
        config, registry = _registry('bedrock')
        created = _patch_build_with_auth(monkeypatch)
        config.auto_backend_mode = 'auto'
        # Pre-create the preferred backend by switching to it and back.
        registry.switch('anthropic')
        registry.switch('bedrock')
        assert 'anthropic' in created

        snap = registry.snapshot(prefer_backend='anthropic')
        assert snap.name == 'anthropic'
        assert snap.session_pinned is False

    def test_prefer_same_as_active_is_noop(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        config.auto_backend_mode = 'auto'
        snap = registry.snapshot(prefer_backend='bedrock')
        assert snap.name == 'bedrock'

    def test_prefer_exhausted_falls_back(self, monkeypatch):
        config, registry = _registry('bedrock')
        created = _patch_build_with_auth(monkeypatch)
        config.auto_backend_mode = 'auto'
        registry.switch('anthropic')
        registry.switch('bedrock')
        # Mark the cached backend instance as exhausted.
        created['anthropic']._available = False

        snap = registry.snapshot(prefer_backend='anthropic')
        # Should fall back to the active backend (bedrock).
        assert snap.name == 'bedrock'

    def test_prefer_unknown_health_honored(self, monkeypatch):
        """When five_hour_status() raises, the preference is honored (conservative)."""
        config, registry = _registry('bedrock')
        created = _patch_build_with_auth(monkeypatch)
        config.auto_backend_mode = 'auto'
        registry.switch('anthropic')
        registry.switch('bedrock')
        created['anthropic']._status_exc = RuntimeError('usage endpoint down')

        snap = registry.snapshot(prefer_backend='anthropic')
        assert snap.name == 'anthropic'

    def test_prefer_unknown_name_silently_ignored(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build(monkeypatch)
        snap = registry.snapshot(prefer_backend='nonexistent')
        assert snap.name == 'bedrock'

    def test_prefer_session_override_wins(self, monkeypatch):
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        config.auto_backend_mode = 'auto'
        # Pin the session to a backend.
        registry.set_session_backend('sess-1', 'anthropic')
        # Now request prefer:codex — the session pin should take precedence.
        snap = registry.snapshot(session_key='sess-1', prefer_backend='codex')
        assert snap.name == 'anthropic'
        assert snap.session_pinned is True

    def test_prefer_bedrock_in_subscription_mode_ignored(self, monkeypatch):
        config, registry = _registry('anthropic')
        _patch_build(monkeypatch)
        config.auto_backend_mode = 'subscription'
        snap = registry.snapshot(prefer_backend='bedrock')
        # Bedrock is not a subscription backend; preference ignored.
        assert snap.name == 'anthropic'

    def test_prefer_construct_on_demand(self, monkeypatch):
        """When the preferred backend is configured but not cached, it is constructed."""
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        # codex is a subscription backend, so this works in default subscription mode.
        # codex is not yet in _instances; preference should construct it.
        snap = registry.snapshot(prefer_backend='codex')
        assert snap.name == 'codex'
        assert snap.session_pinned is False

    def test_prefer_not_configured_falls_back(self, monkeypatch):
        """When the preferred backend cannot be constructed, fall back silently."""
        config, registry = _registry('bedrock')
        # Stub build to fail for codex; also stub auth to succeed.
        _patch_build(monkeypatch, fail_for={'codex'})
        monkeypatch.setattr(
            'anthproxy.codex.auth.ensure_credentials_noninteractive', lambda cfg: None)
        snap = registry.snapshot(prefer_backend='codex')
        # Should fall back to the active backend (bedrock).
        assert snap.name == 'bedrock'

    def test_prefer_preserves_session_routing_override(self, monkeypatch):
        """Per-session routing override is still applied when prefer_backend is set."""
        config, registry = _registry('bedrock')
        _patch_build_with_auth(monkeypatch)
        config.auto_backend_mode = 'auto'
        registry.switch('anthropic')
        registry.switch('bedrock')
        # Set a session routing override.
        registry.set_session_model_routing('sess-1', False)

        snap = registry.snapshot(session_key='sess-1', prefer_backend='anthropic')
        assert snap.name == 'anthropic'
        # The session routing override (False) should be applied to the config.
        assert snap.config.auto_model_routing is False


# ---------------------------------------------------------------------------
# BackendRegistry.cached_subscription_instances
# ---------------------------------------------------------------------------

class TestCachedSubscriptionInstances:
    def test_returns_only_instantiated_subscription_backends(self):
        """Only subscription backends already in _instances are returned."""
        config, registry = _registry('bedrock')
        # Inject a fake anthropic instance directly — no build call.
        fake_anthropic = _FakeBackend('anthropic')
        registry._instances['anthropic'] = fake_anthropic

        result = registry.cached_subscription_instances()

        assert 'anthropic' in result
        assert result['anthropic'] is fake_anthropic
        # codex and openrouter were never instantiated, so must be absent.
        assert 'codex' not in result
        assert 'openrouter' not in result
        # Non-subscription backends must never appear.
        assert 'bedrock' not in result

    def test_empty_instances_returns_empty_dict(self):
        """When no subscription backends are in _instances, returns {}."""
        config, registry = _registry('bedrock')
        # bedrock is in _instances; no subscription backend is.
        result = registry.cached_subscription_instances()
        assert result == {}

    def test_does_not_trigger_instantiation(self, monkeypatch):
        """Calling cached_subscription_instances never calls build_backend."""
        config, registry = _registry('bedrock')
        build_calls = []

        def fake_build(name, cfg):
            build_calls.append(name)
            return _FakeBackend(name)

        monkeypatch.setattr('anthproxy.server.build_backend', fake_build)
        registry.cached_subscription_instances()
        assert build_calls == []


# ---------------------------------------------------------------------------
# BackendRegistry.usage_snapshot — age_secs field
# ---------------------------------------------------------------------------

class TestUsageSnapshotAgeSecs:
    def _make_backend_with_cache(self, name, cache, cached_at):
        """Construct a fake backend with _usage_cache and _usage_cached_at set."""
        b = _FakeBackend(name)
        b._usage_cache = cache
        b._usage_cached_at = cached_at
        return b

    def test_age_secs_is_integer_when_cached_at_is_positive(self, monkeypatch):
        """age_secs is a non-negative integer when _usage_cached_at > 0."""
        import time as _time
        config, registry = _registry('bedrock')
        # Use a cached_at value a few seconds in the past.
        cached_at = _time.monotonic() - 5.0
        fake = self._make_backend_with_cache(
            'anthropic',
            {'five_hour': {'utilization': 10, 'resets_at': None},
             'seven_day': {'utilization': 20, 'resets_at': None}},
            cached_at,
        )
        registry._instances['anthropic'] = fake

        result = registry.usage_snapshot()

        assert 'anthropic' in result
        age = result['anthropic']['age_secs']
        assert isinstance(age, int)
        assert age >= 5  # at least 5 s have elapsed

    def test_age_secs_is_none_when_cached_at_is_zero(self):
        """age_secs is None when _usage_cached_at == 0.0 (never populated)."""
        config, registry = _registry('bedrock')
        fake = self._make_backend_with_cache(
            'anthropic',
            {'five_hour': {'utilization': 10, 'resets_at': None},
             'seven_day': {'utilization': 20, 'resets_at': None}},
            0.0,
        )
        registry._instances['anthropic'] = fake

        result = registry.usage_snapshot()

        assert 'anthropic' in result
        assert result['anthropic']['age_secs'] is None

    def test_unpopulated_cache_backend_absent_from_result(self):
        """Backend with no _usage_cache dict must not appear in the snapshot."""
        config, registry = _registry('bedrock')
        fake = _FakeBackend('anthropic')
        # _usage_cache is not set (or None) — mirrors SubscriptionBackend.__init__
        registry._instances['anthropic'] = fake

        result = registry.usage_snapshot()

        assert 'anthropic' not in result
