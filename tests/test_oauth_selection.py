import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from anthproxy.config import Config
from anthproxy.oauth_registry import OAuthTokenRegistry
from anthproxy.selector import AutoSelector
from anthproxy.server import BackendRegistry


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
