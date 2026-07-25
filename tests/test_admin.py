"""Tests for anthproxy.admin — the admin REST API handler module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from anthproxy import admin
from anthproxy.server import SwitchResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_registry(*, backend_names=None, active='bedrock', switch_result=None):
    """Return a MagicMock registry with sensible defaults."""
    registry = MagicMock()
    registry.active_name.return_value = active
    registry.list_backends.return_value = list(
        backend_names or ['bedrock', 'gauss', 'codex', 'anthropic', 'local', 'openrouter']
    )
    _switch_result = switch_result or SwitchResult(kind='changed', previous=active, current='bedrock')
    registry.switch.return_value = _switch_result
    registry.backend_status.return_value = {
        'bedrock': {'name': 'bedrock', 'available': True, 'active': True},
        'anthropic': {'name': 'anthropic', 'available': True, 'active': False},
    }
    # Config sub-object
    cfg = MagicMock()
    cfg.auto_model_routing = True
    cfg.auto_backend_mode = 'subscription'
    cfg.auto_backend = True
    cfg.auto_model_routing_classifier_model = 'haiku'
    cfg.auto_model_routing_long_context_threshold = 150_000
    cfg.auto_model_routing_affirmation_inherit = True
    cfg.auto_model_routing_mode = 'classifier'
    registry.config = cfg
    registry.usage_snapshot.return_value = {}
    registry.cached_subscription_instances.return_value = {}
    return registry


def _make_selector(status_line='auto-selection: on'):
    """Return a MagicMock AutoSelector with a fixed status_line."""
    selector = MagicMock()
    selector.status_line.return_value = status_line
    selector.resume.return_value = 'anthropic'
    selector.restrict_subscription.return_value = 'anthropic'
    return selector


def _make_db():
    """Return a MagicMock db with sensible default return values."""
    db = MagicMock()
    db.get_sessions.return_value = [
        {'session_id': 'sess1', 'request_count': 10},
        {'session_id': 'sess2', 'request_count': 5},
    ]
    db.get_sessions_count.return_value = 2
    db.get_session.return_value = {
        'session_id': 'sess1',
        'request_count': 10,
        'total_input_tokens': 1000,
    }
    db.get_session_metadata.return_value = {
        'pinned_backend': None,
        'pinned_tier': None,
    }
    db.get_trace.return_value = [
        {'request_ts': '2026-01-01T00:00:00', 'model': 'sonnet'},
    ]
    db.get_cost.return_value = [
        {'key': 'sonnet', 'requests': 100, 'cost_usd': 1.23},
    ]
    db.get_routing.return_value = {
        'reason_code_distribution': [
            {'reason_code': 'classifier', 'applied': 1, 'classification': 'standard', 'cnt': 5},
        ],
        'tier_transitions': [
            {'requested_model': 'sonnet', 'routed_model': 'haiku', 'cnt': 3},
        ],
    }
    db.get_config_changes.return_value = []
    db.get_session_overrides.return_value = []
    db.get_request.return_value = {
        'id': 123,
        'session_id': 'sess1',
        'model': 'sonnet',
        'system_prompt_content': 'You are helpful.',
        'tools_content': None,
    }
    db.get_stats.return_value = {
        'period': 'week',
        'backend_filter': None,
        'buckets': [
            {
                'label': '2026-07-01',
                'rows': [{'backend': 'anthropic', 'model_tier': 'sonnet', 'requests': 5}],
                'subtotal': {'requests': 5},
            }
        ],
        'total': {'requests': 5},
    }
    db.get_prompt.return_value = {
        'content_hash': 'a' * 64,
        'content_type': 'system',
        'content': 'You are helpful.',
        'char_count': 16,
        'first_seen_at': '2026-07-01T00:00:00',
    }
    db.get_session_summary.return_value = {
        'session_id': 'sess1',
        'request_count': 10,
        'total_input_tokens': 1000,
        'total_output_tokens': 500,
        'total_cache_creation': 200,
        'total_cache_read': 800,
        'estimated_cost_usd': 0.05,
    }
    return db


# ---------------------------------------------------------------------------
# GET /admin/sessions
# ---------------------------------------------------------------------------

class TestGetSessions:
    def test_returns_200_with_items_envelope(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/sessions', {}, registry, db)
        assert status == 200
        assert 'items' in body
        assert 'total' in body
        assert 'limit' in body
        assert 'offset' in body

    def test_passes_limit_and_offset(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions', {'limit': '10', 'offset': '20'}, registry, db)
        db.get_sessions.assert_called_once_with(limit=10, offset=20, q=None)

    def test_default_limit_offset(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions', {}, registry, db)
        db.get_sessions.assert_called_once_with(limit=50, offset=0, q=None)

    def test_invalid_limit_uses_default(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions', {'limit': 'bad'}, registry, db)
        db.get_sessions.assert_called_once_with(limit=50, offset=0, q=None)

    def test_total_comes_from_get_sessions_count(self):
        """M2: total must use db.get_sessions_count(), not len(items)."""
        registry = _make_registry()
        db = _make_db()
        # Return only 1 item but say count=999 — they differ deliberately
        db.get_sessions.return_value = [{'session_id': 's1'}]
        db.get_sessions_count.return_value = 999
        status, body = admin.handle_get('/admin/sessions', {}, registry, db)
        assert body['total'] == 999
        db.get_sessions_count.assert_called_once()

    def test_sort_by_accepted(self):
        """sort_by query param should be accepted without error."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(
            '/admin/sessions', {'sort_by': 'request_count'}, registry, db
        )
        assert status == 200

    def test_limit_clamped_to_1000(self):
        """M4: limit > 1000 must be clamped to 1000."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions', {'limit': '9999'}, registry, db)
        db.get_sessions.assert_called_once_with(limit=1000, offset=0, q=None)

    def test_passes_q_to_get_sessions(self):
        """Filter: q param must be forwarded to db.get_sessions()."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions', {'q': 'xyz123xyz'}, registry, db)
        db.get_sessions.assert_called_once_with(limit=50, offset=0, q='xyz123xyz')
        db.get_sessions_count.assert_called_once_with(q='xyz123xyz')

    def test_blank_q_treated_as_none(self):
        """Filter: whitespace-only q must be treated as None."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions', {'q': '   '}, registry, db)
        db.get_sessions.assert_called_once_with(limit=50, offset=0, q=None)
        db.get_sessions_count.assert_called_once_with(q=None)

    def test_q_in_response_envelope(self):
        """Filter: response envelope must include the q param."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/sessions', {'q': 'test'}, registry, db)
        assert body['q'] == 'test'


# ---------------------------------------------------------------------------
# GET /admin/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_returns_200_for_known_session(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/sessions/sess1', {}, registry, db)
        assert status == 200
        assert body['session_id'] == 'sess1'

    def test_returns_404_when_not_found(self):
        registry = _make_registry()
        db = _make_db()
        db.get_session.return_value = None
        status, body = admin.handle_get('/admin/sessions/ghost', {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'

    def test_calls_get_session_with_id(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions/sess1', {}, registry, db)
        db.get_session.assert_called_once_with('sess1')

    def test_percent_encoded_session_id_is_decoded(self):
        # Session keys are JSON blobs; the HTTP layer passes the raw percent-encoded
        # path without decoding.  admin.handle_get must decode each path segment so
        # the DB lookup matches the stored key.
        raw_id = '{"device_id":"abc","account_uuid":"","session_id":"362954d5-af9e-49"}'
        from urllib.parse import quote
        encoded_path = f'/admin/sessions/{quote(raw_id, safe="")}'
        registry = _make_registry()
        db = _make_db()
        db.get_session.return_value = {'session_id': raw_id}
        status, body = admin.handle_get(encoded_path, {}, registry, db)
        assert status == 200
        db.get_session.assert_called_once_with(raw_id)

    def test_percent_encoded_session_id_decoded_for_trace(self):
        raw_id = '{"device_id":"abc","account_uuid":"","session_id":"362954d5-af9e-49"}'
        from urllib.parse import quote
        encoded_path = f'/admin/sessions/{quote(raw_id, safe="")}/trace'
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(encoded_path, {}, registry, db)
        assert status == 200
        db.get_trace.assert_called_once_with(raw_id, anchor=None, limit=100, offset=0, q=None)

    def test_percent_encoded_session_id_decoded_for_post_set_backend(self):
        raw_id = '{"device_id":"abc","account_uuid":"","session_id":"362954d5-af9e-49"}'
        from urllib.parse import quote
        encoded_path = f'/admin/sessions/{quote(raw_id, safe="")}/set-backend'
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(encoded_path, {'backend': 'bedrock'}, registry, db)
        assert status == 200
        registry.set_session_backend.assert_called_once_with(raw_id, 'bedrock')


# ---------------------------------------------------------------------------
# GET /admin/sessions/{session_id}/trace
# ---------------------------------------------------------------------------

class TestGetTrace:
    def test_returns_200_with_items_envelope(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/sessions/sess1/trace', {}, registry, db)
        assert status == 200
        assert 'items' in body
        assert body['session_id'] == 'sess1'
        assert 'anchor' in body

    def test_passes_anchor_limit_offset(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get(
            '/admin/sessions/sess1/trace',
            {'anchor': 'abc', 'limit': '20', 'offset': '5'},
            registry,
            db,
        )
        db.get_trace.assert_called_once_with('sess1', anchor='abc', limit=20, offset=5, q=None)

    def test_anchor_none_when_absent(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/sessions/sess1/trace', {}, registry, db)
        assert body['anchor'] is None

    def test_anchor_propagated(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(
            '/admin/sessions/sess1/trace', {'anchor': 'myanchor'}, registry, db
        )
        assert body['anchor'] == 'myanchor'

    def test_trace_limit_clamped_to_1000(self):
        """M4: trace limit > 1000 must be clamped."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/sessions/s1/trace', {'limit': '5000'}, registry, db)
        db.get_trace.assert_called_once_with('s1', anchor=None, limit=1000, offset=0, q=None)

    def test_passes_q_to_get_trace(self):
        """Filter: q param must be forwarded to db.get_trace()."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get(
            '/admin/sessions/sess1/trace',
            {'q': 'xyz123xyz'},
            registry,
            db,
        )
        db.get_trace.assert_called_once_with('sess1', anchor=None, limit=100, offset=0, q='xyz123xyz')
        db.get_trace_count.assert_called_once_with('sess1', anchor=None, q='xyz123xyz')

    def test_q_and_anchor_both_forwarded(self):
        """Filter: q and anchor params must both be forwarded."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get(
            '/admin/sessions/sess1/trace',
            {'q': 'test', 'anchor': 'conv123'},
            registry,
            db,
        )
        db.get_trace.assert_called_once_with('sess1', anchor='conv123', limit=100, offset=0, q='test')
        db.get_trace_count.assert_called_once_with('sess1', anchor='conv123', q='test')

    def test_blank_q_trace_treated_as_none(self):
        """Filter: whitespace-only q in trace must be treated as None."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get(
            '/admin/sessions/sess1/trace',
            {'q': '  '},
            registry,
            db,
        )
        db.get_trace.assert_called_once_with('sess1', anchor=None, limit=100, offset=0, q=None)
        db.get_trace_count.assert_called_once_with('sess1', anchor=None, q=None)

    def test_q_in_trace_response_envelope(self):
        """Filter: trace response envelope must include the q param."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(
            '/admin/sessions/sess1/trace',
            {'q': 'filter_term'},
            registry,
            db,
        )
        assert body['q'] == 'filter_term'


# ---------------------------------------------------------------------------
# GET /admin/cost
# ---------------------------------------------------------------------------

class TestGetCost:
    def test_returns_200_with_items(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/cost', {}, registry, db)
        assert status == 200
        assert 'items' in body
        assert body['group_by'] == 'model'
        assert body['time_range'] == '7d'

    def test_passes_group_by_and_since(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/cost', {'group_by': 'tier', 'time_range': '30d'}, registry, db)
        db.get_cost.assert_called_once_with(group_by='tier', since='-30 days')

    def test_passes_session_id_when_present(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/cost', {'session_id': 'sess1'}, registry, db)
        db.get_cost.assert_called_once_with(group_by='model', since='-7 days', session_id='sess1')

    def test_invalid_time_range_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(
            '/admin/cost', {'time_range': 'banana'}, registry, db
        )
        assert status == 400
        assert body['error'] == 'INVALID_TIME_RANGE'

    @pytest.mark.parametrize('tr,since', [
        ('1d', '-1 days'),
        ('7d', '-7 days'),
        ('30d', '-30 days'),
    ])
    def test_time_range_mapping(self, tr, since):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/cost', {'time_range': tr}, registry, db)
        db.get_cost.assert_called_once_with(group_by='model', since=since)

    def test_items_use_name_not_key(self):
        """DB rows have 'key'; response must expose it as 'name' for the frontend."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/cost', {}, registry, db)
        assert status == 200
        row = body['items'][0]
        assert 'name' in row
        assert row['name'] == 'sonnet'
        assert 'key' not in row


# ---------------------------------------------------------------------------
# GET /admin/routing
# ---------------------------------------------------------------------------

class TestGetRouting:
    def test_returns_200_with_reason_codes_and_tier_transitions(self):
        """Response must use frontend-expected field names."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert status == 200
        assert 'reason_codes' in body
        assert 'tier_transitions' in body

    def test_reason_codes_use_count_field(self):
        """Each reason_code entry must have a 'count' key (not 'cnt')."""
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert len(body['reason_codes']) == 1
        rc = body['reason_codes'][0]
        assert 'reason_code' in rc
        assert 'count' in rc
        assert 'cnt' not in rc
        assert rc['count'] == 5

    def test_tier_transitions_use_requested_tier_routed_tier_count(self):
        """Tier transition rows must use requested_tier/routed_tier/count (not model/cnt)."""
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert len(body['tier_transitions']) == 1
        tt = body['tier_transitions'][0]
        assert 'requested_tier' in tt
        assert 'routed_tier' in tt
        assert 'count' in tt
        assert 'requested_model' not in tt
        assert 'cnt' not in tt
        assert tt['requested_tier'] == 'sonnet'
        assert tt['routed_tier'] == 'haiku'
        assert tt['count'] == 3

    def test_passes_since(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/routing', {'time_range': '1d'}, registry, db)
        db.get_routing.assert_called_once_with(since='-1 days')

    def test_passes_session_id(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/routing', {'session_id': 'sess1'}, registry, db)
        db.get_routing.assert_called_once_with(since='-7 days', session_id='sess1')

    def test_invalid_time_range_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/routing', {'time_range': 'bad'}, registry, db)
        assert status == 400
        assert body['error'] == 'INVALID_TIME_RANGE'

    def test_empty_routing_data(self):
        """Empty distributions should return empty lists, not crash."""
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {'reason_code_distribution': [], 'tier_transitions': []}
        status, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert status == 200
        assert body['reason_codes'] == []
        assert body['tier_transitions'] == []

    def test_upgrade_downgrade_unchanged_counts(self):
        """sonnet→haiku transition is a downgrade."""
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['downgrade_count'] == 3
        assert body['upgrade_count'] == 0
        assert body['unchanged_count'] == 0

    def test_upgrade_count(self):
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [],
            'tier_transitions': [
                {'requested_model': 'haiku', 'routed_model': 'sonnet', 'cnt': 7},
            ],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['upgrade_count'] == 7
        assert body['downgrade_count'] == 0
        assert body['unchanged_count'] == 0

    def test_unchanged_count(self):
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [],
            'tier_transitions': [
                {'requested_model': 'sonnet', 'routed_model': 'sonnet', 'cnt': 4},
            ],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['unchanged_count'] == 4
        assert body['upgrade_count'] == 0
        assert body['downgrade_count'] == 0

    def test_fable_transitions_counted_as_known_tier(self):
        """fable is now rank 3; fable→fable is unchanged_count."""
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [],
            'tier_transitions': [
                {'requested_model': 'fable', 'routed_model': 'fable', 'cnt': 2},
            ],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['upgrade_count'] == 0
        assert body['downgrade_count'] == 0
        assert body['unchanged_count'] == 2

    def test_size_forced_count(self):
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [
                {'reason_code': 'size_forced_long_context', 'applied': 1, 'classification': None, 'cnt': 12},
            ],
            'tier_transitions': [],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['size_forced_count'] == 12
        assert body['affirmation_count'] == 0
        assert body['cached_tier_count'] == 0

    def test_affirmation_count(self):
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [
                {'reason_code': 'affirmation_inherited', 'applied': 1, 'classification': None, 'cnt': 6},
                {'reason_code': 'affirmation_floored_standard', 'applied': 1, 'classification': None, 'cnt': 2},
            ],
            'tier_transitions': [],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['affirmation_count'] == 8

    def test_cached_tier_count_all_four_codes(self):
        """All four cached_tier reason codes contribute to cached_tier_count."""
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [
                {'reason_code': 'session_cached_tier', 'applied': 1, 'classification': None, 'cnt': 1},
                {'reason_code': 'session_cached_walkback', 'applied': 1, 'classification': None, 'cnt': 2},
                {'reason_code': 'session_cached_tier_capped', 'applied': 0, 'classification': None, 'cnt': 3},
                {'reason_code': 'session_cached_walkback_capped', 'applied': 0, 'classification': None, 'cnt': 4},
            ],
            'tier_transitions': [],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert body['cached_tier_count'] == 10

    def test_original_model_distribution(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert 'original_model_distribution' in body
        assert len(body['original_model_distribution']) == 1
        entry = body['original_model_distribution'][0]
        assert entry['model'] == 'sonnet'
        assert entry['count'] == 3

    def test_routed_model_distribution(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        assert 'routed_model_distribution' in body
        assert len(body['routed_model_distribution']) == 1
        entry = body['routed_model_distribution'][0]
        assert entry['model'] == 'haiku'
        assert entry['count'] == 3

    def test_model_distributions_aggregate_across_transitions(self):
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {
            'reason_code_distribution': [],
            'tier_transitions': [
                {'requested_model': 'sonnet', 'routed_model': 'haiku', 'cnt': 3},
                {'requested_model': 'sonnet', 'routed_model': 'opus', 'cnt': 2},
            ],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        orig = {e['model']: e['count'] for e in body['original_model_distribution']}
        assert orig['sonnet'] == 5
        routed = {e['model']: e['count'] for e in body['routed_model_distribution']}
        assert routed['haiku'] == 3
        assert routed['opus'] == 2

    def test_new_fields_present_in_empty_data(self):
        """All new fields must be present even with empty distributions."""
        registry = _make_registry()
        db = _make_db()
        db.get_routing.return_value = {'reason_code_distribution': [], 'tier_transitions': []}
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        for field in (
            'upgrade_count', 'downgrade_count', 'unchanged_count',
            'size_forced_count', 'affirmation_count', 'cached_tier_count',
            'original_model_distribution', 'routed_model_distribution',
        ):
            assert field in body, f'missing field: {field}'
        assert body['upgrade_count'] == 0
        assert body['original_model_distribution'] == []

    def test_reason_codes_collapsed_by_reason_code(self):
        """Duplicate (reason_code, applied, classification) DB rows must be summed
        into a single entry per reason_code in the response."""
        registry = _make_registry()
        db = _make_db()
        # The DB groups by (reason_code, applied, classification), so the same
        # reason_code can appear multiple times with different applied/classification
        # values.  _get_routing() must collapse them before building the list.
        db.get_routing.return_value = {
            'reason_code_distribution': [
                {'reason_code': 'classifier', 'applied': 1, 'classification': 'standard', 'cnt': 10},
                {'reason_code': 'classifier', 'applied': 1, 'classification': 'deep', 'cnt': 5},
                {'reason_code': 'classifier', 'applied': 0, 'classification': 'trivial', 'cnt': 3},
                {'reason_code': 'size_forced_long_context', 'applied': 1, 'classification': None, 'cnt': 2},
            ],
            'tier_transitions': [],
        }
        _, body = admin.handle_get('/admin/routing', {}, registry, db)
        reason_codes = body['reason_codes']
        # Must have exactly two entries (one per distinct reason_code)
        assert len(reason_codes) == 2
        rc_map = {rc['reason_code']: rc['count'] for rc in reason_codes}
        # classifier rows (cnt 10+5+3=18) must be collapsed
        assert rc_map['classifier'] == 18
        assert rc_map['size_forced_long_context'] == 2
        # No duplicate entries for the same reason_code
        codes = [rc['reason_code'] for rc in reason_codes]
        assert len(codes) == len(set(codes))


# ---------------------------------------------------------------------------
# GET /admin/backends
# ---------------------------------------------------------------------------

class TestGetBackends:
    def test_returns_200_with_backends_list(self):
        """Response must be {backends: [...], active: str} — not a flat dict."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/backends', {}, registry, db)
        assert status == 200
        assert 'backends' in body
        assert isinstance(body['backends'], list)
        assert 'active' in body

    def test_backends_list_contains_backend_dicts(self):
        """Each entry in backends must have name, active, available."""
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/backends', {}, registry, db)
        names = {b['name'] for b in body['backends']}
        assert 'bedrock' in names
        assert 'anthropic' in names

    def test_active_field_is_active_backend_name(self):
        """active field must be the name of the currently active backend."""
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/backends', {}, registry, db)
        assert body['active'] == 'bedrock'

    def test_active_empty_string_when_none_active(self):
        """If no backend is active, active should be empty string."""
        registry = _make_registry()
        db = _make_db()
        registry.backend_status.return_value = {
            'bedrock': {'name': 'bedrock', 'active': False, 'available': True},
        }
        _, body = admin.handle_get('/admin/backends', {}, registry, db)
        assert body['active'] == ''

    def test_active_backend_available_inferred_true_when_null(self):
        """Active backend with available=None is returned as available=True."""
        registry = _make_registry()
        db = _make_db()
        registry.backend_status.return_value = {
            'bedrock': {'name': 'bedrock', 'active': True, 'available': None},
        }
        _, body = admin.handle_get('/admin/backends', {}, registry, db)
        bedrock = next(b for b in body['backends'] if b['name'] == 'bedrock')
        assert bedrock['available'] is True

    def test_inactive_backend_available_none_preserved(self):
        """Non-active backend with available=None stays None (unknown health)."""
        registry = _make_registry()
        db = _make_db()
        registry.backend_status.return_value = {
            'bedrock': {'name': 'bedrock', 'active': True, 'available': True},
            'anthropic': {'name': 'anthropic', 'active': False, 'available': None},
        }
        _, body = admin.handle_get('/admin/backends', {}, registry, db)
        anthropic = next(b for b in body['backends'] if b['name'] == 'anthropic')
        assert anthropic['available'] is None

    def test_calls_backend_status(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/backends', {}, registry, db)
        registry.backend_status.assert_called_once()


# ---------------------------------------------------------------------------
# GET /admin/config
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_returns_200_with_routing_enabled(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/config', {}, registry, db)
        assert status == 200
        assert 'routing_enabled' in body
        assert 'auto_backend_mode' in body

    def test_routing_enabled_reflects_config(self):
        registry = _make_registry()
        db = _make_db()
        registry.config.auto_model_routing = False
        status, body = admin.handle_get('/admin/config', {}, registry, db)
        assert body['routing_enabled'] is False

    def test_config_fields_present(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/config', {}, registry, db)
        expected_keys = {
            'routing_enabled',
            'auto_backend_mode',
            'auto_backend',
            'active_backend',
            'auto_model_routing_classifier_model',
            'auto_model_routing_long_context_threshold',
            'auto_model_routing_affirmation_inherit',
            'auto_model_routing_mode',
        }
        assert expected_keys.issubset(body.keys())

    def test_active_backend_from_registry_active_name(self):
        """active_backend must come from registry.active_name(), not cfg.auto_backend."""
        registry = _make_registry(active='anthropic')
        db = _make_db()
        _, body = admin.handle_get('/admin/config', {}, registry, db)
        assert body['active_backend'] == 'anthropic'

    def test_uses_public_config_property(self):
        """M6: _get_config must access registry.config (public), not registry._config."""
        registry = _make_registry()
        db = _make_db()
        # Verify the public .config property is accessed (not ._config)
        status, body = admin.handle_get('/admin/config', {}, registry, db)
        assert status == 200
        # If _config were accessed on a MagicMock, it would return a different mock object
        # and the config fields would be wrong (not matching our cfg setup).
        assert body['routing_enabled'] is True  # only true if .config (our cfg) was used


# ---------------------------------------------------------------------------
# GET /admin/config-changes
# ---------------------------------------------------------------------------

class TestGetConfigChanges:
    def test_returns_200_with_items(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/config-changes', {}, registry, db)
        assert status == 200
        assert 'items' in body

    def test_passes_limit(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/config-changes', {'limit': '25'}, registry, db)
        db.get_config_changes.assert_called_once_with(limit=25)

    def test_default_limit(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/config-changes', {}, registry, db)
        db.get_config_changes.assert_called_once_with(limit=100)

    def test_config_changes_limit_clamped_to_1000(self):
        """M4: config-changes limit > 1000 must be clamped."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/config-changes', {'limit': '9000'}, registry, db)
        db.get_config_changes.assert_called_once_with(limit=1000)


# ---------------------------------------------------------------------------
# GET unknown paths → 404
# ---------------------------------------------------------------------------

class TestGetUnknownPaths:
    @pytest.mark.parametrize('path', [
        '/admin/nonexistent',
        '/admin/sessions/s1/unknown',
        '/other/path',
        '/admin',
    ])
    def test_unknown_paths_return_404(self, path):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(path, {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'


# ---------------------------------------------------------------------------
# POST /admin/sessions/{session_id}/set-backend
# ---------------------------------------------------------------------------

class TestPostSetSessionBackend:
    def test_returns_200_ok(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': 'bedrock'},
            registry,
            db,
        )
        assert status == 200
        assert body['status'] == 'ok'
        assert body['session_id'] == 'sess1'
        assert body['backend'] == 'bedrock'

    def test_calls_registry_set_session_backend(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': 'bedrock'},
            registry,
            db,
        )
        registry.set_session_backend.assert_called_once_with('sess1', 'bedrock')

    def test_calls_db_set_session_backend(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': 'bedrock'},
            registry,
            db,
        )
        db.set_session_backend.assert_called_once_with('sess1', 'bedrock')

    def test_records_config_change(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': 'bedrock'},
            registry,
            db,
        )
        db.record_config_change.assert_called_once()

    def test_null_backend_calls_clear_session_backend(self):
        """M1: backend=null must call clear_session_backend, not set_session_backend(None)."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': None},
            registry,
            db,
        )
        assert status == 200
        assert body['backend'] is None
        registry.clear_session_backend.assert_called_once_with('sess1')
        registry.set_session_backend.assert_not_called()

    def test_null_backend_still_calls_db_set_backend(self):
        """db.set_session_backend must be called even when clearing (to NULL the column)."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': None},
            registry,
            db,
        )
        db.set_session_backend.assert_called_once_with('sess1', None)

    def test_invalid_backend_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': 'not-a-real-backend'},
            registry,
            db,
        )
        assert status == 400
        assert body['error'] == 'INVALID_BACKEND'

    def test_invalid_backend_does_not_call_registry(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-backend',
            {'backend': 'not-a-real-backend'},
            registry,
            db,
        )
        registry.set_session_backend.assert_not_called()


# ---------------------------------------------------------------------------
# POST /admin/sessions/{session_id}/set-global-tier
# ---------------------------------------------------------------------------

class TestPostSetGlobalTier:
    @pytest.mark.parametrize('tier', ['haiku', 'sonnet', 'opus'])
    def test_valid_tier_returns_200(self, tier):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': tier},
            registry,
            db,
        )
        assert status == 200
        assert body['status'] == 'ok'
        assert body['tier'] == tier

    def test_calls_registry_set_session_tier_global(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': 'sonnet'},
            registry,
            db,
        )
        registry.set_session_tier_global.assert_called_once_with('sess1', 'sonnet')

    def test_calls_db_set_session_tier(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': 'sonnet'},
            registry,
            db,
        )
        db.set_session_tier.assert_called_once_with('sess1', 'sonnet')

    def test_records_config_change(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': 'opus'},
            registry,
            db,
        )
        db.record_config_change.assert_called_once()

    def test_null_tier_is_valid(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': None},
            registry,
            db,
        )
        assert status == 200
        assert body['tier'] is None

    def test_invalid_tier_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': 'gpt-4o'},
            registry,
            db,
        )
        assert status == 400
        assert body['error'] == 'INVALID_TIER'

    def test_invalid_tier_does_not_call_registry(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post(
            '/admin/sessions/sess1/set-global-tier',
            {'tier': 'gpt-4o'},
            registry,
            db,
        )
        registry.set_session_tier_global.assert_not_called()


# ---------------------------------------------------------------------------
# POST /admin/global/routing
# ---------------------------------------------------------------------------

class TestPostGlobalRouting:
    def test_enable_routing_returns_200(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/routing', {'enabled': True}, registry, db
        )
        assert status == 200
        assert body['status'] == 'ok'
        assert body['routing_enabled'] is True

    def test_disable_routing_returns_200(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/routing', {'enabled': False}, registry, db
        )
        assert status == 200
        assert body['routing_enabled'] is False

    def test_calls_set_model_routing(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post('/admin/global/routing', {'enabled': True}, registry, db)
        registry.set_model_routing.assert_called_once_with(True)

    def test_records_config_change(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post('/admin/global/routing', {'enabled': False}, registry, db)
        db.record_config_change.assert_called_once()

    def test_non_bool_enabled_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/routing', {'enabled': 'yes'}, registry, db
        )
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'

    def test_missing_enabled_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post('/admin/global/routing', {}, registry, db)
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'


# ---------------------------------------------------------------------------
# POST /admin/global/backend
# ---------------------------------------------------------------------------

class TestPostGlobalBackend:
    @pytest.mark.parametrize('mode', ['auto', 'subscription'])
    def test_valid_backend_mode_returns_200(self, mode):
        """'auto' and 'subscription' call set_auto_backend_mode and return 200."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': mode}, registry, db
        )
        assert status == 200
        assert body['status'] == 'ok'
        assert body['prefer'] == mode

    @pytest.mark.parametrize('backend', ['bedrock', 'anthropic', 'codex', 'openrouter', 'local'])
    def test_specific_backend_name_calls_switch(self, backend):
        """Backend names route to registry.switch(), not set_auto_backend_mode."""
        registry = _make_registry(
            switch_result=SwitchResult(kind='changed', previous='bedrock', current=backend)
        )
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': backend}, registry, db
        )
        assert status == 200
        assert body['prefer'] == backend
        registry.switch.assert_called_once_with(backend, reason='admin_api')
        registry.set_auto_backend_mode.assert_not_called()

    def test_calls_set_auto_backend_mode_for_mode(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post('/admin/global/backend', {'prefer': 'auto'}, registry, db)
        registry.set_auto_backend_mode.assert_called_once_with('auto')

    def test_records_config_change(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post('/admin/global/backend', {'prefer': 'auto'}, registry, db)
        db.record_config_change.assert_called_once()

    def test_missing_prefer_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post('/admin/global/backend', {}, registry, db)
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'

    def test_unknown_prefer_returns_400(self):
        """Unrecognised values (not a mode and not a backend name) return 400."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'unknown_xyz'}, registry, db
        )
        assert status == 400
        assert body['error'] == 'INVALID_BACKEND_MODE'

    def test_empty_prefer_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': ''}, registry, db
        )
        assert status == 400

    def test_invalid_prefer_does_not_call_registry(self):
        """Unrecognised prefer must not reach set_auto_backend_mode or switch."""
        registry = _make_registry()
        db = _make_db()
        admin.handle_post('/admin/global/backend', {'prefer': 'invalid'}, registry, db)
        registry.set_auto_backend_mode.assert_not_called()
        registry.switch.assert_not_called()

    def test_switch_failure_returns_500(self):
        """If registry.switch returns kind='failed', the API returns 500."""
        registry = _make_registry(
            switch_result=SwitchResult(kind='failed', previous='bedrock', current='bedrock', error='timeout')
        )
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db
        )
        assert status == 500
        assert body['error'] == 'SWITCH_FAILED'

    @pytest.mark.parametrize('invalid_mode', ['', 'AUTO', 'SUBSCRIPTION', 'unknown_xyz'])
    def test_only_known_values_accepted(self, invalid_mode):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': invalid_mode}, registry, db
        )
        assert status == 400
        assert body['error'] == 'INVALID_BACKEND_MODE'


class TestPostGlobalBackendSelector:
    """Selector integration for POST /admin/global/backend (Bug A)."""

    def test_concrete_backend_pins_selector(self):
        registry = _make_registry(
            switch_result=SwitchResult(kind='changed', previous='bedrock', current='anthropic')
        )
        db = _make_db()
        selector = _make_selector('auto-selection: paused (manual `anthropic`)')
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db,
            selector=selector,
        )
        assert status == 200
        selector.pin.assert_called_once_with('anthropic')
        assert body['auto_selection'] == selector.status_line()

    def test_unchanged_switch_still_pins_selector(self):
        """kind='unchanged' mirrors the local-command path: pin anyway."""
        registry = _make_registry(
            switch_result=SwitchResult(kind='unchanged', previous='anthropic', current='anthropic')
        )
        db = _make_db()
        selector = _make_selector()
        status, _ = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db,
            selector=selector,
        )
        assert status == 200
        selector.pin.assert_called_once_with('anthropic')

    def test_prefer_auto_resumes_selector(self):
        registry = _make_registry()
        db = _make_db()
        selector = _make_selector()
        status, _ = admin.handle_post(
            '/admin/global/backend', {'prefer': 'auto'}, registry, db,
            selector=selector,
        )
        assert status == 200
        selector.resume.assert_called_once_with()
        registry.set_auto_backend_mode.assert_called_once_with('auto')
        selector.pin.assert_not_called()
        selector.restrict_subscription.assert_not_called()

    def test_prefer_subscription_restricts_selector(self):
        registry = _make_registry()
        db = _make_db()
        selector = _make_selector()
        status, _ = admin.handle_post(
            '/admin/global/backend', {'prefer': 'subscription'}, registry, db,
            selector=selector,
        )
        assert status == 200
        selector.restrict_subscription.assert_called_once_with()
        registry.set_auto_backend_mode.assert_called_once_with('subscription')
        selector.pin.assert_not_called()
        selector.resume.assert_not_called()

    def test_legacy_call_without_selector_kwarg(self):
        """Existing callers without selector still work; auto_selection is None."""
        registry = _make_registry(
            switch_result=SwitchResult(kind='changed', previous='bedrock', current='anthropic')
        )
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db
        )
        assert status == 200
        assert body['status'] == 'ok'
        assert body['auto_selection'] is None

    def test_failed_switch_does_not_pin(self):
        registry = _make_registry(
            switch_result=SwitchResult(kind='failed', previous='bedrock', current='bedrock', error='timeout')
        )
        db = _make_db()
        selector = _make_selector()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db,
            selector=selector,
        )
        assert status == 500
        assert body['error'] == 'SWITCH_FAILED'
        selector.pin.assert_not_called()

    def test_invalid_switch_does_not_pin(self):
        registry = _make_registry(
            switch_result=SwitchResult(kind='invalid', previous='bedrock', current='bedrock')
        )
        db = _make_db()
        selector = _make_selector()
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db,
            selector=selector,
        )
        assert status == 400
        assert body['error'] == 'INVALID_BACKEND'
        selector.pin.assert_not_called()

    def test_invalid_prefer_never_touches_selector(self):
        registry = _make_registry()
        db = _make_db()
        selector = _make_selector()
        status, _ = admin.handle_post(
            '/admin/global/backend', {'prefer': 'unknown_xyz'}, registry, db,
            selector=selector,
        )
        assert status == 400
        selector.pin.assert_not_called()
        selector.resume.assert_not_called()
        selector.restrict_subscription.assert_not_called()

    def test_success_response_schema(self):
        registry = _make_registry(
            active='anthropic',
            switch_result=SwitchResult(kind='changed', previous='bedrock', current='anthropic'),
        )
        db = _make_db()
        selector = _make_selector('auto-selection: paused (manual `anthropic`)')
        status, body = admin.handle_post(
            '/admin/global/backend', {'prefer': 'anthropic'}, registry, db,
            selector=selector,
        )
        assert status == 200
        assert body == {
            'status': 'ok',
            'prefer': 'anthropic',
            'active_backend': 'anthropic',
            'auto_selection': 'auto-selection: paused (manual `anthropic`)',
        }


# ---------------------------------------------------------------------------
# POST /admin/export
# ---------------------------------------------------------------------------

class TestPostExport:
    def test_returns_200_with_export_list(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/export', {'session_id': 'sess1'}, registry, db
        )
        assert status == 200
        assert 'export' in body

    def test_calls_get_trace(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_post('/admin/export', {'session_id': 'sess1'}, registry, db)
        db.get_trace.assert_called_once_with('sess1')

    def test_missing_session_id_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post('/admin/export', {}, registry, db)
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'

    def test_filename_in_response(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(
            '/admin/export', {'session_id': 'sess1'}, registry, db
        )
        assert '_filename' in body
        assert 'sess1' in body['_filename']


# ---------------------------------------------------------------------------
# POST unknown paths → 404
# ---------------------------------------------------------------------------

class TestPostUnknownPaths:
    @pytest.mark.parametrize('path', [
        '/admin/nonexistent',
        '/admin/sessions/s1',           # missing action segment
        '/admin/sessions/s1/bad-action',
        '/admin/global/unknown',
        '/other/path',
        '/admin',
    ])
    def test_unknown_paths_return_404(self, path):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_post(path, {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'


# ---------------------------------------------------------------------------
# Error format contract
# ---------------------------------------------------------------------------

class TestErrorFormat:
    def test_error_response_has_error_and_message(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/bogus', {}, registry, db)
        assert 'error' in body
        assert 'message' in body

    def test_500_on_unexpected_registry_exception(self):
        registry = _make_registry()
        db = _make_db()
        registry.backend_status.side_effect = RuntimeError('boom')
        status, body = admin.handle_get('/admin/backends', {}, registry, db)
        assert status == 500
        assert body['error'] == 'INTERNAL_ERROR'


# ---------------------------------------------------------------------------
# _int_param helper
# ---------------------------------------------------------------------------

class TestIntParam:
    def test_valid_int(self):
        assert admin._int_param({'k': '42'}, 'k', 0) == 42

    def test_missing_key_uses_default(self):
        assert admin._int_param({}, 'k', 99) == 99

    def test_invalid_value_uses_default(self):
        assert admin._int_param({'k': 'bad'}, 'k', 5) == 5

    def test_max_val_clamps_large_value(self):
        """M4: max_val must clamp values above the ceiling."""
        assert admin._int_param({'k': '9999'}, 'k', 50, max_val=1000) == 1000

    def test_max_val_allows_value_at_ceiling(self):
        assert admin._int_param({'k': '1000'}, 'k', 50, max_val=1000) == 1000

    def test_max_val_allows_value_below_ceiling(self):
        assert admin._int_param({'k': '50'}, 'k', 10, max_val=1000) == 50

    def test_max_val_none_disables_clamping(self):
        assert admin._int_param({'k': '99999'}, 'k', 10, max_val=None) == 99999


# ---------------------------------------------------------------------------
# GET /admin/status  (APXY-API-014)
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_200_with_required_keys(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/status', {}, registry, db)
        assert status == 200
        required = {
            'active_backend', 'routing_enabled', 'routing_mode', 'classifier_model',
            'long_context_threshold', 'affirmation_inherit', 'backends',
            'session_overrides', 'subscription_usage',
        }
        assert required.issubset(body.keys())

    def test_active_backend_from_registry(self):
        registry = _make_registry(active='anthropic')
        db = _make_db()
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert body['active_backend'] == 'anthropic'

    def test_routing_enabled_from_config(self):
        registry = _make_registry()
        db = _make_db()
        registry.config.auto_model_routing = False
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert body['routing_enabled'] is False

    def test_backends_is_list(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert isinstance(body['backends'], list)
        assert len(body['backends']) >= 1

    def test_session_overrides_from_db(self):
        registry = _make_registry()
        db = _make_db()
        db.get_session_overrides.return_value = [
            {'session_id': 'sess1', 'pinned_backend': 'bedrock', 'pinned_tier': None}
        ]
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert body['session_overrides'] == [
            {'session_id': 'sess1', 'pinned_backend': 'bedrock', 'pinned_tier': None}
        ]
        db.get_session_overrides.assert_called_once()

    def test_subscription_usage_from_registry(self):
        registry = _make_registry()
        db = _make_db()
        registry.usage_snapshot.return_value = {
            'anthropic': {'five_hour': {'pct': 42.0, 'reset_at': None, 'reset_in_secs': None}}
        }
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert 'anthropic' in body['subscription_usage']
        registry.usage_snapshot.assert_called_once()

    def test_empty_subscription_usage_when_none_cached(self):
        registry = _make_registry()
        db = _make_db()
        registry.usage_snapshot.return_value = {}
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert body['subscription_usage'] == {}

    def test_active_backend_available_inferred_true(self):
        """Active backend with available=None is reported as available=True."""
        registry = _make_registry()
        db = _make_db()
        registry.backend_status.return_value = {
            'bedrock': {'name': 'bedrock', 'active': True, 'available': None},
        }
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        bedrock = next(b for b in body['backends'] if b['name'] == 'bedrock')
        assert bedrock['available'] is True

    def test_calls_usage_snapshot(self):
        registry = _make_registry()
        db = _make_db()
        admin.handle_get('/admin/status', {}, registry, db)
        registry.usage_snapshot.assert_called_once()


class TestGetStatusSelector:
    """Selector and usage-refresh integration for GET /admin/status (Bug B)."""

    def test_auto_selection_from_selector(self):
        registry = _make_registry()
        db = _make_db()
        selector = _make_selector('auto-selection: subscription-only')
        _, body = admin.handle_get('/admin/status', {}, registry, db, selector=selector)
        assert body['auto_selection'] == 'auto-selection: subscription-only'

    def test_auto_selection_none_without_selector(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/status', {}, registry, db)
        assert body['auto_selection'] is None

    def test_fetch_usage_caches_called_before_snapshot(self, monkeypatch):
        registry = _make_registry()
        db = _make_db()
        calls = []
        monkeypatch.setattr(
            admin, '_fetch_usage_caches', lambda r: calls.append('fetch')
        )
        registry.usage_snapshot.side_effect = lambda: calls.append('snapshot') or {}
        status, _ = admin.handle_get('/admin/status', {}, registry, db)
        assert status == 200
        assert calls == ['fetch', 'snapshot']


class TestFetchUsageCaches:
    """_fetch_usage_caches concurrency helper."""

    @staticmethod
    def _registry_with(instances):
        registry = MagicMock()
        registry.cached_subscription_instances.return_value = instances
        registry.config = MagicMock()
        return registry

    def test_calls_get_usage_on_all_backends(self):
        called = []
        backends = {}
        for name in ('anthropic', 'codex', 'openrouter'):
            b = MagicMock(spec=['get_usage'])
            b.get_usage.side_effect = (
                lambda cfg, _n=name: called.append(_n)
            )
            backends[name] = b
        registry = self._registry_with(backends)
        admin._fetch_usage_caches(registry)
        assert sorted(called) == ['anthropic', 'codex', 'openrouter']
        for b in backends.values():
            b.get_usage.assert_called_once_with(registry.config)

    def test_one_raising_does_not_block_others(self):
        called = []
        ok1 = MagicMock(spec=['get_usage'])
        ok1.get_usage.side_effect = lambda cfg: called.append('ok1')
        bad = MagicMock(spec=['get_usage'])
        bad.get_usage.side_effect = RuntimeError('usage endpoint down')
        ok2 = MagicMock(spec=['get_usage'])
        ok2.get_usage.side_effect = lambda cfg: called.append('ok2')
        registry = self._registry_with({'a': ok1, 'b': bad, 'c': ok2})
        admin._fetch_usage_caches(registry)  # must not raise
        assert sorted(called) == ['ok1', 'ok2']
        bad.get_usage.assert_called_once()

    def test_empty_instances_is_noop(self):
        registry = self._registry_with({})
        admin._fetch_usage_caches(registry)  # must not raise

    def test_backend_without_get_usage_is_skipped(self):
        no_usage = MagicMock(spec=[])  # no get_usage attribute
        has_usage = MagicMock(spec=['get_usage'])
        registry = self._registry_with({'x': no_usage, 'anthropic': has_usage})
        admin._fetch_usage_caches(registry)
        has_usage.get_usage.assert_called_once_with(registry.config)


# ---------------------------------------------------------------------------
# GET /admin/stats  (APXY-API-015)
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_default_period_is_week(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/stats', {}, registry, db)
        assert status == 200
        assert body['period'] == 'week'
        db.get_stats.assert_called_once_with('week', None)

    def test_period_day(self):
        registry = _make_registry()
        db = _make_db()
        db.get_stats.return_value = {'buckets': [], 'total': {}}
        admin.handle_get('/admin/stats', {'period': 'day'}, registry, db)
        db.get_stats.assert_called_once_with('day', None)

    def test_period_month(self):
        registry = _make_registry()
        db = _make_db()
        db.get_stats.return_value = {'buckets': [], 'total': {}}
        admin.handle_get('/admin/stats', {'period': 'month'}, registry, db)
        db.get_stats.assert_called_once_with('month', None)

    def test_period_quarter(self):
        registry = _make_registry()
        db = _make_db()
        db.get_stats.return_value = {'buckets': [], 'total': {}}
        admin.handle_get('/admin/stats', {'period': 'quarter'}, registry, db)
        db.get_stats.assert_called_once_with('quarter', None)

    def test_invalid_period_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/stats', {'period': 'invalid'}, registry, db)
        assert status == 400
        assert body['error'] == 'INVALID_PERIOD'
        db.get_stats.assert_not_called()

    def test_backend_filter_passed_to_db(self):
        registry = _make_registry()
        db = _make_db()
        db.get_stats.return_value = {'buckets': [], 'total': {}}
        admin.handle_get('/admin/stats', {'period': 'day', 'backend': 'anthropic'}, registry, db)
        db.get_stats.assert_called_once_with('day', 'anthropic')

    def test_subscription_backend_filter(self):
        registry = _make_registry()
        db = _make_db()
        db.get_stats.return_value = {'buckets': [], 'total': {}}
        admin.handle_get('/admin/stats', {'backend': 'subscription'}, registry, db)
        db.get_stats.assert_called_once_with('week', 'subscription')

    def test_response_shape(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/stats', {}, registry, db)
        assert status == 200
        assert 'period' in body
        assert 'backend_filter' in body
        assert 'buckets' in body
        assert 'total' in body

    def test_backend_filter_none_when_absent(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/stats', {}, registry, db)
        assert body['backend_filter'] is None

    def test_buckets_from_db(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/stats', {}, registry, db)
        assert len(body['buckets']) == 1
        assert body['buckets'][0]['label'] == '2026-07-01'


# ---------------------------------------------------------------------------
# GET /admin/requests/{id}  (APXY-API-016)
# ---------------------------------------------------------------------------

class TestGetRequestDetail:
    def test_returns_200_with_row_for_known_id(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/requests/123', {}, registry, db)
        assert status == 200
        assert body['id'] == 123
        db.get_request.assert_called_once_with(123)

    def test_returns_404_when_not_found(self):
        registry = _make_registry()
        db = _make_db()
        db.get_request.return_value = None
        status, body = admin.handle_get('/admin/requests/999', {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'

    def test_non_integer_id_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/requests/abc', {}, registry, db)
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'
        db.get_request.assert_not_called()

    def test_zero_id_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/requests/0', {}, registry, db)
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'

    def test_negative_id_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/requests/-5', {}, registry, db)
        assert status == 400
        assert body['error'] == 'BAD_REQUEST'

    def test_response_includes_prompt_join_fields(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get('/admin/requests/123', {}, registry, db)
        assert 'system_prompt_content' in body

    def test_requests_without_id_returns_404(self):
        """GET /admin/requests (no id) must return 404 — not routed."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/requests', {}, registry, db)
        assert status == 404


# ---------------------------------------------------------------------------
# GET /admin/prompts/{sha256}  (APXY-API-017)
# ---------------------------------------------------------------------------

_VALID_SHA = 'a' * 64


class TestGetPrompt:
    def test_returns_200_with_prompt_row(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get(f'/admin/prompts/{_VALID_SHA}', {}, registry, db)
        assert status == 200
        assert body['content_hash'] == _VALID_SHA
        db.get_prompt.assert_called_once_with(_VALID_SHA)

    def test_returns_404_when_not_found(self):
        registry = _make_registry()
        db = _make_db()
        db.get_prompt.return_value = None
        valid_sha = 'b' * 64
        status, body = admin.handle_get(f'/admin/prompts/{valid_sha}', {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'

    def test_invalid_hash_not_hex_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/prompts/not-hex-at-all', {}, registry, db)
        assert status == 400
        assert body['error'] == 'INVALID_HASH'
        db.get_prompt.assert_not_called()

    def test_too_short_hash_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/prompts/' + 'a' * 32, {}, registry, db)
        assert status == 400
        assert body['error'] == 'INVALID_HASH'

    def test_too_long_hash_returns_400(self):
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/prompts/' + 'a' * 65, {}, registry, db)
        assert status == 400
        assert body['error'] == 'INVALID_HASH'

    def test_uppercase_hex_returns_400(self):
        """Hash must be lowercase — uppercase is invalid."""
        registry = _make_registry()
        db = _make_db()
        upper_sha = 'A' * 64
        status, body = admin.handle_get(f'/admin/prompts/{upper_sha}', {}, registry, db)
        assert status == 400
        assert body['error'] == 'INVALID_HASH'

    def test_response_includes_expected_fields(self):
        registry = _make_registry()
        db = _make_db()
        _, body = admin.handle_get(f'/admin/prompts/{_VALID_SHA}', {}, registry, db)
        for field in ('content_hash', 'content_type', 'content', 'char_count', 'first_seen_at'):
            assert field in body

    def test_prompts_without_hash_returns_404(self):
        """GET /admin/prompts (no hash) must return 404."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/prompts', {}, registry, db)
        assert status == 404


# ---------------------------------------------------------------------------
# GET /admin/sessions/{session_id}/summary
# ---------------------------------------------------------------------------

class TestGetSessionSummary:
    def _make_summary(self):
        return {
            'session_id': 'sess1',
            'request_count': 10,
            'total_input_tokens': 1000,
            'total_output_tokens': 500,
            'total_cache_creation': 200,
            'total_cache_read': 800,
            'estimated_cost_usd': 0.05,
        }

    def test_returns_200_for_known_session(self):
        registry = _make_registry()
        db = _make_db()
        db.get_session_summary.return_value = self._make_summary()
        status, body = admin.handle_get('/admin/sessions/sess1/summary', {}, registry, db)
        assert status == 200
        assert body['session_id'] == 'sess1'
        assert body['request_count'] == 10

    def test_calls_get_session_summary_with_id(self):
        registry = _make_registry()
        db = _make_db()
        db.get_session_summary.return_value = self._make_summary()
        admin.handle_get('/admin/sessions/sess1/summary', {}, registry, db)
        db.get_session_summary.assert_called_once_with('sess1')

    def test_returns_404_when_not_found(self):
        registry = _make_registry()
        db = _make_db()
        db.get_session_summary.return_value = None
        status, body = admin.handle_get('/admin/sessions/ghost/summary', {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'

    def test_percent_encoded_session_id_decoded_for_summary(self):
        raw_id = '{"device_id":"abc","account_uuid":"","session_id":"362954d5-af9e-49"}'
        from urllib.parse import quote
        encoded_path = f'/admin/sessions/{quote(raw_id, safe="")}/summary'
        registry = _make_registry()
        db = _make_db()
        db.get_session_summary.return_value = {'session_id': raw_id, 'request_count': 5}
        status, body = admin.handle_get(encoded_path, {}, registry, db)
        assert status == 200
        db.get_session_summary.assert_called_once_with(raw_id)

    def test_summary_sub_path_not_matched_by_trace(self):
        """The /summary path must not accidentally fall through to /trace handler."""
        registry = _make_registry()
        db = _make_db()
        db.get_session_summary.return_value = self._make_summary()
        admin.handle_get('/admin/sessions/sess1/summary', {}, registry, db)
        db.get_trace.assert_not_called()

    def test_unknown_sessions_subpath_returns_404(self):
        """Paths beyond /trace and /summary must return 404."""
        registry = _make_registry()
        db = _make_db()
        status, body = admin.handle_get('/admin/sessions/sess1/unknown-sub', {}, registry, db)
        assert status == 404
        assert body['error'] == 'NOT_FOUND'
