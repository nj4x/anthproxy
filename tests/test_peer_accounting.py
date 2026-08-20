"""Per-hop independent accounting (ADR-0025, SRS-Chaining-004).

Every instance records the traffic it dispatched, including traffic dispatched
to a peer.  No suppression signal crosses the hop in either direction, and no
instance is authoritative for chain-wide cost: both hops record the same
request under the same session key, so the two figures must never be summed.

The failure half is a general fix rather than a peer one — before ADR-0025 a
dispatch that failed *before* the stream was primed left a stats line and no
request row at all, so the most diagnostically interesting traffic in a chain
was the one kind that left no trace.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from anthproxy.config import Config
from anthproxy.db import SessionDB
from anthproxy.handlers import AnthropicRequestError, ProxyRequestHandler
from anthproxy.peer.backend import PeerBackend

_SESSION_KEY = json.dumps({'session_id': 'chain-sess-1', 'account_uuid': 'acct-1'})

_PEER_USAGE = {
    'input_tokens': 1200,
    'output_tokens': 340,
    'cache_creation_input_tokens': 500,
    'cache_read_input_tokens': 900,
}

_PEER_REPLY = json.dumps({
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-6',
    'content': [{'type': 'text', 'text': 'ok'}],
    'stop_reason': 'end_turn',
    'usage': _PEER_USAGE,
}).encode()


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    session_db = SessionDB(path)
    try:
        yield session_db
    finally:
        session_db.close()
        os.unlink(path)


def _config(**kwargs) -> Config:
    defaults = dict(backend='peer', peer_base_url='http://peer.internal:9099')
    defaults.update(kwargs)
    return Config(**defaults)


def _conn(status=200, body=_PEER_REPLY):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.getheader.side_effect = lambda name, default='': default
    conn = MagicMock()
    conn.getresponse.return_value = resp
    return conn


def _snapshot(config, backend=None, name='peer'):
    snapshot = MagicMock()
    snapshot.name = name
    snapshot.backend = backend if backend is not None else PeerBackend()
    snapshot.config = config
    snapshot.session_pinned = False
    snapshot.session_subscription = False
    snapshot.credentials = None
    return snapshot


def _registry(snapshot):
    registry = MagicMock()
    registry.snapshot.return_value = snapshot
    registry.snapshot_for_request.return_value = snapshot
    registry.session_context.return_value = (0, 1.0)
    registry.session_routed_tier.return_value = None
    registry.active_name.return_value = snapshot.name
    registry.session_model_routing.return_value = None
    registry.session_backend.return_value = None
    return registry


def _handler(registry, config, session_db, *, stats=None, headers=None):
    handler = object.__new__(ProxyRequestHandler)
    handler.registry = registry
    handler.config = config
    handler.session_db = session_db
    handler.stats_collector = stats
    handler.selector = None
    handler.path = '/v1/messages'
    handler.headers = {'x-api-key': 'client-key', 'Content-Type': 'application/json',
                       **(headers or {})}
    handler._send_json = MagicMock()
    handler._send_sse = MagicMock()
    handler._read_body = MagicMock(return_value=b'{}')
    return handler


def _payload(model='claude-sonnet-4-6', content='refactor the parser'):
    return {
        'model': model,
        'max_tokens': 1024,
        'system': 'You are a helpful assistant.',
        'messages': [{'role': 'user', 'content': content}],
        'metadata': {'user_id': _SESSION_KEY},
    }


def _post(handler, payload):
    handler._parse_json = MagicMock(return_value=payload)
    handler.do_POST()


def _rows(session_db):
    return session_db._read_conn().execute(
        'SELECT * FROM requests ORDER BY id'
    ).fetchall()


def _only_row(session_db):
    rows = _rows(session_db)
    assert len(rows) == 1, f'expected exactly one request row, got {len(rows)}'
    return rows[0]


# ---------------------------------------------------------------------------
# ADR-0025 §1 — the dispatching hop records what it dispatched
# ---------------------------------------------------------------------------

class TestDispatchingHopRecords:
    def test_successful_peer_dispatch_is_recorded_attributed_to_peer(self, db):
        config = _config()
        stats = MagicMock()
        handler = _handler(_registry(_snapshot(config)), config, db, stats=stats)

        with patch('anthproxy.peer.backend._make_connection', return_value=_conn()):
            _post(handler, _payload())

        row = _only_row(db)
        assert row['backend'] == 'peer'
        assert row['status'] == 'success'
        assert row['session_id'] == _SESSION_KEY
        assert row['cost_estimate'] > 0
        assert stats.record.call_args.args[0] == 'peer'
        assert stats.record.call_args.kwargs['status'] == 'success'

    def test_peer_response_usage_already_arrives_in_anthropic_convention(self, db):
        """Verification, not a build: the peer speaks native Anthropic, so its
        usage needs no token-semantics translation the way Codex's does.  The
        recorded columns are the response's own numbers, unscaled and disjoint.
        """
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, db)

        with patch('anthproxy.peer.backend._make_connection', return_value=_conn()):
            _post(handler, _payload())

        row = _only_row(db)
        assert row['input_tokens'] == _PEER_USAGE['input_tokens']
        assert row['output_tokens'] == _PEER_USAGE['output_tokens']
        assert row['cache_creation_tokens'] == _PEER_USAGE['cache_creation_input_tokens']
        assert row['cache_read_tokens'] == _PEER_USAGE['cache_read_input_tokens']

    def test_session_key_is_transmitted_unchanged_and_recorded_at_both_hops(self, db):
        """Identical session keys across hops are the diagnosis asset that makes
        naive summing so easy: the same request is inspectable at both ends.
        """
        outer_config = _config()
        conn = _conn()
        outer = _handler(_registry(_snapshot(outer_config)), outer_config, db)

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(outer, _payload())

        forwarded = json.loads(conn.request.call_args.kwargs['body'])
        assert forwarded['metadata']['user_id'] == _SESSION_KEY

        # The inner hop receives that payload verbatim and records its own row.
        fd, inner_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        inner_db = SessionDB(inner_path)
        try:
            inner_config = Config(backend='anthropic')
            inner_backend = MagicMock()
            inner_backend.parse_credentials.return_value = {}
            inner_backend.send_message.return_value = json.loads(_PEER_REPLY)
            inner_snapshot = _snapshot(inner_config, inner_backend, name='anthropic')
            inner = _handler(_registry(inner_snapshot), inner_config, inner_db)
            _post(inner, forwarded)

            assert _only_row(inner_db)['session_id'] == _only_row(db)['session_id']
            assert _only_row(inner_db)['backend'] == 'anthropic'
        finally:
            inner_db.close()
            os.unlink(inner_path)

    def test_no_suppression_signal_is_sent_and_none_is_honoured(self, db):
        """Designating one hop authoritative would need a directive to cross the
        hop, which ADR-0024 forbids — so neither hop can silence the other.
        """
        config = _config(peer_api_key='peer-secret')
        handler = _handler(
            _registry(_snapshot(config)), config, db,
            headers={'X-Anthproxy-Override': 'prefer:peer'},
        )
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(handler, _payload())

        headers = conn.request.call_args.kwargs['headers']
        assert set(headers) == {'Content-Type', 'Accept', 'X-Anthproxy-Peer-Key'}
        # Inbound control state did not stop this hop recording its own dispatch.
        assert _only_row(db)['backend'] == 'peer'


# ---------------------------------------------------------------------------
# ADR-0025 §4 — a failed dispatch is recorded too, for every backend
# ---------------------------------------------------------------------------

def _assert_error_row(row, backend):
    assert row['backend'] == backend
    assert row['status'] == 'error'
    # Absent, not zero: a zero is summable and reads as a request that genuinely
    # cost nothing; a null says this hop never learned.
    for column in ('input_tokens', 'output_tokens',
                   'cache_creation_tokens', 'cache_read_tokens'):
        assert row[column] is None, f'{column} must be NULL, got {row[column]!r}'
    assert row['cost_estimate'] is None


class TestFailedDispatchIsRecorded:
    def test_peer_error_status_produces_a_row(self, db):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, db)

        with patch('anthproxy.peer.backend._make_connection',
                   return_value=_conn(status=400, body=b'{"error":{"message":"bad"}}')):
            _post(handler, _payload())

        _assert_error_row(_only_row(db), 'peer')

    def test_peer_connection_refused_produces_a_row(self, db):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, db)

        with patch('anthproxy.peer.backend._make_connection',
                   side_effect=ConnectionRefusedError('refused')), \
                patch('anthproxy.peer.backend.time.sleep'):
            _post(handler, _payload())

        _assert_error_row(_only_row(db), 'peer')

    def test_peer_timeout_produces_a_row(self, db):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, db)
        conn = MagicMock()
        conn.getresponse.side_effect = TimeoutError('timed out')

        with patch('anthproxy.peer.backend._make_connection', return_value=conn), \
                patch('anthproxy.peer.backend.time.sleep'):
            _post(handler, _payload())

        _assert_error_row(_only_row(db), 'peer')

    def test_provider_failure_produces_the_same_row(self, db):
        """Regression proving the fix is general.  Gating the new row on
        ``peer`` would leave provider failures invisible while making peer
        failures visible — a worse inconsistency than the one being fixed.
        """
        config = Config(backend='anthropic')
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = AnthropicRequestError(
            'upstream exploded', error_type='api_error', status_code=500,
        )
        handler = _handler(
            _registry(_snapshot(config, backend, name='anthropic')), config, db)

        _post(handler, _payload())

        _assert_error_row(_only_row(db), 'anthropic')

    def test_non_anthropic_exception_produces_a_row(self, db):
        config = Config(backend='anthropic')
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = RuntimeError('boom')
        handler = _handler(
            _registry(_snapshot(config, backend, name='anthropic')), config, db)

        _post(handler, _payload())

        _assert_error_row(_only_row(db), 'anthropic')

    def test_error_row_carries_the_status_code_and_error_type(self, db):
        config = Config(backend='anthropic')
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = AnthropicRequestError(
            'upstream exploded', error_type='api_error', status_code=503,
        )
        handler = _handler(
            _registry(_snapshot(config, backend, name='anthropic')), config, db)

        _post(handler, _payload())

        assert '503' in _only_row(db)['error']
        assert 'api_error' in _only_row(db)['error']

    def test_streaming_failure_before_the_first_chunk_is_not_recorded_as_success(self, db):
        """The client-facing error frame is written out of band by ``_send_sse``,
        so nothing in the stream itself marks the request as failed — without
        the upstream-error capture this lands as a zero-cost success.
        """
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, db)
        handler.wfile = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        del handler._send_sse  # exercise the real SSE path
        payload = _payload()
        payload['stream'] = True

        with patch('anthproxy.peer.backend._make_connection',
                   return_value=_conn(status=400, body=b'{"error":{"message":"bad"}}')):
            _post(handler, payload)

        _assert_error_row(_only_row(db), 'peer')

    def test_unretried_429_is_recorded_as_rate_limited(self, db):
        """A 429 that finds nowhere to fail over to keeps the status the retry
        path's pre-record uses, so the two are not conflated in the UI.
        """
        config = Config(backend='anthropic')
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = AnthropicRequestError(
            'slow down', error_type='rate_limit_error', status_code=429,
        )
        snapshot = _snapshot(config, backend, name='anthropic')
        snapshot.session_pinned = True  # no auto-switch, so no retry
        handler = _handler(_registry(snapshot), config, db)

        _post(handler, _payload())

        row = _only_row(db)
        assert row['status'] == 'rate_limited'
        assert row['cost_estimate'] is None

    def test_pre_dispatch_client_error_writes_no_row(self, db):
        """No backend was involved, so there is nothing to attribute."""
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, db)
        handler._parse_json = MagicMock(side_effect=AnthropicRequestError(
            'Malformed JSON body', error_type='invalid_request_error', status_code=400,
        ))

        handler.do_POST()

        assert _rows(db) == []
