"""The peer hop is a control boundary (ADR-0024, SRS-Chaining-003).

Content — messages, system prompt, the requested model — crosses a peer
dispatch unchanged.  Control directives (``X-Anthproxy-Override`` and the
``proxy-*`` local commands) name things in *this* instance's configuration and
are consumed at the hop that receives them.  Peer failures propagate as
Anthropic error envelopes with no local fallback.

These are cross-cutting boundary tests: the behaviours are built in the peer
mapper, the routing suppression, and the selector, and are pinned here as one
suite so the boundary cannot erode a piece at a time.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from anthproxy.config import Config
from anthproxy.handlers import ProxyRequestHandler
from anthproxy.peer.backend import PeerBackend
from anthproxy.server import BackendRegistry

_ALL_DIRECTIVES = 'prefer:peer; route:tag; task:refactor; no-classifier'

_PEER_REPLY = json.dumps({
    'type': 'message',
    'role': 'assistant',
    'model': 'sonnet',
    'content': [{'type': 'text', 'text': 'ok'}],
    'stop_reason': 'end_turn',
    'usage': {'input_tokens': 10, 'output_tokens': 5},
}).encode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _sent(conn, index=0):
    """Return ``(headers, body)`` of the *index*-th request on a mocked conn."""
    call = conn.request.call_args_list[index]
    return call.kwargs['headers'], json.loads(call.kwargs['body'])


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


def _handler(registry, config, override=None):
    handler = object.__new__(ProxyRequestHandler)
    handler.registry = registry
    handler.config = config
    handler.path = '/v1/messages'
    headers = {'x-api-key': 'client-key', 'Content-Type': 'application/json'}
    if override is not None:
        headers['X-Anthproxy-Override'] = override
    handler.headers = headers
    handler._send_json = MagicMock()
    handler._send_sse = MagicMock()
    handler._read_body = MagicMock(return_value=b'{}')
    return handler


def _payload(model='sonnet', content='refactor the parser', **extra):
    payload = {
        'model': model,
        'max_tokens': 1024,
        'system': 'You are a helpful assistant.',
        'messages': [{'role': 'user', 'content': content}],
        'metadata': {'user_id': 'boundary-sess'},
    }
    payload.update(extra)
    return payload


def _post(handler, payload):
    handler._parse_json = MagicMock(return_value=payload)
    handler.do_POST()


# ---------------------------------------------------------------------------
# ADR-0024 §1 — directives are consumed at the hop that receives them
# ---------------------------------------------------------------------------

class TestDirectivesDoNotCross:
    def test_outbound_headers_carry_no_part_of_the_override_header(self):
        # peer_api_key set so the assertion distinguishes "no anthproxy header
        # crosses" from "the peer credential is the only one that does".
        config = _config(peer_api_key='peer-secret')
        handler = _handler(_registry(_snapshot(config)), config, override=_ALL_DIRECTIVES)
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(handler, _payload())

        headers, _body = _sent(conn)
        assert set(headers) == {'Content-Type', 'Accept', 'X-Anthproxy-Peer-Key'}
        assert headers['X-Anthproxy-Peer-Key'] == 'peer-secret'

    @pytest.mark.parametrize('directive,attr,value', [
        ('prefer:peer', '_prefer_backend', 'peer'),
        ('route:tag', '_override_mode', 'tag'),
        ('task:refactor', '_task_tag', 'refactor'),
        ('no-classifier', '_no_classifier', True),
    ])
    def test_each_directive_is_consumed_here_and_absent_downstream(
            self, directive, attr, value):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, override=directive)
        payload = _payload()
        expected = copy.deepcopy(payload)
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(handler, payload)

        # Consumed: the outer hop parsed the directive and acted on it here.
        assert getattr(handler, attr) == value
        # Not forwarded: the outbound request is exactly what it would have been
        # without the directive — no extra header, no extra body key.
        headers, body = _sent(conn)
        assert set(headers) == {'Content-Type', 'Accept'}
        assert body == expected

    def test_local_command_is_intercepted_and_never_dispatched(self):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config)
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn) as make_conn:
            _post(handler, _payload(content='proxy-help'))

        make_conn.assert_not_called()
        conn.request.assert_not_called()
        status, envelope = handler._send_json.call_args.args
        assert status == 200
        # The reply is this instance's own, not a peer round-trip.
        assert 'anthproxy' in envelope['content'][0]['text'].lower()

    def test_prefer_peer_is_honoured_at_the_outer_hop(self, monkeypatch):
        """``prefer:peer`` is the escape hatch for addressing the peer: a
        directive naming the peer is meaningful *here* and is honoured here.

        Uses a real registry so the ``ROTATABLE_BACKENDS`` preference gate is
        exercised — against a ``SUBSCRIPTION_BACKENDS`` gate this fails, since
        ``subscription`` is the default mode and the peer is not a subscription.
        """
        config = _config(backend='anthropic')
        assert config.auto_backend_mode == 'subscription'
        peer = PeerBackend()
        monkeypatch.setattr(
            'anthproxy.server.build_backend',
            lambda name, cfg: peer if name == 'peer' else MagicMock(),
        )
        active = MagicMock()
        registry = BackendRegistry(config, active)
        handler = _handler(registry, config, override='prefer:peer')
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(handler, _payload())

        assert conn.request.called
        active.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# ADR-0024 §2 — content crosses unchanged, and nothing is added
# ---------------------------------------------------------------------------

class TestContentCrossesUnchanged:
    def test_outbound_body_is_inbound_minus_the_two_internal_keys(self):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config, override=_ALL_DIRECTIVES)
        payload = _payload(
            temperature=0.2,
            tools=[{'name': 'Read', 'input_schema': {'type': 'object'}}],
        )
        payload['_anthproxy_internal_classifier'] = True
        payload['_anthropic_beta'] = ['context-1m-2025-08-07']
        expected = copy.deepcopy(
            {k: v for k, v in payload.items()
             if k not in ('_anthropic_beta', '_anthproxy_internal_classifier')})
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(handler, payload)

        _headers, body = _sent(conn)
        assert body == expected

    def test_client_beta_survives_the_hop_as_an_outbound_header(self):
        config = _config()
        handler = _handler(_registry(_snapshot(config)), config)
        handler.headers['anthropic-beta'] = 'context-1m-2025-08-07, fine-grained-tool-streaming-2025-05-14'
        conn = _conn()

        with patch('anthproxy.peer.backend._make_connection', return_value=conn):
            _post(handler, _payload())

        headers, body = _sent(conn)
        assert headers['anthropic-beta'] == (
            'context-1m-2025-08-07,fine-grained-tool-streaming-2025-05-14')
        assert '_anthropic_beta' not in body


# ---------------------------------------------------------------------------
# ADR-0024 §3 — peer failures propagate; no fallback, no substitution
# ---------------------------------------------------------------------------

class TestPeerFailuresPropagate:
    def _run(self, conn, model='claude-sonnet-4-5-20250929'):
        config = _config()
        registry = _registry(_snapshot(config))
        handler = _handler(registry, config)
        with patch('anthproxy.peer.backend._make_connection', return_value=conn), \
                patch('anthproxy.peer.backend.time.sleep'):
            _post(handler, _payload(model=model))
        return handler

    def test_model_rejection_surfaces_naming_the_clients_own_model(self):
        model = 'claude-opus-9-does-not-exist'
        body = json.dumps({'type': 'error', 'error': {
            'type': 'invalid_request_error',
            'message': f'model: {model} is not supported',
        }}).encode()
        conn = _conn(status=400, body=body)

        handler = self._run(conn, model=model)

        status, envelope = handler._send_json.call_args.args
        assert status == 400
        assert envelope['type'] == 'error'
        assert envelope['error']['type'] == 'invalid_request_error'
        assert model in envelope['error']['message']
        # One attempt, and it named the client's model — no local retry against
        # a substitute.
        assert conn.request.call_count == 1
        _headers, sent = _sent(conn)
        assert sent['model'] == model

    def test_peer_5xx_surfaces_as_an_error_envelope(self):
        conn = _conn(status=503, body=b'upstream unavailable')

        handler = self._run(conn)

        status, envelope = handler._send_json.call_args.args
        assert status >= 500
        assert envelope['type'] == 'error'

    def test_connection_failure_surfaces_as_an_error_envelope(self):
        conn = _conn()
        conn.request.side_effect = ConnectionRefusedError('connection refused')

        handler = self._run(conn)

        status, envelope = handler._send_json.call_args.args
        assert status == 502
        assert envelope['type'] == 'error'
        assert 'connection' in envelope['error']['message'].lower()

    def _with_live_selector(self, status, model):
        """Run one peer failure with the auto-selector armed to hand off.

        Without a selector the 429 branch is unreachable, so a no-fallback
        assertion would pass on every status for the wrong reason.
        """
        config = _config()
        registry = _registry(_snapshot(config))
        fallback_backend = MagicMock()
        fallback_backend.parse_credentials.return_value = {}
        fallback_backend.send_message.return_value = json.loads(_PEER_REPLY)
        registry.snapshot.return_value = _snapshot(
            config, backend=fallback_backend, name='anthropic')
        handler = _handler(registry, config)
        handler.selector = MagicMock()
        handler.selector.is_paused.return_value = False
        handler.selector.on_rate_limited.return_value = 'anthropic'
        conn = _conn(status=status, body=b'{}')

        with patch('anthproxy.peer.backend._make_connection', return_value=conn), \
                patch('anthproxy.peer.backend.time.sleep'):
            _post(handler, _payload(model=model))
        return handler, conn, fallback_backend

    @pytest.mark.parametrize('status', [400, 404, 500, 503])
    def test_no_local_fallback_or_peer_default_substitution(self, status):
        """A non-429 peer failure reaches the client; no other backend runs."""
        model = 'sonnet-the-client-asked-for'

        handler, conn, fallback_backend = self._with_live_selector(status, model)

        fallback_backend.send_message.assert_not_called()
        handler.selector.on_rate_limited.assert_not_called()
        for i in range(conn.request.call_count):
            assert _sent(conn, i)[1]['model'] == model
        _status, envelope = handler._send_json.call_args.args
        assert envelope['type'] == 'error'

    def test_429_hands_off_to_another_backend_at_the_clients_own_model(self):
        """The one carve-out: a 429 is the selector's reactive path, not a
        fallback — it changes backend, never the model the client asked for."""
        model = 'sonnet-the-client-asked-for'

        handler, conn, fallback_backend = self._with_live_selector(429, model)

        # Surfaced to the selector on the first attempt, never retried in-backend.
        assert conn.request.call_count == 1
        assert _sent(conn)[1]['model'] == model
        handler.selector.on_rate_limited.assert_called_once()
        fallback_backend.send_message.assert_called_once()
