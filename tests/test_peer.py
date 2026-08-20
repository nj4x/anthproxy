"""Unit tests for the peer backend (anthproxy-to-anthproxy dispatch)."""

import http.client
import json
import socket
import threading
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from anthproxy.config import Config, parse_args
from anthproxy.mapper import AnthropicRequestError
from anthproxy.peer.backend import (
    PeerBackend,
    _make_connection,
    _request_headers,
)
from anthproxy.peer.mapper import _beta_header, _build_body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> Config:
    defaults = dict(
        host='127.0.0.1', port=8082, backend='peer', log_level='INFO',
        log_file='', anthproxy_home='', codex_home='', bedrock_home='',
        anthropic_home='', stats_dir='',
        peer_base_url='http://127.0.0.1:9099',
        auto_backend=False,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _fake_response(status, body, headers=None):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body if isinstance(body, bytes) else body.encode()
    resp.getheader.side_effect = lambda name, default='': (headers or {}).get(
        name.lower(), default)
    return resp


def _mock_conn(response):
    conn = MagicMock()
    conn.getresponse.return_value = response
    return conn


def _sent_headers(conn) -> dict:
    """Return the headers of the last request issued on a mocked connection."""
    return conn.request.call_args.kwargs['headers']


def _sent_body(conn) -> dict:
    return json.loads(conn.request.call_args.kwargs['body'])


# ---------------------------------------------------------------------------
# mapper — model pass-through
# ---------------------------------------------------------------------------

class TestPeerBuildBody:
    def test_alias_model_transmitted_verbatim(self):
        body = json.loads(_build_body({'model': 'sonnet', 'messages': []}))
        assert body['model'] == 'sonnet'

    def test_full_model_id_transmitted_verbatim(self):
        body = json.loads(_build_body({'model': 'claude-sonnet-4-6', 'messages': []}))
        assert body['model'] == 'claude-sonnet-4-6'

    def test_no_default_substitution_for_unknown_model(self):
        body = json.loads(_build_body({'model': 'some-peer-only-model', 'messages': []}))
        assert body['model'] == 'some-peer-only-model'

    def test_missing_model_is_not_invented(self):
        body = json.loads(_build_body({'messages': []}))
        assert 'model' not in body

    def test_internal_keys_stripped(self):
        payload = {
            'model': 'sonnet',
            '_anthropic_beta': ['context-1m-2025-08-07'],
            '_anthproxy_internal_classifier': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert '_anthropic_beta' not in body
        assert '_anthproxy_internal_classifier' not in body

    def test_other_keys_pass_through(self):
        payload = {
            'model': 'opus', 'max_tokens': 256, 'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert body['max_tokens'] == 256
        assert body['stream'] is True
        assert body['messages'] == [{'role': 'user', 'content': 'hi'}]


class TestPeerBetaHeader:
    def test_absent_when_no_betas(self):
        assert _beta_header({'model': 'sonnet'}) == ''

    def test_joined_and_deduplicated(self):
        payload = {'_anthropic_beta': ['a', 'b', 'a', '  c  ', '']}
        assert _beta_header(payload) == 'a,b,c'

    def test_classifier_sentinel_is_not_a_beta(self):
        assert _beta_header({'_anthproxy_internal_classifier': True}) == ''

    def test_bare_string_is_accepted(self):
        assert _beta_header({'_anthropic_beta': 'context-1m-2025-08-07'}) == \
            'context-1m-2025-08-07'

    @pytest.mark.parametrize('raw', [5, {'a': 1}, 1.5, True])
    def test_non_list_value_from_client_body_yields_no_header(self, raw):
        """A client may set this key in the body, bypassing the handler's header lift."""
        assert _beta_header({'_anthropic_beta': raw}) == ''

    def test_non_string_entries_are_skipped(self):
        assert _beta_header({'_anthropic_beta': ['a', 7, None, 'b']}) == 'a,b'


# ---------------------------------------------------------------------------
# Headers — credential and beta forwarding
# ---------------------------------------------------------------------------

class TestPeerRequestHeaders:
    def test_peer_key_header_present_when_configured(self):
        headers = _request_headers(_make_config(peer_api_key='s3cret'), '', stream=False)
        assert headers['X-Anthproxy-Peer-Key'] == 's3cret'

    def test_peer_key_header_absent_when_unset(self):
        headers = _request_headers(_make_config(), '', stream=False)
        assert 'X-Anthproxy-Peer-Key' not in headers

    def test_authorization_header_is_never_sent(self):
        for cfg in (_make_config(), _make_config(peer_api_key='s3cret')):
            headers = _request_headers(cfg, 'beta-a', stream=True)
            assert not any(k.lower() == 'authorization' for k in headers)

    def test_beta_header_forwarded_when_present(self):
        headers = _request_headers(_make_config(), 'context-1m-2025-08-07', stream=False)
        assert headers['anthropic-beta'] == 'context-1m-2025-08-07'

    def test_beta_header_absent_when_empty(self):
        headers = _request_headers(_make_config(), '', stream=False)
        assert 'anthropic-beta' not in headers

    def test_accept_reflects_stream_flag(self):
        assert _request_headers(_make_config(), '', stream=True)['Accept'] == 'text/event-stream'
        assert _request_headers(_make_config(), '', stream=False)['Accept'] == 'application/json'


class TestPeerOutboundRequest:
    @patch('anthproxy.peer.backend._make_connection')
    def test_client_beta_survives_hop_as_header_not_body(self, mock_conn):
        conn = _mock_conn(_fake_response(200, json.dumps({'id': 'msg_01'})))
        mock_conn.return_value = conn

        PeerBackend().send_message(
            {
                'model': 'sonnet',
                '_anthropic_beta': ['context-1m-2025-08-07'],
                '_anthproxy_internal_classifier': True,
                'messages': [{'role': 'user', 'content': 'hi'}],
            },
            {}, _make_config(),
        )
        assert _sent_headers(conn)['anthropic-beta'] == 'context-1m-2025-08-07'
        body = _sent_body(conn)
        assert '_anthropic_beta' not in body
        assert '_anthproxy_internal_classifier' not in body
        assert body['model'] == 'sonnet'


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

class TestPeerConnectionHelper:
    def test_http_url_uses_plain_http_connection(self):
        conn = _make_connection(_make_config(peer_base_url='http://192.168.1.50:8082'))
        assert isinstance(conn, http.client.HTTPConnection)
        assert not isinstance(conn, http.client.HTTPSConnection)
        assert conn.port == 8082

    def test_https_url_uses_https_connection(self):
        conn = _make_connection(_make_config(peer_base_url='https://peer.example:8443'))
        assert isinstance(conn, http.client.HTTPSConnection)
        assert conn.port == 8443

    def test_default_ports_by_scheme(self):
        assert _make_connection(_make_config(peer_base_url='http://peer.example')).port == 80
        assert _make_connection(_make_config(peer_base_url='https://peer.example')).port == 443

    @pytest.mark.parametrize('base_url', [
        'localhost:8787',            # scheme-less: parses as scheme='localhost'
        '127.0.0.1:9099',            # scheme-less
        'htttps://peer.example',     # mistyped scheme
        'ftp://peer.example',        # unsupported scheme
        'http://',                   # no host
        'http://peer.example:abc',   # uncastable port
    ])
    def test_malformed_target_is_rejected_not_defaulted(self, base_url):
        """A malformed target must never silently retarget a credentialled request."""
        cfg = _make_config(peer_base_url=base_url, peer_api_key='s3cret')
        with pytest.raises(AnthropicRequestError) as exc_info:
            _make_connection(cfg)
        assert '--peer-base-url' in exc_info.value.message
        assert exc_info.value.error_type == 'api_error'

    def test_rejection_message_redacts_url_credentials(self):
        cfg = _make_config(peer_base_url='ftp://user:hunter2@peer.example:8082')
        with pytest.raises(AnthropicRequestError) as exc_info:
            _make_connection(cfg)
        assert 'hunter2' not in exc_info.value.message
        assert 'user' not in exc_info.value.message
        assert 'peer.example' in exc_info.value.message


class TestPeerPathPrefix:
    @patch('anthproxy.peer.backend._make_connection')
    def test_messages_path_carries_base_url_prefix(self, mock_conn):
        conn = _mock_conn(_fake_response(200, json.dumps({'id': 'msg_01'})))
        mock_conn.return_value = conn
        PeerBackend().send_message(
            {'model': 'sonnet', 'messages': []}, {},
            _make_config(peer_base_url='http://gw.internal/anthproxy/'))
        assert conn.request.call_args.args[1] == '/anthproxy/v1/messages'

    @patch('anthproxy.peer.backend._make_connection')
    def test_count_tokens_path_carries_base_url_prefix(self, mock_conn):
        conn = _mock_conn(_fake_response(200, json.dumps({'input_tokens': 1})))
        mock_conn.return_value = conn
        PeerBackend().count_tokens(
            {'model': 'sonnet', 'messages': []}, {},
            _make_config(peer_base_url='http://gw.internal/anthproxy'))
        assert conn.request.call_args.args[1] == '/anthproxy/v1/messages/count_tokens'

    @patch('anthproxy.peer.backend._make_connection')
    def test_bare_base_url_leaves_path_unprefixed(self, mock_conn):
        conn = _mock_conn(_fake_response(200, json.dumps({'id': 'msg_01'})))
        mock_conn.return_value = conn
        PeerBackend().send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert conn.request.call_args.args[1] == '/v1/messages'


# ---------------------------------------------------------------------------
# Unconfigured target
# ---------------------------------------------------------------------------

class TestPeerUnconfiguredTarget:
    def test_from_config_succeeds_with_empty_base_url(self):
        backend = PeerBackend.from_config(_make_config(peer_base_url=''))
        assert isinstance(backend, PeerBackend)

    @pytest.mark.parametrize('call', [
        lambda b, cfg: b.send_message({'model': 'sonnet', 'messages': []}, {}, cfg),
        lambda b, cfg: list(b.send_message_stream({'model': 'sonnet', 'messages': []}, {}, cfg)),
        lambda b, cfg: b.count_tokens({'model': 'sonnet', 'messages': []}, {}, cfg),
    ])
    def test_dispatch_raises_envelope_naming_the_flag(self, call):
        cfg = _make_config(peer_base_url='')
        with pytest.raises(AnthropicRequestError) as exc_info:
            call(PeerBackend(), cfg)
        assert '--peer-base-url' in exc_info.value.message
        assert exc_info.value.error_type == 'api_error'


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------

class TestPeerSendMessage:
    @patch('anthproxy.peer.backend._make_connection')
    def test_success_returns_peer_response(self, mock_conn):
        mock_conn.return_value = _mock_conn(_fake_response(200, json.dumps({
            'id': 'msg_01', 'model': 'claude-sonnet-4-6', 'stop_reason': 'end_turn',
        })))
        result = PeerBackend().send_message(
            {'model': 'claude-sonnet-4-6', 'messages': []}, {}, _make_config())
        assert result['stop_reason'] == 'end_turn'

    @patch('anthproxy.peer.backend._make_connection')
    def test_400_raises(self, mock_conn):
        mock_conn.return_value = _mock_conn(
            _fake_response(400, json.dumps({'error': {'message': 'bad'}})))
        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 400

    @patch('anthproxy.peer.backend._make_connection')
    def test_non_json_200_raises_502(self, mock_conn):
        mock_conn.return_value = _mock_conn(_fake_response(200, b'not json'))
        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# Retry policy — 429 always surfaces
# ---------------------------------------------------------------------------

class TestPeerRetryPolicy:
    @patch('anthproxy.peer.backend.time.sleep')
    @patch('anthproxy.peer.backend._make_connection')
    def test_429_with_retry_after_raises_without_sleeping(self, mock_conn, mock_sleep):
        resp = _fake_response(429, json.dumps({'error': {'message': 'slow down'}}),
                              headers={'retry-after': '3600'})
        conn = _mock_conn(resp)
        mock_conn.return_value = conn

        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())

        assert exc_info.value.status_code == 429
        assert exc_info.value.error_type == 'rate_limit_error'
        assert exc_info.value.retry_after == 3600.0
        mock_sleep.assert_not_called()
        assert conn.request.call_count == 1

    @patch('anthproxy.peer.backend.time.sleep')
    @patch('anthproxy.peer.backend._make_connection')
    def test_429_without_retry_after_raises_immediately(self, mock_conn, mock_sleep):
        conn = _mock_conn(_fake_response(429, json.dumps({'error': {'message': 'slow down'}})))
        mock_conn.return_value = conn

        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after is None
        mock_sleep.assert_not_called()
        assert conn.request.call_count == 1

    @patch('anthproxy.peer.backend.time.sleep')
    @patch('anthproxy.peer.backend._make_connection')
    def test_503_is_retried(self, mock_conn, mock_sleep):
        mock_conn.side_effect = [
            _mock_conn(_fake_response(503, b'unavailable')),
            _mock_conn(_fake_response(200, json.dumps({'id': 'msg_01'}))),
        ]
        result = PeerBackend().send_message(
            {'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert result['id'] == 'msg_01'
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestPeerSendMessageStream:
    @patch('anthproxy.peer.backend._make_connection')
    def test_sse_frames_pass_through_byte_for_byte(self, mock_conn):
        sse = (
            b'event: message_start\ndata: {"type":"message_start"}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        resp = MagicMock()
        resp.status = 200
        chunks = [sse[i:i + 16] for i in range(0, len(sse), 16)] + [b'']
        resp.read.side_effect = chunks
        mock_conn.return_value = _mock_conn(resp)

        frames = list(PeerBackend().send_message_stream(
            {'model': 'sonnet', 'stream': True, 'messages': []}, {}, _make_config()))
        assert frames == [
            'event: message_start\ndata: {"type":"message_start"}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        assert ''.join(frames).encode() == sse

    @patch('anthproxy.peer.backend._make_connection')
    def test_truncated_final_frame_is_terminated(self, mock_conn):
        """A stream ending without its blank line still yields a complete frame."""
        sse = b'event: message_stop\ndata: {"type":"message_stop"}\n'
        resp = MagicMock()
        resp.status = 200
        resp.read.side_effect = [sse, b'']
        mock_conn.return_value = _mock_conn(resp)

        frames = list(PeerBackend().send_message_stream(
            {'model': 'sonnet', 'stream': True, 'messages': []}, {}, _make_config()))
        assert frames == ['event: message_stop\ndata: {"type":"message_stop"}\n\n']


# ---------------------------------------------------------------------------
# count_tokens — proxied, no fallback
# ---------------------------------------------------------------------------

class TestPeerCountTokens:
    @patch('anthproxy.peer.backend._make_connection')
    def test_returns_peer_answer_not_local_estimate(self, mock_conn):
        conn = _mock_conn(_fake_response(200, json.dumps({'input_tokens': 4242})))
        mock_conn.return_value = conn

        result = PeerBackend().count_tokens(
            {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'Hello world'}]},
            {}, _make_config(peer_api_key='s3cret'),
        )
        assert result == {'input_tokens': 4242}
        assert conn.request.call_args.args[1] == '/v1/messages/count_tokens'
        headers = _sent_headers(conn)
        assert headers['X-Anthproxy-Peer-Key'] == 's3cret'
        assert not any(k.lower() == 'authorization' for k in headers)

    @pytest.mark.parametrize('status,expected_code', [
        (400, 400), (401, 401), (403, 403), (404, 502), (429, 429), (500, 502), (503, 502),
    ])
    @patch('anthproxy.peer.backend._make_connection')
    def test_non_200_surfaces_as_error_envelope(self, mock_conn, status, expected_code):
        mock_conn.return_value = _mock_conn(
            _fake_response(status, json.dumps({'error': {'message': 'nope'}})))
        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().count_tokens({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == expected_code

    @patch('anthproxy.peer.backend._make_connection')
    def test_404_message_points_at_misconfiguration(self, mock_conn):
        mock_conn.return_value = _mock_conn(_fake_response(404, b'Not Found'))
        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().count_tokens({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert '--peer-base-url' in exc_info.value.message

    @patch('anthproxy.peer.backend._make_connection')
    def test_timeout_surfaces_as_error_envelope(self, mock_conn):
        conn = MagicMock()
        conn.request.side_effect = socket.timeout('timed out')
        mock_conn.return_value = conn
        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().count_tokens({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 502
        assert exc_info.value.error_type == 'api_error'

    @patch('anthproxy.peer.backend._make_connection')
    def test_connection_refused_surfaces_as_error_envelope(self, mock_conn):
        conn = MagicMock()
        conn.request.side_effect = ConnectionRefusedError('refused')
        mock_conn.return_value = conn
        with pytest.raises(AnthropicRequestError) as exc_info:
            PeerBackend().count_tokens({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

class TestPeerCredentials:
    def test_parse_credentials_ignores_client_api_key(self):
        assert PeerBackend().parse_credentials('sk-ant-whatever') == {}

    def test_summary_credentials_is_empty(self):
        assert PeerBackend.summary_credentials(None) == {}


# ---------------------------------------------------------------------------
# Configuration and registration
# ---------------------------------------------------------------------------

class TestPeerConfiguration:
    def test_flags_default_to_unset(self, monkeypatch):
        monkeypatch.delenv('ANTHPROXY_PEER_BASE_URL', raising=False)
        monkeypatch.delenv('ANTHPROXY_PEER_API_KEY', raising=False)
        cfg = parse_args([])
        assert cfg.peer_base_url == ''
        assert cfg.peer_api_key == ''

    def test_flags_parsed_from_cli(self, monkeypatch):
        monkeypatch.delenv('ANTHPROXY_PEER_BASE_URL', raising=False)
        monkeypatch.delenv('ANTHPROXY_PEER_API_KEY', raising=False)
        cfg = parse_args(['--peer-base-url', 'http://10.0.0.5:8082', '--peer-api-key', 'k'])
        assert cfg.peer_base_url == 'http://10.0.0.5:8082'
        assert cfg.peer_api_key == 'k'

    def test_flags_read_from_env(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_PEER_BASE_URL', 'http://env-peer:8082')
        monkeypatch.setenv('ANTHPROXY_PEER_API_KEY', 'env-key')
        cfg = parse_args([])
        assert cfg.peer_base_url == 'http://env-peer:8082'
        assert cfg.peer_api_key == 'env-key'


class TestPeerRegistration:
    def test_registered_under_peer(self):
        from anthproxy import backends_registry
        assert backends_registry.get_backend('peer') is PeerBackend

    def test_declared_last(self):
        from anthproxy import backends_registry
        assert backends_registry._DECLARED_ORDER[-1] == 'peer'

    def test_discovery_completeness_passes(self):
        from anthproxy import backends_registry
        backends_registry.discover_backends()
        assert 'peer' in backends_registry.backend_names()


# ---------------------------------------------------------------------------
# End-to-end: two anthproxy instances on loopback
# ---------------------------------------------------------------------------

class _InnerBackend:
    """Stand-in provider backend for the inner anthproxy instance."""

    def parse_credentials(self, api_key: str) -> dict:
        return {}

    def send_message(self, payload, credentials, config) -> dict:
        return {
            'id': 'msg_inner', 'type': 'message', 'role': 'assistant',
            'model': payload.get('model', ''),
            'content': [{'type': 'text', 'text': 'from-inner'}],
            'stop_reason': 'end_turn',
            'usage': {'input_tokens': 3, 'output_tokens': 2},
        }

    def send_message_stream(self, payload, credentials, config):
        for event in (
            {'type': 'message_start', 'message': {
                'id': 'msg_inner', 'type': 'message', 'role': 'assistant',
                'model': payload.get('model', ''), 'content': [],
                'usage': {'input_tokens': 3, 'output_tokens': 0}}},
            {'type': 'content_block_start', 'index': 0,
             'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0,
             'delta': {'type': 'text_delta', 'text': 'from-inner'}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'},
             'usage': {'output_tokens': 2}},
            {'type': 'message_stop'},
        ):
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    def count_tokens(self, payload, credentials, config) -> dict:
        return {'input_tokens': 1234}

    def store_cached_credential(self, key, value):
        pass


def _serve(config, backend):
    """Bind on port 0 and return ``(server, thread, port)`` for the actual port.

    Binding directly avoids the bind-read-close-rebind race of picking a free
    port up front, which can hand the same port to two servers.
    """
    from anthproxy.server import BackendRegistry, create_server
    server = create_server(config, BackendRegistry(config, backend))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


def _shutdown(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def chained_proxies(tmp_path):
    inner_cfg = _make_config(port=0, backend='local',
                             anthproxy_home=str(tmp_path / 'inner'))
    inner, inner_thread, inner_port = _serve(inner_cfg, _InnerBackend())
    try:
        outer_cfg = _make_config(port=0, backend='peer',
                                 peer_base_url=f'http://127.0.0.1:{inner_port}',
                                 anthproxy_home=str(tmp_path / 'outer'))
        outer, outer_thread, outer_port = _serve(
            outer_cfg, PeerBackend.from_config(outer_cfg))
        try:
            yield f'http://127.0.0.1:{outer_port}'
        finally:
            _shutdown(outer, outer_thread)
    finally:
        _shutdown(inner, inner_thread)


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'x-api-key': 'unused'},
    )
    return urllib.request.urlopen(request, timeout=10)


class TestPeerEndToEnd:
    def test_non_streaming_returns_inner_response(self, chained_proxies):
        with _post(f'{chained_proxies}/v1/messages', {
            'model': 'claude-sonnet-4-6', 'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }) as response:
            body = json.loads(response.read())
        assert body['content'][0]['text'] == 'from-inner'
        assert body['model'] == 'claude-sonnet-4-6'

    def test_streaming_returns_inner_sse(self, chained_proxies):
        with _post(f'{chained_proxies}/v1/messages', {
            'model': 'claude-sonnet-4-6', 'max_tokens': 16, 'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }) as response:
            text = response.read().decode()
        assert 'event: message_start' in text
        assert 'from-inner' in text
        assert 'event: message_stop' in text

    def test_count_tokens_returns_inner_answer(self, chained_proxies):
        with _post(f'{chained_proxies}/v1/messages/count_tokens', {
            'model': 'claude-sonnet-4-6',
            'messages': [{'role': 'user', 'content': 'hi'}],
        }) as response:
            body = json.loads(response.read())
        assert body['input_tokens'] == 1234


# ---------------------------------------------------------------------------
# Startup self-reference check (ADR-0026)
# ---------------------------------------------------------------------------

class TestPeerSelfReferenceCheck:
    def _check(self, **kwargs):
        from anthproxy.peer.backend import check_self_reference
        check_self_reference(_make_config(**kwargs))

    def _assert_refuses(self, **kwargs):
        from anthproxy.peer.backend import PeerSelfReferenceError
        with pytest.raises(PeerSelfReferenceError) as excinfo:
            self._check(**kwargs)
        assert '--peer-base-url' in str(excinfo.value)

    def test_same_loopback_literal_and_port_refuses(self):
        self._assert_refuses(host='127.0.0.1', port=8082,
                             peer_base_url='http://127.0.0.1:8082')

    def test_localhost_target_against_loopback_bind_refuses(self):
        self._assert_refuses(host='127.0.0.1', port=8082,
                             peer_base_url='http://localhost:8082')

    def test_wildcard_bind_matches_local_target(self):
        self._assert_refuses(host='0.0.0.0', port=8082,
                             peer_base_url='http://127.0.0.1:8082')

    def test_ipv4_wildcard_bind_matches_ipv6_target(self):
        self._assert_refuses(host='0.0.0.0', port=8082,
                             peer_base_url='http://[::1]:8082')

    def test_ipv6_wildcard_bind_matches_ipv4_target(self):
        self._assert_refuses(host='::', port=8082,
                             peer_base_url='http://127.0.0.1:8082')

    def test_both_sides_resolved_across_families(self):
        """Every resolved peer address is compared against every resolved bind
        address, so the family the resolver happens to return first cannot
        decide whether the instance boots into a self-loop."""
        real = socket.getaddrinfo

        def ipv6_first(host, port, *args, **kwargs):
            if host in ('localhost', 'peerbox'):
                return [(socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('::1', port, 0, 0)),
                        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', port))]
            return real(host, port, *args, **kwargs)

        with patch('socket.getaddrinfo', side_effect=ipv6_first):
            self._assert_refuses(host='127.0.0.1', port=8082,
                                 peer_base_url='http://peerbox:8082')

    def test_same_host_different_port_starts(self):
        self._check(host='127.0.0.1', port=8082,
                    peer_base_url='http://127.0.0.1:9099')

    def test_distinct_host_starts(self):
        real = socket.getaddrinfo

        def resolver(host, port, *args, **kwargs):
            if host == 'peerbox':
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.9.9.9', port))]
            return real(host, port, *args, **kwargs)

        with patch('socket.getaddrinfo', side_effect=resolver):
            self._check(host='127.0.0.1', port=8082,
                        peer_base_url='http://peerbox:8082')

    def test_wildcard_bind_with_remote_target_starts(self):
        """Two boxes each bound to 0.0.0.0:8082 is an ordinary deployment: a
        shared port number is not a self-reference."""
        real = socket.getaddrinfo

        def resolver(host, port, *args, **kwargs):
            if host == 'peerbox':
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.9.9.9', port))]
            return real(host, port, *args, **kwargs)

        with patch('socket.getaddrinfo', side_effect=resolver):
            self._check(host='0.0.0.0', port=8082,
                        peer_base_url='http://peerbox:8082')

    def test_unresolvable_target_warns_and_proceeds(self, caplog):
        real = socket.getaddrinfo

        def resolver(host, port, *args, **kwargs):
            if host == 'nx.invalid':
                raise socket.gaierror('Name or service not known')
            return real(host, port, *args, **kwargs)

        with patch('socket.getaddrinfo', side_effect=resolver):
            with caplog.at_level('WARNING', logger='anthproxy.peer.backend'):
                self._check(host='127.0.0.1', port=8082,
                            peer_base_url='http://nx.invalid:8082')
        assert '--peer-base-url' in caplog.text

    def test_idna_failure_warns_and_proceeds(self, caplog):
        """``getaddrinfo`` raises ``UnicodeError`` — not ``OSError`` — on an IDNA
        encoding failure, which must take the warn-and-proceed path rather than
        crashing startup with a traceback carrying the unredacted URL."""
        long_label = 'x' * 64
        with caplog.at_level('WARNING', logger='anthproxy.peer.backend'):
            self._check(host='127.0.0.1', port=8082,
                        peer_base_url=f'http://{long_label}.example:8082')
        assert '--peer-base-url' in caplog.text

    def test_malformed_target_is_not_a_boot_failure(self):
        self._check(host='127.0.0.1', port=8082,
                    peer_base_url='ftp://127.0.0.1:8082')


class TestPeerSelfReferenceGuard:
    """The ``__main__`` gate: enabled-ness, fatality, and resolution avoidance."""

    def _guard(self, **kwargs):
        from anthproxy.__main__ import _guard_peer_self_reference
        _guard_peer_self_reference(_make_config(**kwargs))

    def test_positive_match_refuses_to_start(self, caplog):
        with caplog.at_level('ERROR', logger='anthproxy'):
            with pytest.raises(SystemExit) as excinfo:
                self._guard(host='127.0.0.1', port=8082,
                            peer_base_url='http://127.0.0.1:8082')
        assert excinfo.value.code != 0
        assert '--peer-base-url' in caplog.text

    def test_unset_target_resolves_nothing(self):
        with patch('socket.getaddrinfo') as resolve:
            self._guard(host='127.0.0.1', port=8082, peer_base_url='')
        resolve.assert_not_called()

    def test_peer_excluded_from_allowlist_starts_cleanly(self, caplog):
        from anthproxy import backends_registry
        backends_registry.set_enabled_backends(frozenset({'anthropic'}))
        with caplog.at_level('WARNING', logger='anthproxy'):
            with patch('socket.getaddrinfo') as resolve:
                self._guard(host='127.0.0.1', port=8082,
                            peer_base_url='http://127.0.0.1:8082')
        resolve.assert_not_called()
        assert caplog.text == ''
