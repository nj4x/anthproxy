"""Unit tests for the local (LM Studio) backend."""

import json
from unittest.mock import MagicMock, patch

import pytest

from anthproxy.config import Config
from anthproxy.local.backend import LocalBackend, _make_connection, _request_headers
from anthproxy.local.mapper import _build_body, _resolve_model
from anthproxy.mapper import AnthropicRequestError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> Config:
    defaults = dict(
        host='127.0.0.1', port=8082, region='us-east-1',
        use_inference_profile=True, use_global_inference_profile=False,
        backend='local', log_level='INFO',
        no_prompt_translate=False, request_history_size=5, log_file='',
        codex_home='', bedrock_home='', anthropic_home='',
        local_base_url='http://127.0.0.1:1235',
        auto_backend=False, auto_backend_interval=60.0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _fake_response(status, body):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body if isinstance(body, bytes) else body.encode()
    resp.getheader.return_value = ''
    return resp


# ---------------------------------------------------------------------------
# mapper — _resolve_model
# ---------------------------------------------------------------------------

class TestLocalResolveModel:
    def test_short_alias_sonnet_resolves_to_default(self):
        # 'sonnet' is not an explicit key in the local aliases dict;
        # should fall back to 'default' = lmstudio-community/gemma-4-12B-it-MLX-4bit
        result = _resolve_model('sonnet')
        assert result == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_short_alias_opus_resolves_to_default(self):
        assert _resolve_model('opus') == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_full_claude_id_resolves_to_default(self):
        assert _resolve_model('claude-sonnet-4-6') == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_native_local_model_name_resolves_to_default(self):
        assert _resolve_model('llama-3') == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_exact_key_match_wins(self):
        from anthproxy import model_config
        # Temporarily inject a specific alias for 'sonnet' into the local backend
        original = model_config._cache
        # Build an override config
        import copy
        cfg = copy.deepcopy(model_config._DEFAULTS)
        cfg['model_aliases']['local']['sonnet'] = 'my-sonnet-model'
        model_config._cache = cfg
        try:
            assert _resolve_model('sonnet') == 'my-sonnet-model'
        finally:
            model_config._cache = original

    def test_empty_string_passes_through_default(self):
        # empty string → not in dict, 'default' key applies
        assert _resolve_model('') == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'


# ---------------------------------------------------------------------------
# mapper — _build_body
# ---------------------------------------------------------------------------

class TestLocalBuildBody:
    def test_model_is_resolved(self):
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'hi'}]}
        body = json.loads(_build_body(payload))
        assert body['model'] == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'

    def test_internal_key_stripped(self):
        payload = {
            'model': 'sonnet',
            '_anthropic_beta': ['foo'],
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert '_anthropic_beta' not in body

    def test_other_keys_pass_through(self):
        payload = {
            'model': 'opus',
            'max_tokens': 256,
            'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert body['max_tokens'] == 256
        assert body['stream'] is True
        assert body['messages'] == [{'role': 'user', 'content': 'hi'}]


# ---------------------------------------------------------------------------
# LocalBackend — parse_credentials
# ---------------------------------------------------------------------------

class TestLocalParseCredentials:
    def test_returns_empty_dict(self):
        backend = LocalBackend()
        assert backend.parse_credentials('any-key') == {}
        assert backend.parse_credentials('') == {}


# ---------------------------------------------------------------------------
# LocalBackend — count_tokens
# ---------------------------------------------------------------------------

class TestLocalCountTokens:
    def test_count_tokens_returns_local_estimate(self):
        backend = LocalBackend()
        cfg = _make_config()
        payload = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'Hello world'}],
        }
        result = backend.count_tokens(payload, {}, cfg)
        assert 'input_tokens' in result
        assert isinstance(result['input_tokens'], int)
        assert result['input_tokens'] > 0
        assert result['model'] == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'


# ---------------------------------------------------------------------------
# LocalBackend — connection helpers
# ---------------------------------------------------------------------------

class TestLocalConnectionHelper:
    def test_http_connection_for_http_url(self):
        import http.client
        cfg = _make_config(local_base_url='http://127.0.0.1:1235')
        conn = _make_connection(cfg)
        assert isinstance(conn, http.client.HTTPConnection)
        assert not isinstance(conn, http.client.HTTPSConnection)

    def test_custom_host_and_port(self):
        import http.client
        cfg = _make_config(local_base_url='http://192.168.1.100:9090')
        conn = _make_connection(cfg)
        assert isinstance(conn, http.client.HTTPConnection)

    def test_https_connection_for_https_url(self):
        import http.client
        cfg = _make_config(local_base_url='https://myserver.local:4433')
        conn = _make_connection(cfg)
        assert isinstance(conn, http.client.HTTPSConnection)

    def test_request_headers_streaming(self):
        headers = _request_headers(stream=True)
        assert headers['Content-Type'] == 'application/json'
        assert headers['Accept'] == 'text/event-stream'

    def test_request_headers_non_streaming(self):
        headers = _request_headers(stream=False)
        assert headers['Accept'] == 'application/json'


# ---------------------------------------------------------------------------
# LocalBackend — send_message (mocked HTTP)
# ---------------------------------------------------------------------------

class TestLocalSendMessage:
    def _mock_conn(self, response):
        conn = MagicMock()
        conn.getresponse.return_value = response
        return conn

    @patch('anthproxy.local.backend._make_connection')
    def test_send_message_success(self, mock_conn):
        response_body = json.dumps({
            'id': 'msg_01', 'type': 'message',
            'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hello!'}],
            'model': 'lmstudio-community/gemma-4-12B-it-MLX-4bit', 'stop_reason': 'end_turn',
            'usage': {'input_tokens': 10, 'output_tokens': 5},
        }).encode()
        resp = _fake_response(200, response_body)
        resp.read.return_value = response_body
        conn = self._mock_conn(resp)
        mock_conn.return_value = conn

        backend = LocalBackend()
        cfg = _make_config()
        result = backend.send_message(
            {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {}, cfg,
        )
        assert result['model'] == 'lmstudio-community/gemma-4-12B-it-MLX-4bit'
        assert result['stop_reason'] == 'end_turn'

    @patch('anthproxy.local.backend._make_connection')
    def test_send_message_400_raises(self, mock_conn):
        resp = _fake_response(400, json.dumps({'error': {'message': 'bad request'}}))
        conn = self._mock_conn(resp)
        mock_conn.return_value = conn

        backend = LocalBackend()
        cfg = _make_config()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, cfg)
        assert exc_info.value.status_code == 400

    @patch('anthproxy.local.backend._make_connection')
    def test_send_message_non_json_raises_502(self, mock_conn):
        resp = _fake_response(200, b'not json at all')
        conn = self._mock_conn(resp)
        mock_conn.return_value = conn

        backend = LocalBackend()
        cfg = _make_config()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, cfg)
        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# LocalBackend — send_message_stream (mocked HTTP)
# ---------------------------------------------------------------------------

class TestLocalSendMessageStream:
    def _make_sse_response(self, events):
        """Build a fake streaming response yielding SSE event strings."""
        sse_bytes = b''
        for ev_type, ev_data in events:
            sse_bytes += f'event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n'.encode()

        resp = MagicMock()
        resp.status = 200

        # read_sse_lines reads in chunks; simulate chunk-at-a-time
        chunk_size = 64
        chunks = [sse_bytes[i:i + chunk_size] for i in range(0, len(sse_bytes), chunk_size)]
        chunks.append(b'')   # EOF sentinel
        resp.read.side_effect = chunks
        return resp

    @patch('anthproxy.local.backend._make_connection')
    def test_stream_yields_events(self, mock_conn):
        events = [
            ('message_start', {'type': 'message_start', 'message': {'id': 'msg_01', 'model': 'lmstudio-community/gemma-4-12B-it-MLX-4bit'}}),
            ('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
            ('content_block_delta', {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello!'}}),
            ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
            ('message_stop', {'type': 'message_stop'}),
        ]
        resp = self._make_sse_response(events)
        conn = MagicMock()
        conn.getresponse.return_value = resp
        mock_conn.return_value = conn

        backend = LocalBackend()
        cfg = _make_config()
        chunks = list(backend.send_message_stream(
            {'model': 'sonnet', 'stream': True, 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {}, cfg,
        ))
        # Should have yielded at least one event block
        assert len(chunks) > 0
        # First chunk should contain 'message_start'
        assert any('message_start' in c for c in chunks)
