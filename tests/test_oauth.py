import http.client
import json
import threading
from unittest.mock import MagicMock

import pytest

from anthproxy.config import Config
from anthproxy.mapper import AnthropicRequestError
from anthproxy.oauth.backend import OAuthBackend, _request_headers, _send
from anthproxy.oauth_registry import OAuthRequestCredentials


def test_requires_request_credentials():
    backend = OAuthBackend()

    with pytest.raises(AnthropicRequestError, match='Bearer'):
        backend.send_message({'model': 'haiku', 'messages': []}, {}, Config())


def test_request_headers_use_only_selected_bearer_token():
    headers = _request_headers('enterprise-secret', 'oauth-2025-04-20', False)

    assert headers['Authorization'] == 'Bearer enterprise-secret'
    assert 'x-api-key' not in headers


def test_send_message_reuses_anthropic_mapper(monkeypatch):
    response = MagicMock()
    response.status = 200
    response.read.return_value = json.dumps({'type': 'message', 'model': 'claude-haiku'}).encode()
    connection = MagicMock()
    connection.getresponse.return_value = response
    monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)
    backend = OAuthBackend()

    result = backend.send_message(
        {'model': 'haiku', 'messages': [{'role': 'user', 'content': 'hello'}]},
        {'oauth': OAuthRequestCredentials(3, 'enterprise-secret')},
        Config(),
    )

    assert result['type'] == 'message'
    request = connection.request.call_args
    body = json.loads(request.kwargs['body'])
    assert body['model'].startswith('claude-haiku')
    assert request.kwargs['headers']['Authorization'] == 'Bearer enterprise-secret'


def test_send_message_stream_uses_selected_bearer_token(monkeypatch):
    response = MagicMock()
    response.status = 200
    response.read.side_effect = [b'data: {"type": "message_start"}\n\n', b'']
    connection = MagicMock()
    connection.getresponse.return_value = response
    monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)
    backend = OAuthBackend()

    chunks = list(backend.send_message_stream(
        {'model': 'haiku', 'messages': [{'role': 'user', 'content': 'hi'}]},
        {'oauth': OAuthRequestCredentials(2, 'enterprise-secret')},
        Config(),
    ))

    assert any('message_start' in chunk for chunk in chunks)
    request = connection.request.call_args
    assert request.kwargs['headers']['Authorization'] == 'Bearer enterprise-secret'
    assert request.kwargs['headers']['Accept'] == 'text/event-stream'


def test_send_message_attaches_retry_after_on_429(monkeypatch):
    response = MagicMock()
    response.status = 429
    response.read.return_value = json.dumps(
        {'error': {'type': 'rate_limit_error', 'message': 'slow down'}}
    ).encode()
    response.getheader.side_effect = lambda name, default='': (
        {'Retry-After': '12'}.get(name, default)
    )
    connection = MagicMock()
    connection.getresponse.return_value = response
    monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)
    backend = OAuthBackend()

    with pytest.raises(AnthropicRequestError) as excinfo:
        backend.send_message(
            {'model': 'haiku', 'messages': [{'role': 'user', 'content': 'hi'}]},
            {'oauth': OAuthRequestCredentials(1, 'enterprise-secret')},
            Config(),
        )

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 12.0


def test_count_tokens_falls_back_to_estimate_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise AnthropicRequestError('upstream down', error_type='api_error', status_code=502)

    monkeypatch.setattr('anthproxy.oauth.backend._send', boom)
    backend = OAuthBackend()

    result = backend.count_tokens(
        {'model': 'haiku', 'messages': [{'role': 'user', 'content': 'hello there'}]},
        {'oauth': OAuthRequestCredentials(1, 'enterprise-secret')},
        Config(),
    )

    assert isinstance(result['input_tokens'], int)
    assert result['model'].startswith('claude-haiku')


def test_model_aliases_match_anthropic():
    from anthproxy import model_config

    assert OAuthBackend.model_aliases() == model_config.model_aliases('anthropic')


def test_summary_credentials_are_never_available():
    assert OAuthBackend.summary_credentials(object()) is None


# ---------------------------------------------------------------------------
# _send: thinking-signature 400 recovery (parity with anthropic backend)
# ---------------------------------------------------------------------------

class TestSendThinkingSignatureRecovery:
    """When the OAuth backend gets HTTP 400 "Invalid signature in thinking
    block" (caused by cross-model/cross-backend thinking blocks in history),
    _send must strip all thinking blocks and retry exactly once.  Mirrors
    TestSendWithRetriesThinkingSignatureRecovery in test_anthropic.py.
    """

    def _make_response(self, status, body_dict, headers=None):
        r = MagicMock(spec=http.client.HTTPResponse)
        r.status = status
        r.read.return_value = json.dumps(body_dict).encode()
        headers = headers or {}
        r.getheader.side_effect = lambda name, default='': headers.get(name, default)
        return r

    def _thinking_400_body(self):
        return {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'messages.3.content.0: Invalid `signature` in `thinking` block',
            },
        }

    def _ok_response(self):
        return {
            'type': 'message',
            'id': 'msg_ok',
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'ok'}],
            'model': 'claude-sonnet-4-6',
            'stop_reason': 'end_turn',
            'usage': {'input_tokens': 10, 'output_tokens': 5},
        }

    def _creds(self):
        return {'oauth': OAuthRequestCredentials(1, 'enterprise-secret')}

    def test_strips_thinking_and_retries_on_400_signature_error(self, monkeypatch):
        connection = MagicMock()
        connection.getresponse.side_effect = [
            self._make_response(400, self._thinking_400_body()),
            self._make_response(200, self._ok_response()),
        ]
        monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)

        payload = {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'foreign reasoning', 'signature': 'rawSig=='},
                    {'type': 'text', 'text': 'response'},
                ]},
                {'role': 'user', 'content': 'follow up'},
            ],
        }
        conn, resp = _send(payload, self._creds(), stream=False)
        assert resp.status == 200
        assert connection.request.call_count == 2
        second_body = json.loads(connection.request.call_args_list[1].kwargs['body'])
        asst_blocks = second_body['messages'][1]['content']
        assert all(b.get('type') != 'thinking' for b in asst_blocks)
        assert any(b.get('type') == 'text' for b in asst_blocks)

    def test_retries_at_most_once_on_repeated_signature_error(self, monkeypatch):
        connection = MagicMock()
        connection.getresponse.side_effect = [
            self._make_response(400, self._thinking_400_body()),
            self._make_response(400, self._thinking_400_body()),
        ]
        monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)

        payload = {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'reason', 'signature': 'sig=='},
                    {'type': 'text', 'text': 'response'},
                ]},
                {'role': 'user', 'content': 'continue'},
            ],
        }
        with pytest.raises(AnthropicRequestError) as exc:
            _send(payload, self._creds(), stream=False)
        assert exc.value.status_code == 400
        assert connection.request.call_count == 2

    def test_no_retry_when_no_thinking_blocks_to_strip(self, monkeypatch):
        connection = MagicMock()
        connection.getresponse.return_value = self._make_response(400, self._thinking_400_body())
        monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)

        payload = {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'no thinking here'}]},
            ],
        }
        with pytest.raises(AnthropicRequestError) as exc:
            _send(payload, self._creds(), stream=False)
        assert exc.value.status_code == 400
        assert connection.request.call_count == 1

    def test_non_signature_400_not_retried(self, monkeypatch):
        other_400 = {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'some other 400 error',
            },
        }
        connection = MagicMock()
        connection.getresponse.return_value = self._make_response(400, other_400)
        monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)

        payload = {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'reason', 'signature': 'sig=='},
                    {'type': 'text', 'text': 'response'},
                ]},
            ],
        }
        with pytest.raises(AnthropicRequestError) as exc:
            _send(payload, self._creds(), stream=False)
        assert exc.value.status_code == 400
        assert connection.request.call_count == 1

    def test_strips_redacted_thinking_on_data_error(self, monkeypatch):
        redacted_400 = {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'messages.1.content.0: Invalid `data` in `redacted_thinking` block',
            },
        }
        connection = MagicMock()
        connection.getresponse.side_effect = [
            self._make_response(400, redacted_400),
            self._make_response(200, self._ok_response()),
        ]
        monkeypatch.setattr('anthproxy.oauth.backend._make_connection', lambda: connection)

        payload = {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'redacted_thinking', 'data': 'opaque=='},
                    {'type': 'text', 'text': 'response'},
                ]},
                {'role': 'user', 'content': 'continue'},
            ],
        }
        conn, resp = _send(payload, self._creds(), stream=False)
        assert resp.status == 200
        second_body = json.loads(connection.request.call_args_list[1].kwargs['body'])
        asst_blocks = second_body['messages'][1]['content']
        assert all(b.get('type') != 'redacted_thinking' for b in asst_blocks)
