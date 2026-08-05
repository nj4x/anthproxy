import json
from unittest.mock import MagicMock

import pytest

from anthproxy.config import Config
from anthproxy.mapper import AnthropicRequestError
from anthproxy.oauth.backend import OAuthBackend, _request_headers
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
