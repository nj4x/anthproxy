"""Unit tests for the OpenRouter backend."""

import json
import socket
from unittest.mock import MagicMock, patch

import pytest

from anthproxy.config import Config
from anthproxy.openrouter.backend import OpenRouterBackend, _send_with_retries
from anthproxy.openrouter.mapper import _build_body, _request_headers, _resolve_model
from anthproxy.mapper import AnthropicRequestError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _pin_default_model_config():
    """Isolate tests from any user ~/.anthproxy/config.json alias overrides."""
    import copy
    from anthproxy import model_config
    original = model_config._cache
    model_config._cache = copy.deepcopy(model_config._DEFAULTS)
    try:
        yield
    finally:
        model_config._cache = original


def _make_config(**kwargs) -> Config:
    defaults = dict(
        host='127.0.0.1', port=8082, region='us-east-1',
        use_inference_profile=True, use_global_inference_profile=False,
        backend='openrouter', log_level='INFO',
        no_prompt_translate=False, request_history_size=5, log_file='',
        anthproxy_home='', codex_home='', bedrock_home='', anthropic_home='',
        stats_dir='',
        openrouter_api_key='sk-or-test',
        local_base_url='http://127.0.0.1:1235',
        auto_backend=False, auto_backend_interval=60.0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _fake_response(status, body, headers=None):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body if isinstance(body, bytes) else body.encode()
    resp.getheader.side_effect = lambda name, default='': (headers or {}).get(name, default)
    return resp


def _mock_conn(response):
    conn = MagicMock()
    conn.getresponse.return_value = response
    return conn


# ---------------------------------------------------------------------------
# mapper — _resolve_model
# ---------------------------------------------------------------------------

class TestOpenRouterResolveModel:
    def test_haiku_alias(self):
        assert _resolve_model('haiku') == 'deepseek/deepseek-v4-flash'

    def test_sonnet_alias(self):
        assert _resolve_model('sonnet') == 'z-ai/glm-5.2'

    def test_opus_alias(self):
        assert _resolve_model('opus') == 'moonshotai/kimi-k3'

    def test_opus_long_context_suffix_stripped(self):
        # resolve_alias strips the [1m] suffix before lookup
        assert _resolve_model('opus[1m]') == 'moonshotai/kimi-k3'

    def test_opus_colon_1m_suffix_stripped(self):
        assert _resolve_model('opus:1m') == 'moonshotai/kimi-k3'

    def test_unknown_id_redirected_to_default(self):
        assert _resolve_model('claude-sonnet-4-6') == 'z-ai/glm-5.2'

    def test_native_openrouter_slug_redirected_to_default(self):
        # When 'default' is configured, unknown slugs resolve to the default catch-all.
        assert _resolve_model('mistralai/mistral-large') == 'z-ai/glm-5.2'

    def test_empty_model_raises_400(self):
        with pytest.raises(AnthropicRequestError) as exc_info:
            _resolve_model('')
        assert exc_info.value.status_code == 400


class TestOpenRouterResolveModelWithDefault:
    """_resolve_model behaviour when a 'default' catch-all is configured."""

    @pytest.fixture(autouse=True)
    def _pin_with_default(self):
        import copy
        from anthproxy import model_config
        original = model_config._cache
        patched = copy.deepcopy(model_config._DEFAULTS)
        patched['model_aliases']['openrouter']['default'] = 'some/catch-all-model'
        model_config._cache = patched
        try:
            yield
        finally:
            model_config._cache = original

    def test_unknown_claude_id_redirected_to_default(self):
        assert _resolve_model('claude-sonnet-4-6') == 'some/catch-all-model'

    def test_native_openrouter_slug_redirected_to_default(self):
        # When 'default' is set, all unrecognized IDs — including native slugs
        # — redirect to it. This matches local mapper semantics.
        assert _resolve_model('mistralai/mistral-large') == 'some/catch-all-model'

    def test_explicit_alias_not_overridden_by_default(self):
        # 'opus' is an explicit alias; default must not shadow it.
        assert _resolve_model('opus') == 'moonshotai/kimi-k3'

    def test_suffix_stripped_alias_not_overridden_by_default(self):
        assert _resolve_model('opus[1m]') == 'moonshotai/kimi-k3'

    def test_empty_model_raises_400_with_default_configured(self):
        with pytest.raises(AnthropicRequestError) as exc_info:
            _resolve_model('')
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# mapper — _build_body
# ---------------------------------------------------------------------------

class TestOpenRouterBuildBody:
    def test_model_is_resolved(self):
        payload = {'model': 'opus', 'messages': [{'role': 'user', 'content': 'hi'}]}
        body = json.loads(_build_body(payload))
        assert body['model'] == 'moonshotai/kimi-k3'

    def test_anthropic_beta_stripped(self):
        payload = {
            'model': 'sonnet',
            '_anthropic_beta': ['context-1m'],
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert '_anthropic_beta' not in body

    def test_internal_classifier_sentinel_stripped(self):
        payload = {
            'model': 'sonnet',
            '_anthproxy_internal_classifier': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert '_anthproxy_internal_classifier' not in body

    def test_other_keys_pass_through(self):
        payload = {
            'model': 'haiku',
            'max_tokens': 256,
            'stream': True,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert body['max_tokens'] == 256
        assert body['stream'] is True
        assert body['messages'] == [{'role': 'user', 'content': 'hi'}]
        assert body['model'] == 'deepseek/deepseek-v4-flash'


# ---------------------------------------------------------------------------
# mapper — _request_headers
# ---------------------------------------------------------------------------

class TestOpenRouterRequestHeaders:
    def test_streaming_accept(self):
        headers = _request_headers('sk-or-abc', stream=True)
        assert headers['Accept'] == 'text/event-stream'

    def test_non_streaming_accept(self):
        headers = _request_headers('sk-or-abc', stream=False)
        assert headers['Accept'] == 'application/json'

    def test_authorization_bearer(self):
        headers = _request_headers('sk-or-abc', stream=False)
        assert headers['Authorization'] == 'Bearer sk-or-abc'

    def test_content_type(self):
        headers = _request_headers('sk-or-abc', stream=False)
        assert headers['Content-Type'] == 'application/json'

    def test_attribution_headers_present(self):
        headers = _request_headers('sk-or-abc', stream=False)
        assert 'HTTP-Referer' in headers
        assert 'X-Title' in headers

    def test_no_anthropic_specific_headers(self):
        headers = _request_headers('sk-or-abc', stream=False)
        assert 'anthropic-version' not in headers
        assert 'x-api-key' not in headers


# ---------------------------------------------------------------------------
# OpenRouterBackend — parse_credentials
# ---------------------------------------------------------------------------

class TestOpenRouterParseCredentials:
    def test_returns_empty_dict(self):
        backend = OpenRouterBackend()
        assert backend.parse_credentials('any-key') == {}
        assert backend.parse_credentials('') == {}


# ---------------------------------------------------------------------------
# OpenRouterBackend — count_tokens
# ---------------------------------------------------------------------------

class TestOpenRouterCountTokens:
    def test_count_tokens_returns_local_estimate(self):
        backend = OpenRouterBackend()
        cfg = _make_config()
        payload = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'Hello world'}],
        }
        result = backend.count_tokens(payload, {}, cfg)
        assert isinstance(result['input_tokens'], int)
        assert result['input_tokens'] > 0
        assert result['model'] == 'z-ai/glm-5.2'


# ---------------------------------------------------------------------------
# OpenRouterBackend — send_message (mocked HTTP)
# ---------------------------------------------------------------------------

class TestOpenRouterSendMessage:
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_success(self, mock_make_conn):
        response_body = json.dumps({
            'id': 'msg_01', 'type': 'message', 'role': 'assistant',
            'content': [{'type': 'text', 'text': 'Hello!'}],
            'model': 'z-ai/glm-5.2', 'stop_reason': 'end_turn',
            'usage': {'input_tokens': 10, 'output_tokens': 5},
        })
        mock_make_conn.return_value = _mock_conn(_fake_response(200, response_body))

        backend = OpenRouterBackend()
        result = backend.send_message(
            {'model': 'opus', 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {}, _make_config(),
        )
        assert result['model'] == 'z-ai/glm-5.2'
        assert result['stop_reason'] == 'end_turn'

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_missing_api_key_raises_401(self, mock_make_conn):
        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message(
                {'model': 'sonnet', 'messages': []}, {},
                _make_config(openrouter_api_key=''),
            )
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_type == 'authentication_error'
        # No network call should have been attempted.
        mock_make_conn.assert_not_called()

    @patch('anthproxy.openrouter.backend.time.sleep')
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_402_credit_exhaustion_raises_429_without_retry(self, mock_make_conn, mock_sleep):
        mock_make_conn.return_value = _mock_conn(
            _fake_response(402, json.dumps({'error': {'message': 'insufficient credits'}})))

        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_type == 'rate_limit_error'
        # 402 is surfaced immediately — exactly one connection, no sleeps.
        assert mock_make_conn.call_count == 1
        mock_sleep.assert_not_called()

    @patch('anthproxy.openrouter.backend.time.sleep')
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_429_retried_then_raised(self, mock_make_conn, mock_sleep):
        resp = _fake_response(429, json.dumps({'error': {'message': 'rate limited'}}))
        mock_make_conn.side_effect = lambda host: _mock_conn(resp)

        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 429
        # 3 retries + final attempt = 4 connections; 3 sleeps.
        assert mock_make_conn.call_count == 4
        assert mock_sleep.call_count == 3

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_400_raises_immediately(self, mock_make_conn):
        mock_make_conn.return_value = _mock_conn(
            _fake_response(400, json.dumps({'error': {'message': 'bad request'}})))

        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 400
        assert mock_make_conn.call_count == 1

    @patch('anthproxy.openrouter.backend.time.sleep')
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_network_error_retried_then_502(self, mock_make_conn, mock_sleep):
        conn = MagicMock()
        conn.request.side_effect = socket.error('connection refused')
        mock_make_conn.side_effect = lambda host: conn

        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 502
        assert mock_make_conn.call_count == 4
        assert mock_sleep.call_count == 3

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_non_json_200_raises_502(self, mock_make_conn):
        mock_make_conn.return_value = _mock_conn(_fake_response(200, b'not json at all'))

        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            backend.send_message({'model': 'sonnet', 'messages': []}, {}, _make_config())
        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# OpenRouterBackend — send_message_stream (mocked HTTP)
# ---------------------------------------------------------------------------

class TestOpenRouterSendMessageStream:
    def _make_sse_response(self, raw_lines):
        """Build a fake streaming response yielding the given raw SSE text."""
        sse_bytes = raw_lines.encode()
        resp = MagicMock()
        resp.status = 200
        chunk_size = 64
        chunks = [sse_bytes[i:i + chunk_size] for i in range(0, len(sse_bytes), chunk_size)]
        chunks.append(b'')
        resp.read.side_effect = chunks
        return resp

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_stream_yields_events(self, mock_make_conn):
        sse = (
            'event: message_start\n'
            'data: {"type":"message_start","message":{"id":"msg_01","model":"z-ai/glm-5.2"}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n'
            '\n'
            'event: message_stop\n'
            'data: {"type":"message_stop"}\n'
            '\n'
        )
        resp = self._make_sse_response(sse)
        mock_make_conn.return_value = _mock_conn(resp)

        backend = OpenRouterBackend()
        chunks = list(backend.send_message_stream(
            {'model': 'opus', 'stream': True, 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {}, _make_config(),
        ))
        assert len(chunks) > 0
        assert any('message_start' in c for c in chunks)
        assert any('message_stop' in c for c in chunks)

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_done_sentinel_filtered(self, mock_make_conn):
        sse = (
            'event: message_stop\n'
            'data: {"type":"message_stop"}\n'
            '\n'
            'data: [DONE]\n'
            '\n'
        )
        resp = self._make_sse_response(sse)
        mock_make_conn.return_value = _mock_conn(resp)

        backend = OpenRouterBackend()
        chunks = list(backend.send_message_stream(
            {'model': 'opus', 'stream': True, 'messages': [{'role': 'user', 'content': 'Hi'}]},
            {}, _make_config(),
        ))
        # The OpenAI-style [DONE] sentinel must never be forwarded to the client.
        assert not any('[DONE]' in c for c in chunks)

    @patch('anthproxy.openrouter.backend.time.sleep')
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_pre_stream_402_raises(self, mock_make_conn, mock_sleep):
        mock_make_conn.return_value = _mock_conn(
            _fake_response(402, json.dumps({'error': {'message': 'insufficient credits'}})))

        backend = OpenRouterBackend()
        with pytest.raises(AnthropicRequestError) as exc_info:
            list(backend.send_message_stream(
                {'model': 'opus', 'stream': True, 'messages': []}, {}, _make_config(),
            ))
        assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# OpenRouterBackend — get_usage_markdown (credits monitoring)
# ---------------------------------------------------------------------------

class TestOpenRouterGetUsageMarkdown:
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_success(self, mock_make_conn):
        body = json.dumps({'data': {'total_credits': 100.0, 'total_usage': 25.5}})
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        backend = OpenRouterBackend()
        md = backend.get_usage_markdown(_make_config())
        assert '## OpenRouter credits' in md
        assert '$25.50 used' in md
        assert '$74.50 remaining' in md
        assert '$100.00 purchased' in md

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_missing_api_key(self, mock_make_conn):
        backend = OpenRouterBackend()
        md = backend.get_usage_markdown(_make_config(openrouter_api_key=''))
        assert 'OPENROUTER_API_KEY is not set' in md
        # No network call is attempted when the key is missing.
        mock_make_conn.assert_not_called()

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_403_management_key_required(self, mock_make_conn):
        # A regular inference key yields HTTP 403 on /api/v1/credits.
        body = json.dumps({'error': {'message': 'No permission for this resource'}})
        mock_make_conn.return_value = _mock_conn(_fake_response(403, body))

        backend = OpenRouterBackend()
        md = backend.get_usage_markdown(_make_config())
        assert '## OpenRouter credits' in md
        assert 'Usage information is unavailable' in md

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_network_failure(self, mock_make_conn):
        conn = MagicMock()
        conn.request.side_effect = socket.error('connection refused')
        mock_make_conn.return_value = conn

        backend = OpenRouterBackend()
        md = backend.get_usage_markdown(_make_config())
        assert '## OpenRouter credits' in md
        assert 'Usage information is unavailable' in md
        assert 'connection refused' in md

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_non_json_body(self, mock_make_conn):
        mock_make_conn.return_value = _mock_conn(_fake_response(200, b'not json at all'))

        backend = OpenRouterBackend()
        md = backend.get_usage_markdown(_make_config())
        assert '## OpenRouter credits' in md
        assert 'Usage information is unavailable' in md

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_success_is_cached(self, mock_make_conn):
        body = json.dumps({'data': {'total_credits': 100.0, 'total_usage': 25.5}})
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        backend = OpenRouterBackend()
        first = backend.get_usage_markdown(_make_config())
        second = backend.get_usage_markdown(_make_config())
        assert '$25.50 used' in first
        assert second == first
        assert mock_make_conn.call_count == 1


# ---------------------------------------------------------------------------
# OpenRouterBackend — five_hour_status (selector integration)
# ---------------------------------------------------------------------------

class TestOpenRouterFiveHourStatus:
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_available_with_credit_utilization(self, mock_make_conn):
        body = json.dumps({'data': {'total_credits': 100.0, 'total_usage': 25.5}})
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        status = OpenRouterBackend().five_hour_status(_make_config())
        assert status.available is True
        assert status.resets_at is None
        assert status.utilization == 25.5
        assert status.weekly_utilization == 25.5

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_exhausted_when_credit_balance_empty(self, mock_make_conn):
        body = json.dumps({'data': {'total_credits': 10.0, 'total_usage': 10.0}})
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        status = OpenRouterBackend().five_hour_status(_make_config())
        assert status.available is False
        assert status.utilization == 100.0
        assert status.weekly_utilization == 100.0

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_missing_key_is_unavailable(self, mock_make_conn):
        status = OpenRouterBackend().five_hour_status(_make_config(openrouter_api_key=''))
        assert status.available is False
        assert status.resets_at is None
        mock_make_conn.assert_not_called()

    @patch('anthproxy.openrouter.backend.time.time', return_value=1000.0)
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_usage_rate_limit_reports_retry_time(self, mock_make_conn, _mock_time):
        body = json.dumps({'error': {'message': 'rate limited'}})
        mock_make_conn.return_value = _mock_conn(_fake_response(429, body, {'Retry-After': '17'}))

        status = OpenRouterBackend().five_hour_status(_make_config())
        assert status.available is None
        assert status.resets_at == 1017.0

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_management_key_error_is_unknown_not_exhausted(self, mock_make_conn):
        body = json.dumps({'error': {'message': 'No permission for this resource'}})
        mock_make_conn.return_value = _mock_conn(_fake_response(403, body))

        status = OpenRouterBackend().five_hour_status(_make_config())
        assert status.available is None
        assert status.resets_at is None


# ---------------------------------------------------------------------------
# _send_with_retries — missing key is caught upstream (defensive)
# ---------------------------------------------------------------------------

class TestSendWithRetriesDirect:
    @patch('anthproxy.openrouter.backend.make_connection')
    def test_returns_conn_and_resp_on_200(self, mock_make_conn):
        resp = _fake_response(200, json.dumps({'ok': True}))
        conn = _mock_conn(resp)
        mock_make_conn.return_value = conn

        got_conn, got_resp = _send_with_retries(
            {'model': 'sonnet', 'messages': []}, 'sk-or-test', stream=False)
        assert got_conn is conn
        assert got_resp is resp


# ---------------------------------------------------------------------------
# send_classifier_message — disable thinking + bump max_tokens for reasoning models
# ---------------------------------------------------------------------------

class TestOpenRouterSendClassifierMessage:
    def _resp_body(self, content):
        return json.dumps({'content': content, 'stop_reason': 'end_turn'}).encode()

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_disables_thinking_and_bumps_max_tokens(self, mock_make_conn):
        body = self._resp_body([{'type': 'text', 'text': 'standard'}])
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        backend = OpenRouterBackend()
        backend.send_classifier_message(
            {'model': 'haiku', 'max_tokens': 4, 'messages': [{'role': 'user', 'content': 'q'}]},
            {}, _make_config(),
        )

        sent_body = json.loads(mock_make_conn.return_value.request.call_args[1]['body'])
        assert sent_body['thinking'] == {'type': 'disabled'}
        assert sent_body['max_tokens'] == 64

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_existing_max_tokens_preserved_if_larger(self, mock_make_conn):
        body = self._resp_body([{'type': 'text', 'text': 'standard'}])
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        backend = OpenRouterBackend()
        backend.send_classifier_message(
            {'model': 'haiku', 'max_tokens': 200, 'messages': [{'role': 'user', 'content': 'q'}]},
            {}, _make_config(),
        )

        sent_body = json.loads(mock_make_conn.return_value.request.call_args[1]['body'])
        assert sent_body['max_tokens'] == 200

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_response_with_thinking_block_parsed(self, mock_make_conn):
        from anthproxy.model_router import parse_classifier_label
        body = self._resp_body([
            {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig'},
            {'type': 'text', 'text': 'standard'},
        ])
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        resp = OpenRouterBackend().send_classifier_message(
            {'model': 'haiku', 'max_tokens': 4, 'messages': [{'role': 'user', 'content': 'q'}]},
            {}, _make_config(),
        )
        assert parse_classifier_label(resp) == 'standard'

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_thinking_only_response_fails_closed(self, mock_make_conn):
        from anthproxy.model_router import parse_classifier_label
        body = self._resp_body([
            {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig'},
        ])
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        resp = OpenRouterBackend().send_classifier_message(
            {'model': 'haiku', 'max_tokens': 4, 'messages': [{'role': 'user', 'content': 'q'}]},
            {}, _make_config(),
        )
        assert parse_classifier_label(resp) is None


# ---------------------------------------------------------------------------
# Thinking-signature tagging — prevent 400 on OpenRouter→Anthropic switch
# ---------------------------------------------------------------------------

class TestOpenRouterSignatureTagging:
    """OpenRouter stamps `or:` onto its response signatures so the Anthropic/
    Bedrock mappers can strip them on a backend switch (foreign signatures are
    rejected with HTTP 400 'Invalid signature in thinking block')."""

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_non_streaming_thinking_signature_tagged(self, mock_make_conn):
        from anthproxy.mapper import OPENROUTER_REASONING_SIG_PREFIX
        body = json.dumps({
            'content': [
                {'type': 'thinking', 'thinking': 'r', 'signature': 'upstreamSig=='},
                {'type': 'text', 'text': 'answer'},
            ],
            'stop_reason': 'end_turn',
        }).encode()
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        resp = OpenRouterBackend().send_message(
            {'model': 'sonnet', 'messages': []}, {}, _make_config(),
        )
        thinking = resp['content'][0]
        assert thinking['signature'] == OPENROUTER_REASONING_SIG_PREFIX + 'upstreamSig=='

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_non_streaming_already_tagged_not_double_prefixed(self, mock_make_conn):
        from anthproxy.mapper import OPENROUTER_REASONING_SIG_PREFIX
        sig = OPENROUTER_REASONING_SIG_PREFIX + 'upstreamSig=='
        body = json.dumps({
            'content': [
                {'type': 'thinking', 'thinking': 'r', 'signature': sig},
            ],
        }).encode()
        mock_make_conn.return_value = _mock_conn(_fake_response(200, body))

        resp = OpenRouterBackend().send_message(
            {'model': 'sonnet', 'messages': []}, {}, _make_config(),
        )
        assert resp['content'][0]['signature'] == sig

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_streaming_signature_delta_tagged(self, mock_make_conn):
        from anthproxy.mapper import OPENROUTER_REASONING_SIG_PREFIX
        sse = (
            'event: content_block_delta\n'
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"upstreamSig=="}}\n'
            '\n'
            'data: [DONE]\n'
        )
        sse_bytes = sse.encode()
        resp = MagicMock()
        resp.status = 200
        chunks = [sse_bytes[i:i + 64] for i in range(0, len(sse_bytes), 64)]
        chunks.append(b'')
        resp.read.side_effect = chunks
        mock_make_conn.return_value = _mock_conn(resp)

        events = list(OpenRouterBackend().send_message_stream(
            {'model': 'sonnet', 'messages': []}, {}, _make_config(),
        ))
        tagged = [e for e in events if 'signature_delta' in e]
        assert tagged
        assert OPENROUTER_REASONING_SIG_PREFIX + 'upstreamSig==' in tagged[0]

    @patch('anthproxy.openrouter.backend.make_connection')
    def test_streaming_non_signature_event_passes_through(self, mock_make_conn):
        sse = (
            'event: content_block_delta\n'
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n'
            '\n'
            'data: [DONE]\n'
        )
        sse_bytes = sse.encode()
        resp = MagicMock()
        resp.status = 200
        chunks = [sse_bytes[i:i + 64] for i in range(0, len(sse_bytes), 64)]
        chunks.append(b'')
        resp.read.side_effect = chunks
        mock_make_conn.return_value = _mock_conn(resp)

        events = list(OpenRouterBackend().send_message_stream(
            {'model': 'sonnet', 'messages': []}, {}, _make_config(),
        ))
        assert any('text_delta' in e for e in events)


class TestOpenRouterMapperStripsThinkingFromRequests:
    """The openrouter mapper strips ALL thinking blocks from request history so
    the upstream never receives a foreign signature (from a prior OpenRouter turn
    or a prior Anthropic-subscription turn)."""

    def _build(self, messages):
        return json.loads(_build_body({
            'model': 'sonnet',
            'messages': messages,
        }))

    def test_strips_thinking_block_from_assistant_history(self):
        body = self._build([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'r', 'signature': 'or:foreign=='},
                {'type': 'text', 'text': 'answer'},
            ]},
            {'role': 'user', 'content': 'follow up'},
        ])
        asst = body['messages'][1]
        assert all(b.get('type') != 'thinking' for b in asst['content'])
        assert any(b.get('type') == 'text' for b in asst['content'])

    def test_strips_redacted_thinking_block(self):
        body = self._build([
            {'role': 'assistant', 'content': [
                {'type': 'redacted_thinking', 'data': 'opaque'},
                {'type': 'text', 'text': 'answer'},
            ]},
        ])
        assert all(b.get('type') != 'redacted_thinking' for b in body['messages'][0]['content'])

    def test_degenerate_all_thinking_inserts_empty_text(self):
        body = self._build([
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'r', 'signature': 'sig'},
            ]},
        ])
        assert body['messages'][0]['content'] == [{'type': 'text', 'text': ''}]

    def test_user_messages_untouched(self):
        body = self._build([
            {'role': 'user', 'content': [
                {'type': 'thinking', 'thinking': 'x', 'signature': 'sig'},
            ]},
        ])
        assert body['messages'][0]['content'] == [
            {'type': 'thinking', 'thinking': 'x', 'signature': 'sig'}]

    def test_no_thinking_blocks_passes_through_unchanged(self):
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]},
        ]
        body = self._build(msgs)
        assert body['messages'][1]['content'] == [{'type': 'text', 'text': 'ok'}]
