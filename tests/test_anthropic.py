"""Unit tests for the Anthropic backend — pure functions, no network calls.

Tests cover model resolution, beta-header merging, request-body preparation
(system-prefix safety net), and auth-credential expiry logic.
"""

import datetime as dt
import json
import pathlib
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import http.client
import pytest

from anthproxy._shared import UsageRateLimitError
from anthproxy.anthropic.backend import (
    AnthropicBackend,
    _fetch_usage,
    _handle_error_response,
    _max_weekly_utilization,
)
from anthproxy.anthropic.mapper import (
    ANTHROPIC_MODEL_ALIASES,
    REQUIRED_BETAS,
    _CC_SYSTEM_PREFIX,
    _build_body,
    _ensure_cc_system_prefix,
    _merge_betas,
    _normalize_cache_ttl_ordering,
    _resolve_model,
)
from anthproxy.anthropic.auth import (
    ACCESS_REFRESH_WINDOW_SECS,
    _anthropic_home,
    _auth_file,
    _pkce,
    _write_auth,
    load_credentials,
    needs_access_refresh,
)
from anthproxy._shared.http_util import should_retry
from anthproxy.mapper import AnthropicRequestError


# ---------------------------------------------------------------------------
# should_retry helper
# ---------------------------------------------------------------------------

class TestShouldRetry:
    """Unit tests for _shared.http_util.should_retry."""

    def _response(self, headers=None):
        response = MagicMock(spec=http.client.HTTPResponse)
        headers = headers or {}
        response.getheader.side_effect = lambda name, default='': headers.get(name, default)
        return response

    def test_429_without_retry_after_is_not_retried(self):
        """A bare 429 with no timing guidance must surface immediately."""
        assert should_retry(429, self._response()) is False

    def test_429_with_retry_after_seconds_is_retried(self):
        assert should_retry(429, self._response({'Retry-After': '30'})) is True

    def test_429_with_retry_after_ms_is_retried(self):
        assert should_retry(429, self._response({'retry-after-ms': '5000'})) is True

    def test_5xx_always_retried_regardless_of_headers(self):
        for status in (500, 502, 503, 504):
            assert should_retry(status, self._response()) is True, f'status {status} should retry'

    def test_4xx_other_than_429_never_retried(self):
        for status in (400, 401, 403, 404):
            assert should_retry(status, self._response()) is False, f'status {status} must not retry'

    def test_none_resp_429_not_retried(self):
        """None response (no headers available) behaves as no Retry-After."""
        assert should_retry(429, None) is False


class TestAnthropicUsage:
    def _response(self, status, data=None, headers=None):
        response = MagicMock(spec=http.client.HTTPResponse)
        response.status = status
        response.read.return_value = json.dumps(data or {}).encode()
        headers = headers or {}
        response.getheader.side_effect = lambda name, default='': headers.get(name, default)
        return response

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_429_uses_retry_after_header(self, mock_get_access, mock_connection):
        mock_get_access.return_value = ('access', None)
        connection = MagicMock()
        connection.getresponse.return_value = self._response(429, headers={'Retry-After': '20'})
        mock_connection.return_value = connection

        with pytest.raises(UsageRateLimitError) as exc_info:
            _fetch_usage(MagicMock(), threading.Lock())
        assert exc_info.value.retry_after == 20.0

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_429_prefers_retry_after_ms(self, mock_get_access, mock_connection):
        mock_get_access.return_value = ('access', None)
        connection = MagicMock()
        connection.getresponse.return_value = self._response(429, headers={
            'retry-after-ms': '2500',
            'Retry-After': '20',
        })
        mock_connection.return_value = connection

        with pytest.raises(UsageRateLimitError) as exc_info:
            _fetch_usage(MagicMock(), threading.Lock())
        assert exc_info.value.retry_after == 2.5

    @patch('anthproxy.anthropic.backend.time.time', return_value=1_000.0)
    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_429_parses_http_date_retry_after(self, mock_get_access, mock_connection, _mock_time):
        mock_get_access.return_value = ('access', None)
        future = dt.datetime.fromtimestamp(1_045.0, tz=dt.timezone.utc)
        connection = MagicMock()
        connection.getresponse.return_value = self._response(429, headers={
            'Retry-After': future.strftime('%a, %d %b %Y %H:%M:%S GMT'),
        })
        mock_connection.return_value = connection

        with pytest.raises(UsageRateLimitError) as exc_info:
            _fetch_usage(MagicMock(), threading.Lock())
        assert exc_info.value.retry_after == 45.0


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    def test_short_alias_opus(self):
        assert _resolve_model('opus') == ANTHROPIC_MODEL_ALIASES['opus']

    def test_short_alias_sonnet(self):
        assert _resolve_model('sonnet') == ANTHROPIC_MODEL_ALIASES['sonnet']

    def test_short_alias_haiku(self):
        assert _resolve_model('haiku') == ANTHROPIC_MODEL_ALIASES['haiku']

    def test_full_anthropic_id_pass_through(self):
        model = 'claude-opus-4-8'
        assert _resolve_model(model) == model

    def test_context_suffix_1m_stripped(self):
        base = 'opus'
        result = _resolve_model(f'{base}:1m')
        assert result == ANTHROPIC_MODEL_ALIASES[base]

    def test_context_suffix_bracket_stripped(self):
        base = 'sonnet'
        result = _resolve_model(f'{base}[1m]')
        assert result == ANTHROPIC_MODEL_ALIASES[base]

    def test_unknown_model_passes_through(self):
        model = 'gpt-5-custom'
        assert _resolve_model(model) == model

    def test_empty_model_raises(self):
        with pytest.raises(AnthropicRequestError) as exc_info:
            _resolve_model('')
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _merge_betas
# ---------------------------------------------------------------------------

class TestMergeBetas:
    def test_required_betas_always_present(self):
        result = _merge_betas({})
        betas = result.split(',')
        for required in REQUIRED_BETAS:
            assert required in betas, f'{required} missing from betas: {result}'

    def test_oauth_beta_always_present(self):
        result = _merge_betas({'_anthropic_beta': ['some-other-beta']})
        assert 'oauth-2025-04-20' in result.split(',')

    def test_client_betas_appended(self):
        extra = 'interleaved-thinking-2025-05-14'
        result = _merge_betas({'_anthropic_beta': [extra]})
        assert extra in result.split(',')

    def test_no_duplicates(self):
        # Even if client sends a required beta again, no duplication
        payload = {'_anthropic_beta': ['oauth-2025-04-20', 'claude-code-20250219']}
        betas = _merge_betas(payload).split(',')
        assert betas.count('oauth-2025-04-20') == 1
        assert betas.count('claude-code-20250219') == 1

    def test_required_betas_come_first(self):
        extra = 'extra-beta-2025'
        result = _merge_betas({'_anthropic_beta': [extra]})
        betas = result.split(',')
        for required in REQUIRED_BETAS:
            assert betas.index(required) < betas.index(extra)

    def test_empty_anthropic_beta_list(self):
        result = _merge_betas({'_anthropic_beta': []})
        betas = result.split(',')
        for required in REQUIRED_BETAS:
            assert required in betas

    def test_no_anthropic_beta_key(self):
        # payload with no _anthropic_beta key at all
        result = _merge_betas({'model': 'claude-opus-4-8', 'max_tokens': 10})
        for required in REQUIRED_BETAS:
            assert required in result.split(',')

    def test_clear_thinking_beta_dropped_for_haiku_adaptive(self):
        # Haiku strips adaptive thinking → clear_thinking_20251015 must be removed
        # from the beta header to avoid HTTP 400 from Anthropic.
        payload = {
            'model': 'haiku',
            'thinking': {'type': 'adaptive', 'budget_tokens': 1000},
            '_anthropic_beta': ['clear_thinking_20251015'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'clear_thinking_20251015' not in betas

    def test_clear_thinking_beta_kept_for_haiku_enabled(self):
        # Haiku supports manual extended thinking (type='enabled'), so the
        # clear_thinking beta is valid and must be preserved.
        payload = {
            'model': 'haiku',
            'thinking': {'type': 'enabled', 'budget_tokens': 1000},
            '_anthropic_beta': ['clear_thinking_20251015'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'clear_thinking_20251015' in betas

    def test_clear_thinking_beta_kept_for_sonnet_adaptive(self):
        # Sonnet supports adaptive thinking → clear_thinking_20251015 stays.
        payload = {
            'model': 'sonnet',
            'thinking': {'type': 'adaptive', 'budget_tokens': 1000},
            '_anthropic_beta': ['clear_thinking_20251015'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'clear_thinking_20251015' in betas

    # --- long-context (1m) beta strip ---

    def test_long_context_beta_dropped_for_haiku_alias(self):
        # Haiku has a 200k context window; forwarding context-1m-* causes HTTP 400.
        payload = {
            'model': 'haiku',
            '_anthropic_beta': ['context-1m-2025-08-07'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'context-1m-2025-08-07' not in betas

    def test_long_context_beta_dropped_for_haiku_resolved_id(self):
        # Strip also applies when the model is already the concrete dated ID.
        payload = {
            'model': 'claude-haiku-4-5-20251001',
            '_anthropic_beta': ['context-1m-2025-08-07'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'context-1m-2025-08-07' not in betas

    def test_long_context_beta_dropped_for_sonnet(self):
        # Long context is opus-only on the subscription; Sonnet returns HTTP 429
        # ("Usage credits are required for long context requests"), so a routed
        # opus->sonnet request must not forward the context-1m beta.
        payload = {
            'model': 'sonnet',
            '_anthropic_beta': ['context-1m-2025-08-07'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'context-1m-2025-08-07' not in betas

    def test_long_context_beta_kept_for_opus(self):
        # Opus supports 1M context; beta must pass through.
        payload = {
            'model': 'opus',
            '_anthropic_beta': ['context-1m-2025-08-07'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'context-1m-2025-08-07' in betas

    def test_long_context_beta_prefix_future_datestamp(self):
        # Gate is prefix-matched so a hypothetical future revision still drops.
        payload = {
            'model': 'haiku',
            '_anthropic_beta': ['context-1m-2026-01-01'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'context-1m-2026-01-01' not in betas

    def test_required_betas_preserved_after_long_context_strip(self):
        # Stripping context-1m must not remove REQUIRED_BETAS.
        payload = {
            'model': 'haiku',
            '_anthropic_beta': ['context-1m-2025-08-07'],
        }
        betas = _merge_betas(payload).split(',')
        for required in REQUIRED_BETAS:
            assert required in betas

    def test_unrelated_betas_preserved_alongside_long_context_strip(self):
        # Only context-1m* is removed; other client betas pass through.
        payload = {
            'model': 'haiku',
            '_anthropic_beta': ['context-1m-2025-08-07', 'some-other-beta'],
        }
        betas = _merge_betas(payload).split(',')
        assert 'context-1m-2025-08-07' not in betas
        assert 'some-other-beta' in betas


# ---------------------------------------------------------------------------
# _ensure_cc_system_prefix
# ---------------------------------------------------------------------------

class TestEnsureCCSystemPrefix:
    def test_none_system_inserts_prefix(self):
        blocks = _ensure_cc_system_prefix(None)
        assert isinstance(blocks, list)
        assert blocks[0]['type'] == 'text'
        assert blocks[0]['text'] == _CC_SYSTEM_PREFIX

    def test_empty_string_inserts_prefix(self):
        blocks = _ensure_cc_system_prefix('')
        assert blocks[0]['text'] == _CC_SYSTEM_PREFIX

    def test_string_system_gets_prefix_prepended(self):
        blocks = _ensure_cc_system_prefix('Be helpful.')
        assert blocks[0]['text'] == _CC_SYSTEM_PREFIX
        assert blocks[1]['text'] == 'Be helpful.'

    def test_list_without_prefix_gets_prefix(self):
        original = [{'type': 'text', 'text': 'Custom system prompt.'}]
        blocks = _ensure_cc_system_prefix(original)
        assert blocks[0]['text'] == _CC_SYSTEM_PREFIX
        assert blocks[1]['text'] == 'Custom system prompt.'

    def test_list_already_has_prefix_not_duplicated(self):
        original = [
            {'type': 'text', 'text': _CC_SYSTEM_PREFIX},
            {'type': 'text', 'text': 'Do more things.'},
        ]
        blocks = _ensure_cc_system_prefix(original)
        prefix_count = sum(
            1 for b in blocks
            if isinstance(b, dict) and b.get('text', '').strip().startswith(_CC_SYSTEM_PREFIX)
        )
        assert prefix_count == 1

    def test_list_prefix_with_extra_text_not_duplicated(self):
        # Some CC versions append extra text to the prefix block
        original = [
            {'type': 'text', 'text': _CC_SYSTEM_PREFIX + ' Additional context.'},
        ]
        blocks = _ensure_cc_system_prefix(original)
        # Should NOT prepend a second prefix block
        prefix_count = sum(
            1 for b in blocks
            if isinstance(b, dict) and b.get('text', '').strip().startswith(_CC_SYSTEM_PREFIX)
        )
        assert prefix_count == 1

    def test_cache_control_on_prefix_block(self):
        blocks = _ensure_cc_system_prefix(None)
        assert blocks[0].get('cache_control') == {'type': 'ephemeral'}


# ---------------------------------------------------------------------------
# _build_body
# ---------------------------------------------------------------------------

class TestBuildBody:
    def test_internal_beta_key_stripped(self):
        payload = {
            'model': 'sonnet',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            '_anthropic_beta': ['some-beta'],
        }
        body = json.loads(_build_body(payload))
        assert '_anthropic_beta' not in body

    def test_model_resolved(self):
        payload = {
            'model': 'opus',
            'max_tokens': 10,
            'messages': [],
        }
        body = json.loads(_build_body(payload))
        assert body['model'] == ANTHROPIC_MODEL_ALIASES['opus']

    def test_system_prefix_injected_when_absent(self):
        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': [],
        }
        body = json.loads(_build_body(payload))
        system = body.get('system', [])
        assert isinstance(system, list)
        assert system[0]['text'] == _CC_SYSTEM_PREFIX

    def test_system_prefix_not_duplicated(self):
        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': [],
            'system': [{'type': 'text', 'text': _CC_SYSTEM_PREFIX}],
        }
        body = json.loads(_build_body(payload))
        prefix_blocks = [
            b for b in body['system']
            if b.get('text', '').strip().startswith(_CC_SYSTEM_PREFIX)
        ]
        assert len(prefix_blocks) == 1

    def test_other_payload_fields_forwarded(self):
        payload = {
            'model': 'haiku',
            'max_tokens': 50,
            'temperature': 0.7,
            'messages': [{'role': 'user', 'content': 'Hello'}],
        }
        body = json.loads(_build_body(payload))
        assert body['max_tokens'] == 50
        assert body['temperature'] == 0.7
        assert len(body['messages']) == 1


# ---------------------------------------------------------------------------
# _build_body: model-aware effort sanitization
# ---------------------------------------------------------------------------

class TestBuildBodyEffortSanitization:
    """output_config.effort is dropped for Haiku; kept for Sonnet/Opus/Fable."""

    def _payload(self, model, output_config=None, extra=None):
        p = {
            'model': model,
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        if output_config is not None:
            p['output_config'] = output_config
        if extra:
            p.update(extra)
        return p

    def test_haiku_effort_only_stripped_entirely(self):
        payload = self._payload('haiku', output_config={'effort': 'high'})
        body = json.loads(_build_body(payload))
        assert 'output_config' not in body

    def test_haiku_effort_stripped_format_retained(self):
        payload = self._payload('haiku', output_config={'effort': 'high', 'format': {'type': 'json'}})
        body = json.loads(_build_body(payload))
        assert 'effort' not in body.get('output_config', {})
        assert body['output_config']['format'] == {'type': 'json'}

    def test_haiku_no_output_config_unchanged(self):
        payload = self._payload('haiku')
        body = json.loads(_build_body(payload))
        assert 'output_config' not in body

    def test_haiku_output_config_no_effort_unchanged(self):
        payload = self._payload('haiku', output_config={'format': {'type': 'json'}})
        body = json.loads(_build_body(payload))
        assert body['output_config'] == {'format': {'type': 'json'}}

    def test_sonnet_effort_preserved(self):
        payload = self._payload('sonnet', output_config={'effort': 'high'})
        body = json.loads(_build_body(payload))
        assert body['output_config']['effort'] == 'high'

    def test_opus_effort_preserved(self):
        payload = self._payload('opus', output_config={'effort': 'xhigh'})
        body = json.loads(_build_body(payload))
        assert body['output_config']['effort'] == 'xhigh'

    def test_fable_effort_preserved(self):
        payload = self._payload('fable', output_config={'effort': 'max'})
        body = json.loads(_build_body(payload))
        assert body['output_config']['effort'] == 'max'

    def test_haiku_caller_payload_not_mutated(self):
        """Stripping effort must not mutate the caller's nested dict."""
        oc = {'effort': 'high'}
        payload = self._payload('haiku', output_config=oc)
        _build_body(payload)
        assert oc == {'effort': 'high'}  # original dict untouched


# ---------------------------------------------------------------------------
# _build_body: model-aware adaptive-thinking sanitization
# ---------------------------------------------------------------------------

class TestBuildBodyThinkingSanitization:
    """thinking.type='adaptive' is dropped for Haiku; preserved elsewhere.

    Haiku rejects adaptive thinking with HTTP 400 but still supports manual
    extended thinking (type='enabled'); Sonnet/Opus/Fable accept adaptive.
    """

    def _payload(self, model, thinking=None, extra=None):
        p = {
            'model': model,
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        if thinking is not None:
            p['thinking'] = thinking
        if extra:
            p.update(extra)
        return p

    def test_haiku_adaptive_thinking_stripped(self):
        payload = self._payload('haiku', thinking={'type': 'adaptive'})
        body = json.loads(_build_body(payload))
        assert 'thinking' not in body

    def test_haiku_enabled_thinking_preserved(self):
        payload = self._payload('haiku', thinking={'type': 'enabled', 'budget_tokens': 1024})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'enabled', 'budget_tokens': 1024}

    def test_haiku_disabled_thinking_preserved(self):
        payload = self._payload('haiku', thinking={'type': 'disabled'})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'disabled'}

    def test_sonnet_adaptive_thinking_preserved(self):
        payload = self._payload('sonnet', thinking={'type': 'adaptive'})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'adaptive'}

    def test_opus_adaptive_thinking_preserved(self):
        payload = self._payload('opus', thinking={'type': 'adaptive'})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'adaptive'}

    def test_fable_adaptive_thinking_preserved(self):
        payload = self._payload('fable', thinking={'type': 'adaptive'})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'adaptive'}

    def test_fable_disabled_thinking_stripped(self):
        payload = self._payload('fable', thinking={'type': 'disabled'})
        body = json.loads(_build_body(payload))
        assert 'thinking' not in body

    def test_fable_enabled_thinking_preserved(self):
        payload = self._payload('fable', thinking={'type': 'enabled', 'budget_tokens': 1024})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'enabled', 'budget_tokens': 1024}

    def test_sonnet_disabled_thinking_preserved(self):
        payload = self._payload('sonnet', thinking={'type': 'disabled'})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'disabled'}

    def test_opus_disabled_thinking_preserved(self):
        payload = self._payload('opus', thinking={'type': 'disabled'})
        body = json.loads(_build_body(payload))
        assert body['thinking'] == {'type': 'disabled'}

    def test_haiku_caller_payload_not_mutated(self):
        thinking = {'type': 'adaptive'}
        payload = self._payload('haiku', thinking=thinking)
        _build_body(payload)
        assert thinking == {'type': 'adaptive'}  # original untouched


# ---------------------------------------------------------------------------
# _build_body: Codex synthetic thinking-block stripping
# ---------------------------------------------------------------------------

class TestBuildBodyStripCodexThinking:
    """Thinking blocks whose signature was minted by the Codex mapper (codexenc:
    prefix) must be stripped before forwarding to Anthropic, which validates
    signatures cryptographically and rejects foreign ones (HTTP 400).
    """

    def _codex_sig(self):
        import base64
        from anthproxy.mapper.common import CODEX_REASONING_SIG_PREFIX
        raw = json.dumps({'id': 'rs_x', 'enc': 'OPAQUE=='}).encode()
        return CODEX_REASONING_SIG_PREFIX + base64.b64encode(raw).decode()

    def _payload(self, messages):
        return {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': messages,
        }

    def test_codex_thinking_block_stripped(self):
        sig = self._codex_sig()
        payload = self._payload([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'ponder', 'signature': sig},
                {'type': 'text', 'text': 'answer'},
            ]},
            {'role': 'user', 'content': 'follow up'},
        ])
        body = json.loads(_build_body(payload))
        asst_content = body['messages'][1]['content']
        assert all(b.get('type') != 'thinking' for b in asst_content)
        assert any(b.get('type') == 'text' for b in asst_content)

    def test_genuine_anthropic_signature_preserved(self):
        payload = self._payload([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'think', 'signature': 'genuineAnthropicSig123'},
                {'type': 'text', 'text': 'answer'},
            ]},
        ])
        body = json.loads(_build_body(payload))
        asst_content = body['messages'][1]['content']
        thinking_blocks = [b for b in asst_content if b.get('type') == 'thinking']
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]['signature'] == 'genuineAnthropicSig123'

    def test_original_payload_not_mutated(self):
        sig = self._codex_sig()
        original_content = [
            {'type': 'thinking', 'thinking': 'x', 'signature': sig},
            {'type': 'text', 'text': 'y'},
        ]
        payload = self._payload([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': original_content},
        ])
        _build_body(payload)
        assert len(original_content) == 2  # not mutated

    def test_openrouter_tagged_thinking_block_stripped(self):
        # OpenRouter stamps `or:` onto its response signatures; on an
        # OpenRouter→Anthropic switch those must be stripped exactly like codex
        # ones, otherwise Anthropic rejects with HTTP 400 'Invalid signature'.
        from anthproxy.mapper.common import OPENROUTER_REASONING_SIG_PREFIX
        sig = OPENROUTER_REASONING_SIG_PREFIX + 'foreignUpstreamSig=='
        payload = self._payload([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'ponder', 'signature': sig},
                {'type': 'text', 'text': 'answer'},
            ]},
            {'role': 'user', 'content': 'follow up'},
        ])
        body = json.loads(_build_body(payload))
        asst_content = body['messages'][1]['content']
        assert all(b.get('type') != 'thinking' for b in asst_content)
        assert any(b.get('type') == 'text' for b in asst_content)


# ---------------------------------------------------------------------------
# _build_body: clear_thinking context-management strategy sanitization
# ---------------------------------------------------------------------------

class TestBuildBodyClearThinkingStrategy:
    """clear_thinking* context_management edits are dropped whenever thinking is
    not active (e.g. adaptive thinking stripped for Haiku); kept when thinking is
    active.  Pairs with the clear_thinking beta strip in _merge_betas.
    """

    def _payload(self, model, thinking, edits):
        return {
            'model': model,
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'thinking': thinking,
            'context_management': {'edits': edits},
        }

    def test_haiku_adaptive_strategy_stripped_block_dropped(self):
        # Haiku strips adaptive thinking → orphaned clear_thinking edit removed.
        # It was the only edit, so context_management is dropped entirely.
        payload = self._payload(
            'haiku', {'type': 'adaptive'},
            [{'type': 'clear_thinking_20251015'}],
        )
        body = json.loads(_build_body(payload))
        assert 'thinking' not in body
        assert 'context_management' not in body

    def test_haiku_strategy_stripped_other_edits_retained(self):
        payload = self._payload(
            'haiku', {'type': 'adaptive'},
            [{'type': 'clear_thinking_20251015'}, {'type': 'clear_tool_uses_20250919'}],
        )
        body = json.loads(_build_body(payload))
        kept = [e['type'] for e in body['context_management']['edits']]
        assert kept == ['clear_tool_uses_20250919']

    def test_haiku_enabled_thinking_strategy_kept(self):
        # Manual extended thinking stays active on Haiku → strategy preserved.
        payload = self._payload(
            'haiku', {'type': 'enabled', 'budget_tokens': 1024},
            [{'type': 'clear_thinking_20251015'}],
        )
        body = json.loads(_build_body(payload))
        assert body['context_management']['edits'] == [{'type': 'clear_thinking_20251015'}]

    def test_sonnet_adaptive_strategy_kept(self):
        # Sonnet keeps adaptive thinking → strategy stays valid.
        payload = self._payload(
            'sonnet', {'type': 'adaptive'},
            [{'type': 'clear_thinking_20251015'}],
        )
        body = json.loads(_build_body(payload))
        assert body['context_management']['edits'] == [{'type': 'clear_thinking_20251015'}]

    def test_haiku_no_thinking_strategy_stripped(self):
        # No thinking at all → thinking inactive → clear_thinking edit removed.
        payload = {
            'model': 'haiku',
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'context_management': {'edits': [{'type': 'clear_thinking_20251015'}]},
        }
        body = json.loads(_build_body(payload))
        assert 'context_management' not in body

    def test_caller_payload_not_mutated(self):
        edits = [{'type': 'clear_thinking_20251015'}]
        payload = self._payload('haiku', {'type': 'adaptive'}, edits)
        _build_body(payload)
        assert edits == [{'type': 'clear_thinking_20251015'}]  # original untouched


# ---------------------------------------------------------------------------
# _build_body: inline role:'system' message folding
# ---------------------------------------------------------------------------

class TestBuildBodyInlineSystemFold:
    """role:'system' messages are folded into the top-level system field (some
    models reject a system role in messages[]); the CC prefix block stays first.
    """

    def _payload(self, messages):
        return {'model': 'sonnet', 'max_tokens': 10, 'messages': messages}

    def test_inline_system_removed_from_messages(self):
        payload = self._payload([
            {'role': 'system', 'content': 'Be terse.'},
            {'role': 'user', 'content': 'Hi'},
        ])
        body = json.loads(_build_body(payload))
        assert all(m['role'] != 'system' for m in body['messages'])
        assert body['messages'] == [{'role': 'user', 'content': 'Hi'}]

    def test_inline_system_text_appended_to_system(self):
        payload = self._payload([
            {'role': 'system', 'content': 'Be terse.'},
            {'role': 'user', 'content': 'Hi'},
        ])
        body = json.loads(_build_body(payload))
        texts = [b['text'] for b in body['system'] if b.get('type') == 'text']
        assert 'Be terse.' in texts

    def test_cc_prefix_stays_first(self):
        payload = self._payload([
            {'role': 'system', 'content': 'Be terse.'},
            {'role': 'user', 'content': 'Hi'},
        ])
        body = json.loads(_build_body(payload))
        assert body['system'][0]['text'].strip().startswith(_CC_SYSTEM_PREFIX)

    def test_inline_system_block_content_flattened(self):
        payload = self._payload([
            {'role': 'system', 'content': [{'type': 'text', 'text': 'Block sys.'}]},
            {'role': 'user', 'content': 'Hi'},
        ])
        body = json.loads(_build_body(payload))
        texts = [b['text'] for b in body['system'] if b.get('type') == 'text']
        assert 'Block sys.' in texts

    def test_empty_inline_system_dropped_without_appending(self):
        # An empty system message is removed from messages but adds no system block.
        payload = self._payload([
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': 'Hi'},
        ])
        body = json.loads(_build_body(payload))
        assert all(m['role'] != 'system' for m in body['messages'])
        # Only the CC prefix block remains in system.
        assert body['system'] == [
            {'type': 'text', 'text': _CC_SYSTEM_PREFIX, 'cache_control': {'type': 'ephemeral'}}
        ]

    def test_no_system_message_messages_unchanged(self):
        payload = self._payload([{'role': 'user', 'content': 'Hi'}])
        body = json.loads(_build_body(payload))
        assert body['messages'] == [{'role': 'user', 'content': 'Hi'}]

    def test_caller_payload_not_mutated(self):
        messages = [
            {'role': 'system', 'content': 'Be terse.'},
            {'role': 'user', 'content': 'Hi'},
        ]
        payload = self._payload(messages)
        _build_body(payload)
        assert messages[0] == {'role': 'system', 'content': 'Be terse.'}
        assert len(messages) == 2  # original list untouched


# ---------------------------------------------------------------------------
# _build_body: model-aware sampling-control sanitization
# ---------------------------------------------------------------------------

class TestBuildBodySamplingSanitization:
    """temperature/top_p/top_k are dropped for fixed-sampling models.

    Opus 4.7+/4.8 and Fable reject non-default sampling parameters with HTTP
    400; older Opus, Sonnet, and Haiku keep them.
    """

    def _payload(self, model, sampling=None):
        p = {
            'model': model,
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        if sampling:
            p.update(sampling)
        return p

    _ALL = {'temperature': 0.7, 'top_p': 0.9, 'top_k': 40}

    def test_opus_4_8_sampling_stripped(self):
        body = json.loads(_build_body(self._payload('opus', self._ALL)))
        assert 'temperature' not in body
        assert 'top_p' not in body
        assert 'top_k' not in body

    def test_opus_4_8_full_id_sampling_stripped(self):
        body = json.loads(_build_body(self._payload('claude-opus-4-8', self._ALL)))
        assert not ({'temperature', 'top_p', 'top_k'} & body.keys())

    def test_opus_4_7_sampling_stripped(self):
        body = json.loads(_build_body(self._payload('claude-opus-4-7', self._ALL)))
        assert not ({'temperature', 'top_p', 'top_k'} & body.keys())

    def test_fable_sampling_stripped(self):
        body = json.loads(_build_body(self._payload('fable', self._ALL)))
        assert not ({'temperature', 'top_p', 'top_k'} & body.keys())

    def test_opus_4_8_none_values_still_omitted(self):
        payload = self._payload('opus', {'temperature': None, 'top_p': None, 'top_k': None})
        body = json.loads(_build_body(payload))
        assert 'temperature' not in body
        assert 'top_p' not in body
        assert 'top_k' not in body

    def test_opus_4_6_sampling_preserved(self):
        body = json.loads(_build_body(self._payload('claude-opus-4-6', self._ALL)))
        assert body['temperature'] == 0.7
        assert body['top_p'] == 0.9
        assert body['top_k'] == 40

    def test_sonnet_sampling_preserved(self):
        body = json.loads(_build_body(self._payload('sonnet', {'temperature': 0.7})))
        assert body['temperature'] == 0.7

    def test_haiku_sampling_preserved(self):
        body = json.loads(_build_body(self._payload('haiku', {'temperature': 0.7})))
        assert body['temperature'] == 0.7


# ---------------------------------------------------------------------------
# _normalize_cache_ttl_ordering
# ---------------------------------------------------------------------------

def _ttl(block):
    return block.get('cache_control', {}).get('ttl')


class TestNormalizeCacheTtlOrdering:
    def test_injected_5m_prefix_before_client_1h_system_block_promoted(self):
        # Non-CC caller: proxy prepends a default (5m) prefix ahead of a client
        # system block carrying ttl='1h' -> the 1h would land at system[2].
        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'system': [
                {'type': 'text', 'text': 'context'},
                {'type': 'text', 'text': 'cached',
                 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}},
            ],
            'messages': [],
        }
        body = json.loads(_build_body(payload))
        system = body['system']
        # Every breakpoint at or before the 1h block must be 1h.
        ttls = [_ttl(b) for b in system if 'cache_control' in b]
        assert '5m' not in [t or '5m' for t in ttls[:ttls.index('1h') + 1]]
        # The injected prefix breakpoint is promoted to 1h.
        assert system[0]['cache_control']['ttl'] == '1h'

    def test_tool_1h_after_injected_5m_is_valid_ordering(self):
        # tools precede system; a 1h tool ahead of injected 5m system blocks is
        # already valid and must be left untouched.
        body = {
            'tools': [{'name': 't', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
            'system': [{'type': 'text', 'text': 'x', 'cache_control': {'type': 'ephemeral'}}],
            'messages': [],
        }
        out = _normalize_cache_ttl_ordering(body)
        assert out['tools'][0]['cache_control']['ttl'] == '1h'
        # System breakpoint stays 5m (it legitimately follows the 1h tool).
        assert out['system'][0]['cache_control'].get('ttl') is None

    def test_system_5m_before_1h_promoted(self):
        body = {
            'system': [
                {'type': 'text', 'text': 'a', 'cache_control': {'type': 'ephemeral'}},
                {'type': 'text', 'text': 'b', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}},
            ],
            'messages': [],
        }
        out = _normalize_cache_ttl_ordering(body)
        assert out['system'][0]['cache_control']['ttl'] == '1h'
        assert out['system'][1]['cache_control']['ttl'] == '1h'

    def test_all_5m_unchanged(self):
        body = {
            'system': [{'type': 'text', 'text': 'a', 'cache_control': {'type': 'ephemeral'}}],
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}},
                ]},
            ],
        }
        out = _normalize_cache_ttl_ordering(body)
        assert out is body
        assert out['system'][0]['cache_control'].get('ttl') is None

    def test_trailing_5m_after_last_1h_stays_5m(self):
        body = {
            'system': [{'type': 'text', 'text': 'a', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'hi', 'cache_control': {'type': 'ephemeral'}},
                ]},
            ],
        }
        out = _normalize_cache_ttl_ordering(body)
        # The message breakpoint follows the last 1h block -> stays 5m.
        assert out['messages'][0]['content'][0]['cache_control'].get('ttl') is None

    def test_input_payload_not_mutated(self):
        sys_blocks = [
            {'type': 'text', 'text': 'a', 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': 'b', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}},
        ]
        body = {'system': sys_blocks, 'messages': []}
        _normalize_cache_ttl_ordering(body)
        # Original block dicts untouched.
        assert sys_blocks[0]['cache_control'] == {'type': 'ephemeral'}


# ---------------------------------------------------------------------------
# needs_access_refresh (anthropic_auth)
# ---------------------------------------------------------------------------

class TestNeedsAccessRefresh:
    def test_expires_at_in_past_needs_refresh(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref',
                 'expires_at': time.time() - 10}
        assert needs_access_refresh(creds) is True

    def test_expires_at_far_future_no_refresh(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref',
                 'expires_at': time.time() + 3600}
        assert needs_access_refresh(creds) is False

    def test_expires_at_within_window_needs_refresh(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref',
                 'expires_at': time.time() + ACCESS_REFRESH_WINDOW_SECS - 10}
        assert needs_access_refresh(creds) is True

    def test_expires_at_outside_window_no_refresh(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref',
                 'expires_at': time.time() + ACCESS_REFRESH_WINDOW_SECS + 60}
        assert needs_access_refresh(creds) is False

    def test_no_expires_at_uses_last_refresh_fresh(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref',
                 'expires_at': None, 'last_refresh': time.time() - 3600}
        assert needs_access_refresh(creds) is False

    def test_no_expires_at_uses_last_refresh_stale(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref',
                 'expires_at': None, 'last_refresh': time.time() - 9 * 86400}
        assert needs_access_refresh(creds) is True

    def test_no_expires_at_no_last_refresh_needs_refresh(self):
        creds = {'access_token': 'tok', 'refresh_token': 'ref'}
        assert needs_access_refresh(creds) is True


# ---------------------------------------------------------------------------
# load_credentials / _write_auth
# ---------------------------------------------------------------------------

class TestLoadCredentials:
    def test_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d) / 'auth_home'
            home.mkdir()
            assert load_credentials(home) is None

    def test_returns_none_for_empty_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d)
            _write_auth(home, {})
            assert load_credentials(home) is None

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d)
            data = {
                'access_token': 'sk-ant-oat-test',
                'refresh_token': 'ref-test',
                'expires_at': time.time() + 3600,
                'account_uuid': 'uuid-abc',
                'email': 'test@example.com',
                'last_refresh': time.time(),
            }
            _write_auth(home, data)
            creds = load_credentials(home)
            assert creds is not None
            assert creds['access_token'] == 'sk-ant-oat-test'
            assert creds['refresh_token'] == 'ref-test'
            assert creds['account_uuid'] == 'uuid-abc'

    def test_auth_file_mode_600(self):
        import stat
        with tempfile.TemporaryDirectory() as d:
            home = pathlib.Path(d)
            data = {'access_token': 'tok', 'refresh_token': 'ref',
                    'expires_at': None, 'account_uuid': None,
                    'email': None, 'last_refresh': None}
            _write_auth(home, data)
            auth_file = _auth_file(home)
            mode = stat.S_IMODE(auth_file.stat().st_mode)
            assert mode == 0o600


# ---------------------------------------------------------------------------
# _pkce
# ---------------------------------------------------------------------------

class TestPkce:
    def test_lengths(self):
        import hashlib
        import base64
        verifier, challenge = _pkce()
        # challenge must equal BASE64URL(SHA256(verifier))
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b'=').decode()
        )
        assert challenge == expected

    def test_uniqueness(self):
        v1, c1 = _pkce()
        v2, c2 = _pkce()
        assert v1 != v2
        assert c1 != c2


# ---------------------------------------------------------------------------
# _anthropic_home
# ---------------------------------------------------------------------------

class TestAnthropicHome:
    def test_default_is_dot_anthropic(self):
        import os
        # Unset env vars for a clean test
        env_backup = os.environ.pop('ANTHROPIC_HOME', None)
        try:
            home = _anthropic_home(None)
            assert home == pathlib.Path.home() / '.anthropic'
        finally:
            if env_backup is not None:
                os.environ['ANTHROPIC_HOME'] = env_backup

    def test_config_override(self):
        class FakeConfig:
            anthropic_home = '/tmp/custom_anthproxy_home'
        home = _anthropic_home(FakeConfig())
        assert home == pathlib.Path('/tmp/custom_anthproxy_home')


# ---------------------------------------------------------------------------
# AnthropicBackend.parse_credentials
# ---------------------------------------------------------------------------

class TestAnthropicBackendParseCredentials:
    def test_ignores_api_key(self):
        backend = AnthropicBackend()
        assert backend.parse_credentials('sk-ant-api-anything') == {}
        assert backend.parse_credentials('') == {}


# ---------------------------------------------------------------------------
# _handle_error_response
# ---------------------------------------------------------------------------

class TestHandleErrorResponse:
    def _call(self, status, body):
        _handle_error_response(status, json.dumps(body).encode())

    def test_400_invalid_request(self):
        with pytest.raises(AnthropicRequestError) as exc:
            self._call(400, {'error': {'message': 'bad param', 'type': 'invalid_request_error'}})
        assert exc.value.status_code == 400

    def test_401_authentication_error(self):
        with pytest.raises(AnthropicRequestError) as exc:
            self._call(401, {'error': {'message': 'auth fail'}})
        assert exc.value.status_code == 401
        assert exc.value.error_type == 'authentication_error'

    def test_403_permission_error(self):
        with pytest.raises(AnthropicRequestError) as exc:
            self._call(403, {'error': {'message': 'forbidden'}})
        assert exc.value.status_code == 403
        assert exc.value.error_type == 'permission_error'

    def test_429_rate_limit_error(self):
        with pytest.raises(AnthropicRequestError) as exc:
            self._call(429, {'error': {'message': 'too many'}})
        assert exc.value.status_code == 429
        assert exc.value.error_type == 'rate_limit_error'

    def test_502_api_error(self):
        with pytest.raises(AnthropicRequestError) as exc:
            self._call(500, {'error': {'message': 'server error'}})
        assert exc.value.status_code == 502
        assert exc.value.error_type == 'api_error'


# ---------------------------------------------------------------------------
# _send_with_retries: thinking-signature 400 recovery
# ---------------------------------------------------------------------------

class TestSendWithRetriesThinkingSignatureRecovery:
    """When Anthropic returns HTTP 400 "Invalid signature in thinking block"
    (caused by model-tier routing switching models between turns — opus
    signatures are invalid for sonnet/haiku), _send_with_retries must strip
    all thinking blocks from history and retry exactly once.
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
                'message': 'messages.1.content.67: Invalid `signature` in `thinking` block',
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

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_strips_thinking_and_retries_on_400_signature_error(self, mock_get_access, mock_conn_cls):
        mock_get_access.return_value = ('token', None)
        conn = MagicMock()
        # First call → 400 "Invalid signature"; second call → 200 OK
        conn.getresponse.side_effect = [
            self._make_response(400, self._thinking_400_body()),
            self._make_response(200, self._ok_response()),
        ]
        mock_conn_cls.return_value = conn

        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'opus reasoning', 'signature': 'rawOpusSig=='},
                    {'type': 'text', 'text': 'response'},
                ]},
                {'role': 'user', 'content': 'follow up'},
            ],
        }
        from anthproxy.anthropic.backend import _send_with_retries
        result_conn, result_resp = _send_with_retries(payload, MagicMock(), threading.Lock(), stream=False)
        assert result_resp.status == 200
        # Two HTTP requests were made: original + thinking-stripped retry
        assert conn.request.call_count == 2
        # The second request body must not contain the thinking block
        # body= is passed as a keyword argument to conn.request
        second_call = conn.request.call_args_list[1]
        second_body = json.loads(second_call.kwargs.get('body') or second_call[1]['body'])
        asst_blocks = second_body['messages'][1]['content']
        assert all(b.get('type') != 'thinking' for b in asst_blocks)
        assert any(b.get('type') == 'text' for b in asst_blocks)

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_retries_at_most_once_on_repeated_signature_error(self, mock_get_access, mock_conn_cls):
        mock_get_access.return_value = ('token', None)
        conn = MagicMock()
        # Both calls return 400 "Invalid signature" — should raise after 2 attempts
        conn.getresponse.side_effect = [
            self._make_response(400, self._thinking_400_body()),
            self._make_response(400, self._thinking_400_body()),
        ]
        mock_conn_cls.return_value = conn

        payload = {
            'model': 'sonnet',
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
        from anthproxy.anthropic.backend import _send_with_retries
        with pytest.raises(AnthropicRequestError) as exc:
            _send_with_retries(payload, MagicMock(), threading.Lock(), stream=False)
        assert exc.value.status_code == 400
        assert conn.request.call_count == 2

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_no_retry_when_no_thinking_blocks_to_strip(self, mock_get_access, mock_conn_cls):
        # No thinking blocks in history → the 400 is not a signature issue we can fix
        mock_get_access.return_value = ('token', None)
        conn = MagicMock()
        conn.getresponse.return_value = self._make_response(400, self._thinking_400_body())
        mock_conn_cls.return_value = conn

        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'no thinking here'}]},
            ],
        }
        from anthproxy.anthropic.backend import _send_with_retries
        with pytest.raises(AnthropicRequestError) as exc:
            _send_with_retries(payload, MagicMock(), threading.Lock(), stream=False)
        assert exc.value.status_code == 400
        assert conn.request.call_count == 1

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_strips_redacted_thinking_on_data_error(self, mock_get_access, mock_conn_cls):
        # "Invalid `data` in `redacted_thinking` block" is the sister error;
        # the retry must also strip redacted_thinking blocks.
        mock_get_access.return_value = ('token', None)
        conn = MagicMock()
        redacted_400 = {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'messages.1.content.68: Invalid `data` in `redacted_thinking` block',
            },
        }
        conn.getresponse.side_effect = [
            self._make_response(400, redacted_400),
            self._make_response(200, self._ok_response()),
        ]
        mock_conn_cls.return_value = conn

        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'redacted_thinking', 'data': 'opaque=='},
                    {'type': 'text', 'text': 'response'},
                ]},
                {'role': 'user', 'content': 'follow up'},
            ],
        }
        from anthproxy.anthropic.backend import _send_with_retries
        _, result_resp = _send_with_retries(payload, MagicMock(), threading.Lock(), stream=False)
        assert result_resp.status == 200
        assert conn.request.call_count == 2
        second_call = conn.request.call_args_list[1]
        second_body = json.loads(second_call.kwargs.get('body') or second_call[1]['body'])
        asst_blocks = second_body['messages'][1]['content']
        assert all(b.get('type') != 'redacted_thinking' for b in asst_blocks)
        assert any(b.get('type') == 'text' for b in asst_blocks)

    @patch('anthproxy.anthropic.backend.http.client.HTTPSConnection')
    @patch('anthproxy.anthropic.backend.get_access')
    def test_non_signature_400_not_retried(self, mock_get_access, mock_conn_cls):
        mock_get_access.return_value = ('token', None)
        conn = MagicMock()
        conn.getresponse.return_value = self._make_response(400, {
            'error': {'type': 'invalid_request_error', 'message': 'some other 400 error'},
        })
        mock_conn_cls.return_value = conn

        payload = {
            'model': 'sonnet',
            'max_tokens': 10,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'reason', 'signature': 'sig=='},
                    {'type': 'text', 'text': 'response'},
                ]},
            ],
        }
        from anthproxy.anthropic.backend import _send_with_retries
        with pytest.raises(AnthropicRequestError) as exc:
            _send_with_retries(payload, MagicMock(), threading.Lock(), stream=False)
        assert exc.value.status_code == 400
        assert conn.request.call_count == 1


# ---------------------------------------------------------------------------
# _max_weekly_utilization helper + five_hour_status weekly plumbing
# ---------------------------------------------------------------------------

def _weekly_usage(utilization, resets_at='2099-01-01T00:00:00Z'):
    """Minimal weekly-window dict."""
    if utilization is None:
        return {}
    return {'utilization': utilization, 'resets_at': resets_at}


def _five_hour_window(utilization=50.0, resets_at='2099-01-01T00:00:00Z'):
    """Minimal five-hour window dict that yields available=True."""
    return {'utilization': utilization, 'resets_at': resets_at}


class TestMaxWeeklyUtilization:
    """Unit tests for _max_weekly_utilization (pure function)."""

    def test_all_three_windows_returns_max(self):
        usage = {
            'seven_day': {'utilization': 30.0},
            'seven_day_sonnet': {'utilization': 55.0},
            'seven_day_opus': {'utilization': 90.0},
        }
        assert _max_weekly_utilization(usage) == 90.0

    def test_single_valid_seven_day_returns_that_value(self):
        usage = {'seven_day': {'utilization': 42.0}}
        assert _max_weekly_utilization(usage) == 42.0

    def test_opus_is_binding_window(self):
        usage = {
            'seven_day': {'utilization': 20.0},
            'seven_day_opus': {'utilization': 80.0},
        }
        assert _max_weekly_utilization(usage) == 80.0

    def test_window_with_none_utilization_skipped(self):
        # seven_day_sonnet is present but has no numeric utilization
        usage = {
            'seven_day': {'utilization': 60.0},
            'seven_day_sonnet': {},  # utilization key missing
        }
        assert _max_weekly_utilization(usage) == 60.0

    def test_window_with_explicit_none_utilization_skipped(self):
        usage = {
            'seven_day': {'utilization': 60.0},
            'seven_day_sonnet': {'utilization': None},
        }
        assert _max_weekly_utilization(usage) == 60.0

    def test_no_weekly_windows_returns_none(self):
        assert _max_weekly_utilization({}) is None
        assert _max_weekly_utilization({'five_hour': {'utilization': 50.0}}) is None

    def test_lone_seven_day_missing_utilization_returns_none(self):
        # Behavioral delta vs. old code: old yielded 0.0; new yields None.
        # A window with no parseable utilization should not be treated as 0% consumed.
        usage = {'seven_day': {}}  # utilization key absent
        assert _max_weekly_utilization(usage) is None

    def test_lone_seven_day_explicit_none_utilization_returns_none(self):
        # Same intentional change: old code yielded 0.0; new yields None.
        usage = {'seven_day': {'utilization': None}}
        assert _max_weekly_utilization(usage) is None

    def test_malformed_value_skipped_no_raise(self):
        usage = {
            'seven_day': {'utilization': 'bad'},
            'seven_day_sonnet': {'utilization': 77.0},
        }
        assert _max_weekly_utilization(usage) == 77.0

    def test_non_dict_window_ignored(self):
        usage = {
            'seven_day': 'not a dict',
            'seven_day_sonnet': {'utilization': 25.0},
        }
        assert _max_weekly_utilization(usage) == 25.0


class TestFiveHourStatusWeekly:
    """five_hour_status correctly populates weekly_utilization via _max_weekly_utilization."""

    def _backend_with_usage(self, usage_dict):
        backend = AnthropicBackend()
        backend.get_usage = lambda config: usage_dict
        return backend

    def _status(self, usage_dict, five_hour_util=50.0):
        """Run five_hour_status with a crafted usage payload."""
        usage = dict(usage_dict)
        usage.setdefault('five_hour', _five_hour_window(five_hour_util))
        backend = self._backend_with_usage(usage)
        return backend.five_hour_status(None)

    def test_all_three_windows_max_is_used(self):
        st = self._status({
            'seven_day': {'utilization': 30.0},
            'seven_day_sonnet': {'utilization': 55.0},
            'seven_day_opus': {'utilization': 90.0},
        })
        assert st.weekly_utilization == 90.0

    def test_lone_valid_seven_day_preserved(self):
        st = self._status({'seven_day': {'utilization': 42.0}})
        assert st.weekly_utilization == 42.0

    def test_opus_binding_window(self):
        st = self._status({
            'seven_day': {'utilization': 30.0},
            'seven_day_sonnet': {'utilization': 40.0},
            'seven_day_opus': {'utilization': 80.0},
        })
        assert st.weekly_utilization == 80.0

    def test_window_with_none_utilization_skipped(self):
        st = self._status({
            'seven_day': {'utilization': 60.0},
            'seven_day_sonnet': {'utilization': None},
        })
        assert st.weekly_utilization == 60.0

    def test_lone_seven_day_missing_utilization_returns_none(self):
        # Intentional change: old code returned 0.0 here; new returns None.
        st = self._status({'seven_day': {}})
        assert st.weekly_utilization is None

    def test_lone_seven_day_explicit_none_utilization_returns_none(self):
        # Same intentional change.
        st = self._status({'seven_day': {'utilization': None}})
        assert st.weekly_utilization is None

    def test_no_weekly_windows_returns_none(self):
        st = self._status({})
        assert st.weekly_utilization is None

    def test_five_hour_window_drives_available_and_utilization(self):
        # Sanity: weekly computation does not interfere with five-hour fields.
        st = self._status({
            'seven_day': {'utilization': 90.0},
            'seven_day_opus': {'utilization': 95.0},
        }, five_hour_util=75.0)
        assert st.available is True
        assert st.utilization == 75.0
        assert st.weekly_utilization == 95.0
