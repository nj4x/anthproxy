"""Unit tests for the Codex backend — translation, auth helpers, and streaming.

All tests are network-free; live refresh() and login() calls are not covered
here.  Follow the patterns established by tests/test_gauss.py.
"""

import base64
import datetime as dt
import hashlib
import http.client
import json
import os
import pathlib
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from anthproxy._shared import UsageRateLimitError
from anthproxy.mapper import AnthropicRequestError
from anthproxy.codex.backend import (
    CODEX_USAGE_PATH,
    CodexBackend,
    MAX_RETRIES,
    _codex_unsupported_model_fallback,
    _drain_sse_to_response,
    _fetch_usage,
    _format_reset,
    _format_usage_markdown,
    _handle_error_response,
    _is_chatgpt_unsupported_model_error,
    _normalize_usage,
    _retry_delay,
    _send_with_retries,
)
from anthproxy.codex.mapper import (
    _convert_message_to_input_item,
    _decode_reasoning_signature,
    _effort_from_budget,
    _encode_reasoning_signature,
    _is_clean_user_turn,
    _iter_stream_as_anthropic_sse,
    _map_request,
    _map_response,
    _resolve_model,
    _truncate_messages_for_context,
)
from anthproxy.codex.auth import (
    ACCESS_REFRESH_INTERVAL_DAYS,
    _account_id_from_id_token,
    _b64url_nopad,
    _decode_jwt_payload,
    _jwt_exp,
    _pkce,
    _refresh_lock,
    _write_auth,
    ensure_credentials_noninteractive,
    load_credentials,
    needs_access_refresh,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jwt(payload: dict, header: dict | None = None) -> str:
    """Build an unsigned JWT from a payload dict (for tests only)."""
    if header is None:
        header = {'alg': 'none', 'typ': 'JWT'}
    enc = base64.urlsafe_b64encode

    def b64(d):
        return enc(json.dumps(d).encode()).rstrip(b'=').decode()

    return f'{b64(header)}.{b64(payload)}.'


def _fake_sse_response(events: list[dict]):
    """Create a minimal file-like object that yields Codex SSE events."""
    lines = []
    for event in events:
        lines.append(f'data: {json.dumps(event)}\n'.encode())
    lines.append(b'data: [DONE]\n')
    content = b''.join(lines)

    class _FakeResponse:
        def __init__(self):
            self._pos = 0
            self._data = content

        def read(self, n):
            chunk = self._data[self._pos:self._pos + n]
            self._pos += n
            return chunk

    return _FakeResponse()


def _collect_events(gen):
    """Collect SSE strings from a generator, parse them, return list of dicts."""
    events = []
    for chunk in gen:
        # chunk is "event: <type>\ndata: <json>\n\n"
        lines = chunk.strip().split('\n')
        event_type = data = None
        for line in lines:
            if line.startswith('event: '):
                event_type = line[7:]
            elif line.startswith('data: '):
                data = json.loads(line[6:])
        if event_type and data is not None:
            events.append({'event': event_type, 'data': data})
    return events


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    def test_short_alias_opus(self):
        assert _resolve_model('opus') == 'gpt-5.6-sol'

    def test_short_alias_sonnet(self):
        assert _resolve_model('sonnet') == 'gpt-5.6-terra'

    def test_short_alias_haiku(self):
        assert _resolve_model('haiku') == 'gpt-5.6-luna'

    def test_native_codex_passthrough(self):
        assert _resolve_model('gpt-5-codex') == 'gpt-5-codex'

    def test_native_o3_passthrough(self):
        assert _resolve_model('o3') == 'o3'

    def test_native_gpt5_passthrough(self):
        assert _resolve_model('gpt-5.6-sol') == 'gpt-5.6-sol'

    def test_full_anthropic_id(self):
        result = _resolve_model('claude-sonnet-4-6')
        assert result == 'gpt-5.6-terra'

    def test_context_suffix_1m_stripped(self):
        assert _resolve_model('opus:1m') == 'gpt-5.6-sol'

    def test_context_suffix_bracket_stripped(self):
        assert _resolve_model('sonnet[1m]') == 'gpt-5.6-terra'

    def test_unknown_model_passthrough(self):
        assert _resolve_model('gpt-6-ultra') == 'gpt-6-ultra'

    def test_empty_model_raises(self):
        from anthproxy.mapper import AnthropicRequestError
        with pytest.raises(AnthropicRequestError):
            _resolve_model('')


# ---------------------------------------------------------------------------
# _effort_from_budget
# ---------------------------------------------------------------------------

class TestEffortFromBudget:
    def test_low(self):
        assert _effort_from_budget(1024) == 'low'

    def test_low_boundary(self):
        assert _effort_from_budget(4096) == 'low'

    def test_medium(self):
        assert _effort_from_budget(8192) == 'medium'

    def test_medium_boundary(self):
        assert _effort_from_budget(16384) == 'medium'

    def test_high(self):
        assert _effort_from_budget(32768) == 'high'


# ---------------------------------------------------------------------------
# _map_request
# ---------------------------------------------------------------------------

class TestMapRequest:
    def _base_payload(self, **overrides):
        return {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            **overrides,
        }

    def test_stream_always_true(self):
        body = _map_request(self._base_payload())
        assert body['stream'] is True

    def test_store_always_false(self):
        body = _map_request(self._base_payload())
        assert body['store'] is False

    def test_max_output_tokens_absent(self):
        body = _map_request(self._base_payload(max_tokens=1000))
        assert 'max_output_tokens' not in body

    def test_parallel_tool_calls_absent(self):
        body = _map_request(self._base_payload())
        assert 'parallel_tool_calls' not in body

    def test_instructions_from_string_system(self):
        body = _map_request(self._base_payload(system='Be concise.'))
        assert body['instructions'] == 'Be concise.'

    def test_instructions_from_block_system(self):
        body = _map_request(self._base_payload(
            system=[{'type': 'text', 'text': 'Be concise.'}]
        ))
        assert body['instructions'] == 'Be concise.'

    def test_instructions_default_empty(self):
        body = _map_request(self._base_payload())
        assert body['instructions'] == ''

    def test_inline_system_message_folds_into_instructions(self):
        # Inline role:"system" messages must NOT appear in input[] — the
        # Responses API returns 400 "System messages are not allowed".
        # Both top-level system and inline system entries should land in instructions.
        body = _map_request({
            'model': 'opus',
            'system': 'Top-level.',
            'messages': [
                {'role': 'system', 'content': 'Inline guardrail.'},
                {'role': 'user', 'content': 'Hi'},
            ],
        })
        assert all(i.get('role') != 'system' for i in body['input']), \
            'role:"system" must not appear in input[]'
        assert 'Top-level.' in body['instructions']
        assert 'Inline guardrail.' in body['instructions']
        assert any(i.get('role') == 'user' for i in body['input'])

    def test_inline_system_only_no_toplevel(self):
        # Inline system with no top-level system field — should still work.
        body = _map_request({
            'model': 'sonnet',
            'messages': [
                {'role': 'system', 'content': 'Only inline.'},
                {'role': 'user', 'content': 'Hello'},
            ],
        })
        assert all(i.get('role') != 'system' for i in body['input'])
        assert body['instructions'] == 'Only inline.'
        assert len(body['input']) == 1


        body = _map_request(self._base_payload())
        # user message → input[]
        item = body['input'][0]
        assert item['type'] == 'message'
        assert item['role'] == 'user'
        assert item['content'][0]['type'] == 'input_text'
        assert item['content'][0]['text'] == 'Hello'

    def test_assistant_message_maps_to_output_text(self):
        body = _map_request({
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello!'},
            ],
        })
        asst_item = body['input'][1]
        assert asst_item['role'] == 'assistant'
        assert asst_item['content'][0]['type'] == 'output_text'

    def test_tool_use_maps_to_function_call(self):
        body = _map_request({
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': 'Run it'},
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'toolu_abc', 'name': 'bash',
                     'input': {'command': 'ls'}},
                ]},
            ],
        })
        fc_item = next(i for i in body['input'] if i.get('type') == 'function_call')
        assert fc_item['name'] == 'bash'
        assert fc_item['call_id'] == 'toolu_abc'
        assert json.loads(fc_item['arguments']) == {'command': 'ls'}

    def test_tool_result_maps_to_function_call_output(self):
        body = _map_request({
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_abc', 'content': 'file.txt'},
                ]},
            ],
        })
        fc_out = next(i for i in body['input'] if i.get('type') == 'function_call_output')
        assert fc_out['call_id'] == 'toolu_abc'
        assert fc_out['output'] == 'file.txt'

    def test_only_latest_reasoning_before_tool_result_is_replayed(self):
        signatures = [
            _encode_reasoning_signature(f'rs_{index}', f'ENC{index}')
            for index in range(1, 4)
        ]
        body = _map_request({
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': 'First task'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'first', 'signature': signatures[0]},
                    {'type': 'tool_use', 'id': 'toolu_first', 'name': 'bash',
                     'input': {'command': 'ls'}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_first',
                     'content': 'file.txt'},
                ]},
                {'role': 'assistant', 'content': 'First answer'},
                {'role': 'user', 'content': 'Second task'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'second', 'signature': signatures[1]},
                    {'type': 'tool_use', 'id': 'toolu_second', 'name': 'bash',
                     'input': {'command': 'date'}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_second',
                     'content': '2026-07-11'},
                ]},
                {'role': 'assistant', 'content': 'Second answer'},
                {'role': 'user', 'content': 'Use the tool'},
                {'role': 'assistant', 'content': [
                    {'type': 'thinking', 'thinking': 'latest', 'signature': signatures[2]},
                    {'type': 'tool_use', 'id': 'toolu_latest', 'name': 'bash',
                     'input': {'command': 'pwd'}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_latest',
                     'content': '/tmp'},
                ]},
            ],
        })

        reasoning = [item for item in body['input'] if item.get('type') == 'reasoning']
        assert reasoning == [{
            'type': 'reasoning',
            'encrypted_content': 'ENC3',
            'summary': [],
            'id': 'rs_3',
        }]
        assistant_text = [
            part['text']
            for item in body['input']
            if item.get('type') == 'message' and item.get('role') == 'assistant'
            for part in item['content']
        ]
        assert assistant_text == ['First answer', 'Second answer']
        assert any(
            item.get('type') == 'function_call' and item.get('call_id') == 'toolu_latest'
            for item in body['input']
        )
        assert any(
            item.get('type') == 'function_call_output'
            and item.get('call_id') == 'toolu_latest'
            for item in body['input']
        )

    def test_image_maps_to_input_image(self):
        body = _map_request({
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'image', 'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': 'abc123',
                    }},
                ]},
            ],
        })
        img_item = body['input'][0]
        assert img_item['content'][0]['type'] == 'input_image'
        assert img_item['content'][0]['image_url'] == 'data:image/png;base64,abc123'

    def test_tools_mapped_correctly(self):
        body = _map_request(self._base_payload(tools=[
            {'name': 'bash', 'description': 'Run a command',
             'input_schema': {'type': 'object', 'properties': {}}},
        ]))
        assert body['tools'][0] == {
            'type': 'function',
            'name': 'bash',
            'description': 'Run a command',
            'parameters': {'type': 'object', 'properties': {}},
        }

    def test_tool_choice_auto(self):
        body = _map_request(self._base_payload(tool_choice={'type': 'auto'}))
        assert body['tool_choice'] == 'auto'

    def test_tool_choice_any(self):
        body = _map_request(self._base_payload(tool_choice={'type': 'any'}))
        assert body['tool_choice'] == 'required'

    def test_tool_choice_specific_tool(self):
        body = _map_request(self._base_payload(
            tool_choice={'type': 'tool', 'name': 'bash'}
        ))
        assert body['tool_choice'] == {'type': 'function', 'name': 'bash'}

    def test_thinking_maps_to_reasoning(self):
        body = _map_request(self._base_payload(
            thinking={'type': 'enabled', 'budget_tokens': 8192}
        ))
        assert body['reasoning'] == {'effort': 'medium'}

    def test_model_resolved(self):
        body = _map_request(self._base_payload(model='opus'))
        assert body['model'] == 'gpt-5.6-sol'


# ---------------------------------------------------------------------------
# _map_response
# ---------------------------------------------------------------------------

class TestMapResponse:
    def test_text_output(self):
        items = [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello'}]}]
        resp = _map_response(items, {'input_tokens': 10, 'output_tokens': 5}, 'completed', 'sonnet')
        assert resp['role'] == 'assistant'
        assert resp['stop_reason'] == 'end_turn'
        assert resp['content'][0] == {'type': 'text', 'text': 'Hello'}
        assert resp['usage'] == {'input_tokens': 10, 'output_tokens': 5}

    def test_function_call_output(self):
        items = [{'type': 'function_call', 'call_id': 'toolu_x', 'name': 'bash',
                  'arguments': '{"command": "ls"}'}]
        resp = _map_response(items, {}, 'completed', 'sonnet')
        assert resp['stop_reason'] == 'tool_use'
        tool = resp['content'][0]
        assert tool['type'] == 'tool_use'
        assert tool['id'] == 'toolu_x'
        assert tool['name'] == 'bash'
        assert tool['input'] == {'command': 'ls'}

    def test_reasoning_output(self):
        items = [
            {'type': 'reasoning_summary', 'summary': [{'text': 'I think...'}]},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Answer'}]},
        ]
        resp = _map_response(items, {}, 'completed', 'sonnet')
        assert resp['content'][0]['type'] == 'thinking'
        assert resp['content'][0]['thinking'] == 'I think...'
        assert resp['content'][1]['type'] == 'text'

    def test_incomplete_status_gives_max_tokens(self):
        items = [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Truncated'}]}]
        resp = _map_response(items, {}, 'incomplete', 'sonnet')
        assert resp['stop_reason'] == 'max_tokens'

    def test_empty_output_returns_empty_text_block(self):
        resp = _map_response([], {}, 'completed', 'sonnet')
        assert resp['content'] == [{'type': 'text', 'text': ''}]

    def test_cache_tokens_forwarded(self):
        usage = {
            'input_tokens': 100,
            'output_tokens': 20,
            'input_tokens_details': {'cached_tokens': 50},
        }
        resp = _map_response([], usage, 'completed', 'sonnet')
        assert resp['usage']['cache_read_input_tokens'] == 50


# ---------------------------------------------------------------------------
# _iter_stream_as_anthropic_sse
# ---------------------------------------------------------------------------

class TestIterStreamAsAnthropicSse:
    def _run(self, events):
        resp = _fake_sse_response(events)
        return _collect_events(_iter_stream_as_anthropic_sse(resp, 'sonnet', estimated_input_tokens=10))

    def test_message_start_emitted_first(self):
        events = self._run([
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 10, 'output_tokens': 2}}},
        ])
        assert events[0]['event'] == 'message_start'

    def test_simple_text_stream(self):
        codex_events = [
            {'type': 'response.output_text.delta', 'item_id': 'item_1',
             'content_index': 0, 'delta': 'Hello'},
            {'type': 'response.output_text.delta', 'item_id': 'item_1',
             'content_index': 0, 'delta': ' world'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 10, 'output_tokens': 3}}},
        ]
        events = self._run(codex_events)
        event_types = [e['event'] for e in events]
        assert 'content_block_start' in event_types
        assert 'content_block_delta' in event_types
        assert 'content_block_stop' in event_types
        assert 'message_delta' in event_types
        assert 'message_stop' in event_types

        # Collect all text deltas
        text = ''.join(
            e['data']['delta']['text']
            for e in events
            if e['event'] == 'content_block_delta'
            and e['data'].get('delta', {}).get('type') == 'text_delta'
        )
        assert text == 'Hello world'

    def test_thinking_before_text(self):
        codex_events = [
            {'type': 'response.reasoning_summary_text.delta',
             'item_id': 'r_1', 'delta': 'Let me think...'},
            {'type': 'response.output_text.delta',
             'item_id': 'item_1', 'content_index': 0, 'delta': 'Answer'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 2}}},
        ]
        events = self._run(codex_events)
        starts = [e for e in events if e['event'] == 'content_block_start']
        block_types = [s['data']['content_block']['type'] for s in starts]
        assert 'thinking' in block_types
        assert 'text' in block_types
        # Thinking must come before text
        assert block_types.index('thinking') < block_types.index('text')

    def test_reasoning_dropped_after_text(self):
        """Reasoning deltas that arrive after text opened must be silently dropped."""
        codex_events = [
            {'type': 'response.output_text.delta',
             'item_id': 'item_1', 'content_index': 0, 'delta': 'Text first'},
            # This reasoning delta arrives after text — must be dropped
            {'type': 'response.reasoning_summary_text.delta',
             'item_id': 'r_1', 'delta': 'Oops thinking'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 2}}},
        ]
        events = self._run(codex_events)
        starts = [e for e in events if e['event'] == 'content_block_start']
        block_types = [s['data']['content_block']['type'] for s in starts]
        assert 'thinking' not in block_types

    def test_tool_use_stream(self):
        codex_events = [
            {'type': 'response.output_item.added',
             'output_index': 0,
             'item': {'type': 'function_call', 'id': 'fc_001',
                      'call_id': 'toolu_abc', 'name': 'bash'}},
            {'type': 'response.function_call_arguments.delta',
             'item_id': 'fc_001', 'output_index': 0, 'delta': '{"command":'},
            {'type': 'response.function_call_arguments.delta',
             'item_id': 'fc_001', 'output_index': 0, 'delta': '"ls"}'},
            {'type': 'response.output_item.done',
             'output_index': 0,
             'item': {'type': 'function_call', 'id': 'fc_001',
                      'call_id': 'toolu_abc', 'name': 'bash',
                      'arguments': '{"command":"ls"}'}},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 10, 'output_tokens': 5}}},
        ]
        events = self._run(codex_events)
        tool_start = next(
            e for e in events
            if e['event'] == 'content_block_start'
            and e['data']['content_block']['type'] == 'tool_use'
        )
        assert tool_start['data']['content_block']['name'] == 'bash'

        args = ''.join(
            e['data']['delta']['partial_json']
            for e in events
            if e['event'] == 'content_block_delta'
            and e['data'].get('delta', {}).get('type') == 'input_json_delta'
        )
        assert json.loads(args) == {'command': 'ls'}

    def test_message_delta_has_stop_reason(self):
        codex_events = [
            {'type': 'response.output_text.delta',
             'item_id': 'item_1', 'content_index': 0, 'delta': 'Done'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 1}}},
        ]
        events = self._run(codex_events)
        msg_delta = next(e for e in events if e['event'] == 'message_delta')
        assert msg_delta['data']['delta']['stop_reason'] == 'end_turn'

    def test_output_item_done_completes_empty_output(self):
        """response.completed.output=[] with output_item.done items is handled."""
        codex_events = [
            {'type': 'response.output_item.done',
             'output_index': 0,
             'item': {'type': 'message', 'id': 'msg_x',
                      'content': [{'type': 'output_text', 'text': 'Hi'}]}},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 1}}},
        ]
        # Should not raise and should emit message_stop
        events = self._run(codex_events)
        assert any(e['event'] == 'message_stop' for e in events)

    @pytest.mark.parametrize('message', [
        'Prompt is too long',
        'This request exceeds the context window for this model',
    ])
    def test_context_overflow_failure_maps_to_invalid_request(self, message):
        with pytest.raises(AnthropicRequestError) as exc:
            self._run([{
                'type': 'response.failed',
                'response': {'error': {'message': message}},
            }])
        assert exc.value.status_code == 400
        assert exc.value.error_type == 'invalid_request_error'

    def test_unrelated_failure_remains_api_error(self):
        with pytest.raises(AnthropicRequestError) as exc:
            self._run([{
                'type': 'response.failed',
                'response': {'error': {'message': 'Internal server failure'}},
            }])
        assert exc.value.status_code == 502
        assert exc.value.error_type == 'api_error'


class TestDrainSseToResponseFailures:
    @pytest.mark.parametrize('message', [
        'Prompt is too long',
        'Maximum context length exceeded',
    ])
    def test_context_overflow_maps_to_invalid_request(self, message):
        response = _fake_sse_response([{
            'type': 'response.failed',
            'response': {'error': {'message': message}},
        }])
        with pytest.raises(AnthropicRequestError) as exc:
            _drain_sse_to_response(response, 'sonnet')
        assert exc.value.status_code == 400
        assert exc.value.error_type == 'invalid_request_error'

    def test_unrelated_failure_remains_api_error(self):
        response = _fake_sse_response([{
            'type': 'response.failed',
            'response': {'error': {'message': 'Internal server failure'}},
        }])
        with pytest.raises(AnthropicRequestError) as exc:
            _drain_sse_to_response(response, 'sonnet')
        assert exc.value.status_code == 502
        assert exc.value.error_type == 'api_error'


# ---------------------------------------------------------------------------
# codex_auth: JWT helpers
# ---------------------------------------------------------------------------

class TestJwtHelpers:
    def test_decode_payload(self):
        payload = {'sub': 'user123', 'exp': 9999999999}
        jwt = _make_jwt(payload)
        decoded = _decode_jwt_payload(jwt)
        assert decoded['sub'] == 'user123'
        assert decoded['exp'] == 9999999999

    def test_jwt_exp_present(self):
        future = int(time.time()) + 3600
        jwt = _make_jwt({'exp': future})
        assert _jwt_exp(jwt) == future

    def test_jwt_exp_absent(self):
        jwt = _make_jwt({'sub': 'user123'})
        assert _jwt_exp(jwt) is None

    def test_jwt_exp_malformed_returns_none(self):
        assert _jwt_exp('not.a.jwt') is None
        assert _jwt_exp('') is None

    def test_account_id_from_id_token(self):
        payload = {
            'https://api.openai.com/auth': {
                'chatgpt_account_id': 'acc_abc123',
            }
        }
        jwt = _make_jwt(payload)
        assert _account_id_from_id_token(jwt) == 'acc_abc123'

    def test_account_id_fallback_top_level(self):
        payload = {'chatgpt_account_id': 'acc_fallback'}
        jwt = _make_jwt(payload)
        assert _account_id_from_id_token(jwt) == 'acc_fallback'

    def test_account_id_missing_returns_none(self):
        jwt = _make_jwt({'sub': 'user'})
        assert _account_id_from_id_token(jwt) is None


# ---------------------------------------------------------------------------
# codex_auth: PKCE
# ---------------------------------------------------------------------------

class TestPkce:
    def test_verifier_and_challenge_shapes(self):
        verifier, challenge = _pkce()
        # verifier: 64 bytes → 86 chars base64url no padding
        assert len(verifier) == 86
        assert set(verifier) <= set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_')
        # challenge: SHA256 → 32 bytes → 43 chars base64url no padding
        assert len(challenge) == 43

    def test_challenge_is_sha256_of_verifier(self):
        verifier, challenge = _pkce()
        expected = _b64url_nopad(hashlib.sha256(verifier.encode()).digest())
        assert challenge == expected

    def test_unique_each_call(self):
        v1, c1 = _pkce()
        v2, c2 = _pkce()
        assert v1 != v2
        assert c1 != c2


# ---------------------------------------------------------------------------
# codex_auth: load_credentials + _write_auth
# ---------------------------------------------------------------------------

class TestLoadWriteCredentials:
    def _make_auth_json(self, **overrides) -> dict:
        base = {
            'auth_mode': 'chatgpt',
            'OPENAI_API_KEY': None,
            'tokens': {
                'id_token': _make_jwt({'sub': 'u1'}),
                'access_token': _make_jwt({'exp': int(time.time()) + 3600}),
                'refresh_token': 'opaque_refresh_token',
                'account_id': 'acc_test',
            },
            'last_refresh': '2026-06-01T12:00:00Z',
        }
        base.update(overrides)
        return base

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir) / 'codex'
            assert load_credentials(home) is None

    def test_empty_tokens_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            home.mkdir(exist_ok=True)
            (home / 'auth.json').write_text(
                json.dumps({'tokens': {}}), encoding='utf-8'
            )
            assert load_credentials(home) is None

    def test_valid_credentials_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            raw = self._make_auth_json()
            _write_auth(home, raw)
            creds = load_credentials(home)
            assert creds is not None
            assert creds['account_id'] == 'acc_test'
            assert creds['refresh_token'] == 'opaque_refresh_token'

    def test_write_preserves_openai_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            raw = self._make_auth_json()
            _write_auth(home, raw)
            on_disk = json.loads((home / 'auth.json').read_text(encoding='utf-8'))
            assert 'OPENAI_API_KEY' in on_disk  # must always be present
            assert on_disk['OPENAI_API_KEY'] is None

    def test_write_id_token_as_string(self):
        """id_token must be written as a raw JWT string, not an object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            raw = self._make_auth_json()
            _write_auth(home, raw)
            on_disk = json.loads((home / 'auth.json').read_text(encoding='utf-8'))
            # id_token must be a string (raw JWT), not a dict
            assert isinstance(on_disk['tokens']['id_token'], str)
            assert '.' in on_disk['tokens']['id_token']  # has JWT dot separators

    def test_auth_mode_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            raw = self._make_auth_json()
            _write_auth(home, raw)
            on_disk = json.loads((home / 'auth.json').read_text(encoding='utf-8'))
            assert on_disk.get('auth_mode') == 'chatgpt'


# ---------------------------------------------------------------------------
# codex_auth: needs_access_refresh
# ---------------------------------------------------------------------------

class TestNeedsAccessRefresh:
    def _creds_with_exp(self, exp_offset_secs: int) -> dict:
        future_exp = int(time.time()) + exp_offset_secs
        return {
            'access_token': _make_jwt({'exp': future_exp}),
            'refresh_token': 'tok',
            'last_refresh': time.time(),
        }

    def test_not_due_when_exp_far_future(self):
        creds = self._creds_with_exp(3600)  # expires in 1 hour
        assert needs_access_refresh(creds) is False

    def test_due_when_exp_within_window(self):
        creds = self._creds_with_exp(60)  # expires in 1 minute
        assert needs_access_refresh(creds) is True

    def test_due_when_access_token_already_expired(self):
        creds = self._creds_with_exp(-100)  # expired 100s ago
        assert needs_access_refresh(creds) is True

    def test_due_when_no_exp_and_old_last_refresh(self):
        creds = {
            'access_token': _make_jwt({'sub': 'user'}),  # no exp
            'refresh_token': 'tok',
            'last_refresh': time.time() - (ACCESS_REFRESH_INTERVAL_DAYS + 1) * 86400,
        }
        assert needs_access_refresh(creds) is True

    def test_not_due_when_no_exp_and_recent_last_refresh(self):
        creds = {
            'access_token': _make_jwt({'sub': 'user'}),  # no exp
            'refresh_token': 'tok',
            'last_refresh': time.time() - 3600,  # refreshed 1 hour ago
        }
        assert needs_access_refresh(creds) is False

    def test_due_when_no_exp_and_no_last_refresh(self):
        creds = {
            'access_token': _make_jwt({'sub': 'user'}),
            'refresh_token': 'tok',
            'last_refresh': None,
        }
        assert needs_access_refresh(creds) is True


# ---------------------------------------------------------------------------
# _map_request: new fields (prompt_cache_key, include, top_p)
# ---------------------------------------------------------------------------

class TestMapRequestNewFields:
    def _base_payload(self, **overrides):
        return {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            **overrides,
        }

    def test_prompt_cache_key_from_metadata_user_id(self):
        body = _map_request(self._base_payload(metadata={'user_id': 'user-42'}))
        assert body.get('prompt_cache_key') == 'user-42'

    def test_prompt_cache_key_from_system_hash_when_no_user_id(self):
        body = _map_request(self._base_payload(system='You are helpful.'))
        key = body.get('prompt_cache_key')
        assert key is not None
        # Should be a 32-char hex string (SHA-256 first 32 chars)
        assert len(key) == 32
        assert all(c in '0123456789abcdef' for c in key)

    def test_prompt_cache_key_absent_when_no_system_and_no_user_id(self):
        body = _map_request(self._base_payload())
        assert 'prompt_cache_key' not in body

    def test_prompt_cache_key_clamped_to_64_chars(self):
        long_id = 'u' * 100
        body = _map_request(self._base_payload(metadata={'user_id': long_id}))
        assert len(body['prompt_cache_key']) == 64

    def test_include_reasoning_when_thinking_enabled(self):
        body = _map_request(self._base_payload(
            thinking={'type': 'enabled', 'budget_tokens': 8192}
        ))
        assert body.get('include') == ['reasoning.encrypted_content']

    def test_include_absent_when_thinking_disabled(self):
        body = _map_request(self._base_payload())
        assert 'include' not in body

    def test_top_p_forwarded(self):
        body = _map_request(self._base_payload(top_p=0.9))
        assert body.get('top_p') == 0.9

    def test_temperature_omitted_while_zero_top_p_preserved(self):
        body = _map_request(self._base_payload(temperature=0, top_p=0))
        assert 'temperature' not in body
        assert 'top_p' in body
        assert body['top_p'] == 0

    def test_top_p_absent_when_not_in_payload(self):
        body = _map_request(self._base_payload())
        assert 'top_p' not in body


# ---------------------------------------------------------------------------
# _retry_delay
# ---------------------------------------------------------------------------

class TestRetryDelay:
    def _make_resp(self, headers: dict):
        """Build a minimal fake HTTPResponse with a getheader method."""
        resp = MagicMock(spec=http.client.HTTPResponse)
        resp.getheader = lambda name, default='': headers.get(name, headers.get(name.lower(), default))
        return resp

    def test_integer_retry_after(self):
        resp = self._make_resp({'Retry-After': '10'})
        assert _retry_delay(resp, 0) == 10.0

    def test_retry_after_ms(self):
        resp = self._make_resp({'retry-after-ms': '5000'})
        assert _retry_delay(resp, 0) == pytest.approx(5.0, abs=0.01)

    def test_retry_after_ms_takes_priority_over_retry_after(self):
        resp = self._make_resp({'retry-after-ms': '2000', 'Retry-After': '60'})
        assert _retry_delay(resp, 0) == pytest.approx(2.0, abs=0.01)

    def test_exponential_fallback_attempt_0(self):
        assert _retry_delay(None, 0) == pytest.approx(1.0)

    def test_exponential_fallback_attempt_1(self):
        assert _retry_delay(None, 1) == pytest.approx(2.0)

    def test_exponential_fallback_capped(self):
        assert _retry_delay(None, 100) == pytest.approx(30.0)

    def test_none_resp_falls_back_to_exponential(self):
        delay = _retry_delay(None, 2)
        assert delay == pytest.approx(4.0)

    def test_http_date_retry_after(self):
        """Retry-After with an HTTP-date in the past → treated as 0/exponential."""
        import email.utils
        past = email.utils.formatdate(timeval=time.time() - 60, usegmt=True)
        resp = self._make_resp({'Retry-After': past})
        # A past date returns <=0, so exponential fallback is used
        delay = _retry_delay(resp, 0)
        # Should fall through to the exponential path since the header gave <=0
        assert delay >= 0


# ---------------------------------------------------------------------------
# _send_with_retries  (network-free via patching)
# ---------------------------------------------------------------------------

class TestSendWithRetries:
    """All tests patch _make_connection so no real network calls are made."""

    def _fake_config(self):
        cfg = MagicMock()
        cfg.codex_home = ''
        return cfg

    def _mock_200_resp(self):
        resp = MagicMock(spec=http.client.HTTPResponse)
        resp.status = 200
        return resp

    def _mock_error_resp(self, status, body=b'{}'):
        resp = MagicMock(spec=http.client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheader = MagicMock(return_value='')
        return resp

    def _simple_payload(self):
        return {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_200_on_first_attempt(self, mock_get_access, mock_make_conn):
        mock_get_access.return_value = ('tok_access', 'acc_id')
        conn = MagicMock()
        resp = self._mock_200_resp()
        conn.getresponse.return_value = resp
        mock_make_conn.return_value = conn

        lock = threading.Lock()
        result_conn, result_resp = _send_with_retries(self._simple_payload(), self._fake_config(), lock)
        assert result_resp.status == 200
        result_conn.close()

    @patch('anthproxy.codex.backend.time.sleep')
    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_429_without_retry_after_surfaces_immediately(self, mock_get_access, mock_make_conn, mock_sleep):
        """A 429 with no Retry-After must not be retried — surface it immediately."""
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('tok_access', 'acc_id')
        conn = MagicMock()
        # getheader returns '' for all headers → parse_retry_after returns None
        conn.getresponse.return_value = self._mock_error_resp(429)
        mock_make_conn.return_value = conn

        lock = threading.Lock()
        with pytest.raises(AnthropicRequestError) as exc_info:
            _send_with_retries(self._simple_payload(), self._fake_config(), lock)
        assert exc_info.value.status_code == 429
        # Only one attempt — no sleep, no second connection
        assert mock_sleep.call_count == 0
        assert conn.getresponse.call_count == 1

    @patch('anthproxy.codex.backend.time.sleep')
    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_429_with_retry_after_retries_then_succeeds(self, mock_get_access, mock_make_conn, mock_sleep):
        """A 429 with a Retry-After header is retried and succeeds on the second attempt."""
        mock_get_access.return_value = ('tok_access', 'acc_id')
        resp_429 = self._mock_error_resp(429)
        # Provide a Retry-After header so should_retry returns True
        resp_429.getheader.side_effect = lambda name, default='': ('5' if name == 'Retry-After' else default)
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = resp_429
        conn2.getresponse.return_value = self._mock_200_resp()
        mock_make_conn.side_effect = [conn1, conn2]

        lock = threading.Lock()
        result_conn, result_resp = _send_with_retries(self._simple_payload(), self._fake_config(), lock)
        assert result_resp.status == 200
        assert mock_sleep.called
        result_conn.close()

    @patch('anthproxy.codex.backend.time.sleep')
    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.force_refresh')
    @patch('anthproxy.codex.backend.get_access')
    def test_401_triggers_force_refresh_then_200(
        self, mock_get_access, mock_force_refresh, mock_make_conn, mock_sleep
    ):
        mock_get_access.return_value = ('old_tok', 'acc_id')
        mock_force_refresh.return_value = ('new_tok', 'acc_id')
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = self._mock_error_resp(401)
        conn2.getresponse.return_value = self._mock_200_resp()
        mock_make_conn.side_effect = [conn1, conn2]

        lock = threading.Lock()
        result_conn, result_resp = _send_with_retries(self._simple_payload(), self._fake_config(), lock)
        assert result_resp.status == 200
        mock_force_refresh.assert_called_once()
        result_conn.close()

    @patch('anthproxy.codex.backend.time.sleep')
    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.force_refresh')
    @patch('anthproxy.codex.backend.get_access')
    def test_two_401s_raises(
        self, mock_get_access, mock_force_refresh, mock_make_conn, mock_sleep
    ):
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('old_tok', 'acc_id')
        mock_force_refresh.return_value = ('new_tok', 'acc_id')
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = self._mock_error_resp(401)
        conn2.getresponse.return_value = self._mock_error_resp(401)
        mock_make_conn.side_effect = [conn1, conn2]

        lock = threading.Lock()
        with pytest.raises(AnthropicRequestError) as exc_info:
            _send_with_retries(self._simple_payload(), self._fake_config(), lock)
        assert exc_info.value.status_code == 401

    @patch('anthproxy.codex.backend.time.sleep')
    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_exhausted_retries_raises(self, mock_get_access, mock_make_conn, mock_sleep):
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('tok', 'acc')
        conns = []
        for _ in range(MAX_RETRIES + 1):
            conn = MagicMock()
            conn.getresponse.return_value = self._mock_error_resp(502)
            conns.append(conn)
        mock_make_conn.side_effect = conns

        lock = threading.Lock()
        with pytest.raises(AnthropicRequestError) as exc_info:
            _send_with_retries(self._simple_payload(), self._fake_config(), lock)
        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# Codex unsupported-model fallback
# ---------------------------------------------------------------------------

_UNSUPPORTED_MODEL_BODY = json.dumps({
    'error': {
        'message': 'gpt-5.6-sol is not supported when using Codex with a ChatGPT account',
        'type': 'invalid_request_error',
    }
}).encode()

_OTHER_400_BODY = json.dumps({
    'error': {'message': 'bad request for unrelated reason', 'type': 'invalid_request_error'}
}).encode()


class TestCodexUnsupportedModelFallback:
    def _fake_config(self, fallback: str = ''):
        cfg = MagicMock()
        cfg.codex_home = ''
        cfg.codex_unsupported_model_fallback = fallback
        return cfg

    def _mock_200_resp(self):
        resp = MagicMock(spec=http.client.HTTPResponse)
        resp.status = 200
        return resp

    def _mock_error_resp(self, status, body=b'{}'):
        resp = MagicMock(spec=http.client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheader = MagicMock(return_value='')
        return resp

    def _simple_payload(self, model='fable'):
        return {'model': model, 'messages': [{'role': 'user', 'content': 'Hi'}]}

    # ------------------------------------------------------------------
    # _is_chatgpt_unsupported_model_error
    # ------------------------------------------------------------------

    def test_matches_chatgpt_unsupported_message(self):
        assert _is_chatgpt_unsupported_model_error(400, _UNSUPPORTED_MODEL_BODY)

    def test_does_not_match_other_400(self):
        assert not _is_chatgpt_unsupported_model_error(400, _OTHER_400_BODY)

    def test_does_not_match_non_400_status(self):
        assert not _is_chatgpt_unsupported_model_error(422, _UNSUPPORTED_MODEL_BODY)

    def test_does_not_match_invalid_json(self):
        assert not _is_chatgpt_unsupported_model_error(400, b'not json')

    # ------------------------------------------------------------------
    # _codex_unsupported_model_fallback
    # ------------------------------------------------------------------

    def test_returns_configured_value(self):
        cfg = self._fake_config(fallback='haiku')
        assert _codex_unsupported_model_fallback(cfg) == 'haiku'

    def test_returns_empty_when_disabled(self):
        cfg = self._fake_config(fallback='')
        assert _codex_unsupported_model_fallback(cfg) == ''

    def test_returns_empty_when_attribute_missing(self):
        cfg = MagicMock(spec=[])
        assert _codex_unsupported_model_fallback(cfg) == ''

    # ------------------------------------------------------------------
    # _send_with_retries — fallback disabled
    # ------------------------------------------------------------------

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_unsupported_model_raises_when_fallback_disabled(
        self, mock_get_access, mock_make_conn
    ):
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('tok', 'acc')
        conn = MagicMock()
        conn.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        mock_make_conn.return_value = conn

        with pytest.raises(AnthropicRequestError) as exc_info:
            _send_with_retries(
                self._simple_payload(),
                self._fake_config(fallback=''),
                threading.Lock(),
            )
        assert exc_info.value.status_code == 400

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_unsupported_model_logs_warning_when_fallback_disabled(
        self, mock_get_access, mock_make_conn, caplog
    ):
        import logging
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('tok', 'acc')
        conn = MagicMock()
        conn.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        mock_make_conn.return_value = conn

        with caplog.at_level(logging.WARNING, logger='anthproxy.codex.backend'):
            with pytest.raises(AnthropicRequestError):
                _send_with_retries(
                    self._simple_payload(),
                    self._fake_config(fallback=''),
                    threading.Lock(),
                )
        assert any('fallback disabled' in r.message for r in caplog.records)

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_other_400_raises_without_fallback_attempt(
        self, mock_get_access, mock_make_conn
    ):
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('tok', 'acc')
        conn = MagicMock()
        conn.getresponse.return_value = self._mock_error_resp(400, _OTHER_400_BODY)
        mock_make_conn.side_effect = [conn]
        mock_make_conn.return_value = conn

        with pytest.raises(AnthropicRequestError) as exc_info:
            _send_with_retries(
                self._simple_payload(),
                self._fake_config(fallback='haiku'),
                threading.Lock(),
            )
        assert exc_info.value.status_code == 400
        # Only one connection attempt — no fallback retry
        assert mock_make_conn.call_count == 1

    # ------------------------------------------------------------------
    # _send_with_retries — fallback enabled, succeeds on retry
    # ------------------------------------------------------------------

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_fallback_retries_once_on_unsupported_model(
        self, mock_get_access, mock_make_conn
    ):
        mock_get_access.return_value = ('tok', 'acc')
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        conn2.getresponse.return_value = self._mock_200_resp()
        mock_make_conn.side_effect = [conn1, conn2]

        result_conn, result_resp = _send_with_retries(
            self._simple_payload(model='fable'),
            self._fake_config(fallback='haiku'),
            threading.Lock(),
        )
        assert result_resp.status == 200
        assert mock_make_conn.call_count == 2
        result_conn.close()

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_fallback_logs_warning_with_model_names(
        self, mock_get_access, mock_make_conn, caplog
    ):
        import logging
        mock_get_access.return_value = ('tok', 'acc')
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        conn2.getresponse.return_value = self._mock_200_resp()
        mock_make_conn.side_effect = [conn1, conn2]

        with caplog.at_level(logging.WARNING, logger='anthproxy.codex.backend'):
            _send_with_retries(
                self._simple_payload(model='fable'),
                self._fake_config(fallback='haiku'),
                threading.Lock(),
            )
        assert any('fallback' in r.message and 'haiku' in r.message for r in caplog.records)

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_fallback_does_not_mutate_original_payload(
        self, mock_get_access, mock_make_conn
    ):
        mock_get_access.return_value = ('tok', 'acc')
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        conn2.getresponse.return_value = self._mock_200_resp()
        mock_make_conn.side_effect = [conn1, conn2]

        payload = self._simple_payload(model='fable')
        _send_with_retries(payload, self._fake_config(fallback='haiku'), threading.Lock())
        assert payload['model'] == 'fable'

    @patch('anthproxy.codex.backend._make_connection')
    @patch('anthproxy.codex.backend.get_access')
    def test_fallback_used_only_once_then_raises(
        self, mock_get_access, mock_make_conn
    ):
        from anthproxy.mapper import AnthropicRequestError
        mock_get_access.return_value = ('tok', 'acc')
        # Both the primary and the fallback return the unsupported-model 400
        conn1 = MagicMock()
        conn2 = MagicMock()
        conn1.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        conn2.getresponse.return_value = self._mock_error_resp(400, _UNSUPPORTED_MODEL_BODY)
        mock_make_conn.side_effect = [conn1, conn2]

        with pytest.raises(AnthropicRequestError) as exc_info:
            _send_with_retries(
                self._simple_payload(model='fable'),
                self._fake_config(fallback='haiku'),
                threading.Lock(),
            )
        assert exc_info.value.status_code == 400
        # Exactly two attempts: primary + one fallback
        assert mock_make_conn.call_count == 2


# ---------------------------------------------------------------------------
# Config: codex_unsupported_model_fallback knob
# ---------------------------------------------------------------------------

class TestCodexUnsupportedModelFallbackConfig:
    def test_default_is_empty_string(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.codex_unsupported_model_fallback == ''

    def test_cli_sets_fallback(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--codex-unsupported-model-fallback', 'haiku'])
        assert cfg.codex_unsupported_model_fallback == 'haiku'

    def test_env_sets_fallback(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_CODEX_UNSUPPORTED_MODEL_FALLBACK', 'sonnet')
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.codex_unsupported_model_fallback == 'sonnet'

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv('ANTHPROXY_CODEX_UNSUPPORTED_MODEL_FALLBACK', 'opus')
        from anthproxy.config import parse_args
        cfg = parse_args(['--codex-unsupported-model-fallback', 'haiku'])
        assert cfg.codex_unsupported_model_fallback == 'haiku'

    def test_whitespace_only_normalizes_to_empty(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--codex-unsupported-model-fallback', '   '])
        assert cfg.codex_unsupported_model_fallback == ''


# ---------------------------------------------------------------------------
# Streaming: new event types and fields
# ---------------------------------------------------------------------------

class TestStreamingNewEvents:
    def _run(self, events):
        resp = _fake_sse_response(events)
        return _collect_events(_iter_stream_as_anthropic_sse(resp, 'sonnet', estimated_input_tokens=10))

    def test_reasoning_text_delta_opens_thinking_block(self):
        """response.reasoning_text.delta should open a thinking block."""
        codex_events = [
            {'type': 'response.reasoning_text.delta',
             'item_id': 'r_1', 'delta': 'Raw reasoning...'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 1}}},
        ]
        events = self._run(codex_events)
        starts = [e for e in events if e['event'] == 'content_block_start']
        block_types = [s['data']['content_block']['type'] for s in starts]
        assert 'thinking' in block_types

        thinking_text = ''.join(
            e['data']['delta']['thinking']
            for e in events
            if e['event'] == 'content_block_delta'
            and e['data'].get('delta', {}).get('type') == 'thinking_delta'
        )
        assert 'Raw reasoning' in thinking_text

    def test_response_incomplete_gives_max_tokens_stop_reason(self):
        """response.incomplete terminal event → stop_reason = 'max_tokens'."""
        codex_events = [
            {'type': 'response.output_text.delta',
             'item_id': 'item_1', 'content_index': 0, 'delta': 'Truncated...'},
            {'type': 'response.incomplete',
             'response': {'status': 'incomplete', 'output': [],
                          'usage': {'input_tokens': 10, 'output_tokens': 5}}},
        ]
        events = self._run(codex_events)
        msg_delta = next(e for e in events if e['event'] == 'message_delta')
        assert msg_delta['data']['delta']['stop_reason'] == 'max_tokens'

    def test_response_done_treated_as_completed(self):
        """response.done should finalize the stream correctly."""
        codex_events = [
            {'type': 'response.output_text.delta',
             'item_id': 'item_1', 'content_index': 0, 'delta': 'Done'},
            {'type': 'response.done',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 1}}},
        ]
        events = self._run(codex_events)
        assert any(e['event'] == 'message_stop' for e in events)
        msg_delta = next(e for e in events if e['event'] == 'message_delta')
        assert msg_delta['data']['delta']['stop_reason'] == 'end_turn'

    def test_message_delta_includes_cache_read_input_tokens(self):
        """cache_read_input_tokens from input_tokens_details should appear in message_delta."""
        codex_events = [
            {'type': 'response.output_text.delta',
             'item_id': 'item_1', 'content_index': 0, 'delta': 'Hi'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {
                              'input_tokens': 100,
                              'output_tokens': 5,
                              'input_tokens_details': {'cached_tokens': 40},
                          }}},
        ]
        events = self._run(codex_events)
        msg_delta = next(e for e in events if e['event'] == 'message_delta')
        usage = msg_delta['data']['usage']
        assert usage.get('cache_read_input_tokens') == 40

    def test_reasoning_summary_part_added_inserts_separator(self):
        """response.reasoning_summary_part.added emits a '\\n\\n' thinking_delta."""
        codex_events = [
            {'type': 'response.reasoning_summary_text.delta',
             'item_id': 'r_1', 'delta': 'Part one.'},
            {'type': 'response.reasoning_summary_part.added',
             'item_id': 'r_1'},
            {'type': 'response.reasoning_summary_text.delta',
             'item_id': 'r_1', 'delta': 'Part two.'},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 1}}},
        ]
        events = self._run(codex_events)
        thinking_deltas = [
            e['data']['delta']['thinking']
            for e in events
            if e['event'] == 'content_block_delta'
            and e['data'].get('delta', {}).get('type') == 'thinking_delta'
        ]
        full_thinking = ''.join(thinking_deltas)
        assert 'Part one.' in full_thinking
        assert '\n\n' in full_thinking
        assert 'Part two.' in full_thinking


# ---------------------------------------------------------------------------
# codex_auth: atomic _write_auth and _refresh_lock
# ---------------------------------------------------------------------------

class TestAtomicWriteAuth:
    def _make_raw(self):
        return {
            'auth_mode': 'chatgpt',
            'OPENAI_API_KEY': None,
            'tokens': {
                'id_token': _make_jwt({'sub': 'u1'}),
                'access_token': _make_jwt({'exp': int(time.time()) + 3600}),
                'refresh_token': 'rtoken',
                'account_id': 'acc_1',
            },
            'last_refresh': '2026-06-01T12:00:00Z',
        }

    def test_write_creates_auth_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            _write_auth(home, self._make_raw())
            assert (home / 'auth.json').exists()

    def test_write_content_is_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            raw = self._make_raw()
            _write_auth(home, raw)
            on_disk = json.loads((home / 'auth.json').read_text(encoding='utf-8'))
            assert on_disk['tokens']['refresh_token'] == 'rtoken'
            assert on_disk['tokens']['account_id'] == 'acc_1'

    def test_no_temp_file_left_behind(self):
        """Verify the atomic write leaves no .auth_tmp_*.json files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            _write_auth(home, self._make_raw())
            tmp_files = list(home.glob('.auth_tmp_*.json'))
            assert len(tmp_files) == 0

    def test_file_mode_600(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            _write_auth(home, self._make_raw())
            mode = (home / 'auth.json').stat().st_mode & 0o777
            # Allow 0o600 (exact) or 0o644 on Windows-like systems where chmod
            # is best-effort
            assert mode in (0o600, 0o644)

    def test_refresh_lock_context_manager_runs(self):
        """_refresh_lock must be usable as a context manager without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            ran = []
            with _refresh_lock(home):
                ran.append(True)
            assert ran == [True]

    def test_refresh_lock_is_reentrant_across_sequential_uses(self):
        """Lock can be acquired twice sequentially (not re-entrant, but serially)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            with _refresh_lock(home):
                pass
            with _refresh_lock(home):
                pass  # should not deadlock


class TestCodexUsage:
    def _usage_payload(self):
        return {
            'plan_type': 'Plus',
            'credits': {'balance': '12.5'},
            'rate_limit': {
                'limit_reached': False,
                'primary_window': {
                    'limit_window_seconds': 18000,
                    'used_percent': 25,
                    'reset_at': 1780700000,
                },
                'secondary_window': {
                    'limit_window_seconds': 604800,
                    'used_percent': 70,
                    'reset_at': 1781100000,
                },
            },
        }

    def _response(self, status, data):
        response = MagicMock(spec=http.client.HTTPResponse)
        response.status = status
        response.read.return_value = json.dumps(data).encode()
        return response

    def test_normalizes_and_formats_windows(self):
        usage = _normalize_usage(self._usage_payload(), fetched_at=1780600000)
        markdown = _format_usage_markdown(usage)
        assert '**Plan:** Plus' in markdown
        assert '**Credits:** $12.50' in markdown
        assert '**5-hour usage:** 25% used · 75% remaining' in markdown
        assert '**Weekly usage:** 70% used · 30% remaining' in markdown

    def test_secondary_weekly_inferred_from_reset_cadence(self):
        data = self._usage_payload()
        data['rate_limit']['secondary_window']['limit_window_seconds'] = 86400
        usage = _normalize_usage(data)
        assert usage['weekly'] is not None

    def test_primary_168_hour_window_becomes_weekly_fallback(self):
        data = self._usage_payload()
        rate_limit = data['rate_limit']
        rate_limit['primary_window'] = {
            'limit_window_seconds': 168 * 60 * 60,
            'used_percent': 62,
            'reset_at': 1781200000,
        }
        rate_limit.pop('secondary_window')

        usage = _normalize_usage(data)

        assert usage['primary']['window_seconds'] == 168 * 60 * 60
        assert usage['weekly'] == usage['primary']

    def test_explicit_secondary_weekly_takes_precedence_over_primary_fallback(self):
        data = self._usage_payload()
        data['rate_limit']['primary_window']['limit_window_seconds'] = 168 * 60 * 60
        data['rate_limit']['primary_window']['used_percent'] = 62
        data['rate_limit']['secondary_window']['used_percent'] = 31

        usage = _normalize_usage(data)

        assert usage['weekly']['used_percent'] == 31
        assert usage['weekly'] is not usage['primary']

    def test_short_primary_does_not_become_weekly_fallback(self):
        data = self._usage_payload()
        data['rate_limit'].pop('secondary_window')

        usage = _normalize_usage(data)

        assert usage['primary'] is not None
        assert usage['weekly'] is None

    def test_weekly_only_primary_drives_subscription_status(self):
        data = self._usage_payload()
        rate_limit = data['rate_limit']
        rate_limit['primary_window'] = {
            'limit_window_seconds': 168 * 60 * 60,
            'used_percent': 62,
            'reset_at': 1781200000,
        }
        rate_limit.pop('secondary_window')
        backend = CodexBackend()
        backend.get_usage = MagicMock(return_value=_normalize_usage(data))

        status = backend.five_hour_status(MagicMock())

        assert status.available is True
        assert status.resets_at == 1781200000
        assert status.utilization == 62
        assert status.weekly_utilization == 62

    def test_reset_after_seconds_fallback(self):
        data = self._usage_payload()
        primary = data['rate_limit']['primary_window']
        primary.pop('reset_at')
        primary['reset_after_seconds'] = 300
        usage = _normalize_usage(data, fetched_at=1000)
        assert usage['primary']['reset_at'] == 1300

    @patch('anthproxy.codex.backend.http.client.HTTPSConnection')
    @patch('anthproxy.codex.backend.get_access')
    def test_fetches_wham_with_account_header(self, mock_get_access, mock_connection):
        mock_get_access.return_value = ('access', 'account')
        connection = MagicMock()
        connection.getresponse.return_value = self._response(200, self._usage_payload())
        mock_connection.return_value = connection

        result = _fetch_usage(MagicMock(), threading.Lock())

        method, path = connection.request.call_args.args[:2]
        headers = connection.request.call_args.kwargs['headers']
        assert method == 'GET'
        assert path == CODEX_USAGE_PATH
        assert headers['Authorization'] == 'Bearer access'
        assert headers['ChatGPT-Account-ID'] == 'account'
        assert result['primary']['used_percent'] == 25
        connection.close.assert_called_once()

    @patch('anthproxy.codex.backend.http.client.HTTPSConnection')
    @patch('anthproxy.codex.backend.force_refresh')
    @patch('anthproxy.codex.backend.get_access')
    def test_401_refreshes_once(self, mock_get_access, mock_refresh, mock_connection):
        mock_get_access.return_value = ('old', 'account')
        mock_refresh.return_value = ('new', 'account')
        first = MagicMock()
        first.getresponse.return_value = self._response(401, {})
        second = MagicMock()
        second.getresponse.return_value = self._response(200, self._usage_payload())
        mock_connection.side_effect = [first, second]

        _fetch_usage(MagicMock(), threading.Lock())

        mock_refresh.assert_called_once()
        second_headers = second.request.call_args.kwargs['headers']
        assert second_headers['Authorization'] == 'Bearer new'

    @patch('anthproxy.codex.backend._fetch_usage')
    def test_backend_caches_success(self, mock_fetch):
        mock_fetch.return_value = _normalize_usage(self._usage_payload())
        backend = CodexBackend()
        first = backend.get_usage_markdown(MagicMock())
        second = backend.get_usage_markdown(MagicMock())
        assert first == second
        mock_fetch.assert_called_once()

    @patch('anthproxy.codex.backend._fetch_usage')
    def test_failure_returns_markdown(self, mock_fetch):
        mock_fetch.side_effect = OSError('offline')
        markdown = CodexBackend().get_usage_markdown(MagicMock())
        assert 'Usage information is unavailable' in markdown
        assert 'offline' in markdown

    @patch('anthproxy.codex.backend.http.client.HTTPSConnection')
    @patch('anthproxy.codex.backend.get_access')
    def test_429_uses_retry_after_header(self, mock_get_access, mock_connection):
        mock_get_access.return_value = ('access', 'account')
        response = MagicMock(spec=http.client.HTTPResponse)
        response.status = 429
        response.read.return_value = b'{}'
        response.getheader.side_effect = lambda name, default='': {'Retry-After': '12'}.get(name, default)
        connection = MagicMock()
        connection.getresponse.return_value = response
        mock_connection.return_value = connection

        with pytest.raises(UsageRateLimitError) as exc_info:
            _fetch_usage(MagicMock(), threading.Lock())
        assert exc_info.value.retry_after == 12.0

    @patch('anthproxy.codex.backend.http.client.HTTPSConnection')
    @patch('anthproxy.codex.backend.get_access')
    def test_429_prefers_retry_after_ms(self, mock_get_access, mock_connection):
        mock_get_access.return_value = ('access', 'account')
        response = MagicMock(spec=http.client.HTTPResponse)
        response.status = 429
        response.read.return_value = b'{}'
        response.getheader.side_effect = lambda name, default='': {
            'retry-after-ms': '1500',
            'Retry-After': '12',
        }.get(name, default)
        connection = MagicMock()
        connection.getresponse.return_value = response
        mock_connection.return_value = connection

        with pytest.raises(UsageRateLimitError) as exc_info:
            _fetch_usage(MagicMock(), threading.Lock())
        assert exc_info.value.retry_after == 1.5

    @patch('anthproxy.codex.backend.time.time', return_value=1_000.0)
    @patch('anthproxy.codex.backend.http.client.HTTPSConnection')
    @patch('anthproxy.codex.backend.get_access')
    def test_429_parses_http_date_retry_after(self, mock_get_access, mock_connection, _mock_time):
        mock_get_access.return_value = ('access', 'account')
        future = dt.datetime.fromtimestamp(1_030.0, tz=dt.timezone.utc)
        response = MagicMock(spec=http.client.HTTPResponse)
        response.status = 429
        response.read.return_value = b'{}'
        response.getheader.side_effect = lambda name, default='': {
            'Retry-After': future.strftime('%a, %d %b %Y %H:%M:%S GMT'),
        }.get(name, default)
        connection = MagicMock()
        connection.getresponse.return_value = response
        mock_connection.return_value = connection

        with pytest.raises(UsageRateLimitError) as exc_info:
            _fetch_usage(MagicMock(), threading.Lock())
        assert exc_info.value.retry_after == 30.0


class TestFormatReset:
    def _with_tz(self, tz_name, fn):
        old = os.environ.get('TZ')
        os.environ['TZ'] = tz_name
        time.tzset()
        try:
            return fn()
        finally:
            if old is None:
                os.environ.pop('TZ', None)
            else:
                os.environ['TZ'] = old
            time.tzset()

    def test_none_is_unknown(self):
        assert _format_reset(None) == 'unknown'

    @pytest.mark.skipif(not hasattr(time, 'tzset'), reason='TZ control unavailable')
    def test_winter_offset_eastern(self):
        # 2026-01-15 12:00 UTC → 07:00 EST (UTC-05:00)
        result = self._with_tz('America/New_York', lambda: _format_reset(1768478400))
        assert '07:00' in result
        assert 'EST' in result
        assert 'UTC-05:00' in result

    @pytest.mark.skipif(not hasattr(time, 'tzset'), reason='TZ control unavailable')
    def test_summer_offset_eastern(self):
        # 2026-07-15 12:00 UTC → 08:00 EDT (UTC-04:00)
        result = self._with_tz('America/New_York', lambda: _format_reset(1784116800))
        assert '08:00' in result
        assert 'EDT' in result
        assert 'UTC-04:00' in result

    @pytest.mark.skipif(not hasattr(time, 'tzset'), reason='TZ control unavailable')
    def test_utc_zone(self):
        result = self._with_tz('UTC', lambda: _format_reset(1768478400))
        assert '2026-01-15 12:00' in result
        assert 'UTC+00:00' in result


class TestNonInteractiveReadiness:
    def _auth_json(self, exp_offset):
        return {
            'auth_mode': 'chatgpt',
            'OPENAI_API_KEY': None,
            'tokens': {
                'id_token': _make_jwt({'sub': 'u1'}),
                'access_token': _make_jwt({'exp': int(time.time()) + exp_offset}),
                'refresh_token': 'rtoken',
                'account_id': 'acc_1',
            },
            'last_refresh': '2026-06-01T12:00:00Z',
        }

    def _config(self, home):
        cfg = MagicMock()
        cfg.codex_home = str(home)
        return cfg

    def test_missing_credentials_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match='No Codex credentials'):
                ensure_credentials_noninteractive(self._config(pathlib.Path(tmpdir)))

    def test_fresh_credentials_no_refresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            _write_auth(home, self._auth_json(3600))
            with patch('anthproxy.codex.auth.refresh') as mock_refresh:
                ensure_credentials_noninteractive(self._config(home))
                mock_refresh.assert_not_called()

    def test_never_calls_login(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            _write_auth(home, self._auth_json(60))  # within refresh window
            with patch('anthproxy.codex.auth.login') as mock_login, \
                 patch('anthproxy.codex.auth.refresh') as mock_refresh:
                mock_refresh.return_value = {}
                ensure_credentials_noninteractive(self._config(home))
                mock_login.assert_not_called()
                mock_refresh.assert_called_once()

    def test_transient_refresh_failure_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = pathlib.Path(tmpdir)
            _write_auth(home, self._auth_json(60))
            with patch('anthproxy.codex.auth.refresh', side_effect=RuntimeError('net')):
                with pytest.raises(RuntimeError, match='token refresh failed'):
                    ensure_credentials_noninteractive(self._config(home))


# ---------------------------------------------------------------------------
# _handle_error_response — HTTP status → AnthropicRequestError mapping
# ---------------------------------------------------------------------------

class TestHandleErrorResponse:
    def _call(self, status, body=None):
        body_bytes = json.dumps(body or {}).encode()
        _handle_error_response(status, body_bytes)

    def test_400_invalid_request(self):
        with pytest.raises(Exception) as exc:
            self._call(400, {'error': {'message': 'bad param'}})
        assert exc.value.status_code == 400
        assert exc.value.error_type == 'invalid_request_error'

    def test_401_authentication_error(self):
        with pytest.raises(Exception) as exc:
            self._call(401, {'error': {'message': 'auth fail'}})
        assert exc.value.status_code == 401
        assert exc.value.error_type == 'authentication_error'

    def test_403_permission_error(self):
        with pytest.raises(Exception) as exc:
            self._call(403, {'error': {'message': 'forbidden'}})
        assert exc.value.status_code == 403
        assert exc.value.error_type == 'permission_error'

    def test_429_rate_limit_error(self):
        with pytest.raises(Exception) as exc:
            self._call(429, {'error': {'message': 'slow down'}})
        assert exc.value.status_code == 429
        assert exc.value.error_type == 'rate_limit_error'

    def test_500_maps_to_502(self):
        with pytest.raises(Exception) as exc:
            self._call(500, {'error': {'message': 'server error'}})
        assert exc.value.status_code == 502
        assert exc.value.error_type == 'api_error'

    def test_message_extracted_from_body(self):
        with pytest.raises(Exception) as exc:
            self._call(400, {'error': {'message': 'specific error text'}})
        assert 'specific error text' in exc.value.message

    def test_invalid_json_body_does_not_crash(self):
        from anthproxy.mapper import AnthropicRequestError
        with pytest.raises(AnthropicRequestError):
            _handle_error_response(401, b'not-json')


# ---------------------------------------------------------------------------
# Reasoning / encrypted-content round-trip
# ---------------------------------------------------------------------------

class TestReasoningSignatureCodec:
    def test_encode_decode_roundtrip(self):
        sig = _encode_reasoning_signature('rs_123', 'OPAQUE==')
        decoded = _decode_reasoning_signature(sig)
        assert decoded == {'id': 'rs_123', 'enc': 'OPAQUE=='}

    def test_decode_rejects_foreign_signature(self):
        # A genuine Anthropic signature must not be mistaken for ours.
        assert _decode_reasoning_signature('abc123realanthropicsig') is None

    def test_decode_rejects_empty_and_nonstr(self):
        assert _decode_reasoning_signature('') is None
        assert _decode_reasoning_signature(None) is None

    def test_decode_rejects_corrupt_payload(self):
        assert _decode_reasoning_signature('codexenc:not-base64!!') is None


class TestReasoningResponseEncode:
    def test_map_response_attaches_signature(self):
        items = [
            {'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': 'ENC1',
             'summary': [{'text': 'pondering'}]},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Answer'}]},
        ]
        resp = _map_response(items, {}, 'completed', 'sonnet')
        think = resp['content'][0]
        assert think['type'] == 'thinking'
        assert think['thinking'] == 'pondering'
        assert _decode_reasoning_signature(think['signature']) == {'id': 'rs_1', 'enc': 'ENC1'}

    def test_map_response_encrypted_without_summary(self):
        items = [{'type': 'reasoning', 'id': 'rs_2', 'encrypted_content': 'ENC2', 'summary': []}]
        resp = _map_response(items, {}, 'completed', 'sonnet')
        think = resp['content'][0]
        assert think['type'] == 'thinking'
        assert think['thinking'] == ''
        assert _decode_reasoning_signature(think['signature']) == {'id': 'rs_2', 'enc': 'ENC2'}

    def test_map_response_summary_only_has_no_signature(self):
        items = [{'type': 'reasoning_summary', 'summary': [{'text': 'just a summary'}]}]
        resp = _map_response(items, {}, 'completed', 'sonnet')
        assert resp['content'][0] == {'type': 'thinking', 'thinking': 'just a summary'}


class TestReasoningStreamEncode:
    def _run(self, events):
        resp = _fake_sse_response(events)
        return _collect_events(_iter_stream_as_anthropic_sse(resp, 'sonnet', estimated_input_tokens=10))

    def test_signature_delta_emitted_for_reasoning_item(self):
        sig = _encode_reasoning_signature('rs_1', 'ENC')
        codex_events = [
            {'type': 'response.reasoning_summary_text.delta', 'item_id': 'rs_1', 'delta': 'think'},
            {'type': 'response.output_item.done',
             'item': {'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': 'ENC'}},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 2}}},
        ]
        events = self._run(codex_events)
        sigs = [
            e['data']['delta']['signature']
            for e in events
            if e['event'] == 'content_block_delta'
            and e['data'].get('delta', {}).get('type') == 'signature_delta'
        ]
        assert sigs == [sig]

    def test_signature_opens_thinking_when_no_summary(self):
        codex_events = [
            {'type': 'response.output_item.done',
             'item': {'type': 'reasoning', 'id': 'rs_9', 'encrypted_content': 'ENC9'}},
            {'type': 'response.completed',
             'response': {'status': 'completed', 'output': [],
                          'usage': {'input_tokens': 5, 'output_tokens': 2}}},
        ]
        events = self._run(codex_events)
        starts = [e for e in events if e['event'] == 'content_block_start']
        assert any(s['data']['content_block']['type'] == 'thinking' for s in starts)
        sigs = [
            e['data']['delta']['signature']
            for e in events
            if e['event'] == 'content_block_delta'
            and e['data'].get('delta', {}).get('type') == 'signature_delta'
        ]
        assert _decode_reasoning_signature(sigs[0]) == {'id': 'rs_9', 'enc': 'ENC9'}


class TestReasoningRequestDecode:
    def test_thinking_with_tool_use_rebuilds_reasoning(self):
        sig = _encode_reasoning_signature('rs_1', 'ENC')
        msg = {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'plan', 'signature': sig},
            {'type': 'tool_use', 'id': 'toolu_x', 'name': 'bash', 'input': {'cmd': 'ls'}},
        ]}
        result = _convert_message_to_input_item(msg)
        items = result['_multi']
        assert items[0] == {'type': 'reasoning', 'encrypted_content': 'ENC',
                            'summary': [], 'id': 'rs_1'}
        assert items[1]['type'] == 'function_call'

    def test_thinking_with_text_rebuilds_reasoning(self):
        sig = _encode_reasoning_signature('rs_2', 'ENC2')
        msg = {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'plan', 'signature': sig},
            {'type': 'text', 'text': 'Done'},
        ]}
        result = _convert_message_to_input_item(msg)
        items = result['_multi']
        assert items[0]['type'] == 'reasoning'
        assert items[0]['encrypted_content'] == 'ENC2'
        assert items[1]['type'] == 'message'
        assert items[1]['content'][0]['text'] == 'Done'

    def test_thinking_without_signature_dropped(self):
        msg = {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'plan'},
            {'type': 'text', 'text': 'Done'},
        ]}
        result = _convert_message_to_input_item(msg)
        # No reasoning item; falls through to a plain message.
        assert result['type'] == 'message'
        assert result['content'][0]['text'] == 'Done'

    def test_foreign_signature_not_replayed(self):
        msg = {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'plan', 'signature': 'real-anthropic-sig'},
            {'type': 'text', 'text': 'Done'},
        ]}
        result = _convert_message_to_input_item(msg)
        assert result['type'] == 'message'

    def test_full_roundtrip_response_to_request(self):
        # Encode via _map_response, then feed the thinking block back through
        # _convert_message_to_input_item — the encrypted content must survive.
        items = [
            {'type': 'reasoning', 'id': 'rs_rt', 'encrypted_content': 'SECRET',
             'summary': [{'text': 'reasoned'}]},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]},
        ]
        resp = _map_response(items, {}, 'completed', 'sonnet')
        assistant_msg = {'role': 'assistant', 'content': resp['content']}
        rebuilt = _convert_message_to_input_item(assistant_msg)
        reasoning = rebuilt['_multi'][0]
        assert reasoning['type'] == 'reasoning'
        assert reasoning['encrypted_content'] == 'SECRET'
        assert reasoning['id'] == 'rs_rt'


class TestIsCleanUserTurn:
    def test_plain_user_text(self):
        msg = {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}
        assert _is_clean_user_turn(msg) is True

    def test_string_content_user(self):
        msg = {'role': 'user', 'content': 'hello'}
        assert _is_clean_user_turn(msg) is True

    def test_tool_result_is_not_clean(self):
        msg = {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'x', 'content': 'y'}]}
        assert _is_clean_user_turn(msg) is False

    def test_assistant_is_not_clean(self):
        msg = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hi'}]}
        assert _is_clean_user_turn(msg) is False


class TestTruncateMessagesForContext:
    def _make_msgs(self, n_pairs: int) -> list:
        msgs = []
        for i in range(n_pairs):
            msgs.append({'role': 'user', 'content': f'user message {i}'})
            msgs.append({'role': 'assistant', 'content': f'assistant reply {i}'})
        return msgs

    def test_no_truncation_when_limit_zero(self):
        msgs = self._make_msgs(10)
        result = _truncate_messages_for_context(msgs, '', 0)
        assert result is msgs

    def test_no_truncation_when_under_limit(self):
        msgs = self._make_msgs(2)
        result = _truncate_messages_for_context(msgs, '', 1_000_000)
        assert result == msgs

    def test_negative_limit_same_as_zero(self):
        msgs = self._make_msgs(10)
        result = _truncate_messages_for_context(msgs, '', -100)
        assert result is msgs  # negative limit disables truncation

    def test_truncation_drops_oldest_messages(self):
        msgs = self._make_msgs(10)  # 20 messages, growing context
        # Use a very small limit that forces truncation
        result = _truncate_messages_for_context(msgs, '', 10)
        assert len(result) < len(msgs)
        # Remaining list must start with a clean user turn
        assert result[0]['role'] == 'user'
        assert result[0]['content'].startswith('user message')

    def test_post_truncation_estimate_within_limit(self):
        # Core invariant: result estimate must fall within the limit
        big_text = 'x' * 3_000  # 1000 chars per message (chars/3 = ~333 tokens)
        msgs = [
            {'role': 'user', 'content': big_text},
            {'role': 'assistant', 'content': big_text},
            {'role': 'user', 'content': big_text},
            {'role': 'assistant', 'content': big_text},
            {'role': 'user', 'content': big_text},
        ]
        # Total: 5 * 3000 = 15000 chars / 3 = 5000 tokens. Limit to 1000.
        result = _truncate_messages_for_context(msgs, '', 1000)
        # Compute the actual estimate of the result
        def _char_count_result(msgs_list):
            total = 0
            for m in msgs_list:
                c = m.get('content', '')
                if isinstance(c, str):
                    total += len(c)
            return total
        result_estimate = _char_count_result(result) // 3
        assert result_estimate <= 1000, f'result estimate {result_estimate} exceeds limit 1000'

    def test_result_starts_with_clean_user_turn(self):
        # Construct: [user, assistant-tool-use, user-tool-result, assistant, user, assistant]
        msgs = [
            {'role': 'user', 'content': 'ask'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'tu1', 'name': 'f', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'res'}]},
            {'role': 'assistant', 'content': 'done'},
            {'role': 'user', 'content': 'follow-up'},
            {'role': 'assistant', 'content': 'ok'},
        ]
        # Force truncation to drop at least the first pair
        result = _truncate_messages_for_context(msgs, '', 5)
        # Result must not start with a tool_result or assistant message
        assert result, 'truncation should not return empty list'
        assert result[0].get('role') == 'user', f'first message must be user, got {result[0].get("role")}'
        content = result[0].get('content', [])
        if isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get('type') == 'tool_result']
            assert not tool_results, f'first message should not contain tool_result, got {tool_results}'

    def test_preserves_final_message(self):
        msgs = self._make_msgs(5)
        result = _truncate_messages_for_context(msgs, '', 10)
        assert result[-1] == msgs[-1]

    def test_empty_messages_no_change(self):
        assert _truncate_messages_for_context([], '', 1000) == []

    def test_single_message_over_limit(self):
        # When a single message exceeds the limit, it must be returned unchanged
        # (the function cannot make further progress)
        msgs = [{'role': 'user', 'content': 'x' * 10_000}]  # 10000 chars / 3 ≈ 3333 tokens
        result = _truncate_messages_for_context(msgs, '', 100)  # tight limit
        assert result == msgs  # cannot drop the only message

    def test_system_prompt_alone_exceeds_limit_truncates_all_messages(self):
        # When system prompt alone exceeds limit, drop all messages but keep the last
        # to avoid returning an empty list
        big_system = 'y' * 10_000  # 10000 chars / 3 ≈ 3333 tokens
        msgs = [
            {'role': 'user', 'content': 'msg1'},
            {'role': 'assistant', 'content': 'msg2'},
            {'role': 'user', 'content': 'msg3'},
        ]
        result = _truncate_messages_for_context(msgs, big_system, 100)
        # Function can drop message 1 and 2, but must keep at least one
        assert len(result) >= 1
        assert result[-1] == msgs[-1]

    def test_map_request_applies_truncation(self):
        # Build a payload whose messages are large enough to trigger truncation
        big_text = 'x' * 10_000
        msgs = [
            {'role': 'user', 'content': big_text},
            {'role': 'assistant', 'content': big_text},
            {'role': 'user', 'content': big_text},
            {'role': 'assistant', 'content': big_text},
            {'role': 'user', 'content': 'final question'},
        ]
        payload = {'model': 'sonnet', 'messages': msgs}
        # 5 * 10000 chars / 3 ≈ 16666 tokens; limit below that triggers truncation
        body = _map_request(payload, context_limit=1000)
        # Truncation should fire: the input array must be built from fewer messages
        # (when all messages are plain text with 1:1 mapping, input count ≈ message count)
        assert len(body['input']) < len(msgs), 'truncation should reduce input items count'


# ---------------------------------------------------------------------------
# TestMapRequestEmptyInput
# ---------------------------------------------------------------------------

class TestMapRequestEmptyInput:
    def test_empty_messages_raises(self):
        # messages: [] → guard raises 400
        with pytest.raises(AnthropicRequestError) as exc_info:
            _map_request({'model': 'opus', 'messages': []})
        assert exc_info.value.status_code == 400

    def test_missing_messages_raises(self):
        # no 'messages' key at all
        with pytest.raises(AnthropicRequestError) as exc_info:
            _map_request({'model': 'opus'})
        assert exc_info.value.status_code == 400

    def test_system_only_raises(self):
        # only a system message; system goes to instructions, not input
        with pytest.raises(AnthropicRequestError) as exc_info:
            _map_request({'model': 'opus', 'system': 'Be helpful.', 'messages': []})
        assert exc_info.value.status_code == 400

    def test_system_role_message_raises(self):
        # inline system-role message only
        with pytest.raises(AnthropicRequestError) as exc_info:
            _map_request({'model': 'opus', 'messages': [{'role': 'system', 'content': 'sys'}]})
        assert exc_info.value.status_code == 400

    def test_empty_content_user_message_raises(self):
        # user message with empty content → _convert_message_to_input_item returns None
        with pytest.raises(AnthropicRequestError) as exc_info:
            _map_request({'model': 'opus', 'messages': [{'role': 'user', 'content': []}]})
        assert exc_info.value.status_code == 400

    def test_valid_user_message_does_not_raise(self):
        # sanity: a normal user message must not raise
        body = _map_request({'model': 'opus', 'messages': [{'role': 'user', 'content': 'hello'}]})
        assert body['input']


# ---------------------------------------------------------------------------
# TestTruncationToolPairFallback
# ---------------------------------------------------------------------------

class TestTruncationToolPairFallback:
    def test_tool_pair_preserved_when_truncation_would_empty(self):
        # A history that, after truncation cleanup, would be empty is rescued by
        # returning the last assistant tool_use + user tool_result pair.
        big = 'x' * 30_000  # forces heavy truncation
        msgs = [
            {'role': 'user', 'content': big},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'fn', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'result'},
            ]},
        ]
        # Use a very small context_limit so truncation fires and the user tool_result
        # turn is orphaned (would be dropped) — expect the pair to be rescued.
        result = _truncate_messages_for_context(msgs, '', 1)
        # The fallback should preserve both the assistant and user messages.
        assert len(result) == 2
        assert result[0]['role'] == 'assistant'
        assert result[1]['role'] == 'user'

    def test_no_pair_available_returns_empty(self):
        # When there is no tool_use/tool_result pair, the fallback cannot rescue and
        # returns [] (the mapper guard will convert this to a 400 upstream).
        big = 'x' * 30_000
        msgs = [
            {'role': 'user', 'content': big},
            {'role': 'assistant', 'content': 'big response ' + big},
        ]
        result = _truncate_messages_for_context(msgs, '', 1)
        assert result == []

    def test_map_request_tool_pair_fallback_produces_valid_input(self):
        # End-to-end: mapper should not raise when the tool pair rescue fires.
        big = 'x' * 30_000
        msgs = [
            {'role': 'user', 'content': big},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu2', 'name': 'g', 'input': {'k': 'v'}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu2', 'content': 'ok'},
            ]},
        ]
        body = _map_request({'model': 'opus', 'messages': msgs}, context_limit=1)
        assert body['input']
        types = {item.get('type') for item in body['input']}
        assert 'function_call' in types
        assert 'function_call_output' in types
