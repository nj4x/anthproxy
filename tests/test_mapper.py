"""Characterization tests for anthproxy/bedrock/mapper.py.

These tests pin the current behaviour of the mapper before any refactoring so
that Phase 4 refactors (dead-code removal, request/count-tokens factoring, SSE
builder extraction) can run safely against a green test suite.

Tests are intentionally exhaustive on the content-block dispatchers and on the
two request-mapping functions that share ~50 lines of copy-pasted logic.
"""

import base64
import json
import pytest

from anthproxy.mapper import AnthropicRequestError
from anthproxy.bedrock.mapper import (
    _decode_image_source,
    _guess_bedrock_model_id,
    _has_cache_control,
    _map_tool_result_content,
    _normalize_system_blocks,
    map_anthropic_content_to_bedrock,
    map_anthropic_request_to_bedrock,
    map_anthropic_tools_to_bedrock,
    map_count_tokens_request_to_bedrock,
    map_tool_choice_to_bedrock,
    normalize_model_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# normalize_model_id / _guess_bedrock_model_id
# ---------------------------------------------------------------------------

class TestNormalizeModelId:
    def test_empty_raises(self):
        with pytest.raises(AnthropicRequestError) as exc:
            normalize_model_id('')
        assert exc.value.status_code == 400

    def test_none_raises(self):
        with pytest.raises(AnthropicRequestError):
            normalize_model_id(None)

    def test_short_alias_sonnet(self):
        result = normalize_model_id('sonnet')
        assert result.startswith('anthropic.')

    def test_short_alias_opus(self):
        result = normalize_model_id('opus')
        assert result.startswith('anthropic.')

    def test_context_suffix_bracket_alias(self):
        result = normalize_model_id('sonnet[1m]')
        assert '1m' in result.lower()

    def test_context_suffix_colon_alias(self):
        result = normalize_model_id('opus:1m')
        assert '1m' in result.lower()

    def test_bedrock_prefix_passthrough(self):
        model = 'anthropic.claude-3-haiku-20240307-v1:0'
        assert normalize_model_id(model) == model

    def test_arn_passthrough(self):
        model = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3'
        assert normalize_model_id(model) == model

    def test_inference_profile_prefix_passthrough(self):
        model = 'us.anthropic.claude-3-haiku-20240307-v1:0'
        assert normalize_model_id(model) == model

    def test_unknown_claude_guessed(self):
        model = 'claude-new-model-20991231'
        result = normalize_model_id(model)
        assert result.startswith('anthropic.')
        assert 'new-model' in result

    def test_non_claude_unknown_passthrough(self):
        model = 'some-completely-unknown-model'
        assert normalize_model_id(model) == model


class TestGuessBedrock:
    def test_adds_anthropic_prefix(self):
        result = _guess_bedrock_model_id('claude-opus-4-8')
        assert result.startswith('anthropic.')

    def test_appends_v1_0_when_date_suffix(self):
        result = _guess_bedrock_model_id('claude-sonnet-4-5-20250929')
        assert result.endswith('-v1:0')

    def test_no_v1_0_without_date(self):
        result = _guess_bedrock_model_id('claude-sonnet-4-6')
        assert not result.endswith('-v1:0')

    def test_1m_suffix_preserved(self):
        result = _guess_bedrock_model_id('claude-opus-4-8[1m]')
        assert result.endswith(':1m')
        assert '[1m]' not in result

    def test_colon_1m_suffix_preserved(self):
        result = _guess_bedrock_model_id('claude-opus-4-8:1m')
        assert result.endswith(':1m')


# ---------------------------------------------------------------------------
# _normalize_system_blocks
# ---------------------------------------------------------------------------

class TestNormalizeSystemBlocks:
    def test_none_returns_empty(self):
        assert _normalize_system_blocks(None) == []

    def test_string_wraps_in_text(self):
        result = _normalize_system_blocks('hello')
        assert result == [{'text': 'hello'}]

    def test_list_of_text_blocks(self):
        blocks = [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]
        result = _normalize_system_blocks(blocks)
        assert result == [{'text': 'a'}, {'text': 'b'}]

    def test_cache_control_inserts_cache_point(self):
        blocks = [{'type': 'text', 'text': 'x', 'cache_control': {'type': 'ephemeral'}}]
        result = _normalize_system_blocks(blocks)
        assert result[0] == {'text': 'x'}
        assert result[1] == {'cachePoint': {'type': 'default'}}

    def test_non_text_block_raises(self):
        blocks = [{'type': 'image', 'source': {}}]
        with pytest.raises(AnthropicRequestError):
            _normalize_system_blocks(blocks)

    def test_invalid_type_raises(self):
        with pytest.raises(AnthropicRequestError):
            _normalize_system_blocks(123)


# ---------------------------------------------------------------------------
# _decode_image_source
# ---------------------------------------------------------------------------

class TestDecodeImageSource:
    def _make_source(self, **kwargs):
        base = {'type': 'base64', 'media_type': 'image/png', 'data': _b64(b'PNGDATA')}
        base.update(kwargs)
        return base

    def test_valid_base64_png(self):
        result = _decode_image_source(self._make_source())
        assert 'image' in result
        assert result['image']['format'] == 'png'

    def test_jpeg_format(self):
        result = _decode_image_source(self._make_source(media_type='image/jpeg'))
        assert result['image']['format'] == 'jpeg'

    def test_url_type_raises(self):
        with pytest.raises(AnthropicRequestError, match='URL'):
            _decode_image_source({'type': 'url', 'url': 'http://example.com/img.png'})

    def test_unknown_type_raises(self):
        with pytest.raises(AnthropicRequestError):
            _decode_image_source({'type': 'file', 'file_id': 'abc'})

    def test_missing_media_type_raises(self):
        with pytest.raises(AnthropicRequestError):
            _decode_image_source({'type': 'base64', 'media_type': 'notseparated', 'data': _b64(b'x')})

    def test_not_a_dict_raises(self):
        with pytest.raises(AnthropicRequestError):
            _decode_image_source('inline-string')


# ---------------------------------------------------------------------------
# map_anthropic_content_to_bedrock — all branches
# ---------------------------------------------------------------------------

class TestMapContentToBedrock:
    def test_text_block(self):
        result = map_anthropic_content_to_bedrock([{'type': 'text', 'text': 'hello'}])
        assert result == [{'text': 'hello'}]

    def test_string_content(self):
        result = map_anthropic_content_to_bedrock('hello world')
        assert result == [{'text': 'hello world'}]

    def test_image_block(self):
        src = {'type': 'base64', 'media_type': 'image/png', 'data': _b64(b'IMG')}
        result = map_anthropic_content_to_bedrock([{'type': 'image', 'source': src}])
        assert result[0]['image']['format'] == 'png'

    def test_tool_use_block(self):
        block = {'type': 'tool_use', 'id': 'toolu_001', 'name': 'get_weather', 'input': {'city': 'SF'}}
        result = map_anthropic_content_to_bedrock([block])
        assert result[0]['toolUse']['toolUseId'] == 'toolu_001'
        assert result[0]['toolUse']['name'] == 'get_weather'
        assert result[0]['toolUse']['input'] == {'city': 'SF'}

    def test_tool_use_generates_id_when_absent(self):
        block = {'type': 'tool_use', 'name': 'my_tool', 'input': {}}
        result = map_anthropic_content_to_bedrock([block])
        assert result[0]['toolUse']['toolUseId'].startswith('toolu_')

    def test_tool_use_missing_name_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_content_to_bedrock([{'type': 'tool_use', 'id': 'x', 'input': {}}])

    def test_tool_result_string_content(self):
        block = {'type': 'tool_result', 'tool_use_id': 'toolu_001', 'content': 'the result'}
        result = map_anthropic_content_to_bedrock([block])
        assert result[0]['toolResult']['toolUseId'] == 'toolu_001'
        assert result[0]['toolResult']['content'] == [{'text': 'the result'}]
        assert result[0]['toolResult']['status'] == 'success'

    def test_tool_result_error_flag(self):
        block = {'type': 'tool_result', 'tool_use_id': 'toolu_001', 'content': 'err', 'is_error': True}
        result = map_anthropic_content_to_bedrock([block])
        assert result[0]['toolResult']['status'] == 'error'

    def test_tool_result_missing_tool_use_id_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_content_to_bedrock([{'type': 'tool_result', 'content': 'x'}])

    def test_thinking_block(self):
        block = {'type': 'thinking', 'thinking': 'let me reason', 'signature': 'sig123'}
        result = map_anthropic_content_to_bedrock([block])
        rc = result[0]['reasoningContent']
        assert rc['reasoningText']['text'] == 'let me reason'
        assert rc['reasoningText']['signature'] == 'sig123'

    def test_thinking_block_no_signature(self):
        block = {'type': 'thinking', 'thinking': 'reasoning'}
        result = map_anthropic_content_to_bedrock([block])
        assert 'signature' not in result[0]['reasoningContent']['reasoningText']

    def test_redacted_thinking_block(self):
        data_b64 = _b64(b'encrypted-thinking-blob')
        block = {'type': 'redacted_thinking', 'data': data_b64}
        result = map_anthropic_content_to_bedrock([block])
        rc = result[0]['reasoningContent']
        assert rc['redactedContent'] == base64.b64decode(data_b64)

    def test_redacted_thinking_invalid_base64_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_content_to_bedrock([{'type': 'redacted_thinking', 'data': '!!!bad!!!'}])

    def test_cache_control_appends_cache_point(self):
        block = {'type': 'text', 'text': 'cached', 'cache_control': {'type': 'ephemeral'}}
        result = map_anthropic_content_to_bedrock([block])
        assert result[0] == {'text': 'cached'}
        assert result[1] == {'cachePoint': {'type': 'default'}}

    def test_unsupported_type_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_content_to_bedrock([{'type': 'document', 'source': {}}])

    def test_non_dict_block_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_content_to_bedrock([['bad', 'block']])


# ---------------------------------------------------------------------------
# map_anthropic_tools_to_bedrock
# ---------------------------------------------------------------------------

class TestMapToolsToBedrock:
    def test_empty_returns_none(self):
        assert map_anthropic_tools_to_bedrock([]) is None

    def test_none_returns_none(self):
        assert map_anthropic_tools_to_bedrock(None) is None

    def test_single_tool(self):
        tools = [{'name': 'get_weather', 'description': 'Get weather', 'input_schema': {'type': 'object'}}]
        result = map_anthropic_tools_to_bedrock(tools)
        assert len(result) == 1
        spec = result[0]['toolSpec']
        assert spec['name'] == 'get_weather'
        assert spec['description'] == 'Get weather'
        assert spec['inputSchema']['json'] == {'type': 'object'}

    def test_missing_name_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_tools_to_bedrock([{'description': 'no name'}])


# ---------------------------------------------------------------------------
# map_tool_choice_to_bedrock
# ---------------------------------------------------------------------------

class TestMapToolChoiceToBedrock:
    def test_none_returns_auto(self):
        assert map_tool_choice_to_bedrock(None) == {'auto': {}}

    def test_string_auto(self):
        assert map_tool_choice_to_bedrock('auto') == {'auto': {}}

    def test_string_any(self):
        assert map_tool_choice_to_bedrock('any') == {'any': {}}

    def test_string_none(self):
        assert map_tool_choice_to_bedrock('none') is None

    def test_dict_auto(self):
        assert map_tool_choice_to_bedrock({'type': 'auto'}) == {'auto': {}}

    def test_dict_any(self):
        assert map_tool_choice_to_bedrock({'type': 'any'}) == {'any': {}}

    def test_dict_tool(self):
        result = map_tool_choice_to_bedrock({'type': 'tool', 'name': 'get_weather'})
        assert result == {'tool': {'name': 'get_weather'}}

    def test_unsupported_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_tool_choice_to_bedrock({'type': 'unknown'})


# ---------------------------------------------------------------------------
# map_anthropic_request_to_bedrock
# ---------------------------------------------------------------------------

_BASE_PAYLOAD = {
    'model': 'sonnet',
    'messages': [{'role': 'user', 'content': 'Hello'}],
}


class TestMapRequestToBedrock:
    def test_basic_request(self):
        result = map_anthropic_request_to_bedrock(_BASE_PAYLOAD)
        assert 'modelId' in result
        assert result['messages'][0]['role'] == 'user'
        assert result['messages'][0]['content'] == [{'text': 'Hello'}]

    def test_model_resolved(self):
        result = map_anthropic_request_to_bedrock(_BASE_PAYLOAD)
        assert result['modelId'].startswith('anthropic.')

    def test_system_string_included(self):
        payload = dict(_BASE_PAYLOAD, system='Be helpful')
        result = map_anthropic_request_to_bedrock(payload)
        assert any(b.get('text') == 'Be helpful' for b in result['system'])

    def test_system_list_blocks_included(self):
        payload = dict(_BASE_PAYLOAD, system=[{'type': 'text', 'text': 'You are helpful'}])
        result = map_anthropic_request_to_bedrock(payload)
        assert any(b.get('text') == 'You are helpful' for b in result['system'])

    def test_inline_system_message_folded_into_system(self):
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'system', 'content': 'Injected system'},
                {'role': 'user', 'content': 'Hi'},
            ],
        }
        result = map_anthropic_request_to_bedrock(payload)
        system_texts = [b.get('text') for b in result.get('system', [])]
        assert 'Injected system' in system_texts
        # The system message itself should not be in the messages array
        assert all(m['role'] in ('user', 'assistant') for m in result['messages'])

    def test_max_tokens_in_inference_config(self):
        payload = dict(_BASE_PAYLOAD, max_tokens=1024)
        result = map_anthropic_request_to_bedrock(payload)
        assert result['inferenceConfig']['maxTokens'] == 1024

    def test_temperature_in_inference_config(self):
        payload = dict(_BASE_PAYLOAD, temperature=0.7)
        result = map_anthropic_request_to_bedrock(payload)
        assert result['inferenceConfig']['temperature'] == 0.7

    def test_no_inference_config_when_absent(self):
        result = map_anthropic_request_to_bedrock(_BASE_PAYLOAD)
        assert 'inferenceConfig' not in result

    def test_tools_mapped(self):
        payload = dict(_BASE_PAYLOAD, tools=[
            {'name': 'tool1', 'description': 'd', 'input_schema': {'type': 'object'}}
        ])
        result = map_anthropic_request_to_bedrock(payload)
        assert 'toolConfig' in result
        assert result['toolConfig']['tools'][0]['toolSpec']['name'] == 'tool1'

    def test_thinking_in_additional_fields(self):
        payload = dict(_BASE_PAYLOAD, thinking={'type': 'enabled', 'budget_tokens': 1000})
        result = map_anthropic_request_to_bedrock(payload)
        assert result['additionalModelRequestFields']['thinking'] == {'type': 'enabled', 'budget_tokens': 1000}

    def test_empty_messages_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_request_to_bedrock({'model': 'sonnet', 'messages': []})

    def test_non_dict_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_anthropic_request_to_bedrock([])

    def test_stop_sequences_in_inference_config(self):
        payload = dict(_BASE_PAYLOAD, stop_sequences=['END'])
        result = map_anthropic_request_to_bedrock(payload)
        assert result['inferenceConfig']['stopSequences'] == ['END']


# ---------------------------------------------------------------------------
# map_count_tokens_request_to_bedrock
# — must produce identical messages/system assembly to map_anthropic_request_to_bedrock
# ---------------------------------------------------------------------------

class TestMapCountTokensRequest:
    def test_basic_structure(self):
        result = map_count_tokens_request_to_bedrock(_BASE_PAYLOAD)
        assert 'modelId' in result
        assert 'messages' in result

    def test_same_message_assembly_as_request_mapper(self):
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'system', 'content': 'Be helpful'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there'},
            ],
        }
        request_result = map_anthropic_request_to_bedrock(payload)
        count_result = map_count_tokens_request_to_bedrock(payload)

        # Messages list must be identical
        assert request_result['messages'] == count_result['messages']
        # System list must be identical
        assert request_result.get('system') == count_result.get('system')
        # Model must be identical
        assert request_result['modelId'] == count_result['modelId']

    def test_same_system_string_handling(self):
        payload = dict(_BASE_PAYLOAD, system='System context')
        request_result = map_anthropic_request_to_bedrock(payload)
        count_result = map_count_tokens_request_to_bedrock(payload)
        assert request_result.get('system') == count_result.get('system')

    def test_same_tools_mapping(self):
        payload = dict(_BASE_PAYLOAD, tools=[
            {'name': 'search', 'description': 'Search', 'input_schema': {'type': 'object'}}
        ])
        request_result = map_anthropic_request_to_bedrock(payload)
        count_result = map_count_tokens_request_to_bedrock(payload)
        assert request_result['toolConfig']['tools'] == count_result['toolConfig']['tools']

    def test_count_tokens_excludes_inference_config(self):
        payload = dict(_BASE_PAYLOAD, max_tokens=512, temperature=0.5)
        result = map_count_tokens_request_to_bedrock(payload)
        # Count-tokens must NOT include inferenceConfig — it doesn't make sense for counting
        assert 'inferenceConfig' not in result

    def test_count_tokens_excludes_thinking(self):
        payload = dict(_BASE_PAYLOAD, thinking={'type': 'enabled', 'budget_tokens': 1000})
        result = map_count_tokens_request_to_bedrock(payload)
        assert 'additionalModelRequestFields' not in result

    def test_empty_messages_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_count_tokens_request_to_bedrock({'model': 'sonnet', 'messages': []})

    def test_non_dict_raises(self):
        with pytest.raises(AnthropicRequestError):
            map_count_tokens_request_to_bedrock('bad')

    def test_same_cache_control_on_system(self):
        payload = dict(
            _BASE_PAYLOAD,
            system=[{'type': 'text', 'text': 'ctx', 'cache_control': {'type': 'ephemeral'}}]
        )
        request_result = map_anthropic_request_to_bedrock(payload)
        count_result = map_count_tokens_request_to_bedrock(payload)
        # cache_control must be translated identically (cachePoint blocks present)
        assert request_result['system'] == count_result['system']


# ---------------------------------------------------------------------------
# _has_cache_control helper
# ---------------------------------------------------------------------------

class TestHasCacheControl:
    def test_ephemeral_returns_true(self):
        assert _has_cache_control({'cache_control': {'type': 'ephemeral'}}) is True

    def test_missing_returns_false(self):
        assert _has_cache_control({'type': 'text', 'text': 'x'}) is False

    def test_non_dict_returns_false(self):
        assert _has_cache_control('plain string') is False

    def test_unknown_type_returns_false(self):
        assert _has_cache_control({'cache_control': {'type': 'persistent'}}) is False


# ---------------------------------------------------------------------------
# _map_tool_result_content
# ---------------------------------------------------------------------------

class TestMapToolResultContent:
    def test_string_wraps_in_text(self):
        result = _map_tool_result_content('result text')
        assert result == [{'text': 'result text'}]

    def test_list_with_text_block(self):
        result = _map_tool_result_content([{'type': 'text', 'text': 'ok'}])
        assert result == [{'text': 'ok'}]

    def test_list_with_image_block(self):
        src = {'type': 'base64', 'media_type': 'image/png', 'data': _b64(b'IMG')}
        result = _map_tool_result_content([{'type': 'image', 'source': src}])
        assert 'image' in result[0]

    def test_invalid_type_raises(self):
        with pytest.raises(AnthropicRequestError):
            _map_tool_result_content([{'type': 'video'}])

    def test_non_list_non_string_raises(self):
        with pytest.raises(AnthropicRequestError):
            _map_tool_result_content(123)


# ---------------------------------------------------------------------------
# estimate_input_tokens — text heuristic + calibrated tool-use overhead
# ---------------------------------------------------------------------------

class TestEstimateInputTokens:
    def _est(self, payload):
        from anthproxy.mapper.common import estimate_input_tokens
        return estimate_input_tokens(payload)

    def test_text_only_uses_four_chars_per_token(self):
        payload = {'model': 'sonnet',
                   'messages': [{'role': 'user', 'content': 'x' * 4000}]}
        assert self._est(payload) == 1000

    def test_system_and_messages_counted(self):
        payload = {'model': 'sonnet', 'system': 'y' * 400,
                   'messages': [{'role': 'user', 'content': 'x' * 400}]}
        assert self._est(payload) == 200  # (400 + 400) // 4

    def test_empty_messages_returns_at_least_one(self):
        assert self._est({'model': 'sonnet', 'messages': []}) == 1

    def test_tools_add_base_and_per_tool_overhead(self):
        from anthproxy.mapper.common import (
            _TOOL_USE_BASE_OVERHEAD, _TOOL_FRAMING_OVERHEAD, _count_chars)
        tool = {'name': 'a', 'description': 'b',
                'input_schema': {'type': 'object'}}
        msg = 'hi'
        with_tools = {'model': 'sonnet', 'tools': [tool, tool, tool],
                      'messages': [{'role': 'user', 'content': msg}]}
        # Composition: (all text chars) // 4 + base + framing*n_tools.
        total_chars = len(msg) + _count_chars(tool) * 3
        expected = (total_chars // 4
                    + _TOOL_USE_BASE_OVERHEAD + _TOOL_FRAMING_OVERHEAD * 3)
        assert self._est(with_tools) == expected

    def test_no_tool_overhead_when_tools_absent_or_empty(self):
        base = {'model': 'sonnet',
                'messages': [{'role': 'user', 'content': 'x' * 40}]}
        assert self._est(base) == self._est(dict(base, tools=[]))

    def test_more_tools_increase_estimate(self):
        tool = {'name': 'a', 'description': 'b', 'input_schema': {}}
        few = {'model': 'sonnet', 'tools': [tool],
               'messages': [{'role': 'user', 'content': 'hi'}]}
        many = dict(few, tools=[tool] * 10)
        assert self._est(many) > self._est(few)


class TestStripCodexThinkingBlocks:
    def _strip(self, messages):
        from anthproxy.mapper.common import strip_codex_thinking_blocks
        return strip_codex_thinking_blocks(messages)

    def _sig(self, payload='OPAQUE=='):
        from anthproxy.mapper.common import CODEX_REASONING_SIG_PREFIX
        import base64
        raw = json.dumps({'id': 'rs_1', 'enc': payload}).encode()
        return CODEX_REASONING_SIG_PREFIX + base64.b64encode(raw).decode()

    def test_returns_same_list_when_nothing_to_strip(self):
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]},
        ]
        assert self._strip(msgs) is msgs

    def test_strips_codex_thinking_block_from_assistant(self):
        sig = self._sig()
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'ponder', 'signature': sig},
                {'type': 'text', 'text': 'answer'},
            ]},
        ]
        result = self._strip(msgs)
        assert result is not msgs
        assert len(result[1]['content']) == 1
        assert result[1]['content'][0] == {'type': 'text', 'text': 'answer'}

    def test_preserves_genuine_anthropic_signature(self):
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'think', 'signature': 'genuineAnthropicSig123'},
                {'type': 'text', 'text': 'answer'},
            ]},
        ]
        result = self._strip(msgs)
        assert result is msgs

    def test_degenerate_all_blocks_codex_inserts_empty_text(self):
        sig = self._sig()
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'only thinking', 'signature': sig},
            ]},
        ]
        result = self._strip(msgs)
        assert result[1]['content'] == [{'type': 'text', 'text': ''}]

    def test_strips_multiple_codex_blocks_keeps_text(self):
        sig = self._sig()
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'a', 'signature': sig},
                {'type': 'thinking', 'thinking': 'b', 'signature': sig},
                {'type': 'text', 'text': 'final'},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': 'final'}]

    def test_does_not_touch_user_messages(self):
        sig = self._sig()
        msgs = [
            {'role': 'user', 'content': [
                {'type': 'thinking', 'thinking': 'x', 'signature': sig},
            ]},
        ]
        result = self._strip(msgs)
        assert result is msgs

    def test_empty_list_returns_empty(self):
        assert self._strip([]) == []

    def test_non_list_content_assistant_passes_through(self):
        msgs = [{'role': 'assistant', 'content': 'plain string'}]
        assert self._strip(msgs) is msgs

    def test_strips_openrouter_tagged_thinking_block(self):
        # OpenRouter stamps `or:` onto its response signatures; on an
        # OpenRouter→Anthropic switch those must be stripped just like codex.
        from anthproxy.mapper.common import OPENROUTER_REASONING_SIG_PREFIX
        sig = OPENROUTER_REASONING_SIG_PREFIX + 'foreignBase64Sig=='
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'ponder', 'signature': sig},
                {'type': 'text', 'text': 'answer'},
            ]},
        ]
        result = self._strip(msgs)
        assert result is not msgs
        assert result[1]['content'] == [{'type': 'text', 'text': 'answer'}]

    def test_strips_both_codex_and_openrouter_blocks(self):
        from anthproxy.mapper.common import OPENROUTER_REASONING_SIG_PREFIX
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'a', 'signature': self._sig()},
                {'type': 'thinking', 'thinking': 'b',
                 'signature': OPENROUTER_REASONING_SIG_PREFIX + 'x'},
                {'type': 'text', 'text': 'final'},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': 'final'}]

    def test_preserves_genuine_anthropic_signature_unaffected_by_or_prefix(self):
        # A genuine Anthropic signature that happens to start with 'or' (but not
        # 'or:') must be preserved — the prefix match is on the full 'or:' sentinel.
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'think', 'signature': 'orderlyBase64Sig=='},
                {'type': 'text', 'text': 'answer'},
            ]},
        ]
        result = self._strip(msgs)
        assert result is msgs


class TestStripAllThinkingBlocks:
    """Tests for strip_all_thinking_blocks — the model-switch recovery fallback."""

    def _strip(self, messages):
        from anthproxy.mapper.common import strip_all_thinking_blocks
        return strip_all_thinking_blocks(messages)

    def test_returns_same_list_when_no_thinking(self):
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]},
        ]
        assert self._strip(msgs) is msgs

    def test_strips_native_anthropic_thinking_block(self):
        # Thinking blocks with raw Anthropic signatures (no sentinel prefix)
        # are stripped — these arise after a model-tier switch within Anthropic.
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'reason', 'signature': 'rawAnthropicOpusSig=='},
                {'type': 'text', 'text': 'answer'},
            ]},
        ]
        result = self._strip(msgs)
        assert result is not msgs
        assert result[1]['content'] == [{'type': 'text', 'text': 'answer'}]

    def test_strips_all_thinking_regardless_of_signature_prefix(self):
        # All thinking blocks are stripped regardless of prefix.
        from anthproxy.mapper.common import CODEX_REASONING_SIG_PREFIX, OPENROUTER_REASONING_SIG_PREFIX
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'a', 'signature': 'rawSig=='},
                {'type': 'thinking', 'thinking': 'b', 'signature': OPENROUTER_REASONING_SIG_PREFIX + 'x'},
                {'type': 'thinking', 'thinking': 'c', 'signature': CODEX_REASONING_SIG_PREFIX + 'y'},
                {'type': 'text', 'text': 'final'},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': 'final'}]

    def test_strips_redacted_thinking_blocks(self):
        # redacted_thinking blocks also carry model-specific data and trigger
        # "Invalid `data` in `redacted_thinking` block" (HTTP 400) on a model
        # switch; they must be stripped together with thinking blocks.
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'redacted_thinking', 'data': 'opaqueData=='},
                {'type': 'thinking', 'thinking': 'visible', 'signature': 'rawSig=='},
                {'type': 'text', 'text': 'answer'},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': 'answer'}]

    def test_degenerate_only_redacted_thinking_inserts_empty_text(self):
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'redacted_thinking', 'data': 'opaqueData=='},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': ''}]

    def test_degenerate_all_blocks_thinking_inserts_empty_text(self):
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'only', 'signature': 'rawSig=='},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': ''}]

    def test_does_not_touch_user_messages(self):
        msgs = [
            {'role': 'user', 'content': [
                {'type': 'thinking', 'thinking': 'x', 'signature': 'sig'},
            ]},
        ]
        assert self._strip(msgs) is msgs

    def test_empty_list_returns_empty(self):
        assert self._strip([]) == []

    def test_strips_across_multiple_assistant_messages(self):
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'a', 'signature': 'sig1'},
                {'type': 'text', 'text': 'first'},
            ]},
            {'role': 'user', 'content': 'continue'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'b', 'signature': 'sig2'},
                {'type': 'text', 'text': 'second'},
            ]},
        ]
        result = self._strip(msgs)
        assert result[0]['content'] == [{'type': 'text', 'text': 'first'}]
        assert result[2]['content'] == [{'type': 'text', 'text': 'second'}]
