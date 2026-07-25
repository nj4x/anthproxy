import json
import pathlib

from anthproxy.bedrock.token_estimator import BedrockTokenEstimator
from anthproxy.config import Config
from anthproxy.bedrock.mapper import (
    apply_inference_profile_model_id,
    _map_bedrock_usage_to_anthropic,
    iter_bedrock_stream_as_anthropic_sse,
    map_anthropic_request_to_bedrock,
    map_bedrock_response_to_anthropic,
)


def _parse_sse(chunks):
    events = []
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        event = next(line[7:] for line in lines if line.startswith('event: '))
        data = json.loads(next(line[6:] for line in lines if line.startswith('data: ')))
        events.append((event, data))
    return events


def test_usage_mapping_prefers_current_cache_fields():
    usage = {
        'inputTokens': 100,
        'outputTokens': 20,
        'cacheReadInputTokens': 80,
        'cacheWriteInputTokens': 10,
        'cacheReadInputTokenCount': 1,
        'cacheWriteInputTokenCount': 2,
    }
    mapped = _map_bedrock_usage_to_anthropic(usage)
    assert mapped['cache_read_input_tokens'] == 80
    assert mapped['cache_creation_input_tokens'] == 10


def test_usage_mapping_falls_back_to_legacy_cache_fields():
    usage = {
        'inputTokens': 100,
        'outputTokens': 20,
        'cacheReadInputTokenCount': 80,
        'cacheWriteInputTokenCount': 10,
    }
    mapped = _map_bedrock_usage_to_anthropic(usage)
    assert mapped['cache_read_input_tokens'] == 80
    assert mapped['cache_creation_input_tokens'] == 10


def test_non_streaming_response_includes_current_cache_usage():
    response = {
        'output': {'message': {'role': 'assistant', 'content': [{'text': 'Hello'}]}},
        'stopReason': 'end_turn',
        'usage': {
            'inputTokens': 100,
            'outputTokens': 10,
            'cacheReadInputTokens': 80,
            'cacheWriteInputTokens': 20,
        },
    }
    mapped = map_bedrock_response_to_anthropic(response, 'sonnet')
    assert mapped['usage']['input_tokens'] == 100
    assert mapped['usage']['cache_read_input_tokens'] == 80
    assert mapped['usage']['cache_creation_input_tokens'] == 20


def test_streaming_message_start_uses_structured_estimate_and_terminal_usage_is_actual():
    seen = []
    stream_response = {
        'stream': [
            {'contentBlockStart': {'contentBlockIndex': 0, 'start': {}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'Hi'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'messageStop': {'stopReason': 'end_turn'}},
            {'metadata': {'usage': {
                'inputTokens': 319,
                'outputTokens': 10,
                'cacheReadInputTokens': 129400,
                'cacheWriteInputTokens': 0,
            }}},
        ]
    }
    events = _parse_sse(iter_bedrock_stream_as_anthropic_sse(
        stream_response,
        'sonnet',
        estimated_usage={
            'input_tokens': 12,
            'cache_read_input_tokens': 120000,
            'cache_creation_input_tokens': 0,
        },
        on_actual_usage=seen.append,
    ))
    assert events[0][0] == 'message_start'
    assert events[0][1]['message']['usage']['input_tokens'] == 12
    assert events[0][1]['message']['usage']['cache_read_input_tokens'] == 120000
    assert events[-2][1]['usage']['input_tokens'] == 319
    assert events[-2][1]['usage']['cache_read_input_tokens'] == 129400
    assert seen == [{'input_tokens': 319, 'output_tokens': 10, 'cache_read_input_tokens': 129400, 'cache_creation_input_tokens': 0}]


def test_streaming_usage_callback_failure_does_not_abort_stream():
    stream_response = {
        'stream': [
            {'contentBlockStart': {'contentBlockIndex': 0, 'start': {}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'Hi'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'messageStop': {'stopReason': 'end_turn'}},
            {'metadata': {'usage': {'inputTokens': 42, 'outputTokens': 7}}},
        ]
    }

    def fail(_usage):
        raise RuntimeError('boom')

    events = _parse_sse(iter_bedrock_stream_as_anthropic_sse(
        stream_response,
        'sonnet',
        estimated_usage={'input_tokens': 10},
        on_actual_usage=fail,
    ))
    assert events[-2][0] == 'message_delta'
    assert events[-2][1]['usage']['input_tokens'] == 42
    assert events[-1][0] == 'message_stop'


def test_estimator_learns_cache_prefix(tmp_path):
    estimator = BedrockTokenEstimator(tmp_path)
    request = {
        'modelId': 'us.anthropic.claude-sonnet-4-6',
        'system': [
            {'text': 'You are Claude Code'},
            {'cachePoint': {'type': 'default'}},
        ],
        'messages': [{'role': 'user', 'content': [{'text': 'Count README words'}]}],
        'toolConfig': {'tools': [{'toolSpec': {'name': 'read'}}]},
    }
    context = estimator.build_context(request)
    cold = estimator.estimate(context).as_anthropic()
    assert cold['input_tokens'] >= 1
    assert cold['cache_creation_input_tokens'] >= 1
    estimator.observe(context, {
        'input_tokens': 319,
        'output_tokens': 12,
        'cache_read_input_tokens': 129400,
        'cache_creation_input_tokens': 0,
    })
    warm = estimator.estimate(context).as_anthropic()
    assert warm['cache_read_input_tokens'] == 129400
    assert 'cache_creation_input_tokens' not in warm


def test_estimator_state_persists(tmp_path):
    request = {
        'modelId': 'us.anthropic.claude-sonnet-4-6',
        'system': [
            {'text': 'You are Claude Code'},
            {'cachePoint': {'type': 'default'}},
        ],
        'messages': [{'role': 'user', 'content': [{'text': 'Count README words'}]}],
        'toolConfig': {'tools': [{'toolSpec': {'name': 'read'}}]},
    }
    estimator = BedrockTokenEstimator(tmp_path)
    context = estimator.build_context(request)
    estimator.observe(context, {
        'input_tokens': 319,
        'output_tokens': 12,
        'cache_read_input_tokens': 129400,
        'cache_creation_input_tokens': 0,
    })
    reloaded = BedrockTokenEstimator(tmp_path)
    warm = reloaded.estimate(reloaded.build_context(request)).as_anthropic()
    assert warm['cache_read_input_tokens'] == 129400


def test_stream_wiring_uses_estimator_context_and_learning():
    payload = {
        'model': 'sonnet',
        'max_tokens': 32,
        'system': [
            {'type': 'text', 'text': 'You are Claude Code', 'cache_control': {'type': 'ephemeral'}},
        ],
        'messages': [{'role': 'user', 'content': 'Count README words'}],
        'tools': [{'name': 'read', 'description': 'Read a file', 'input_schema': {'type': 'object'}}],
    }
    config = Config()
    bedrock_request = map_anthropic_request_to_bedrock(payload)
    bedrock_request['modelId'] = apply_inference_profile_model_id(
        bedrock_request['modelId'],
        region_name=config.region,
        use_inference_profile=config.use_inference_profile,
        use_global=config.use_global_inference_profile,
    )
    estimator = BedrockTokenEstimator(pathlib.Path('/tmp/bedrock-estimator-unused'))
    estimator._save_state = lambda: None
    context = estimator.build_context(bedrock_request)
    estimated = estimator.estimate(context).as_anthropic()
    assert estimated['input_tokens'] >= 1
    assert 'cache_creation_input_tokens' in estimated

    seen = []
    stream_response = {
        'stream': [
            {'contentBlockStart': {'contentBlockIndex': 0, 'start': {}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'Hi'}}},
            {'contentBlockStop': {'contentBlockIndex': 0}},
            {'messageStop': {'stopReason': 'end_turn'}},
            {'metadata': {'usage': {
                'inputTokens': 319,
                'outputTokens': 10,
                'cacheReadInputTokens': 129400,
            }}},
        ]
    }
    events = _parse_sse(iter_bedrock_stream_as_anthropic_sse(
        stream_response,
        payload['model'],
        estimated_usage=estimated,
        on_actual_usage=lambda usage: (seen.append(usage), estimator.observe(context, usage)),
    ))
    assert events[-2][1]['usage']['cache_read_input_tokens'] == 129400
    assert seen and seen[0]['cache_read_input_tokens'] == 129400
    warmed = estimator.estimate(context).as_anthropic()
    assert warmed['cache_read_input_tokens'] == 129400
    assert 'cache_creation_input_tokens' not in warmed


def test_estimator_state_file_created(tmp_path):
    estimator = BedrockTokenEstimator(tmp_path)
    request = {
        'modelId': 'us.anthropic.claude-sonnet-4-6',
        'system': [
            {'text': 'You are Claude Code'},
            {'cachePoint': {'type': 'default'}},
        ],
        'messages': [{'role': 'user', 'content': [{'text': 'Count README words'}]}],
        'toolConfig': {'tools': [{'toolSpec': {'name': 'read'}}]},
    }
    context = estimator.build_context(request)
    estimator.observe(context, {
        'input_tokens': 319,
        'output_tokens': 12,
        'cache_read_input_tokens': 129400,
        'cache_creation_input_tokens': 0,
    })
    assert (tmp_path / 'token-estimator.json').exists()
    text = (tmp_path / 'token-estimator.json').read_text(encoding='utf-8')
    assert 'You are Claude Code' not in text
    assert 'Count README words' not in text
    assert '129400' in text
    assert context.prefix_hash in text


class TestBedrockRegression:
    def test_usage_mapping_prefers_current_cache_fields(self):
        test_usage_mapping_prefers_current_cache_fields()

    def test_usage_mapping_falls_back_to_legacy_cache_fields(self):
        test_usage_mapping_falls_back_to_legacy_cache_fields()

    def test_non_streaming_response_includes_current_cache_usage(self):
        test_non_streaming_response_includes_current_cache_usage()

    def test_streaming_message_start_uses_structured_estimate_and_terminal_usage_is_actual(self):
        test_streaming_message_start_uses_structured_estimate_and_terminal_usage_is_actual()

    def test_streaming_usage_callback_failure_does_not_abort_stream(self):
        test_streaming_usage_callback_failure_does_not_abort_stream()

    def test_estimator_learns_cache_prefix(self, tmp_path):
        test_estimator_learns_cache_prefix(tmp_path)

    def test_estimator_state_persists(self, tmp_path):
        test_estimator_state_persists(tmp_path)

    def test_stream_wiring_uses_estimator_context_and_learning(self):
        test_stream_wiring_uses_estimator_context_and_learning()

    def test_estimator_state_file_created(self, tmp_path):
        test_estimator_state_file_created(tmp_path)


if __name__ == '__main__':
    TestBedrockRegression().test_usage_mapping_prefers_current_cache_fields()
    TestBedrockRegression().test_usage_mapping_falls_back_to_legacy_cache_fields()
    TestBedrockRegression().test_non_streaming_response_includes_current_cache_usage()
    TestBedrockRegression().test_streaming_message_start_uses_structured_estimate_and_terminal_usage_is_actual()
    TestBedrockRegression().test_streaming_usage_callback_failure_does_not_abort_stream()
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    path = __import__('pathlib').Path(tmp.name)
    TestBedrockRegression().test_estimator_learns_cache_prefix(path)
    TestBedrockRegression().test_estimator_state_persists(path)
    TestBedrockRegression().test_stream_wiring_uses_estimator_context_and_learning()
    TestBedrockRegression().test_estimator_state_file_created(path)
    print('ok')
