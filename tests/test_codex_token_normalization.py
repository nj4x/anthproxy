"""Tests for Codex token normalization fixes (E1b floor double-count).

Verifies that:
1. Non-streaming _map_response subtracts cached tokens from inclusive input_tokens
2. Streaming _iter_stream_as_anthropic_sse normalizes both message_start (→ 0)
   and message_delta (→ exclusive input)
3. Cache-key derivation uses session_id from user_id JSON blob

All tests are network-free.
"""

import json
from unittest.mock import MagicMock

from anthproxy.codex.mapper import (
    _map_response,
    _iter_stream_as_anthropic_sse,
    _map_request,
)


class TestMapResponseTokenNormalization:
    """Non-streaming path: _map_response must emit exclusive input_tokens."""

    def test_inclusive_input_tokens_subtracts_cached(self):
        """Codex reports input_tokens inclusive of cached; emit exclusive."""
        items = [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]}
        ]
        usage = {
            'input_tokens': 1000,  # inclusive (includes cached)
            'output_tokens': 100,
            'input_tokens_details': {
                'cached_tokens': 400  # this is a subset of input_tokens
            }
        }
        resp = _map_response(items, usage, 'completed', 'sonnet')
        # Should emit exclusive: 1000 - 400 = 600
        assert resp['usage']['input_tokens'] == 600
        assert resp['usage']['cache_read_input_tokens'] == 400

    def test_no_cached_tokens_uses_raw_input(self):
        """If no cache info, emit raw input as-is."""
        items = [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]}
        ]
        usage = {
            'input_tokens': 500,
            'output_tokens': 50,
            'input_tokens_details': {}
        }
        resp = _map_response(items, usage, 'completed', 'sonnet')
        assert resp['usage']['input_tokens'] == 500
        assert 'cache_read_input_tokens' not in resp['usage']

    def test_zero_cached_tokens_omitted(self):
        """If cached is 0 or falsy, don't emit cache_read_input_tokens."""
        items = [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]}
        ]
        usage = {
            'input_tokens': 500,
            'output_tokens': 50,
            'input_tokens_details': {
                'cached_tokens': 0
            }
        }
        resp = _map_response(items, usage, 'completed', 'sonnet')
        assert resp['usage']['input_tokens'] == 500
        assert 'cache_read_input_tokens' not in resp['usage']

    def test_negative_exclusive_clamped_to_zero(self):
        """Malformed response with cached > input → clamp to 0."""
        items = [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hi'}]}
        ]
        usage = {
            'input_tokens': 100,
            'output_tokens': 50,
            'input_tokens_details': {
                'cached_tokens': 200  # impossible: cached > input
            }
        }
        resp = _map_response(items, usage, 'completed', 'sonnet')
        assert resp['usage']['input_tokens'] == 0  # max(0, 100 - 200)
        assert resp['usage']['cache_read_input_tokens'] == 200


class TestStreamingTokenNormalization:
    """Streaming path: _iter_stream_as_anthropic_sse normalizes both events."""

    def _parse_sse_data(self, event_str):
        """Extract and parse the JSON payload from an SSE string."""
        for line in event_str.split('\n'):
            if line.startswith('data: '):
                return json.loads(line[6:])
        return {}

    def test_message_start_emits_zero_input_tokens(self):
        """message_start must emit input_tokens: 0 to prevent max() inflation."""
        # Use Codex Response API event types; side_effect terminates after one chunk
        sse_data = (
            b'data: {"type":"response.output_text.delta","delta":"hello"}\n'
            b'data: {"type":"response.completed","response":{"status":"completed",'
            b'"usage":{"input_tokens":1000,"output_tokens":10,'
            b'"input_tokens_details":{"cached_tokens":400}}}}\n'
        )

        response = MagicMock()
        response.read.side_effect = [sse_data, b'']

        events = list(_iter_stream_as_anthropic_sse(
            response=response,
            requested_model='sonnet',
            estimated_input_tokens=1000,
        ))

        # message_start is always the first yielded event
        msg_start = self._parse_sse_data(events[0])
        assert msg_start['type'] == 'message_start'
        assert msg_start['message']['usage']['input_tokens'] == 0  # normalized to 0

        # message_delta is the second-to-last event (before message_stop)
        msg_delta = self._parse_sse_data(events[-2])
        assert msg_delta['type'] == 'message_delta'
        # Delta should have normalized exclusive input: 1000 - 400 = 600
        assert msg_delta['usage']['input_tokens'] == 600
        assert msg_delta['usage'].get('cache_read_input_tokens') == 400


_MINIMAL_MSGS = [{'role': 'user', 'content': 'hi'}]


class TestCacheKeyDerivation:
    """Cache-key derivation must use session_id, not 64-char prefix."""

    def test_cache_key_uses_session_id_from_json(self):
        """Derived from user_id JSON blob's session_id, not 64-char prefix."""
        payload = {
            'messages': _MINIMAL_MSGS,
            'metadata': {
                'user_id': json.dumps({
                    'device_id': '550e8400-e29b-41d4-a716-446655440000',
                    'session_id': 'my-stable-session-id',
                    'account_uuid': 'acc-uuid',
                })
            }
        }
        body = _map_request({**payload, 'model': 'sonnet'})
        assert body['prompt_cache_key'] == 'my-stable-session-id'

    def test_cache_key_fallback_on_malformed_json(self):
        """Fallback to 64-char truncation if user_id isn't valid JSON."""
        payload = {
            'messages': _MINIMAL_MSGS,
            'metadata': {
                'user_id': 'not-json-string-that-is-longer-than-64-characters-so-we-can-test-truncation-behavior'
            }
        }
        body = _map_request({**payload, 'model': 'sonnet'})
        assert body['prompt_cache_key'] == 'not-json-string-that-is-longer-than-64-characters-so-we-can-test'  # 64 chars

    def test_cache_key_fallback_on_missing_session_id(self):
        """Fallback if JSON lacks session_id field."""
        payload = {
            'messages': _MINIMAL_MSGS,
            'metadata': {
                'user_id': json.dumps({
                    'device_id': '550e8400-e29b-41d4-a716-446655440000',
                    'account_uuid': 'acc-uuid',
                    # no session_id
                })
            }
        }
        body = _map_request({**payload, 'model': 'sonnet'})
        # Should fallback to 64-char prefix of the JSON string
        json_str = json.dumps({
            'device_id': '550e8400-e29b-41d4-a716-446655440000',
            'account_uuid': 'acc-uuid',
        })
        assert body['prompt_cache_key'] == json_str[:64]

    def test_cache_key_uses_instructions_hash_if_no_metadata(self):
        """Fallback to instructions hash if user_id unavailable."""
        import hashlib
        instructions = 'you are a helpful assistant'
        payload = {
            'messages': _MINIMAL_MSGS,
            'metadata': {},
            'system': instructions,
        }
        body = _map_request({**payload, 'model': 'sonnet'})
        expected = hashlib.sha256(instructions.encode('utf-8')).hexdigest()[:32]
        assert body['prompt_cache_key'] == expected

    def test_different_sessions_same_device_get_different_keys(self):
        """Key derivation must isolate by session_id, not device_id prefix."""
        session1_payload = {
            'messages': _MINIMAL_MSGS,
            'metadata': {
                'user_id': json.dumps({
                    'device_id': 'shared-device-id',
                    'session_id': 'session-1',
                    'account_uuid': 'acc-1',
                })
            }
        }
        session2_payload = {
            'messages': _MINIMAL_MSGS,
            'metadata': {
                'user_id': json.dumps({
                    'device_id': 'shared-device-id',  # same device
                    'session_id': 'session-2',  # different session
                    'account_uuid': 'acc-1',
                })
            }
        }
        body1 = _map_request({**session1_payload, 'model': 'sonnet'})
        body2 = _map_request({**session2_payload, 'model': 'sonnet'})
        # Keys must differ by session_id
        assert body1['prompt_cache_key'] == 'session-1'
        assert body2['prompt_cache_key'] == 'session-2'
        assert body1['prompt_cache_key'] != body2['prompt_cache_key']
