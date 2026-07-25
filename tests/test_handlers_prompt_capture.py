"""Tests for prompt-capture extraction wired in handlers.py.

Covers:
- _extract_user_prompt_text: last user message raw text
- _extract_prompt_capture: SHA-256 fields, prompt_store_entries, routing fields
- Graceful handling of missing/malformed payload fields
- Classifier field pass-through from ModelRoutingDecision
- _db_sse_wrapper: streaming DB recording with prompt-capture fields
"""
import time
import hashlib
import json
from unittest.mock import MagicMock

from anthproxy.handlers import ProxyRequestHandler
from anthproxy.model_router import ModelRoutingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(routing=None):
    """Return a bare ProxyRequestHandler with _routing pre-set."""
    handler = object.__new__(ProxyRequestHandler)
    handler._routing = routing
    handler._session_prefix = None
    handler._session_hash = None
    handler._req_start = None
    return handler


def _decision(**kwargs):
    """Build a minimal ModelRoutingDecision with optional overrides."""
    base = dict(
        requested_model='sonnet',
        routed_model='sonnet',
        classification=None,
        applied=False,
        reason_code='disabled',
    )
    base.update(kwargs)
    return ModelRoutingDecision(**base)


# ---------------------------------------------------------------------------
# _extract_user_prompt_text
# ---------------------------------------------------------------------------

class TestExtractUserPromptText:
    def test_string_content(self):
        payload = {'messages': [{'role': 'user', 'content': 'hello world'}]}
        result = ProxyRequestHandler._extract_user_prompt_text(payload)
        assert result == 'hello world'

    def test_text_block_content(self):
        payload = {'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'hello '},
            {'type': 'text', 'text': 'world'},
        ]}]}
        result = ProxyRequestHandler._extract_user_prompt_text(payload)
        assert result == 'hello world'

    def test_mixed_blocks_only_text_concatenated(self):
        """Non-text blocks (tool_result, image) are ignored; text blocks joined."""
        payload = {'messages': [{'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'result'},
            {'type': 'text', 'text': 'please review'},
        ]}]}
        result = ProxyRequestHandler._extract_user_prompt_text(payload)
        assert result == 'please review'

    def test_last_user_message_selected(self):
        """The last user message is used, not the first."""
        payload = {'messages': [
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'ok'},
            {'role': 'user', 'content': 'last'},
        ]}
        result = ProxyRequestHandler._extract_user_prompt_text(payload)
        assert result == 'last'

    def test_raw_text_no_stripping(self):
        """system-reminder blocks are NOT stripped — raw storage."""
        raw = '<system-reminder>injected</system-reminder>\nactual message'
        payload = {'messages': [{'role': 'user', 'content': raw}]}
        result = ProxyRequestHandler._extract_user_prompt_text(payload)
        assert result == raw

    def test_empty_messages_returns_none(self):
        assert ProxyRequestHandler._extract_user_prompt_text({'messages': []}) is None

    def test_missing_messages_returns_none(self):
        assert ProxyRequestHandler._extract_user_prompt_text({}) is None

    def test_no_user_role_returns_none(self):
        payload = {'messages': [{'role': 'assistant', 'content': 'hi'}]}
        assert ProxyRequestHandler._extract_user_prompt_text(payload) is None

    def test_empty_string_returns_none(self):
        payload = {'messages': [{'role': 'user', 'content': ''}]}
        assert ProxyRequestHandler._extract_user_prompt_text(payload) is None

    def test_empty_text_blocks_returns_none(self):
        payload = {'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': ''},
        ]}]}
        assert ProxyRequestHandler._extract_user_prompt_text(payload) is None

    def test_non_list_non_string_content_returns_none(self):
        payload = {'messages': [{'role': 'user', 'content': 42}]}
        assert ProxyRequestHandler._extract_user_prompt_text(payload) is None


# ---------------------------------------------------------------------------
# _extract_prompt_capture — SHA-256 and prompt_store
# ---------------------------------------------------------------------------

class TestExtractPromptCaptureHashes:
    def _payload_with_system(self, system_value):
        return {
            'model': 'sonnet',
            'system': system_value,
            'messages': [{'role': 'user', 'content': 'hello'}],
        }

    def test_string_system_sha256(self):
        system = 'You are a helpful assistant.'
        handler = _make_handler()
        result = handler._extract_prompt_capture(self._payload_with_system(system))
        expected = hashlib.sha256(system.encode('utf-8')).hexdigest()
        assert result['system_prompt_sha256'] == expected
        assert len(result['system_prompt_sha256']) == 64

    def test_list_system_sha256(self):
        system = [{'type': 'text', 'text': 'You are helpful.'}]
        handler = _make_handler()
        result = handler._extract_prompt_capture(self._payload_with_system(system))
        serialized = json.dumps(system, sort_keys=True, ensure_ascii=False)
        expected = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        assert result['system_prompt_sha256'] == expected

    def test_no_system_sha256_is_none(self):
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'hi'}]}
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['system_prompt_sha256'] is None

    def test_tools_sha256(self):
        tools = [{'name': 'read_file', 'description': 'reads a file', 'input_schema': {}}]
        payload = {
            'model': 'sonnet',
            'tools': tools,
            'messages': [{'role': 'user', 'content': 'use the tool'}],
        }
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        serialized = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        expected = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        assert result['tools_sha256'] == expected

    def test_no_tools_sha256_is_none(self):
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'hi'}]}
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['tools_sha256'] is None


# ---------------------------------------------------------------------------
# _extract_prompt_capture — prompt_store_entries
# ---------------------------------------------------------------------------

class TestExtractPromptStoreEntries:
    def test_system_entry_present(self):
        system = 'You are an assistant.'
        payload = {
            'model': 'sonnet',
            'system': system,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        entries = result['prompt_store_entries']
        sha = result['system_prompt_sha256']
        assert sha in entries
        content_type, content = entries[sha]
        assert content_type == 'system'
        assert content == system

    def test_tools_entry_present(self):
        tools = [{'name': 'bash', 'description': 'runs bash'}]
        payload = {
            'model': 'sonnet',
            'tools': tools,
            'messages': [{'role': 'user', 'content': 'run it'}],
        }
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        entries = result['prompt_store_entries']
        sha = result['tools_sha256']
        assert sha in entries
        content_type, content = entries[sha]
        assert content_type == 'tools'
        assert content == json.dumps(tools, sort_keys=True, ensure_ascii=False)

    def test_both_system_and_tools_entries(self):
        system = 'sys'
        tools = [{'name': 't'}]
        payload = {
            'model': 'sonnet',
            'system': system,
            'tools': tools,
            'messages': [{'role': 'user', 'content': 'q'}],
        }
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        entries = result['prompt_store_entries']
        assert len(entries) == 2
        assert result['system_prompt_sha256'] in entries
        assert result['tools_sha256'] in entries

    def test_empty_entries_when_no_system_no_tools(self):
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'q'}]}
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['prompt_store_entries'] == {}


# ---------------------------------------------------------------------------
# _extract_prompt_capture — classifier fields pass-through
# ---------------------------------------------------------------------------

class TestExtractPromptCaptureClassifierFields:
    def test_classifier_fields_from_routing_decision(self):
        routing = _decision(
            classification='standard',
            applied=True,
            reason_code='classifier_standard',
            classifier_model='claude-haiku-4-5',
            classifier_summary_json='{"final_user_text":"fix bug"}',
            classifier_raw_response='standard',
            classifier_format='standard',
            classifier_confidence=None,
        )
        handler = _make_handler(routing=routing)
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'fix bug'}]}
        result = handler._extract_prompt_capture(payload)
        assert result['classifier_model'] == 'claude-haiku-4-5'
        assert result['classifier_summary_json'] == '{"final_user_text":"fix bug"}'
        assert result['classifier_raw_response'] == 'standard'
        assert result['classifier_format'] == 'standard'
        assert result['classifier_confidence'] is None

    def test_classifier_fields_none_when_no_routing(self):
        handler = _make_handler(routing=None)
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'hi'}]}
        result = handler._extract_prompt_capture(payload)
        assert result['classifier_model'] is None
        assert result['classifier_summary_json'] is None
        assert result['classifier_raw_response'] is None
        assert result['classifier_format'] is None
        assert result['classifier_confidence'] is None

    def test_classifier_confidence_float(self):
        routing = _decision(
            classification='deep',
            applied=True,
            reason_code='classifier_deep',
            classifier_confidence=0.95,
            classifier_format='json',
        )
        handler = _make_handler(routing=routing)
        payload = {'model': 'sonnet', 'messages': [{'role': 'user', 'content': 'arch'}]}
        result = handler._extract_prompt_capture(payload)
        assert result['classifier_confidence'] == 0.95
        assert result['classifier_format'] == 'json'


# ---------------------------------------------------------------------------
# _extract_prompt_capture — routing_recovered_via_walkback
# ---------------------------------------------------------------------------

class TestExtractPromptCaptureWalkback:
    def test_walkback_false_for_direct_final_message(self):
        """A final message with real text: recovered_via_walkback should be False."""
        payload = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'what is 2+2?'}],
        }
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['routing_recovered_via_walkback'] is False

    def test_walkback_true_for_tool_result_only_final_message(self):
        """A tool_result-only final message forces text recovery from walk-back."""
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': 'what is 2+2?'},
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}}
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': '4'}
                ]},
            ],
        }
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['routing_recovered_via_walkback'] is True

    def test_walkback_none_for_empty_messages(self):
        """No summary can be built → None."""
        payload = {'model': 'sonnet', 'messages': []}
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['routing_recovered_via_walkback'] is None


# ---------------------------------------------------------------------------
# _extract_prompt_capture — graceful malformed payload handling
# ---------------------------------------------------------------------------

class TestExtractPromptCaptureMalformed:
    def test_empty_payload_returns_dict_with_nones(self):
        handler = _make_handler()
        result = handler._extract_prompt_capture({})
        assert isinstance(result, dict)
        assert result['user_prompt_text'] is None
        assert result['system_prompt_sha256'] is None
        assert result['tools_sha256'] is None
        assert result['prompt_store_entries'] == {}

    def test_messages_not_list_returns_nones(self):
        handler = _make_handler()
        result = handler._extract_prompt_capture({'messages': 'not a list'})
        assert result['user_prompt_text'] is None

    def test_user_prompt_text_extracted_even_without_routing(self):
        payload = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'hello'}],
        }
        handler = _make_handler(routing=None)
        result = handler._extract_prompt_capture(payload)
        assert result['user_prompt_text'] == 'hello'
        assert result['classifier_model'] is None

    def test_result_always_dict(self):
        """Even a completely broken payload returns a dict (never raises)."""
        handler = _make_handler()
        # Extremely malformed
        result = handler._extract_prompt_capture({'messages': [None, 42, {}]})
        assert isinstance(result, dict)

    def test_image_block_skipped(self):
        """Image blocks in message content are ignored; only text blocks extracted."""
        payload = {'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'abc'}},
            {'type': 'text', 'text': 'describe this'},
        ]}]}
        handler = _make_handler()
        result = handler._extract_prompt_capture(payload)
        assert result['user_prompt_text'] == 'describe this'


# ---------------------------------------------------------------------------
# _db_sse_wrapper — streaming DB recording path (Major finding B1)
# ---------------------------------------------------------------------------

def _make_streaming_handler(prompt_capture=None, db_request_id=None, routing=None):
    """Return a handler wired for streaming DB recording tests."""
    handler = object.__new__(ProxyRequestHandler)
    handler.session_db = MagicMock()
    handler._routing = routing
    handler._session_prefix = None
    handler._session_hash = None
    handler._req_start = None
    handler._prompt_capture = prompt_capture if prompt_capture is not None else {}
    handler._db_request_id = db_request_id
    return handler


def _drain(gen):
    """Consume the generator and return all yielded chunks."""
    return list(gen)


class TestDbSseWrapperStreamingPath:
    """Verify _db_sse_wrapper passes prompt-capture fields into DB on success."""

    def _routing_decision(self, routed_model='sonnet'):
        return ModelRoutingDecision(
            requested_model='sonnet',
            routed_model=routed_model,
            classification=None,
            applied=False,
            reason_code='disabled',
        )

    def test_record_request_called_on_stream_complete(self):
        """record_request is called exactly once after the generator is exhausted."""
        handler = _make_streaming_handler()
        handler.stats_collector = None
        chunks = ['data: {"type":"message_start"}\n\n', 'data: [DONE]\n\n']
        gen = handler._usage_sse_wrapper(
            iter(chunks), None, 'anthropic', time.monotonic() - 0.01, 'opus',
            session_id='ses1', conversation_anchor='anchor',
            routing_decision=self._routing_decision(),
        )
        result = _drain(gen)
        assert result == chunks
        assert handler.session_db.record_request.call_count == 1

    def test_prompt_capture_fields_forwarded(self):
        """Prompt-capture dict is forwarded verbatim into record_request kwargs."""
        capture = {
            'user_prompt_text': 'hello world',
            'system_prompt_sha256': 'a' * 64,
            'tools_sha256': None,
            'prompt_store_entries': {},
            'routing_recovered_via_walkback': False,
            'classifier_model': 'claude-haiku-4-5',
            'classifier_summary_json': '{"label":"trivial"}',
            'classifier_raw_response': 'trivial',
            'classifier_confidence': 0.95,
            'classifier_format': 'standard',
        }
        handler = _make_streaming_handler(prompt_capture=capture)
        handler.stats_collector = None
        gen = handler._usage_sse_wrapper(
            iter(['data: chunk\n\n']), None, 'anthropic', time.monotonic(), 'opus',
            session_id='ses1', conversation_anchor='anchor',
            routing_decision=self._routing_decision(),
        )
        _drain(gen)
        kwargs = handler.session_db.record_request.call_args[1]
        for key, val in capture.items():
            assert kwargs.get(key) == val, f'Mismatch for {key}'

    def test_success_records_error_none(self):
        """Normal completion records error=None."""
        handler = _make_streaming_handler()
        handler.stats_collector = None
        gen = handler._usage_sse_wrapper(
            iter(['data: ok\n\n']), None, 'anthropic', time.monotonic(), 'opus',
            session_id='s', conversation_anchor=None,
            routing_decision=self._routing_decision(),
        )
        _drain(gen)
        kwargs = handler.session_db.record_request.call_args[1]
        assert kwargs['error'] is None
        assert kwargs['status'] == 'success'

    def test_sse_error_event_records_sse_error(self):
        """An in-band SSE error event causes error='sse_error' in the DB row."""
        handler = _make_streaming_handler()
        handler.stats_collector = None
        chunks = ['event: error\ndata: {"type":"error"}\n\n']
        gen = handler._usage_sse_wrapper(
            iter(chunks), None, 'anthropic', time.monotonic(), 'opus',
            session_id='s', conversation_anchor=None,
            routing_decision=self._routing_decision(),
        )
        _drain(gen)
        kwargs = handler.session_db.record_request.call_args[1]
        assert kwargs['error'] == 'sse_error'
        assert kwargs['status'] == 'error'

    def test_no_session_db_skips_recording(self):
        """When session_db is None no record_request is attempted."""
        handler = _make_streaming_handler()
        handler.session_db = None
        handler.stats_collector = None
        gen = handler._usage_sse_wrapper(
            iter(['data: chunk\n\n']), None, 'anthropic', time.monotonic(), 'opus',
            session_id='s', conversation_anchor=None,
            routing_decision=self._routing_decision(),
        )
        # Should not raise even though session_db is None
        chunks = _drain(gen)
        assert chunks == ['data: chunk\n\n']

    def test_empty_prompt_capture_records_with_none_fields(self):
        """Empty _prompt_capture (extraction error path) records None for new fields."""
        handler = _make_streaming_handler(prompt_capture={})
        handler.stats_collector = None
        gen = handler._usage_sse_wrapper(
            iter(['data: x\n\n']), None, 'anthropic', time.monotonic(), 'opus',
            session_id='s', conversation_anchor=None,
            routing_decision=self._routing_decision(),
        )
        _drain(gen)
        kwargs = handler.session_db.record_request.call_args[1]
        # No prompt-capture keys injected — original params only
        assert 'user_prompt_text' not in kwargs
        assert 'system_prompt_sha256' not in kwargs

    def test_retry_path_calls_update_not_record(self):
        """When _db_request_id is set, update_request_on_retry is called instead."""
        handler = _make_streaming_handler(db_request_id=42)
        handler.stats_collector = None
        gen = handler._usage_sse_wrapper(
            iter(['data: chunk\n\n']), None, 'anthropic', time.monotonic(), 'opus',
            session_id='s', conversation_anchor=None,
            routing_decision=self._routing_decision(),
        )
        _drain(gen)
        assert handler.session_db.record_request.call_count == 0
        assert handler.session_db.update_request_on_retry.call_count == 1
