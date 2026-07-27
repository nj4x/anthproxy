import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from anthproxy.handlers import (
    AnthropicRequestError,
    ProxyRequestHandler,
    _context_key,
    _conversation_anchor,
    _parse_local_command,
    _session_key,
    _session_short_id,
    _system_fingerprint,
)


def _msg(content):
    return {'messages': [{'role': 'user', 'content': content}]}


def _handler_with_registry(registry):
    handler = object.__new__(ProxyRequestHandler)
    handler.registry = registry
    handler.config = MagicMock()
    handler._send_json = MagicMock()
    handler._send_sse = MagicMock()
    # Session-context floor reads (floor, ratio) and must unpack to a real tuple;
    # default to the identity so a bare MagicMock registry doesn't break routing.
    if isinstance(registry, MagicMock):
        registry.session_context.return_value = (0, 1.0)
    # do_POST normally initializes these; tests call _handle_* directly.
    handler._ctx_key = None
    handler._route_est = 0
    # Per-request override defaults (do_POST parses the header; tests that
    # don't set these explicitly get the no-override path).
    handler._no_classifier = False
    handler._prefer_backend = None
    return handler


def _fake_snapshot(name, backend, session_pinned=False, session_subscription=False):
    snapshot = MagicMock()
    snapshot.name = name
    snapshot.backend = backend
    snapshot.config = MagicMock()
    # Keepalive off by default in tests → synchronous priming, no thread spawned.
    snapshot.config.sse_keepalive_interval = 0
    # Concrete int so the model-router size floor never fires on small test
    # payloads (a MagicMock threshold would compare truthy and force opus[1m]).
    snapshot.config.auto_model_routing_long_context_threshold = 150_000
    # Concrete dict/string (not MagicMocks) so label->tier mapping and the
    # long-context floor's forced model are well-defined for tests that don't
    # override them.
    snapshot.config.auto_model_routing_classification = {
        'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'opus',
    }
    snapshot.config.auto_model_routing_long = 'opus[1m]'
    # Disable confidence-bump and set explicit mode/confidence defaults so the
    # classifier path uses parse_classifier_label (one-word response) rather than
    # parse_classifier_label_json (JSON response) — tests mock single-word replies.
    snapshot.config.auto_model_routing_confidence_bump = False
    snapshot.config.auto_model_routing_min_confidence = 0.0
    snapshot.config.auto_model_routing_mode = 'classifier'
    snapshot.config.lock_requested_model = 'off'
    snapshot.config.auto_model_routing_prior_response_summary_limit = 1000
    # ADR 0010/0012: weighted blend config — concrete values so comparisons work.
    snapshot.config.auto_model_routing_system_prompt_weight = 0.30
    snapshot.config.auto_model_routing_user_prompt_weight = 0.70
    snapshot.config.auto_model_routing_trivial_threshold = 0.75
    snapshot.config.auto_model_routing_standard_threshold = 1.50
    snapshot.config.auto_model_routing_system_prompt_cache_size = 256
    snapshot.config.auto_model_routing_system_prompt_preview_limit = 500
    snapshot.session_pinned = session_pinned
    snapshot.session_subscription = session_subscription
    return snapshot


class TestFreshSessionLocalCommands:
    def test_reminder_prefix_in_string_stripped(self):
        text = '<system-reminder>some\ninjected\ncontext</system-reminder>\nproxy-status'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_multiple_reminders_in_string_stripped(self):
        text = (
            '<system-reminder>first</system-reminder>\n'
            '<system-reminder>second</system-reminder>\n'
            'proxy-status'
        )
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_reminder_only_content_not_command(self):
        text = '<system-reminder>some injected context</system-reminder>'
        assert _parse_local_command(_msg(text)) is None

    def test_reminder_removed_but_prose_remains_not_command(self):
        text = '<system-reminder>ctx</system-reminder>\nplease run proxy-status'
        assert _parse_local_command(_msg(text)) is None

    def test_reminder_with_attributes_before_command_stripped(self):
        text = '<system-reminder data-source="cli">ctx</system-reminder>\nproxy-status'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_reminder_with_tag_whitespace_before_command_stripped(self):
        text = '<system-reminder >ctx</system-reminder>\nproxy-status'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_reminder_case_insensitive_before_command_stripped(self):
        text = '<System-Reminder>ctx</SYSTEM-REMINDER>\nproxy-status'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_unclosed_reminder_after_command_is_stripped(self):
        text = 'proxy-status\n<system-reminder data-source="cli">ctx'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_unclosed_reminder_tag_not_stripped(self):
        text = '<system-reminder>ctx\nproxy-status'
        assert _parse_local_command(_msg(text)) is None

    def test_local_command_wrapper_prefix_stripped(self):
        text = '<command-name>/clear</command-name>\nproxy-status'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_local_command_wrapper_blocks_only_not_command(self):
        payload = _msg([
            {'type': 'text', 'text':
                '<local-command-caveat>Caveat…</local-command-caveat>'},
            {'type': 'text', 'text': '<command-name>/clear</command-name>'},
            {'type': 'text', 'text': '<local-command-stdout></local-command-stdout>'},
        ])
        assert _parse_local_command(payload) is None

    def test_unclosed_command_wrapper_not_stripped(self):
        text = '<command-name>/clear\nproxy-status'
        assert _parse_local_command(_msg(text)) is None

    def test_multiple_reminder_only_blocks_before_command_stripped(self):
        payload = _msg([
            {'type': 'text', 'text': '<system-reminder>first</system-reminder>'},
            {'type': 'text', 'text': '<system-reminder>second</system-reminder>'},
            {'type': 'text', 'text': 'proxy-get-backend'},
        ])
        assert _parse_local_command(payload) == ('get-backend', None)

    def test_prose_and_command_blocks_still_rejected_after_reminder_strip(self):
        payload = _msg([
            {'type': 'text', 'text': '<system-reminder>ctx</system-reminder>'},
            {'type': 'text', 'text': 'hello'},
            {'type': 'text', 'text': 'proxy-status'},
        ])
        assert _parse_local_command(payload) is None

    def test_command_and_prose_same_block_after_multiple_reminders_rejected(self):
        payload = _msg([
            {
                'type': 'text',
                'text': '<system-reminder>one</system-reminder>'
                        '<system-reminder>two</system-reminder>\nproxy-status now',
            },
        ])
        assert _parse_local_command(payload) is None

    def test_reminder_suffix_in_string_stripped(self):
        text = 'proxy-status\n<system-reminder>end reminder</system-reminder>'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_reminder_as_separate_text_block_stripped(self):
        payload = _msg([
            {'type': 'text', 'text': '<system-reminder>injected</system-reminder>'},
            {'type': 'text', 'text': 'proxy-get-backend'},
        ])
        assert _parse_local_command(payload) == ('get-backend', None)

    def test_reminder_and_command_in_same_block_stripped(self):
        payload = _msg([
            {'type': 'text', 'text': '<system-reminder>ctx</system-reminder>\nproxy-get-usage'},
        ])
        assert _parse_local_command(payload) == ('get-usage', None)

    def test_two_real_text_blocks_still_rejected(self):
        payload = _msg([
            {'type': 'text', 'text': 'hello'},
            {'type': 'text', 'text': 'proxy-status'},
        ])
        assert _parse_local_command(payload) is None

    def test_reminder_only_block_plus_non_text_still_rejected(self):
        payload = _msg([
            {'type': 'text', 'text': '<system-reminder>ctx</system-reminder>'},
            {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'done'},
        ])
        assert _parse_local_command(payload) is None

    def test_session_wrap_with_spaces_detected(self):
        text = '<session> proxy-status </session>'
        assert _parse_local_command(_msg(text)) == ('status', None)

    def test_session_wrap_no_spaces_detected(self):
        text = '<session>proxy-get-backend</session>'
        assert _parse_local_command(_msg(text)) == ('get-backend', None)

    def test_session_wrap_prose_outside_not_a_command(self):
        text = 'hi <session>proxy-status</session>'
        assert _parse_local_command(_msg(text)) is None

    def test_session_wrap_unclosed_not_a_command(self):
        text = '<session>proxy-status'
        assert _parse_local_command(_msg(text)) is None

    def test_session_wrap_non_command_inner_not_a_command(self):
        text = '<session>hello</session>'
        assert _parse_local_command(_msg(text)) is None

    def test_proxy_get_backend_bypasses_backend_parsing_on_fresh_session(self):
        registry = MagicMock()
        registry.active_name.return_value = 'codex'
        handler = _handler_with_registry(registry)
        payload = _msg([
            {'type': 'text', 'text': '<system-reminder>ctx</system-reminder>'},
            {'type': 'text', 'text': 'proxy-get-backend'},
        ])
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        registry.snapshot.assert_not_called()
        handler._send_json.assert_called_once()
        text = handler._send_json.call_args.args[1]['content'][0]['text']
        assert '`codex`' in text


class TestRoutingLogs:
    def test_messages_route_logs_backend(self, caplog):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.return_value = {'type': 'message', 'content': []}
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot('openrouter', backend)
        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=_msg('Hello'))

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert 'operation=messages backend=openrouter stream=False attempt=1' in caplog.text

    def test_streaming_messages_route_logs_backend(self, caplog):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message_stream.return_value = iter(['chunk-1', 'chunk-2'])
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot('anthropic', backend)
        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})
        handler._send_sse = MagicMock()

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert 'operation=messages backend=anthropic stream=True attempt=1' in caplog.text
        # Priming now happens inside _send_sse, so the generator passed is
        # un-primed. The test checks that _send_sse received the correct
        # generator and config.
        assert handler._send_sse.call_count == 1
        assert handler._send_sse.call_args.kwargs['config'] is not None

    def test_retry_429_uses_original_snapshot_name_and_logs_attempts(self, caplog):
        first_backend = MagicMock()
        first_backend.parse_credentials.return_value = {'token': 'first'}
        first_backend.send_message.side_effect = AnthropicRequestError('rate limited', status_code=429)
        second_backend = MagicMock()
        second_backend.parse_credentials.return_value = {'token': 'second'}
        second_backend.send_message.return_value = {'type': 'message', 'content': []}
        first_snapshot = _fake_snapshot('anthropic', first_backend)
        second_snapshot = _fake_snapshot('bedrock', second_backend)
        registry = MagicMock()
        registry.snapshot.side_effect = [first_snapshot, second_snapshot]
        registry.active_name.return_value = 'codex'
        selector = MagicMock()
        selector.is_paused.return_value = False
        selector.on_rate_limited.return_value = 'bedrock'
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=_msg('Hello'))

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        selector.on_rate_limited.assert_called_once_with('anthropic', None)
        assert 'operation=messages backend=anthropic stream=False attempt=1' in caplog.text
        assert 'operation=messages backend=bedrock stream=False attempt=2' in caplog.text
        assert 'retrying request on bedrock after 429 from anthropic' in caplog.text

    def test_local_command_logs_active_backend(self, caplog):
        registry = MagicMock()
        registry.active_name.return_value = 'codex'
        handler = _handler_with_registry(registry)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_local_command(('get-backend', None), {'stream': True})

        assert 'operation=local_command command=get-backend active_backend=codex stream=True' in caplog.text

    def test_status_local_command_logs_active_backend(self, caplog):
        backend = MagicMock()
        backend.get_usage_markdown.return_value = '## Usage'
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        registry.snapshot.return_value = _fake_snapshot('anthropic', backend)
        # instance() is called for both 'anthropic' and 'codex' — return a mock
        # whose get_usage_markdown returns a string so ''.join(lines) doesn't fail.
        instance_mock = MagicMock()
        instance_mock.get_usage_markdown.return_value = '## Usage'
        registry.instance.return_value = instance_mock
        handler = _handler_with_registry(registry)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_local_command(('status', None), {'stream': False})

        assert 'operation=local_command command=status active_backend=anthropic stream=False' in caplog.text
        assert 'backend=anthropic' not in caplog.text.replace('active_backend=anthropic', '')

    def test_count_tokens_logs_backend(self, caplog):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.count_tokens.return_value = {'input_tokens': 1}
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot('codex', backend)
        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={'model': 'sonnet', 'messages': []})

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_count_tokens()

        assert 'operation=count_tokens backend=codex' in caplog.text

    def test_cache_logs_bedrock_backend_without_leaking_secret(self, caplog):
        backend = MagicMock()
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        registry.instance.return_value = backend
        handler = _handler_with_registry(registry)
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={'key': 'alias', 'value': 'super-secret'})

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_cache()

        assert 'operation=cache backend=bedrock' in caplog.text
        assert 'alias' not in caplog.text
        assert 'super-secret' not in caplog.text
        registry.instance.assert_called_once_with('bedrock')
        registry.snapshot.assert_not_called()
        registry.switch.assert_not_called()

    def test_cache_validation_failure_does_not_log_backend(self, caplog):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={'key': '', 'value': 'super-secret'})

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_cache()

        assert 'operation=cache backend=' not in caplog.text
        registry.instance.assert_not_called()
        registry.snapshot.assert_not_called()
        registry.switch.assert_not_called()

    def test_cache_bedrock_lookup_failure_does_not_log_backend(self, caplog):
        registry = MagicMock()
        registry.instance.side_effect = RuntimeError('boom')
        handler = _handler_with_registry(registry)
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={'key': 'alias', 'value': 'super-secret'})

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_cache()

        assert 'operation=cache backend=bedrock' not in caplog.text
        registry.instance.assert_called_once_with('bedrock')
        registry.snapshot.assert_not_called()
        registry.switch.assert_not_called()
        assert 'super-secret' not in caplog.text
        assert 'alias' not in caplog.text
        assert 'Cache store failure:' in caplog.text
        assert 'boom' in caplog.text
        handler._send_json.assert_called_once()
        status, payload = handler._send_json.call_args.args
        assert status == 502
        assert payload['error']['type'] == 'api_error'
        assert payload['error']['message'] == 'Cache store failed'

    def test_pre_snapshot_request_error_does_not_consult_selector(self):
        registry = MagicMock()
        selector = MagicMock()
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler._validate_content_type = MagicMock(side_effect=AnthropicRequestError('bad request', status_code=429))
        handler._send_json = MagicMock()

        handler._handle_messages()

        selector.on_rate_limited.assert_not_called()
        registry.snapshot.assert_not_called()
        handler._send_json.assert_called_once()

    def test_streaming_retry_logs_backend(self, caplog):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message_stream.return_value = iter(['first', 'second'])
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot('codex', backend)
        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'secret'}
        handler._send_sse = MagicMock()

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._retry_on_new_backend({'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

        assert 'operation=messages backend=codex stream=True attempt=2' in caplog.text
        # Priming now happens inside _send_sse, so the generator passed is
        # un-primed. Check that _send_sse received config.
        assert handler._send_sse.call_count == 1
        assert handler._send_sse.call_args.kwargs['config'] is not None

    def test_streaming_429_during_priming_sends_sse_error_no_retry(self, caplog):
        """429 during priming (post-header) is delivered as an SSE error, not retried."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'first'}

        def raise_on_first_chunk():
            raise AnthropicRequestError('rate limited', status_code=429)
            yield  # pragma: no cover

        backend.send_message_stream.return_value = raise_on_first_chunk()
        snapshot = _fake_snapshot('anthropic', backend)
        registry = MagicMock()
        registry.snapshot.return_value = snapshot
        selector = MagicMock()
        selector.is_paused.return_value = False
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # No retry: the 429 now surfaces inside _send_sse (post-header) as an SSE
        # error, so _handle_messages never sees it and never triggers a backend
        # switch. (The SSE-error frame itself is covered in TestSsePriming.)
        selector.on_rate_limited.assert_not_called()
        assert 'operation=messages backend=anthropic stream=True attempt=1' in caplog.text
        assert 'attempt=2' not in caplog.text  # no retry


# ---------------------------------------------------------------------------
# _session_key helper
# ---------------------------------------------------------------------------

class TestSessionKey:
    def test_extracts_metadata_user_id(self):
        payload = {'metadata': {'user_id': 'user_abc123_session_xyz'}}
        assert _session_key(payload) == 'user_abc123_session_xyz'

    def test_absent_metadata_returns_none(self):
        assert _session_key({}) is None

    def test_absent_user_id_returns_none(self):
        assert _session_key({'metadata': {}}) is None

    def test_blank_user_id_returns_none(self):
        assert _session_key({'metadata': {'user_id': ''}}) is None

    def test_non_string_user_id_returns_none(self):
        assert _session_key({'metadata': {'user_id': 123}}) is None

    def test_non_dict_metadata_returns_none(self):
        assert _session_key({'metadata': 'oops'}) is None

    def test_user_id_truncated_to_128(self):
        long_id = 'x' * 200
        result = _session_key({'metadata': {'user_id': long_id}})
        assert result == 'x' * 128


# ---------------------------------------------------------------------------
# _session_short_id: UUID extraction from plain and JSON-blob session keys
# ---------------------------------------------------------------------------

class TestSessionShortId:
    def test_plain_uuid_returns_first_segment(self):
        uid = '550e8400-e29b-41d4-a716-446655440000'
        assert _session_short_id(uid) == '550e8400'

    def test_json_blob_prefers_session_id_field(self):
        # Claude Code sends {"device_id":"...", "session_id":"24b79b69-...", ...}
        blob = '{"device_id":"33c2e38ab8c2ca51","session_id":"24b79b69-3092-424e-b2e1-56d26a76064c","account_uuid":""}'
        assert _session_short_id(blob) == '24b79b69'

    def test_json_blob_session_id_non_uuid_value(self):
        # session_id present but not UUID-shaped — use first 8 chars of its value
        blob = '{"device_id":"device123","session_id":"sess_plain_text"}'
        assert _session_short_id(blob) == 'sess_pla'

    def test_json_blob_without_session_id_falls_back_to_sha256(self):
        import hashlib
        blob = '{"device_id":"device_abc123xyz"}'
        assert _session_short_id(blob) == hashlib.sha256(blob.encode()).hexdigest()[:8]

    def test_uuid_case_insensitive(self):
        uid = 'ABCDEF01-1234-5678-9ABC-DEF012345678'
        assert _session_short_id(uid) == 'ABCDEF01'

    def test_plain_no_uuid_falls_back_to_sha256(self):
        import hashlib
        key = 'plainkey'
        assert _session_short_id(key) == hashlib.sha256(key.encode()).hexdigest()[:8]

    def test_invalid_json_falls_back_to_raw_uuid_scan_then_sha256(self):
        import hashlib
        key = '{invalid json}'
        # No UUID in the string, so sha256 fallback
        assert _session_short_id(key) == hashlib.sha256(key.encode()).hexdigest()[:8]

    def test_empty_key_returns_sha256_of_empty(self):
        import hashlib
        assert _session_short_id('') == hashlib.sha256(b'').hexdigest()[:8]


# ---------------------------------------------------------------------------
# Parsing: proxy-set-backend:<name>:session (renamed from proxy-session-set-backend)
# ---------------------------------------------------------------------------

class TestParseSessionSetBackend:
    def _msg(self, text):
        return {'messages': [{'role': 'user', 'content': text}]}

    def test_parse_valid_backend(self):
        assert _parse_local_command(self._msg('proxy-set-backend:openrouter:session')) == \
               ('session-set-backend', 'openrouter')

    def test_parse_auto(self):
        assert _parse_local_command(self._msg('proxy-set-backend:auto:session')) == \
               ('session-set-backend', 'auto')

    def test_parse_unknown_backend_returns_none_arg(self):
        assert _parse_local_command(self._msg('proxy-set-backend:nope:session')) == \
               ('session-set-backend', None)

    def test_parse_subscription_session(self):
        assert _parse_local_command(self._msg('proxy-set-backend:subscription:session')) == \
               ('session-set-backend', 'subscription')

    def test_empty_target_session_returns_none_arg(self):
        # proxy-set-backend::session → target='' → invalid
        assert _parse_local_command(self._msg('proxy-set-backend::session')) == \
               ('session-set-backend', None)

    def test_global_set_backend_still_works(self):
        assert _parse_local_command(self._msg('proxy-set-backend:codex')) == \
               ('set-backend', 'codex')

    def test_old_session_prefix_not_recognized(self):
        # The old proxy-session-set-backend: prefix is removed.
        assert _parse_local_command(self._msg('proxy-session-set-backend:codex')) is None
        assert _parse_local_command(self._msg('proxy-session-set-backend:auto')) is None


# ---------------------------------------------------------------------------
# Routing: session-pinned 429 does NOT trigger auto-switch
# ---------------------------------------------------------------------------

class TestSessionPinned429:
    def test_session_pinned_429_skips_auto_switch(self):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.side_effect = AnthropicRequestError(
            'rate limited', error_type='rate_limit_error', status_code=429)
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot(
            'codex', backend, session_pinned=True)
        selector = MagicMock()
        selector.is_paused.return_value = False
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(
            return_value={'messages': [{'role': 'user', 'content': 'Hello'}]})

        handler._handle_messages()

        # session is pinned — auto-switch must not happen
        selector.on_rate_limited.assert_not_called()
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 429

    def test_not_session_pinned_429_triggers_auto_switch(self):
        """Baseline: session_pinned=False behaves as before — auto-switch fires."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.side_effect = AnthropicRequestError(
            'rate limited', status_code=429)
        second_backend = MagicMock()
        second_backend.parse_credentials.return_value = {}
        second_backend.send_message.return_value = {'type': 'message', 'content': []}
        registry = MagicMock()
        registry.snapshot.side_effect = [
            _fake_snapshot('anthropic', backend, session_pinned=False),
            _fake_snapshot('bedrock', second_backend, session_pinned=False),
        ]
        selector = MagicMock()
        selector.is_paused.return_value = False
        selector.on_rate_limited.return_value = 'bedrock'
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(
            return_value={'messages': [{'role': 'user', 'content': 'Hello'}]})

        handler._handle_messages()

        selector.on_rate_limited.assert_called_once_with('anthropic', None)
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 200

    def test_session_subscription_429_records_exhaustion_and_returns_429(self):
        """session_subscription=True: note_exhausted recorded, on_rate_limited NOT called, 429 returned."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.side_effect = AnthropicRequestError(
            'rate limited', error_type='rate_limit_error', status_code=429)
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot(
            'codex', backend, session_pinned=True, session_subscription=True)
        selector = MagicMock()
        selector.is_paused.return_value = False
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(
            return_value={'messages': [{'role': 'user', 'content': 'Hello'}]})

        handler._handle_messages()

        # Exhaustion is recorded so next request resolves to the other sub
        selector.note_exhausted.assert_called_once_with('codex', None)
        # Global auto-switch must NOT fire
        selector.on_rate_limited.assert_not_called()
        # 429 is returned to the client
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 429

    def test_fixed_session_pin_429_does_not_record_exhaustion(self):
        """session_pinned=True, session_subscription=False: neither note_exhausted nor on_rate_limited called."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.side_effect = AnthropicRequestError(
            'rate limited', error_type='rate_limit_error', status_code=429)
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot(
            'codex', backend, session_pinned=True, session_subscription=False)
        selector = MagicMock()
        selector.is_paused.return_value = False
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(
            return_value={'messages': [{'role': 'user', 'content': 'Hello'}]})

        handler._handle_messages()

        selector.note_exhausted.assert_not_called()
        selector.on_rate_limited.assert_not_called()
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 429


# ---------------------------------------------------------------------------
# Handler: _session_set_backend_markdown
# ---------------------------------------------------------------------------

class TestSessionSetBackendMarkdown:
    def _handler(self):
        registry = MagicMock()
        registry.active_name.return_value = 'bedrock'
        registry.session_backend.return_value = None
        handler = object.__new__(ProxyRequestHandler)
        handler.registry = registry
        handler.config = MagicMock()
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        return handler, registry

    def test_no_session_key_returns_error(self):
        handler, registry = self._handler()
        result = handler._session_set_backend_markdown('openrouter', None)
        assert 'metadata.user_id' in result
        registry.set_session_backend.assert_not_called()

    def test_auto_clears_existing_override(self):
        handler, registry = self._handler()
        registry.clear_session_backend.return_value = True
        result = handler._session_set_backend_markdown('auto', 'sess-a')
        assert 'cleared' in result
        assert 'bedrock' in result
        registry.clear_session_backend.assert_called_once_with('sess-a')

    def test_auto_when_no_override(self):
        handler, registry = self._handler()
        registry.clear_session_backend.return_value = False
        result = handler._session_set_backend_markdown('auto', 'sess-a')
        assert 'already follows' in result

    def test_invalid_arg_returns_error(self):
        handler, registry = self._handler()
        result = handler._session_set_backend_markdown(None, 'sess-a')
        assert 'invalid' in result.lower()
        registry.set_session_backend.assert_not_called()

    def test_success_changed(self):
        from anthproxy.server import SwitchResult
        handler, registry = self._handler()
        registry.set_session_backend.return_value = SwitchResult(
            kind='changed', previous='bedrock', current='openrouter')
        result = handler._session_set_backend_markdown('openrouter', 'sess-a')
        assert 'openrouter' in result
        assert 'only' in result.lower() or 'session' in result.lower()
        assert 'proxy-set-backend:auto:session' in result
        registry.set_session_backend.assert_called_once_with('sess-a', 'openrouter')

    def test_unchanged(self):
        from anthproxy.server import SwitchResult
        handler, registry = self._handler()
        registry.set_session_backend.return_value = SwitchResult(
            kind='unchanged', previous='openrouter', current='openrouter')
        result = handler._session_set_backend_markdown('openrouter', 'sess-a')
        assert 'already' in result.lower()

    def test_failed(self):
        from anthproxy.server import SwitchResult
        handler, registry = self._handler()
        registry.set_session_backend.return_value = SwitchResult(
            kind='failed', previous='bedrock', current='bedrock',
            error='no credentials')
        result = handler._session_set_backend_markdown('codex', 'sess-a')
        assert 'no credentials' in result

    def test_subscription_locks_session(self):
        from anthproxy.server import SwitchResult
        handler, registry = self._handler()
        registry.set_session_subscription.return_value = SwitchResult(
            kind='changed', previous='bedrock', current='subscription')
        result = handler._session_set_backend_markdown('subscription', 'sess-a')
        registry.set_session_subscription.assert_called_once_with('sess-a')
        assert 'subscription' in result.lower()
        assert 'bedrock' in result.lower() or 'never' in result.lower()
        assert 'proxy-set-backend:auto:session' in result

    def test_subscription_unchanged(self):
        from anthproxy.server import SwitchResult
        handler, registry = self._handler()
        registry.set_session_subscription.return_value = SwitchResult(
            kind='unchanged', previous='subscription', current='subscription')
        result = handler._session_set_backend_markdown('subscription', 'sess-a')
        assert 'already' in result.lower()

    def test_subscription_no_session_key_errors(self):
        handler, registry = self._handler()
        result = handler._session_set_backend_markdown('subscription', None)
        assert 'metadata.user_id' in result
        registry.set_session_subscription.assert_not_called()


class TestSetBackendSubscriptionGlobal:
    def _handler(self, selector=None):
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        handler = object.__new__(ProxyRequestHandler)
        handler.registry = registry
        handler.config = MagicMock()
        handler.selector = selector
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        return handler, registry

    def test_subscription_restricts_selector(self):
        selector = MagicMock()
        selector.restrict_subscription.return_value = 'anthropic'
        handler, registry = self._handler(selector=selector)
        result = handler._set_backend_markdown('subscription')
        selector.restrict_subscription.assert_called_once()
        assert 'anthropic' in result
        assert 'bedrock' in result.lower()

    def test_subscription_without_selector_explains(self):
        handler, registry = self._handler(selector=None)
        result = handler._set_backend_markdown('subscription')
        assert 'auto-backend' in result.lower() or 'auto_backend' in result.lower() or 'auto' in result.lower()
        assert 'subscription' in result.lower() or 'enabled' in result.lower()


# ---------------------------------------------------------------------------
# proxy-set-model-routing:on|off command
# ---------------------------------------------------------------------------

class TestSetModelRoutingCommand:
    """Tests for the proxy-set-model-routing:on|off local command."""

    def test_parse_on(self):
        payload = _msg('proxy-set-model-routing:on')
        result = _parse_local_command(payload)
        assert result == ('set-model-routing', True)

    def test_parse_off(self):
        payload = _msg('proxy-set-model-routing:off')
        result = _parse_local_command(payload)
        assert result == ('set-model-routing', False)

    def test_parse_invalid_value(self):
        payload = _msg('proxy-set-model-routing:maybe')
        result = _parse_local_command(payload)
        assert result == ('set-model-routing', 'invalid')

    def test_parse_on_with_reminder_prefix(self):
        payload = _msg('<system-reminder>ctx</system-reminder>\nproxy-set-model-routing:on')
        result = _parse_local_command(payload)
        assert result == ('set-model-routing', True)

    def test_dispatch_on_calls_registry_and_returns_markdown(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        handler._handle_local_command(('set-model-routing', True), {'stream': False})
        registry.set_model_routing.assert_called_once_with(True)
        handler._send_json.assert_called_once()
        body = handler._send_json.call_args.args[1]
        text = body['content'][0]['text']
        assert 'on' in text

    def test_dispatch_off_calls_registry_and_returns_markdown(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        handler._handle_local_command(('set-model-routing', False), {'stream': False})
        registry.set_model_routing.assert_called_once_with(False)
        handler._send_json.assert_called_once()
        body = handler._send_json.call_args.args[1]
        text = body['content'][0]['text']
        assert 'off' in text

    def test_dispatch_invalid_value_returns_error_and_does_not_call_registry(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        handler._handle_local_command(('set-model-routing', 'invalid'), {'stream': False})
        registry.set_model_routing.assert_not_called()
        handler._send_json.assert_called_once()
        body = handler._send_json.call_args.args[1]
        text = body['content'][0]['text']
        assert 'on' in text and 'off' in text  # error explains valid values


# ---------------------------------------------------------------------------
# proxy-status: model routing line
# ---------------------------------------------------------------------------

class TestStatusModelRoutingLine:
    """Tests that _status_markdown includes a Model routing line."""

    def _make_handler(self, routing_on, classifier_model='haiku'):
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        registry.session_backend.return_value = None
        registry.instance.side_effect = Exception('no usage')
        handler = object.__new__(ProxyRequestHandler)
        handler.registry = registry
        handler.selector = None
        cfg = MagicMock()
        cfg.auto_model_routing = routing_on
        cfg.auto_model_routing_classifier_model = classifier_model
        cfg.auto_model_routing_mode = 'classifier'
        handler.config = cfg
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        return handler

    def test_status_shows_routing_off(self):
        handler = self._make_handler(routing_on=False)
        text = handler._status_markdown()
        assert '**Model routing:** off' in text

    def test_status_shows_routing_on_with_classifier(self):
        handler = self._make_handler(routing_on=True, classifier_model='haiku')
        text = handler._status_markdown()
        assert '**Model routing:** on' in text
        assert 'haiku' in text

    def test_status_classifier_model_shown_when_routing_on(self):
        handler = self._make_handler(routing_on=True, classifier_model='sonnet')
        text = handler._status_markdown()
        assert 'sonnet' in text  # classifier model name present



# ---------------------------------------------------------------------------
# proxy-set-model-routing:on|off:session command
# ---------------------------------------------------------------------------

class TestSessionSetModelRoutingCommand:
    """Tests for the proxy-set-model-routing:*:session local commands."""

    def test_parse_on_session(self):
        payload = _msg('proxy-set-model-routing:on:session')
        result = _parse_local_command(payload)
        assert result == ('session-set-model-routing', True)

    def test_parse_off_session(self):
        payload = _msg('proxy-set-model-routing:off:session')
        result = _parse_local_command(payload)
        assert result == ('session-set-model-routing', False)

    def test_parse_auto_session(self):
        payload = _msg('proxy-set-model-routing:auto:session')
        result = _parse_local_command(payload)
        assert result == ('session-set-model-routing', None)

    def test_parse_invalid_session(self):
        payload = _msg('proxy-set-model-routing:bogus:session')
        result = _parse_local_command(payload)
        assert result == ('session-set-model-routing', 'invalid')

    def test_parse_on_with_reminder_prefix_session(self):
        payload = _msg('<system-reminder>ctx</system-reminder>\nproxy-set-model-routing:on:session')
        result = _parse_local_command(payload)
        assert result == ('session-set-model-routing', True)

    def test_dispatch_on_session_calls_registry(self):
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        handler = _handler_with_registry(registry)
        payload = {'messages': [{'role': 'user', 'content': 'proxy-set-model-routing:on:session',
                                 'metadata': {'user_id': 'u1'}}],
                   'metadata': {'user_id': 'u1'}, 'stream': False}
        handler._handle_local_command(('session-set-model-routing', True), payload)
        registry.set_session_model_routing.assert_called_once()
        call_args = registry.set_session_model_routing.call_args
        assert call_args.args[1] is True
        handler._send_json.assert_called_once()
        body = handler._send_json.call_args.args[1]
        text = body['content'][0]['text']
        assert 'on' in text

    def test_dispatch_off_session_calls_registry(self):
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        handler = _handler_with_registry(registry)
        payload = {'metadata': {'user_id': 'u1'}, 'stream': False}
        handler._handle_local_command(('session-set-model-routing', False), payload)
        registry.set_session_model_routing.assert_called_once()
        call_args = registry.set_session_model_routing.call_args
        assert call_args.args[1] is False

    def test_dispatch_auto_session_clears_override(self):
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        handler = _handler_with_registry(registry)
        payload = {'metadata': {'user_id': 'u1'}, 'stream': False}
        handler._handle_local_command(('session-set-model-routing', None), payload)
        registry.set_session_model_routing.assert_called_once()
        call_args = registry.set_session_model_routing.call_args
        assert call_args.args[1] is None
        body = handler._send_json.call_args.args[1]
        text = body['content'][0]['text']
        assert 'global' in text.lower() or 'following' in text.lower()

    def test_dispatch_invalid_session_returns_error_and_does_not_call_registry(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {'metadata': {'user_id': 'u1'}, 'stream': False}
        handler._handle_local_command(('session-set-model-routing', 'invalid'), payload)
        registry.set_session_model_routing.assert_not_called()
        body = handler._send_json.call_args.args[1]
        text = body['content'][0]['text']
        assert 'on' in text and 'off' in text


class TestStatusSessionRoutingLine:
    """Tests that _status_markdown shows session routing override when set."""

    def _make_handler(self, routing_on, session_routing=None, sess_key='u1'):
        registry = MagicMock()
        registry.active_name.return_value = 'anthropic'
        registry.session_backend.return_value = None
        registry.session_model_routing.return_value = session_routing
        registry.instance.side_effect = Exception('no usage')
        handler = object.__new__(ProxyRequestHandler)
        handler.registry = registry
        handler.selector = None
        cfg = MagicMock()
        cfg.auto_model_routing = routing_on
        cfg.auto_model_routing_classifier_model = 'haiku'
        handler.config = cfg
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        return handler

    def test_status_shows_session_routing_on(self):
        handler = self._make_handler(routing_on=False, session_routing=True)
        text = handler._status_markdown('u1')
        assert 'on' in text.lower()
        assert 'session' in text.lower()

    def test_status_shows_session_routing_off(self):
        handler = self._make_handler(routing_on=True, session_routing=False)
        text = handler._status_markdown('u1')
        assert 'off' in text.lower()
        assert 'session' in text.lower()

    def test_status_no_session_note_when_no_override(self):
        handler = self._make_handler(routing_on=True, session_routing=None)
        text = handler._status_markdown('u1')
        # No session-routing note when there's no override
        assert 'proxy-set-model-routing:auto:session' not in text


# ---------------------------------------------------------------------------
# Session tier cache — handler integration tests
# ---------------------------------------------------------------------------

class TestSessionTierCache:
    """Handler-level tests for the session-cached-tier feature.

    Verifies that:
    (a) a successful classification stores the tier for the session;
    (b) a text-less (tool_result-only) follow-up reuses it and logs
        reason=session_cached_tier, making no extra classifier call;
    (c) text-less turn with no cache still fails closed
        (reason=missing_final_user_text, original model kept);
    (d) requests without a session key are unaffected.
    """

    _TOOL_RESULT_PAYLOAD = {
        'model': 'sonnet',
        'metadata': {'user_id': 'test-sess-1'},
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
            ]},
        ],
    }

    _TEXT_PAYLOAD = {
        'model': 'sonnet',
        'metadata': {'user_id': 'test-sess-1'},
        'messages': [
            {'role': 'user', 'content': 'please fix this bug'},
        ],
    }

    def _make_handler(self, *, auto_routing=True, cached_tier=None,
                      send_message_return=None):
        """Return a handler wired to a registry mock with controllable tier cache."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        if send_message_return is None:
            send_message_return = {'type': 'message', 'content': []}
        backend.send_message.return_value = send_message_return
        # Classifier response — always 'standard' for simplicity
        backend.send_classifier_message = MagicMock(
            return_value={
                'content': [{'type': 'text', 'text': 'standard'}],
            }
        )

        snapshot = _fake_snapshot('codex', backend)
        snapshot.config.auto_model_routing = auto_routing

        registry = MagicMock()
        registry.snapshot.return_value = snapshot
        registry.session_routed_tier.return_value = cached_tier
        registry.set_session_routed_tier = MagicMock()

        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'key'}
        handler._validate_content_type = MagicMock()
        return handler, registry, backend

    def _make_handler_with_custom_map(self, *, classification, auto_routing=True,
                                      cached_tier=None, send_message_return=None):
        """Same wiring as _make_handler, but with a custom
        auto_model_routing_classification dict on the snapshot's config — used
        to exercise the cap's label_map-aware reverse lookup at the handler
        layer (e.g. a custom target like 'fable' ranked as 'deep')."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        if send_message_return is None:
            send_message_return = {'type': 'message', 'content': []}
        backend.send_message.return_value = send_message_return
        # Classifier response — always 'standard' for simplicity
        backend.send_classifier_message = MagicMock(
            return_value={
                'content': [{'type': 'text', 'text': 'standard'}],
            }
        )

        snapshot = _fake_snapshot('codex', backend)
        snapshot.config.auto_model_routing = auto_routing
        snapshot.config.auto_model_routing_classification = dict(classification)

        registry = MagicMock()
        registry.snapshot.return_value = snapshot
        registry.session_routed_tier.return_value = cached_tier
        registry.set_session_routed_tier = MagicMock()

        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'key'}
        handler._validate_content_type = MagicMock()
        return handler, registry, backend

    def test_successful_classification_stores_tier(self, caplog):
        """A real-text turn that classifies to 'sonnet' should persist 'sonnet' in the cache."""
        import copy
        handler, registry, backend = self._make_handler(auto_routing=True, cached_tier=None)
        payload = copy.deepcopy(self._TEXT_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Tier 'sonnet' is written under the per-conversation key (session_id +
        # first-user-message hash), so a sub-agent never reuses the parent's tier.
        expected_key = 'test-sess-1\x00' + _conversation_anchor(self._TEXT_PAYLOAD)
        registry.set_session_routed_tier.assert_called_once_with(
            expected_key, 'sonnet')

    def test_tool_result_only_reuses_cached_tier(self, caplog):
        """A text-less continuation reuses cached 'opus' and logs session_cached_tier.

        Requested 'opus' here so the no-upgrade cap does not fire — this test
        exercises the plain reuse path (see test_session_cached_tier_capped_to_
        requested for the cap)."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='opus')
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)
        payload['model'] = 'opus'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Model should be rewritten to the cached tier
        assert payload['model'] == 'opus'
        # Classifier must NOT have been called (no extra round trip)
        backend.send_classifier_message.assert_not_called()
        # Log should show session_cached_tier
        assert 'reason=session_cached_tier' in caplog.text

    def test_tool_result_no_cache_fails_closed(self, caplog):
        """A text-less continuation with nothing cached keeps the original model."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier=None)
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)
        original_model = payload['model']
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert payload['model'] == original_model
        assert 'reason=missing_final_user_text' in caplog.text
        backend.send_classifier_message.assert_not_called()

    def test_size_floor_forces_opus_1m_and_does_not_cache_tier(self, caplog):
        """A huge request is forced to opus[1m] without a classifier call or tier write."""
        import copy
        handler, registry, backend = self._make_handler(auto_routing=True, cached_tier=None)
        snapshot = registry.snapshot.return_value
        snapshot.config.auto_model_routing_long_context_threshold = 10
        payload = {
            'model': 'sonnet',
            'metadata': {'user_id': 'test-sess-1'},
            'messages': [{'role': 'user', 'content': 'x' * 4000}],
        }
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=copy.deepcopy(payload))

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert 'reason=size_forced_long_context' in caplog.text
        assert 'routed=opus[1m]' in caplog.text
        # Deterministic floor: no classifier call, and classification is None so the
        # session tier cache is not written.
        backend.send_classifier_message.assert_not_called()
        registry.set_session_routed_tier.assert_not_called()

    def test_no_session_key_no_caching(self, caplog):
        """Requests without metadata.user_id are never read from or written to the cache."""
        import copy
        payload = {
            'model': 'sonnet',
            # no metadata
            'messages': [{'role': 'user', 'content': 'what is 2+2?'}],
        }
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='haiku')
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=copy.deepcopy(payload))

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Cache should not be consulted or written for sess_key=None
        registry.session_routed_tier.assert_not_called()
        registry.set_session_routed_tier.assert_not_called()

    def test_routing_disabled_no_cache_ops(self):
        """When auto_model_routing is False, neither cache read nor write occurs."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=False, cached_tier='sonnet')
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        registry.session_routed_tier.assert_not_called()
        registry.set_session_routed_tier.assert_not_called()
        # Model must be unchanged
        assert payload['model'] == 'sonnet'

    def test_cache_hit_applied_flag_reflects_model_change(self, caplog):
        """applied=True when cached tier differs from requested model."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='haiku')
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)  # model='sonnet'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert payload['model'] == 'haiku'
        assert 'applied=True' in caplog.text
        assert 'reason=session_cached_tier' in caplog.text

    def test_cache_hit_applied_false_when_tier_matches_requested(self, caplog):
        """applied=False when cached tier equals the originally-requested model."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='sonnet')  # same as payload model
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)  # model='sonnet'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert payload['model'] == 'sonnet'
        assert 'applied=False' in caplog.text
        assert 'reason=session_cached_tier' in caplog.text

    _WALKBACK_PAYLOAD = {
        'model': 'sonnet',
        'metadata': {'user_id': 'test-sess-1'},
        'messages': [
            {'role': 'user', 'content': 'redesign the auth layer'},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
            ]},
        ],
    }

    def test_walkback_reuses_cached_tier_without_classifying(self, caplog):
        """A tool_result-only final turn (walk-back continuation) with a cached
        session tier reuses the cache instead of re-classifying recovered prose.
        This avoids misclassifying boilerplate and saves a classifier call."""
        import copy
        # Cache holds 'opus' from a prior turn with real intent.
        # Walk-back recovers boilerplate from a skill prompt, but we skip
        # classification and reuse 'opus' instead of downgrading to 'trivial'.
        # Requested 'opus' so the no-upgrade cap does not fire (see
        # TestNoUpgradeCap / test_walkback_cap_logged_at_handler for the cap).
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='opus')
        payload = copy.deepcopy(self._WALKBACK_PAYLOAD)
        payload['model'] = 'opus'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Classifier should NOT be called — we skip straight to cache reuse.
        backend.send_classifier_message.assert_not_called()
        assert 'reason=session_cached_walkback' in caplog.text
        assert 'reason=classifier_' not in caplog.text
        assert payload['model'] == 'opus'
        # Cache is NOT updated (no new classification), preserving the prior tier.

    def test_session_cached_tier_capped_to_requested(self, caplog):
        """A text-less continuation (missing_final_user_text) must not replay a
        cached tier above the requested model — logs session_cached_tier_capped."""
        import copy
        # Cache holds 'opus' but this turn requested 'sonnet'.
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='opus')
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)  # model='sonnet'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Kept the requested tier, not the cached upgrade.
        assert payload['model'] == 'sonnet'
        assert 'reason=session_cached_tier_capped' in caplog.text
        assert 'applied=False' in caplog.text
        backend.send_classifier_message.assert_not_called()
        # The cap must not poison the cache.
        registry.set_session_routed_tier.assert_not_called()

    def test_walkback_cap_logged_at_handler(self, caplog):
        """End-to-end: a haiku-requested walk-back continuation with a cached
        'sonnet' bypasses the cap (tool_result-only agentic turns), replaying
        sonnet instead of being capped to haiku."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='sonnet')
        payload = copy.deepcopy(self._WALKBACK_PAYLOAD)
        payload['model'] = 'claude-haiku-4-5-20251001'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Tool_result-only bypass: model is now sonnet (cached), not haiku (requested)
        assert payload['model'] == 'sonnet'
        assert 'reason=session_cached_walkback_tool_result' in caplog.text
        assert 'applied=True' in caplog.text
        backend.send_classifier_message.assert_not_called()

    def test_lock_requested_model_baseline_for_routing(self, caplog):
        """When lock_requested_model is set, routing uses the locked baseline for tier
        decisions while preserving the client's original model for response echo and applied flag."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier=None)
        payload = copy.deepcopy(self._TEXT_PAYLOAD)  # client model='sonnet'
        payload['model'] = 'haiku'  # client requests haiku
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)
        # Classifier will say 'deep' (should route to opus)
        handler.registry.snapshot.return_value.config.lock_requested_model = 'claude-sonnet-4-6'
        backend.send_classifier_message.return_value = {
            'content': [{'type': 'text', 'text': 'deep'}],
        }

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Routing used sonnet baseline: 'deep' → opus, not haiku
        assert payload['model'] == 'opus'
        # applied=True: routed (opus) != requested (haiku)
        assert 'applied=True' in caplog.text
        # The log should show the lock was applied
        assert 'Model lock: forcing routing baseline from haiku to claude-sonnet-4-6' in caplog.text
        backend.send_classifier_message.assert_called_once()

    def test_lock_requested_model_missing_text_cap_uses_lock_baseline(self, caplog):
        """When lock_requested_model is set and missing_final_user_text fires,
        the cached-tier no-upgrade cap uses the lock baseline, not the client model."""
        import copy
        # Cache holds 'opus' but client requested 'haiku' and lock='sonnet'
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='opus')
        payload = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)  # tool-result-only (no text)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)
        handler.registry.snapshot.return_value.config.lock_requested_model = 'claude-sonnet-4-6'

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Cached opus is capped against sonnet baseline: sonnet < opus, so return sonnet
        assert payload['model'] == 'claude-sonnet-4-6'
        assert 'reason=session_cached_tier_capped' in caplog.text
        # applied=False: routed (sonnet) == requested (sonnet)... wait, requested is the
        # client's original model from the payload. Let's check:
        # The payload from _TOOL_RESULT_PAYLOAD has model='sonnet', but we don't change it.
        # The lock only affects routing; applied compares routed vs requested (client model).
        # Since routed is capped to the lock baseline (sonnet) and requested is the client's
        # 'sonnet', applied=False. But the cap WAS applied (cached opus → sonnet).
        # The reason_code='session_cached_tier_capped' indicates the cap fired.
        backend.send_classifier_message.assert_not_called()

    def test_lock_requested_model_applies_even_routing_disabled(self, caplog):
        """Lock is applied at input, before routing is even checked."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=False)  # routing off
        payload = copy.deepcopy(self._TEXT_PAYLOAD)
        payload['model'] = 'haiku'
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)
        handler.registry.snapshot.return_value.config.lock_requested_model = 'claude-sonnet-4-6'

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Lock is applied at input substitution, regardless of routing state
        assert payload['model'] == 'claude-sonnet-4-6'
        assert 'reason=disabled' in caplog.text
        backend.send_classifier_message.assert_not_called()

    # --- session-context floor (record + consume) ------------------------------

    def test_records_session_context_includes_cache_tokens(self):
        """Non-streaming response records floor = input + cache_read + cache_creation + output."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier=None,
            send_message_return={
                'type': 'message', 'content': [],
                'usage': {
                    'input_tokens': 500,
                    'cache_read_input_tokens': 180_000,
                    'cache_creation_input_tokens': 2_000,
                    'output_tokens': 1_500,
                },
            })
        payload = copy.deepcopy(self._TEXT_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        registry.record_session_context.assert_called_once()
        sess, floor, ratio = registry.record_session_context.call_args[0]
        # Floor is keyed on (session_id, first-user-message hash) so a sub-agent
        # does not collide with its parent.
        assert sess == 'test-sess-1\x00' + _conversation_anchor(self._TEXT_PAYLOAD)
        assert floor == 500 + 2_000 + 180_000 + 1_500  # 184000 — caching included
        assert ratio == 1.0  # tiny request → baseline too small, prior ratio kept

    def test_large_cached_floor_forces_opus_1m_on_small_request(self, caplog):
        """A small turn in a session already near the window is forced to opus[1m]."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier=None,
            send_message_return={'type': 'message', 'content': [],
                                 'usage': {'input_tokens': 10}})
        registry.session_context.return_value = (195_000, 1.0)
        payload = copy.deepcopy(self._TEXT_PAYLOAD)  # small text
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert payload['model'] == 'opus[1m]'
        assert 'reason=size_forced_long_context' in caplog.text
        backend.send_classifier_message.assert_not_called()

    def test_subagent_floor_key_isolated_from_parent(self):
        """A sub-agent (same metadata.user_id, different launch prompt) reads and
        records its size-context floor under a DISTINCT key from its parent.

        Claude Code reuses the parent's metadata.user_id for Task sub-agents, so
        without folding the first-user-message hash into the key, a sub-agent would
        inherit (and clobber) the parent's large floor.  Both turns share the same
        session_id but their initiating user messages differ, so the floor keys
        must differ while keeping the common session prefix.
        """
        import copy

        def _floor_key_for(launch_text):
            handler, registry, backend = self._make_handler(
                auto_routing=True, cached_tier=None,
                send_message_return={'type': 'message', 'content': [],
                                     'usage': {'input_tokens': 10}})
            payload = copy.deepcopy(self._TEXT_PAYLOAD)
            payload['messages'][0]['content'] = launch_text
            handler._read_body = MagicMock(return_value=b'{}')
            handler._parse_json = MagicMock(return_value=payload)
            handler._handle_messages()
            # Same key feeds the read (session_context) and the write.
            read_key = registry.session_context.call_args[0][0]
            write_key = registry.record_session_context.call_args[0][0]
            assert read_key == write_key
            return read_key

        parent_key = _floor_key_for('Please review the whole repository for me.')
        subagent_key = _floor_key_for('You are an Explore sub-agent. Read foo.py.')

        assert parent_key != subagent_key
        assert parent_key.startswith('test-sess-1\x00')
        assert subagent_key.startswith('test-sess-1\x00')

    def test_subagent_tier_cache_isolated_from_parent(self):
        """A sub-agent reads/writes the routed-tier cache under a DISTINCT key
        from its parent (same session_id, different launch prompt).

        Observed in production: the parent cached 'opus', and haiku sub-agent
        continuation turns reused it via session_cached_walkback because the tier
        cache was keyed on the bare (shared) session.  Per-conversation keying
        prevents the leak.
        """
        import copy

        def _tier_key_for(launch_text):
            handler, registry, backend = self._make_handler(
                auto_routing=True, cached_tier=None)
            payload = copy.deepcopy(self._TEXT_PAYLOAD)
            payload['messages'][0]['content'] = launch_text
            handler._read_body = MagicMock(return_value=b'{}')
            handler._parse_json = MagicMock(return_value=payload)
            handler._handle_messages()
            return registry.set_session_routed_tier.call_args[0][0]

        parent_key = _tier_key_for('Please review the whole repository for me.')
        subagent_key = _tier_key_for('You are an Explore sub-agent. Read foo.py.')

        assert parent_key != subagent_key
        assert parent_key.startswith('test-sess-1\x00')
        assert subagent_key.startswith('test-sess-1\x00')

    def test_routing_off_does_not_record_session_context(self):
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=False,
            send_message_return={'type': 'message', 'content': [],
                                 'usage': {'input_tokens': 50_000}})
        payload = copy.deepcopy(self._TEXT_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        registry.record_session_context.assert_not_called()

    # --- short-affirmation continuation (inherit / floor, no cache poison) -----

    _AFFIRMATION_PAYLOAD = {
        'model': 'sonnet',
        'metadata': {'user_id': 'test-sess-1'},
        'messages': [
            {'role': 'user', 'content': 'design the migration plan'},
            {'role': 'assistant', 'content': 'Here is the plan. Ready to start?'},
            {'role': 'user', 'content': 'yes'},
        ],
    }

    def test_affirmation_inherits_cached_tier_without_classifying(self, caplog):
        """A bare 'yes' with a cached opus tier inherits opus, no classifier call,
        and crucially does NOT write the tier cache (no haiku poisoning)."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='opus')
        payload = copy.deepcopy(self._AFFIRMATION_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert payload['model'] == 'opus'
        backend.send_classifier_message.assert_not_called()
        assert 'reason=affirmation_inherited' in caplog.text
        # classification is None → the tier cache is never written, so the
        # affirmation can never poison the conversation's established tier.
        registry.set_session_routed_tier.assert_not_called()

    def test_affirmation_classifies_when_uncached_with_prior_response(self, caplog):
        """A bare 'yes' with no cached tier and a prior assistant message calls the
        classifier with enriched input; the result is written to the tier cache."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier=None)
        payload = copy.deepcopy(self._AFFIRMATION_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        assert payload['model'] == 'sonnet'  # classifier returned 'standard' → sonnet
        backend.send_classifier_message.assert_called_once()
        assert 'reason=affirmation_classified' in caplog.text
        registry.set_session_routed_tier.assert_called_once()

    def test_affirmation_full_chain_complex_task_keeps_opus(self, caplog):
        """End-to-end: a cached opus tier survives a 'yes' affirmation and the
        following tool_result-only continuation still replays opus via
        session_cached_walkback — the kicked-off complex task is NOT downgraded."""
        import copy
        # Turn A: the affirmation. Cache holds opus from the planning turns.
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier='opus')
        affirm = copy.deepcopy(self._AFFIRMATION_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=affirm)
        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()
        assert affirm['model'] == 'opus'
        assert 'reason=affirmation_inherited' in caplog.text
        # The affirmation left the cache untouched (still opus, not haiku).
        registry.set_session_routed_tier.assert_not_called()

        # Turn B: the complex task now runs; its first continuation is a
        # tool_result-only turn.  In a real opus session Claude Code sends these
        # continuations with model='opus' (the main loop's model), so the cached
        # opus replays uncapped — the kicked-off task is NOT downgraded.
        caplog.clear()
        handler2, registry2, backend2 = self._make_handler(
            auto_routing=True, cached_tier='opus')
        cont = copy.deepcopy(self._WALKBACK_PAYLOAD)
        cont['model'] = 'opus'
        handler2._read_body = MagicMock(return_value=b'{}')
        handler2._parse_json = MagicMock(return_value=cont)
        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler2._handle_messages()
        assert cont['model'] == 'opus'
        assert 'reason=session_cached_walkback' in caplog.text
        backend2.send_classifier_message.assert_not_called()

    def test_affirmation_knob_disabled_classifies_normally(self, caplog):
        """With the knob off, a 'yes' is classified normally (documents the known
        limitation: the classifier may return trivial and poison the cache)."""
        import copy
        handler, registry, backend = self._make_handler(
            auto_routing=True, cached_tier=None)
        snapshot = registry.snapshot.return_value
        snapshot.config.auto_model_routing_affirmation_inherit = False
        # Classifier mock returns 'standard' by default → tier write happens.
        payload = copy.deepcopy(self._AFFIRMATION_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # Knob off → the classifier IS consulted for the affirmation turn.
        backend.send_classifier_message.assert_called_once()
        assert 'reason=affirmation_inherited' not in caplog.text
        assert 'reason=affirmation_floored_standard' not in caplog.text
        registry.set_session_routed_tier.assert_called_once()

    def test_session_cached_tier_capped_with_custom_label_map(self, caplog):
        """The post-route cap (session_cached_tier_capped) respects a custom
        auto_model_routing_classification label_map: a cached custom target
        like 'fable' (mapped to the 'deep' slot, rank=2) still caps down to a
        lower-ranked requested model ('haiku', rank=0)."""
        import copy
        classification = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}

        # --- Step 1: prime the session tier cache with a fresh classification
        # that routes to the custom 'deep' target ('fable').
        handler, registry, backend = self._make_handler_with_custom_map(
            classification=classification, auto_routing=True, cached_tier=None)
        backend.send_classifier_message.return_value = {
            'content': [{'type': 'text', 'text': 'deep'}],
        }
        payload = copy.deepcopy(self._TEXT_PAYLOAD)
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)
        handler._handle_messages()

        expected_key = 'test-sess-1\x00' + _conversation_anchor(self._TEXT_PAYLOAD)
        registry.set_session_routed_tier.assert_called_once_with(
            expected_key, 'fable')

        # --- Step 2: a new tool-result-only (text-less) turn requesting
        # 'haiku'.  Simulate the persisted cache from step 1 via a fresh
        # handler whose session_routed_tier mock returns the primed value.
        handler2, registry2, backend2 = self._make_handler_with_custom_map(
            classification=classification, auto_routing=True, cached_tier='fable')
        payload2 = copy.deepcopy(self._TOOL_RESULT_PAYLOAD)
        payload2['model'] = 'haiku'
        handler2._read_body = MagicMock(return_value=b'{}')
        handler2._parse_json = MagicMock(return_value=payload2)

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler2._handle_messages()

        assert payload2['model'] == 'haiku'
        assert 'reason=session_cached_tier_capped' in caplog.text
        backend2.send_classifier_message.assert_not_called()
        # The cap must not poison the cache with a fresh write.
        registry2.set_session_routed_tier.assert_not_called()


class TestContextKey:
    @staticmethod
    def _turn1(text, system='SYS'):
        return {'system': system, 'messages': [{'role': 'user', 'content': text}]}

    @staticmethod
    def _continuation(text, system='SYS'):
        # Same conversation, later turn: full history resent, final turn is a
        # tool_result-only user message (the case that defeated the system hash).
        return {
            'system': system,
            'messages': [
                {'role': 'user', 'content': text},
                {'role': 'assistant', 'content': 'doing work'},
                {'role': 'user', 'content': [{'type': 'tool_result', 'content': 'ok'}]},
            ],
        }

    def test_stable_across_turns_despite_system_change(self):
        # Same first user message, different system on the continuation turn ->
        # SAME key.  This is the regression the old system-hash key failed.
        turn1 = self._turn1('You are fixing 6 findings in adapters', system='SYS-A')
        cont = self._continuation('You are fixing 6 findings in adapters', system='SYS-B-different')
        assert _context_key('s', turn1) == _context_key('s', cont)

    def test_distinct_subagents_distinct_keys(self):
        # Same session + same system, different launch prompts -> distinct keys.
        a = _context_key('s', self._turn1('You are fixing 6 findings in adapters'))
        b = _context_key('s', self._turn1('You are fixing 3 findings in config'))
        assert a != b
        assert a.startswith('s\x00') and b.startswith('s\x00')

    def test_system_change_does_not_change_key(self):
        same_msg = 'identical launch prompt'
        assert (_context_key('s', self._turn1(same_msg, system='parent'))
                == _context_key('s', self._turn1(same_msg, system='sub-agent')))

    def test_no_session_key_returns_none(self):
        assert _context_key(None, self._turn1('x')) is None
        assert _context_key('', self._turn1('x')) is None

    def test_no_messages_uses_sentinel(self):
        assert _context_key('s', {}) == 's\x00--------'
        assert _context_key('s', {'messages': []}) == 's\x00--------'

    def test_tool_result_only_first_turn_uses_sentinel(self):
        payload = {'messages': [{'role': 'user',
                                 'content': [{'type': 'tool_result', 'content': 'r'}]}]}
        assert _context_key('s', payload) == 's\x00--------'

    def test_string_and_block_message_forms(self):
        # String content and a single text block with the same text hash equally.
        text = 'shared launch text'
        as_str = _context_key('s', self._turn1(text))
        as_block = _context_key('s', {'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': text}]}]})
        assert as_str == as_block

    def test_reminders_stripped_before_hashing(self):
        wrapped = '<system-reminder>noise</system-reminder>real task'
        assert (_context_key('s', self._turn1(wrapped))
                == _context_key('s', self._turn1('real task')))

    def test_anchor_skips_leading_assistant_turn(self):
        # First *user* turn anchors even if not at index 0.
        payload = {'messages': [
            {'role': 'assistant', 'content': 'preamble'},
            {'role': 'user', 'content': 'the real launch'},
        ]}
        assert _conversation_anchor(payload) == _conversation_anchor(
            {'messages': [{'role': 'user', 'content': 'the real launch'}]})

    def test_fingerprint_head_and_counts(self):
        blocks, chars, digest, head = _system_fingerprint(
            {'system': [{'type': 'text', 'text': 'hello'},
                        {'type': 'text', 'text': 'world'}]})
        assert blocks == 2
        assert chars == len('hello\nworld')
        assert len(digest) == 8
        assert head == 'hello world'


class TestStreamingSessionRecording:
    """_stats_sse_wrapper records the session-context floor from stream usage."""

    def _chunk(self, body):
        return 'event: x\ndata: ' + json.dumps(body) + '\n\n'

    def test_records_floor_and_ratio_from_stream(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        handler.stats_collector = MagicMock()
        handler.session_db = None
        handler._ctx_key = 'sess-strm'
        handler._route_est = 50_000  # large enough baseline → ratio computed
        chunks = [
            self._chunk({'type': 'message_start', 'message': {'usage': {
                'input_tokens': 1_000,
                'cache_read_input_tokens': 160_000,
                'cache_creation_input_tokens': 4_000,
            }}}),
            self._chunk({'type': 'message_delta', 'usage': {'output_tokens': 2_000}}),
        ]
        list(handler._usage_sse_wrapper(iter(chunks), None, 'anthropic', 0.0, 'opus'))

        registry.record_session_context.assert_called_once()
        sess, floor, ratio = registry.record_session_context.call_args[0]
        assert sess == 'sess-strm'
        assert floor == 1_000 + 4_000 + 160_000 + 2_000  # 167000
        # measured_input 165000 / est 50000 = 3.3 → clamped to _RATIO_MAX (3.0)
        assert ratio == 3.0

    def test_no_sess_key_does_not_record(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        handler.stats_collector = MagicMock()
        handler.session_db = None
        handler._ctx_key = None
        chunks = [self._chunk({'type': 'message_start',
                               'message': {'usage': {'input_tokens': 9_000}}})]
        list(handler._usage_sse_wrapper(iter(chunks), None, 'anthropic', 0.0, 'opus'))
        registry.record_session_context.assert_not_called()


class TestLogTag:
    def _bare_handler(self):
        handler = object.__new__(ProxyRequestHandler)
        return handler

    def test_tag_with_session_and_elapsed(self):
        handler = self._bare_handler()
        handler._session_prefix = 'abcde'
        handler._session_hash = 'deadbeef'
        handler._req_start = time.monotonic()
        tag = handler._log_tag()
        assert tag.startswith('[abcde deadbeef +')
        assert tag.endswith('s]')

    def test_tag_without_session_uses_placeholder(self):
        handler = self._bare_handler()
        handler._session_prefix = None
        handler._req_start = time.monotonic()
        # Neither _session_hash set → falls back to '--------' for both slots.
        assert handler._log_tag().startswith('[-------- -------- +')

    def test_tag_missing_start_omits_elapsed(self):
        # Paths that emit before do_POST stamping (e.g. an early 404).
        handler = self._bare_handler()
        assert handler._log_tag() == '[-------- --------]'


def _disconnect_handler(payload, *, stream, backend_result):
    """Build a handler driving _handle_messages with real _send_sse/_send_json
    but a socket whose header commit raises BrokenPipeError."""
    backend = MagicMock()
    backend.parse_credentials.return_value = {'token': 'ok'}
    if stream:
        backend.send_message_stream.return_value = iter(backend_result)
    else:
        backend.send_message.return_value = backend_result
    registry = MagicMock()
    registry.snapshot.return_value = _fake_snapshot('anthropic', backend)
    registry.session_routed_tier.return_value = None
    handler = object.__new__(ProxyRequestHandler)
    handler.registry = registry
    handler.config = MagicMock()
    handler.selector = None
    handler.stats_collector = None
    handler.headers = {'x-api-key': 'secret'}
    handler._req_start = time.monotonic()
    handler._session_prefix = None
    handler._validate_content_type = MagicMock()
    handler._read_body = MagicMock(return_value=b'{}')
    handler._parse_json = MagicMock(return_value=payload)
    # Mock the header plumbing; break the socket at end_headers (the very spot
    # where a cancelled request surfaces BrokenPipeError in production).
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock(side_effect=BrokenPipeError())
    handler.wfile = MagicMock()
    return handler


class TestClientDisconnect:
    def test_streaming_disconnect_logs_single_info_no_traceback(self, caplog):
        payload = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}
        handler = _disconnect_handler(payload, stream=True, backend_result=['c1', 'c2'])

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()  # must not raise

        assert 'client disconnected mid-stream' in caplog.text
        assert 'Proxy failure' not in caplog.text
        disconnects = [r for r in caplog.records if 'client disconnected' in r.getMessage()]
        assert len(disconnects) == 1
        # The single line carries the session/elapsed tag.
        assert '+' in disconnects[0].getMessage() and 's]' in disconnects[0].getMessage()

    def test_nonstream_disconnect_swallowed_no_double_fault(self, caplog):
        payload = {'messages': [{'role': 'user', 'content': 'Hi'}]}
        handler = _disconnect_handler(
            payload, stream=False, backend_result={'type': 'message', 'content': []})

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()  # must not raise

        assert 'client disconnected before response' in caplog.text
        assert 'Proxy failure' not in caplog.text

    def test_streaming_disconnect_log_has_ttfb_and_counters(self, caplog):
        # The enriched log should include ttfb=, chunks_sent=, and bytes_sent=.
        payload = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}
        handler = _disconnect_handler(payload, stream=True, backend_result=['c1', 'c2'])

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        disconnect_msgs = [r.getMessage() for r in caplog.records
                           if 'client disconnected mid-stream' in r.getMessage()]
        assert len(disconnect_msgs) == 1
        msg = disconnect_msgs[0]
        # The mock breaks at end_headers before any write → no bytes sent.
        assert 'ttfb=' in msg
        assert 'chunks_sent=0' in msg
        assert 'bytes_sent=0' in msg

    def test_streaming_disconnect_mid_loop_counters_and_generator_closed(self, caplog):
        # A disconnect that happens after at least one chunk is written should
        # reflect chunks_sent>0, and the upstream generator should be closed.
        import time as _time

        cleaned_up = []

        def real_backend_gen():
            # Wrap all yields so GeneratorExit (thrown by .close()) is caught
            # regardless of which yield the generator is suspended at.
            try:
                yield 'chunk_one'
                yield 'chunk_two'
                yield 'chunk_three'
            except GeneratorExit:
                cleaned_up.append(True)
                raise

        payload = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message_stream.return_value = real_backend_gen()
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot('anthropic', backend)
        registry.session_routed_tier.return_value = None

        handler = object.__new__(ProxyRequestHandler)
        handler.registry = registry
        handler.config = MagicMock()
        handler.selector = None
        handler.stats_collector = None
        handler.headers = {'x-api-key': 'secret'}
        handler._req_start = _time.monotonic()
        handler._session_prefix = None
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        call_count = [0]

        def write_side_effect(data):
            call_count[0] += 1
            if call_count[0] >= 2:  # fail on the second chunk write
                raise BrokenPipeError()

        handler.wfile = MagicMock()
        handler.wfile.write.side_effect = write_side_effect
        handler.wfile.flush = MagicMock()

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        disconnect_msgs = [r.getMessage() for r in caplog.records
                           if 'client disconnected mid-stream' in r.getMessage()]
        assert len(disconnect_msgs) == 1
        msg = disconnect_msgs[0]
        assert 'chunks_sent=1' in msg     # one successful write before disconnect
        assert 'bytes_sent=9' in msg      # len('chunk_one') == 9
        assert 'Proxy failure' not in caplog.text
        # Generator's finally ran — upstream was closed.
        assert cleaned_up == [True]

    def test_routing_log_carries_session_prefix(self, caplog):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.return_value = {'type': 'message', 'content': []}
        registry = MagicMock()
        registry.snapshot.return_value = _fake_snapshot('anthropic', backend)
        registry.session_routed_tier.return_value = None
        handler = _handler_with_registry(registry)
        handler.selector = None
        handler.stats_collector = None
        handler.headers = {'x-api-key': 'secret'}
        handler._req_start = time.monotonic()
        handler._session_prefix = None
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value={
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'metadata': {'user_id': 'abcdefgh-session'},
        })

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._handle_messages()

        # 'abcdefgh-session' has no UUID → sha256 fallback; no system → '--------'
        assert '[e2fd8dfc -------- +' in caplog.text


def _sse_handler(*, keepalive_interval):
    """Build a handler with a real _send_sse and a recording wfile.

    ``handler.wfile.write`` appends each written bytes object to
    ``handler.written`` so tests can assert on the emitted SSE stream.
    """
    handler = object.__new__(ProxyRequestHandler)
    handler._req_start = time.monotonic()
    handler._session_prefix = None
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.written = []
    handler.wfile = MagicMock()
    handler.wfile.write.side_effect = handler.written.append
    handler.wfile.flush = MagicMock()
    cfg = SimpleNamespace(sse_keepalive_interval=keepalive_interval)
    return handler, cfg


class TestSsePriming:
    def test_keepalive_sent_while_priming_slow(self, caplog):
        # Generator whose first chunk arrives only after a short delay; with a
        # tiny keepalive interval the client should receive keepalive comments
        # before the first real chunk.
        ready = threading.Event()

        def slow_gen():
            ready.wait(timeout=2.0)
            yield 'event: message_start\ndata: {}\n\n'
            yield 'event: message_stop\ndata: {}\n\n'

        handler, cfg = _sse_handler(keepalive_interval=0.02)

        def release_soon():
            time.sleep(0.1)  # ~5 keepalive intervals
            ready.set()

        releaser = threading.Thread(target=release_soon, daemon=True)
        releaser.start()

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._send_sse(slow_gen(), config=cfg)
        releaser.join(timeout=2.0)

        written = b''.join(handler.written)
        assert b': keepalive\n\n' in written          # at least one keepalive
        assert b'event: message_start' in written     # real chunks followed
        assert b'event: message_stop' in written
        # keepalive comments precede the first real data frame
        assert written.index(b': keepalive') < written.index(b'event: message_start')
        assert 'SSE keepalive complete' in caplog.text

    def test_keepalive_disabled_no_comments(self):
        def gen():
            yield 'event: message_start\ndata: {}\n\n'

        handler, cfg = _sse_handler(keepalive_interval=0)
        handler._send_sse(gen(), config=cfg)

        written = b''.join(handler.written)
        assert b': keepalive' not in written
        assert b'event: message_start' in written

    def test_priming_429_emits_sse_error_event(self, caplog):
        def raise_gen():
            raise AnthropicRequestError('rate limited', error_type='rate_limit_error',
                                        status_code=429)
            yield  # pragma: no cover

        handler, cfg = _sse_handler(keepalive_interval=0)

        with caplog.at_level(logging.WARNING, logger='anthproxy.handlers'):
            handler._send_sse(raise_gen(), config=cfg)

        written = b''.join(handler.written)
        # Canonical Anthropic SSE error frame is emitted in-band.
        assert b'event: error' in written
        assert b'"type": "error"' in written
        assert b'rate_limit_error' in written
        assert b'rate limited' in written
        assert 'SSE priming error (post-header)' in caplog.text

    def test_mid_stream_request_error_emits_sse_error_event(self, caplog):
        first_frame = 'event: message_start\ndata: {}\n\n'

        def raise_after_first_frame():
            yield first_frame
            raise AnthropicRequestError('Prompt is too long',
                                        error_type='invalid_request_error',
                                        status_code=400)

        handler, cfg = _sse_handler(keepalive_interval=0)

        with caplog.at_level(logging.WARNING, logger='anthproxy.handlers'):
            handler._send_sse(raise_after_first_frame(), config=cfg)

        written = b''.join(handler.written)
        assert first_frame.encode('utf-8') in written
        assert b'event: error' in written
        assert b'"type": "error"' in written
        assert b'invalid_request_error' in written
        assert b'Prompt is too long' in written
        assert written.index(first_frame.encode('utf-8')) < written.index(b'event: error')
        assert 'SSE stream request error after headers committed' in caplog.text

    def test_keepalive_client_disconnect_logged_and_thread_joined(self, caplog):
        # Client drops during the keepalive phase: the keepalive write raises
        # BrokenPipeError. The handler must log it, wait for priming, not crash.
        ready = threading.Event()
        primed_cleanly = []

        def slow_gen():
            ready.wait(timeout=2.0)
            primed_cleanly.append(True)
            yield 'event: message_start\ndata: {}\n\n'

        handler, cfg = _sse_handler(keepalive_interval=0.02)
        handler.wfile.write.side_effect = BrokenPipeError()

        def release_soon():
            time.sleep(0.1)
            ready.set()

        releaser = threading.Thread(target=release_soon, daemon=True)
        releaser.start()

        with caplog.at_level(logging.INFO, logger='anthproxy.handlers'):
            handler._send_sse(slow_gen(), config=cfg)  # must not raise
        releaser.join(timeout=2.0)

        assert 'client disconnected during SSE keepalive' in caplog.text
        # Priming thread was awaited (not abandoned): the generator advanced.
        assert primed_cleanly == [True]


# ---------------------------------------------------------------------------
# TestStatsFailureRecording — failure paths emit stats records
# ---------------------------------------------------------------------------

class TestStatsFailureRecording:
    """Verify that _handle_messages emits a failure stats record on error paths.

    Tests use _handle_messages directly (the full request path), so that the
    exception handlers under test are the same code as production.
    """

    def _make_handler(self, backend, *, session_pinned=False, session_subscription=False,
                      selector=None):
        """Build a handler wired to a single-snapshot registry with a mock stats_collector."""
        registry = MagicMock()
        registry.session_context.return_value = (0, 1.0)
        registry.snapshot.return_value = _fake_snapshot(
            'anthropic', backend,
            session_pinned=session_pinned,
            session_subscription=session_subscription,
        )
        handler = _handler_with_registry(registry)
        handler.selector = selector
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(
            return_value={'messages': [{'role': 'user', 'content': 'hello'}], 'model': 'sonnet'})
        handler.stats_collector = MagicMock()
        return handler

    def test_non_429_error_emits_failure_record(self):
        """Non-retrying AnthropicRequestError → failure record with correct fields."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = AnthropicRequestError(
            'bad request', error_type='invalid_request_error', status_code=400)
        handler = self._make_handler(backend)

        handler._handle_messages()

        handler.stats_collector.record.assert_called_once()
        _, kwargs = handler.stats_collector.record.call_args
        assert kwargs['status'] == 'error'
        assert kwargs['status_code'] == 400
        assert kwargs['error'] == 'invalid_request_error'
        # Error response was still sent to client
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 400

    def test_generic_exception_emits_502_failure_record(self):
        """Unhandled exception → failure record with status=error, status_code=502."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = RuntimeError('connection died')
        handler = self._make_handler(backend)

        handler._handle_messages()

        handler.stats_collector.record.assert_called_once()
        _, kwargs = handler.stats_collector.record.call_args
        assert kwargs['status'] == 'error'
        assert kwargs['status_code'] == 502
        assert kwargs['error'] == 'upstream_failure'
        # 502 was sent to client
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 502

    def test_429_with_retry_does_not_emit_failure_record_for_first_attempt(self):
        """429 that triggers auto-retry → NO failure record for the depleted attempt.

        The retry path calls _dispatch on the new backend; that call records its
        own outcome (success or failure).  The first 429 must not also emit an
        extra record to avoid double-counting.
        """
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = AnthropicRequestError(
            'rate limited', error_type='rate_limit_error', status_code=429)
        second_backend = MagicMock()
        second_backend.parse_credentials.return_value = {}
        second_backend.send_message.return_value = {'usage': {'input_tokens': 1, 'output_tokens': 1}}

        registry = MagicMock()
        registry.session_context.return_value = (0, 1.0)
        registry.snapshot.side_effect = [
            _fake_snapshot('anthropic', backend, session_pinned=False),
            _fake_snapshot('bedrock', second_backend, session_pinned=False),
        ]
        selector = MagicMock()
        selector.is_paused.return_value = False
        selector.on_rate_limited.return_value = 'bedrock'

        handler = object.__new__(ProxyRequestHandler)
        handler.registry = registry
        handler.selector = selector
        handler.config = MagicMock()
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        handler._ctx_key = None
        handler._route_est = 0
        handler.headers = {'x-api-key': 'secret'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(
            return_value={'messages': [{'role': 'user', 'content': 'hello'}], 'model': 'sonnet'})
        handler.stats_collector = MagicMock()

        handler._handle_messages()

        # Only one record: the successful retry (non-stream → status='success').
        # If a failure record were emitted for the 429, call_count would be 2.
        assert handler.stats_collector.record.call_count == 1
        _, kwargs = handler.stats_collector.record.call_args
        assert kwargs['status'] == 'success'

    def test_telemetry_exception_does_not_mask_original_error(self):
        """If stats_collector.record() raises, the real error response is still sent."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = AnthropicRequestError(
            'server error', error_type='api_error', status_code=500)
        handler = self._make_handler(backend)
        handler.stats_collector.record.side_effect = Exception('disk full')

        # Must not raise — the telemetry failure is swallowed
        handler._handle_messages()

        # Real error response still sent
        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args.args
        assert status == 500


# ---------------------------------------------------------------------------
# Per-request no-classifier override (X-Anthproxy-Override: no-classifier)
# ---------------------------------------------------------------------------

class TestNoClassifierOverride:
    """Verify that ``no-classifier`` bypasses route_model() entirely — no
    classifier call, no size floor, no session-tier cache read/write."""

    def _make_handler(self, *, routing_on=True):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.return_value = {'type': 'message', 'content': []}
        backend.send_classifier_message = MagicMock(
            return_value={'content': [{'type': 'text', 'text': 'standard'}]}
        )
        snapshot = _fake_snapshot('codex', backend)
        snapshot.config.auto_model_routing = routing_on

        registry = MagicMock()
        registry.snapshot.return_value = snapshot
        registry.session_context.return_value = (0, 1.0)
        registry.session_routed_tier.return_value = None
        registry.set_session_routed_tier = MagicMock()
        registry.record_session_context = MagicMock()

        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'key'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        return handler, registry, backend, snapshot

    def test_no_classifier_skips_classifier_call(self):
        handler, registry, backend, _ = self._make_handler()
        handler._no_classifier = True
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'plan a refactor'}],
            'metadata': {'user_id': 'test-sess-1'},
        })

        handler._handle_messages()

        # The classifier must not be called.
        backend.send_classifier_message.assert_not_called()
        # The model is unchanged — no tier rewriting.
        assert handler._send_json.call_count == 1
        # routing reason_code is override_no_classifier
        assert handler._routing.reason_code == 'override_no_classifier'
        assert handler._routing.applied is False

    def test_no_classifier_bypasses_size_floor(self):
        """Even with a large threshold + large payload, model is not forced to opus[1m]."""
        handler, registry, backend, snapshot = self._make_handler()
        # Force the size floor to fire under normal routing.
        snapshot.config.auto_model_routing_long_context_threshold = 1
        handler._no_classifier = True
        # Large payload that would normally trigger the floor.
        big_text = 'x' * 100_000
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': big_text}],
            'metadata': {'user_id': 'test-sess-1'},
        })

        handler._handle_messages()

        # No opus[1m] rewrite, no context-1m beta injection.
        assert handler._routing.reason_code == 'override_no_classifier'
        # The dispatched payload's model should remain 'sonnet'.
        dispatch_args = backend.send_message.call_args
        assert dispatch_args.args[0]['model'] == 'sonnet'

    def test_no_classifier_no_session_cache_write(self):
        handler, registry, _, _ = self._make_handler()
        handler._no_classifier = True
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'plan a refactor'}],
            'metadata': {'user_id': 'test-sess-1'},
        })

        handler._handle_messages()

        # classification is None → no cache write.
        registry.set_session_routed_tier.assert_not_called()

    def test_no_classifier_route_est_populated(self):
        """estimated_input_tokens is populated (defensive consistency with route_model)."""
        from anthproxy.mapper import estimate_input_tokens
        handler, _, _, _ = self._make_handler()
        handler._no_classifier = True
        payload = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'plan a refactor'}],
            'metadata': {'user_id': 'test-sess-1'},
        }
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        expected = estimate_input_tokens(payload)
        assert handler._routing.estimated_input_tokens == expected

    def test_no_classifier_no_session_context_record(self):
        """_record_session_context is not called when no-classifier is active."""
        handler, registry, _, _ = self._make_handler()
        handler._no_classifier = True
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'plan a refactor'}],
            'metadata': {'user_id': 'test-sess-1'},
        })

        handler._handle_messages()

        # Even though routing is on, the floor isn't active for no-classifier
        # requests, so no context recording.
        registry.record_session_context.assert_not_called()


class TestOverrideHeaderInDoPost:
    """Verify that do_POST parses X-Anthproxy-Override and resets per-request
    state so HTTP keep-alive connection reuse does not leak overrides."""

    def _make_handler(self):
        handler = object.__new__(ProxyRequestHandler)
        handler.registry = MagicMock()
        handler.config = MagicMock()
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        handler._handle_messages = MagicMock()
        handler._handle_count_tokens = MagicMock()
        handler._handle_cache = MagicMock()
        handler.path = '/v1/messages'
        handler.headers = {}
        return handler

    def test_no_classifier_header_sets_attribute(self):
        handler = self._make_handler()
        handler.headers = {'X-Anthproxy-Override': 'no-classifier'}
        handler.do_POST()
        assert handler._no_classifier is True

    def test_prefer_header_sets_attribute(self):
        handler = self._make_handler()
        handler.headers = {'X-Anthproxy-Override': 'prefer:codex'}
        handler.do_POST()
        assert handler._prefer_backend == 'codex'

    def test_no_header_resets_to_defaults(self):
        """A request without the header on a reused connection does not inherit
        the previous request's overrides."""
        handler = self._make_handler()
        # Simulate a previous request having set these.
        handler._no_classifier = True
        handler._prefer_backend = 'codex'
        handler.headers = {}
        handler.do_POST()
        assert handler._no_classifier is False
        assert handler._prefer_backend is None

    def test_count_tokens_receives_prefer_backend(self):
        """count_tokens inherits self._prefer_backend from do_POST."""
        handler = self._make_handler()
        handler.path = '/v1/messages/count_tokens'
        handler.headers = {'X-Anthproxy-Override': 'prefer:codex'}
        handler.do_POST()
        # count_tokens handler was called.
        handler._handle_count_tokens.assert_called_once()
        # The prefer_backend attribute was set before dispatch.
        assert handler._prefer_backend == 'codex'


# ---------------------------------------------------------------------------
# Admin POST — malformed JSON body returns 400 not 500
# ---------------------------------------------------------------------------

class TestAdminPostMalformedJson:
    """Regression test: a malformed JSON body in an admin POST must return 400,
    not 500 'Admin handler failed'."""

    def _make_handler(self, path='/admin/export'):
        handler = object.__new__(ProxyRequestHandler)
        handler.registry = MagicMock()
        handler.config = MagicMock()
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        handler._read_body = MagicMock(return_value=b'{not valid json}')
        handler.enable_ui = True
        handler.path = path
        handler.headers = {}
        handler.session_db = None
        return handler

    def test_malformed_json_returns_400(self):
        """POST /admin/export with invalid JSON body → HTTP 400, not 500."""
        handler = self._make_handler('/admin/export')
        handler.do_POST()

        handler._send_json.assert_called_once()
        status, payload = handler._send_json.call_args[0]
        assert status == 400
        assert payload.get('error', {}).get('type') == 'invalid_request_error'

    def test_malformed_json_does_not_return_500(self):
        """The 500 'Admin handler failed' response must not be sent for bad JSON."""
        handler = self._make_handler('/admin/sessions/sess1/tier')
        handler.do_POST()

        handler._send_json.assert_called_once()
        status, _ = handler._send_json.call_args[0]
        assert status != 500

    def test_valid_json_still_calls_admin_handle_post(self, monkeypatch):
        """A valid JSON body still reaches admin.handle_post (no regression)."""
        from anthproxy import admin as admin_mod
        mock_post = MagicMock(return_value=(200, {'ok': True}))
        monkeypatch.setattr(admin_mod, 'handle_post', mock_post)

        handler = self._make_handler('/admin/export')
        handler._read_body = MagicMock(return_value=b'{"key": "value"}')
        handler.do_POST()

        mock_post.assert_called_once()
        status, _ = handler._send_json.call_args[0]
        assert status == 200


# ---------------------------------------------------------------------------
# Phase 0 — Handler-level routing ordering invariants
# ---------------------------------------------------------------------------

class TestHandlerRoutingOrderInvariants:
    """Handler-level assertions for Phase 0 invariants 2 and 6.

    Invariant 2: no-classifier suppresses ctx_key (no context floor recording)
    Invariant 6: response model echo rewrites client-facing model iff applied=True
    """

    def _make_handler(self, *, routing_on=True):
        backend = MagicMock()
        backend.parse_credentials.return_value = {'token': 'ok'}
        backend.send_message.return_value = {'type': 'message', 'content': [], 'model': 'haiku'}
        backend.send_classifier_message = MagicMock(
            return_value={'content': [{'type': 'text', 'text': 'trivial'}]}
        )
        snapshot = _fake_snapshot('anthropic', backend)
        snapshot.config.auto_model_routing = routing_on
        snapshot.config.auto_model_routing_affirmation_inherit = True

        registry = MagicMock()
        registry.snapshot.return_value = snapshot
        registry.session_context.return_value = (0, 1.0)
        registry.session_routed_tier.return_value = None
        registry.set_session_routed_tier = MagicMock()
        registry.record_session_context = MagicMock()

        handler = _handler_with_registry(registry)
        handler.headers = {'x-api-key': 'key'}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        return handler, registry, backend, snapshot

    # ------------------------------------------------------------------
    # Invariant 2: no-classifier suppresses context-floor bookkeeping
    # ------------------------------------------------------------------

    def test_no_classifier_ctx_key_not_set(self):
        """When _no_classifier=True, _ctx_key is never set, so context floor is off
        and _record_session_context is a no-op for this request."""
        handler, registry, _, snapshot = self._make_handler()
        snapshot.config.auto_model_routing_long_context_threshold = 1  # would normally fire
        handler._no_classifier = True
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'plan a refactor'}],
            'metadata': {'user_id': 'test-sess-inv2'},
        })

        handler._handle_messages()

        # The context-floor gate must not have been armed.
        assert handler._ctx_key is None
        # No context recording should have happened.
        registry.record_session_context.assert_not_called()

    def test_no_classifier_no_tier_cache_write(self):
        """no-classifier: classification is None → set_session_routed_tier not called."""
        handler, registry, _, _ = self._make_handler()
        handler._no_classifier = True
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'plan a refactor'}],
            'metadata': {'user_id': 'test-sess-inv2b'},
        })

        handler._handle_messages()

        assert handler._routing.reason_code == 'override_no_classifier'
        assert handler._routing.applied is False
        registry.set_session_routed_tier.assert_not_called()

    # ------------------------------------------------------------------
    # Invariant 6: response model echo — only when applied=True
    # ------------------------------------------------------------------

    def test_response_model_echo_when_applied(self):
        """Non-streaming: result['model'] is rewritten to requested_model when applied=True."""
        handler, _, backend, snapshot = self._make_handler()
        snapshot.config.auto_model_routing = True
        # Classifier returns 'trivial' so routing goes sonnet → haiku (applied=True).
        # Backend returns a response with model='haiku' (the routed model).
        backend.send_message.return_value = {
            'type': 'message',
            'content': [],
            'model': 'haiku',
        }
        # send_classifier_message is already mocked to return 'trivial'.
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'metadata': {'user_id': 'test-sess-echo1'},
        })

        handler._handle_messages()

        # routing.applied must be True (sonnet → haiku).
        assert handler._routing.applied is True
        assert handler._routing.requested_model == 'sonnet'
        assert handler._routing.routed_model == 'haiku'
        # The client-facing _send_json must have been called with the requested model.
        assert handler._send_json.call_count == 1
        _status, result = handler._send_json.call_args.args
        assert result['model'] == 'sonnet'  # echoed requested model, not 'haiku'

    def test_response_model_echo_noop_when_not_applied(self):
        """Non-streaming: result['model'] is NOT changed when applied=False."""
        handler, _, backend, snapshot = self._make_handler()
        snapshot.config.auto_model_routing = True
        # Make classifier return 'standard' so sonnet → sonnet (applied=False).
        backend.send_classifier_message.return_value = {
            'content': [{'type': 'text', 'text': 'standard'}]
        }
        backend.send_message.return_value = {
            'type': 'message',
            'content': [],
            'model': 'sonnet',
        }
        handler._parse_json = MagicMock(return_value={
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'normal task'}],
            'metadata': {'user_id': 'test-sess-echo2'},
        })

        handler._handle_messages()

        assert handler._routing.applied is False
        _status, result = handler._send_json.call_args.args
        # model in the response is unchanged (no rewrite).
        assert result['model'] == 'sonnet'

    def test_routing_fields_returns_original_requested_model(self):
        """_routing_fields['requested_model'] is always the original client model,
        even after routing rewrote payload['model'] to the routed tier.
        Stats receive the original requested model separately from the serving model."""
        handler, _, backend, snapshot = self._make_handler()
        snapshot.config.auto_model_routing = True
        # Classifier: trivial → haiku; requested was 'opus'.
        backend.send_classifier_message.return_value = {
            'content': [{'type': 'text', 'text': 'trivial'}]
        }
        backend.send_message.return_value = {
            'type': 'message', 'content': [], 'model': 'haiku',
        }
        handler._parse_json = MagicMock(return_value={
            'model': 'opus',
            'messages': [{'role': 'user', 'content': 'quick question'}],
            'metadata': {'user_id': 'test-sess-fields'},
        })

        handler._handle_messages()

        rf = handler._routing_fields()
        # The stats 'requested_model' field carries the original client model.
        assert rf['requested_model'] == 'opus'
        # But the routing decision's routed_model is 'haiku' (the serving tier).
        assert handler._routing.routed_model == 'haiku'
        assert handler._routing.requested_model == 'opus'
