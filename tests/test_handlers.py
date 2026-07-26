import json
from unittest.mock import MagicMock

import anthproxy.handlers as _handlers_module
from anthproxy.constants import HAPPY_BIRTHDAY_REPLY, HAPPY_NEW_YEAR_PREFIX
from anthproxy.handlers import (
    ProxyRequestHandler,
    _extract_sse_stats,
    _has_happy_new_year_system_prompt,
    _local_message,
    _local_message_sse,
    _parse_local_command,
    _parse_override_header,
    _parse_stats_selector,
    _qualify_first_heading,
    _rewrite_message_start_model,
)
from anthproxy.stats import StatsPeriod, resolve_stats_period


def _parse_sse(chunks):
    events = []
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        event = next(line[7:] for line in lines if line.startswith('event: '))
        data = json.loads(next(line[6:] for line in lines if line.startswith('data: ')))
        events.append((event, data))
    return events


def _msg(content):
    return {'messages': [{'role': 'user', 'content': content}]}


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

class TestParseLocalCommand:
    def test_get_usage_string(self):
        assert _parse_local_command(_msg('proxy-get-usage')) == ('get-usage', None)

    def test_session_wrap_status(self):
        assert _parse_local_command(_msg('<session> proxy-status </session>')) == ('status', None)

    def test_session_wrap_set_backend(self):
        assert _parse_local_command(_msg('<session>proxy-set-backend:auto</session>')) == ('set-backend', 'auto')

    def test_session_wrap_prose_outside_rejected(self):
        assert _parse_local_command(_msg('hi <session>proxy-status</session>')) is None

    def test_session_wrap_trailing_titlegen_prose(self):
        # openclaw appends title-generation boilerplate after </session>; the
        # trailing prose must not defeat command recognition (regression for the
        # <session> fullmatch rejecting any text after the closing tag).
        content = (
            '<session> proxy-status </session>  Write the title in the language '
            'the user wrote in, regardless of the language of the examples above.'
        )
        assert _parse_local_command(_msg(content)) == ('status', None)

    def test_session_wrap_trailing_prose_set_backend(self):
        content = '<session>proxy-set-backend:codex</session> please summarize'
        assert _parse_local_command(_msg(content)) == ('set-backend', 'codex')

    def test_session_wrap_trailing_system_reminder(self):
        # Claude Code appends a <system-reminder> after the session-wrapped
        # command; it must be stripped before unwrapping so the command is still
        # recognized.
        content = (
            '<session> proxy-status </session>\n'
            '<system-reminder>Write the title in the language the user wrote in, '
            'regardless of the language of the examples above.</system-reminder>'
        )
        assert _parse_local_command(_msg(content)) == ('status', None)

    def test_session_wrap_leading_system_reminder(self):
        content = (
            '<system-reminder>Some injected reminder.</system-reminder>\n'
            '<session> proxy-status </session>'
        )
        assert _parse_local_command(_msg(content)) == ('status', None)

    def test_session_wrap_unclosed_rejected(self):
        assert _parse_local_command(_msg('<session>proxy-status')) is None

    def test_get_usage_text_block(self):
        payload = _msg([{'type': 'text', 'text': '  proxy-get-usage\n'}])
        assert _parse_local_command(payload) == ('get-usage', None)

    def test_get_backend(self):
        assert _parse_local_command(_msg('proxy-get-backend')) == ('get-backend', None)

    def test_proxy_stats_default(self):
        name, period = _parse_local_command(_msg('proxy-stats'))
        assert name == 'stats'
        assert period == resolve_stats_period('1d')

    def test_proxy_stats_day_alias(self):
        name, period = _parse_local_command(_msg('proxy-stats:d'))
        assert name == 'stats'
        assert period == resolve_stats_period('d')

    def test_proxy_stats_week(self):
        name, period = _parse_local_command(_msg('proxy-stats:1w'))
        assert name == 'stats'
        assert period == resolve_stats_period('1w')

    def test_proxy_stats_historical_day(self):
        name, period = _parse_local_command(_msg('proxy-stats:-2d'))
        assert name == 'stats'
        assert period.token == '-2d'
        assert period.bucket == 'hour'

    def test_proxy_stats_month_alias(self):
        name, period = _parse_local_command(_msg('proxy-stats:m'))
        assert name == 'stats'
        assert period == resolve_stats_period('m')

    def test_proxy_stats_historical_month(self):
        name, period = _parse_local_command(_msg('proxy-stats:-12m'))
        assert name == 'stats'
        assert period.token == '-12m'
        assert period.bucket == 'week'

    def test_proxy_stats_invalid_period_defaults_to_day(self):
        name, period = _parse_local_command(_msg('proxy-stats:7d'))
        assert name == 'stats'
        assert period == resolve_stats_period('1d')

    def test_proxy_stats_invalid_forms_default_to_day(self):
        # '-1d:extra' is no longer "extra" — if 'extra' were a valid backend it
        # would be picked up as the filter.  Since 'extra' is not a recognized
        # backend name the fallback behaviour is: period_token='-1d:extra'
        # (re-joined leftover), which resolve_stats_period rejects → today.
        for token in ('2m', '+1d', '0d', '-0d', '-01d', '-9999999d', '-1D'):
            name, period = _parse_local_command(_msg(f'proxy-stats:{token}'))
            assert name == 'stats'
            assert period == resolve_stats_period('1d')

    def test_proxy_stats_week_alias(self):
        name, period = _parse_local_command(_msg('proxy-stats:w'))
        assert name == 'stats'
        assert period.bucket == 'day'
        assert period.label == 'this week'
        assert period.backend is None

    def test_proxy_stats_quarter_alias(self):
        name, period = _parse_local_command(_msg('proxy-stats:q'))
        assert name == 'stats'
        assert period.bucket == 'month'
        assert period.label == 'this quarter'
        assert period.backend is None

    def test_proxy_stats_historical_week(self):
        name, period = _parse_local_command(_msg('proxy-stats:-1w'))
        assert name == 'stats'
        assert period.token == '-1w'
        assert period.bucket == 'day'
        assert period.backend is None

    def test_proxy_stats_historical_quarter(self):
        name, period = _parse_local_command(_msg('proxy-stats:-2q'))
        assert name == 'stats'
        assert period.token == '-2q'
        assert period.bucket == 'month'
        assert period.backend is None

    def test_proxy_stats_backend_with_week_alias(self):
        name, period = _parse_local_command(_msg('proxy-stats:bedrock:w'))
        assert name == 'stats'
        assert period.backend == 'bedrock'
        assert period.bucket == 'day'

    def test_proxy_stats_backend_with_historical_quarter(self):
        name, period = _parse_local_command(_msg('proxy-stats:bedrock:-1q'))
        assert name == 'stats'
        assert period.backend == 'bedrock'
        assert period.bucket == 'month'

    # --- backend filter ---

    def test_proxy_stats_backend_suffix_period_first(self):
        name, period = _parse_local_command(_msg('proxy-stats:-1d:bedrock'))
        assert name == 'stats'
        assert period.backend == 'bedrock'
        assert period.bucket == 'hour'
        assert period.token == '-1d'

    def test_proxy_stats_backend_suffix_backend_first(self):
        name, period = _parse_local_command(_msg('proxy-stats:bedrock:-1d'))
        assert name == 'stats'
        assert period.backend == 'bedrock'
        assert period.bucket == 'hour'
        assert period.token == '-1d'

    def test_proxy_stats_backend_only_defaults_period_to_today(self):
        name, period = _parse_local_command(_msg('proxy-stats:bedrock'))
        assert name == 'stats'
        assert period.backend == 'bedrock'
        assert period.token == '1d'

    def test_proxy_stats_subscription_filter(self):
        name, period = _parse_local_command(_msg('proxy-stats:-1d:subscription'))
        assert name == 'stats'
        assert period.backend == 'subscription'
        assert period.bucket == 'hour'

    def test_proxy_stats_no_backend_filter_unchanged(self):
        name, period = _parse_local_command(_msg('proxy-stats:-1d'))
        assert name == 'stats'
        assert period.backend is None
        assert period.bucket == 'hour'

    def test_proxy_stats_unknown_filter_token_falls_back_to_today(self):
        # 'xyz' is not a backend name; '-1d:xyz' becomes the period token (both
        # parts are leftover), which is invalid and falls back to today.
        name, period = _parse_local_command(_msg('proxy-stats:-1d:xyz'))
        assert name == 'stats'
        assert period.backend is None
        assert period == resolve_stats_period('1d')

    def test_proxy_stats_all_backends_accepted_as_filter(self):
        for backend in ('bedrock', 'codex', 'anthropic', 'local', 'openrouter'):
            name, period = _parse_local_command(_msg(f'proxy-stats:{backend}'))
            assert name == 'stats'
            assert period.backend == backend, f'expected backend={backend}'

    def test_set_backend_valid(self):
        for name in ('bedrock', 'codex', 'anthropic', 'local'):
            assert _parse_local_command(_msg(f'proxy-set-backend:{name}')) == ('set-backend', name)

    def test_set_backend_whitespace_trimmed_outer(self):
        assert _parse_local_command(_msg('  proxy-set-backend:codex ')) == ('set-backend', 'codex')

    def test_set_backend_unknown_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:foo')) == ('set-backend', None)

    def test_set_backend_empty_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:')) == ('set-backend', None)

    def test_set_backend_uppercase_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:Codex')) == ('set-backend', None)

    def test_set_backend_space_after_colon_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend: codex')) == ('set-backend', None)

    def test_set_backend_extra_suffix_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:codex:extra')) == ('set-backend', None)

    def test_set_backend_session_suffix_valid(self):
        assert _parse_local_command(_msg('proxy-set-backend:codex:session')) == ('session-set-backend', 'codex')

    def test_set_backend_auto_session(self):
        assert _parse_local_command(_msg('proxy-set-backend:auto:session')) == ('session-set-backend', 'auto')

    def test_set_backend_subscription(self):
        assert _parse_local_command(_msg('proxy-set-backend:subscription')) == ('set-backend', 'subscription')

    def test_set_backend_subscription_session(self):
        assert _parse_local_command(_msg('proxy-set-backend:subscription:session')) == ('session-set-backend', 'subscription')

    def test_set_backend_unknown_session_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:nope:session')) == ('session-set-backend', None)

    def test_set_backend_bare_session_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:session')) == ('set-backend', None)

    def test_set_backend_empty_session_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend::session')) == ('session-set-backend', None)

    def test_set_backend_subscription_extra_suffix_is_malformed(self):
        assert _parse_local_command(_msg('proxy-set-backend:subscription:extra')) == ('set-backend', None)

    def test_prose_near_match_not_command(self):
        assert _parse_local_command(_msg('please proxy-set-backend:codex')) is None

    def test_get_backend_trailing_text_not_command(self):
        assert _parse_local_command(_msg('proxy-get-backend now')) is None

    def test_proxy_statsx_not_command(self):
        assert _parse_local_command(_msg('proxy-statsx')) is None

    def test_command_in_history_ignored(self):
        payload = {'messages': [
            {'role': 'user', 'content': 'proxy-get-backend'},
            {'role': 'assistant', 'content': 'ok'},
        ]}
        assert _parse_local_command(payload) is None

    def test_mixed_blocks_rejected(self):
        payload = _msg([
            {'type': 'text', 'text': 'proxy-get-usage'},
            {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'done'},
        ])
        assert _parse_local_command(payload) is None

    def test_final_assistant_message_rejected(self):
        payload = {'messages': [{'role': 'assistant', 'content': 'proxy-get-backend'}]}
        assert _parse_local_command(payload) is None

    def test_trailing_newline_status(self):
        assert _parse_local_command(_msg('some prose\nproxy-status')) == ('status', None)

    def test_trailing_newline_get_usage(self):
        assert _parse_local_command(_msg('some prose\nproxy-get-usage')) == ('get-usage', None)

    def test_trailing_newline_set_backend(self):
        assert _parse_local_command(_msg('some prose\nproxy-set-backend:openrouter')) == ('set-backend', 'openrouter')

    def test_trailing_newline_set_model_routing(self):
        assert _parse_local_command(_msg('some prose\nproxy-set-model-routing:on')) == ('set-model-routing', True)

    def test_trailing_newline_stats(self):
        name, period = _parse_local_command(_msg('some prose\nproxy-stats:1w'))
        assert name == 'stats'
        assert period == resolve_stats_period('1w')

    def test_command_not_at_end_is_rejected(self):
        assert _parse_local_command(_msg('proxy-status\nsome trailing text')) is None

    # --- transcript stripping ---

    def test_transcript_before_command_is_stripped(self):
        # Claude Code may embed a <transcript> block in the same message as
        # a trailing proxy-* command; the block must be stripped so the
        # command on the last line is still detected.
        content = (
            '<transcript>User: ls\nAssistant: file.py</transcript>\n'
            'proxy-status'
        )
        assert _parse_local_command(_msg(content)) == ('status', None)

    def test_transcript_only_not_mistaken_for_command(self):
        # A message that is only a transcript block (no proxy-* command on the
        # last segment after stripping) must not be detected as a command.
        content = (
            '<transcript>User: proxy-status\nAssistant: ok</transcript>'
        )
        assert _parse_local_command(_msg(content)) is None

    def test_unclosed_transcript_after_command_is_stripped(self):
        # Unclosed transcript appears after the command; stripping it leaves
        # the command on the last segment → still detected.
        content = 'proxy-status\n<transcript>User: earlier turn'
        assert _parse_local_command(_msg(content)) == ('status', None)

    def test_transcript_command_on_last_segment(self):
        # Command follows the transcript block on the last \n-segment.
        content = (
            'Some preamble\n'
            '<transcript>User: hi\nAssistant: hello</transcript>\n'
            'proxy-get-usage'
        )
        assert _parse_local_command(_msg(content)) == ('get-usage', None)


# ---------------------------------------------------------------------------
# _parse_stats_selector unit tests
# ---------------------------------------------------------------------------

class TestParseStatsSelector:
    def test_empty_raw(self):
        assert _parse_stats_selector('') == ('', None)

    def test_period_only(self):
        assert _parse_stats_selector('-1d') == ('-1d', None)

    def test_backend_only(self):
        assert _parse_stats_selector('bedrock') == ('', 'bedrock')

    def test_period_then_backend(self):
        assert _parse_stats_selector('-1d:bedrock') == ('-1d', 'bedrock')

    def test_backend_then_period(self):
        assert _parse_stats_selector('bedrock:-1d') == ('-1d', 'bedrock')

    def test_subscription_keyword(self):
        assert _parse_stats_selector('-1d:subscription') == ('-1d', 'subscription')

    def test_unknown_part_treated_as_period(self):
        # 'xyz' is not a backend → both parts end up as leftover → re-joined
        period_token, backend = _parse_stats_selector('-1d:xyz')
        assert backend is None
        assert period_token == '-1d:xyz'

    def test_week_period_no_backend(self):
        assert _parse_stats_selector('1w') == ('1w', None)

    def test_subscription_backend_first(self):
        assert _parse_stats_selector('subscription:1w') == ('1w', 'subscription')

    def test_all_backend_names(self):
        for name in ('bedrock', 'openrouter', 'codex', 'anthropic', 'local'):
            period_token, backend = _parse_stats_selector(name)
            assert backend == name, f'expected backend={name}'
            assert period_token == '', f'expected empty period for {name}'


# ---------------------------------------------------------------------------
# Local response shape
# ---------------------------------------------------------------------------

def test_local_message_is_anthropic_shape():
    result = _local_message('## Usage', 'sonnet')
    assert result['type'] == 'message'
    assert result['content'] == [{'type': 'text', 'text': '## Usage'}]
    assert result['stop_reason'] == 'end_turn'
    assert result['usage'] == {'input_tokens': 0, 'output_tokens': 0}


def test_local_message_sse_has_complete_lifecycle():
    events = _parse_sse(_local_message_sse('## Usage', 'sonnet'))
    assert [event for event, _ in events] == [
        'message_start',
        'content_block_start',
        'content_block_delta',
        'content_block_stop',
        'message_delta',
        'message_stop',
    ]
    assert events[2][1]['delta']['text'] == '## Usage'
    assert events[4][1]['delta']['stop_reason'] == 'end_turn'


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

def _handler_with_registry(registry):
    handler = object.__new__(ProxyRequestHandler)
    handler.registry = registry
    handler.selector = None
    handler.config = MagicMock()
    handler._send_json = MagicMock()
    handler._send_sse = MagicMock()
    return handler


def _fake_snapshot(name, backend, session_pinned=False, session_subscription=False):
    snapshot = MagicMock()
    snapshot.name = name
    snapshot.backend = backend
    snapshot.config = MagicMock()
    snapshot.config.lock_requested_model = 'off'
    snapshot.session_pinned = session_pinned
    snapshot.session_subscription = session_subscription
    return snapshot


def test_get_backend_command_reports_active():
    registry = MagicMock()
    registry.active_name.return_value = 'openrouter'
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('get-backend', None), {'model': 'sonnet'})

    response = handler._send_json.call_args.args[1]
    assert '`openrouter`' in response['content'][0]['text']


def test_set_backend_changed_message():
    registry = MagicMock()
    registry.switch.return_value = MagicMock(kind='changed', previous='bedrock', current='codex', error=None)
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('set-backend', 'codex'), {})

    registry.switch.assert_called_once_with('codex', reason='manual command')
    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert 'Switched from `bedrock` to `codex`' in text


def test_set_backend_malformed_does_not_switch():
    registry = MagicMock()
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('set-backend', None), {})

    registry.switch.assert_not_called()
    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert 'invalid command' in text


def test_set_backend_failed_keeps_backend():
    registry = MagicMock()
    registry.switch.return_value = MagicMock(
        kind='failed', previous='bedrock', current='bedrock', error='No Codex credentials found.')
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('set-backend', 'codex'), {})

    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert 'remains `bedrock`' in text
    assert 'No Codex credentials' in text


def test_usage_command_only_active_codex():
    backend = MagicMock()
    backend.get_usage_markdown.return_value = '## Codex subscription usage'
    registry = MagicMock()
    registry.snapshot.return_value = _fake_snapshot('codex', backend)
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('get-usage', None), {})

    backend.get_usage_markdown.assert_called_once()
    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert text == '## Codex subscription usage'


def test_usage_command_non_subscription_explains():
    backend = MagicMock(spec=['parse_credentials', 'send_message', 'send_message_stream', 'count_tokens'])
    registry = MagicMock()
    registry.snapshot.return_value = _fake_snapshot('bedrock', backend)
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('get-usage', None), {})

    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert 'subscription' in text.lower()


def test_qualify_first_heading_updates_only_first_heading():
    markdown = '## Codex subscription usage\n\nUsage information is unavailable: offline\n## Details\nstill unavailable'

    qualified = _qualify_first_heading(markdown, 'not the active backend')

    assert qualified == (
        '## Codex subscription usage · not the active backend\n\n'
        'Usage information is unavailable: offline\n'
        '## Details\n'
        'still unavailable'
    )


def test_qualify_first_heading_without_heading_is_unchanged():
    markdown = 'Usage information is unavailable: offline'

    assert _qualify_first_heading(markdown, 'not the active backend') == markdown


def test_status_command_qualifies_inactive_codex_heading_only():
    anthropic = MagicMock()
    anthropic.get_usage_markdown.return_value = '## Anthropic subscription usage\n\n**5-hour usage:** 21% used'
    codex = MagicMock()
    codex.get_usage_markdown.return_value = '## Codex subscription usage\n\n**Plan:** plus'
    registry = MagicMock()
    registry.active_name.return_value = 'anthropic'
    registry.instance.side_effect = lambda name: {
        'anthropic': anthropic,
        'codex': codex,
    }[name]
    handler = _handler_with_registry(registry)

    text = handler._status_markdown()

    assert '## Anthropic subscription usage\n\n**5-hour usage:** 21% used' in text
    assert '## Codex subscription usage · *not the active backend*\n\n**Plan:** plus' in text
    assert '_`codex` — not the active backend_' not in text


def test_status_command_qualifies_inactive_anthropic_heading_only():
    anthropic = MagicMock()
    anthropic.get_usage_markdown.return_value = '## Anthropic subscription usage\n\n**5-hour usage:** 21% used'
    codex = MagicMock()
    codex.get_usage_markdown.return_value = '## Codex subscription usage\n\n**Plan:** plus'
    registry = MagicMock()
    registry.active_name.return_value = 'codex'
    registry.instance.side_effect = lambda name: {
        'anthropic': anthropic,
        'codex': codex,
    }[name]
    handler = _handler_with_registry(registry)

    text = handler._status_markdown()

    assert '## Anthropic subscription usage · *not the active backend*\n\n**5-hour usage:** 21% used' in text
    assert '## Codex subscription usage\n\n**Plan:** plus' in text
    assert '_`anthropic` — not the active backend_' not in text


def test_status_command_qualifies_both_subscription_headings_when_global_backend_is_non_subscription():
    anthropic = MagicMock()
    anthropic.get_usage_markdown.return_value = '## Anthropic subscription usage\n\n**5-hour usage:** 21% used'
    codex = MagicMock()
    codex.get_usage_markdown.return_value = '## Codex subscription usage\n\n**Plan:** plus'
    registry = MagicMock()
    registry.active_name.return_value = 'bedrock'
    registry.instance.side_effect = lambda name: {
        'anthropic': anthropic,
        'codex': codex,
    }[name]
    handler = _handler_with_registry(registry)

    text = handler._status_markdown()

    assert '## Anthropic subscription usage · *not the active backend*' in text
    assert '## Codex subscription usage · *not the active backend*' in text
    assert '_`anthropic` — not the active backend_' not in text
    assert '_`codex` — not the active backend_' not in text


def test_status_command_qualifies_failure_heading_without_changing_body():
    anthropic = MagicMock()
    anthropic.get_usage_markdown.return_value = '## Anthropic subscription usage\n\n**5-hour usage:** 21% used'
    codex_failure = (
        '## Codex subscription usage\n\n'
        'Usage information is unavailable: offline\n'
        '## Details\n'
        'still unavailable'
    )
    codex = MagicMock()
    codex.get_usage_markdown.return_value = codex_failure
    registry = MagicMock()
    registry.active_name.return_value = 'anthropic'
    registry.instance.side_effect = lambda name: {
        'anthropic': anthropic,
        'codex': codex,
    }[name]
    handler = _handler_with_registry(registry)

    text = handler._status_markdown()

    assert (
        '## Codex subscription usage · *not the active backend*\n\n'
        'Usage information is unavailable: offline\n'
        '## Details\n'
        'still unavailable'
    ) in text


def test_stats_command_reads_collector_with_range():
    registry = MagicMock()
    registry.active_name.return_value = 'codex'
    handler = _handler_with_registry(registry)
    handler.stats_collector = MagicMock()
    period = StatsPeriod('1w', 100.0, 200.0, 'this week', 'day')
    handler.stats_collector.read_records.return_value = [{'backend': 'codex', 'ts': 150.0}]

    handler._handle_local_command(('stats', period), {})

    handler.stats_collector.read_records.assert_called_once_with(100.0, 200.0)
    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert 'anthproxy stats' in text
    assert 'this week' in text


def test_stats_command_without_collector_explains():
    registry = MagicMock()
    registry.active_name.return_value = 'codex'
    handler = _handler_with_registry(registry)
    handler.stats_collector = None

    handler._handle_local_command(('stats', resolve_stats_period('1d')), {})

    text = handler._send_json.call_args.args[1]['content'][0]['text']
    assert 'not enabled' in text.lower()


def test_extract_sse_stats_accumulates_usage():
    stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_creation_tokens': 0,
        'cache_read_tokens': 0,
    }
    chunk = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":10,'
        '"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
    )

    _extract_sse_stats(chunk, stats)

    assert stats == {
        'input_tokens': 10,
        'output_tokens': 7,
        'cache_creation_tokens': 3,
        'cache_read_tokens': 4,
    }


def test_extract_sse_stats_does_not_double_count_cumulative_usage():
    # Regression: Anthropic's final message_delta re-states the cumulative usage
    # snapshot — for server-tool turns it re-echoes input/cache_* with a LARGER
    # total than message_start (the server-side loop consumed more context).
    # Each field must be tracked as max, not summed, or the cached prefix is
    # double-counted (a ~110K context measured ~218K → spurious opus[1m]).
    stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_creation_tokens': 0,
        'cache_read_tokens': 0,
    }
    chunk = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":2679,'
        '"cache_creation_input_tokens":4000,"cache_read_input_tokens":105000,'
        '"output_tokens":2}}}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"input_tokens":10682,'
        '"cache_creation_input_tokens":4000,"cache_read_input_tokens":105000,'
        '"output_tokens":510}}\n\n'
    )
    _extract_sse_stats(chunk, stats)
    # max per field — NOT message_start + message_delta (which would yield
    # input=13361, cache_creation=8000, cache_read=210000).
    assert stats == {
        'input_tokens': 10682,
        'output_tokens': 510,
        'cache_creation_tokens': 4000,
        'cache_read_tokens': 105000,
    }


def test_extract_sse_stats_cache_on_message_delta():
    # Codex emits real usage only at stream end, so cache_read lands on
    # message_delta rather than message_start.
    stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_creation_tokens': 0,
        'cache_read_tokens': 0,
    }
    chunk = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":100}}}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"output_tokens":20,'
        '"cache_read_input_tokens":500}}\n\n'
    )
    _extract_sse_stats(chunk, stats)
    assert stats == {
        'input_tokens': 100,
        'output_tokens': 20,
        'cache_creation_tokens': 0,
        'cache_read_tokens': 500,
    }


def test_stats_sse_wrapper_records_codex_cache_read():
    # Full streaming path: cache_read arrives on the final message_delta chunk.
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.stats_collector = MagicMock()
    first_chunk = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":100}}}\n\n'
    )
    second_chunk = (
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"output_tokens":20,'
        '"cache_read_input_tokens":500}}\n\n'
    )
    handler.session_db = None
    list(handler._usage_sse_wrapper(iter([second_chunk]), first_chunk, 'codex', 0.0, 'sonnet'))
    args, kwargs = handler.stats_collector.record.call_args
    assert args == ('codex',)
    assert kwargs['input_tokens'] == 100
    assert kwargs['output_tokens'] == 20
    assert kwargs['cache_creation_tokens'] == 0
    assert kwargs['cache_read_tokens'] == 500
    assert kwargs['status'] == 'success'
    assert kwargs['status_code'] == 200


def test_dispatch_records_nonstream_usage():
    backend = MagicMock()
    backend.parse_credentials.return_value = {'token': 'ok'}
    backend.send_message.return_value = {
        'usage': {
            'input_tokens': 11,
            'output_tokens': 22,
            'cache_creation_input_tokens': 3,
            'cache_read_input_tokens': 4,
        }
    }
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = MagicMock()
    snapshot = _fake_snapshot('codex', backend)

    handler._dispatch({'model': 'sonnet'}, snapshot, 1)

    handler.stats_collector.record.assert_called_once()
    args, kwargs = handler.stats_collector.record.call_args
    assert args == ('codex',)
    assert kwargs['input_tokens'] == 11
    assert kwargs['output_tokens'] == 22
    assert kwargs['cache_creation_tokens'] == 3
    assert kwargs['cache_read_tokens'] == 4
    assert kwargs['streaming'] is False
    assert kwargs['model'] == 'sonnet'
    # Enriched observability fields
    assert kwargs['status'] == 'success'
    assert kwargs['status_code'] == 200


def test_dispatch_records_nonstream_with_routing_metadata():
    """Non-stream record includes routing metadata when _routing is set on handler."""
    from anthproxy.model_router import ModelRoutingDecision

    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    backend.send_message.return_value = {'usage': {'input_tokens': 5, 'output_tokens': 2}}
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = MagicMock()
    handler._routing = ModelRoutingDecision(
        requested_model='opus',
        routed_model='sonnet',
        classification='standard',
        applied=True,
        reason_code='classifier_standard',
    )
    snapshot = _fake_snapshot('anthropic', backend)

    handler._dispatch({'model': 'sonnet'}, snapshot, 1)

    _, kwargs = handler.stats_collector.record.call_args
    assert kwargs['requested_model'] == 'opus'
    assert kwargs['classification'] == 'standard'
    assert kwargs['reason_code'] == 'classifier_standard'
    assert kwargs['status'] == 'success'
    assert kwargs['status_code'] == 200


def test_stats_sse_wrapper_records_stream_usage():
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.stats_collector = MagicMock()
    first_chunk = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{"input_tokens":5,'
        '"cache_creation_input_tokens":1,"cache_read_input_tokens":2}}}\n\n'
    )
    second_chunk = (
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
    )

    handler.session_db = None
    chunks = list(handler._usage_sse_wrapper(iter([second_chunk]), first_chunk, 'anthropic', 0.0, 'opus'))

    assert chunks == [first_chunk, second_chunk]
    handler.stats_collector.record.assert_called_once()
    args, kwargs = handler.stats_collector.record.call_args
    assert args == ('anthropic',)
    assert kwargs['input_tokens'] == 5
    assert kwargs['output_tokens'] == 9
    assert kwargs['cache_creation_tokens'] == 1
    assert kwargs['cache_read_tokens'] == 2
    assert kwargs['streaming'] is True
    assert kwargs['model'] == 'opus'
    # Enriched observability fields on success
    assert kwargs['status'] == 'success'
    assert kwargs['status_code'] == 200


def test_stats_sse_wrapper_records_error_on_sse_error_event():
    """An in-band ``event: error`` chunk causes status='error', error='sse_error'."""
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.stats_collector = MagicMock()
    # Simulate a pre-stream 429 delivered as in-band SSE error (post-HTTP-200)
    error_chunk = (
        'event: error\n'
        'data: {"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}\n\n'
    )

    handler.session_db = None
    chunks = list(handler._usage_sse_wrapper(iter([error_chunk]), None, 'anthropic', 0.0, 'opus'))

    assert chunks == [error_chunk]
    _, kwargs = handler.stats_collector.record.call_args
    assert kwargs['status'] == 'error'
    assert kwargs['error'] == 'sse_error'
    assert kwargs['status_code'] is None
    assert kwargs['streaming'] is True


def test_stats_sse_wrapper_records_error_in_first_chunk():
    """``event: error`` in the first (primed) chunk is also detected."""
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.stats_collector = MagicMock()
    error_chunk = 'event: error\ndata: {"type":"error","error":{"type":"api_error"}}\n\n'

    handler.session_db = None
    list(handler._usage_sse_wrapper(iter([]), error_chunk, 'anthropic', 0.0, 'opus'))

    _, kwargs = handler.stats_collector.record.call_args
    assert kwargs['status'] == 'error'
    assert kwargs['error'] == 'sse_error'


def test_command_streaming_uses_sse():
    registry = MagicMock()
    registry.active_name.return_value = 'codex'
    handler = _handler_with_registry(registry)

    handler._handle_local_command(('get-backend', None), {'stream': True})

    handler._send_sse.assert_called_once()
    handler._send_json.assert_not_called()


def test_cache_always_uses_bedrock_instance_without_switching_active_backend():
    backend = MagicMock()
    registry = MagicMock()
    registry.active_name.return_value = 'anthropic'
    registry.instance.return_value = backend
    handler = _handler_with_registry(registry)
    handler.headers = {}
    handler._validate_content_type = MagicMock()
    handler._read_body = MagicMock(return_value=b'{}')
    handler._parse_json = MagicMock(return_value={'key': 'alias', 'value': 'secret'})

    handler._handle_cache()

    registry.instance.assert_called_once_with('bedrock')
    registry.switch.assert_not_called()


# ---------------------------------------------------------------------------
# Requested-model echo: the client must see its originally-requested model in
# the response even when model-tier routing served the request on another tier.
# ---------------------------------------------------------------------------


def _routing(requested, routed, applied=True, reason_code='classifier_standard'):
    from anthproxy.model_router import ModelRoutingDecision
    return ModelRoutingDecision(
        requested_model=requested,
        routed_model=routed,
        classification='standard',
        applied=applied,
        reason_code=reason_code,
    )


def _msg_start(model, **extra):
    message = {'id': 'msg_1', 'type': 'message', 'role': 'assistant',
               'model': model, 'usage': {'input_tokens': 5, 'output_tokens': 0}}
    message.update(extra)
    return (
        'event: message_start\n'
        'data: ' + json.dumps({'type': 'message_start', 'message': message}) + '\n\n'
    )


def test_rewrite_message_start_model_rewrites_routed_model():
    chunk = _msg_start('claude-sonnet-4-6')
    out = _rewrite_message_start_model(chunk, 'claude-haiku-4-5-20251001')
    event = json.loads(out.split('data: ', 1)[1].strip())
    assert event['message']['model'] == 'claude-haiku-4-5-20251001'
    # Other fields preserved.
    assert event['message']['usage'] == {'input_tokens': 5, 'output_tokens': 0}


def test_rewrite_message_start_model_noop_when_already_requested():
    chunk = _msg_start('haiku')
    # Returns the SAME object so the dispatch latch does not fire prematurely.
    assert _rewrite_message_start_model(chunk, 'haiku') is chunk


def test_rewrite_message_start_model_ignores_non_message_start():
    chunk = (
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    )
    assert _rewrite_message_start_model(chunk, 'haiku') is chunk


def test_rewrite_message_start_model_ignores_message_delta():
    chunk = (
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
    )
    assert _rewrite_message_start_model(chunk, 'haiku') is chunk


def test_rewrite_message_start_model_passes_malformed_json():
    chunk = 'event: message_start\ndata: {not valid json}\n\n'
    assert _rewrite_message_start_model(chunk, 'haiku') is chunk


def test_dispatch_nonstream_echoes_requested_model_when_routed():
    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    backend.send_message.return_value = {
        'type': 'message', 'model': 'sonnet', 'content': [],
        'usage': {'input_tokens': 5, 'output_tokens': 2},
    }
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = None
    handler._ctx_key = None
    handler._route_est = 0
    handler._routing = _routing('haiku', 'sonnet', applied=True)
    snapshot = _fake_snapshot('anthropic', backend)

    # payload['model'] is the routed model (route_model mutated it in place).
    handler._dispatch({'model': 'sonnet'}, snapshot, 1)

    status, body = handler._send_json.call_args.args
    assert status == 200
    assert body['model'] == 'haiku'


def test_dispatch_nonstream_passthrough_model_when_not_routed():
    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    backend.send_message.return_value = {
        'type': 'message', 'model': 'sonnet', 'content': [],
        'usage': {'input_tokens': 5, 'output_tokens': 2},
    }
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = None
    handler._ctx_key = None
    handler._route_est = 0
    handler._routing = _routing('sonnet', 'sonnet', applied=False,
                                reason_code='classifier_standard')
    snapshot = _fake_snapshot('anthropic', backend)

    handler._dispatch({'model': 'sonnet'}, snapshot, 1)

    _, body = handler._send_json.call_args.args
    assert body['model'] == 'sonnet'


def test_dispatch_nonstream_records_routed_model_in_stats():
    """Stats keep the routed model even though the body echoes the requested one."""
    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    backend.send_message.return_value = {
        'type': 'message', 'model': 'sonnet',
        'usage': {'input_tokens': 5, 'output_tokens': 2},
    }
    registry = MagicMock()
    registry.session_context.return_value = (0, 1.0)
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = MagicMock()
    handler._ctx_key = None
    handler._route_est = 0
    handler._routing = _routing('haiku', 'sonnet', applied=True)
    snapshot = _fake_snapshot('anthropic', backend)

    handler._dispatch({'model': 'sonnet'}, snapshot, 1)

    _, kwargs = handler.stats_collector.record.call_args
    # stats 'model' is the routed/serving model; 'requested_model' is the client's.
    assert kwargs['model'] == 'sonnet'
    assert kwargs['requested_model'] == 'haiku'
    # Body still echoes the requested model.
    _, body = handler._send_json.call_args.args
    assert body['model'] == 'haiku'


def test_dispatch_stream_echoes_requested_model_in_message_start():
    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    backend.send_message_stream.return_value = iter([
        _msg_start('sonnet'),
        'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n',
    ])
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = None
    handler._ctx_key = None
    handler._route_est = 0
    handler._routing = _routing('haiku', 'sonnet', applied=True)
    snapshot = _fake_snapshot('anthropic', backend)

    handler._dispatch({'model': 'sonnet', 'stream': True}, snapshot, 1)

    gen = handler._send_sse.call_args.args[0]
    chunks = list(gen)
    start_event = json.loads(chunks[0].split('data: ', 1)[1].strip())
    assert start_event['message']['model'] == 'haiku'
    # message_delta untouched.
    assert 'message_delta' in chunks[1]


def test_dispatch_stream_passthrough_when_not_routed():
    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    original = [_msg_start('sonnet')]
    backend.send_message_stream.return_value = iter(original)
    registry = MagicMock()
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = None
    handler._ctx_key = None
    handler._route_est = 0
    handler._routing = _routing('sonnet', 'sonnet', applied=False)
    snapshot = _fake_snapshot('anthropic', backend)

    handler._dispatch({'model': 'sonnet', 'stream': True}, snapshot, 1)

    gen = handler._send_sse.call_args.args[0]
    chunks = list(gen)
    # Byte-for-byte identical: no rewrite when routing did not apply.
    assert chunks == original


def test_dispatch_stream_echo_survives_stats_wrapper():
    """The echo wrapper sits outside the stats wrapper and still rewrites."""
    backend = MagicMock()
    backend.parse_credentials.return_value = {}
    backend.send_message_stream.return_value = iter([_msg_start('sonnet')])
    registry = MagicMock()
    registry.session_context.return_value = (0, 1.0)
    handler = _handler_with_registry(registry)
    handler.headers = {'x-api-key': 'key'}
    handler.stats_collector = MagicMock()
    handler._ctx_key = None
    handler._route_est = 0
    handler._routing = _routing('haiku', 'sonnet', applied=True)
    snapshot = _fake_snapshot('anthropic', backend)

    handler._dispatch({'model': 'sonnet', 'stream': True}, snapshot, 1)

    gen = handler._send_sse.call_args.args[0]
    chunks = list(gen)
    start_event = json.loads(chunks[0].split('data: ', 1)[1].strip())
    assert start_event['message']['model'] == 'haiku'
    # Stats still recorded with the routed model.
    _, kwargs = handler.stats_collector.record.call_args
    assert kwargs['model'] == 'sonnet'


# ---------------------------------------------------------------------------
# X-Anthproxy-Override header parsing
# ---------------------------------------------------------------------------

class TestParseOverrideHeader:
    def test_empty(self):
        assert _parse_override_header(None) == {}
        assert _parse_override_header('') == {}
        assert _parse_override_header('   ') == {}

    def test_no_classifier(self):
        assert _parse_override_header('no-classifier') == {'no_classifier': True}

    def test_no_classifier_case_insensitive(self):
        assert _parse_override_header('No-Classifier') == {'no_classifier': True}
        assert _parse_override_header('NO-CLASSIFIER') == {'no_classifier': True}

    def test_prefer(self):
        assert _parse_override_header('prefer:codex') == {'prefer_backend': 'codex'}
        assert _parse_override_header('prefer:openrouter') == {'prefer_backend': 'openrouter'}

    def test_prefer_case_insensitive(self):
        assert _parse_override_header('PREFER:CODEX') == {'prefer_backend': 'codex'}

    def test_prefer_unknown_backend(self):
        # Unknown backend names are silently ignored (not added to result).
        assert _parse_override_header('prefer:unknown') == {}

    def test_multiple_directives(self):
        assert _parse_override_header('no-classifier; prefer:codex') == {
            'no_classifier': True,
            'prefer_backend': 'codex',
        }

    def test_multiple_prefer_last_wins(self):
        assert _parse_override_header('prefer:codex; prefer:openrouter') == {
            'prefer_backend': 'openrouter',
        }

    def test_unknown_directive_silently_ignored(self):
        assert _parse_override_header('unknown-directive') == {}
        assert _parse_override_header('no-classifier; unknown-thing') == {
            'no_classifier': True,
        }

    def test_whitespace_trimmed(self):
        assert _parse_override_header(' no-classifier ; prefer: codex ') == {
            'no_classifier': True,
            'prefer_backend': 'codex',
        }

    def test_too_long_header_ignored(self):
        # Headers longer than 2048 chars are ignored entirely.
        long_value = 'no-classifier; ' * 200  # well over 2048 chars
        assert _parse_override_header(long_value) == {}


# ---------------------------------------------------------------------------
# _serve_ui_file  (M3: path resolution, traversal, SPA fallback, headers)
# ---------------------------------------------------------------------------


def _make_ui_handler(tmp_path, monkeypatch):
    """Return a ProxyRequestHandler instance wired to serve from tmp_path/ui/dist."""
    # Redirect __file__ so ui_dist resolves to tmp_path/anthproxy/ui/dist
    fake_handlers_py = tmp_path / 'anthproxy' / 'handlers.py'
    fake_handlers_py.parent.mkdir(parents=True, exist_ok=True)
    fake_handlers_py.touch()
    monkeypatch.setattr(_handlers_module, '__file__', str(fake_handlers_py))

    ui_dist = tmp_path / 'anthproxy' / 'ui' / 'dist'
    ui_dist.mkdir(parents=True, exist_ok=True)

    handler = object.__new__(ProxyRequestHandler)
    handler.written = []
    handler.headers_sent = []

    handler.send_response = MagicMock()
    handler.end_headers = MagicMock()
    handler.send_header = MagicMock(side_effect=lambda k, v: handler.headers_sent.append((k, v)))
    handler.wfile = MagicMock()
    handler.wfile.write.side_effect = handler.written.append
    handler._send_json = MagicMock()

    return handler, ui_dist


class TestServeUiFile:
    def test_serves_index_html_at_ui_root(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'index.html').write_bytes(b'<html>hello</html>')

        handler._serve_ui_file('/ui')

        handler.send_response.assert_called_once_with(200)
        written = b''.join(handler.written)
        assert written == b'<html>hello</html>'

    def test_serves_index_html_at_ui_slash(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'index.html').write_bytes(b'<html>root</html>')

        handler._serve_ui_file('/ui/')

        handler.send_response.assert_called_once_with(200)

    def test_serves_asset_file(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        assets = ui_dist / 'assets'
        assets.mkdir()
        (assets / 'app.js').write_bytes(b'console.log("hi")')

        handler._serve_ui_file('/ui/assets/app.js')

        handler.send_response.assert_called_once_with(200)
        written = b''.join(handler.written)
        assert written == b'console.log("hi")'

    def test_content_type_html(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'index.html').write_bytes(b'<html/>')

        handler._serve_ui_file('/ui/index.html')

        ct = dict(handler.headers_sent).get('Content-Type', '')
        assert 'text/html' in ct

    def test_content_type_javascript(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'app.js').write_bytes(b'x=1')

        handler._serve_ui_file('/ui/app.js')

        ct = dict(handler.headers_sent).get('Content-Type', '')
        assert 'javascript' in ct

    def test_content_length_header(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        payload = b'<html>test</html>'
        (ui_dist / 'index.html').write_bytes(payload)

        handler._serve_ui_file('/ui/index.html')

        cl = dict(handler.headers_sent).get('Content-Length')
        assert cl == str(len(payload))

    def test_cache_control_no_cache_for_index(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'index.html').write_bytes(b'<html/>')

        handler._serve_ui_file('/ui/index.html')

        cc = dict(handler.headers_sent).get('Cache-Control', '')
        assert cc == 'no-cache'

    def test_cache_control_long_for_asset(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'app.js').write_bytes(b'x=1')

        handler._serve_ui_file('/ui/app.js')

        cc = dict(handler.headers_sent).get('Cache-Control', '')
        assert 'max-age=' in cc
        assert 'public' in cc

    def test_missing_file_falls_back_to_index_html(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        (ui_dist / 'index.html').write_bytes(b'<html>spa</html>')

        handler._serve_ui_file('/ui/settings')

        # SPA fallback: serves index.html, returns 200
        handler.send_response.assert_called_once_with(200)
        written = b''.join(handler.written)
        assert written == b'<html>spa</html>'

    def test_missing_file_and_no_index_returns_404(self, tmp_path, monkeypatch):
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        # No index.html created → both the asset and the fallback are missing

        handler._serve_ui_file('/ui/nonexistent-page')

        handler._send_json.assert_called_once()
        status_code = handler._send_json.call_args[0][0]
        assert status_code == 404

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        """../../../etc/passwd must not escape ui/dist."""
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)
        # Place a sentinel above ui/dist to prove it's not served
        (tmp_path / 'anthproxy' / 'secret.txt').write_text('secret')

        handler._serve_ui_file('/ui/../secret.txt')

        # Must NOT serve the file — either 404 _send_json or fallback to index.html
        # (no index.html exists here, so _send_json 404 fires)
        handler._send_json.assert_called_once()
        status_code = handler._send_json.call_args[0][0]
        assert status_code == 404
        # Must not have written the secret content
        written = b''.join(handler.written)
        assert b'secret' not in written

    def test_path_traversal_absolute_blocked(self, tmp_path, monkeypatch):
        """An absolute-looking path that resolves outside ui/dist must be blocked."""
        handler, ui_dist = _make_ui_handler(tmp_path, monkeypatch)

        # Craft a path that after /ui prefix stripping tries to escape
        handler._serve_ui_file('/ui/../../anthproxy/secret.txt')

        handler._send_json.assert_called_once()
        assert handler._send_json.call_args[0][0] == 404


# ---------------------------------------------------------------------------
# Admin route selector forwarding
# ---------------------------------------------------------------------------


def _make_admin_handler():
    """Return a bare ProxyRequestHandler wired for /admin dispatch."""
    handler = object.__new__(ProxyRequestHandler)
    handler.enable_ui = True
    handler.registry = MagicMock()
    handler.session_db = MagicMock()
    handler.selector = MagicMock()
    handler.headers = {}
    handler._send_json = MagicMock()
    return handler


class TestAdminSelectorForwarding:
    def test_do_post_forwards_selector_to_handle_post(self, monkeypatch):
        import anthproxy.admin as admin_module

        handler = _make_admin_handler()
        handler.path = '/admin/global/backend'
        handler._read_body = MagicMock(return_value=b'{"prefer": "anthropic"}')

        spy = MagicMock(return_value=(200, {'status': 'ok'}))
        monkeypatch.setattr(admin_module, 'handle_post', spy)

        handler.do_POST()

        spy.assert_called_once_with(
            '/admin/global/backend',
            {'prefer': 'anthropic'},
            handler.registry,
            handler.session_db,
            selector=handler.selector,
        )
        handler._send_json.assert_called_once_with(200, {'status': 'ok'})

    def test_do_get_forwards_selector_to_handle_get(self, monkeypatch):
        import anthproxy.admin as admin_module

        handler = _make_admin_handler()
        handler.path = '/admin/status'

        spy = MagicMock(return_value=(200, {'active_backend': 'bedrock'}))
        monkeypatch.setattr(admin_module, 'handle_get', spy)

        handler.do_GET()

        spy.assert_called_once_with(
            '/admin/status',
            {},
            handler.registry,
            handler.session_db,
            selector=handler.selector,
        )
        handler._send_json.assert_called_once_with(200, {'active_backend': 'bedrock'})


# ---------------------------------------------------------------------------
# _has_happy_new_year_system_prompt
# ---------------------------------------------------------------------------

class TestHasHappyNewYearSystemPrompt:
    def test_string_exact_prefix(self):
        assert _has_happy_new_year_system_prompt({'system': HAPPY_NEW_YEAR_PREFIX}) is True

    def test_string_with_suffix(self):
        assert _has_happy_new_year_system_prompt({'system': f'{HAPPY_NEW_YEAR_PREFIX}, friend!'}) is True

    def test_string_wrong_case(self):
        assert _has_happy_new_year_system_prompt({'system': 'happy new year'}) is False

    def test_string_mismatch(self):
        assert _has_happy_new_year_system_prompt({'system': 'Hello world'}) is False

    def test_string_empty(self):
        assert _has_happy_new_year_system_prompt({'system': ''}) is False

    def test_list_first_block_matches(self):
        assert _has_happy_new_year_system_prompt(
            {'system': [{'type': 'text', 'text': f'{HAPPY_NEW_YEAR_PREFIX} everyone'}]}) is True

    def test_list_second_block_matches(self):
        assert _has_happy_new_year_system_prompt(
            {'system': [
                {'type': 'text', 'text': 'Preamble'},
                {'type': 'text', 'text': f'{HAPPY_NEW_YEAR_PREFIX}!'},
            ]}) is True

    def test_list_no_block_matches(self):
        assert _has_happy_new_year_system_prompt(
            {'system': [{'type': 'text', 'text': 'Just a normal prompt'}]}) is False

    def test_list_non_dict_items_skipped(self):
        assert _has_happy_new_year_system_prompt(
            {'system': [HAPPY_NEW_YEAR_PREFIX, {'type': 'text', 'text': 'other'}]}) is False

    def test_list_block_missing_text_key(self):
        assert _has_happy_new_year_system_prompt({'system': [{'type': 'text'}]}) is False

    def test_list_any_type_matches(self):
        assert _has_happy_new_year_system_prompt(
            {'system': [{'type': 'cache_control', 'text': f'{HAPPY_NEW_YEAR_PREFIX}'}]}) is True

    def test_no_system_key(self):
        assert _has_happy_new_year_system_prompt({}) is False

    def test_system_none(self):
        assert _has_happy_new_year_system_prompt({'system': None}) is False

    def test_system_integer(self):
        assert _has_happy_new_year_system_prompt({'system': 42}) is False

    def test_system_empty_list(self):
        assert _has_happy_new_year_system_prompt({'system': []}) is False


# ---------------------------------------------------------------------------
# _handle_happy_new_year short-circuit
# ---------------------------------------------------------------------------

class TestHappyNewYearShortCircuit:
    def test_non_streaming_calls_send_json(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {'system': HAPPY_NEW_YEAR_PREFIX, 'model': 'claude-opus-4-5', 'stream': False}
        handler._handle_happy_new_year(payload)
        handler._send_json.assert_called_once()
        handler._send_sse.assert_not_called()

    def test_non_streaming_reply_text(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {'system': HAPPY_NEW_YEAR_PREFIX, 'model': 'claude-opus-4-5', 'stream': False}
        handler._handle_happy_new_year(payload)
        body = handler._send_json.call_args[0][1]
        assert body['content'][0]['text'] == HAPPY_BIRTHDAY_REPLY

    def test_streaming_calls_send_sse(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {'system': HAPPY_NEW_YEAR_PREFIX, 'model': 'claude-opus-4-5', 'stream': True}
        handler._handle_happy_new_year(payload)
        handler._send_sse.assert_called_once()
        handler._send_json.assert_not_called()

    def test_streaming_reply_text(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {'system': HAPPY_NEW_YEAR_PREFIX, 'model': 'claude-opus-4-5', 'stream': True}
        handler._handle_happy_new_year(payload)
        gen = handler._send_sse.call_args[0][0]
        events = _parse_sse(list(gen))
        delta = next(d for name, d in events if name == 'content_block_delta')
        assert delta['delta']['text'] == HAPPY_BIRTHDAY_REPLY

    def test_list_system_trigger(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {
            'system': [{'type': 'text', 'text': f'{HAPPY_NEW_YEAR_PREFIX}, everyone!'}],
            'model': 'claude-opus-4-5',
            'stream': False,
        }
        handler._handle_happy_new_year(payload)
        body = handler._send_json.call_args[0][1]
        assert body['content'][0]['text'] == HAPPY_BIRTHDAY_REPLY

    def test_registry_snapshot_never_called(self):
        registry = MagicMock()
        handler = _handler_with_registry(registry)
        payload = {'system': HAPPY_NEW_YEAR_PREFIX, 'model': 'claude-opus-4-5', 'stream': False}
        handler._handle_happy_new_year(payload)
        registry.snapshot.assert_not_called()
