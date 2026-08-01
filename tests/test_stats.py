"""Tests for anthproxy.stats — storage, pricing helpers, and bucketed rendering."""
import dataclasses
import datetime
import json
import time

import pytest

from anthproxy.stats import (
    RoutingEconomics,
    StatsCollector,
    StatsPeriod,
    _classify_model,
    _fmt_cost,
    _matches_backend_filter,
    _record_cost,
    format_stats_markdown,
    resolve_stats_period,
    routing_economics,
)


# ---------------------------------------------------------------------------
# _classify_model
# ---------------------------------------------------------------------------

class TestClassifyModel:
    def test_bare_aliases(self):
        assert _classify_model('opus') == 'opus'
        assert _classify_model('sonnet') == 'sonnet'
        assert _classify_model('haiku') == 'haiku'

    def test_full_anthropic_ids(self):
        assert _classify_model('claude-opus-4-8') == 'opus'
        assert _classify_model('claude-sonnet-4-6') == 'sonnet'
        assert _classify_model('claude-haiku-4-5-20251001') == 'haiku'

    def test_context_window_suffixes(self):
        assert _classify_model('claude-sonnet-4-5-20250929[1m]') == 'sonnet'
        assert _classify_model('claude-opus-4-8:1m') == 'opus'

    def test_legacy_ids(self):
        assert _classify_model('claude-3-haiku-20240307') == 'haiku'
        assert _classify_model('claude-3-5-haiku-20241022') == 'haiku'

    def test_unknown_returns_other(self):
        assert _classify_model('') == 'other'
        assert _classify_model('gpt-5.5') == 'other'
        assert _classify_model('plugin-model-1') == 'other'
        assert _classify_model(None) == 'other'

    def test_case_insensitive(self):
        assert _classify_model('OPUS') == 'opus'
        assert _classify_model('Sonnet') == 'sonnet'


# ---------------------------------------------------------------------------
# _record_cost
# ---------------------------------------------------------------------------

class TestRecordCost:
    def _r(self, model, **kw):
        return {'model': model, **kw}

    def test_opus_input_only(self):
        cost = _record_cost(self._r('opus', input_tokens=1_000_000))
        assert abs(cost - 5.0) < 1e-9

    def test_opus_output_only(self):
        cost = _record_cost(self._r('opus', output_tokens=1_000_000))
        assert abs(cost - 25.0) < 1e-9

    def test_opus_cache_read(self):
        cost = _record_cost(self._r('opus', cache_read_tokens=1_000_000))
        assert abs(cost - 0.50) < 1e-9

    def test_opus_cache_write(self):
        cost = _record_cost(self._r('opus', cache_creation_tokens=1_000_000))
        assert abs(cost - 6.25) < 1e-9

    def test_sonnet_full(self):
        cost = _record_cost(self._r(
            'sonnet',
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        ))
        assert abs(cost - 18.30) < 1e-6

    def test_haiku(self):
        cost = _record_cost(self._r('haiku', input_tokens=1_000_000))
        assert abs(cost - 1.0) < 1e-9

    def test_other_zero_cost(self):
        assert _record_cost({'model': 'gpt-5.5', 'input_tokens': 9_999_999}) == 0.0
        assert _record_cost({'model': '', 'output_tokens': 9_999_999}) == 0.0

    def test_zero_tokens(self):
        assert _record_cost(self._r('opus')) == 0.0

    def test_combined_opus(self):
        cost = _record_cost(self._r(
            'opus',
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        ))
        assert abs(cost - 30.50) < 1e-6


# ---------------------------------------------------------------------------
# _fmt_cost
# ---------------------------------------------------------------------------

class TestFmtCost:
    def test_zero(self):
        assert _fmt_cost(0.0) == '$0.00'
        assert _fmt_cost(-1.0) == '$0.00'

    def test_small(self):
        assert _fmt_cost(0.0012) == '$0.0012'

    def test_normal(self):
        assert _fmt_cost(1.5) == '$1.50'
        assert _fmt_cost(1234.56) == '$1,234.56'


# ---------------------------------------------------------------------------
# resolve_stats_period
# ---------------------------------------------------------------------------

class TestResolveStatsPeriod:
    def test_default_day_aliases(self):
        now = datetime.datetime(2026, 6, 8, 15, 45, 0)
        bare = resolve_stats_period('', now=now)
        day = resolve_stats_period('1d', now=now)
        alias = resolve_stats_period('d', now=now)
        assert bare == day == alias
        assert day.label == 'today'
        assert day.bucket == 'hour'
        assert day.token == '1d'

    def test_current_month_alias(self):
        now = datetime.datetime(2026, 6, 8, 15, 45, 0)
        assert resolve_stats_period('1m', now=now) == resolve_stats_period('m', now=now)

    def test_yesterday_period(self):
        now = datetime.datetime(2026, 6, 8, 15, 45, 0)
        period = resolve_stats_period('-1d', now=now)
        start = datetime.datetime.fromtimestamp(period.start_ts)
        end = datetime.datetime.fromtimestamp(period.end_ts)
        assert start == datetime.datetime(2026, 6, 7, 0, 0, 0)
        assert end == datetime.datetime(2026, 6, 8, 0, 0, 0)
        assert period.bucket == 'hour'
        assert period.label == 'yesterday (2026-06-07)'

    def test_day_rolls_across_year(self):
        now = datetime.datetime(2026, 1, 1, 8, 0, 0)
        period = resolve_stats_period('-1d', now=now)
        assert datetime.datetime.fromtimestamp(period.start_ts) == datetime.datetime(2025, 12, 31, 0, 0, 0)
        assert datetime.datetime.fromtimestamp(period.end_ts) == datetime.datetime(2026, 1, 1, 0, 0, 0)

    def test_last_month_rolls_across_year(self):
        now = datetime.datetime(2026, 1, 20, 12, 0, 0)
        period = resolve_stats_period('-1m', now=now)
        assert datetime.datetime.fromtimestamp(period.start_ts) == datetime.datetime(2025, 12, 1, 0, 0, 0)
        assert datetime.datetime.fromtimestamp(period.end_ts) == datetime.datetime(2026, 1, 1, 0, 0, 0)
        assert period.bucket == 'week'
        assert period.label == 'last month (December 2025)'

    def test_leap_year_february(self):
        now = datetime.datetime(2024, 3, 15, 12, 0, 0)
        period = resolve_stats_period('-1m', now=now)
        assert datetime.datetime.fromtimestamp(period.start_ts) == datetime.datetime(2024, 2, 1, 0, 0, 0)
        assert datetime.datetime.fromtimestamp(period.end_ts) == datetime.datetime(2024, 3, 1, 0, 0, 0)

    def test_week_and_quarter_bounds(self):
        now = datetime.datetime(2026, 6, 10, 15, 30, 0)
        week = resolve_stats_period('1w', now=now)
        quarter = resolve_stats_period('1q', now=now)
        assert datetime.datetime.fromtimestamp(week.start_ts) == datetime.datetime(2026, 6, 8, 0, 0, 0)
        assert datetime.datetime.fromtimestamp(week.end_ts) == datetime.datetime(2026, 6, 15, 0, 0, 0)
        assert datetime.datetime.fromtimestamp(quarter.start_ts) == datetime.datetime(2026, 4, 1, 0, 0, 0)
        assert datetime.datetime.fromtimestamp(quarter.end_ts) == datetime.datetime(2026, 7, 1, 0, 0, 0)

    @pytest.mark.skipif(not hasattr(time, 'tzset'), reason='tzset not available')
    def test_day_boundaries_use_local_midnights_under_dst(self, monkeypatch):
        old_tz = __import__('os').environ.get('TZ')
        monkeypatch.setenv('TZ', 'America/New_York')
        time.tzset()
        try:
            now = datetime.datetime(2026, 3, 9, 12, 0, 0)
            period = resolve_stats_period('-1d', now=now)
            start = datetime.datetime.fromtimestamp(period.start_ts)
            end = datetime.datetime.fromtimestamp(period.end_ts)
            assert start == datetime.datetime(2026, 3, 8, 0, 0, 0)
            assert end == datetime.datetime(2026, 3, 9, 0, 0, 0)
            assert start.date() + datetime.timedelta(days=1) == end.date()
        finally:
            if old_tz is None:
                monkeypatch.delenv('TZ', raising=False)
            else:
                monkeypatch.setenv('TZ', old_tz)
            time.tzset()

    @pytest.mark.parametrize('token', [
        '7d', '2m', '+1d', '0d', '-0d', '-01d', '-9999999d', '-9999999m',
        '-9999999w', '-9999999q', '', 'bad',
    ])
    def test_invalid_tokens_fall_back_to_today(self, token):
        now = datetime.datetime(2026, 6, 8, 15, 45, 0)
        assert resolve_stats_period(token, now=now) == resolve_stats_period('1d', now=now)

    def test_current_week_alias(self):
        now = datetime.datetime(2026, 6, 10, 15, 45, 0)
        assert resolve_stats_period('w', now=now) == resolve_stats_period('1w', now=now)

    def test_current_quarter_alias(self):
        now = datetime.datetime(2026, 6, 10, 15, 45, 0)
        assert resolve_stats_period('q', now=now) == resolve_stats_period('1q', now=now)

    def test_last_week_period(self):
        # Wednesday 2026-06-10: previous week is Mon 2026-06-01 → Mon 2026-06-08
        now = datetime.datetime(2026, 6, 10, 15, 30, 0)
        period = resolve_stats_period('-1w', now=now)
        start = datetime.datetime.fromtimestamp(period.start_ts)
        end = datetime.datetime.fromtimestamp(period.end_ts)
        assert start == datetime.datetime(2026, 6, 1, 0, 0, 0)
        assert end == datetime.datetime(2026, 6, 8, 0, 0, 0)
        assert period.bucket == 'day'
        assert period.label == 'last week (week of 2026-06-01)'

    def test_weeks_ago(self):
        # Wednesday 2026-06-10: 2 weeks ago is Mon 2026-05-25 → Mon 2026-06-01
        now = datetime.datetime(2026, 6, 10, 15, 30, 0)
        period = resolve_stats_period('-2w', now=now)
        start = datetime.datetime.fromtimestamp(period.start_ts)
        end = datetime.datetime.fromtimestamp(period.end_ts)
        assert start == datetime.datetime(2026, 5, 25, 0, 0, 0)
        assert end == datetime.datetime(2026, 6, 1, 0, 0, 0)
        assert period.bucket == 'day'
        assert period.label == '2 weeks ago (week of 2026-05-25)'

    def test_last_quarter_period(self):
        # Q2 2026 (Jun 10): last quarter is Q1 2026 = Jan 1 → Apr 1
        now = datetime.datetime(2026, 6, 10, 15, 30, 0)
        period = resolve_stats_period('-1q', now=now)
        start = datetime.datetime.fromtimestamp(period.start_ts)
        end = datetime.datetime.fromtimestamp(period.end_ts)
        assert start == datetime.datetime(2026, 1, 1, 0, 0, 0)
        assert end == datetime.datetime(2026, 4, 1, 0, 0, 0)
        assert period.bucket == 'month'
        assert period.label == 'last quarter (Q1 2026)'

    def test_quarters_ago_rolls_year(self):
        # Q2 2026: 2 quarters ago = Q4 2025 = Oct 1 2025 → Jan 1 2026
        now = datetime.datetime(2026, 6, 10, 15, 30, 0)
        period = resolve_stats_period('-2q', now=now)
        start = datetime.datetime.fromtimestamp(period.start_ts)
        end = datetime.datetime.fromtimestamp(period.end_ts)
        assert start == datetime.datetime(2025, 10, 1, 0, 0, 0)
        assert end == datetime.datetime(2026, 1, 1, 0, 0, 0)
        assert period.bucket == 'month'
        assert period.label == '2 quarters ago (Q4 2025)'


# ---------------------------------------------------------------------------
# StatsCollector — storage
# ---------------------------------------------------------------------------

def _record_for(ts: float, backend: str, model: str, **kw) -> dict:
    return {
        'ts': ts,
        'backend': backend,
        'model': model,
        'input_tokens': kw.get('input_tokens', 0),
        'output_tokens': kw.get('output_tokens', 0),
        'cache_creation_tokens': 0,
        'cache_read_tokens': 0,
        'duration_ms': kw.get('duration_ms', 100),
        'streaming': False,
    }


class TestStatsCollectorStorage:
    def _no_legacy(self, monkeypatch, tmp_path):
        import anthproxy.stats as stats_mod
        monkeypatch.setattr(stats_mod, '_LEGACY_STATS_FILE', tmp_path / 'no_legacy.jsonl')

    def test_record_writes_day_file(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record('bedrock', model='sonnet', input_tokens=10, output_tokens=5, duration_ms=200)
        files = list((tmp_path / 'stats').glob('*.jsonl'))
        assert len(files) == 1
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        assert files[0].name == f'{today}.jsonl'
        rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]['backend'] == 'bedrock'
        assert rows[0]['model'] == 'sonnet'
        assert rows[0]['input_tokens'] == 10

    def test_read_records_single_day(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record('anthropic', model='opus', input_tokens=100)
        sc.record('anthropic', model='haiku', input_tokens=200)
        now = time.time()
        records = sc.read_records(now - 86400, now + 10)
        assert len(records) == 2
        assert records[0]['model'] == 'opus'
        assert records[1]['model'] == 'haiku'

    def test_read_records_filters_by_half_open_bounds(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        base = datetime.datetime(2026, 6, 8, 0, 0, 0).timestamp()
        day_file = (tmp_path / 'stats') / '2026-06-08.jsonl'
        day_file.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            _record_for(base - 1, 'bedrock', 'before'),
            _record_for(base, 'bedrock', 'at-start'),
            _record_for(base + 10, 'bedrock', 'inside'),
            _record_for(base + 3600, 'bedrock', 'at-end'),
        ]
        day_file.write_text(''.join(json.dumps(entry) + '\n' for entry in entries))
        records = sc.read_records(base, base + 3600)
        models = [r['model'] for r in records]
        assert models == ['at-start', 'inside']

    def test_read_records_multiple_day_files(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        stats_dir = tmp_path / 'stats'
        stats_dir.mkdir()
        ts_a = datetime.datetime(2026, 6, 6, 10, 0, 0).timestamp()
        ts_b = datetime.datetime(2026, 6, 8, 10, 0, 0).timestamp()
        day_a = '2026-06-06'
        day_b = '2026-06-08'

        def _write(path, entry):
            p = stats_dir / path
            with open(p, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

        _write(f'{day_a}.jsonl', _record_for(ts_a, 'plugin', 'haiku', input_tokens=1))
        _write(f'{day_b}.jsonl', _record_for(ts_b, 'codex', 'sonnet', input_tokens=2))

        sc = StatsCollector(stats_dir=stats_dir)
        records = sc.read_records(ts_a - 1, ts_b + 1)
        assert len(records) == 2
        assert {r['model'] for r in records} == {'haiku', 'sonnet'}

    def test_read_records_does_not_include_newer_files_for_historical_query(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        stats_dir = tmp_path / 'stats'
        stats_dir.mkdir()
        older = datetime.datetime(2026, 6, 7, 10, 0, 0).timestamp()
        newer = datetime.datetime(2026, 6, 8, 10, 0, 0).timestamp()
        (stats_dir / '2026-06-07.jsonl').write_text(json.dumps(_record_for(older, 'bedrock', 'older')) + '\n')
        (stats_dir / '2026-06-08.jsonl').write_text(json.dumps(_record_for(newer, 'bedrock', 'newer')) + '\n')
        sc = StatsCollector(stats_dir=stats_dir)
        records = sc.read_records(datetime.datetime(2026, 6, 7, 0, 0, 0).timestamp(), datetime.datetime(2026, 6, 8, 0, 0, 0).timestamp())
        assert [r['model'] for r in records] == ['older']

    def test_legacy_file_is_read_with_bounds(self, tmp_path, monkeypatch):
        import anthproxy.stats as stats_mod

        stats_dir = tmp_path / 'stats'
        legacy = tmp_path / 'stats.jsonl'
        ts_old = datetime.datetime(2026, 6, 7, 10, 0, 0).timestamp()
        ts_new = datetime.datetime(2026, 6, 8, 10, 0, 0).timestamp()
        legacy.write_text(
            json.dumps(_record_for(ts_old, 'bedrock', 'opus', input_tokens=999)) + '\n'
            + json.dumps(_record_for(ts_new, 'bedrock', 'sonnet', input_tokens=1)) + '\n'
        )
        monkeypatch.setattr(stats_mod, '_LEGACY_STATS_FILE', legacy)

        sc = StatsCollector(stats_dir=stats_dir)
        records = sc.read_records(datetime.datetime(2026, 6, 7, 0, 0, 0).timestamp(), datetime.datetime(2026, 6, 8, 0, 0, 0).timestamp())
        assert [r['model'] for r in records] == ['opus']

    def test_read_records_missing_dir_returns_empty(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'nonexistent' / 'stats')
        records = sc.read_records(time.time() - 3600, time.time() + 3600)
        assert records == []

    def test_read_records_invalid_ranges_raise(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        with pytest.raises(ValueError):
            sc.read_records(10, 10)
        with pytest.raises(ValueError):
            sc.read_records(11, 10)
        with pytest.raises(ValueError):
            sc.read_records(float('nan'), 10)
        with pytest.raises(ValueError):
            sc.read_records(0, float('inf'))

    def test_read_records_rejects_too_many_partitions(self, tmp_path, monkeypatch):
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        start = datetime.datetime(2026, 1, 1, 0, 0, 0).timestamp()
        end = datetime.datetime(2026, 4, 4, 0, 0, 0).timestamp()
        with pytest.raises(ValueError):
            sc.read_records(start, end)

    def test_record_never_raises_on_bad_dir(self, tmp_path):
        bad = tmp_path / 'stats2'
        bad.write_text('not a directory')
        sc2 = StatsCollector.__new__(StatsCollector)
        sc2._dir = bad
        import threading
        sc2._lock = threading.Lock()
        sc2.record('bedrock', model='sonnet', input_tokens=1)


# ---------------------------------------------------------------------------
# format_stats_markdown — rendering
# ---------------------------------------------------------------------------

def _ts_today_at(hour: int) -> float:
    now = datetime.datetime.now()
    return now.replace(hour=hour, minute=0, second=0, microsecond=0).timestamp()


def _ts_days_ago(days: int, hour: int = 10) -> float:
    now = datetime.datetime.now()
    return (now - datetime.timedelta(days=days)).replace(
        hour=hour, minute=0, second=0, microsecond=0).timestamp()


def _period(label: str, bucket: str) -> StatsPeriod:
    return StatsPeriod('test', 0.0, 1.0, label, bucket)  # type: ignore[arg-type]


class TestFormatStatsMarkdown:
    def test_empty_records(self):
        md = format_stats_markdown([], _period('today', 'hour'))
        assert 'No requests recorded' in md

    def test_day_groups_by_hour(self):
        records = [
            _record_for(_ts_today_at(9), 'anthropic', 'sonnet', input_tokens=100),
            _record_for(_ts_today_at(9), 'anthropic', 'sonnet', output_tokens=50),
            _record_for(_ts_today_at(14), 'anthropic', 'opus', input_tokens=200),
        ]
        md = format_stats_markdown(records, _period('today', 'hour'))
        assert '### 09:00' in md
        assert '### 14:00' in md
        assert 'Sonnet' in md
        assert 'Opus' in md
        assert '$' in md

    def test_week_groups_by_day(self):
        records = [
            _record_for(_ts_days_ago(0), 'bedrock', 'haiku', input_tokens=500),
            _record_for(_ts_days_ago(2), 'bedrock', 'sonnet', input_tokens=500),
        ]
        md = format_stats_markdown(records, _period('this week', 'day'))
        assert __import__('re').search(r'\d{4}-\d{2}-\d{2}', md)
        assert 'Haiku' in md
        assert 'Sonnet' in md

    def test_month_groups_by_week(self):
        records = [_record_for(_ts_days_ago(0), 'codex', 'opus', output_tokens=1000)]
        md = format_stats_markdown(records, _period('this month', 'week'))
        assert 'Week of' in md
        assert 'Opus' in md

    def test_quarter_groups_by_month(self):
        records = [_record_for(_ts_days_ago(0), 'plugin', 'sonnet', input_tokens=100)]
        md = format_stats_markdown(records, _period('this quarter', 'month'))
        assert __import__('re').search(r'[A-Z][a-z]+ \d{4}', md)
        assert 'Sonnet' in md

    def test_historical_labels_appear(self):
        records = [_record_for(_ts_today_at(9), 'bedrock', 'haiku', input_tokens=10)]
        day_md = format_stats_markdown(records, _period('yesterday (2026-06-07)', 'hour'))
        month_md = format_stats_markdown(records, _period('last month (May 2026)', 'week'))
        assert 'yesterday (2026-06-07)' in day_md
        assert 'last month (May 2026)' in month_md

    def test_cost_columns_present_in_all_periods(self):
        records = [_record_for(_ts_today_at(10), 'anthropic', 'opus', input_tokens=10_000)]
        for period in (
            _period('today', 'hour'),
            _period('this week', 'day'),
            _period('this month', 'week'),
            _period('this quarter', 'month'),
        ):
            md = format_stats_markdown(records, period)
            assert 'Cost' in md
            assert '$' in md

    def test_subtotal_and_total_rows(self):
        records = [
            _record_for(_ts_today_at(9), 'anthropic', 'sonnet', input_tokens=1_000_000),
            _record_for(_ts_today_at(10), 'bedrock', 'opus', output_tokens=1_000_000),
        ]
        md = format_stats_markdown(records, _period('today', 'hour'))
        assert 'Subtotal' in md
        assert 'Total' in md

    def test_other_model_shows_zero_cost(self):
        records = [_record_for(_ts_today_at(9), 'plugin', 'plugin-model-1', input_tokens=9_999_999)]
        md = format_stats_markdown(records, _period('today', 'hour'))
        assert 'Other' in md
        assert '$0.00' in md

    def test_multiple_backends_and_tiers(self):
        records = [
            _record_for(_ts_today_at(8), 'anthropic', 'opus', input_tokens=100),
            _record_for(_ts_today_at(8), 'bedrock', 'sonnet', input_tokens=200),
            _record_for(_ts_today_at(8), 'bedrock', 'haiku', input_tokens=300),
        ]
        md = format_stats_markdown(records, _period('today', 'hour'))
        assert 'anthropic' in md
        assert 'bedrock' in md
        assert 'Opus' in md
        assert 'Sonnet' in md
        assert 'Haiku' in md

    # --- backend filter ---

    def _scoped_period(self, backend: str) -> StatsPeriod:
        """Return a today/hour period scoped to the given backend."""
        return dataclasses.replace(_period('today', 'hour'), backend=backend)

    def test_backend_filter_excludes_other_backends(self):
        records = [
            _record_for(_ts_today_at(9), 'bedrock', 'sonnet', input_tokens=100),
            _record_for(_ts_today_at(9), 'anthropic', 'opus', input_tokens=200),
        ]
        md = format_stats_markdown(records, self._scoped_period('bedrock'))
        assert 'bedrock' in md
        assert 'Sonnet' in md
        assert 'anthropic' not in md
        assert 'Opus' not in md

    def test_backend_filter_heading_includes_backend_name(self):
        records = [_record_for(_ts_today_at(9), 'codex', 'sonnet', input_tokens=50)]
        md = format_stats_markdown(records, self._scoped_period('codex'))
        assert '· codex' in md

    def test_subscription_filter_includes_subscription_backends(self):
        records = [
            _record_for(_ts_today_at(9), 'anthropic', 'sonnet', input_tokens=100),
            _record_for(_ts_today_at(9), 'codex', 'opus', input_tokens=200),
            _record_for(_ts_today_at(9), 'openrouter', 'haiku', input_tokens=300),
            _record_for(_ts_today_at(9), 'bedrock', 'haiku', input_tokens=400),
        ]
        md = format_stats_markdown(records, self._scoped_period('subscription'))
        assert 'anthropic' in md
        assert 'codex' in md
        assert 'openrouter' in md
        assert 'bedrock' not in md
        assert '· subscription' in md

    def test_backend_filter_empty_result_shows_no_requests(self):
        records = [_record_for(_ts_today_at(9), 'bedrock', 'sonnet', input_tokens=100)]
        md = format_stats_markdown(records, self._scoped_period('codex'))
        assert 'No requests recorded' in md
        assert '· codex' in md

    def test_no_backend_filter_regression(self):
        """Unfiltered output is unchanged when period.backend is None."""
        records = [
            _record_for(_ts_today_at(9), 'anthropic', 'opus', input_tokens=100),
            _record_for(_ts_today_at(9), 'bedrock', 'sonnet', input_tokens=200),
        ]
        md_all = format_stats_markdown(records, _period('today', 'hour'))
        assert 'anthropic' in md_all
        assert 'bedrock' in md_all
        # No '· <backend>' in heading when filter is absent
        assert '· anthropic' not in md_all.split('\n')[0]


# ---------------------------------------------------------------------------
# _matches_backend_filter
# ---------------------------------------------------------------------------

class TestMatchesBackendFilter:
    def test_exact_match(self):
        assert _matches_backend_filter('bedrock', 'bedrock') is True

    def test_exact_no_match(self):
        assert _matches_backend_filter('plugin', 'bedrock') is False

    def test_subscription_matches_anthropic(self):
        assert _matches_backend_filter('anthropic', 'subscription') is True

    def test_subscription_matches_codex(self):
        assert _matches_backend_filter('codex', 'subscription') is True

    def test_subscription_matches_openrouter(self):
        assert _matches_backend_filter('openrouter', 'subscription') is True

    def test_subscription_excludes_bedrock(self):
        assert _matches_backend_filter('bedrock', 'subscription') is False

    def test_subscription_excludes_non_subscription_backend(self):
        assert _matches_backend_filter('plugin', 'subscription') is False

    def test_subscription_excludes_local(self):
        assert _matches_backend_filter('local', 'subscription') is False


# ---------------------------------------------------------------------------
# TestRecordEnrichedFields — new observability fields on StatsCollector.record()
# ---------------------------------------------------------------------------

class TestRecordEnrichedFields:
    """Tests for the six new keyword-only observability fields on record()."""

    def _no_legacy(self, monkeypatch, tmp_path):
        import anthproxy.stats as stats_mod
        monkeypatch.setattr(stats_mod, '_LEGACY_STATS_FILE', tmp_path / 'no_legacy.jsonl')

    def _read_row(self, sc, tmp_path) -> dict:
        """Read the single JSONL row written into stats_dir today."""
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        rows = [
            json.loads(line)
            for line in (tmp_path / 'stats' / f'{today}.jsonl').read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1, f'expected 1 row, got {len(rows)}'
        return rows[0]

    # ------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------

    def test_defaults_write_success_fields(self, tmp_path, monkeypatch):
        """A bare record() call writes status='success', status_code=None, error=None."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record('anthropic', model='sonnet')

        row = self._read_row(sc, tmp_path)
        assert row['status'] == 'success'
        assert row['status_code'] is None
        assert row['error'] is None
        assert row['classification'] is None
        assert row['reason_code'] is None

    def test_requested_model_defaults_to_model(self, tmp_path, monkeypatch):
        """When requested_model is not passed, it falls back to the routed model."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record('anthropic', model='haiku')

        row = self._read_row(sc, tmp_path)
        assert row['requested_model'] == 'haiku'
        assert row['model'] == 'haiku'

    # ------------------------------------------------------------------
    # Success round-trip with routing metadata
    # ------------------------------------------------------------------

    def test_success_roundtrip_with_routing(self, tmp_path, monkeypatch):
        """Full success record with routing fields round-trips through read_records."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record(
            'anthropic',
            model='haiku',
            input_tokens=100,
            output_tokens=10,
            duration_ms=500,
            streaming=False,
            status='success',
            status_code=200,
            requested_model='opus',
            classification='trivial',
            reason_code='classifier_trivial',
        )

        now = time.time()
        records = sc.read_records(now - 86400, now + 10)
        assert len(records) == 1
        r = records[0]
        assert r['status'] == 'success'
        assert r['status_code'] == 200
        assert r['error'] is None
        assert r['model'] == 'haiku'           # routed model
        assert r['requested_model'] == 'opus'  # original request
        assert r['classification'] == 'trivial'
        assert r['reason_code'] == 'classifier_trivial'

    def test_explicit_requested_model_is_preserved(self, tmp_path, monkeypatch):
        """Explicit requested_model is NOT overridden by model fallback."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record('anthropic', model='sonnet', requested_model='opus', status_code=200)

        row = self._read_row(sc, tmp_path)
        assert row['requested_model'] == 'opus'
        assert row['model'] == 'sonnet'

    # ------------------------------------------------------------------
    # Error round-trip
    # ------------------------------------------------------------------

    def test_error_roundtrip(self, tmp_path, monkeypatch):
        """Error record with status_code and error type round-trips."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record(
            'anthropic',
            model='opus',
            duration_ms=300,
            status='error',
            status_code=429,
            error='rate_limit_error',
        )

        now = time.time()
        records = sc.read_records(now - 86400, now + 10)
        assert len(records) == 1
        r = records[0]
        assert r['status'] == 'error'
        assert r['status_code'] == 429
        assert r['error'] == 'rate_limit_error'
        # Token fields default to 0 for error records
        assert r['input_tokens'] == 0
        assert r['output_tokens'] == 0

    def test_error_with_none_status_code(self, tmp_path, monkeypatch):
        """status_code=None (SSE mid-flight error) is stored as JSON null."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')
        sc.record('anthropic', model='opus', status='error', error='sse_error', status_code=None)

        row = self._read_row(sc, tmp_path)
        assert row['status'] == 'error'
        assert row['status_code'] is None
        assert row['error'] == 'sse_error'

    # ------------------------------------------------------------------
    # Backward compatibility: old records without new keys
    # ------------------------------------------------------------------

    def test_old_record_reads_without_crash(self, tmp_path, monkeypatch):
        """read_records on a legacy record (no new keys) does not raise."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')

        # Hand-write an old-style record (pre-enrichment schema)
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        stats_dir = tmp_path / 'stats'
        stats_dir.mkdir(parents=True, exist_ok=True)
        old_record = {
            'ts': time.time(),
            'backend': 'bedrock',
            'model': 'sonnet',
            'input_tokens': 50,
            'output_tokens': 5,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'duration_ms': 1000,
            'streaming': True,
        }
        (stats_dir / f'{today}.jsonl').write_text(json.dumps(old_record) + '\n')

        now = time.time()
        records = sc.read_records(now - 86400, now + 10)
        assert len(records) == 1
        r = records[0]
        # New fields are absent — .get() must return None / safe defaults
        assert r.get('status') is None
        assert r.get('status_code') is None
        assert r.get('error') is None
        assert r.get('requested_model') is None
        assert r.get('classification') is None
        assert r.get('reason_code') is None
        # Existing fields preserved
        assert r['model'] == 'sonnet'
        assert r['input_tokens'] == 50

    def test_format_stats_markdown_works_on_old_records(self, tmp_path, monkeypatch):
        """format_stats_markdown renders cleanly when new fields are absent."""
        self._no_legacy(monkeypatch, tmp_path)
        sc = StatsCollector(stats_dir=tmp_path / 'stats')

        today = datetime.datetime.now().strftime('%Y-%m-%d')
        stats_dir = tmp_path / 'stats'
        stats_dir.mkdir(parents=True, exist_ok=True)
        old_record = {
            'ts': time.time(),
            'backend': 'anthropic',
            'model': 'opus',
            'input_tokens': 1000,
            'output_tokens': 100,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'duration_ms': 2000,
            'streaming': False,
        }
        (stats_dir / f'{today}.jsonl').write_text(json.dumps(old_record) + '\n')

        now = time.time()
        records = sc.read_records(now - 86400, now + 10)
        period = StatsPeriod('1d', now - 86400, now + 10, 'today', 'hour')
        md = format_stats_markdown(records, period)
        # Should render without error and include expected sections
        assert 'anthproxy stats' in md
        assert 'anthropic' in md
        assert 'Opus' in md


# ---------------------------------------------------------------------------
# routing_economics
# ---------------------------------------------------------------------------


class TestRoutingEconomics:
    """Tests for routing_economics() — per-request cost breakdown."""

    def test_normal_routed_savings_computed_correctly(self):
        """Routing to haiku on 1M input tokens yields savings vs the opus baseline."""
        econ = routing_economics(
            routed_model='haiku',
            classifier_model='haiku',
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        assert isinstance(econ, RoutingEconomics)
        assert econ.pricing_available is True
        # opus input = $5.0/M; haiku input = $1.0/M
        assert abs(econ.opus_baseline_cost - 5.0) < 1e-9
        assert abs(econ.routed_cost - 1.0) < 1e-9
        assert econ.classifier_overhead_usd == 0.0
        assert abs(econ.net_savings_usd - 4.0) < 1e-9

    def test_baseline_is_opus_regardless_of_routed_tier(self):
        """Baseline is always the opus tier, not the routed model."""
        # Serving on sonnet still measures against the opus baseline.
        econ = routing_economics(
            routed_model='sonnet',
            classifier_model='haiku',
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        assert econ.pricing_available is True
        assert abs(econ.opus_baseline_cost - 5.0) < 1e-9  # opus input $5.0/M
        assert abs(econ.routed_cost - 3.0) < 1e-9         # sonnet input $3.0/M
        assert abs(econ.net_savings_usd - 2.0) < 1e-9

    def test_unknown_routed_model_pricing_unavailable_no_crash(self):
        """Unknown routed model → pricing_available=False, no raise."""
        econ = routing_economics(
            routed_model='plugin-model-1',
            classifier_model='haiku',
            input_tokens=500_000,
            output_tokens=100_000,
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )
        assert econ.pricing_available is False
        assert econ.opus_baseline_cost == 0.0
        assert econ.routed_cost == 0.0
        assert econ.classifier_overhead_usd == 0.0
        assert econ.net_savings_usd == 0.0

    def test_classifier_overhead_included_when_nonzero(self):
        """Nonzero classifier tokens are priced and reduce net_savings."""
        # haiku pricing: input=$1.0/M, output=$5.0/M
        clf_in = 1_000
        clf_out = 10
        econ = routing_economics(
            routed_model='haiku',
            classifier_model='haiku',
            input_tokens=100_000,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            classifier_input_tokens=clf_in,
            classifier_output_tokens=clf_out,
        )
        assert econ.pricing_available is True
        expected_clf = (clf_in * 1.0 + clf_out * 5.0) / 1_000_000
        assert abs(econ.classifier_overhead_usd - expected_clf) < 1e-9
        # net_savings = baseline - routed - overhead
        expected_net = econ.opus_baseline_cost - econ.routed_cost - econ.classifier_overhead_usd
        assert abs(econ.net_savings_usd - expected_net) < 1e-9

    def test_cache_tokens_included_in_cost_basis(self):
        """cache_creation_tokens and cache_read_tokens are priced in the cost basis."""
        # opus pricing: cache_write=$6.25/M, cache_read=$0.50/M
        # Route to opus so baseline and routed match, isolating cache math (no savings).
        econ = routing_economics(
            routed_model='opus',
            classifier_model='haiku',
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        assert econ.pricing_available is True
        expected_cost = 6.25 + 0.50  # $6.25/M write + $0.50/M read
        assert abs(econ.opus_baseline_cost - expected_cost) < 1e-6
        assert abs(econ.routed_cost - expected_cost) < 1e-6
        # Same model → no savings (and no classifier overhead)
        assert abs(econ.net_savings_usd) < 1e-6
