"""Request statistics collector for anthproxy.

Records per-request metrics (tokens, cache hits, duration) to JSONL files
partitioned by local day under ``~/.anthproxy/stats/``. One file per day:
``~/.anthproxy/stats/YYYY-MM-DD.jsonl``. Reading is done by iterating the
day-files that fall within the requested window — no separate index.

Back-compat: if a legacy ``~/.anthproxy/stats.jsonl`` exists it is read in
addition to the day-files so existing data is not lost on upgrade.

Each record:
  {
    "ts":                    float,
    "backend":               str,
    "model":                 str,          # routed model (post-tier-routing)
    "input_tokens":          int,
    "output_tokens":         int,
    "cache_creation_tokens": int,
    "cache_read_tokens":     int,
    "duration_ms":           int,
    "streaming":             bool,
    "status":                str,          # "success" | "error"
    "status_code":           int | None,   # HTTP status (200/4xx/502); None when unknown
    "error":                 str | None,   # error_type string or "upstream_failure"; None on success
    "requested_model":       str,          # model before tier routing (== model when routing off)
    "classification":        str | None,   # classifier label ("trivial"/"standard"/"deep") or None
    "reason_code":           str | None    # router ReasonCode (e.g. "classifier_deep",
                                           # "size_forced_long_context") or None
  }

  New fields (status, status_code, error, requested_model, classification, reason_code) are absent
  on records written before this version — consumers must use .get() with safe defaults.
  Failure records (status="error") are emitted even when the upstream returned no usage data;
  token fields default to 0 in that case.
"""
import datetime
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import model_config as _model_config
from .constants import SESSION_SUBSCRIPTION_SENTINEL, SUBSCRIPTION_BACKENDS

logger = logging.getLogger(__name__)

DEFAULT_STATS_DIR = Path.home() / '.anthproxy' / 'stats'
_LEGACY_STATS_FILE = Path.home() / '.anthproxy' / 'stats.jsonl'
_MAX_PARTITION_DAYS = 92
_HISTORICAL_PERIOD_RE = re.compile(r'^-([1-9]\d{0,5})([dmwq])$')

PeriodBucket = Literal['hour', 'day', 'week', 'month']


@dataclass(frozen=True)
class StatsPeriod:
    token: str
    start_ts: float
    end_ts: float
    label: str
    bucket: PeriodBucket
    backend: str | None = None


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# tier -> (input, output, cache_read, cache_write) USD per million tokens
#   cache_read  = 0.1  × input rate  (Anthropic standard prompt-cache pricing)
#   cache_write = 1.25 × input rate
# Tables are driven by ~/.anthproxy/config.json (see anthproxy/model_config.py).
MODEL_PRICING = _model_config.model_pricing()

_MODEL_LABEL = _model_config.model_labels()


def _classify_model(model: str) -> str:
    """Return 'fable', 'opus', 'sonnet', 'haiku', or 'other' for an Anthropic model string."""
    m = (model or '').lower()
    for tier in ('fable', 'opus', 'sonnet', 'haiku'):
        if tier in m:
            return tier
    return 'other'


def _record_cost(r: dict) -> float:
    """Return approximate USD cost for one stats record."""
    price = MODEL_PRICING.get(_classify_model(r.get('model', '')))
    if price is None:
        return 0.0
    in_p, out_p, cr_p, cw_p = price
    return (
        r.get('input_tokens', 0) * in_p
        + r.get('output_tokens', 0) * out_p
        + r.get('cache_read_tokens', 0) * cr_p
        + r.get('cache_creation_tokens', 0) * cw_p
    ) / 1_000_000


@dataclass(frozen=True)
class RoutingEconomics:
    """Per-request routing cost breakdown returned by :func:`routing_economics`."""

    pricing_available: bool
    opus_baseline_cost: float = 0.0
    routed_cost: float = 0.0
    classifier_overhead_usd: float = 0.0
    net_savings_usd: float = 0.0


def routing_economics(
    routed_model: str,
    classifier_model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    classifier_input_tokens: int = 0,
    classifier_output_tokens: int = 0,
) -> RoutingEconomics:
    """Compute per-request routing cost breakdown against the opus baseline.

    Uses the same pricing tables as :func:`_record_cost` (sourced from
    :mod:`model_config`).  Never raises.

    Returns a :class:`RoutingEconomics` with:

    ``pricing_available``
        ``False`` when pricing is unavailable for the opus baseline or the
        routed model; all numeric fields are ``0.0`` in that case.
    ``opus_baseline_cost``
        Cost if the request had been served on the ``opus`` tier — the fixed
        baseline every routed request is measured against.
    ``routed_cost``
        Actual cost on the routed (serving) model.
    ``classifier_overhead_usd``
        Cost of the classifier call; ``0.0`` when
        ``classifier_input_tokens`` and ``classifier_output_tokens`` are both
        ``0``, or when classifier model pricing is unavailable.
    ``net_savings_usd``
        ``opus_baseline_cost - routed_cost - classifier_overhead_usd``.
    """
    routed_tier = _classify_model(routed_model or '')

    opus_price = MODEL_PRICING.get('opus')
    routed_price = MODEL_PRICING.get(routed_tier)
    if opus_price is None or routed_price is None:
        return RoutingEconomics(pricing_available=False)

    inp = int(input_tokens or 0)
    out = int(output_tokens or 0)
    cc = int(cache_creation_tokens or 0)
    cr = int(cache_read_tokens or 0)

    def _cost(price: tuple) -> float:
        in_p, out_p, cr_p, cw_p = price
        return (inp * in_p + out * out_p + cr * cr_p + cc * cw_p) / 1_000_000

    opus_baseline_cost = _cost(opus_price)
    routed_cost = _cost(routed_price)

    classifier_overhead_usd = 0.0
    if classifier_input_tokens or classifier_output_tokens:
        clf_tier = _classify_model(classifier_model or '')
        clf_price = MODEL_PRICING.get(clf_tier)
        if clf_price is not None:
            clf_in_p, clf_out_p, _cr_p, _cw_p = clf_price
            classifier_overhead_usd = (
                int(classifier_input_tokens or 0) * clf_in_p
                + int(classifier_output_tokens or 0) * clf_out_p
            ) / 1_000_000

    return RoutingEconomics(
        pricing_available=True,
        opus_baseline_cost=opus_baseline_cost,
        routed_cost=routed_cost,
        classifier_overhead_usd=classifier_overhead_usd,
        net_savings_usd=opus_baseline_cost - routed_cost - classifier_overhead_usd,
    )


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def _local_midnight(day: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(day, datetime.time.min)


def _month_start(dt: datetime.datetime) -> datetime.datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _quarter_start(dt: datetime.datetime) -> datetime.datetime:
    month = ((dt.month - 1) // 3) * 3 + 1
    return dt.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def _week_start(dt: datetime.datetime) -> datetime.datetime:
    return _local_midnight(dt.date() - datetime.timedelta(days=dt.weekday()))


def _shift_month_start(start: datetime.datetime, months: int) -> datetime.datetime:
    month_index = (start.year - 1) * 12 + (start.month - 1) + months
    year, month0 = divmod(month_index, 12)
    year += 1
    if year < 1 or year > 9999:
        raise ValueError('month shift out of range')
    return start.replace(year=year, month=month0 + 1, day=1)


def _make_period(
    token: str,
    start: datetime.datetime,
    end: datetime.datetime,
    label: str,
    bucket: PeriodBucket,
) -> StatsPeriod:
    return StatsPeriod(
        token=token,
        start_ts=start.timestamp(),
        end_ts=end.timestamp(),
        label=label,
        bucket=bucket,
    )


def _today_period(now: datetime.datetime) -> StatsPeriod:
    start = _local_midnight(now.date())
    end = _local_midnight(now.date() + datetime.timedelta(days=1))
    return _make_period('1d', start, end, 'today', 'hour')


def resolve_stats_period(token: str, *, now: datetime.datetime | None = None) -> StatsPeriod:
    """Resolve a stats selector into a bounded local-calendar period."""
    now = now or datetime.datetime.now()
    token = (token or '').strip()

    try:
        if token in ('', '1d', 'd'):
            return _today_period(now)

        if token in ('1w', 'w'):
            start = _week_start(now)
            end = start + datetime.timedelta(days=7)
            return _make_period('1w', start, end, 'this week', 'day')

        if token in ('1m', 'm'):
            start = _month_start(now)
            end = _shift_month_start(start, 1)
            return _make_period('1m', start, end, 'this month', 'week')

        if token in ('1q', 'q'):
            start = _quarter_start(now)
            end = _shift_month_start(start, 3)
            return _make_period('1q', start, end, 'this quarter', 'month')

        match = _HISTORICAL_PERIOD_RE.fullmatch(token)
        if match is not None:
            offset = int(match.group(1))
            unit = match.group(2)

            if unit == 'd':
                day = now.date() - datetime.timedelta(days=offset)
                start = _local_midnight(day)
                end = _local_midnight(day + datetime.timedelta(days=1))
                if offset == 1:
                    label = f'yesterday ({day.isoformat()})'
                else:
                    label = f'{offset} days ago ({day.isoformat()})'
                return _make_period(token, start, end, label, 'hour')

            if unit == 'w':
                start = _week_start(now) - datetime.timedelta(days=7 * offset)
                end = start + datetime.timedelta(days=7)
                week_of = start.date().isoformat()
                if offset == 1:
                    label = f'last week (week of {week_of})'
                else:
                    label = f'{offset} weeks ago (week of {week_of})'
                return _make_period(token, start, end, label, 'day')

            if unit == 'm':
                month_start = _month_start(now)
                start = _shift_month_start(month_start, -offset)
                end = _shift_month_start(start, 1)
                month_label = start.strftime('%B %Y')
                if offset == 1:
                    label = f'last month ({month_label})'
                else:
                    label = f'{offset} months ago ({month_label})'
                return _make_period(token, start, end, label, 'week')

            # unit == 'q'
            start = _shift_month_start(_quarter_start(now), -3 * offset)
            end = _shift_month_start(start, 3)
            q_num = (start.month - 1) // 3 + 1
            if offset == 1:
                label = f'last quarter (Q{q_num} {start.year})'
            else:
                label = f'{offset} quarters ago (Q{q_num} {start.year})'
            return _make_period(token, start, end, label, 'month')
    except (OverflowError, OSError, ValueError):
        pass

    return _today_period(now)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class StatsCollector:
    """Thread-safe JSONL stats writer/reader backed by per-day files."""

    def __init__(self, stats_dir: str | Path | None = None):
        self._dir = Path(stats_dir) if stats_dir else DEFAULT_STATS_DIR
        self._lock = threading.Lock()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning('Stats: could not create directory %s: %s', self._dir, exc)

    def _file_for(self, ts: float) -> Path:
        """Return the day-file path for the given POSIX timestamp."""
        day = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        return self._dir / f'{day}.jsonl'

    def record(
        self,
        backend: str,
        *,
        model: str = '',
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        duration_ms: int = 0,
        streaming: bool = False,
        # Observability fields (all keyword-only with safe defaults for back-compat).
        status: str = 'success',
        status_code: int | None = None,
        error: str | None = None,
        requested_model: str = '',
        classification: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        """Append one stats record. Never raises — silently drops on I/O error."""
        ts = time.time()
        resolved_model = str(model or '')
        entry = {
            'ts': ts,
            'backend': backend,
            'model': resolved_model,
            'input_tokens': int(input_tokens or 0),
            'output_tokens': int(output_tokens or 0),
            'cache_creation_tokens': int(cache_creation_tokens or 0),
            'cache_read_tokens': int(cache_read_tokens or 0),
            'duration_ms': int(duration_ms or 0),
            'streaming': bool(streaming),
            'status': str(status or 'success'),
            'status_code': int(status_code) if status_code is not None else None,
            'error': str(error) if error is not None else None,
            # If requested_model not supplied, default to routed model so
            # "requested != model" reliably indicates routing changed the tier.
            'requested_model': str(requested_model) if requested_model else resolved_model,
            'classification': str(classification) if classification is not None else None,
            'reason_code': str(reason_code) if reason_code is not None else None,
        }
        line = json.dumps(entry) + '\n'
        target = self._file_for(ts)
        with self._lock:
            try:
                with open(target, 'a', encoding='utf-8') as fh:
                    fh.write(line)
            except OSError as exc:
                logger.debug('Stats: write failed: %s', exc)

    def read_records(self, start_ts: float, end_ts: float) -> list[dict]:
        """Return all records with start_ts <= ts < end_ts, oldest first."""
        if not isinstance(start_ts, (int, float)) or not math.isfinite(start_ts):
            raise ValueError('start_ts must be finite')
        if not isinstance(end_ts, (int, float)) or not math.isfinite(end_ts):
            raise ValueError('end_ts must be finite')
        if end_ts <= start_ts:
            raise ValueError('end_ts must be greater than start_ts')

        start_dt = datetime.datetime.fromtimestamp(start_ts)
        end_prev_dt = datetime.datetime.fromtimestamp(math.nextafter(end_ts, -math.inf))
        start_date = start_dt.date()
        end_date = end_prev_dt.date()
        if end_date < start_date:
            raise ValueError('stats range resolves to no partitions')
        if (end_date - start_date).days + 1 > _MAX_PARTITION_DAYS:
            raise ValueError('stats range spans too many partitions')

        today = datetime.datetime.now().date()
        last_date = min(end_date, today)

        records: list[dict] = []
        lines_per_file: list[tuple[Path, list[str]]] = []

        with self._lock:
            cur = start_date
            while cur <= last_date:
                path = self._dir / f'{cur.isoformat()}.jsonl'
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        lines_per_file.append((path, fh.readlines()))
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.debug('Stats: read failed %s: %s', path, exc)
                cur += datetime.timedelta(days=1)

            if _LEGACY_STATS_FILE.exists():
                try:
                    with open(_LEGACY_STATS_FILE, 'r', encoding='utf-8') as fh:
                        lines_per_file.append((_LEGACY_STATS_FILE, fh.readlines()))
                except OSError as exc:
                    logger.debug('Stats: legacy read failed: %s', exc)

        for _path, lines in lines_per_file:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                try:
                    record_ts = float(record.get('ts', 0))
                except (TypeError, ValueError):
                    continue
                if start_ts <= record_ts < end_ts:
                    records.append(record)

        records.sort(key=lambda r: float(r.get('ts', 0)))
        return records


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _fmt_tokens(n: int) -> str:
    """Format token count with thousands separator."""
    return f'{n:,}'


def _fmt_duration(ms: int) -> str:
    """Format milliseconds as human-readable string."""
    if ms < 1000:
        return f'{ms}ms'
    if ms < 60_000:
        return f'{ms / 1000:.1f}s'
    minutes = ms // 60_000
    secs = (ms % 60_000) / 1000
    return f'{minutes}m {secs:.0f}s'


def _fmt_active_time(total_ms: int) -> str:
    """Format total active time in minutes."""
    minutes = total_ms / 60_000
    if minutes < 1:
        return f'{total_ms / 1000:.1f}s'
    return f'{minutes:.1f} min'


def _fmt_cost(d: float) -> str:
    """Format a USD dollar amount."""
    if d <= 0:
        return '$0.00'
    if d < 0.01:
        return f'${d:.4f}'
    return f'${d:,.2f}'


def _bucket_key_label(ts: float, bucket: PeriodBucket) -> tuple[str, str]:
    """Return (sort_key, display_label) for a timestamp and bucket kind."""
    dt = datetime.datetime.fromtimestamp(ts)
    if bucket == 'day':
        return dt.strftime('%Y-%m-%d'), dt.strftime('%a %Y-%m-%d')
    if bucket == 'week':
        monday = (dt - datetime.timedelta(days=dt.weekday())).date()
        iso = monday.isoformat()
        return iso, f'Week of {iso}'
    if bucket == 'month':
        return dt.strftime('%Y-%m'), dt.strftime('%B %Y')
    return dt.strftime('%Y-%m-%d %H'), dt.strftime('%H:00')


def _matches_backend_filter(record_backend: str, backend_filter: str) -> bool:
    """Return True when record_backend satisfies backend_filter.

    'subscription' matches any backend in SUBSCRIPTION_BACKENDS; any other
    value is an exact match.
    """
    if backend_filter == SESSION_SUBSCRIPTION_SENTINEL:
        return record_backend in SUBSCRIPTION_BACKENDS
    return record_backend == backend_filter


def format_stats_markdown(records: list[dict], period: StatsPeriod) -> str:
    """Format stats records as Markdown for the given period."""
    if period.backend:
        records = [r for r in records if _matches_backend_filter(r.get('backend', 'unknown'), period.backend)]
    heading_label = f'{period.label} · {period.backend}' if period.backend else period.label

    if not records:
        return f'## anthproxy stats — {heading_label}\n\nNo requests recorded.'

    agg: dict[str, dict] = {}

    for r in records:
        bkey, blabel = _bucket_key_label(r.get('ts', 0), period.bucket)
        if bkey not in agg:
            agg[bkey] = {'label': blabel, 'rows': {}}
        backend = r.get('backend', 'unknown')
        tier = _classify_model(r.get('model', ''))
        row_key = (backend, tier)
        rows = agg[bkey]['rows']
        if row_key not in rows:
            rows[row_key] = {
                'requests': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_read_tokens': 0,
                'cache_creation_tokens': 0,
                'duration_ms': 0,
                'cost': 0.0,
            }
        acc = rows[row_key]
        acc['requests'] += 1
        acc['input_tokens'] += r.get('input_tokens', 0)
        acc['output_tokens'] += r.get('output_tokens', 0)
        acc['cache_read_tokens'] += r.get('cache_read_tokens', 0)
        acc['cache_creation_tokens'] += r.get('cache_creation_tokens', 0)
        acc['duration_ms'] += r.get('duration_ms', 0)
        acc['cost'] += _record_cost(r)

    lines = [f'## anthproxy stats — {heading_label}\n']

    grand = {
        'requests': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_read_tokens': 0,
        'cache_creation_tokens': 0,
        'duration_ms': 0,
        'cost': 0.0,
    }

    for bkey in sorted(agg):
        bucket = agg[bkey]
        blabel = bucket['label']
        rows = bucket['rows']

        sub = {k: 0 for k in grand}
        sub['cost'] = 0.0

        lines.append(f'\n### {blabel}\n')
        lines.append('| Backend | Model | Requests | Input | Output | Cache Read | Cache Created | Cost |')
        lines.append('|---|---|---|---|---|---|---|---|')

        for (backend, tier) in sorted(rows):
            acc = rows[(backend, tier)]
            lines.append(
                f'| {backend} '
                f'| {_MODEL_LABEL.get(tier, tier)} '
                f'| {acc["requests"]} '
                f'| {_fmt_tokens(acc["input_tokens"])} '
                f'| {_fmt_tokens(acc["output_tokens"])} '
                f'| {_fmt_tokens(acc["cache_read_tokens"])} '
                f'| {_fmt_tokens(acc["cache_creation_tokens"])} '
                f'| {_fmt_cost(acc["cost"])} |'
            )
            for key in sub:
                sub[key] += acc[key]  # type: ignore[operator]

        lines.append(
            f'| **Subtotal** | | **{sub["requests"]}** '
            f'| **{_fmt_tokens(sub["input_tokens"])}** '
            f'| **{_fmt_tokens(sub["output_tokens"])}** '
            f'| **{_fmt_tokens(sub["cache_read_tokens"])}** '
            f'| **{_fmt_tokens(sub["cache_creation_tokens"])}** '
            f'| **{_fmt_cost(sub["cost"])}** |'
        )
        lines.append(f'\n_{_fmt_active_time(sub["duration_ms"])} active_')

        for key in grand:
            grand[key] += sub[key]  # type: ignore[operator]

    lines.append(
        f'\n---\n**Total — {grand["requests"]} requests · '
        f'{_fmt_tokens(grand["input_tokens"])} in · '
        f'{_fmt_tokens(grand["output_tokens"])} out · '
        f'**{_fmt_cost(grand["cost"])}** · '
        f'{_fmt_active_time(grand["duration_ms"])} active**'
    )

    return '\n'.join(lines)
