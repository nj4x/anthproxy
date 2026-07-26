"""Shared HTTP utilities for Codex and Anthropic backends.

Contains byte-identical (or trivially parameterizable) HTTP scaffolding that
was previously duplicated between ``codex.py`` and ``anthropic.py``.

Plugin backends may define their own private copies.

Public API:
    MAX_RETRIES, RETRYABLE_STATUSES, RETRY_BASE_DELAY, RETRY_MAX_DELAY
    make_connection(host, timeout=300)
    read_sse_lines(response)
    parse_retry_after(resp)
    retry_delay(resp, attempt)
    should_retry(status, resp)
    handle_error_response(status, body_bytes, *, provider_name, forward_error_type=False)
    format_relative(delta_secs)
    format_reset(reset_at)
"""

import email.utils
import http.client
import json
import logging
import ssl
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry policy constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_BASE_DELAY = 1.0   # seconds, exponential base
RETRY_MAX_DELAY = 30.0   # seconds


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def make_connection(host: str, timeout: int = 300) -> http.client.HTTPSConnection:
    """Create an HTTPS connection with a custom SSL context."""
    ctx = ssl.create_default_context()
    return http.client.HTTPSConnection(host, timeout=timeout, context=ctx)


# ---------------------------------------------------------------------------
# SSE line reader
# ---------------------------------------------------------------------------

def read_sse_lines(response):
    """Read newline-delimited lines from an ``http.client.HTTPResponse``.

    Buffers reads in 4096-byte chunks and yields one decoded line at a time.
    A trailing non-empty fragment without a trailing newline is yielded after
    the response is exhausted (handles servers that omit the final newline).
    """
    buf = b''
    while True:
        chunk = response.read(4096)
        if not chunk:
            if buf:
                yield buf.decode('utf-8', errors='replace')
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            yield line.decode('utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Retry delay
# ---------------------------------------------------------------------------

def parse_retry_after(resp: http.client.HTTPResponse | None) -> float | None:
    """Return upstream retry guidance in seconds, if present and parseable."""
    if resp is None:
        return None

    ms_val = resp.getheader('retry-after-ms', '')
    if ms_val:
        try:
            return max(0.0, float(ms_val) / 1000.0)
        except (ValueError, TypeError):
            pass

    ra = resp.getheader('Retry-After', '')
    if ra:
        try:
            return max(0.0, float(ra))
        except (ValueError, TypeError):
            pass
        try:
            dt = email.utils.parsedate_to_datetime(ra)
            delta = dt.timestamp() - time.time()
            if delta > 0:
                return delta
        except Exception:
            pass

    return None



def retry_delay(resp: http.client.HTTPResponse | None, attempt: int) -> float:
    """Compute how long to sleep before the next retry.

    Prefers the upstream ``retry-after-ms`` (milliseconds, numeric) and
    ``Retry-After`` (seconds or HTTP-date) headers; falls back to exponential
    backoff capped at ``RETRY_MAX_DELAY``.
    """
    delay = parse_retry_after(resp)
    if delay is not None:
        return delay
    return min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)


def should_retry(status: int, resp: http.client.HTTPResponse | None) -> bool:
    """Whether an upstream error status warrants a transparent retry.

    5xx errors are always retried (transient server faults).  A 429 is retried
    ONLY when the upstream provided explicit timing guidance via
    ``Retry-After`` or ``retry-after-ms``; a 429 without guidance won't clear
    on a short exponential backoff, so it is surfaced immediately and the
    auto-selector fails over to another backend instead of stalling for ~9s.
    """
    if status not in RETRYABLE_STATUSES:
        return False
    if status == 429:
        return parse_retry_after(resp) is not None
    return True


# ---------------------------------------------------------------------------
# Error response handler
# ---------------------------------------------------------------------------

def handle_error_response(
    status: int,
    body_bytes: bytes,
    *,
    provider_name: str,
    forward_error_type: bool = False,
) -> None:
    """Map an upstream HTTP error status to an ``AnthropicRequestError``.

    Always raises — never returns.

    Args:
        status: the upstream HTTP status code.
        body_bytes: the raw response body.
        provider_name: used in log and error messages (e.g. ``'Codex'``).
        forward_error_type: when ``True``, pass through the upstream
            ``error.type`` field from the response JSON for 400 errors
            (used by the Anthropic backend to preserve native error types).
    """
    # Local import to avoid a circular dependency at module load time.
    from ..mapper import AnthropicRequestError

    try:
        detail = json.loads(body_bytes)
        error = detail.get('error') or {}
        if not isinstance(error, dict):
            error = {}
        message = error.get('message', '') or detail.get('message', '') or str(detail)
        error_type = error.get('type', '') if forward_error_type else ''
    except (json.JSONDecodeError, TypeError):
        message = body_bytes.decode('utf-8', errors='replace')[:500]
        error_type = ''

    logger.warning('%s error HTTP %d: %s', provider_name, status, message[:300])

    if status == 400:
        raise AnthropicRequestError(
            message, error_type=error_type or 'invalid_request_error', status_code=400)
    if status == 401:
        raise AnthropicRequestError(message, error_type='authentication_error', status_code=401)
    if status == 403:
        raise AnthropicRequestError(message, error_type='permission_error', status_code=403)
    if status == 429:
        raise AnthropicRequestError(message, error_type='rate_limit_error', status_code=429)
    raise AnthropicRequestError(
        f'{provider_name} error (HTTP {status}): {message}',
        error_type='api_error',
        status_code=502,
    )


# ---------------------------------------------------------------------------
# Usage formatting helpers
# ---------------------------------------------------------------------------

def format_relative(delta_secs: float) -> str:
    """Format a time delta (seconds from now) as a human-readable string.

    E.g. ``in 3 minutes``, ``in 2h 15m``, ``in 1 day``.
    """
    s = int(delta_secs)
    if s <= 0:
        return 'now'
    if s < 3600:
        m = max(1, s // 60)
        return f'in {m} minute{"s" if m != 1 else ""}'
    if s < 86400:
        h = s // 3600
        m = (s % 3600) // 60
        return f'in {h}h {m}m' if m else f'in {h} hour{"s" if h != 1 else ""}'
    d = s // 86400
    h = (s % 86400) // 3600
    return f'in {d}d {h}h' if h else f'in {d} day{"s" if d != 1 else ""}'


def format_reset(reset_at) -> str:
    """Format a reset timestamp as a human-readable local-time string.

    Accepts either a float/int Unix epoch timestamp (Codex usage response) or
    an ISO 8601 string (Anthropic usage response).  Returns ``'unknown'`` when
    ``reset_at`` is ``None`` or empty.
    """
    if reset_at is None or reset_at == '':
        return 'unknown'
    try:
        if isinstance(reset_at, (int, float)):
            ts = float(reset_at)
        else:
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(str(reset_at).replace('Z', '+00:00')).timestamp()
    except (ValueError, AttributeError, TypeError):
        return str(reset_at)  # return as-is if unparseable

    local = datetime.fromtimestamp(ts).astimezone()
    label = local.strftime('%Z') or 'local'
    offset = local.strftime('%z')
    offset_fmt = f'{offset[:3]}:{offset[3:]}' if offset else ''
    suffix = f' ({label} UTC{offset_fmt})' if offset_fmt else f' ({label})'
    absolute = local.strftime('%Y-%m-%d %H:%M') + suffix
    delta = ts - time.time()
    return f'{absolute} · {format_relative(delta)}' if delta > 0 else absolute
