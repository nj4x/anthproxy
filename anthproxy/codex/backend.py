import http.client
import json
import logging
import math
import socket
import ssl
import threading
import time

from .._shared import Backend, FiveHourStatus, SubscriptionBackend, UsageRateLimitError
from .._shared.http_util import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    format_reset as _format_reset,
    handle_error_response as _handle_error_response_shared,
    make_connection as _make_connection_shared,
    parse_retry_after as _parse_retry_after,
    read_sse_lines as _read_sse_lines,
    retry_delay as _retry_delay,
    should_retry as _should_retry,
)
from ..config import Config
from ..mapper import AnthropicRequestError, estimate_input_tokens
from .auth import force_refresh, get_access
from .mapper import (
    _iter_stream_as_anthropic_sse,
    _map_request,
    _map_response,
    _raise_response_failed,
    _resolve_model,
)

logger = logging.getLogger(__name__)

CODEX_HOST = 'chatgpt.com'
CODEX_PATH = '/backend-api/codex/responses'
CODEX_USAGE_PATH = '/backend-api/wham/usage'
CODEX_CLI_VERSION = '0.144.1'
CODEX_ORIGINATOR = 'codex_cli_rs'
USAGE_TIMEOUT_SECONDS = 3
WEEKLY_RESET_GAP_SECONDS = 3 * 24 * 60 * 60
WEEKLY_WINDOW_SECONDS = 168 * 60 * 60


def _make_connection() -> http.client.HTTPSConnection:
    return _make_connection_shared(CODEX_HOST)


def _codex_identity_headers(access_token: str, account_id: str | None) -> dict:
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': f'codex_cli_rs/{CODEX_CLI_VERSION} (darwin; aarch64)',
        'originator': CODEX_ORIGINATOR,
        'version': CODEX_CLI_VERSION,
    }
    if account_id:
        headers['ChatGPT-Account-ID'] = account_id
    return headers


def _codex_headers(access_token: str, account_id: str | None) -> dict:
    return {
        **_codex_identity_headers(access_token, account_id),
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
    }


def _usage_headers(access_token: str, account_id: str | None) -> dict:
    return {
        **_codex_identity_headers(access_token, account_id),
        'Accept': 'application/json',
    }


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_usage_window(raw, fetched_at: float) -> dict | None:
    if not isinstance(raw, dict):
        return None
    used = _finite_number(raw.get('used_percent'))
    if used is None:
        return None
    used = min(100.0, max(0.0, used))
    window_seconds = _finite_number(raw.get('limit_window_seconds'))
    reset_at = _finite_number(raw.get('reset_at'))
    if reset_at is None:
        reset_after = _finite_number(raw.get('reset_after_seconds'))
        if reset_after is not None and reset_after >= 0:
            reset_at = fetched_at + reset_after
    return {
        'used_percent': used,
        'remaining_percent': 100.0 - used,
        'window_seconds': window_seconds,
        'reset_at': reset_at,
    }


def _normalize_usage(data: dict, fetched_at: float | None = None) -> dict:
    if not isinstance(data, dict):
        raise ValueError('usage response is not a JSON object')
    rate_limit = data.get('rate_limit')
    if not isinstance(rate_limit, dict):
        raise ValueError('usage response has no rate_limit object')

    fetched_at = time.time() if fetched_at is None else fetched_at
    primary = _normalize_usage_window(rate_limit.get('primary_window'), fetched_at)
    secondary = _normalize_usage_window(rate_limit.get('secondary_window'), fetched_at)
    if primary is None and secondary is None:
        raise ValueError('usage response has no recognizable limit windows')

    weekly = None
    if secondary is not None:
        seconds = secondary.get('window_seconds')
        primary_reset = primary.get('reset_at') if primary else None
        secondary_reset = secondary.get('reset_at')
        is_week_duration = seconds is not None and seconds >= WEEKLY_WINDOW_SECONDS
        has_week_cadence = (
            primary_reset is not None
            and secondary_reset is not None
            and secondary_reset - primary_reset >= WEEKLY_RESET_GAP_SECONDS
        )
        if is_week_duration or has_week_cadence:
            weekly = secondary

    if weekly is None and primary is not None:
        seconds = primary.get('window_seconds')
        if seconds is not None and seconds >= WEEKLY_WINDOW_SECONDS:
            weekly = primary

    credits = data.get('credits')
    balance = _finite_number(credits.get('balance')) if isinstance(credits, dict) else None
    return {
        'plan_type': data.get('plan_type') if isinstance(data.get('plan_type'), str) else None,
        'credit_balance': balance,
        'limit_reached': rate_limit.get('limit_reached') is True,
        'primary': primary,
        'weekly': weekly,
        'fetched_at': fetched_at,
    }


def _format_percent(value: float) -> str:
    return f'{value:.1f}'.rstrip('0').rstrip('.')


def _format_usage_markdown(usage: dict) -> str:
    lines = ['## `Codex` subscription usage']
    plan = usage.get('plan_type')
    balance = usage.get('credit_balance')
    if plan:
        lines.append(f'\n**Plan:** {plan}')
    if balance is not None:
        lines.append(f'**Credits:** ${balance:.2f}')

    def add_window(label: str, window: dict | None):
        if window is None:
            lines.extend(['', f'**{label}:** unavailable'])
            return
        used = _format_percent(window['used_percent'])
        remaining = _format_percent(window['remaining_percent'])
        lines.extend([
            '',
            f'**{label}:** {used}% used · {remaining}% remaining',
            f'Resets: {_format_reset(window.get("reset_at"))}',
        ])

    add_window('5-hour usage', usage.get('primary'))
    add_window('Weekly usage', usage.get('weekly'))
    if usage.get('limit_reached'):
        lines.extend(['', '**Limit reached:** yes'])
    return '\n'.join(lines)


def _usage_failure_markdown(message: str) -> str:
    return f'## Codex subscription usage\n\nUsage information is unavailable: {message}'


def _fetch_usage(config, lock: threading.Lock) -> dict:
    access_token, account_id = get_access(config, lock)
    refreshed = False

    while True:
        conn = http.client.HTTPSConnection(CODEX_HOST, timeout=USAGE_TIMEOUT_SECONDS)
        try:
            conn.request('GET', CODEX_USAGE_PATH, headers=_usage_headers(access_token, account_id))
            resp = conn.getresponse()
            body = resp.read()
            status = resp.status
        finally:
            conn.close()

        if status == 200:
            try:
                return _normalize_usage(json.loads(body))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f'unrecognized response from ChatGPT ({exc})') from exc

        if status == 401 and not refreshed:
            access_token, account_id = force_refresh(config, lock)
            refreshed = True
            continue
        if status == 429:
            raise UsageRateLimitError(retry_after=_parse_retry_after(resp))
        if status in (401, 403):
            raise RuntimeError('authentication failed; re-login with the Codex CLI may be required')
        raise RuntimeError(f'ChatGPT usage service returned HTTP {status}')


def _handle_error_response(status: int, body_bytes: bytes):
    _handle_error_response_shared(status, body_bytes, provider_name='Codex')


def _codex_error_message(body_bytes: bytes) -> str:
    try:
        detail = json.loads(body_bytes)
    except (json.JSONDecodeError, TypeError):
        return body_bytes.decode('utf-8', errors='replace')[:500]

    if isinstance(detail, dict):
        error = detail.get('error')
        if isinstance(error, dict) and error.get('message'):
            return str(error['message'])
        for key in ('detail', 'message'):
            if detail.get(key):
                return str(detail[key])
    return str(detail)


def _is_chatgpt_unsupported_model_error(status: int, body_bytes: bytes) -> bool:
    return (
        status == 400
        and 'is not supported when using Codex with a ChatGPT account'
        in _codex_error_message(body_bytes)
    )


def _codex_unsupported_model_fallback(config) -> str:
    value = getattr(config, 'codex_unsupported_model_fallback', '')
    return value if isinstance(value, str) else ''


def _drain_sse_to_response(response, requested_model: str) -> dict:
    output_items: list[dict] = []
    usage: dict = {}
    status = 'completed'

    for raw_line in _read_sse_lines(response):
        line = raw_line.strip()
        if not line or not line.startswith('data: '):
            continue
        data = line[6:]
        if data.strip() in ('[DONE]', ''):
            continue

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        etype = event.get('type', '')

        if etype == 'response.output_item.done':
            item = event.get('item', {})
            if item:
                output_items.append(item)

        elif etype in ('response.completed', 'response.done', 'response.incomplete'):
            resp = event.get('response', {}) or {}
            usage = resp.get('usage', {}) or {}
            status = resp.get('status', 'completed')
            if etype == 'response.incomplete':
                status = 'incomplete'

        elif etype == 'response.failed':
            _raise_response_failed(event)

    return _map_response(output_items, usage, status, requested_model)


def _send_with_retries(payload: dict, config, lock: threading.Lock) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    try:
        context_limit = int(getattr(config, 'codex_context_limit', 0))
    except (TypeError, ValueError):
        context_limit = 0
    body = _map_request(payload, context_limit=context_limit)
    body['stream'] = True
    body_bytes = json.dumps(body).encode('utf-8')

    access_token, account_id = get_access(config, lock)
    auth_refreshed = False
    fallback_model = _codex_unsupported_model_fallback(config)
    fallback_used = False

    for attempt in range(MAX_RETRIES + 1):
        conn = _make_connection()
        try:
            headers = _codex_headers(access_token, account_id)
            conn.request('POST', CODEX_PATH, body=body_bytes, headers=headers)
            resp = conn.getresponse()
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            conn.close()
            if attempt < MAX_RETRIES:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    'Codex request failed (network, attempt %d/%d): %s — retrying in %.1fs',
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue
            raise AnthropicRequestError(
                f'Codex connection error after {MAX_RETRIES} retries: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc

        if resp.status == 200:
            return conn, resp

        if resp.status == 401 and not auth_refreshed:
            resp.read()
            conn.close()
            logger.info('Codex returned 401 — forcing token refresh before retry.')
            try:
                access_token, account_id = force_refresh(config, lock)
            except AnthropicRequestError:
                raise
            auth_refreshed = True
            continue

        if _should_retry(resp.status, resp) and attempt < MAX_RETRIES:
            delay = _retry_delay(resp, attempt)
            resp.read()
            conn.close()
            logger.warning(
                'Codex HTTP %d (attempt %d/%d) — retrying in %.1fs',
                resp.status, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue

        resp_body = resp.read()
        conn.close()
        if _is_chatgpt_unsupported_model_error(resp.status, resp_body):
            upstream_message = _codex_error_message(resp_body)
            if fallback_model and not fallback_used:
                fallback_body = _map_request({**payload, 'model': fallback_model}, context_limit=context_limit)
                fallback_body['stream'] = True
                body_bytes = json.dumps(fallback_body).encode('utf-8')
                fallback_used = True
                logger.warning(
                    'Codex unsupported model for ChatGPT account; retrying once with fallback %s '
                    '(requested=%s resolved=%s): %s',
                    fallback_model,
                    payload.get('model', ''),
                    fallback_body.get('model', ''),
                    upstream_message,
                )
                continue
            if not fallback_model:
                logger.warning(
                    'Codex unsupported model for ChatGPT account with fallback disabled '
                    '(requested=%s): %s',
                    payload.get('model', ''),
                    upstream_message,
                )
        _handle_error_response(resp.status, resp_body)

    raise AnthropicRequestError('Codex request failed', error_type='api_error', status_code=502)


class CodexBackend(SubscriptionBackend, Backend):
    _PROVIDER_NAME = 'Codex'

    def __init__(self):
        self._lock = threading.Lock()
        super().__init__()

    @classmethod
    def summary_credentials(cls, snapshot) -> dict:
        return snapshot.backend.parse_credentials('')

    def _fetch_usage_data(self, config) -> dict:
        return _fetch_usage(config, self._lock)

    def _format_usage_markdown_impl(self, usage: dict) -> str:
        return _format_usage_markdown(usage)

    def _usage_failure_markdown_impl(self, message: str) -> str:
        return _usage_failure_markdown(message)

    def five_hour_status(self, config: Config) -> FiveHourStatus:
        try:
            usage = self.get_usage(config)
        except AnthropicRequestError:
            return FiveHourStatus(available=False, resets_at=None)
        except Exception:
            return FiveHourStatus(available=None, resets_at=None)

        primary = usage.get('primary') if isinstance(usage, dict) else None
        weekly_window = usage.get('weekly') if isinstance(usage, dict) else None
        primary_seconds = primary.get('window_seconds') if isinstance(primary, dict) else None
        primary_is_weekly = (
            isinstance(primary_seconds, (int, float))
            and not isinstance(primary_seconds, bool)
            and primary_seconds >= WEEKLY_WINDOW_SECONDS
        )
        status_window = weekly_window if primary_is_weekly else primary
        if not isinstance(status_window, dict):
            return FiveHourStatus(available=None, resets_at=None)

        remaining = status_window.get('remaining_percent')
        limit_reached = usage.get('limit_reached', False)
        resets_at = status_window.get('reset_at')

        weekly_utilization: float | None = None
        weekly_resets_at: float | None = None
        weekly_window_hours: float | None = None
        if isinstance(weekly_window, dict):
            try:
                weekly_utilization = float(weekly_window.get('used_percent', 0))
            except (TypeError, ValueError):
                pass
            weekly_reset = weekly_window.get('reset_at')
            if isinstance(weekly_reset, (int, float)) and not isinstance(weekly_reset, bool):
                weekly_resets_at = float(weekly_reset)
            weekly_seconds = weekly_window.get('window_seconds')
            if (
                isinstance(weekly_seconds, (int, float))
                and not isinstance(weekly_seconds, bool)
                and weekly_seconds > 0
            ):
                weekly_window_hours = float(weekly_seconds) / 3600.0

        try:
            remaining_pct = float(remaining)
            available = remaining_pct > 0.0 and not limit_reached
        except (TypeError, ValueError):
            return FiveHourStatus(available=None, resets_at=resets_at,
                                  weekly_utilization=weekly_utilization,
                                  weekly_resets_at=weekly_resets_at,
                                  weekly_window_hours=weekly_window_hours)

        utilization = 100.0 - remaining_pct
        return FiveHourStatus(available=available, resets_at=resets_at,
                              utilization=utilization, weekly_utilization=weekly_utilization,
                              weekly_resets_at=weekly_resets_at,
                              weekly_window_hours=weekly_window_hours)

    def parse_credentials(self, api_key: str) -> dict:
        return {}

    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        requested_model = payload.get('model', '')
        conn, resp = _send_with_retries(payload, config, self._lock)
        try:
            result = _drain_sse_to_response(resp, requested_model)
            logger.debug('<<< Codex response: model=%s stop=%s',
                         requested_model, result.get('stop_reason'))
            return result
        finally:
            conn.close()

    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        requested_model = payload.get('model', '')
        estimated = estimate_input_tokens(payload)
        conn, resp = _send_with_retries(payload, config, self._lock)
        try:
            yield from _iter_stream_as_anthropic_sse(resp, requested_model, estimated)
        finally:
            conn.close()

    def count_tokens(self, payload: dict, credentials: dict, config: Config) -> dict:
        estimated = estimate_input_tokens(payload)
        model = _resolve_model(payload.get('model', ''))
        return {
            'input_tokens': estimated,
            'model': model,
        }
