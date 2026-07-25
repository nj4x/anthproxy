import http.client
import json
import logging
import socket
import ssl
import threading
import time

from .._shared import Backend, FiveHourStatus, SubscriptionBackend, UsageRateLimitError
from .._shared.http_util import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    format_reset as _format_reset_time,
    handle_error_response as _handle_error_response_shared,
    make_connection as _make_connection_shared,
    parse_retry_after as _parse_retry_after,
    read_sse_lines as _read_sse_lines,
    retry_delay as _retry_delay,
    should_retry as _should_retry,
)
from ..config import Config
from ..mapper import AnthropicRequestError, estimate_input_tokens, strip_all_thinking_blocks
from .auth import force_refresh, get_access
from .mapper import _build_body, _merge_betas, _resolve_model

logger = logging.getLogger(__name__)

ANTHROPIC_HOST = 'api.anthropic.com'
MESSAGES_PATH = '/v1/messages?beta=true'
COUNT_TOKENS_PATH = '/v1/messages/count_tokens?beta=true'
USAGE_PATH = '/api/oauth/usage'
ANTHROPIC_VERSION = '2023-06-01'
CLAUDE_CLI_VERSION = '2.1.88'
USER_AGENT = f'claude-cli/{CLAUDE_CLI_VERSION} (external, cli)'
USAGE_TIMEOUT_SECONDS = 5


def _make_connection() -> http.client.HTTPSConnection:
    return _make_connection_shared(ANTHROPIC_HOST)


def _request_headers(access_token: str, betas: str, stream: bool) -> dict:
    return {
        'Authorization': f'Bearer {access_token}',
        'anthropic-version': ANTHROPIC_VERSION,
        'anthropic-beta': betas,
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream' if stream else 'application/json',
        'anthropic-dangerous-direct-browser-access': 'true',
        'x-app': 'cli',
        'User-Agent': USER_AGENT,
    }


def _handle_error_response(status: int, body_bytes: bytes):
    _handle_error_response_shared(status, body_bytes, provider_name='Anthropic',
                                  forward_error_type=True)


def _send_with_retries(payload: dict, config, lock: threading.Lock, stream: bool,
                       path: str = MESSAGES_PATH, method: str = 'POST',
                       body_bytes: bytes | None = None) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    if body_bytes is None:
        body_bytes = _build_body(payload)
    betas = _merge_betas(payload)

    access_token, _account_uuid = get_access(config, lock)
    auth_refreshed = False
    thinking_stripped = False

    for attempt in range(MAX_RETRIES + 1):
        conn = _make_connection()
        try:
            headers = _request_headers(access_token, betas, stream)
            conn.request(method, path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            conn.close()
            if attempt < MAX_RETRIES:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    'Anthropic request failed (network, attempt %d/%d): %s — retrying in %.1fs',
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue
            raise AnthropicRequestError(
                f'Anthropic connection error after {MAX_RETRIES} retries: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc

        if resp.status == 200:
            return conn, resp

        if resp.status == 401 and not auth_refreshed:
            resp.read()
            conn.close()
            logger.info('Anthropic returned 401 — forcing token refresh before retry.')
            try:
                access_token, _account_uuid = force_refresh(config, lock)
            except AnthropicRequestError:
                raise
            auth_refreshed = True
            continue

        if _should_retry(resp.status, resp) and attempt < MAX_RETRIES:
            delay = _retry_delay(resp, attempt)
            resp.read()
            conn.close()
            logger.warning(
                'Anthropic HTTP %d (attempt %d/%d) — retrying in %.1fs',
                resp.status, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue

        resp_body = resp.read()
        conn.close()

        # Anthropic thinking/redacted_thinking blocks are model-specific: an
        # ``opus``-generated block is invalid for ``sonnet``/``haiku`` and vice
        # versa.  After a model-tier routing switch the history may contain
        # blocks from a different model; strip_codex_thinking_blocks() only
        # removes foreign-backend markers (or:/codexenc:).  Detect both error
        # forms ("Invalid `signature` in `thinking` block" and "Invalid `data`
        # in `redacted_thinking` block") and recover by stripping ALL thinking
        # and redacted_thinking blocks, then retrying once.
        if (not thinking_stripped and resp.status == 400
                and b'thinking' in resp_body
                and (b'signature' in resp_body or b'redacted_thinking' in resp_body)):
            messages = payload.get('messages')
            if isinstance(messages, list):
                stripped_msgs = strip_all_thinking_blocks(messages)
                if stripped_msgs is not messages:
                    body_bytes = _build_body({**payload, 'messages': stripped_msgs})
                    thinking_stripped = True
                    logger.warning(
                        'Retrying after thinking-block 400 with all thinking/'
                        'redacted_thinking blocks stripped from history',
                    )
                    continue

        _handle_error_response(resp.status, resp_body)

    raise AnthropicRequestError('Anthropic request failed', error_type='api_error', status_code=502)


def _fetch_usage(config, lock: threading.Lock) -> dict:
    access_token, _account_uuid = get_access(config, lock)
    refreshed = False

    while True:
        conn = http.client.HTTPSConnection(ANTHROPIC_HOST, timeout=USAGE_TIMEOUT_SECONDS)
        try:
            conn.request('GET', USAGE_PATH, headers={
                'Authorization': f'Bearer {access_token}',
                'anthropic-version': ANTHROPIC_VERSION,
                'anthropic-beta': 'oauth-2025-04-20',
                'Accept': 'application/json',
                'User-Agent': USER_AGENT,
            })
            resp = conn.getresponse()
            body = resp.read()
            status = resp.status
        finally:
            conn.close()

        if status == 200:
            try:
                return json.loads(body)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f'Unrecognized usage response from Anthropic: {exc}') from exc

        if status == 401 and not refreshed:
            access_token, _account_uuid = force_refresh(config, lock)
            refreshed = True
            continue
        if status == 429:
            raise UsageRateLimitError(retry_after=_parse_retry_after(resp))
        if status in (401, 403):
            raise RuntimeError('Authentication failed for usage endpoint; re-login may be required')
        raise RuntimeError(f'Anthropic usage endpoint returned HTTP {status}')


def _format_usage_markdown(data: dict) -> str:
    lines = ['## `Anthropic` subscription usage']
    limit_reached = False

    def add_window(label: str, window: dict | None, track_limit: bool = False):
        nonlocal limit_reached
        if not isinstance(window, dict):
            return
        utilization = window.get('utilization')
        resets_at = window.get('resets_at')
        if utilization is not None:
            try:
                pct = float(utilization)
                remaining = max(0.0, 100.0 - pct)
                used_str = f'{pct:.1f}'.rstrip('0').rstrip('.')
                remaining_str = f'{remaining:.1f}'.rstrip('0').rstrip('.')
                lines.append(f'\n**{label}:** {used_str}% used · {remaining_str}% remaining')
                if track_limit and pct >= 100.0:
                    limit_reached = True
            except (TypeError, ValueError):
                lines.append(f'\n**{label}:** {utilization}')
        if resets_at:
            lines.append(f'Resets: {_format_reset_time(resets_at)}')

    add_window('5-hour usage', data.get('five_hour'), track_limit=True)
    add_window('Weekly usage', data.get('seven_day'))
    add_window('Weekly Sonnet usage', data.get('seven_day_sonnet'))
    add_window('Weekly Opus usage', data.get('seven_day_opus'))

    if limit_reached:
        lines.extend(['', '**Limit reached:** yes'])
    return '\n'.join(lines)


def _usage_failure_markdown(message: str) -> str:
    return f'## Anthropic subscription usage\n\nUsage information is unavailable: {message}'


_WEEKLY_KEYS = ('seven_day', 'seven_day_sonnet', 'seven_day_opus')


def _max_weekly_utilization(usage: dict) -> float | None:
    """Highest utilization across all weekly windows present in the payload.

    Anthropic exposes up to three weekly windows (overall, Sonnet, Opus). The
    binding weekly constraint is whichever is closest to its cap, so the
    selector should react to the max. Returns None when no window has a
    numeric utilization.
    """
    values = []
    for key in _WEEKLY_KEYS:
        window = usage.get(key)
        if isinstance(window, dict):
            util = window.get('utilization')
            if util is not None:
                try:
                    values.append(float(util))
                except (TypeError, ValueError):
                    pass
    return max(values) if values else None


class AnthropicBackend(SubscriptionBackend, Backend):
    _PROVIDER_NAME = 'Anthropic'

    def __init__(self):
        self._lock = threading.Lock()
        super().__init__()

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

        window = usage.get('five_hour') if isinstance(usage, dict) else None
        if not isinstance(window, dict):
            return FiveHourStatus(available=None, resets_at=None)

        utilization = window.get('utilization')
        resets_at_str = window.get('resets_at')

        resets_at: float | None = None
        if resets_at_str:
            try:
                import datetime as _dt
                dt = _dt.datetime.fromisoformat(resets_at_str.replace('Z', '+00:00'))
                resets_at = dt.timestamp()
            except Exception:
                pass

        weekly_utilization = _max_weekly_utilization(usage) if isinstance(usage, dict) else None

        try:
            pct = float(utilization)
            available = pct < 100.0
        except (TypeError, ValueError):
            return FiveHourStatus(available=None, resets_at=resets_at,
                                  weekly_utilization=weekly_utilization)

        return FiveHourStatus(available=available, resets_at=resets_at,
                              utilization=pct, weekly_utilization=weekly_utilization)

    def parse_credentials(self, api_key: str) -> dict:
        return {}

    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        stream = bool(payload.get('stream'))
        conn, resp = _send_with_retries(payload, config, self._lock, stream=stream)
        try:
            body = resp.read()
        finally:
            conn.close()

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnthropicRequestError(
                f'Upstream returned non-JSON response: {body[:200]!r}',
                error_type='api_error',
                status_code=502,
            ) from exc

        logger.debug('<<< Anthropic response: model=%s stop=%s',
                     payload.get('model', ''), result.get('stop_reason'))
        return result

    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        conn, resp = _send_with_retries(payload, config, self._lock, stream=True)
        try:
            pending = ''
            for line in _read_sse_lines(resp):
                pending += line + '\n'
                if line.strip() == '':
                    if pending.strip():
                        yield pending
                    pending = ''
            if pending.strip():
                yield pending + '\n'
        finally:
            conn.close()

    def count_tokens(self, payload: dict, credentials: dict, config: Config) -> dict:
        try:
            body_bytes = _build_body(payload)
            betas = _merge_betas(payload)
            access_token, _ = get_access(config, self._lock)
            conn = _make_connection()
            try:
                conn.request('POST', COUNT_TOKENS_PATH, body=body_bytes, headers=_request_headers(
                    access_token, betas, stream=False))
                resp = conn.getresponse()
                resp_body = resp.read()
            finally:
                conn.close()

            if resp.status == 200:
                result = json.loads(resp_body)
                return result

            logger.warning('count_tokens upstream returned HTTP %d — falling back to estimate',
                           resp.status)
        except Exception as exc:
            logger.warning('count_tokens upstream call failed: %s — using estimate', exc)

        estimated = estimate_input_tokens(payload)
        return {
            'input_tokens': estimated,
            'model': _resolve_model(payload.get('model', '')),
        }
