"""OpenRouter backend for anthproxy.

Relays Anthropic Messages API requests to OpenRouter's Anthropic-compatible
endpoint (https://openrouter.ai/api/v1/messages).

Key properties
--------------
* HTTPS via stdlib ``http.client.HTTPSConnection``.
* Bearer-token auth from ``config.openrouter_api_key`` (``OPENROUTER_API_KEY`` env).
* No SSE translation — OpenRouter returns native Anthropic-format SSE.
* Filters the OpenAI-style ``data: [DONE]`` sentinel that OpenRouter appends.
* Credit-balance usage tracking via OpenRouter's management endpoint.
* Participates in subscription auto-selection when an API key is configured.
"""

import http.client
import json
import logging
import socket
import ssl
import time

from .._shared import Backend, FiveHourStatus, SubscriptionBackend, UsageRateLimitError
from .._shared.http_util import (
    make_connection,
    handle_error_response as _handle_error_response_shared,
    parse_retry_after as _parse_retry_after,
    read_sse_lines as _read_sse_lines,
    MAX_RETRIES,
    RETRYABLE_STATUSES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    retry_delay as _retry_delay,
)
from ..config import Config
from ..mapper import OPENROUTER_REASONING_SIG_PREFIX, AnthropicRequestError, estimate_input_tokens
from .mapper import _build_body, _request_headers, _resolve_model, _HOST, _MESSAGES_PATH

logger = logging.getLogger(__name__)

_CREDITS_PATH = '/api/v1/credits'


def _tag_thinking_signatures(response: dict) -> dict:
    """Prefix thinking-block signatures in a non-streaming response with the OpenRouter sentinel.

    OpenRouter's upstream signatures are foreign to the user's own Anthropic
    subscription; tagging them lets ``strip_codex_thinking_blocks`` identify and
    drop them on an OpenRouter→Anthropic/Bedrock backend switch.
    """
    content = response.get('content')
    if not isinstance(content, list):
        return response
    changed = False
    new_content = []
    for block in content:
        if (isinstance(block, dict)
                and block.get('type') == 'thinking'
                and isinstance(block.get('signature'), str)
                and not block['signature'].startswith(OPENROUTER_REASONING_SIG_PREFIX)):
            new_content.append({**block, 'signature': OPENROUTER_REASONING_SIG_PREFIX + block['signature']})
            changed = True
        else:
            new_content.append(block)
    return {**response, 'content': new_content} if changed else response


def _tag_signature_in_sse_event(event_text: str) -> str:
    """Prefix the signature in a ``signature_delta`` SSE event with the OpenRouter sentinel."""
    if 'signature_delta' not in event_text:
        return event_text
    out_lines = []
    for line in event_text.split('\n'):
        if line.startswith('data: '):
            try:
                data = json.loads(line[len('data: '):])
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            delta = data.get('delta') if isinstance(data, dict) else None
            if (isinstance(delta, dict)
                    and delta.get('type') == 'signature_delta'
                    and isinstance(delta.get('signature'), str)
                    and not delta['signature'].startswith(OPENROUTER_REASONING_SIG_PREFIX)):
                delta['signature'] = OPENROUTER_REASONING_SIG_PREFIX + delta['signature']
                out_lines.append('data: ' + json.dumps(data))
                continue
        out_lines.append(line)
    return '\n'.join(out_lines)


def _handle_error_response(status: int, body_bytes: bytes) -> None:
    _handle_error_response_shared(status, body_bytes, provider_name='OpenRouter')


def _fetch_credits(api_key: str) -> dict:
    """GET the OpenRouter credits endpoint and return the parsed JSON.

    Raises ``AnthropicRequestError`` on any non-200 status (via the shared
    error handler) or on a non-JSON body. Callers surface failures as a
    Markdown notice rather than propagating.
    """
    conn = make_connection(_HOST)
    try:
        conn.request('GET', _CREDITS_PATH, headers={
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        })
        resp = conn.getresponse()
        body = resp.read()
    finally:
        conn.close()

    if resp.status == 429:
        raise UsageRateLimitError(retry_after=_parse_retry_after(resp))
    if resp.status != 200:
        _handle_error_response(resp.status, body)

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise AnthropicRequestError(
            f'OpenRouter credits endpoint returned non-JSON: {body[:200]!r}',
            error_type='api_error',
            status_code=502,
        ) from exc


def _send_with_retries(payload: dict, api_key: str, stream: bool):
    """POST *payload* to the OpenRouter ``/api/v1/messages`` endpoint.

    Returns ``(conn, resp)`` with HTTP 200.  Retries transient errors up to
    ``MAX_RETRIES`` times with exponential back-off or server-supplied
    ``Retry-After``.  Raises ``AnthropicRequestError`` on unrecoverable failures.
    HTTP 402 (credit exhaustion) is surfaced immediately as a 429-equivalent.
    """
    body_bytes = _build_body(payload)

    for attempt in range(MAX_RETRIES + 1):
        conn = make_connection(_HOST)
        try:
            conn.request('POST', _MESSAGES_PATH, body=body_bytes,
                         headers=_request_headers(api_key, stream=stream))
            resp = conn.getresponse()
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            conn.close()
            if attempt < MAX_RETRIES:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    'OpenRouter request failed (network, attempt %d/%d): %s — retrying in %.1fs',
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue
            raise AnthropicRequestError(
                f'OpenRouter connection error after {MAX_RETRIES} retries: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc

        if resp.status == 200:
            return conn, resp

        # 402 = credit exhaustion — surface as rate_limit_error immediately (no retry)
        if resp.status == 402:
            resp.read()
            conn.close()
            raise AnthropicRequestError(
                'OpenRouter credit balance exhausted',
                'rate_limit_error', 429,
            )

        if resp.status in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
            delay = _retry_delay(resp, attempt)
            resp.read()
            conn.close()
            logger.warning(
                'OpenRouter HTTP %d (attempt %d/%d) — retrying in %.1fs',
                resp.status, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue

        resp_body = resp.read()
        conn.close()
        _handle_error_response(resp.status, resp_body)

    raise AnthropicRequestError(
        'OpenRouter request failed', error_type='api_error', status_code=502,
    )


def _credit_totals(usage: dict) -> tuple[float | None, float | None]:
    block = usage.get('data') if isinstance(usage, dict) else None
    if not isinstance(block, dict):
        return None, None
    try:
        total = float(block.get('total_credits', 0.0))
        used = float(block.get('total_usage', 0.0))
    except (TypeError, ValueError):
        return None, None
    return total, used


def _format_credits_markdown(usage: dict) -> str:
    total, used = _credit_totals(usage)
    if total is None or used is None:
        raise ValueError('OpenRouter credits response has no recognizable totals')
    remaining = total - used
    return (
        '## OpenRouter credits\n\n'
        f'**Credits:** ${used:.2f} used · ${remaining:.2f} remaining '
        f'(of ${total:.2f} purchased)'
    )


def _usage_failure_markdown(message: str) -> str:
    return f'## OpenRouter credits\n\n_Usage information is unavailable: {message}_'


class OpenRouterBackend(SubscriptionBackend, Backend):
    """Backend that forwards requests to OpenRouter's Anthropic-compatible endpoint."""

    _PROVIDER_NAME = 'OpenRouter'

    def __init__(self):
        super().__init__()

    def _fetch_usage_data(self, config: Config) -> dict:
        return _fetch_credits(self._get_api_key(config))

    def _format_usage_markdown_impl(self, usage: dict) -> str:
        return _format_credits_markdown(usage)

    def _usage_failure_markdown_impl(self, message: str) -> str:
        return _usage_failure_markdown(message)

    def five_hour_status(self, config: Config) -> FiveHourStatus:
        try:
            usage = self.get_usage(config)
        except AnthropicRequestError as exc:
            if exc.status_code in (401, 402):
                return FiveHourStatus(available=False, resets_at=None)
            return FiveHourStatus(available=None, resets_at=None)
        except UsageRateLimitError as exc:
            reset = time.time() + exc.retry_after if exc.retry_after is not None else None
            return FiveHourStatus(available=None, resets_at=reset)
        except Exception:
            return FiveHourStatus(available=None, resets_at=None)

        total, used = _credit_totals(usage)
        if total is None or used is None or total <= 0:
            return FiveHourStatus(available=None, resets_at=None)
        utilization = max(0.0, used / total * 100.0)
        available = total - used > 0.0
        return FiveHourStatus(
            available=available,
            resets_at=None,
            utilization=utilization,
            weekly_utilization=utilization,
        )

    def parse_credentials(self, api_key: str) -> dict:
        """OpenRouter uses a Bearer token from config, not the x-api-key header."""
        return {}

    def _get_api_key(self, config: Config) -> str:
        key = config.openrouter_api_key
        if not key:
            raise AnthropicRequestError(
                'OPENROUTER_API_KEY is not set. Configure --openrouter-api-key or '
                'set the OPENROUTER_API_KEY environment variable.',
                'authentication_error', 401,
            )
        return key

    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        api_key = self._get_api_key(config)
        conn, resp = _send_with_retries(payload, api_key, stream=False)
        try:
            body = resp.read()
        finally:
            conn.close()
        try:
            response = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnthropicRequestError(
                f'OpenRouter returned non-JSON response: {body[:200]!r}',
                error_type='api_error',
                status_code=502,
            ) from exc
        return _tag_thinking_signatures(response)

    # Reasoning models (e.g. the deepseek alias used as the default classifier)
    # emit thinking/redacted_thinking blocks that can consume the tiny 4-token
    # classifier budget before any text is produced.  Disable thinking and raise
    # max_tokens so a text block is returned.  parse_classifier_label skips any
    # thinking blocks that slip through anyway.
    _CLASSIFIER_MAX_TOKENS = 64

    def send_classifier_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        payload = {**payload, 'thinking': {'type': 'disabled'}}
        if payload.get('max_tokens', 0) < self._CLASSIFIER_MAX_TOKENS:
            payload['max_tokens'] = self._CLASSIFIER_MAX_TOKENS
        return self.send_message(payload, credentials, config)

    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        api_key = self._get_api_key(config)
        conn, resp = _send_with_retries({**payload, 'stream': True}, api_key, stream=True)
        try:
            pending = ''
            for line in _read_sse_lines(resp):
                # OpenRouter appends an OpenAI-style [DONE] sentinel — filter it out.
                # read_sse_lines() strips the trailing \n, so compare without it.
                if line == 'data: [DONE]':
                    continue
                pending += line + '\n'
                if line.strip() == '':
                    if pending.strip():
                        yield _tag_signature_in_sse_event(pending)
                    pending = ''
            if pending.strip():
                yield _tag_signature_in_sse_event(pending + '\n')
        finally:
            conn.close()

    def count_tokens(self, payload: dict, credentials: dict, config: Config) -> dict:
        """Return a local estimate — OpenRouter has no /count_tokens endpoint."""
        return {
            'input_tokens': estimate_input_tokens(payload),
            'model': _resolve_model(payload.get('model', '')),
        }
