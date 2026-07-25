"""Local (LM Studio) backend for anthproxy.

Relays Anthropic Messages API requests to a locally-running LM Studio instance
that exposes an Anthropic-compatible ``/v1/messages`` endpoint.

Key properties
--------------
* Plain HTTP (``http.client.HTTPConnection``) — LM Studio runs on localhost.
* No authentication — ``x-api-key`` is accepted and silently ignored.
* No SSE translation — the upstream returns native Anthropic-format SSE.
* No usage/quota tracking — the backend is never auto-selected.
* Configurable via ``--local-base-url`` / ``ANTHPROXY_LOCAL_BASE_URL``
  (default: ``http://127.0.0.1:1235``).
"""

import http.client
import json
import logging
import socket
import ssl
import time
import urllib.parse

from .._shared import Backend
from .._shared.http_util import (
    MAX_RETRIES,
    RETRYABLE_STATUSES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    handle_error_response as _handle_error_response_shared,
    read_sse_lines as _read_sse_lines,
    retry_delay as _retry_delay,
)
from ..config import Config
from ..mapper import AnthropicRequestError, estimate_input_tokens
from .mapper import _build_body, _resolve_model

logger = logging.getLogger(__name__)

_MESSAGES_PATH = '/v1/messages'


def _make_connection(config: Config):
    """Open a plain HTTP (or HTTPS) connection to the LM Studio endpoint."""
    parsed = urllib.parse.urlsplit(config.local_base_url)
    scheme = (parsed.scheme or 'http').lower()
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or (443 if scheme == 'https' else 1235)
    if scheme == 'https':
        ctx = ssl.create_default_context()
        return http.client.HTTPSConnection(host, port, timeout=300, context=ctx)
    return http.client.HTTPConnection(host, port, timeout=300)


def _request_headers(stream: bool) -> dict:
    return {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream' if stream else 'application/json',
    }


def _handle_error_response(status: int, body_bytes: bytes) -> None:
    _handle_error_response_shared(status, body_bytes, provider_name='LMStudio')


def _send_with_retries(
    payload: dict,
    config: Config,
    stream: bool,
) -> tuple:
    """POST *payload* to the LM Studio ``/v1/messages`` endpoint.

    Returns ``(conn, resp)`` with HTTP 200.  Retries on transient errors up to
    ``MAX_RETRIES`` times with exponential back-off.  Raises
    ``AnthropicRequestError`` on unrecoverable failures.
    """
    body_bytes = _build_body(payload)

    for attempt in range(MAX_RETRIES + 1):
        conn = _make_connection(config)
        try:
            conn.request('POST', _MESSAGES_PATH, body=body_bytes,
                         headers=_request_headers(stream))
            resp = conn.getresponse()
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            conn.close()
            if attempt < MAX_RETRIES:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    'LMStudio request failed (network, attempt %d/%d): %s — retrying in %.1fs',
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue
            raise AnthropicRequestError(
                f'LMStudio connection error after {MAX_RETRIES} retries: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc

        if resp.status == 200:
            return conn, resp

        if resp.status in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
            delay = _retry_delay(resp, attempt)
            resp.read()
            conn.close()
            logger.warning(
                'LMStudio HTTP %d (attempt %d/%d) — retrying in %.1fs',
                resp.status, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue

        resp_body = resp.read()
        conn.close()
        _handle_error_response(resp.status, resp_body)

    raise AnthropicRequestError('LMStudio request failed', error_type='api_error', status_code=502)


class LocalBackend(Backend):
    """Backend that forwards requests to a local LM Studio Anthropic-compatible endpoint.

    Inherits only from ``Backend`` (not ``SubscriptionBackend``) so it carries no
    quota logic, usage endpoint, or ``five_hour_status`` method.  The auto-selector
    never considers this backend; it is activated only via ``proxy-set-backend:local``
    or the ``--backend local`` CLI flag.
    """

    def parse_credentials(self, api_key: str) -> dict:
        """LM Studio does not require authentication — ignore the api key."""
        return {}

    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        conn, resp = _send_with_retries(payload, config, stream=False)
        try:
            body = resp.read()
        finally:
            conn.close()

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnthropicRequestError(
                f'LMStudio returned non-JSON response: {body[:200]!r}',
                error_type='api_error',
                status_code=502,
            ) from exc

        logger.debug('<<< LMStudio response: model=%s stop=%s',
                     payload.get('model', ''), result.get('stop_reason'))
        return result

    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        conn, resp = _send_with_retries(payload, config, stream=True)
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
        """Return a local estimate — LM Studio may not implement count_tokens."""
        return {
            'input_tokens': estimate_input_tokens(payload),
            'model': _resolve_model(payload.get('model', '')),
        }
