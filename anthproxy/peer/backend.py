"""Peer backend for anthproxy — dispatch to another anthproxy instance.

Relays Anthropic Messages API requests to a second anthproxy over its public
``/v1/messages`` interface (ADR-0021).

Key properties
--------------
* Plain HTTP for ``http://`` targets, HTTPS for ``https://`` — a peer is
  commonly reached over loopback, an SSH tunnel, or a private network.
* The requested model is transmitted verbatim; resolving it is the peer's job.
* No SSE translation — the peer emits native Anthropic SSE.
* Credential is ``X-Anthproxy-Peer-Key``, never ``Authorization: Bearer``,
  which the receiving instance would absorb as an OAuth delegation.
* ``count_tokens`` is proxied to the peer with no local-estimate fallback.
* Configured via ``--peer-base-url`` / ``ANTHPROXY_PEER_BASE_URL`` and
  ``--peer-api-key`` / ``ANTHPROXY_PEER_API_KEY``, both unset by default.
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
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    handle_error_response as _handle_error_response_shared,
    parse_retry_after as _parse_retry_after,
    read_sse_lines as _read_sse_lines,
    retry_delay as _retry_delay,
    should_retry as _should_retry,
)
from ..config import Config
from ..mapper import AnthropicRequestError
from .mapper import _beta_header, _build_body

logger = logging.getLogger(__name__)

_MESSAGES_PATH = '/v1/messages'
_COUNT_TOKENS_PATH = '/v1/messages/count_tokens'
_PEER_KEY_HEADER = 'X-Anthproxy-Peer-Key'


def _require_target(config: Config) -> str:
    """Return the configured peer base URL, or raise if none is configured.

    ``from_config`` must be total — the discovery completeness assertion
    constructs every declared backend at startup — so an unconfigured peer is
    constructible and fails here, at dispatch, instead.
    """
    base_url = (config.peer_base_url or '').strip()
    if not base_url:
        raise AnthropicRequestError(
            'peer backend has no target configured: set --peer-base-url '
            '(env ANTHPROXY_PEER_BASE_URL) to another anthproxy instance',
            error_type='api_error',
            status_code=502,
        )
    return base_url


def _redact(base_url: str) -> str:
    """Strip any ``user:pass@`` userinfo before a URL reaches a client-visible error."""
    head, sep, tail = base_url.rpartition('@')
    if not sep:
        return base_url
    scheme, _, _ = head.partition('://')
    return f'{scheme}://***@{tail}' if scheme != head else f'***@{tail}'


def _target(config: Config) -> tuple[str, str, int, str]:
    """Parse the peer base URL into ``(scheme, host, port, path_prefix)``.

    Malformed targets are rejected rather than defaulted. ``local`` can afford to
    fall back to loopback on an unparseable URL because it sends no credential;
    a peer that silently retargets would send ``X-Anthproxy-Peer-Key`` in
    cleartext to a host the operator did not name (e.g. ``htttps://peer.example``
    parses as a non-https scheme and would otherwise become plain HTTP port 80).
    """
    base_url = _require_target(config)
    parsed = urllib.parse.urlsplit(base_url)
    scheme = parsed.scheme.lower()
    if scheme not in ('http', 'https'):
        raise AnthropicRequestError(
            f'--peer-base-url must start with http:// or https://, got {_redact(base_url)!r}',
            error_type='api_error',
            status_code=502,
        )
    if not parsed.hostname:
        raise AnthropicRequestError(
            f'--peer-base-url names no host: {_redact(base_url)!r}',
            error_type='api_error',
            status_code=502,
        )
    try:
        port = parsed.port or (443 if scheme == 'https' else 80)
    except ValueError as exc:
        raise AnthropicRequestError(
            f'--peer-base-url has an invalid port: {_redact(base_url)!r}',
            error_type='api_error',
            status_code=502,
        ) from exc
    return scheme, parsed.hostname, port, parsed.path.rstrip('/')


def _make_connection(config: Config):
    """Open a connection to the peer, preserving plain HTTP for http:// targets."""
    scheme, host, port, _ = _target(config)
    if scheme == 'https':
        ctx = ssl.create_default_context()
        return http.client.HTTPSConnection(host, port, timeout=300, context=ctx)
    return http.client.HTTPConnection(host, port, timeout=300)


def _path(config: Config, path: str) -> str:
    """Prefix *path* with any path component of the base URL.

    ADR-0021 §4 expects a peer to sit behind a reverse proxy or identity-aware
    proxy, where a mount prefix (``http://gw.internal/anthproxy``) is normal.
    """
    return _target(config)[3] + path


def _request_headers(config: Config, betas: str, stream: bool) -> dict:
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream' if stream else 'application/json',
    }
    if betas:
        headers['anthropic-beta'] = betas
    api_key = (config.peer_api_key or '').strip()
    if api_key:
        headers[_PEER_KEY_HEADER] = api_key
    return headers


def _handle_error_response(status: int, body_bytes: bytes) -> None:
    if status == 404:
        detail = body_bytes.decode('utf-8', errors='replace')[:300]
        raise AnthropicRequestError(
            f'Peer returned HTTP 404 — the target does not implement this endpoint '
            f'and is not a conforming anthproxy peer; check --peer-base-url: {detail}',
            error_type='api_error',
            status_code=502,
        )
    _handle_error_response_shared(status, body_bytes, provider_name='Peer')


def _send_with_retries(payload: dict, config: Config, stream: bool) -> tuple:
    """POST *payload* to the peer's ``/v1/messages``, returning ``(conn, resp)`` at 200.

    A peer 429 always surfaces to the caller — never retried in-backend, with or
    without ``Retry-After`` — so the selector can park the peer and fail over
    rather than sleeping against the peer's own retry guidance.
    """
    body_bytes = _build_body(payload)
    headers = _request_headers(config, _beta_header(payload), stream)
    path = _path(config, _MESSAGES_PATH)

    for attempt in range(MAX_RETRIES + 1):
        conn = _make_connection(config)
        try:
            conn.request('POST', path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            conn.close()
            if attempt < MAX_RETRIES:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    'Peer request failed (network, attempt %d/%d): %s — retrying in %.1fs',
                    attempt + 1, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue
            raise AnthropicRequestError(
                f'Peer connection error after {MAX_RETRIES} retries: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc

        if resp.status == 200:
            return conn, resp

        if resp.status != 429 and _should_retry(resp.status, resp) and attempt < MAX_RETRIES:
            delay = _retry_delay(resp, attempt)
            resp.read()
            conn.close()
            logger.warning(
                'Peer HTTP %d (attempt %d/%d) — retrying in %.1fs',
                resp.status, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue

        retry_after = _parse_retry_after(resp) if resp.status == 429 else None
        resp_body = resp.read()
        conn.close()
        try:
            _handle_error_response(resp.status, resp_body)
        except AnthropicRequestError as exc:
            exc.retry_after = retry_after
            raise


class PeerBackend(Backend):
    """Backend that forwards requests to another anthproxy instance.

    Inherits only from ``Backend``: capacity participation in the selector is a
    separate concern from transport and arrives with ``five_hour_status``.
    """

    @classmethod
    def summary_credentials(cls, snapshot) -> dict:
        return {}

    def parse_credentials(self, api_key: str) -> dict:
        """The client's ``x-api-key`` is not forwarded to the peer."""
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
                f'Peer returned non-JSON response: {body[:200]!r}',
                error_type='api_error',
                status_code=502,
            ) from exc

        logger.debug('<<< Peer response: model=%s stop=%s',
                     result.get('model', ''), result.get('stop_reason'))
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
        """Proxy to the peer's ``count_tokens``.

        There is no fallback path: a local estimate would be indistinguishable
        from the peer's real answer, and ``count_tokens`` answers are acted on.
        Every non-200 — including 404, 429, and connection failures — surfaces
        as an Anthropic error envelope.
        """
        body_bytes = _build_body(payload)
        headers = _request_headers(config, _beta_header(payload), stream=False)
        path = _path(config, _COUNT_TOKENS_PATH)
        conn = _make_connection(config)
        try:
            conn.request('POST', path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            status = resp.status
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            raise AnthropicRequestError(
                f'Peer count_tokens connection error: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc
        finally:
            conn.close()

        if status != 200:
            _handle_error_response(status, resp_body)

        try:
            return json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise AnthropicRequestError(
                f'Peer count_tokens returned non-JSON response: {resp_body[:200]!r}',
                error_type='api_error',
                status_code=502,
            ) from exc
