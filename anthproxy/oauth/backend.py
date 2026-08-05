import http.client
import json
import logging
import socket
import ssl
import time

from .. import model_config
from .._shared import Backend
from .._shared.http_util import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    handle_error_response,
    make_connection,
    parse_retry_after,
    read_sse_lines,
    should_retry,
)
from ..mapper.anthropic_protocol import (
    ANTHROPIC_HOST,
    ANTHROPIC_VERSION,
    COUNT_TOKENS_PATH,
    MESSAGES_PATH,
    USER_AGENT,
)
# TODO(arch): move build_body, merge_betas, resolve_model to mapper/anthropic_transform.py
# to avoid importing from sibling anthropic backend package. These are shared Anthropic-
# protocol transformation functions, not anthropic-backend-specific. See issue #13 refactor.
from ..anthropic.mapper import (
    build_body,
    merge_betas,
    resolve_model,
)
from ..config import Config
from ..mapper import AnthropicRequestError, estimate_input_tokens
from ..oauth_registry import OAuthRequestCredentials

logger = logging.getLogger(__name__)


def _make_connection() -> http.client.HTTPSConnection:
    return make_connection(ANTHROPIC_HOST)


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


def _credential(credentials: dict) -> OAuthRequestCredentials:
    selected = credentials.get('oauth')
    if not isinstance(selected, OAuthRequestCredentials):
        raise AnthropicRequestError(
            'OAuth backend requires request Bearer credentials',
            error_type='authentication_error',
            status_code=401,
        )
    return selected


def _send(
    payload: dict, credentials: dict, stream: bool, path: str = MESSAGES_PATH,
) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
    selected = _credential(credentials)
    body = build_body(payload)
    headers = _request_headers(selected.access_token, merge_betas(payload), stream)
    for attempt in range(MAX_RETRIES + 1):
        connection = _make_connection()
        try:
            connection.request('POST', path, body=body, headers=headers)
            response = connection.getresponse()
        except (socket.error, http.client.HTTPException, ssl.SSLError, OSError) as exc:
            connection.close()
            if attempt < MAX_RETRIES:
                time.sleep(min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY))
                continue
            raise AnthropicRequestError(
                f'Anthropic connection error after {MAX_RETRIES} retries: {exc}',
                error_type='api_error',
                status_code=502,
            ) from exc
        if response.status == 200:
            return connection, response
        if response.status != 429 and should_retry(response.status, response) and attempt < MAX_RETRIES:
            response.read()
            connection.close()
            time.sleep(min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY))
            continue
        retry_after = parse_retry_after(response) if response.status == 429 else None
        response_body = response.read()
        connection.close()
        if response.status == 429:
            try:
                handle_error_response(
                    response.status,
                    response_body,
                    provider_name='Anthropic',
                    forward_error_type=True,
                )
            except AnthropicRequestError as exc:
                exc.retry_after = retry_after
                raise
        handle_error_response(
            response.status,
            response_body,
            provider_name='Anthropic',
            forward_error_type=True,
        )
    raise AnthropicRequestError('Anthropic request failed', error_type='api_error', status_code=502)


class OAuthBackend(Backend):
    @classmethod
    def model_aliases(cls) -> dict:
        return model_config.model_aliases('anthropic')

    @classmethod
    def summary_credentials(cls, snapshot) -> None:
        return None

    def parse_credentials(self, api_key: str) -> dict:
        return {}

    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        connection, response = _send(payload, credentials, stream=False)
        try:
            body = response.read()
        finally:
            connection.close()
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnthropicRequestError(
                f'Upstream returned non-JSON response: {body[:200]!r}',
                error_type='api_error',
                status_code=502,
            ) from exc

    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        connection, response = _send(payload, credentials, stream=True)
        try:
            pending = ''
            for line in read_sse_lines(response):
                pending += line + '\n'
                if not line.strip():
                    if pending.strip():
                        yield pending
                    pending = ''
            if pending.strip():
                yield pending + '\n'
        finally:
            connection.close()

    def count_tokens(self, payload: dict, credentials: dict, config: Config) -> dict:
        try:
            connection, response = _send(
                payload, credentials, stream=False, path=COUNT_TOKENS_PATH,
            )
            try:
                return json.loads(response.read())
            finally:
                connection.close()
        except Exception as exc:
            logger.warning('OAuth count_tokens failed: %s — using estimate', exc)
            return {
                'input_tokens': estimate_input_tokens(payload),
                'model': resolve_model(payload.get('model', '')),
            }
