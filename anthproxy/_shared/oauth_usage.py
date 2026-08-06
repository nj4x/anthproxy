"""Shared OAuth usage endpoint transport.

Fetches OAuth enterprise token usage from the Anthropic endpoint.
Shared by backends that support OAuth tokens (e.g., the `oauth` backend).
"""

import http.client
import json

from . import UsageRateLimitError
from .http_util import parse_retry_after as _parse_retry_after
from ..mapper.anthropic_protocol import (
    ANTHROPIC_HOST,
    ANTHROPIC_VERSION,
    USER_AGENT,
)

USAGE_PATH = '/api/oauth/usage'
USAGE_TIMEOUT_SECONDS = 5


def fetch_oauth_usage(access_token: str) -> dict:
    """Fetch OAuth enterprise token usage from Anthropic.

    Raises:
        UsageRateLimitError: if the endpoint returns 429
        RuntimeError: for auth failures or unexpected status codes
        ValueError: if usage response is unparseable
    """
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
    if status == 429:
        raise UsageRateLimitError(retry_after=_parse_retry_after(resp))
    if status in (401, 403):
        raise RuntimeError('Authentication failed for usage endpoint')
    raise RuntimeError(f'Anthropic usage endpoint returned HTTP {status}')
