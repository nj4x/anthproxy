"""Anthropic backend — OAuth credential management."""

import hashlib
import http.client
import json
import logging
import os
import pathlib
import secrets
import threading
import time
import urllib.parse

from .._shared.oauth_base import (
    OAuthProvider,
    TerminalRefreshError,
    _auth_file,
    _b64url_nopad,
    _classify_refresh_error,
    _make_state,
    _write_auth,
    ensure_credentials as _ensure_credentials_shared,
    ensure_credentials_noninteractive as _ensure_credentials_noninteractive_shared,
    force_refresh as _force_refresh_shared,
    get_access as _get_access_shared,
    run_pkce_login_server,
)

logger = logging.getLogger(__name__)

CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e'
AUTH_HOST = 'claude.ai'
AUTHORIZE_PATH = '/oauth/authorize'
TOKEN_HOST = 'platform.claude.com'
TOKEN_PATH = '/v1/oauth/token'
SCOPE = 'org:create_api_key user:profile user:inference'
REDIRECT_PORT = 54545
FALLBACK_PORT = 53692
CALLBACK_PATH = '/callback'
ACCESS_REFRESH_WINDOW_SECS = 5 * 60


def _anthropic_home(config=None) -> pathlib.Path:
    if config is not None and getattr(config, 'anthropic_home', ''):
        return pathlib.Path(config.anthropic_home)
    env = os.environ.get('ANTHROPIC_HOME', '').strip()
    if env:
        p = pathlib.Path(env)
        if p.is_dir():
            return p
    # Try new ANTHPROXY_HOME/anthropic location if available
    if config is not None:
        anthproxy_home = getattr(config, 'anthproxy_home', '')
        if not anthproxy_home or not anthproxy_home.strip():
            # Not set in config, try env var or default
            from ..config import _resolve_home
            anthproxy_home = _resolve_home('')
        p = pathlib.Path(anthproxy_home) / 'anthropic'
        if p.is_dir():
            return p
    # Legacy fallback to ~/.anthropic
    return pathlib.Path.home() / '.anthropic'


def load_credentials(home: pathlib.Path) -> dict | None:
    auth_file = _auth_file(home)
    if not auth_file.exists():
        return None
    try:
        raw = json.loads(auth_file.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None

    access_token = raw.get('access_token', '')
    refresh_token = raw.get('refresh_token', '')
    if not access_token or not refresh_token:
        return None

    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'expires_at': raw.get('expires_at'),
        'account_uuid': raw.get('account_uuid'),
        'email': raw.get('email'),
        'last_refresh': raw.get('last_refresh'),
    }


def needs_access_refresh(creds: dict) -> bool:
    expires_at = creds.get('expires_at')
    if expires_at is not None:
        try:
            return float(expires_at) <= time.time() + ACCESS_REFRESH_WINDOW_SECS
        except (TypeError, ValueError):
            pass
    last_refresh = creds.get('last_refresh')
    if last_refresh is None:
        return True
    try:
        return time.time() - float(last_refresh) > 8 * 86400
    except (TypeError, ValueError):
        return True


def refresh(creds: dict, home: pathlib.Path) -> dict:
    body = json.dumps({
        'client_id': CLIENT_ID,
        'grant_type': 'refresh_token',
        'refresh_token': creds['refresh_token'],
    }).encode('utf-8')

    conn = http.client.HTTPSConnection(TOKEN_HOST, timeout=30)
    try:
        conn.request('POST', TOKEN_PATH, body=body, headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })
        resp = conn.getresponse()
        resp_body = resp.read()
    finally:
        conn.close()

    terminal = _classify_refresh_error(resp.status, resp_body)
    if terminal:
        raise TerminalRefreshError(
            f'Refresh token permanently invalid ({terminal}). Re-login required.'
        )

    if resp.status != 200:
        raise RuntimeError(
            f'Token refresh failed with HTTP {resp.status}: '
            f'{resp_body.decode("utf-8", errors="replace")[:300]}'
        )

    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'Token refresh returned non-JSON: {resp_body[:200]!r}'
        ) from exc

    now = time.time()
    expires_in = data.get('expires_in')
    expires_at = (now + float(expires_in)) if expires_in is not None else None

    new_creds = {
        'access_token': data.get('access_token', creds['access_token']),
        'refresh_token': data.get('refresh_token', creds['refresh_token']),
        'expires_at': expires_at if expires_at is not None else creds.get('expires_at'),
        'account_uuid': creds.get('account_uuid'),
        'email': creds.get('email'),
        'last_refresh': now,
    }
    _write_auth(home, new_creds)
    logger.info('Anthropic tokens refreshed successfully.')
    return new_creds


def _authorize_url(challenge: str, state: str, port: int) -> tuple[str, str]:
    redirect_uri = f'http://localhost:{port}{CALLBACK_PATH}'
    params = {
        'code': 'true',
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': redirect_uri,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
        'state': state,
    }
    qs = urllib.parse.urlencode(params)
    scope_param = 'scope=' + urllib.parse.quote(SCOPE, safe=':').replace('%20', '+')
    return f'https://{AUTH_HOST}{AUTHORIZE_PATH}?{qs}&{scope_param}', redirect_uri


def _exchange_code(code: str, state: str, redirect_uri: str, verifier: str) -> dict:
    body = json.dumps({
        'grant_type': 'authorization_code',
        'code': code,
        'state': state,
        'redirect_uri': redirect_uri,
        'client_id': CLIENT_ID,
        'code_verifier': verifier,
    }).encode('utf-8')

    conn = http.client.HTTPSConnection(TOKEN_HOST, timeout=30)
    try:
        conn.request('POST', TOKEN_PATH, body=body, headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })
        resp = conn.getresponse()
        resp_body = resp.read()
    finally:
        conn.close()

    if resp.status != 200:
        raise RuntimeError(
            f'Token exchange failed with HTTP {resp.status}: '
            f'{resp_body.decode("utf-8", errors="replace")[:500]}'
        )

    data = json.loads(resp_body)
    account = data.get('account') or {}
    return {
        'access_token': data['access_token'],
        'refresh_token': data['refresh_token'],
        'expires_in': data.get('expires_in'),
        'account_uuid': account.get('uuid') or account.get('account_uuid') or data.get('account_uuid'),
        'email': account.get('email_address') or account.get('email') or data.get('email'),
    }


def _pkce() -> tuple[str, str]:
    verifier = _b64url_nopad(secrets.token_bytes(32))
    challenge = _b64url_nopad(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def login(home: pathlib.Path) -> dict:
    verifier, challenge = _pkce()
    state = _make_state()

    code, redirect_uri = run_pkce_login_server(
        (REDIRECT_PORT, FALLBACK_PORT),
        CALLBACK_PATH,
        'Anthropic',
        state,
        lambda port: _authorize_url(challenge, state, port),
    )

    token_data = _exchange_code(code, state, redirect_uri, verifier)

    now = time.time()
    expires_in = token_data.get('expires_in')
    expires_at = (now + float(expires_in)) if expires_in is not None else None

    creds = {
        'access_token': token_data['access_token'],
        'refresh_token': token_data['refresh_token'],
        'expires_at': expires_at,
        'account_uuid': token_data.get('account_uuid'),
        'email': token_data.get('email'),
        'last_refresh': now,
    }
    _write_auth(home, creds)
    logger.info('Anthropic login successful. Credentials saved to %s', _auth_file(home))
    return creds


_PROVIDER = OAuthProvider(
    name='Anthropic',
    account_id_field='account_uuid',
    restart_hint='--backend anthropic',
    no_creds_noninteractive_msg=(
        'No Anthropic credentials found. Restart anthproxy with --backend anthropic '
        'to authenticate interactively.'
    ),
    home_fn=lambda config=None: _anthropic_home(config),
    load_credentials_fn=lambda home: load_credentials(home),
    needs_refresh_fn=lambda creds: needs_access_refresh(creds),
    refresh_fn=lambda creds, home: refresh(creds, home),
    login_fn=lambda home: login(home),
)


def ensure_credentials(config=None) -> None:
    _ensure_credentials_shared(_PROVIDER, config)


def ensure_credentials_noninteractive(config=None) -> None:
    _ensure_credentials_noninteractive_shared(_PROVIDER, config)


def get_access(config, lock: threading.Lock) -> tuple[str, str | None]:
    return _get_access_shared(_PROVIDER, config, lock)


def force_refresh(config, lock: threading.Lock) -> tuple[str, str | None]:
    return _force_refresh_shared(_PROVIDER, config, lock)
