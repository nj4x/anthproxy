"""Codex backend — OAuth credential management.

Reads / refreshes ChatGPT OAuth tokens from ~/.codex/auth.json (written by
the official ``codex`` CLI).  Automatically runs an OAuth2 PKCE login flow
at startup if credentials are missing or the refresh token is dead.

OAuth constants and flow mirror the official Codex CLI (codex-rs/login):
  - client_id:   app_EMoamEEZ73f0CkXaXp7hrann
  - token URL:   https://auth.openai.com/oauth/token
  - upstream:    https://chatgpt.com/backend-api/codex/responses

Shared OAuth scaffolding (TerminalRefreshError, _write_auth, _refresh_lock,
run_pkce_login_server, and the four orchestration functions) lives in
``oauth_base``.  Provider-specific code here: OAuth constants, JWT helpers,
on-disk schema, _authorize_url, _exchange_code, refresh, login.
"""

import base64
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
from datetime import datetime, timezone

from .._shared.oauth_base import (
    OAuthProvider,
    TerminalRefreshError,
    _auth_file,
    _b64url_nopad,
    _classify_refresh_error,
    _make_state,
    _refresh_lock,  # noqa: F401 — re-exported for tests
    _write_auth,
    ensure_credentials as _ensure_credentials_shared,
    ensure_credentials_noninteractive as _ensure_credentials_noninteractive_shared,
    force_refresh as _force_refresh_shared,
    get_access as _get_access_shared,
    run_pkce_login_server,
)

logger = logging.getLogger(__name__)

CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
AUTH_HOST = 'auth.openai.com'
AUTHORIZE_PATH = '/oauth/authorize'
TOKEN_PATH = '/oauth/token'
SCOPE = 'openid profile email offline_access api.connectors.read api.connectors.invoke'
ORIGINATOR = 'codex_cli_rs'
REDIRECT_PORT = 1455
FALLBACK_PORT = 1457
CALLBACK_PATH = '/auth/callback'
ACCESS_REFRESH_WINDOW_SECS = 5 * 60
ACCESS_REFRESH_INTERVAL_DAYS = 8
_AUTH_CLAIM = 'https://api.openai.com/auth'


def _codex_home(config=None) -> pathlib.Path:
    if config is not None and getattr(config, 'codex_home', ''):
        return pathlib.Path(config.codex_home)
    env = os.environ.get('CODEX_HOME', '').strip()
    if env:
        p = pathlib.Path(env)
        if p.is_dir():
            return p
    return pathlib.Path.home() / '.codex'


def _b64url_decode(s: str) -> bytes:
    pad = (4 - len(s) % 4) % 4
    return base64.urlsafe_b64decode(s + '=' * pad)


def _decode_jwt_payload(jwt: str) -> dict:
    try:
        parts = jwt.split('.')
        if len(parts) < 2:
            return {}
        return json.loads(_b64url_decode(parts[1]))
    except Exception:
        return {}


def _jwt_exp(jwt: str) -> int | None:
    payload = _decode_jwt_payload(jwt)
    exp = payload.get('exp')
    return int(exp) if exp is not None else None


def _account_id_from_id_token(id_token: str) -> str | None:
    payload = _decode_jwt_payload(id_token)
    auth_claims = payload.get(_AUTH_CLAIM) or {}
    return auth_claims.get('chatgpt_account_id') or payload.get('chatgpt_account_id')


def load_credentials(home: pathlib.Path) -> dict | None:
    auth_file = _auth_file(home)
    if not auth_file.exists():
        return None
    try:
        raw = json.loads(auth_file.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None

    tokens = raw.get('tokens') or {}
    access_token = tokens.get('access_token', '')
    refresh_token = tokens.get('refresh_token', '')
    if not access_token or not refresh_token:
        return None

    last_refresh: float | None = None
    last_refresh_str = raw.get('last_refresh', '')
    if last_refresh_str:
        try:
            dt = datetime.fromisoformat(last_refresh_str.replace('Z', '+00:00'))
            last_refresh = dt.timestamp()
        except (ValueError, AttributeError):
            pass

    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'account_id': tokens.get('account_id'),
        'id_token': tokens.get('id_token', ''),
        'last_refresh': last_refresh,
        '_raw': raw,
    }


def needs_access_refresh(creds: dict) -> bool:
    access_token = creds.get('access_token', '')
    if access_token:
        exp = _jwt_exp(access_token)
        if exp is not None:
            return exp <= time.time() + ACCESS_REFRESH_WINDOW_SECS
    last_refresh = creds.get('last_refresh')
    if last_refresh is None:
        return True
    return time.time() - last_refresh > ACCESS_REFRESH_INTERVAL_DAYS * 86400


def refresh(creds: dict, home: pathlib.Path) -> dict:
    body = json.dumps({
        'client_id': CLIENT_ID,
        'grant_type': 'refresh_token',
        'refresh_token': creds['refresh_token'],
    }).encode('utf-8')

    conn = http.client.HTTPSConnection(AUTH_HOST, timeout=30)
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

    raw = dict(creds.get('_raw') or {})
    tokens = dict(raw.get('tokens') or {})
    if 'access_token' in data:
        tokens['access_token'] = data['access_token']
    if 'refresh_token' in data:
        tokens['refresh_token'] = data['refresh_token']
    if 'id_token' in data:
        tokens['id_token'] = data['id_token']
        new_account_id = _account_id_from_id_token(data['id_token'])
        if new_account_id:
            tokens['account_id'] = new_account_id

    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    raw['tokens'] = tokens
    raw['last_refresh'] = now_iso
    _write_auth(home, raw)

    logger.info('Codex tokens refreshed successfully.')
    return {
        'access_token': tokens.get('access_token', creds['access_token']),
        'refresh_token': tokens.get('refresh_token', creds['refresh_token']),
        'account_id': tokens.get('account_id', creds.get('account_id')),
        'id_token': tokens.get('id_token', creds.get('id_token', '')),
        'last_refresh': time.time(),
        '_raw': raw,
    }


def _pkce() -> tuple[str, str]:
    verifier = _b64url_nopad(secrets.token_bytes(64))
    challenge = _b64url_nopad(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def _authorize_url(challenge: str, state: str, port: int) -> tuple[str, str]:
    redirect_uri = f'http://localhost:{port}{CALLBACK_PATH}'
    params = [
        ('response_type', 'code'),
        ('client_id', CLIENT_ID),
        ('redirect_uri', redirect_uri),
        ('scope', SCOPE),
        ('code_challenge', challenge),
        ('code_challenge_method', 'S256'),
        ('id_token_add_organizations', 'true'),
        ('codex_cli_simplified_flow', 'true'),
        ('state', state),
        ('originator', ORIGINATOR),
    ]
    qs = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f'https://{AUTH_HOST}{AUTHORIZE_PATH}?{qs}', redirect_uri


def _exchange_code(code: str, redirect_uri: str, verifier: str) -> dict:
    body = urllib.parse.urlencode({
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri,
        'client_id': CLIENT_ID,
        'code_verifier': verifier,
    }).encode('utf-8')

    conn = http.client.HTTPSConnection(AUTH_HOST, timeout=30)
    try:
        conn.request('POST', TOKEN_PATH, body=body, headers={
            'Content-Type': 'application/x-www-form-urlencoded',
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
    return json.loads(resp_body)


def login(home: pathlib.Path) -> dict:
    verifier, challenge = _pkce()
    state = _make_state()

    code, redirect_uri = run_pkce_login_server(
        (REDIRECT_PORT, FALLBACK_PORT),
        CALLBACK_PATH,
        'Codex',
        state,
        lambda port: _authorize_url(challenge, state, port),
    )

    token_data = _exchange_code(code, redirect_uri, verifier)

    access_token = token_data['access_token']
    refresh_token = token_data['refresh_token']
    id_token = token_data.get('id_token', '')
    account_id = _account_id_from_id_token(id_token) if id_token else None

    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    raw = {
        'auth_mode': 'chatgpt',
        'OPENAI_API_KEY': None,
        'tokens': {
            'id_token': id_token,
            'access_token': access_token,
            'refresh_token': refresh_token,
            'account_id': account_id,
        },
        'last_refresh': now_iso,
    }
    _write_auth(home, raw)
    logger.info('Codex login successful. Credentials saved to %s', _auth_file(home))

    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'account_id': account_id,
        'id_token': id_token,
        'last_refresh': time.time(),
        '_raw': raw,
    }


_PROVIDER = OAuthProvider(
    name='Codex',
    account_id_field='account_id',
    restart_hint='--backend codex',
    no_creds_noninteractive_msg=(
        'No Codex credentials found. Run `codex login` or restart anthproxy '
        'with --backend codex to authenticate.'
    ),
    home_fn=lambda config=None: _codex_home(config),
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
