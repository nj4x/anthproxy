"""Shared OAuth2 PKCE scaffolding for Codex and Anthropic backends.

Contains the byte-identical (or trivially parameterizable) OAuth machinery
that was previously duplicated between ``codex_auth.py`` and
``anthropic_auth.py``.  Provider-specific OAuth constants, on-disk schema,
token exchange, and credential refresh stay in each ``*_auth.py``.

Public API (used by each ``*_auth.py``):
    TerminalRefreshError
    _auth_file, _b64url_nopad, _make_state
    _write_auth, _refresh_lock
    _TERMINAL_ERROR_CODES, _classify_refresh_error
    run_pkce_login_server
    OAuthProvider
    ensure_credentials, ensure_credentials_noninteractive, get_access, force_refresh
"""

import contextlib
import dataclasses
import hashlib  # noqa: F401 — kept for callers that import from here
import json
import logging
import os
import pathlib
import secrets
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from base64 import urlsafe_b64encode
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows — cross-process locking not available; best-effort

logger = logging.getLogger(__name__)

# Seconds to wait for the browser to complete login
_LOGIN_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class TerminalRefreshError(Exception):
    """Raised when the refresh token is permanently invalid (expired/reused/revoked)."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _auth_file(home: pathlib.Path) -> pathlib.Path:
    return home / 'auth.json'


# ---------------------------------------------------------------------------
# PKCE / state helpers
# ---------------------------------------------------------------------------

def _b64url_nopad(data: bytes) -> str:
    """Base64url-encode without trailing '=' padding."""
    return urlsafe_b64encode(data).rstrip(b'=').decode()


def _make_state() -> str:
    return _b64url_nopad(secrets.token_bytes(32))


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _write_auth(home: pathlib.Path, data: dict) -> None:
    """Write auth.json atomically with mode 0600.

    Uses a same-directory temp file + ``os.replace()`` so readers never see a
    partially-written file (POSIX atomic on same filesystem).  Cross-process
    callers must additionally hold ``_refresh_lock(home)`` to prevent races.
    """
    home.mkdir(parents=True, exist_ok=True)
    auth_file = _auth_file(home)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(dir=home, prefix='.auth_tmp_', suffix='.json')
    try:
        os.write(fd, text.encode('utf-8'))
        os.close(fd)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass  # Windows — best-effort
        os.replace(tmp_path, auth_file)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Cross-process lock
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _refresh_lock(home: pathlib.Path):
    """Cross-process exclusive lock around auth.json read-modify-write.

    Uses ``fcntl.LOCK_EX`` on ``home/auth.lock`` (POSIX).  On Windows
    (``fcntl`` unavailable) this is a no-op context manager — the in-process
    ``threading.Lock`` in ``get_access`` is the only guard there.

    The ``LOCK_EX | LOCK_NB`` variant is intentionally NOT used: we prefer
    blocking briefly over skipping the lock and triggering
    ``refresh_token_reused`` revocation.
    """
    if not _HAS_FCNTL:
        yield
        return
    home.mkdir(parents=True, exist_ok=True)
    lock_path = home / 'auth.lock'
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Refresh error classification
# ---------------------------------------------------------------------------

_TERMINAL_ERROR_CODES = frozenset({
    'refresh_token_expired',
    'refresh_token_reused',
    'refresh_token_invalidated',
})


def _classify_refresh_error(status: int, body: bytes) -> str | None:
    """Return a terminal error code string if the refresh is permanently dead, else None."""
    try:
        data = json.loads(body)
        error = data.get('error') or ''
        if isinstance(error, dict):
            code = error.get('code', '') or ''
        else:
            code = str(error) if error else ''
        if not code:
            code = data.get('code', '') or ''
        if code in _TERMINAL_ERROR_CODES:
            return code
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    if status == 401:
        return 'http_401'
    return None


# ---------------------------------------------------------------------------
# Login callback server
# ---------------------------------------------------------------------------

def run_pkce_login_server(
    port_candidates: tuple,
    callback_path: str,
    brand_name: str,
    state: str,
    authorize_url_fn,
    *,
    timeout: float = _LOGIN_TIMEOUT,
) -> tuple[str, str]:
    """Bind a local OAuth2 PKCE callback server and return ``(code, redirect_uri)``.

    ``authorize_url_fn(port)`` must return ``(auth_url, redirect_uri)`` — it is
    called after port binding so the redirect URI is known.

    Raises:
        RuntimeError: if no port in ``port_candidates`` is available, or on an
            OAuth error in the callback.
        TimeoutError: if the user does not complete login within ``timeout`` seconds.
    """
    port = None
    for candidate in port_candidates:
        try:
            _probe = HTTPServer(('127.0.0.1', candidate), BaseHTTPRequestHandler)
            _probe.server_close()
            port = candidate
            break
        except OSError:
            continue

    if port is None:
        raise RuntimeError(
            f'Could not bind login callback server on ports '
            f'{" or ".join(str(p) for p in port_candidates)}. '
            'Another process may be using those ports.'
        )

    auth_url, redirect_uri = authorize_url_fn(port)

    print(
        f'\n{brand_name} OAuth authentication required.\n'
        f'Opening browser for login... if it did not open, navigate to:\n\n'
        f'  {auth_url}\n',
        file=sys.stderr,
    )
    webbrowser.open(auth_url)

    result: dict = {}

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != callback_path:
                self.send_response(404)
                self.end_headers()
                return

            params = dict(urllib.parse.parse_qsl(parsed.query or ''))
            code = params.get('code', '')
            recv_state = params.get('state', '')
            error = params.get('error', '')

            if recv_state != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'State mismatch - possible CSRF attack.')
                result['error'] = 'state_mismatch'
                return

            if error:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f'OAuth error: {error}'.encode())
                result['error'] = error
                return

            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'No authorization code received.')
                result['error'] = 'no_code'
                return

            result['code'] = code
            html = (
                '<html><body style="font-family:sans-serif;text-align:center;margin-top:10%">'
                f'<h2>&#10003; Logged in to {brand_name}</h2>'
                '<p>You can close this tab and return to the terminal.</p>'
                '</body></html>'
            ).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, fmt, *args):  # silence request logs
            pass

    callback_server = HTTPServer(('127.0.0.1', port), _CallbackHandler)
    deadline = time.monotonic() + timeout
    while 'code' not in result and 'error' not in result:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            callback_server.server_close()
            raise TimeoutError(f'Login timed out after {timeout}s. Please try again.')
        callback_server.timeout = min(remaining, 5.0)
        callback_server.handle_request()
    callback_server.server_close()

    if 'error' in result:
        raise RuntimeError(f'Login failed: {result["error"]}')

    return result['code'], redirect_uri


# ---------------------------------------------------------------------------
# Provider descriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OAuthProvider:
    """Per-provider constants for the shared OAuth orchestration functions.

    Each ``*_auth.py`` constructs one ``_PROVIDER`` instance and delegates its
    ``ensure_credentials``, ``ensure_credentials_noninteractive``,
    ``get_access``, and ``force_refresh`` to the shared implementations below.
    """
    name: str
    """Human-readable provider name used in log / error messages (e.g. 'Codex')."""

    account_id_field: str
    """Key in the credentials dict that holds the account identifier.

    Codex: ``'account_id'``; Anthropic: ``'account_uuid'``.
    """

    restart_hint: str
    """CLI flag to include in error messages (e.g. ``'--backend codex'``)."""

    no_creds_noninteractive_msg: str
    """Full error message when ``ensure_credentials_noninteractive`` finds no creds."""

    home_fn: object
    """Callable ``(config=None) -> pathlib.Path`` — resolves the credential home dir."""

    load_credentials_fn: object
    """Callable ``(home) -> dict | None`` — loads credentials from disk."""

    needs_refresh_fn: object
    """Callable ``(creds) -> bool`` — True when access token needs refreshing."""

    refresh_fn: object
    """Callable ``(creds, home) -> dict`` — refreshes the access token."""

    login_fn: object
    """Callable ``(home) -> dict`` — runs the interactive PKCE login flow."""


# ---------------------------------------------------------------------------
# Shared orchestration functions
# ---------------------------------------------------------------------------

def ensure_credentials(provider: OAuthProvider, config=None) -> None:
    """Ensure valid credentials exist, running the login flow if necessary.

    Called once at startup (after logging is configured, before
    ``create_server``).  Proactively refreshes the access token so the first
    request is pre-warmed.

    Login is triggered when:
    - auth.json is missing or has no usable tokens, OR
    - the proactive refresh fails with a terminal error (refresh token dead).

    Transient refresh failures are logged and swallowed so the server still
    starts; the per-request refresh path will retry.
    """
    home = provider.home_fn(config)
    creds = provider.load_credentials_fn(home)

    if creds is None:
        logger.info('No %s credentials found — starting interactive login.', provider.name)
        provider.login_fn(home)
        return

    logger.info('%s credentials found — proactively refreshing tokens at startup.', provider.name)
    try:
        with _refresh_lock(home):
            # Another process may have just refreshed while we waited for the lock.
            current = provider.load_credentials_fn(home)
            if current is not None and not provider.needs_refresh_fn(current):
                logger.info('Token already fresh (refreshed by peer) — skipping startup refresh.')
                return
            provider.refresh_fn(current or creds, home)
    except TerminalRefreshError as exc:
        logger.warning('Refresh token permanently invalid (%s). Starting interactive login.', exc)
        provider.login_fn(home)
    except Exception as exc:
        logger.warning(
            'Proactive token refresh failed (non-terminal): %s. '
            'Will retry at request time.',
            exc,
        )


def ensure_credentials_noninteractive(provider: OAuthProvider, config=None) -> None:
    """Validate (and if needed refresh) credentials without any login UI.

    Used by runtime ``proxy-set-backend:<name>`` switches and the background
    ``TokenRefresher``, which must never open a browser or block on an OAuth
    callback.

    Raises:
        RuntimeError: if credentials are missing/malformed, the refresh token is
            permanently invalid, or a transient refresh failure leaves readiness
            unconfirmed.  The caller keeps the previously active backend.
    """
    home = provider.home_fn(config)
    creds = provider.load_credentials_fn(home)
    if creds is None:
        raise RuntimeError(provider.no_creds_noninteractive_msg)

    if not provider.needs_refresh_fn(creds):
        return

    with _refresh_lock(home):
        current = provider.load_credentials_fn(home)
        if current is not None and not provider.needs_refresh_fn(current):
            return  # a peer refreshed while we waited for the lock
        try:
            provider.refresh_fn(current or creds, home)
        except TerminalRefreshError as exc:
            raise RuntimeError(
                f'{provider.name} refresh token permanently invalid ({exc}). '
                f'Restart anthproxy with {provider.restart_hint} to re-login.'
            ) from exc
        except Exception as exc:
            raise RuntimeError(f'{provider.name} token refresh failed: {exc}') from exc


def get_access(
    provider: OAuthProvider,
    config,
    lock: threading.Lock,
) -> tuple[str, str | None]:
    """Return ``(access_token, account_id)``, refreshing the access token if needed.

    Thread-safe: refresh is performed under the in-process ``lock`` and the
    cross-process ``_refresh_lock`` file lock, with a double-checked load so
    only one agent actually calls the refresh endpoint under high concurrency.

    Raises:
        AnthropicRequestError(401): if credentials are missing or the refresh
            token is permanently dead.
    """
    from ..mapper import AnthropicRequestError  # lazy — avoids circular at module load

    home = provider.home_fn(config)
    creds = provider.load_credentials_fn(home)
    if creds is None:
        raise AnthropicRequestError(
            f'No {provider.name} credentials. '
            f'Restart anthproxy ({provider.restart_hint}) to log in.',
            error_type='authentication_error',
            status_code=401,
        )

    if provider.needs_refresh_fn(creds):
        with lock:
            # Double-check under in-process lock first (cheap), then under
            # cross-process file lock (blocks concurrent peers).
            with _refresh_lock(home):
                creds = provider.load_credentials_fn(home)
                if creds is None or provider.needs_refresh_fn(creds):
                    try:
                        creds = provider.refresh_fn(creds, home)
                    except TerminalRefreshError as exc:
                        raise AnthropicRequestError(
                            f'{provider.name} refresh token permanently invalid ({exc}). '
                            f'Restart anthproxy ({provider.restart_hint}) to re-login.',
                            error_type='authentication_error',
                            status_code=401,
                        ) from exc
                    except Exception as exc:
                        logger.error('Token refresh failed during request: %s', exc)
                        # Proceed with existing (possibly stale) token; let the
                        # upstream reject it.
                        if creds is None:
                            raise AnthropicRequestError(
                                f'Token refresh failed and no credentials available: {exc}',
                                error_type='authentication_error',
                                status_code=401,
                            ) from exc

    return creds['access_token'], creds.get(provider.account_id_field)


def force_refresh(
    provider: OAuthProvider,
    config,
    lock: threading.Lock,
) -> tuple[str, str | None]:
    """Unconditionally refresh the access token (called after a 401 from upstream).

    Unlike ``get_access``, this skips the ``needs_access_refresh`` check and
    always hits the refresh endpoint once.  It uses the same locking hierarchy
    (in-process → file) to avoid triggering ``refresh_token_reused``.

    Raises:
        AnthropicRequestError(401): if credentials are missing or the refresh
            token is permanently invalid.
    """
    from ..mapper import AnthropicRequestError

    home = provider.home_fn(config)
    creds = provider.load_credentials_fn(home)
    if creds is None:
        raise AnthropicRequestError(
            f'No {provider.name} credentials. '
            f'Restart anthproxy ({provider.restart_hint}) to log in.',
            error_type='authentication_error',
            status_code=401,
        )

    with lock:
        with _refresh_lock(home):
            # Re-read: a concurrent request may have already refreshed.
            fresh = provider.load_credentials_fn(home)
            if fresh is not None and not provider.needs_refresh_fn(fresh):
                # A peer just refreshed — use the new token without hitting the endpoint again.
                return fresh['access_token'], fresh.get(provider.account_id_field)
            try:
                creds = provider.refresh_fn(fresh or creds, home)
            except TerminalRefreshError as exc:
                raise AnthropicRequestError(
                    f'{provider.name} refresh token permanently invalid ({exc}). '
                    f'Restart anthproxy ({provider.restart_hint}) to re-login.',
                    error_type='authentication_error',
                    status_code=401,
                ) from exc
            except Exception as exc:
                logger.error('Forced token refresh failed: %s', exc)
                raise AnthropicRequestError(
                    f'Token refresh failed: {exc}',
                    error_type='authentication_error',
                    status_code=401,
                ) from exc

    return creds['access_token'], creds.get(provider.account_id_field)
