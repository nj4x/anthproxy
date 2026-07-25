"""Periodic OAuth token refresher for subscription backends.

``TokenRefresher`` runs as a daemon thread and proactively refreshes access
tokens for both subscription backends (anthropic and codex) on a fixed interval,
regardless of which backend is currently active.  This keeps inactive tokens
warm so that backend switching never incurs a stale-token request.

The refresh calls delegate to the existing ``ensure_credentials_noninteractive``
functions, which:
  - do nothing when the token is still fresh (``needs_access_refresh`` is False)
  - call the refresh endpoint and write the new token to disk when within the
    5-minute window (or the 8-day fallback)
  - raise ``RuntimeError`` when credentials are absent (skipped silently)
  - propagate ``TerminalRefreshError`` wrapped in ``RuntimeError`` on dead tokens

Every actual token exchange logs at INFO level (``… tokens refreshed
successfully.``) inside ``refresh()``, so periodic refreshes are visible in
the normal console output without extra noise.
"""
import logging
import threading

logger = logging.getLogger(__name__)


class TokenRefresher:
    """Background daemon that periodically refreshes subscription OAuth tokens.

    Usage::

        refresher = TokenRefresher(config)
        refresher.start()
        try:
            server.serve_forever()
        finally:
            refresher.stop()
    """

    def __init__(self, config, interval: float | None = None):
        self._config = config
        self._interval = interval if interval is not None else config.auto_backend_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background refresh thread (daemon)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='token-refresher', daemon=True)
        self._thread.start()
        logger.info('Token refresher started (interval=%.0fs)', self._interval)

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background poll loop."""
        while not self._stop.wait(self._interval):
            self._refresh_all()

    def _refresh_all(self) -> None:
        """Attempt to refresh tokens for every subscription backend."""
        self._refresh_one('anthropic')
        self._refresh_one('codex')

    def _refresh_one(self, name: str) -> None:
        """Refresh a single backend's token, swallowing expected non-errors."""
        try:
            if name == 'anthropic':
                from .anthropic import auth as anthropic_auth
                logger.debug('Token refresher: checking anthropic credentials')
                anthropic_auth.ensure_credentials_noninteractive(self._config)
            elif name == 'codex':
                from .codex import auth as codex_auth
                logger.debug('Token refresher: checking codex credentials')
                codex_auth.ensure_credentials_noninteractive(self._config)
        except RuntimeError as exc:
            msg = str(exc)
            # Missing credentials are expected (backend not set up) — debug only.
            if 'No ' in msg and 'credentials' in msg.lower():
                logger.debug('Token refresher: %s credentials absent, skipping', name)
            else:
                logger.warning('Token refresher: %s refresh failed: %s', name, exc)
        except Exception:
            logger.exception('Token refresher: unexpected error refreshing %s', name)
