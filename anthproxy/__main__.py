import logging
import sys

from . import model_config
from .config import parse_args
from .server import BackendRegistry, build_backend, create_server, discover_backends


class _ShortNameFormatter(logging.Formatter):
    """Strip the leading 'anthproxy.' package prefix from %(name)s.

    Keeps the hierarchy intact for filtering (child loggers still propagate
    to the 'anthproxy' root handler) while producing compact output:
      INFO handlers: [23cf115f 4706654d +0.01s] ...
    instead of:
      INFO anthproxy.handlers: [23cf115f 4706654d +0.01s] ...
    """

    _PREFIX = 'anthproxy.'

    def format(self, record: logging.LogRecord) -> str:
        # Mutate a copy of the record so the original is never changed
        # (other handlers may still format it with the full name).
        if record.name.startswith(self._PREFIX):
            record = logging.makeLogRecord(record.__dict__)
            record.name = record.name[len(self._PREFIX):]
        return super().format(record)


logger = logging.getLogger('anthproxy')


def main():
    discover_backends()

    config = parse_args()

    root = logging.getLogger('anthproxy')
    root.setLevel(logging.DEBUG)

    # Console handler — at the user-configured level (default INFO)
    _fmt = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, config.log_level))
    console.setFormatter(_ShortNameFormatter(_fmt))
    root.addHandler(console)

    if config.log_file:
        fh = logging.FileHandler(config.log_file)
        fh.setLevel(getattr(logging, config.log_level))
        fh.setFormatter(_ShortNameFormatter(_fmt))
        root.addHandler(fh)

    # Write default ~/.anthproxy/config.json on first run so users have an
    # editable template for model aliases, pricing, etc.
    model_config.ensure_file()

    if config.auto_backend:
        # Auto mode: ensure credentials for both subscription backends regardless
        # of --backend.  Interactive login may run for each.  Priority order:
        # anthropic first so its login prompt appears before codex's.
        from .anthropic import auth as anthropic_auth
        from .codex import auth as codex_auth
        try:
            anthropic_auth.ensure_credentials(config)
        except Exception as exc:
            logger.warning(
                'Anthropic credential setup failed; anthropic backend will be unavailable: %s', exc)
        try:
            codex_auth.ensure_credentials(config)
        except Exception as exc:
            logger.warning(
                'Codex credential setup failed; codex backend will be unavailable: %s', exc)
    else:
        # Static mode: only prepare the single configured backend.
        if config.backend == 'codex':
            from .codex import auth as codex_auth
            codex_auth.ensure_credentials(config)
        elif config.backend == 'anthropic':
            from .anthropic import auth as anthropic_auth
            anthropic_auth.ensure_credentials(config)

    backend = build_backend(config.backend, config)
    from ._shared.oauth_usage import fetch_oauth_usage
    from .oauth_registry import OAuthTokenRegistry

    oauth_registry = OAuthTokenRegistry(usage_probe=fetch_oauth_usage)
    registry = BackendRegistry(config, backend, oauth_registry=oauth_registry)

    from .selector import AutoSelector
    selector = AutoSelector(registry, config, oauth_registry=oauth_registry)
    if config.auto_backend:
        if config.backend == 'local':
            # The initial --backend local was explicit; pin it so the startup
            # evaluate() doesn't immediately switch away to a subscription backend.
            selector.pin('local')
        else:
            # Evaluate once synchronously so the first request lands on the best
            # backend rather than whatever --backend was specified.
            selector.evaluate()

    from .stats import StatsCollector
    stats_collector = StatsCollector()
    logger.info('Stats collection enabled: %s', stats_collector._dir)

    session_db = None
    summary_daemon = None
    if config.enable_ui or config.db_path:
        from pathlib import Path as _Path
        from .db import SessionDB
        _db_path = config.db_path or str(_Path.home() / '.anthproxy' / 'anthproxy.db')
        session_db = SessionDB(_db_path)
        session_db.start_retention_daemon()
        logger.info('Session DB enabled: %s', _db_path)

        if config.enable_ui:
            from .summary import SummaryDaemon
            summary_daemon = SummaryDaemon(session_db, registry)

    server = create_server(config, registry, selector, stats_collector,
                           session_db=session_db)
    logger.info('anthproxy listening on %s:%d (backend=%s%s)',
                config.host, config.port, config.backend,
                ', auto-backend=on' if config.auto_backend else '')

    try:
        selector.start()
        if summary_daemon is not None:
            summary_daemon.start()
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('Shutting down.')
    finally:
        server.shutdown()
        if summary_daemon is not None:
            summary_daemon.stop()
        selector.stop()
        if session_db is not None:
            session_db.stop_retention_daemon()
            session_db.close()
        sys.exit(0)


if __name__ == '__main__':
    main()
