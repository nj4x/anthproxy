import dataclasses
import logging
import math
import threading
import time
from collections import OrderedDict
from http.server import ThreadingHTTPServer

from .backends_registry import (  # noqa: F401
    BackendDiscoveryError,
    backend_names,
    class_hook,
    discover_backends,
    get_backend,
    list_backends,
)
from .config import Config
from .constants import SUBSCRIPTION_BACKENDS, SESSION_SUBSCRIPTION_SENTINEL
from .handlers import ProxyRequestHandler

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Raised when a backend cannot be constructed or prepared for activation."""


def build_backend(name: str, config: Config):
    """Construct a backend instance by canonical name via the from_config hook.

    Codex credential preparation is intentionally NOT performed here — startup
    handles interactive login, and runtime switches use a non-interactive
    readiness check before constructing the backend.
    """
    backend_class = get_backend(name)
    if backend_class is None:
        raise BackendError(f'Unknown backend: {name}')
    factory = class_hook(backend_class, 'from_config')
    if factory is None:
        raise BackendError(
            f'{backend_class.__module__}.{backend_class.__qualname__}: missing or '
            f'malformed from_config hook — inherit anthproxy._shared.Backend or '
            f'declare a classmethod named from_config'
        )
    return factory(config)


@dataclasses.dataclass(frozen=True)
class BackendSnapshot:
    """Immutable view of the active backend captured once per request."""
    name: str
    backend: object
    config: Config
    session_pinned: bool = False
    session_subscription: bool = False
    credentials: object | None = None


@dataclasses.dataclass(frozen=True)
class SwitchResult:
    kind: str  # 'changed' | 'unchanged' | 'invalid' | 'failed'
    previous: str
    current: str
    error: str | None = None


def _format_usage_snapshot(name: str, cache: dict) -> dict:
    """Convert a backend's raw ``_usage_cache`` to the usage-snapshot shape.

    Anthropic stores ``{five_hour: {utilization, resets_at}, seven_day: ...}``.
    Codex stores ``{primary: {used_percent, reset_at}, weekly: ...}``.
    OpenRouter stores ``{data: {total_credits, total_usage}}``.

    Returns ``{}`` when the cache has no recognisable content.
    """
    import datetime as _dt

    def _reset_in_secs(reset_at_iso: str | None) -> int | None:
        if not reset_at_iso:
            return None
        try:
            dt = _dt.datetime.fromisoformat(reset_at_iso.replace('Z', '+00:00'))
            secs = (dt - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
            return int(secs)
        except Exception:  # noqa: BLE001
            return None

    if name == 'anthropic':
        def _fmt_anthropic(window: dict | None) -> dict | None:
            if not isinstance(window, dict):
                return None
            pct_raw = window.get('utilization')
            reset_at = window.get('resets_at')
            try:
                pct: float | None = float(pct_raw)
            except (TypeError, ValueError):
                pct = None
            return {
                'pct': pct,
                'reset_at': reset_at,
                'reset_in_secs': _reset_in_secs(reset_at),
            }

        five_hour = _fmt_anthropic(cache.get('five_hour'))
        weekly = _fmt_anthropic(cache.get('seven_day'))
        if five_hour is not None:
            five_hour['window_hours'] = 5
        if weekly is not None:
            weekly['window_hours'] = 168
        return {'five_hour': five_hour, 'weekly': weekly}

    if name == 'codex':
        def _fmt_codex(window: dict | None) -> dict | None:
            if not isinstance(window, dict):
                return None
            pct_raw = window.get('used_percent')
            reset_at_unix = window.get('reset_at')
            reset_at_iso: str | None = None
            secs_remaining: int | None = None
            if isinstance(reset_at_unix, (int, float)):
                try:
                    dt = _dt.datetime.fromtimestamp(float(reset_at_unix),
                                                    tz=_dt.timezone.utc)
                    reset_at_iso = dt.isoformat()
                    secs_remaining = int(float(reset_at_unix)
                                        - _dt.datetime.now(_dt.timezone.utc).timestamp())
                except Exception:  # noqa: BLE001
                    pass
            try:
                pct: float | None = float(pct_raw)
            except (TypeError, ValueError):
                pct = None
            window_seconds = window.get('window_seconds')
            try:
                window_hours = float(window_seconds) / 3600
            except (TypeError, ValueError):
                window_hours = None
            return {
                'pct': pct,
                'reset_at': reset_at_iso,
                'reset_in_secs': secs_remaining,
                'window_hours': window_hours,
            }

        primary = cache.get('primary')
        weekly = cache.get('weekly')
        primary_usage = _fmt_codex(primary)
        weekly_usage = _fmt_codex(weekly)
        if primary is weekly:
            primary_usage = None
        return {'five_hour': primary_usage, 'weekly': weekly_usage}

    if name == 'openrouter':
        data_block = cache.get('data')
        if not isinstance(data_block, dict):
            return {}
        try:
            total: float | None = float(data_block.get('total_credits', 0) or 0)
            used: float | None = float(data_block.get('total_usage', 0) or 0)
            pct: float | None = (used / total * 100.0) if (total and total > 0) else None
        except (TypeError, ValueError):
            total = used = pct = None
        return {
            'credits': {
                'used_usd': used,
                'total_usd': total,
                'pct': pct,
            }
        }

    return {}


class BackendRegistry:
    """Process-wide owner of the active backend and cached instances.

    One instance per backend is cached for the lifetime of the server so that
    backend-local state (Bedrock credential cache,
    Codex OAuth/usage cache) survives switching away and back.

    The state lock guards only the active-name/config commit, the instance
    cache, and the per-session override map. It is never held across inference,
    streaming, token refresh, or any network I/O. Candidate preparation/
    construction happens before the commit.
    """

    _MAX_SESSION_OVERRIDES = 1024  # cap on the session override map size

    def __init__(self, config: Config, initial_backend, oauth_registry=None):
        self._config = config
        self._oauth_registry = oauth_registry
        self._active = config.backend
        self._instances = {config.backend: initial_backend}
        self._state_lock = threading.Lock()
        # Serialises candidate construction/preparation so concurrent switches
        # to the same backend do not build duplicate instances.
        self._prepare_lock = threading.Lock()
        # Per-session backend overrides keyed on metadata.user_id.  Bounded to
        # _MAX_SESSION_OVERRIDES entries; oldest entries are evicted on overflow
        # so a long-lived proxy does not grow unboundedly.
        self._session_overrides: OrderedDict[str, str] = OrderedDict()
        # Per-session model-routing overrides (bool).  Same bounded-eviction
        # policy as _session_overrides.
        self._session_routing_overrides: OrderedDict[str, bool] = OrderedDict()
        # Per-session routed-tier cache (str: bare short alias produced by the
        # last successful classification, e.g. 'haiku'/'sonnet'/'opus').  Reused
        # on text-less continuation turns so a task stays on a consistent tier
        # without an extra classifier call.  Same bounded-eviction policy.
        self._session_routed_tier: OrderedDict[str, str] = OrderedDict()
        # Per-session observed-context cache: (measured_floor, est_ratio).
        # measured_floor is the most recent response's total context size
        # (input + cache_read + cache_creation + output) — a lower bound on the
        # next turn's input.  est_ratio is the last actual/estimate calibration
        # factor (clamped >= 1.0).  Fed into the long-context size floor in
        # model_router so a session already near the window forces opus[1m] even
        # on a small turn.  Same bounded-eviction policy as the other maps.
        self._session_context_obs: OrderedDict[str, tuple[int, float]] = OrderedDict()
        # Optional callable (set by create_server when a selector is present)
        # that returns a subscription backend name from cached selector state.
        # Called under _state_lock so it must be a leaf (no network I/O, no
        # registry callbacks).
        self._subscription_resolver = None
        self._personal_candidates_resolver = None

    def set_personal_candidates_resolver(self, resolver) -> None:
        self._personal_candidates_resolver = resolver

    def observe_oauth_credential(self, access_token: str):
        if self._oauth_registry is None:
            return None
        return self._oauth_registry.observe(access_token)

    def mark_oauth_cooldown(self, credential, retry_after: float | None = None) -> bool:
        if self._oauth_registry is None or credential is None:
            return False
        return self._oauth_registry.mark_cooldown(credential.generation, retry_after)

    def snapshot_for_request(
        self,
        session_key: str | None = None,
        prefer_backend: str | None = None,
        oauth_credential=None,
    ) -> BackendSnapshot:
        with self._state_lock:
            override = self._session_overrides.get(session_key) if session_key is not None else None
        if override is not None:
            return self.snapshot(session_key, prefer_backend=prefer_backend)

        if prefer_backend is not None:
            return self.snapshot(session_key, prefer_backend=prefer_backend)

        candidates = (
            tuple(self._personal_candidates_resolver())
            if self._personal_candidates_resolver is not None else ()
        )
        personal = min(candidates, key=lambda candidate: candidate.burn, default=None)
        if personal is None:
            base = self.snapshot(session_key)
            personal_name = base.name
            personal_burn = 100.0
        else:
            personal_name = personal.name
            personal_burn = personal.burn

        oauth = (
            self._oauth_registry.snapshot(oauth_credential.generation)
            if self._oauth_registry is not None and oauth_credential is not None
            else None
        )
        oauth_valid = oauth is not None and oauth.eligible
        oauth_wins = (
            oauth_valid
            and oauth.burn is not None
            and oauth.burn < personal_burn
            and not math.isclose(oauth.burn, personal_burn, rel_tol=1e-6)
        )
        if oauth_wins:
            with self._state_lock:
                backend = self._instances.get('oauth')
            if backend is None:
                backend = self._get_or_create('oauth')
            request_config = dataclasses.replace(self._config, backend='oauth')
            return BackendSnapshot(
                name='oauth', backend=backend, config=request_config,
                credentials=oauth_credential,
            )

        with self._state_lock:
            backend = self._instances.get(personal_name)
        if backend is None:
            return self.snapshot(session_key)
        request_config = dataclasses.replace(self._config, backend=personal_name)
        session_routing_override = self.session_model_routing(session_key) \
            if session_key is not None else None
        if session_routing_override is not None:
            request_config = dataclasses.replace(
                request_config, auto_model_routing=session_routing_override,
            )
        return BackendSnapshot(name=personal_name, backend=backend, config=request_config)

    def snapshot(self, session_key: str | None = None,
                 prefer_backend: str | None = None) -> BackendSnapshot:
        # Phase 1: resolve session override and capture per-request state under
        # _state_lock.  The preference gate is checked here (session overrides
        # take precedence, subscription-only mode excludes bedrock), and the
        # cached preferred instance (if any) is captured for the unlocked health
        # check in Phase 2.
        need_prepare: str | None = None
        prefer_cached: object | None = None

        with self._state_lock:
            if session_key is not None:
                override = self._session_overrides.get(session_key)
            else:
                override = None
            if override is not None:
                if override == SESSION_SUBSCRIPTION_SENTINEL:
                    name = None
                    if self._subscription_resolver is not None:
                        resolved = self._subscription_resolver()
                        if resolved in SUBSCRIPTION_BACKENDS:
                            name = resolved
                    if name is None:
                        name = self._active if self._active in SUBSCRIPTION_BACKENDS else 'anthropic'
                    backend = self._instances.get(name) or self._instances[self._active]
                    session_subscription = True
                else:
                    name = override
                    backend = self._instances.get(name) or self._instances[self._active]
                    session_subscription = False
                session_pinned = True
            else:
                name = self._active
                backend = self._instances[name]
                session_pinned = False
                session_subscription = False

            # Per-request backend preference gate — only open when no session
            # override is active and the backend name is valid and allowed in the
            # current mode.
            if (
                prefer_backend is not None
                and prefer_backend in backend_names()
                and not session_pinned
                and not session_subscription
                and not (self._config.auto_backend_mode == 'subscription'
                         and prefer_backend not in SUBSCRIPTION_BACKENDS)
            ):
                prefer_cached = self._instances.get(prefer_backend)
                if prefer_cached is None:
                    need_prepare = prefer_backend

            # Capture session routing override under the lock (reads from a dict
            # that is mutated under _state_lock), matching the existing pattern at
            # the original line ~148.
            session_routing_override = self._session_routing_overrides.get(session_key) \
                if session_key is not None else None

        # Phase 2: unlocked health check on the cached preferred backend.
        # five_hour_status() performs network I/O for subscription backends
        # and must NOT be called under _state_lock (CLAUDE.md lock discipline).
        # Conservative: honor the preference unless the backend is confirmed
        # unavailable (available is False).  Unknown health (None) or a
        # five_hour_status() exception both honor the preference.
        if prefer_cached is not None:
            try:
                status = prefer_cached.five_hour_status(self._config)
            except Exception:
                status = None
            confirmed_unavailable = status is not None and status.available is False
            if not confirmed_unavailable and prefer_backend != name:
                name = prefer_backend  # type: ignore[assignment]
                backend = prefer_cached

        # Phase 3: construct the preferred backend on demand under _prepare_lock.
        # Mirrors the existing pattern in switch() at server.py:215 and
        # set_session_backend() at server.py:257.  Fallback recursion runs
        # OUTSIDE the lock so concurrent unrelated prepares are not stalled.
        if need_prepare is not None:
            prepare_failed = False
            with self._prepare_lock:
                try:
                    prepared = self._prepare_candidate(need_prepare)
                except Exception as exc:
                    logger.warning(
                        'Per-request backend preference: preparing %s failed: %s '
                        '-- falling back to standard selector',
                        need_prepare, exc,
                    )
                    prepare_failed = True
            if prepare_failed:
                return self.snapshot(session_key, prefer_backend=None)
            # Health check on the freshly-prepared instance (unlocked).
            try:
                status = prepared.five_hour_status(self._config)  # type: ignore[possibly-unbound]
            except Exception:
                status = None
            if status is None or status.available is not False:
                name = need_prepare
                backend = prepared  # type: ignore[assignment]

        # Build request_config with session routing override applied.
        request_config = dataclasses.replace(self._config, backend=name)
        if session_routing_override is not None:
            request_config = dataclasses.replace(request_config,
                                                 auto_model_routing=session_routing_override)
        return BackendSnapshot(name=name, backend=backend, config=request_config,
                               session_pinned=session_pinned,
                               session_subscription=session_subscription)

    def active_name(self) -> str:
        with self._state_lock:
            return self._active

    def _get_or_create(self, name: str):
        with self._state_lock:
            existing = self._instances.get(name)
        if existing is not None:
            return existing
        backend = build_backend(name, self._config)
        with self._state_lock:
            # Another thread may have created it while we built ours; prefer the
            # already-published instance so state stays single-sourced.
            published = self._instances.get(name)
            if published is not None:
                return published
            self._instances[name] = backend
            return backend

    def instance(self, name: str):
        """Return the cached backend instance for ``name``, creating it if needed.

        Used by ``AutoSelector`` to call ``five_hour_status()`` without
        triggering a full switch.  Raises ``BackendError`` for unknown names.
        """
        if name not in backend_names():
            raise BackendError(f'Unknown backend: {name}')
        return self._get_or_create(name)

    def _prepare_candidate(self, name: str):
        """Ensure credentials are fresh and a backend instance exists for ``name``.

        Runs non-interactive credential readiness for codex/anthropic, then
        builds or reuses the cached instance.  Raises on failure.  Must be
        called under ``_prepare_lock`` to serialise concurrent switches to the
        same backend.
        """
        if name == 'codex':
            from .codex import auth as codex_auth
            codex_auth.ensure_credentials_noninteractive(self._config)
        if name == 'anthropic':
            from .anthropic import auth as anthropic_auth
            anthropic_auth.ensure_credentials_noninteractive(self._config)
        if name == 'openrouter' and not self._config.openrouter_api_key:
            raise BackendError('OPENROUTER_API_KEY is not set')
        return self._get_or_create(name)

    def switch(self, name: str, reason: str = '') -> SwitchResult:
        if name not in backend_names():
            return SwitchResult(kind='invalid', previous=self.active_name(),
                                current=self.active_name())

        previous = self.active_name()
        if name == previous:
            return SwitchResult(kind='unchanged', previous=previous, current=previous)

        # Prepare/construct the candidate outside the state lock so ordinary
        # requests keep flowing during (bounded) readiness work.
        with self._prepare_lock:
            try:
                backend = self._prepare_candidate(name)
            except Exception as exc:  # noqa: BLE001 — surfaced as a local command reply
                logger.warning('Backend switch to %s failed: %s', name, exc)
                return SwitchResult(kind='failed', previous=previous,
                                    current=previous, error=str(exc))

        with self._state_lock:
            previous = self._active
            self._active = name
            self._instances[name] = backend
            # Mirror into _config so code that holds a reference to the global
            # Config object (e.g. handler.config) can read the current backend
            # name.  _active is the authoritative source; snapshot() always
            # derives its backend from _active, not from _config.backend.
            self._config.backend = name
        logger.info('Backend switched: %s → %s%s', previous, name,
                    f' ({reason})' if reason else '')
        return SwitchResult(kind='changed', previous=previous, current=name)

    def set_session_backend(self, session_key: str, name: str) -> SwitchResult:
        """Pin ``session_key`` to ``name`` without touching the global active backend.

        The per-session override takes precedence over the global active in
        ``snapshot()``.  The existing ``_instances`` cache is shared; ``name``'s
        instance is prepared (credentials refreshed, backend constructed) before
        recording the override so that per-session pins get the same credential-
        readiness guarantee as global switches.

        Returns a ``SwitchResult`` whose ``previous`` is the prior override for
        this session (or the current global active when no override existed).
        """
        if name not in backend_names():
            return SwitchResult(kind='invalid', previous=self.active_name(),
                                current=self.active_name())

        with self._state_lock:
            previous = self._session_overrides.get(session_key) or self._active
            if self._session_overrides.get(session_key) == name:
                return SwitchResult(kind='unchanged', previous=name, current=name)

        with self._prepare_lock:
            try:
                backend = self._prepare_candidate(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Session backend pin to %s failed for %s: %s',
                                name, session_key, exc)
                return SwitchResult(kind='failed', previous=previous,
                                    current=previous, error=str(exc))

        with self._state_lock:
            # Re-read previous under lock for accuracy after prepare_lock gap
            previous = self._session_overrides.get(session_key) or self._active
            self._instances[name] = backend
            if len(self._session_overrides) >= self._MAX_SESSION_OVERRIDES:
                self._session_overrides.popitem(last=False)  # evict oldest
            self._session_overrides[session_key] = name
            self._session_overrides.move_to_end(session_key)  # mark recently used
        logger.info('Session backend pinned: %s → %s (session=%s…)',
                    previous, name, session_key[:16])
        return SwitchResult(kind='changed', previous=previous, current=name)

    def clear_session_backend(self, session_key: str) -> bool:
        """Remove the per-session override for ``session_key``.

        Returns ``True`` if an override existed and was removed, ``False`` if
        the session had no override.
        """
        with self._state_lock:
            existed = session_key in self._session_overrides
            if existed:
                del self._session_overrides[session_key]
        if existed:
            logger.info('Session backend override cleared (session=%s…)', session_key[:16])
        return existed

    def session_backend(self, session_key: str) -> str | None:
        """Return the per-session backend override, or ``None`` if none is set.

        May return ``SESSION_SUBSCRIPTION_SENTINEL`` if the session is locked
        to subscription backends.  Callers that display the value must
        special-case the sentinel.
        """
        with self._state_lock:
            return self._session_overrides.get(session_key)

    def set_model_routing(self, enabled: bool) -> None:
        """Toggle auto model routing on or off globally.

        Mutates ``_config.auto_model_routing`` under ``_state_lock`` so the
        change propagates to every subsequent ``snapshot()`` call and from there
        to ``route_model()``.  Per-session overrides (``set_session_model_routing``)
        take precedence over the global flag.
        """
        with self._state_lock:
            self._config.auto_model_routing = enabled
        logger.info('Auto model routing %s (global)', 'enabled' if enabled else 'disabled')

    def set_session_model_routing(self, session_key: str, enabled: bool | None) -> None:
        """Set or clear a per-session model-routing override.

        ``enabled=True/False`` pins model routing on or off for this session,
        overriding the global flag.  ``enabled=None`` clears any existing
        session override so the session follows the global flag again.
        """
        with self._state_lock:
            if enabled is None:
                self._session_routing_overrides.pop(session_key, None)
            else:
                if len(self._session_routing_overrides) >= self._MAX_SESSION_OVERRIDES:
                    self._session_routing_overrides.popitem(last=False)
                self._session_routing_overrides[session_key] = enabled
                self._session_routing_overrides.move_to_end(session_key)
        if enabled is None:
            logger.info('Session model routing override cleared (session=%s…)',
                        session_key[:16])
        else:
            logger.info('Session model routing %s (session=%s…)',
                        'enabled' if enabled else 'disabled', session_key[:16])

    def session_model_routing(self, session_key: str) -> bool | None:
        """Return the per-session model-routing override, or ``None`` if not set."""
        with self._state_lock:
            return self._session_routing_overrides.get(session_key)

    def set_session_routed_tier(self, session_key: str, tier: str) -> None:
        """Cache the most recent successfully routed tier for a session.

        Stores the bare short alias (``'haiku'``, ``'sonnet'``, ``'opus'``)
        produced by the last successful classifier call for this session.  Used
        by the handler to reuse the prior tier on text-less continuation turns
        (e.g. ``tool_result``-only final messages) without an extra classifier
        call.  Bounded to ``_MAX_SESSION_OVERRIDES`` entries; oldest entry is
        evicted on overflow.  Must be called only under normal routing (not on
        classifier sentinel payloads).
        """
        with self._state_lock:
            if len(self._session_routed_tier) >= self._MAX_SESSION_OVERRIDES:
                self._session_routed_tier.popitem(last=False)
            self._session_routed_tier[session_key] = tier
            self._session_routed_tier.move_to_end(session_key)

    def session_routed_tier(self, session_key: str) -> str | None:
        """Return the cached routed tier for a session, or ``None`` if not set."""
        with self._state_lock:
            return self._session_routed_tier.get(session_key)

    def record_session_context(
        self, session_key: str, measured_floor: int, est_ratio: float,
    ) -> None:
        """Record the latest observed context size and calibration ratio for a session.

        ``measured_floor`` is the most recent response's total context size
        (``input + cache_read + cache_creation + output`` tokens); ``est_ratio``
        is the latest actual/estimate calibration factor.  Replace semantics (not
        a running sum): a response already includes the full resent history, so we
        keep the newest measurement, which also shrinks correctly after a context
        compaction.  Bounded to ``_MAX_SESSION_OVERRIDES`` entries; oldest entry
        is evicted on overflow.
        """
        with self._state_lock:
            if len(self._session_context_obs) >= self._MAX_SESSION_OVERRIDES:
                self._session_context_obs.popitem(last=False)
            self._session_context_obs[session_key] = (measured_floor, est_ratio)
            self._session_context_obs.move_to_end(session_key)

    def session_context(self, session_key: str) -> tuple[int, float]:
        """Return ``(measured_floor, est_ratio)`` for a session.

        Defaults to ``(0, 1.0)`` when nothing is cached — i.e. no floor and an
        identity calibration, reproducing the per-request-estimate-only behavior.
        """
        with self._state_lock:
            return self._session_context_obs.get(session_key, (0, 1.0))

    def set_session_tier_global(self, session_id: str, tier: str | None) -> None:
        """Pin or clear the tier for all ctx_keys belonging to ``session_id``.

        ``session_id`` is the raw ``metadata.user_id`` value.  All ctx_keys of
        the form ``session_id\\x00<anchor>`` are updated atomically under the
        state lock.  ``tier=None`` removes those entries; a string value
        overwrites them with the given tier alias (``'haiku'``, ``'sonnet'``,
        ``'opus'``).  Used by the admin UI to force a session onto a specific
        model tier regardless of the classifier's decision.
        """
        prefix = session_id + '\x00'
        with self._state_lock:
            keys_to_update = [k for k in self._session_routed_tier
                              if k.startswith(prefix)]
            for k in keys_to_update:
                if tier is None:
                    del self._session_routed_tier[k]
                else:
                    self._session_routed_tier[k] = tier
        if keys_to_update:
            action = 'cleared' if tier is None else f'forced to {tier!r}'
            logger.info('Session tier %s for session=%s… (%d ctx_keys updated)',
                        action, session_id[:16], len(keys_to_update))

    def backend_status(self) -> dict:
        """Return cached backend availability state — no network I/O.

        Returns a dict mapping backend name to a summary dict with keys
        ``name``, ``active``, and ``available`` (cached boolean or None when
        no health state has been observed yet).
        """
        with self._state_lock:
            active = self._active
            result = {}
            for name, instance in self._instances.items():
                avail = getattr(instance, '_available', None)
                result[name] = {
                    'name': name,
                    'active': name == active,
                    'available': avail,
                }
        return result

    def usage_snapshot(self) -> dict:
        """Return cached subscription usage without making network calls.

        Reads each subscription backend's ``_usage_cache`` (populated by
        GET /admin/status on-demand fetches and by the selector's evaluate()
        probes when auto-selection is active).  Never holds ``_state_lock``
        during the read — it captures a shallow copy of ``_instances`` under
        the lock then reads the (thread-safe) cache attribute outside it.

        Returns a dict mapping backend_name → per-backend usage summary.
        Each entry includes an ``age_secs`` key: the integer number of seconds
        since the cache was last populated, or ``None`` when never populated.
        Returns ``{}`` when no subscription backend has cached data yet.
        """
        with self._state_lock:
            instances = dict(self._instances)

        result = {}
        for name in SUBSCRIPTION_BACKENDS:
            backend = instances.get(name)
            if backend is None:
                continue
            cache = getattr(backend, '_usage_cache', None)
            cached_at = getattr(backend, '_usage_cached_at', 0.0)
            if not isinstance(cache, dict):
                continue
            entry = _format_usage_snapshot(name, cache)
            if entry:
                age = int(time.monotonic() - cached_at) if cached_at > 0.0 else None
                entry['age_secs'] = age
                result[name] = entry
        return result

    def cached_subscription_instances(self) -> dict:
        """Return a {name: backend} snapshot for subscription backends already in cache.

        Never creates new instances. Never holds the lock across I/O.
        """
        with self._state_lock:
            return {
                name: self._instances[name]
                for name in SUBSCRIPTION_BACKENDS
                if name in self._instances
            }

    def list_backends(self) -> tuple:
        """Return all known backend names."""
        return backend_names()

    @property
    def config(self):
        """Read-only view of the current server configuration."""
        return self._config

    def set_auto_backend_mode(self, mode: str) -> None:
        """Update the global auto_backend_mode preference.

        Mutates ``_config.auto_backend_mode`` under ``_state_lock`` so the
        change propagates to every subsequent ``snapshot()`` call.
        """
        with self._state_lock:
            self._config.auto_backend_mode = mode
        logger.info('Auto backend mode set: %s', mode)

    def set_subscription_resolver(self, fn) -> None:
        """Register a callable that returns a subscription backend name.

        The callable is invoked inside ``snapshot()`` while ``_state_lock`` is
        held, so it MUST be a leaf: no network I/O, no ``registry.*`` callbacks,
        only read-only access to selector cached state.
        """
        self._subscription_resolver = fn

    def set_session_subscription(self, session_key: str) -> 'SwitchResult':
        """Lock ``session_key`` to subscription backends without touching the global active.

        Pre-creates configured subscription backend instances so subsequent
        ``snapshot()`` calls can resolve to any of them without a cold build.
        Succeeds if at least one subscription backend prepares successfully;
        fails only when all fail.

        Returns a ``SwitchResult`` whose ``current`` is
        ``SESSION_SUBSCRIPTION_SENTINEL``.
        """
        previous: str
        with self._state_lock:
            existing = self._session_overrides.get(session_key)
            previous = existing if existing is not None else self._active
            if existing == SESSION_SUBSCRIPTION_SENTINEL:
                return SwitchResult(kind='unchanged', previous=previous,
                                    current=SESSION_SUBSCRIPTION_SENTINEL)

        prepared_any = False
        last_error: str | None = None
        with self._prepare_lock:
            for sub in SUBSCRIPTION_BACKENDS:
                try:
                    self._prepare_candidate(sub)
                    prepared_any = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning('Session subscription: preparing %s failed: %s', sub, exc)
                    last_error = str(exc)

        if not prepared_any:
            return SwitchResult(kind='failed', previous=previous,
                                current=previous, error=last_error)

        with self._state_lock:
            previous = self._session_overrides.get(session_key) or self._active
            if len(self._session_overrides) >= self._MAX_SESSION_OVERRIDES:
                self._session_overrides.popitem(last=False)  # evict oldest
            self._session_overrides[session_key] = SESSION_SUBSCRIPTION_SENTINEL
            self._session_overrides.move_to_end(session_key)  # mark recently used
        logger.info('Session subscription lock set (session=%s…)', session_key[:16])
        return SwitchResult(kind='changed', previous=previous,
                            current=SESSION_SUBSCRIPTION_SENTINEL)


def make_handler_class(registry: BackendRegistry, config: Config,
                       selector=None, stats_collector=None, session_db=None):
    class Handler(ProxyRequestHandler):
        pass
    Handler.registry = registry
    Handler.config = config
    Handler.selector = selector
    Handler.stats_collector = stats_collector
    Handler.session_db = session_db
    Handler.enable_ui = getattr(config, 'enable_ui', False)
    return Handler


def create_server(config: Config, registry: BackendRegistry,
                  selector=None, stats_collector=None,
                  session_db=None) -> ThreadingHTTPServer:
    if selector is not None:
        registry.set_subscription_resolver(selector.current_subscription_backend)
        registry.set_personal_candidates_resolver(selector.personal_candidates)
    handler_class = make_handler_class(registry, config, selector, stats_collector,
                                       session_db=session_db)
    server = ThreadingHTTPServer((config.host, config.port), handler_class)
    if getattr(config, 'enable_ui', False) and config.host not in ('127.0.0.1', 'localhost', '::1', ''):
        logging.getLogger(__name__).warning(
            'SECURITY: --enable-ui is active and the server is bound to %s. '
            'The admin UI has no authentication — any host on this network can '
            'read conversation history and control routing. Bind to 127.0.0.1 '
            'for local-only access.', config.host
        )
    return server
