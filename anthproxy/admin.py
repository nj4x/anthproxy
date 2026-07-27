"""Admin REST API for anthproxy web UI.

Provides pure handler functions called by the HTTP layer:

    status, body = admin.handle_get(path, query_params, registry, db, selector=selector)
    status, body = admin.handle_post(path, body_dict, registry, db, selector=selector)

All responses are JSON-serialisable dicts or lists.  The caller is
responsible for serialising to JSON and writing the HTTP response.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

from .backends_registry import backend_names as _backend_names
from .constants import VALID_BACKEND_MODES
from .model_tier import model_tier_rank

logger = logging.getLogger(__name__)

VALID_TIERS = ('haiku', 'sonnet', 'opus', 'fable')
TIME_RANGE_MAP = {'1d': '-1 days', '7d': '-7 days', '30d': '-30 days'}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _err(code: int, error_code: str, message: str) -> tuple[int, dict]:
    return code, {'error': error_code, 'message': message}


def _int_param(params: dict, key: str, default: int, max_val: int | None = None) -> int:
    """Parse an integer query param, returning ``default`` on missing/invalid."""
    try:
        v = int(params.get(key, default))
    except (ValueError, TypeError):
        return default
    if max_val is not None and v > max_val:
        return max_val
    return v


def _fetch_usage_caches(registry) -> None:
    """Concurrently refresh usage caches for cached subscription backends.

    Called on the admin request thread before usage_snapshot() to populate
    per-backend _usage_cache.  Rate-limited by SubscriptionBackend.get_usage's
    300s TTL — most polls are cache hits.  Never holds registry/selector locks
    across fetches; only backend-local _usage_lock is held per backend.
    """
    instances = registry.cached_subscription_instances()
    config = registry.config
    if not instances:
        return

    def _fetch_one(backend):
        try:
            backend.get_usage(config)
        except Exception:  # noqa: BLE001 — usage refresh is best-effort
            pass

    targets = [b for b in instances.values() if hasattr(b, 'get_usage')]
    if not targets:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = [executor.submit(_fetch_one, b) for b in targets]
        concurrent.futures.wait(futures, timeout=5.0)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def handle_get(
    path: str, query_params: dict, registry, db, *, selector=None,
) -> tuple[int, dict | list]:
    """Route GET requests.  Returns (status_code, response_body)."""
    parts = [unquote(p) for p in path.split('/') if p]
    try:
        if len(parts) < 2 or parts[0] != 'admin':
            return _err(404, 'NOT_FOUND', f'Unknown path: {path}')
        return _route_admin_get(parts[1:], query_params, registry, db, selector=selector)
    except Exception as exc:  # noqa: BLE001
        logger.exception('Unexpected error in admin handle_get: %s', exc)
        return _err(500, 'INTERNAL_ERROR', str(exc))


def handle_post(
    path: str, body: dict, registry, db, *, selector=None,
) -> tuple[int, dict]:
    """Route POST requests.  Returns (status_code, response_body)."""
    parts = [unquote(p) for p in path.split('/') if p]
    try:
        if len(parts) < 2 or parts[0] != 'admin':
            return _err(404, 'NOT_FOUND', f'Unknown path: {path}')
        return _route_admin_post(parts[1:], body, registry, db, selector=selector)
    except Exception as exc:  # noqa: BLE001
        logger.exception('Unexpected error in admin handle_post: %s', exc)
        return _err(500, 'INTERNAL_ERROR', str(exc))


# ---------------------------------------------------------------------------
# GET routing
# ---------------------------------------------------------------------------

def _route_admin_get(
    parts: list[str],
    query_params: dict,
    registry,
    db,
    *,
    selector=None,
) -> tuple[int, dict | list]:
    if not parts:
        return _err(404, 'NOT_FOUND', 'Unknown admin path')

    segment = parts[0]

    # GET /admin/sessions
    if segment == 'sessions' and len(parts) == 1:
        return _get_sessions(query_params, db)

    # GET /admin/sessions/{session_id}
    # GET /admin/sessions/{session_id}/trace
    # GET /admin/sessions/{session_id}/summary
    if segment == 'sessions' and len(parts) >= 2:
        session_id = parts[1]
        if len(parts) == 2:
            return _get_session(session_id, db)
        if len(parts) == 3 and parts[2] == 'trace':
            return _get_trace(session_id, query_params, db)
        if len(parts) == 3 and parts[2] == 'summary':
            return _get_session_summary(session_id, db)
        return _err(404, 'NOT_FOUND', f'Unknown sessions sub-path: {parts[2]}')

    # GET /admin/cost
    if segment == 'cost':
        return _get_cost(query_params, db)

    # GET /admin/routing
    if segment == 'routing':
        return _get_routing(query_params, db)

    # GET /admin/backends
    if segment == 'backends':
        return _get_backends(registry)

    # GET /admin/config
    if segment == 'config':
        return _get_config(registry)

    # GET /admin/config-changes
    if segment == 'config-changes':
        return _get_config_changes(query_params, db)

    # GET /admin/status  (APXY-API-014)
    if segment == 'status':
        return _get_status(registry, db, selector=selector)

    # GET /admin/stats  (APXY-API-015)
    if segment == 'stats':
        return _get_stats(query_params, db)

    # GET /admin/requests/{id}  (APXY-API-016)
    if segment == 'requests' and len(parts) == 2:
        try:
            request_id = int(parts[1])
            if request_id <= 0:
                return _err(400, 'BAD_REQUEST', 'request_id must be positive')
            return _get_request_detail(request_id, db)
        except ValueError:
            return _err(400, 'BAD_REQUEST', 'request_id must be an integer')

    # GET /admin/prompts/{sha256}  (APXY-API-017)
    if segment == 'prompts' and len(parts) == 2:
        sha256_hex = parts[1]
        if not re.match(r'^[a-f0-9]{64}$', sha256_hex):
            return _err(400, 'INVALID_HASH', 'sha256 must be 64 lowercase hex chars')
        return _get_prompt(sha256_hex, db)

    return _err(404, 'NOT_FOUND', f'Unknown admin path: /{segment}')


# ---------------------------------------------------------------------------
# GET handlers
# ---------------------------------------------------------------------------

def _get_sessions(query_params: dict, db) -> tuple[int, dict]:
    limit = _int_param(query_params, 'limit', 50, max_val=1000)
    offset = _int_param(query_params, 'offset', 0)
    q = query_params.get('q')
    q = q.strip() if isinstance(q, str) and q.strip() else None
    items = db.get_sessions(limit=limit, offset=offset, q=q)
    total = db.get_sessions_count(q=q)
    return 200, {
        'items': items,
        'total': total,
        'limit': limit,
        'offset': offset,
        'q': q,
    }


def _get_session(session_id: str, db) -> tuple[int, dict]:
    session = db.get_session(session_id)
    if session is None:
        return _err(404, 'NOT_FOUND', f'Session not found: {session_id}')
    return 200, session


def _get_trace(session_id: str, query_params: dict, db) -> tuple[int, dict]:
    anchor = query_params.get('anchor')
    limit = _int_param(query_params, 'limit', 100, max_val=1000)
    offset = _int_param(query_params, 'offset', 0)
    q = query_params.get('q')
    q = q.strip() if isinstance(q, str) and q.strip() else None
    items = db.get_trace(session_id, anchor=anchor, limit=limit, offset=offset, q=q)
    total = db.get_trace_count(session_id, anchor=anchor, q=q)
    return 200, {
        'items': items,
        'session_id': session_id,
        'anchor': anchor,
        'total': total,
        'limit': limit,
        'offset': offset,
        'q': q,
    }


def _get_session_summary(session_id: str, db) -> tuple[int, dict]:
    """GET /admin/sessions/{session_id}/summary — LLM-generated session summary row."""
    summary = db.get_session_summary(session_id)
    if summary is None:
        return _err(404, 'NOT_FOUND', f'Session summary not found: {session_id}')
    return 200, summary


def _get_cost(query_params: dict, db) -> tuple[int, dict]:
    group_by = query_params.get('group_by', 'model')
    time_range = query_params.get('time_range', '7d')
    session_id = query_params.get('session_id')
    if time_range not in TIME_RANGE_MAP:
        return _err(
            400, 'INVALID_TIME_RANGE',
            f'time_range must be one of: {", ".join(TIME_RANGE_MAP)}',
        )
    since = TIME_RANGE_MAP[time_range]
    kwargs: dict = {'group_by': group_by, 'since': since}
    if session_id is not None:
        kwargs['session_id'] = session_id
    raw = db.get_cost(**kwargs)
    items = [{'name': r.get('key', ''), **{k: v for k, v in r.items() if k != 'key'}} for r in raw]
    return 200, {'items': items, 'group_by': group_by, 'time_range': time_range}


def _tier_rank(model: str) -> int:
    r = model_tier_rank(model)
    return r if r is not None else -1


def _get_routing(query_params: dict, db) -> tuple[int, dict]:
    time_range = query_params.get('time_range', '7d')
    session_id = query_params.get('session_id')
    if time_range not in TIME_RANGE_MAP:
        return _err(
            400, 'INVALID_TIME_RANGE',
            f'time_range must be one of: {", ".join(TIME_RANGE_MAP)}',
        )
    since = TIME_RANGE_MAP[time_range]
    kwargs: dict = {'since': since}
    if session_id is not None:
        kwargs['session_id'] = session_id
    data = db.get_routing(**kwargs)

    # --- KPIs from reason_code_distribution ---
    _CACHED_TIER_CODES = {
        'session_cached_tier',
        'session_cached_walkback',
        'session_cached_tier_capped',
        'session_cached_walkback_capped',
    }
    size_forced_count = 0
    affirmation_count = 0
    cached_tier_count = 0
    for item in data.get('reason_code_distribution', []):
        rc = item['reason_code']
        cnt = item['cnt']
        if rc == 'size_forced_long_context':
            size_forced_count += cnt
        elif rc in ('affirmation_inherited', 'affirmation_floored_standard',
                    'affirmation_classified', 'affirmation_classifier_failed'):
            affirmation_count += cnt
        elif rc in _CACHED_TIER_CODES:
            cached_tier_count += cnt

    # --- Distributions and upgrade/downgrade/unchanged from tier_transitions ---
    # When applied=0, CLAUDE.md guarantees requested_model==routed_model → unchanged.
    original_model_totals: dict[str, int] = {}
    routed_model_totals: dict[str, int] = {}
    upgrade_count = 0
    downgrade_count = 0
    unchanged_count = 0
    for r in data.get('tier_transitions', []):
        req_model = r['requested_model']
        rout_model = r['routed_model']
        cnt = r['cnt']
        original_model_totals[req_model] = original_model_totals.get(req_model, 0) + cnt
        routed_model_totals[rout_model] = routed_model_totals.get(rout_model, 0) + cnt
        req_rank = _tier_rank(req_model)
        rout_rank = _tier_rank(rout_model)
        if req_rank != -1 and rout_rank != -1:
            if rout_rank > req_rank:
                upgrade_count += cnt
            elif rout_rank < req_rank:
                downgrade_count += cnt
            else:
                unchanged_count += cnt

    original_model_distribution = [
        {'model': m, 'count': c}
        for m, c in sorted(original_model_totals.items(), key=lambda x: -x[1])
    ]
    routed_model_distribution = [
        {'model': m, 'count': c}
        for m, c in sorted(routed_model_totals.items(), key=lambda x: -x[1])
    ]

    # Collapse reason_code_distribution by reason_code, summing cnt across
    # (applied, classification) variants that the DB groups separately.
    reason_code_totals: dict[str, int] = {}
    for r in data.get('reason_code_distribution', []):
        rc = r['reason_code']
        reason_code_totals[rc] = reason_code_totals.get(rc, 0) + r['cnt']

    return 200, {
        'reason_codes': [
            {'reason_code': rc, 'count': cnt}
            for rc, cnt in sorted(reason_code_totals.items(), key=lambda x: -x[1])
        ],
        'tier_transitions': [
            {
                'requested_tier': r['requested_model'],
                'routed_tier': r['routed_model'],
                'count': r['cnt'],
            }
            for r in data.get('tier_transitions', [])
        ],
        'upgrade_count': upgrade_count,
        'downgrade_count': downgrade_count,
        'unchanged_count': unchanged_count,
        'size_forced_count': size_forced_count,
        'affirmation_count': affirmation_count,
        'cached_tier_count': cached_tier_count,
        'original_model_distribution': original_model_distribution,
        'routed_model_distribution': routed_model_distribution,
    }


def _window_start(w: dict, hours: float) -> str | None:
    """Return the ISO timestamp at which the current usage window opened.

    Derived from reset_at - window_hours (fixed window boundary), falling back
    to reset_in_secs when reset_at is absent.  Returns None when neither is
    available.
    """
    reset_at_str = w.get('reset_at')
    if reset_at_str:
        try:
            reset_at = datetime.fromisoformat(reset_at_str.replace('Z', '+00:00'))
            start = reset_at - timedelta(hours=hours)
            return start.strftime('%Y-%m-%dT%H:%M:%S.') + f'{start.microsecond // 1000:03d}Z'
        except ValueError:
            pass
    reset_in_secs = w.get('reset_in_secs')
    if reset_in_secs is not None:
        start = datetime.now(timezone.utc) + timedelta(seconds=reset_in_secs) - timedelta(hours=hours)
        return start.strftime('%Y-%m-%dT%H:%M:%S.') + f'{start.microsecond // 1000:03d}Z'
    return None


def _build_backends_list(status: dict) -> list:
    """Convert registry backend_status() dict to a JSON-serialisable list.

    Patches ``available`` to ``True`` for the active backend when no health
    observation has been recorded yet (avoids surfacing ``null`` for a backend
    that is plainly working because it is serving requests).
    """
    out = []
    for b in status.values():
        entry = dict(b)
        if entry.get('active') and entry.get('available') is None:
            entry['available'] = True
        out.append(entry)
    return out


def _get_backends(registry) -> tuple[int, dict]:
    status = registry.backend_status()
    active = next((name for name, b in status.items() if b.get('active')), '')
    return 200, {
        'backends': _build_backends_list(status),
        'active': active,
        'known': list(registry.list_backends()),
        'modes': list(VALID_BACKEND_MODES),
    }


def _get_config(registry) -> tuple[int, dict]:
    cfg = registry.config
    return 200, {
        'routing_enabled': cfg.auto_model_routing,
        'auto_backend_mode': cfg.auto_backend_mode,
        'auto_backend': cfg.auto_backend,
        'active_backend': registry.active_name(),
        'auto_model_routing_classifier_model': cfg.auto_model_routing_classifier_model,
        'auto_model_routing_long_context_threshold': cfg.auto_model_routing_long_context_threshold,
        'auto_model_routing_affirmation_inherit': cfg.auto_model_routing_affirmation_inherit,
        'auto_model_routing_mode': cfg.auto_model_routing_mode,
    }


def _get_config_changes(query_params: dict, db) -> tuple[int, dict]:
    limit = _int_param(query_params, 'limit', 100, max_val=1000)
    items = db.get_config_changes(limit=limit)
    return 200, {'items': items}


_VALID_STATS_PERIODS = ('day', 'week', 'month', 'quarter')


def _get_status(registry, db, *, selector=None) -> tuple[int, dict]:
    """GET /admin/status — current proxy state snapshot (APXY-API-014)."""
    active_backend = registry.active_name()
    status = registry.backend_status()
    cfg = registry.config
    _fetch_usage_caches(registry)
    subscription_usage = registry.usage_snapshot()
    for backend_name, backend_usage in subscription_usage.items():
        for window_key in ('five_hour', 'weekly'):
            w = backend_usage.get(window_key)
            if w is None:
                continue
            hours = w.get('window_hours') or (5 if window_key == 'five_hour' else 168)
            window_start = _window_start(w, hours)
            if window_start is None:
                continue
            busy = db.busy_secs_window(backend_name, window_start)
            if busy is not None:
                w['active_secs'] = busy
    session_overrides = db.get_session_overrides()

    return 200, {
        'active_backend': active_backend,
        'routing_enabled': cfg.auto_model_routing,
        'routing_mode': cfg.auto_model_routing_mode,
        'classifier_model': cfg.auto_model_routing_classifier_model,
        'long_context_threshold': cfg.auto_model_routing_long_context_threshold,
        'affirmation_inherit': cfg.auto_model_routing_affirmation_inherit,
        'backends': _build_backends_list(status),
        'session_overrides': session_overrides,
        'subscription_usage': subscription_usage,
        'auto_selection': selector.status_line() if selector is not None else None,
    }


def _get_stats(query_params: dict, db) -> tuple[int, dict]:
    """GET /admin/stats — time-bucketed aggregates (APXY-API-015)."""
    period = query_params.get('period', 'week')
    if period not in _VALID_STATS_PERIODS:
        return _err(
            400, 'INVALID_PERIOD',
            f'period must be one of: {", ".join(_VALID_STATS_PERIODS)}',
        )
    backend = query_params.get('backend') or None
    data = db.get_stats(period, backend)
    return 200, {
        'period': period,
        'backend_filter': backend,
        'buckets': data.get('buckets', []),
        'total': data.get('total', {}),
    }


def _get_request_detail(request_id: int, db) -> tuple[int, dict]:
    """GET /admin/requests/{id} — full request detail with prompt joins (APXY-API-016)."""
    row = db.get_request(request_id)
    if row is None:
        return _err(404, 'NOT_FOUND', f'Request not found: {request_id}')
    return 200, row


def _get_prompt(sha256_hex: str, db) -> tuple[int, dict]:
    """GET /admin/prompts/{sha256} — prompt store row (APXY-API-017)."""
    row = db.get_prompt(sha256_hex)
    if row is None:
        return _err(404, 'NOT_FOUND', f'Prompt not found: {sha256_hex}')
    return 200, row


# ---------------------------------------------------------------------------
# POST routing
# ---------------------------------------------------------------------------

def _route_admin_post(
    parts: list[str],
    body: dict,
    registry,
    db,
    *,
    selector=None,
) -> tuple[int, dict]:
    if not parts:
        return _err(404, 'NOT_FOUND', 'Unknown admin path')

    segment = parts[0]

    # POST /admin/sessions/{session_id}/set-backend
    # POST /admin/sessions/{session_id}/set-global-tier
    if segment == 'sessions' and len(parts) == 3:
        session_id = parts[1]
        action = parts[2]
        if action == 'set-backend':
            return _post_set_session_backend(session_id, body, registry, db)
        if action == 'set-global-tier':
            return _post_set_global_tier(session_id, body, registry, db)
        return _err(404, 'NOT_FOUND', f'Unknown sessions action: {action}')

    # POST /admin/global/routing
    # POST /admin/global/backend
    if segment == 'global' and len(parts) == 2:
        action = parts[1]
        if action == 'routing':
            return _post_global_routing(body, registry, db)
        if action == 'backend':
            return _post_global_backend(body, registry, db, selector=selector)
        return _err(404, 'NOT_FOUND', f'Unknown global action: {action}')

    # POST /admin/export
    if segment == 'export':
        return _post_export(body, db)

    return _err(404, 'NOT_FOUND', f'Unknown admin path: /{segment}')


# ---------------------------------------------------------------------------
# POST handlers
# ---------------------------------------------------------------------------

def _post_set_session_backend(
    session_id: str, body: dict, registry, db,
) -> tuple[int, dict]:
    backend = body.get('backend')  # str or None (None = clear pin)
    if backend is not None:
        known = registry.list_backends()
        if backend not in known:
            return _err(
                400, 'INVALID_BACKEND',
                f'Unknown backend: {backend!r}. Known: {list(known)}',
            )
        registry.set_session_backend(session_id, backend)
    else:
        registry.clear_session_backend(session_id)
    db.set_session_backend(session_id, backend)
    db.record_config_change(
        'set_session_backend', 'admin_api', session_id,
        prev_value=None, new_value=backend,
    )
    return 200, {'status': 'ok', 'session_id': session_id, 'backend': backend}


def _post_set_global_tier(
    session_id: str, body: dict, registry, db,
) -> tuple[int, dict]:
    tier = body.get('tier')  # str or None (None = clear override)
    if tier is not None and tier not in VALID_TIERS:
        return _err(
            400, 'INVALID_TIER',
            f'tier must be one of: {", ".join(VALID_TIERS)} or null',
        )
    registry.set_session_tier_global(session_id, tier)
    db.set_session_tier(session_id, tier)
    db.record_config_change(
        'set_session_tier', 'admin_api', session_id,
        prev_value=None, new_value=tier,
    )
    return 200, {'status': 'ok', 'session_id': session_id, 'tier': tier}


def _post_global_routing(body: dict, registry, db) -> tuple[int, dict]:
    enabled = body.get('enabled')
    if not isinstance(enabled, bool):
        return _err(400, 'BAD_REQUEST', '"enabled" must be a boolean')
    registry.set_model_routing(enabled)
    db.record_config_change(
        'set_global_routing', 'admin_api', 'global',
        prev_value=None, new_value=enabled,
    )
    return 200, {'status': 'ok', 'routing_enabled': enabled}


def _post_global_backend(body: dict, registry, db, *, selector=None) -> tuple[int, dict]:
    prefer = body.get('prefer')
    if prefer is None:
        return _err(400, 'BAD_REQUEST', '"prefer" is required')
    valid = tuple(VALID_BACKEND_MODES) + tuple(_backend_names())
    if prefer not in valid:
        return _err(
            400, 'INVALID_BACKEND_MODE',
            f'prefer must be one of: {", ".join(valid)}',
        )
    if prefer == 'auto':
        registry.set_auto_backend_mode('auto')
        if selector is not None:
            selector.resume()
    elif prefer == 'subscription':
        registry.set_auto_backend_mode('subscription')
        if selector is not None:
            selector.restrict_subscription()
    else:
        result = registry.switch(prefer, reason='admin_api')
        # Pause auto-selection when an operator manually pins a backend, so
        # the selector's next tick does not immediately switch it back.
        if selector is not None and result.kind in ('changed', 'unchanged'):
            selector.pin(prefer)
        if result.kind == 'failed':
            return _err(500, 'SWITCH_FAILED', f'Could not switch to {prefer!r}: {result.error}')
        if result.kind == 'invalid':
            return _err(400, 'INVALID_BACKEND', f'Unknown backend: {prefer!r}')
    db.record_config_change(
        'set_global_backend', 'admin_api', 'global',
        prev_value=None, new_value=prefer,
    )
    return 200, {
        'status': 'ok',
        'prefer': prefer,
        'active_backend': registry.active_name(),
        'auto_selection': selector.status_line() if selector is not None else None,
    }


def _post_export(body: dict, db) -> tuple[int, dict]:
    session_id = body.get('session_id')
    if not session_id:
        return _err(400, 'BAD_REQUEST', '"session_id" is required')
    items = db.get_trace(session_id)
    return 200, {'export': items, '_filename': f'trace_{session_id}.json'}
