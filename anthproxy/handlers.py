import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler

from .backends_registry import backend_names as _backend_names
from .constants import (
    HAPPY_BIRTHDAY_REPLY,
    HAPPY_NEW_YEAR_PREFIX,
    SESSION_SUBSCRIPTION_SENTINEL,
    SUBSCRIPTION_BACKENDS,
    TOOLS_TO_REMOVE,
)
from .mapper import AnthropicRequestError, anthropic_error_payload, estimate_input_tokens, sse_event
from .model_router import ModelRoutingDecision as _ModelRoutingDecision
from .model_router import _cap_cached_tier as _cap_cached_tier
from .model_router import _extract_user_text as _extract_user_text
from .model_router import build_routing_summary as _build_routing_summary
from .model_router import calibrated_ratio as _calibrated_ratio
from .model_router import route_model as _route_model
from .request_text import _WRAPPER_TAGS, last_transcript_user_turn as _last_transcript_user_turn
from .request_text import strip_reminders as _strip_reminders_shared

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .oauth_registry import OAuthRequestCredentials

logger = logging.getLogger(__name__)

# Client-side socket teardown — the client (e.g. Claude Code) cancelled an
# in-flight request and closed the connection.  Benign: handled quietly, never
# logged as a "Proxy failure" traceback.
_CLIENT_DISCONNECT = (BrokenPipeError, ConnectionResetError)

_SET_BACKEND_PREFIX = 'proxy-set-backend:'
_SET_MODEL_ROUTING_PREFIX = 'proxy-set-model-routing:'
_SESSION_SUFFIX = ':session'
_STATS_PREFIX = 'proxy-stats'
# Unlike _WRAPPER_TAGS (whose blocks are removed), <session>…</session> KEEPS
# its inner content: it is a recognized way to wrap a proxy-* command. Anchored
# at the start (leading whitespace tolerated) so prose *before* the tag prevents
# unwrapping, but trailing text after </session> is ignored — openclaw appends
# title-generation boilerplate (e.g. "Write the title …") after the wrapped
# command, and that suffix must not defeat command recognition. The inner group
# is non-greedy so it stops at the first </session>.
_SESSION_WRAP_RE = re.compile(r'\s*<session>(.*?)</session>', re.DOTALL)
def _stats_backend_filters() -> frozenset:
    """Return valid backend-filter tokens for proxy-stats: all registered names + sentinel."""
    return frozenset(_backend_names()) | {SESSION_SUBSCRIPTION_SENTINEL}
_BACKEND_HEADER = '## anthproxy backend\n\n'

# SSE data line pattern for stats extraction
_SSE_DATA_RE = re.compile(r'^data: (.+)$', re.MULTILINE)
# Cap on stored response_text to prevent unbounded disk growth
_RESPONSE_TEXT_LIMIT = 1024 * 1024  # 1 MB

def _qualify_first_heading(markdown: str, qualifier: str) -> str:
    """Append `` · qualifier`` to the first Markdown heading, if any."""
    lines = markdown.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.rstrip('\r\n')
        if stripped.startswith('#'):
            newline = line[len(stripped):]
            lines[i] = f'{stripped} · {qualifier}{newline}'
            return ''.join(lines)
    return markdown


def _strip_reminders(text: str) -> str:
    """Remove all <system-reminder> blocks and trim outer whitespace."""
    return _strip_reminders_shared(text)


def _unwrap_session(text: str) -> str:
    m = _SESSION_WRAP_RE.match(text)
    return m.group(1).strip() if m else text


def _final_user_text(payload: dict) -> str | None:
    """Return the sole final-user-message text, or None if not a clean command.

    Only the last message is inspected; it must be a user message whose content
    is a string or a list of text blocks.  Non-text blocks (tool_result, image,
    etc.) in a list cause the message to be rejected so ordinary conversation is
    never intercepted.

    <system-reminder> blocks injected by the Claude Code CLI are stripped before
    matching so proxy-* commands work on the first turn of a fresh session.
    After stripping, exactly one non-empty text segment must remain; two or more
    real text segments indicate prose rather than a command.
    """
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        return None
    message = messages[-1]
    if not isinstance(message, dict) or message.get('role') != 'user':
        return None

    content = message.get('content')
    if isinstance(content, str):
        return _strip_reminders(content) or None

    if isinstance(content, list):
        segments = []
        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'text':
                # Any non-text block (tool_result, image, …) disqualifies
                return None
            text = block.get('text')
            if not isinstance(text, str):
                return None
            stripped = _strip_reminders(text)
            if stripped:
                segments.append(stripped)
        # Exactly one real segment must remain; 0 or 2+ → not a command
        return segments[0] if len(segments) == 1 else None

    return None


def _parse_stats_selector(raw: str) -> tuple[str, str | None]:
    """Split the suffix after 'proxy-stats:' into (period_token, backend_or_None).

    The backend token (a registered backend name or the subscription sentinel) may appear in any
    position among the colon-separated parts.  The first matching part becomes
    the backend filter; the remaining parts form the period token.  If no part
    matches, backend is None and the whole raw string is the period token.

    Examples:
      '-1d:bedrock' -> ('-1d', 'bedrock')
      'bedrock:-1d' -> ('-1d', 'bedrock')
      'bedrock'     -> ('',   'bedrock')
      '-1d'         -> ('-1d', None)
      ''            -> ('',    None)
    """
    backend: str | None = None
    leftover: list[str] = []
    filters = _stats_backend_filters()
    for part in (raw.split(':') if raw else []):
        if backend is None and part in filters:
            backend = part
        else:
            leftover.append(part)
    # Single leftover part → period token; 0 or 2+ → pass them joined so that
    # resolve_stats_period fails gracefully and falls back to today.
    period_token = leftover[0] if len(leftover) == 1 else ':'.join(leftover)
    return period_token, backend


def _parse_local_command(payload: dict) -> tuple[str, object | None] | None:
    """Parse an exact local proxy command from the final user message.

    Returns ``(name, arg)`` where name is one of ``help``, ``status``,
    ``get-usage``, ``get-backend``, ``set-backend`` (arg = requested backend or
    None if malformed), or ``stats`` (arg = resolved stats period), or None
    when the message is not a local command.
    """
    text = _final_user_text(payload)
    if text is None:
        return None
    text = _unwrap_session(text)
    if '\n' in text:
        before, last = text.rsplit('\n', 1)
        # Reject if the preceding text contains an unclosed wrapper-tag opener —
        # same as treating it as embedded prose rather than stripped context.
        cmd = text if any(f'<{t}>' in before for t in _WRAPPER_TAGS) else last
    else:
        cmd = text
    if cmd == 'proxy-help':
        return ('help', None)
    if cmd == 'proxy-status':
        return ('status', None)
    if cmd == 'proxy-get-usage':
        return ('get-usage', None)
    if cmd == 'proxy-get-backend':
        return ('get-backend', None)
    if cmd.startswith(_SET_BACKEND_PREFIX):
        rest = cmd[len(_SET_BACKEND_PREFIX):]
        if rest.endswith(_SESSION_SUFFIX):
            name = 'session-set-backend'
            target = rest[:-len(_SESSION_SUFFIX)]
        else:
            name = 'set-backend'
            target = rest
        if target == 'auto':
            arg = 'auto'
        elif target == 'subscription':
            arg = 'subscription'
        elif target in _backend_names():
            arg = target
        else:
            arg = None
        return (name, arg)
    if cmd.startswith(_SET_MODEL_ROUTING_PREFIX):
        rest = cmd[len(_SET_MODEL_ROUTING_PREFIX):]
        if rest.endswith(_SESSION_SUFFIX):
            name = 'session-set-model-routing'
            value = rest[:-len(_SESSION_SUFFIX)]
        else:
            name = 'set-model-routing'
            value = rest
        if value == 'on':
            arg = True
        elif value == 'off':
            arg = False
        elif value == 'auto' and name == 'session-set-model-routing':
            arg = None  # clear session override → follow global
        else:
            arg = 'invalid'
        return (name, arg)
    if cmd == _STATS_PREFIX or cmd.startswith(_STATS_PREFIX + ':'):
        from .stats import resolve_stats_period

        raw = cmd[len(_STATS_PREFIX) + 1:] if cmd.startswith(_STATS_PREFIX + ':') else ''
        period_token, backend = _parse_stats_selector(raw)
        period = resolve_stats_period(period_token)
        if backend is not None:
            period = replace(period, backend=backend)
        return ('stats', period)
    return None


def _has_happy_new_year_system_prompt(payload: dict) -> bool:
    system = payload.get('system')
    if isinstance(system, str):
        return system.startswith(HAPPY_NEW_YEAR_PREFIX)
    if not isinstance(system, list):
        return False
    for item in system:
        if not isinstance(item, dict):
            continue
        text = item.get('text')
        if isinstance(text, str) and text.startswith(HAPPY_NEW_YEAR_PREFIX):
            return True
    return False


_OVERRIDE_HEADER_MAX = 2048


_VALID_OVERRIDE_MODES = frozenset({'classifier', 'rules', 'tag'})


def _parse_override_header(raw: str | None) -> dict:
    """Parse ``X-Anthproxy-Override`` into a directives dict.

    Returns a dict with zero or more of:
      - ``no_classifier``: bool (default False)
      - ``prefer_backend``: str | None (default None)
      - ``override_mode``: str | None — one of ``'classifier'``, ``'rules'``, ``'tag'``
      - ``task_tag``: str | None — task name for ``route:tag`` mode

    Grammar: ``<directive> [; <directive>]*`` where directive is one of:
    ``no-classifier``, ``prefer:<backend>``, ``route:<mode>``, ``task:<name>``.

    Unknown directives and malformed values are silently ignored (fail-open).
    Multiple ``prefer:`` or ``route:`` directives: last one wins.  Backend
    names not in the registry are logged and ignored.  Unknown route
    modes are silently ignored.  Task names preserve original case.
    """
    if not raw or not isinstance(raw, str):
        return {}
    if len(raw) > _OVERRIDE_HEADER_MAX:
        logger.warning(
            'X-Anthproxy-Override header too long (%d chars), ignoring',
            len(raw),
        )
        return {}
    result: dict = {}
    for part in raw.split(';'):
        orig = part.strip()
        directive = orig.lower()
        if not directive:
            continue
        if directive == 'no-classifier':
            result['no_classifier'] = True
        elif directive.startswith('prefer:'):
            name = directive[len('prefer:'):].strip()
            if name in _backend_names():
                result['prefer_backend'] = name
            else:
                logger.warning(
                    'X-Anthproxy-Override: unknown backend %r in prefer: -- ignored',
                    name,
                )
        elif directive.startswith('route:'):
            mode = directive[len('route:'):].strip()
            if mode in _VALID_OVERRIDE_MODES:
                result['override_mode'] = mode
            # else: silently ignore unknown modes
        elif directive.startswith('task:'):
            # Preserve original case for the task name
            task_name = orig[len('task:'):].strip()
            if task_name:
                result['task_tag'] = task_name
        # else: silently ignore unknown directives
    return result


def _session_key(payload: dict) -> str | None:
    """Return the session key from ``payload['metadata']['user_id']``, or None.

    The Claude Code CLI populates ``metadata.user_id`` with a per-session
    identifier on every request; it is the only per-request value that reliably
    distinguishes one session from another.  Returns None when the field is
    absent, blank, or not a string.
    """
    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get('user_id')
    if not isinstance(user_id, str) or not user_id:
        return None
    return user_id


def _economics_kwargs(econ) -> dict:
    """Return ``record_request`` kwargs for the persisted routing-economics columns.

    Yields values only when economics were computed with available pricing; an
    absent or pricing-unavailable ``econ`` produces an empty dict so the columns
    default to NULL.  ``record_request`` itself gates on ``applied``.
    """
    if econ is None or not getattr(econ, 'pricing_available', False):
        return {}
    return {
        'net_savings_usd': econ.net_savings_usd,
        'classifier_overhead_usd': econ.classifier_overhead_usd,
    }


_UUID_RE = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.IGNORECASE,
)


def _session_short_id(sess_key: str) -> str:
    """Return an 8-char display token from a session key.

    ``metadata.user_id`` may be a plain string/UUID, or a JSON blob (Claude
    Code sends ``{"device_id":..., "session_id":..., "account_uuid":...}``).
    Strategy:
    1. If JSON, prefer ``session_id`` field — return its first 8 UUID chars.
    2. Scan raw string for a UUID and return its first 8 chars.
    3. Fall back to the first 8 hex chars of sha256(sess_key).
    """
    if sess_key and sess_key[0] == '{':
        try:
            data = json.loads(sess_key)
            if isinstance(data, dict):
                sid = data.get('session_id')
                if isinstance(sid, str) and sid:
                    m = _UUID_RE.search(sid)
                    return m.group()[:8] if m else sid[:8]
        except (json.JSONDecodeError, TypeError):
            pass

    m = _UUID_RE.search(sess_key)
    if m:
        return m.group()[:8]

    return hashlib.sha256(sess_key.encode()).hexdigest()[:8]


def _system_fingerprint(payload: dict) -> tuple[int, int, str, str]:
    """Return ``(block_count, total_chars, sha8, head)`` for the system prompt.

    The system prompt is the most reliable *in-band* signal that separates a
    Claude Code sub-agent (Task tool) from its parent loop.  Claude Code reuses
    the **same** ``metadata.user_id`` (device_id/account_uuid/session_id) for a
    Task sub-agent and its parent, so ``session_id`` alone cannot tell them
    apart; but each agent type carries a distinct system prompt, so its hash
    differs.  Logging this lets the operator see which size-forced requests
    actually belong to a sub-agent rather than the top-level session.
    """
    system = payload.get('system')
    parts: list[str] = []
    block_count = 0
    if isinstance(system, str):
        if system:
            parts.append(system)
            block_count = 1
    elif isinstance(system, list):
        for blk in system:
            if isinstance(blk, dict):
                text = blk.get('text')
                if isinstance(text, str):
                    parts.append(text)
                    block_count += 1
            elif isinstance(blk, str):
                parts.append(blk)
                block_count += 1
    text = '\n'.join(parts)
    total = len(text)
    digest = hashlib.sha256(text.encode()).hexdigest()[:8] if text else '--------'
    head = text[:60].replace('\n', ' ') if text else ''
    return block_count, total, digest, head


_ANCHOR_LIMIT = 4_000  # mirror model_router._TEXT_LIMIT; cap before hashing


def _conversation_anchor(payload: dict) -> str:
    """Hash the conversation's **first user message** as a per-conversation anchor.

    Returns the first 8 hex chars of sha256 over the (stripped, head-capped) text
    of the earliest ``role == 'user'`` message, or the ``--------`` sentinel when no
    usable text exists.  This is the in-band signal that isolates a Claude Code Task
    sub-agent from its parent loop: both reuse the same ``metadata.user_id``, but
    each agent's initiating user turn is distinct and **stable across that agent's
    own turns** (Claude Code resends the full history, so ``messages[0]`` is present
    on every continuation turn).  Text extraction reuses ``_extract_user_text`` (same
    str/list handling, text-block-only privacy contract, and ``strip_reminders`` as
    the router), with the transcript fallback for a transcript-only first turn.
    """
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        return '--------'
    for msg in messages:
        if not isinstance(msg, dict) or msg.get('role') != 'user':
            continue
        extracted = _extract_user_text(msg.get('content'))
        if extracted is None:
            continue
        stripped, raw, _non_text, _has_images = extracted
        text = stripped or _last_transcript_user_turn(raw)
        if text:
            return hashlib.sha256(text[:_ANCHOR_LIMIT].encode()).hexdigest()[:8]
        break
    return '--------'


def _context_key(sess_key: str | None, payload: dict) -> str | None:
    """Derive the per-conversation routing key from session key + first-user-message hash.

    Claude Code reuses the **same** ``metadata.user_id`` for a Task sub-agent and
    the parent loop that spawned it, so keying the tier cache / size-context floor
    on the session alone makes a sub-agent **inherit and clobber** the parent's
    state.  The original discriminator was the system-prompt hash, but production
    traces show it is both *unstable within* a conversation (a turn-1 request and
    its ``tool_result``-only continuation turns carry different system text) and
    *colliding across* conversations (every agent's continuation turns converged on
    one shared hash, so distinct sub-agents shared a slot and clobbered each other).
    The first user message is the stable, distinct anchor instead: it is identical
    across a conversation's own turns and unique per Task launch / parent prompt.
    The ``\\x00`` separator cannot collide with a hex hash.

    ``/compact`` rewrites history and changes ``messages[0]`` → the key resets to a
    fresh tier/floor slot, which is desirable (a compacted context is materially
    different and warrants a fresh measurement).

    Returns ``None`` when there is no session key (floor disabled for the request).
    """
    if not sess_key:
        return None
    return f'{sess_key}\x00{_conversation_anchor(payload)}'


def _local_message(markdown: str, model: str) -> dict:
    return {
        'id': f'msg_{uuid.uuid4().hex}',
        'type': 'message',
        'role': 'assistant',
        'content': [{'type': 'text', 'text': markdown}],
        'model': model or 'proxy',
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 0, 'output_tokens': 0},
    }


def _local_message_sse(markdown: str, model: str):
    message_id = f'msg_{uuid.uuid4().hex}'
    model = model or 'proxy'
    events = [
        sse_event('message_start', {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': model,
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
            },
        }),
        sse_event('content_block_start', {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''},
        }),
        sse_event('content_block_delta', {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'text_delta', 'text': markdown},
        }),
        sse_event('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        sse_event('message_delta', {
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
            'usage': {'output_tokens': 0},
        }),
        sse_event('message_stop', {'type': 'message_stop'}),
    ]
    return iter(events)


# Wire usage-key → internal stats-key mapping for _extract_sse_stats.
_USAGE_FIELD_MAP = (
    ('input_tokens', 'input_tokens'),
    ('output_tokens', 'output_tokens'),
    ('cache_creation_input_tokens', 'cache_creation_tokens'),
    ('cache_read_input_tokens', 'cache_read_tokens'),
)


def _extract_sse_stats(chunk: str, stats: dict) -> None:
    """Parse Anthropic SSE events in ``chunk`` and track cumulative token counts.

    Every Anthropic usage count is **cumulative**, and the usage object on the
    final ``message_delta`` re-states the full snapshot — for server-tool turns
    it even re-echoes ``input_tokens``/``cache_*`` with a *larger* total than
    ``message_start`` (the server-side loop kept consuming context).  So each
    field is tracked as a running **max**, never summed: summing message_start's
    cache counts with message_delta's re-echo double-counts the cached prefix.
    That bug measured a ~110K-token (post-/compact) context as ~218K and forced
    ``opus[1m]`` on every continuation turn.  ``max`` is also correct for Codex,
    whose mapper reports real usage only on the final ``message_delta`` (a tiny
    ``message_start`` is superseded), so no special-casing per backend is needed.
    """
    for m in _SSE_DATA_RE.finditer(chunk):
        try:
            event = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        etype = event.get('type', '')
        if etype == 'message_start':
            usage = event.get('message', {}).get('usage', {}) or {}
        elif etype == 'message_delta':
            usage = event.get('usage', {}) or {}
        else:
            continue
        for wire_key, stat_key in _USAGE_FIELD_MAP:
            val = usage.get(wire_key)
            if val is None:
                continue
            stats[stat_key] = max(stats[stat_key], int(val or 0))


def _extract_sse_text(chunk: str, text_parts: list) -> None:
    """Accumulate assistant text from content_block_delta SSE events.

    Appends text delta values to *text_parts* in place, stopping when the
    total length would exceed _RESPONSE_TEXT_LIMIT.  Only processes
    ``content_block_delta`` events with ``delta.type == 'text_delta'``.
    Skips thinking, tool-use, and other block types.  Never raises.
    Uses the existing module-level ``_SSE_DATA_RE`` regex.
    """
    current_len = sum(len(p) for p in text_parts)
    if current_len >= _RESPONSE_TEXT_LIMIT:
        return
    for m in _SSE_DATA_RE.finditer(chunk):
        try:
            event = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if event.get('type') != 'content_block_delta':
            continue
        delta = event.get('delta') or {}
        if delta.get('type') == 'text_delta':
            t = delta.get('text', '')
            if isinstance(t, str) and t:
                current_len += len(t)
                if current_len > _RESPONSE_TEXT_LIMIT:
                    text_parts.append(t[: max(0, _RESPONSE_TEXT_LIMIT - (current_len - len(t)))])
                    return
                text_parts.append(t)


def _extract_response_text(result: dict) -> str | None:
    """Pull concatenated text blocks from a non-streaming Messages response dict.

    Stops accumulating when the limit is reached to prevent unbounded storage.
    """
    parts: list[str] = []
    current_len = 0
    for block in (result.get('content') or []):
        if isinstance(block, dict) and block.get('type') == 'text':
            t = block.get('text', '')
            if isinstance(t, str) and t:
                remaining = _RESPONSE_TEXT_LIMIT - current_len
                if remaining <= 0:
                    break
                if len(t) > remaining:
                    parts.append(t[:remaining])
                    break
                parts.append(t)
                current_len += len(t)
    return ''.join(parts) or None


def _rewrite_message_start_model(chunk: str, requested_model: str) -> str:
    """Rewrite ``message.model`` in any ``message_start`` event inside ``chunk``.

    Model-tier routing rewrites the outbound ``payload['model']`` to the routed
    tier, so the upstream (or mapper-synthesized) ``message_start`` carries the
    *routed* model.  This restores the client's originally-requested model in the
    response only, leaving the model that actually served the request unchanged.

    Re-serializes only the events it actually rewrites; chunks without a
    ``message_start`` (and malformed ``data:`` lines) pass through byte-for-byte,
    so the common per-event SSE framing is preserved.  Returns the chunk
    unchanged when no rewrite applies.
    """
    if 'message_start' not in chunk:
        return chunk

    changed = False

    def _sub(m: 're.Match') -> str:
        nonlocal changed
        raw = m.group(1)
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return m.group(0)
        message = event.get('message')
        if event.get('type') != 'message_start' or not isinstance(message, dict):
            return m.group(0)
        if message.get('model') == requested_model:
            return m.group(0)
        message['model'] = requested_model
        changed = True
        return 'data: ' + json.dumps(event)

    result = _SSE_DATA_RE.sub(_sub, chunk)
    return result if changed else chunk


@dataclass
class PreparedRequest:
    """Result of routing preparation before dispatch."""
    payload: dict
    snapshot: object
    credentials: dict
    routing: _ModelRoutingDecision
    ctx_key: str | None
    prompt_capture: dict


def _oauth_credential(handler) -> 'OAuthRequestCredentials | None':
    authorization = handler.headers.get('Authorization', '')
    scheme, separator, token = authorization.partition(' ')
    if separator and scheme.lower() == 'bearer' and token.strip():
        return handler.registry.observe_oauth_credential(token.strip())
    return None


def _snapshot_credentials(handler, snapshot) -> dict:
    """Credentials for a dispatch attempt against ``snapshot``.

    The oauth backend carries its credential on the snapshot itself; every
    other backend parses it from the request headers.
    """
    if snapshot.name == 'oauth':
        return {'oauth': snapshot.credentials}
    return snapshot.backend.parse_credentials(handler.headers.get('x-api-key', ''))


def _derive_and_record_routing(handler, payload, snapshot) -> None:
    """Derive the routing decision and record it on the handler.

    Used by both initial dispatch and retry to avoid duplication. Sets
    ``self._routing``, ``self._ctx_key``, and ``self._prompt_capture``.
    """
    credentials = _snapshot_credentials(handler, snapshot)
    sess_key = _session_key(payload) if isinstance(payload, dict) else None
    routing, ctx_key = derive_routing(
        handler, payload, sess_key, snapshot, credentials,
    )
    handler._routing = routing
    handler._ctx_key = ctx_key
    handler._prompt_capture = handler._extract_prompt_capture(payload, routing=routing)


def derive_routing(handler, payload, sess_key, snapshot, credentials):
    """Derive the model-routing decision for one dispatch attempt.

    Called once per attempt against the snapshot that will actually serve the
    request, so a retry crossing the provider/peer boundary re-derives rather
    than inheriting the previous attempt's decision.  Returns
    ``(routing, ctx_key)``; ``ctx_key`` is the key the post-response
    session-context recording writes under, or ``None`` when it must not fire.
    """
    _blks, _chars, _sys_hash, _head = _system_fingerprint(payload)

    # ADR-0023: model-tier routing authority rests with the innermost hop.
    # When this attempt resolves to the peer backend the peer classifies the
    # same text against its own configuration, so classifying here would bill
    # a second classifier call, risk a conflicting decision, and hand the peer
    # a model no client requested.  Suppression is total — no classifier and
    # no long-context floor, because a floor is a judgement about what the
    # serving backend can hold and this hop is not serving the request.
    peer_bound = snapshot.name == 'peer'

    # Per-conversation routing key: (session_id, first-user-message hash).
    # NOT the bare session — Claude Code shares metadata.user_id between a
    # Task sub-agent and its parent loop, so a session-only key lets a
    # sub-agent inherit and clobber both the parent's tier cache (mis-routing
    # its haiku continuations to the parent's cached opus) and its
    # size-context floor.  The first user message is distinct per agent and
    # stable across that agent's turns, so it isolates them (see
    # _context_key).  Computed whenever routing is on; the floor additionally
    # requires a threshold (below).
    routing_on = (
        sess_key is not None and snapshot.config.auto_model_routing
    )
    ctx_key = _context_key(sess_key, payload) if routing_on else None

    # Classify and potentially rewrite payload['model'] before dispatch.
    # route_model() mutates payload in place and is fail-closed; errors
    # keep the original requested model. Never called for local commands.
    # Pass cached session tier so walk-back-only continuations can reuse
    # the tier established by a turn with real intent — keyed per
    # conversation so a sub-agent never reuses the parent's tier.
    # The peer path reads no routing state either: the short-affirmation
    # lookup and the floor read both live inside route_model, which is not
    # called below, but skipping them here explicitly keeps a later hoist out
    # of route_model from quietly reintroducing them on the peer path.
    cached_tier = (
        handler.registry.session_routed_tier(ctx_key)
        if ctx_key and not peer_bound else None
    )
    # Session-context floor: track measured tokens only when routing is on
    # AND a floor threshold is set. _ctx_key gates the post-response
    # recording below; (floor, ratio) feed the floor decision now.
    floor_active = (
        routing_on
        and not peer_bound
        and snapshot.config.auto_model_routing_long_context_threshold > 0
    )
    ctx_key_result = ctx_key if floor_active else None
    session_floor, session_ratio = (
        handler.registry.session_context(ctx_key) if floor_active else (0, 1.0)
    )
    if peer_bound:
        # A pass-through decision, not None: _routing_fields() and the
        # response emitter consume it unconditionally.  The reason code is
        # distinct from 'disabled' (routing was off) and from a classifier
        # run that happened to choose the same model, so an operator
        # debugging a chain can tell the three apart.
        ctx_key_result = None
        requested = str(payload.get('model', ''))
        routing = _ModelRoutingDecision(
            requested_model=requested,
            routed_model=requested,
            classification=None,
            applied=False,
            reason_code='peer_hop_suppressed',
            estimated_input_tokens=0,
        )
        logger.info(
            '%s Model routing suppressed: dispatch target is the peer backend, '
            'which classifies for itself; transmitting requested=%s verbatim '
            '(no classifier, no context floor, no session state written)',
            handler._log_tag(), requested,
        )
    elif getattr(handler, '_no_classifier', False):
        # Bypass ALL routing: classifier call, size floor, session tier
        # cache, affirmation branch.  The client gets exactly the model
        # it asked for.  _ctx_key must be cleared so the post-response
        # _record_session_context is a no-op for this request (per spec:
        # session-context recording does not fire for bypassed requests).
        ctx_key_result = None
        est_tokens = 0
        try:
            est_tokens = estimate_input_tokens(payload)
        except Exception:
            est_tokens = 0
        routing = _ModelRoutingDecision(
            requested_model=str(payload.get('model', '')),
            routed_model=str(payload.get('model', '')),
            classification=None,
            applied=False,
            reason_code='override_no_classifier',
            estimated_input_tokens=est_tokens,
        )
    else:
        _lock = snapshot.config.lock_requested_model
        baseline_model = _lock if _lock != 'off' else None
        if baseline_model:
            logger.info(
                '%s Model lock: forcing routing baseline from %s to %s',
                handler._log_tag(), payload.get('model'), baseline_model,
            )
        routing = _route_model(payload, snapshot, credentials, cached_tier,
                               session_floor, session_ratio,
                               log_tag=handler._log_tag(),
                               override_mode=getattr(handler, '_override_mode', None),
                               task_tag=getattr(handler, '_task_tag', None),
                               ctx_key=ctx_key,
                               baseline_model=baseline_model)

    # Session tier cache (last-resort fallback): persist each fresh
    # classification, and reuse the last tier only when this turn yields
    # no text at all. build_routing_summary now walks prior messages
    # back to the most-recent user-text turn, so ordinary tool_result-
    # only continuations are freshly classified (taking the write
    # branch); the stale-tier read fires only when neither the final
    # message nor any prior user turn yields text.
    if ctx_key is not None and not peer_bound:
        if routing.classification is not None:
            # A valid classification was produced; persist the tier.
            # For affirmation_classified, write the uncapped cache_tier so
            # subsequent turns can apply their own cap when reading from cache.
            if routing.reason_code == 'affirmation_classified':
                handler.registry.set_session_routed_tier(
                    ctx_key, routing.cache_tier or routing.routed_model
                )
            else:
                handler.registry.set_session_routed_tier(ctx_key, routing.routed_model)
        elif routing.reason_code == 'missing_final_user_text':
            cached = handler.registry.session_routed_tier(ctx_key)
            if cached is not None:
                # No-upgrade cap: never replay a cached tier above the client's
                # requested model. baseline_model lock participates in fresh
                # routing decisions but must not override a prior classifier
                # result replayed from cache.
                capped = _cap_cached_tier(
                    cached,
                    routing.requested_model,
                    label_map=snapshot.config.auto_model_routing_classification,
                )
                payload['model'] = capped
                routing = replace(
                    routing,
                    routed_model=capped,
                    applied=(capped != routing.requested_model),
                    reason_code=(
                        'session_cached_tier_capped'
                        if capped != cached
                        else 'session_cached_tier'
                    ),
                )
    # Update routing after any cached-tier replacement so stats always
    # reflect the final routing decision (reason_code, routed_model).

    # DB tier-pin enforcement: if a session tier is pinned via admin UI,
    # apply it regardless of the classifier's decision.
    if handler.session_db is not None:
        try:
            sk = _session_key(payload) if isinstance(payload, dict) else None
            if sk:
                meta = handler.session_db.get_session_metadata(sk)
                pinned = meta.get('pinned_tier') if meta else None
                if pinned and pinned != routing.routed_model:
                    pinned_alias = pinned
                    payload['model'] = pinned_alias
                    routing = replace(
                        routing,
                        routed_model=pinned_alias,
                        applied=(pinned_alias != routing.requested_model),
                        reason_code='db_tier_pin',
                    )
                # Restore pinned backend from DB on restart (lazy recovery).
                pinned_backend_db = meta.get('pinned_backend') if meta else None
                if pinned_backend_db and handler.registry.session_backend(sk) is None:
                    try:
                        handler.registry.set_session_backend(sk, pinned_backend_db)
                    except Exception:
                        logger.debug('%s db backend-pin restore failed',
                                     handler._log_tag(), exc_info=True)
        except Exception:
            logger.debug('%s db tier-pin check failed', handler._log_tag(), exc_info=True)

    if routing.applied or routing.reason_code not in ('disabled', 'model_not_eligible'):
        logger.info(
            '%s Model routing: backend=%s requested=%s routed=%s '
            'classification=%s applied=%s reason=%s '
            'sys_blocks=%d sys_chars=%d ctx_anchor=%s sys_head=%r '
            'predicted=%d sess_ctx=%d est_ratio=%.2f mode=%s',
            handler._log_tag(),
            snapshot.name,
            routing.requested_model,
            routing.routed_model,
            routing.classification,
            routing.applied,
            routing.reason_code,
            _blks,
            _chars,
            ctx_key.split('\x00', 1)[-1] if ctx_key else '--------',
            _head,
            routing.predicted_input_tokens,
            routing.session_context_tokens,
            routing.session_estimate_ratio,
            routing.classifier_mode,
        )


    return routing, ctx_key_result


def prepare_routing(handler, payload, sess_key):
    """Prepare a request for dispatch: routing, tier cache, DB pin, tool stripping, etc."""
    oauth_credential = _oauth_credential(handler)
    snapshot = handler.registry.snapshot_for_request(
        sess_key,
        prefer_backend=getattr(handler, '_prefer_backend', None),
        oauth_credential=oauth_credential,
    )
    credentials = _snapshot_credentials(handler, snapshot)
    routing, ctx_key_result = derive_routing(
        handler, payload, sess_key, snapshot, credentials,
    )

    # Strip deny-listed tools before the request reaches any backend.
    if TOOLS_TO_REMOVE and payload.get('tools'):
        payload['tools'] = [
            t for t in payload['tools']
            if t.get('name') not in TOOLS_TO_REMOVE
        ]
        if not payload['tools']:
            payload.pop('tools', None)
            payload.pop('tool_choice', None)
            payload.pop('parallel_tool_calls', None)
        else:
            # If tool_choice names a now-stripped tool, remove it so the
            # backend doesn't receive a reference to an absent tool.
            tc = payload.get('tool_choice')
            if isinstance(tc, dict) and tc.get('name') in TOOLS_TO_REMOVE:
                payload.pop('tool_choice', None)

    # Capture after stripping so tools_sha256 reflects dispatched payload.
    prompt_capture = handler._extract_prompt_capture(payload, routing=routing)

    return PreparedRequest(
        payload=payload,
        snapshot=snapshot,
        credentials=credentials,
        routing=routing,
        ctx_key=ctx_key_result,
        prompt_capture=prompt_capture,
    )


class ProxyRequestHandler(BaseHTTPRequestHandler):
    # Set by server factory
    registry = None
    config = None
    selector = None         # AutoSelector instance when --auto-backend is active, else None
    stats_collector = None  # StatsCollector instance, or None if stats are disabled
    session_db = None       # SessionDB instance, or None if DB recording disabled
    enable_ui = False       # Whether /admin/* and /ui/* endpoints are active
    _original_requested_model = None  # Set per-request in do_POST/_handle_messages
    _original_anthropic_beta = None   # Set per-request in do_POST/_handle_messages

    def do_POST(self):
        self._req_start = time.monotonic()
        self._session_prefix = None
        # Session-context floor bookkeeping (set in _handle_messages when the
        # feature is active; read at response time to record measured tokens).
        # Keyed on (session_id, system-prompt hash) so a sub-agent does not share
        # the parent's floor — see _context_key.
        self._ctx_key = None
        self._route_est = 0
        # Per-request override state: parsed from X-Anthproxy-Override and reset
        # per-request so HTTP keep-alive connection reuse does not leak overrides
        # between requests.
        self._no_classifier = False
        self._prefer_backend = None
        self._override_mode = None
        self._task_tag = None
        self._db_request_id = None
        self._prompt_capture: dict = {}
        # The single source of truth for "what the client asked for": captured
        # before any route_model call and never overwritten, so every dispatch
        # attempt derives its own decision from it rather than inheriting the
        # previous attempt's routed output as its input.  The beta list is
        # captured with it because the long-context floor appends to it, and a
        # retry that lands on a different backend must not inherit that either.
        self._original_requested_model = None
        self._original_anthropic_beta = None

        overrides = _parse_override_header(self.headers.get('X-Anthproxy-Override'))
        self._no_classifier = overrides.get('no_classifier', False)
        self._prefer_backend = overrides.get('prefer_backend')
        self._override_mode = overrides.get('override_mode')
        self._task_tag = overrides.get('task_tag')

        path = self.path.split('?')[0].rstrip('/')

        if path == '/v1/messages':
            self._handle_messages()
        elif path == '/v1/messages/count_tokens':
            self._handle_count_tokens()
        elif path == '/cache':
            self._handle_cache()
        elif path.startswith('/admin/') or path == '/admin':
            if not getattr(self, 'enable_ui', False):
                self._send_json(404, anthropic_error_payload('not_found_error', 'Admin UI not enabled'))
                return
            from . import admin  # lazy import to avoid circular at module load
            try:
                body = self._read_body()
                try:
                    body_dict = json.loads(body or b'{}')
                except json.JSONDecodeError as exc:
                    self._send_json(400, anthropic_error_payload('invalid_request_error', f'Malformed JSON body: {exc}'))
                    return
                status, response = admin.handle_post(
                    path, body_dict, self.registry, self.session_db,
                    selector=self.selector,
                )
                self._send_json(status, response)
            except Exception:
                logger.exception('Admin POST handler error')
                self._send_json(500, anthropic_error_payload('api_error', 'Admin handler failed'))
        else:
            self._send_json(404, anthropic_error_payload('not_found_error', 'Not found'))

    def do_GET(self):
        import urllib.parse
        self._req_start = time.monotonic()
        self._session_prefix = None
        self._session_hash = None
        path_full = self.path
        path = path_full.split('?')[0].rstrip('/')
        query_string = path_full.split('?', 1)[1] if '?' in path_full else ''
        query_params: dict[str, str] = {}
        if query_string:
            for part in query_string.split('&'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    query_params[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)

        if not getattr(self, 'enable_ui', False):
            self._send_json(404, anthropic_error_payload('not_found_error', 'Admin UI not enabled'))
            return

        if path.startswith('/ui'):
            self._serve_ui_file(path)
        elif path.startswith('/admin'):
            from . import admin  # lazy import to avoid circular at module load
            try:
                status, body = admin.handle_get(
                    path, query_params, self.registry, self.session_db,
                    selector=self.selector,
                )
                self._send_json(status, body)
            except Exception:
                logger.exception('Admin GET handler error')
                self._send_json(500, anthropic_error_payload('api_error', 'Admin handler failed'))
        else:
            self._send_json(404, anthropic_error_payload('not_found_error', 'Not found'))

    def _handle_messages(self):
        snapshot = None
        payload = None
        prepared = None
        try:
            self._validate_content_type()
            body = self._read_body()
            payload = self._parse_json(body)
            logger.debug('>>> Anthropic request: %s', json.dumps(payload, default=str))

            command = _parse_local_command(payload)
            if command is not None:
                self._handle_local_command(command, payload)
                return

            if _has_happy_new_year_system_prompt(payload):
                self._handle_happy_new_year(payload)
                return

            anthropic_beta = self.headers.get('anthropic-beta', '')
            if anthropic_beta:
                payload['_anthropic_beta'] = [
                    item.strip() for item in anthropic_beta.split(',') if item.strip()
                ]

            # Snapshot the active backend once; the whole request lifetime uses
            # this captured backend/config even if another thread switches.
            sess_key = _session_key(payload)
            self._original_requested_model = str(payload.get('model', ''))
            _client_beta = payload.get('_anthropic_beta')
            self._original_anthropic_beta = (
                list(_client_beta) if isinstance(_client_beta, list) else None
            )
            self._session_prefix = _session_short_id(sess_key) if sess_key else None
            _blks, _chars, sys_hash, _head = _system_fingerprint(payload)
            self._session_hash = sys_hash

            prepared = prepare_routing(self, payload, sess_key)
            snapshot = prepared.snapshot
            self._routing = prepared.routing
            self._ctx_key = prepared.ctx_key
            self._route_est = prepared.routing.estimated_input_tokens
            self._prompt_capture = prepared.prompt_capture
            self._dispatch(payload, snapshot, 1, credentials=prepared.credentials)

        except AnthropicRequestError as exc:
            if exc.status_code == 429 and snapshot is not None:
                retry_after: float | None = getattr(exc, 'retry_after', None)

                if snapshot.name == 'oauth':
                    # Retry-After > 0 → transient cooldown; absent or zero →
                    # spend-cap exhaustion, park the token until the next UTC month.
                    if retry_after is not None and retry_after > 0:
                        self.registry.mark_oauth_cooldown(snapshot.credentials, retry_after)
                    else:
                        self.registry.mark_oauth_cap_exhausted(snapshot.credentials)

                # Session-subscription lock: record exhaustion so the next
                # request from this session resolves to the other subscription
                # backend.  Do NOT switch globally or retry in this request.
                elif (snapshot.session_subscription
                        and self.selector is not None):
                    self.selector.note_exhausted(snapshot.name, retry_after)

                elif (
                    not snapshot.session_pinned
                    and self.selector is not None
                    and not self.selector.is_paused()
                ):
                    # Auto-mode: demote the depleted backend and retry once on
                    # the new one.  payload/creds are safe to reuse (idempotent,
                    # body not consumed twice).
                    depleted_name = snapshot.name
                    new_name = self.selector.on_rate_limited(depleted_name, retry_after)
                    if new_name != depleted_name:
                        logger.info('Auto-selector: retrying request on %s after 429 from %s',
                                    new_name, depleted_name)
                        # DB: record 429-failed attempt before retry so the retry
                        # path can update the same row on success.
                        if self.session_db is not None and not getattr(self, '_db_request_id', None):
                            try:
                                _dur_ms = int((time.monotonic() - getattr(self, '_req_start', time.monotonic())) * 1000)
                                self._db_request_id = self.session_db.record_request(
                                    session_id=_session_key(payload) or '' if isinstance(payload, dict) else '',
                                    conversation_anchor=(
                                        self._ctx_key.split('\x00', 1)[-1]
                                        if getattr(self, '_ctx_key', None) and '\x00' in self._ctx_key else None
                                    ),
                                    routing_decision=getattr(self, '_routing', None),
                                    stats_dict={'input_tokens': 0, 'output_tokens': 0,
                                                'cache_creation_tokens': 0, 'cache_read_tokens': 0},
                                    duration_ms=_dur_ms,
                                    backend=snapshot.name,
                                    status='rate_limited',
                                    error='429 — backend rate limited',
                                    attempt=1,
                                    **getattr(self, '_prompt_capture', {}),
                                )
                            except Exception:
                                logger.debug('%s db 429 record failed', self._log_tag(), exc_info=True)
                        self._retry_on_new_backend(payload)
                        return
            # Record the failure only when a backend was involved (snapshot is
            # not None).  Pre-snapshot client errors (bad content-type, malformed
            # JSON) are not backend reliability signals; backend is unknown there.
            if snapshot is not None and self.stats_collector is not None:
                try:
                    duration_ms = int((time.monotonic() - getattr(self, '_req_start', time.monotonic())) * 1000)
                    self.stats_collector.record(
                        snapshot.name,
                        model=payload.get('model', '') if isinstance(payload, dict) else '',
                        duration_ms=duration_ms,
                        streaming=bool(isinstance(payload, dict) and payload.get('stream')),
                        status='error',
                        status_code=exc.status_code,
                        error=exc.error_type,
                        **self._routing_fields(),
                    )
                except Exception:
                    logger.debug('%s stats failure-record failed', self._log_tag(), exc_info=True)
            if snapshot is not None:
                # A 429 reaching here was not retried (oauth cooldown, a pinned
                # session, or the selector having nowhere else to go); it keeps
                # the status the retry path's pre-record uses so the schema's
                # rate_limited/error distinction survives.
                self._record_failed_request(
                    payload, snapshot, f'{exc.status_code} — {exc.error_type}',
                    status='rate_limited' if exc.status_code == 429 else 'error',
                )
            self._send_error(exc)
        except _CLIENT_DISCONNECT:
            logger.info('%s client disconnected before response', self._log_tag())
        except Exception as exc:
            logger.exception('%s Proxy failure: %s', self._log_tag(), exc)
            if snapshot is not None and self.stats_collector is not None:
                try:
                    duration_ms = int((time.monotonic() - getattr(self, '_req_start', time.monotonic())) * 1000)
                    self.stats_collector.record(
                        snapshot.name,
                        model=payload.get('model', '') if isinstance(payload, dict) else '',
                        duration_ms=duration_ms,
                        streaming=bool(isinstance(payload, dict) and payload.get('stream')),
                        status='error',
                        status_code=502,
                        error='upstream_failure',
                        **self._routing_fields(),
                    )
                except Exception:
                    logger.debug('%s stats failure-record failed', self._log_tag(), exc_info=True)
            if snapshot is not None:
                self._record_failed_request(payload, snapshot, '502 — upstream_failure')
            self._send_json(502, anthropic_error_payload('api_error', 'Upstream request failed'))

    def _record_failed_request(self, payload, snapshot, error: str,
                               status: str = 'error') -> None:
        """Persist a request row for a dispatch that failed before any usage was
        learned (ADR-0025).

        An empty ``stats_dict`` records the usage as absent rather than zero —
        see ``SessionDB.record_request``.  Applies to every backend: recording
        peer failures but not provider ones would be a worse inconsistency than
        the silence it replaces.
        """
        if self.session_db is None or getattr(self, '_db_request_id', None):
            return
        try:
            ctx_key = getattr(self, '_ctx_key', None)
            self._db_request_id = self.session_db.record_request(
                session_id=_session_key(payload) or '' if isinstance(payload, dict) else '',
                conversation_anchor=(
                    ctx_key.split('\x00', 1)[-1] if ctx_key and '\x00' in ctx_key else None
                ),
                routing_decision=getattr(self, '_routing', None),
                stats_dict={},
                duration_ms=int(
                    (time.monotonic() - getattr(self, '_req_start', time.monotonic())) * 1000),
                backend=snapshot.name,
                status=status,
                error=error,
                **getattr(self, '_prompt_capture', {}),
            )
        except Exception:
            logger.debug('%s db failure-record failed', self._log_tag(), exc_info=True)

    def _retry_on_new_backend(self, payload: dict):
        """Retry the current payload on the (now-active) new backend.

        Called exactly once after a 429-triggered auto-switch; errors here are
        surfaced normally (no further retries).  The new snapshot's backend name
        is what gets recorded on failure — this is correct, because the new
        backend is what failed, not the original depleted one.
        """
        snapshot = None
        try:
            # The retry must land on the auto-selector's new choice, not the
            # session backend pin. That pin is consulted only in the initial
            # prepare_routing; here we use the fresh selector state to honour
            # the 429.  We do NOT pass sess_key because that would read
            # _session_overrides (the backend pin) and bypass the retry.
            snapshot = self.registry.snapshot()
            # The routing decision belongs to the snapshot that will actually
            # serve the request, so a retry re-derives it from the client's
            # original model.  Inheriting the first attempt's output would
            # either leave routing disabled for a request this instance now
            # serves itself (peer → provider), or send the peer a routed model
            # the client never asked for (provider → peer).
            credentials = None
            if self._original_requested_model:  # non-empty string
                payload['model'] = self._original_requested_model
                # Undo the long-context floor's beta injection too: the peer
                # re-emits _anthropic_beta as an outbound header, so a carried
                # context-1m token would hand it a floor decision this hop was
                # not entitled to make.
                if self._original_anthropic_beta is None:
                    payload.pop('_anthropic_beta', None)
                else:
                    payload['_anthropic_beta'] = list(self._original_anthropic_beta)
                _derive_and_record_routing(self, payload, snapshot)
            self._dispatch(payload, snapshot, 2)
        except AnthropicRequestError as exc:
            if snapshot is not None and self.stats_collector is not None:
                try:
                    duration_ms = int((time.monotonic() - getattr(self, '_req_start', time.monotonic())) * 1000)
                    self.stats_collector.record(
                        snapshot.name,
                        model=payload.get('model', ''),
                        duration_ms=duration_ms,
                        streaming=bool(payload.get('stream')),
                        status='error',
                        status_code=exc.status_code,
                        error=exc.error_type,
                        **self._routing_fields(),
                    )
                except Exception:
                    logger.debug('%s stats failure-record (retry) failed', self._log_tag(), exc_info=True)
            self._send_error(exc)
        except _CLIENT_DISCONNECT:
            logger.info('%s client disconnected before response', self._log_tag())
        except Exception as exc:
            logger.exception('%s Proxy failure on retry: %s', self._log_tag(), exc)
            if snapshot is not None and self.stats_collector is not None:
                try:
                    duration_ms = int((time.monotonic() - getattr(self, '_req_start', time.monotonic())) * 1000)
                    self.stats_collector.record(
                        snapshot.name,
                        model=payload.get('model', ''),
                        duration_ms=duration_ms,
                        streaming=bool(payload.get('stream')),
                        status='error',
                        status_code=502,
                        error='upstream_failure',
                        **self._routing_fields(),
                    )
                except Exception:
                    logger.debug('%s stats failure-record (retry) failed', self._log_tag(), exc_info=True)
            self._send_json(502, anthropic_error_payload('api_error', 'Upstream request failed'))

    def _oauth_cooldown_wrapper(self, generator, snapshot):
        try:
            yield from generator
        except AnthropicRequestError as exc:
            if exc.status_code == 429:
                retry_after = getattr(exc, 'retry_after', None)
                if retry_after is not None and retry_after > 0:
                    self.registry.mark_oauth_cooldown(snapshot.credentials, retry_after)
                else:
                    self.registry.mark_oauth_cap_exhausted(snapshot.credentials)
            raise

    def _dispatch(self, payload: dict, snapshot, attempt: int,
                  credentials: dict | None = None) -> None:
        """Snapshot → log → parse creds → stream or non-stream send.

        The caller creates and retains the snapshot so the 429 handler knows
        which backend depleted.  Raises ``AnthropicRequestError`` or propagates
        backend exceptions on failure.

        ``credentials`` may be pre-parsed by the caller (e.g. when the same
        credential set was already used for the classifier call).  When None,
        credentials are parsed here from the request headers as before.
        """
        logger.info('%s Routing request: operation=messages backend=%s stream=%s attempt=%d',
                    self._log_tag(), snapshot.name, bool(payload.get('stream')), attempt)
        if credentials is None:
            credentials = snapshot.backend.parse_credentials(self.headers.get('x-api-key', ''))
        start_time = time.monotonic()
        backend_name = snapshot.name
        model = payload.get('model', '')

        if payload.get('stream'):
            sse_gen = snapshot.backend.send_message_stream(payload, credentials, snapshot.config)
            if snapshot.name == 'oauth':
                sse_gen = self._oauth_cooldown_wrapper(sse_gen, snapshot)
            # Priming (and keepalive) happens inside _send_sse after HTTP 200 is
            # committed, so the client sees a response immediately rather than
            # waiting for the upstream first byte (which can take 60-90 s under
            # load).  A pre-stream 429 is now delivered as an in-band SSE error
            # event rather than an HTTP 429 status.
            _anchor = getattr(self, '_ctx_key', None)
            _anchor = _anchor.split('\x00', 1)[-1] if _anchor and '\x00' in _anchor else None
            wrapped = self._usage_sse_wrapper(
                sse_gen, None, backend_name, start_time, model,
                session_id=_session_key(payload) or '',
                conversation_anchor=_anchor,
                routing_decision=getattr(self, '_routing', None),
            )
            # Echo the client's requested model in message_start (no-op unless
            # routing rewrote the model).  Outermost wrapper so stats/db above
            # still record the routed model.
            wrapped = self._rewrite_response_model_sse(wrapped)
            self._send_sse(wrapped, config=snapshot.config)
        else:
            result = snapshot.backend.send_message(payload, credentials, snapshot.config)
            logger.debug('<<< Anthropic response: %s', json.dumps(result, default=str))
            _resp_text = _extract_response_text(result)
            usage = result.get('usage', {}) or {}
            if self.stats_collector is not None:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                self.stats_collector.record(
                    backend_name,
                    model=model,
                    input_tokens=usage.get('input_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    cache_creation_tokens=usage.get('cache_creation_input_tokens', 0),
                    cache_read_tokens=usage.get('cache_read_input_tokens', 0),
                    duration_ms=duration_ms,
                    streaming=False,
                    status='success',
                    status_code=200,
                    **self._routing_fields(),
                )
            # Session-context floor is independent of stats collection.
            self._record_session_context(
                usage.get('input_tokens', 0),
                usage.get('cache_creation_input_tokens', 0),
                usage.get('cache_read_input_tokens', 0),
                usage.get('output_tokens', 0),
            )
            econ = self._log_routing_economics(
                routed_model=model,
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                cache_creation_tokens=usage.get('cache_creation_input_tokens', 0),
                cache_read_tokens=usage.get('cache_read_input_tokens', 0),
            )
            # DB recording (independent of stats collection).
            if self.session_db is not None and not getattr(self, '_db_request_id', None):
                try:
                    duration_ms_db = int((time.monotonic() - start_time) * 1000)
                    self._db_request_id = self.session_db.record_request(
                        session_id=_session_key(payload) or '',
                        conversation_anchor=(
                            self._ctx_key.split('\x00', 1)[-1]
                            if getattr(self, '_ctx_key', None) and '\x00' in self._ctx_key else None
                        ),
                        routing_decision=getattr(self, '_routing', None),
                        stats_dict={
                            'input_tokens': usage.get('input_tokens', 0),
                            'output_tokens': usage.get('output_tokens', 0),
                            'cache_creation_tokens': usage.get('cache_creation_input_tokens', 0),
                            'cache_read_tokens': usage.get('cache_read_input_tokens', 0),
                        },
                        duration_ms=duration_ms_db,
                        backend=backend_name,
                        status='success',
                        response_text=_resp_text,
                        **_economics_kwargs(econ),
                        **getattr(self, '_prompt_capture', {}),
                    )
                except Exception:
                    logger.debug('%s db record (non-streaming) failed', self._log_tag(), exc_info=True)
            elif self.session_db is not None and getattr(self, '_db_request_id', None):
                # Retry success: update pre-created rate_limited row.
                try:
                    from .db import compute_cost
                    _stats_d = {
                        'input_tokens': usage.get('input_tokens', 0),
                        'output_tokens': usage.get('output_tokens', 0),
                        'cache_creation_tokens': usage.get('cache_creation_input_tokens', 0),
                        'cache_read_tokens': usage.get('cache_read_input_tokens', 0),
                    }
                    _routing = getattr(self, '_routing', None)
                    _cost = compute_cost(_routing.routed_model if _routing else '', _stats_d)
                    self.session_db.update_request_on_retry(
                        request_id=self._db_request_id,
                        new_backend=backend_name,
                        attempt=2,
                        cost_estimate=_cost,
                        status='success',
                        error=None,
                        response_text=_resp_text,
                        **_stats_d,
                    )
                except Exception:
                    logger.debug('%s db retry-update (non-streaming) failed', self._log_tag(), exc_info=True)
            # Echo the client's requested model in the response body (stats and
            # session-context above already recorded the routed model).  No-op
            # unless routing actually rewrote the model.
            routing = getattr(self, '_routing', None)
            if routing is not None and routing.applied and isinstance(result, dict) \
                    and 'model' in result:
                result['model'] = routing.requested_model
            self._send_json(200, result)

    def _rewrite_response_model_sse(self, sse_gen):
        """Wrap an SSE generator to echo the client's requested model.

        No-op unless model-tier routing actually rewrote the model
        (``self._routing.applied``); when it did, the first ``message_start``
        event's ``message.model`` is rewritten back to the requested model so the
        client sees what it asked for.  Applied as the *outermost* wrapper so it
        covers both the stats-on and stats-off paths and every backend uniformly.

        Once the rewrite has fired, later chunks pass through untouched (the
        ``rewritten`` latch avoids re-parsing every subsequent event), and the
        wrapped generator's ``close()`` is chained so the backend/stats ``finally``
        blocks still run synchronously.
        """
        routing = getattr(self, '_routing', None)
        requested = routing.requested_model if routing and routing.applied else None
        if not requested:
            yield from sse_gen
            return
        rewritten = False
        try:
            for chunk in sse_gen:
                if not rewritten:
                    new_chunk = _rewrite_message_start_model(chunk, requested)
                    if new_chunk is not chunk:
                        rewritten = True
                    yield new_chunk
                else:
                    yield chunk
        finally:
            close = getattr(sse_gen, 'close', None)
            if close is not None:
                close()

    def _routing_fields(self) -> dict:
        """Return stats kwargs for routing metadata, safe when _routing is unset.

        Reads ``self._routing`` via getattr so handlers that bypass
        ``_handle_messages`` (count_tokens, cache, local commands, and unit
        tests that call ``_dispatch`` directly) return empty/None defaults rather
        than raising AttributeError.
        """
        routing = getattr(self, '_routing', None)
        if routing is None:
            return {'requested_model': '', 'classification': None, 'reason_code': None}
        return {
            'requested_model': routing.requested_model or '',
            'classification': routing.classification,
            'reason_code': routing.reason_code,
        }

    def _log_routing_economics(
        self,
        routed_model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
    ):
        """Compute and log per-request routing cost breakdown as a single INFO line.

        Returns the :class:`RoutingEconomics` result (against the opus baseline) so
        callers can persist it, or ``None`` when economics could not be computed —
        e.g. ``self._routing`` is not set (local commands, count_tokens, classifier
        requests, direct unit-test calls to ``_dispatch``) or an error occurred.
        Never raises — errors are silently swallowed so this call can never affect
        request handling.
        """
        routing = getattr(self, '_routing', None)
        if routing is None:
            return None
        try:
            from .stats import routing_economics as _routing_economics
            config = getattr(self, 'config', None)
            classifier_model = (
                getattr(config, 'auto_model_routing_classifier_model', 'haiku')
                if config is not None else 'haiku'
            )
            econ = _routing_economics(
                routed_model=routed_model,
                classifier_model=classifier_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                classifier_input_tokens=routing.classifier_input_tokens,
                classifier_output_tokens=routing.classifier_output_tokens,
            )
            logger.info(
                '%s Routing economics: req=%s routed=%s reason=%s '
                'pricing=%s opus_baseline=$%.6f routed_cost=$%.6f '
                'clf_overhead=$%.6f net_savings=$%.6f',
                self._log_tag(),
                routing.requested_model or routed_model,
                routed_model,
                routing.reason_code,
                econ.pricing_available,
                econ.opus_baseline_cost,
                econ.routed_cost,
                econ.classifier_overhead_usd,
                econ.net_savings_usd,
            )
            return econ
        except Exception:
            logger.debug('%s routing economics log failed', self._log_tag(), exc_info=True)
            return None

    def _usage_sse_wrapper(self, sse_gen, first_chunk, backend_name: str, start_time: float,
                           model: str = '', *, session_id: str = '',
                           conversation_anchor=None, routing_decision=None):
        """Yield SSE chunks, recording stats and DB data after the stream ends.

        Merges the former ``_stats_sse_wrapper`` and ``_db_sse_wrapper`` so the
        SSE stream is parsed exactly once.  Token stats are accumulated via
        ``_extract_sse_stats`` (max-tracking of cumulative usage restatements);
        response text is accumulated via ``_extract_sse_text`` only when a
        ``SessionDB`` is configured.

        Detects ``event: error`` lines in the stream (including pre-stream 429s
        delivered as in-band SSE errors after HTTP 200 has been committed) so the
        recorded status reflects ``status='error'`` rather than ``status='success'``.
        A rolling line-tail buffer accumulates the tail of the previous chunk so an
        ``event: error`` header split across two chunk boundaries is still detected.

        ``first_chunk`` supports priming: the caller may pull the first chunk to
        surface a pre-stream in-band error before it is yielded downstream.
        """
        stats = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
        }
        routing_fields = self._routing_fields()
        capture_text = self.session_db is not None
        text_parts: list[str] = []
        errored = False
        # An upstream failure raised out of the stream (during priming or
        # mid-stream) is delivered to the client by _send_sse as an out-of-band
        # error frame, so it never appears as an "event: error" line here.
        # Without this, a stream that carried no response at all would be
        # recorded as a success that cost nothing (ADR-0025).
        upstream_error: tuple[int | None, str] | None = None
        # Rolling line buffer: accumulate the tail of the previous chunk so
        # an "event: error" header split across two chunk boundaries is still
        # detected.  We only need the last ~20 bytes from the previous chunk
        # (longest event-line prefix we care about) plus the current chunk.
        _line_tail = ''
        try:
            if first_chunk is not None:
                _extract_sse_stats(first_chunk, stats)
                if capture_text:
                    _extract_sse_text(first_chunk, text_parts)
                _combined = _line_tail + first_chunk
                if 'event: error' in _combined:
                    errored = True
                _line_tail = first_chunk[-20:]
                yield first_chunk
            for chunk in sse_gen:
                _extract_sse_stats(chunk, stats)
                if capture_text:
                    _extract_sse_text(chunk, text_parts)
                _combined = _line_tail + chunk
                if 'event: error' in _combined:
                    errored = True
                _line_tail = chunk[-20:]
                yield chunk
        except AnthropicRequestError as exc:
            upstream_error = (exc.status_code, exc.error_type)
            raise
        except Exception:
            upstream_error = (502, 'upstream_failure')
            raise
        finally:
            close = getattr(sse_gen, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.debug('%s sse_gen close failed', self._log_tag(), exc_info=True)
            duration_ms = int((time.monotonic() - start_time) * 1000)
            if upstream_error is not None:
                errored = True
            if self.stats_collector is not None:
                if upstream_error is not None:
                    status_kwargs = {'status': 'error', 'error': upstream_error[1],
                                     'status_code': upstream_error[0]}
                elif errored:
                    status_kwargs = {'status': 'error', 'error': 'sse_error', 'status_code': None}
                else:
                    status_kwargs = {'status': 'success', 'status_code': 200}
                self.stats_collector.record(
                    backend_name,
                    model=model,
                    duration_ms=duration_ms,
                    streaming=True,
                    **stats,
                    **status_kwargs,
                    **routing_fields,
                )
            # Session-context floor and routing economics are independent of
            # stats collection and always run.
            self._record_session_context(
                stats['input_tokens'],
                stats['cache_creation_tokens'],
                stats['cache_read_tokens'],
                stats['output_tokens'],
            )
            econ = self._log_routing_economics(
                routed_model=model,
                input_tokens=stats['input_tokens'],
                output_tokens=stats['output_tokens'],
                cache_creation_tokens=stats['cache_creation_tokens'],
                cache_read_tokens=stats['cache_read_tokens'],
            )
            if self.session_db is not None:
                response_text = ''.join(text_parts) or None
                status = 'error' if errored else 'success'
                if upstream_error is not None:
                    # Nothing was learned about usage, so record it as absent.
                    db_stats, db_error = {}, f'{upstream_error[0]} — {upstream_error[1]}'
                else:
                    db_stats, db_error = stats, ('sse_error' if errored else None)
                db_request_id = getattr(self, '_db_request_id', None)
                if db_request_id:
                    # Retry path: update the pre-created rate_limited row.
                    try:
                        from .db import compute_cost
                        cost = compute_cost(
                            routing_decision.routed_model if routing_decision else '', stats)
                        self.session_db.update_request_on_retry(
                            request_id=db_request_id,
                            new_backend=backend_name,
                            attempt=2,
                            input_tokens=stats['input_tokens'],
                            output_tokens=stats['output_tokens'],
                            cache_creation_tokens=stats['cache_creation_tokens'],
                            cache_read_tokens=stats['cache_read_tokens'],
                            cost_estimate=cost,
                            status=status,
                            error=None,
                            response_text=response_text,
                        )
                    except Exception:
                        logger.debug('%s db retry-update (streaming) failed',
                                     self._log_tag(), exc_info=True)
                else:
                    # Normal path: create new row.
                    try:
                        self.session_db.record_request(
                            session_id=session_id,
                            conversation_anchor=conversation_anchor,
                            routing_decision=routing_decision,
                            stats_dict=db_stats,
                            duration_ms=duration_ms,
                            backend=backend_name,
                            status=status,
                            error=db_error,
                            response_text=response_text,
                            **_economics_kwargs(econ),
                            **getattr(self, '_prompt_capture', {}),
                        )
                    except Exception:
                        logger.debug('%s db record (streaming) failed',
                                     self._log_tag(), exc_info=True)

    @staticmethod
    def _extract_user_prompt_text(payload: dict) -> str | None:
        """Return text from the last user message for DB storage.

        For text messages: returns joined raw text (no strip_reminders).
        For tool_result-only messages: returns bracketed descriptions so the
        field is non-NULL and the UI can show which tools fired and what they
        returned, instead of "(no text content in final message)".
        Returns None only when no user message or no content exists at all.
        """
        messages = payload.get('messages')
        if not isinstance(messages, list) or not messages:
            return None
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get('role') != 'user':
                continue
            content = msg.get('content')
            if isinstance(content, str):
                return content or None
            if not isinstance(content, list):
                return None
            text_parts: list[str] = []
            tool_result_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get('type')
                if btype == 'text':
                    t = block.get('text', '')
                    if isinstance(t, str) and t:
                        text_parts.append(t)
                elif btype == 'tool_result':
                    tid = block.get('tool_use_id') or '?'
                    result_content = block.get('content')
                    if isinstance(result_content, str):
                        preview = result_content[:300]
                    elif isinstance(result_content, list):
                        parts = [
                            b.get('text', '')
                            for b in result_content
                            if isinstance(b, dict) and b.get('type') == 'text'
                            and isinstance(b.get('text'), str)
                        ]
                        preview = ' '.join(parts)[:300]
                    else:
                        preview = ''
                    error_tag = ' [error]' if block.get('is_error') else ''
                    tool_result_parts.append(
                        f'[tool_result {tid}{error_tag}: {repr(preview)}]'
                    )
            if text_parts:
                return ''.join(text_parts) or None
            if tool_result_parts:
                return '\n'.join(tool_result_parts) or None
            return None
        return None

    def _extract_prompt_capture(self, payload: dict, routing=None) -> dict:
        """Extract prompt-capture fields for DB recording.  Never raises.

        Returns a dict whose keys exactly match the new schema-v2 keyword params
        of ``SessionDB.record_request()``::

            user_prompt_text, system_prompt_sha256, tools_sha256,
            prompt_store_entries, routing_recovered_via_walkback,
            classifier_model, classifier_summary_json, classifier_raw_response,
            classifier_confidence, classifier_format

        Callers pass the dict as ``**getattr(self, '_prompt_capture', {})``.
        """
        try:
            # 1. Raw user prompt text (no strip_reminders)
            user_prompt_text = self._extract_user_prompt_text(payload)

            # 2. System prompt SHA-256
            system = payload.get('system')
            system_content_str: str | None = None
            system_prompt_sha256: str | None = None
            if system:
                if isinstance(system, str):
                    system_content_str = system
                elif isinstance(system, list):
                    system_content_str = json.dumps(system, sort_keys=True, ensure_ascii=False)
                if system_content_str is not None:
                    system_prompt_sha256 = hashlib.sha256(
                        system_content_str.encode('utf-8')
                    ).hexdigest()

            # 3. Tools SHA-256
            tools = payload.get('tools')
            tools_content_str: str | None = None
            tools_sha256: str | None = None
            if tools:
                tools_content_str = json.dumps(tools, sort_keys=True, ensure_ascii=False)
                tools_sha256 = hashlib.sha256(
                    tools_content_str.encode('utf-8')
                ).hexdigest()

            # 4. Prompt store entries: {sha256: (content_type, content_str)}
            prompt_store_entries: dict[str, tuple[str, str]] = {}
            if system_prompt_sha256 is not None and system_content_str is not None:
                prompt_store_entries[system_prompt_sha256] = ('system', system_content_str)
            if tools_sha256 is not None and tools_content_str is not None:
                prompt_store_entries[tools_sha256] = ('tools', tools_content_str)

            # 5. routing_recovered_via_walkback — from RoutingSummary.
            # build_routing_summary() is a pure read from payload['messages'] and
            # does not mutate the payload; calling it again here is safe.
            routing_recovered_via_walkback: bool | None = None
            try:
                summary = _build_routing_summary(payload)
                if summary is not None:
                    routing_recovered_via_walkback = summary.recovered_via_walkback
            except Exception:
                pass  # fail-open: leave as None

            # 6. Classifier transparency fields — from ModelRoutingDecision.
            # Prefer the routing passed in directly (available during prepare_routing);
            # fall back to self._routing for callers that don't pass it.
            _routing = routing if routing is not None else getattr(self, '_routing', None)
            classifier_model = getattr(_routing, 'classifier_model', None)
            classifier_summary_json = getattr(_routing, 'classifier_summary_json', None)
            classifier_raw_response = getattr(_routing, 'classifier_raw_response', None)
            classifier_confidence = getattr(_routing, 'classifier_confidence', None)
            classifier_format = getattr(_routing, 'classifier_format', None)

            # 7. ADR 0010/0011: weighted blend fields from ModelRoutingDecision.
            system_prompt_tier = getattr(_routing, 'system_prompt_tier', None)
            system_prompt_score = getattr(_routing, 'system_prompt_score', None)
            user_prompt_score = getattr(_routing, 'user_prompt_score', None)
            routing_weighted_score = getattr(_routing, 'routing_weighted_score', None)
            system_prompt_classification_failed = getattr(
                _routing, 'system_prompt_classification_failed', False
            )
            user_prompt_tier = getattr(_routing, 'user_prompt_tier', None)

            return {
                'user_prompt_text': user_prompt_text,
                'system_prompt_sha256': system_prompt_sha256,
                'tools_sha256': tools_sha256,
                'prompt_store_entries': prompt_store_entries,
                'routing_recovered_via_walkback': routing_recovered_via_walkback,
                'classifier_model': classifier_model,
                'classifier_summary_json': classifier_summary_json,
                'classifier_raw_response': classifier_raw_response,
                'classifier_confidence': classifier_confidence,
                'classifier_format': classifier_format,
                'system_prompt_tier': system_prompt_tier,
                'system_prompt_score': system_prompt_score,
                'user_prompt_score': user_prompt_score,
                'routing_weighted_score': routing_weighted_score,
                'system_prompt_classification_failed': system_prompt_classification_failed,
                'user_prompt_tier': user_prompt_tier,
            }
        except Exception:
            logger.warning('%s prompt capture extraction failed', self._log_tag(), exc_info=True)
            return {}

    def _record_session_context(self, input_tokens, cache_creation, cache_read, output):
        """Persist the latest measured context size + calibration ratio for a session.

        Called after a response with the upstream usage figures.  No-op unless the
        session-context floor is active for this request (``self._ctx_key`` set in
        ``_handle_messages``).  The key is ``(session_id, first-user-message hash)`` so
        a sub-agent records under its own slot, not the parent's (see ``_context_key``).
        ``measured_input`` excludes output and pairs with the route-time estimate to
        refresh the ratio; the stored floor includes output as a lower bound on the
        next turn's input.  Skips a zero measurement (e.g. an early client
        disconnect) so it cannot wrongly reset a real floor.
        """
        # getattr guards cover handlers that bypass do_POST (e.g. direct unit-test
        # calls to _dispatch / _usage_sse_wrapper) and reused handler instances.
        ctx_key = getattr(self, '_ctx_key', None)
        if not ctx_key:
            return
        measured_input = int(input_tokens or 0) + int(cache_creation or 0) + int(cache_read or 0)
        floor = measured_input + int(output or 0)
        if floor <= 0:
            return
        _prior_floor, prior_ratio = self.registry.session_context(ctx_key)
        ratio = _calibrated_ratio(measured_input, getattr(self, '_route_est', 0), prior_ratio)
        self.registry.record_session_context(ctx_key, floor, ratio)

    def _handle_local_command(self, command, payload: dict):
        name, arg = command
        sk = _session_key(payload)
        self._session_prefix = _session_short_id(sk) if sk else None
        _blks, _chars, sys_hash, _head = _system_fingerprint(payload)
        self._session_hash = sys_hash
        logger.info('%s Routing request: operation=local_command command=%s active_backend=%s stream=%s',
                    self._log_tag(), name, self.registry.active_name(), bool(payload.get('stream')))
        if name == 'help':
            markdown = self._help_markdown()
        elif name == 'status':
            markdown = self._status_markdown(_session_key(payload))
        elif name == 'get-usage':
            markdown = self._usage_markdown()
        elif name == 'get-backend':
            sess_key = _session_key(payload)
            override = self.registry.session_backend(sess_key) if sess_key else None
            if override == SESSION_SUBSCRIPTION_SENTINEL:
                # Resolve to the concrete backend the session would use right now
                resolved = self.registry.snapshot(sess_key).name
                markdown = (
                    f'{_BACKEND_HEADER}'
                    f'**Active backend (this session):** subscription-locked '
                    f'(currently `{resolved}`)\n\n'
                    f'_Global default: `{self.registry.active_name()}`_'
                )
            elif override is not None:
                markdown = (
                    f'{_BACKEND_HEADER}'
                    f'**Active backend (this session):** `{override}`\n\n'
                    f'_Global default: `{self.registry.active_name()}`_'
                )
            else:
                markdown = (
                    f'{_BACKEND_HEADER}'
                    f'**Active backend:** `{self.registry.active_name()}`'
                )
        elif name == 'stats':
            markdown = self._stats_markdown(arg)
        elif name == 'session-set-backend':
            markdown = self._session_set_backend_markdown(arg, _session_key(payload))
        elif name == 'set-model-routing':
            markdown = self._set_model_routing_markdown(arg)
        elif name == 'session-set-model-routing':
            markdown = self._session_set_model_routing_markdown(arg, _session_key(payload))
        else:
            markdown = self._set_backend_markdown(arg)

        model = payload.get('model', '')
        if payload.get('stream'):
            self._send_sse(_local_message_sse(markdown, model))
        else:
            self._send_json(200, _local_message(markdown, model))

    def _handle_happy_new_year(self, payload: dict):
        sk = _session_key(payload)
        self._session_prefix = _session_short_id(sk) if sk else None
        _blks, _chars, sys_hash, _head = _system_fingerprint(payload)
        self._session_hash = sys_hash
        model = payload.get('model', '')
        logger.info('%s Routing request: operation=happy_new_year', self._log_tag())
        if payload.get('stream'):
            self._send_sse(_local_message_sse(HAPPY_BIRTHDAY_REPLY, model))
        else:
            self._send_json(200, _local_message(HAPPY_BIRTHDAY_REPLY, model))

    def _help_markdown(self) -> str:
        available = ', '.join(f'`{n}`' for n in _backend_names())
        subscription = '/'.join(SUBSCRIPTION_BACKENDS)
        auto_row = ''
        subscription_row = ''
        if self.selector is not None:
            auto_row = '| `proxy-set-backend:auto` | Resume auto-selection (clear manual pin or subscription restriction) |\n'
            subscription_row = f'| `proxy-set-backend:subscription` | Auto-select among subscription backends only ({subscription}); never use bedrock |\n'
        return (
            '## anthproxy commands\n\n'
            '| Command | Description |\n'
            '|---|---|\n'
            '| `proxy-help` | Show this help |\n'
            '| `proxy-status` | Active backend + subscription usage |\n'
            '| `proxy-get-backend` | Report the active backend |\n'
            f'| `proxy-set-backend:<name>` | Switch backend globally; `<name>` is one of {available} |\n'
            f'{auto_row}'
            f'{subscription_row}'
            f'| `proxy-set-backend:<name>:session` | Pin backend for this session only; `<name>` is one of {available} |\n'
            '| `proxy-set-backend:auto:session` | Clear this session\'s backend pin (follow global) |\n'
            f'| `proxy-set-backend:subscription:session` | Lock this session to subscription backends only ({subscription}) |\n'
            f'| `proxy-get-usage` | Subscription usage ({subscription} only) |\n'
            '| `proxy-set-model-routing:on\\|off` | Enable/disable LLM model-tier routing globally |\n'
            '| `proxy-set-model-routing:on\\|off:session` | Enable/disable model-tier routing for this session only |\n'
            '| `proxy-set-model-routing:auto:session` | Clear this session\'s routing override (follow global) |\n'
            '\n'
            '**Per-request routing overrides** (send in `X-Anthproxy-Override` header, semicolon-separated):\n\n'
            '| Directive | Effect |\n'
            '|---|---|\n'
            '| `no-classifier` | Bypass all routing: no classifier call, no size floor, no session cache |\n'
            '| `prefer:<backend>` | Soft-prefer a backend for this request (falls back if unavailable) |\n'
            '| `route:classifier` | Use LLM classifier for this request (overrides config mode) |\n'
            '| `route:rules` | Use keyword rules for this request (no LLM call) |\n'
            '| `route:tag` | Use task-tag lookup for this request (requires `task:<name>`) |\n'
            '| `task:<name>` | Task name for `route:tag` mode; looked up in `--auto-model-routing-task-tiers` |\n'
            '\n'
            '| `proxy-stats` / `proxy-stats:1d` / `proxy-stats:d` | Today grouped by hour, per model + cost |\n'
            '| `proxy-stats:-Nd` | N days ago grouped by hour, per model + cost |\n'
            '| `proxy-stats:1w` / `proxy-stats:w` | This week grouped by day, per model + cost |\n'
            '| `proxy-stats:-Nw` | N weeks ago grouped by day, per model + cost |\n'
            '| `proxy-stats:1m` / `proxy-stats:m` | This month grouped by week, per model + cost |\n'
            '| `proxy-stats:-Nm` | N months ago grouped by week, per model + cost |\n'
            '| `proxy-stats:1q` / `proxy-stats:q` | This quarter grouped by month, per model + cost |\n'
            '| `proxy-stats:-Nq` | N quarters ago grouped by month, per model + cost |\n'
            f'| `proxy-stats:<period>:<name>` | Scope any stats period to one backend; `<name>` is one of {available} or `subscription` ({"+".join(SUBSCRIPTION_BACKENDS)}) |'
        )

    def _set_model_routing_markdown(self, arg) -> str:
        """Toggle auto model routing on (True) or off (False) globally.

        ``arg`` is ``True``, ``False``, or ``'invalid'``.
        """
        _HEADER = '## anthproxy model routing\n\n'
        if arg == 'invalid':
            return (
                f'{_HEADER}'
                '**Error:** unrecognised value. '
                'Use `proxy-set-model-routing:on` or `proxy-set-model-routing:off`.'
            )
        self.registry.set_model_routing(arg)
        state = 'on' if arg else 'off'
        return f'{_HEADER}**Auto model routing:** {state} (global)'

    def _session_set_model_routing_markdown(self, arg, sess_key: str | None) -> str:
        """Set or clear per-session model-routing override.

        ``arg`` is ``True`` (on), ``False`` (off), ``None`` (auto/clear), or
        ``'invalid'`` (unrecognised value in the command).
        """
        _HEADER = '## anthproxy model routing\n\n'
        if arg == 'invalid':
            return (
                f'{_HEADER}'
                '**Error:** unrecognised value. '
                'Use `proxy-set-model-routing:on:session`, '
                '`proxy-set-model-routing:off:session`, or '
                '`proxy-set-model-routing:auto:session` to clear.'
            )
        if sess_key is None:
            return (
                f'{_HEADER}'
                '**Error:** session key not available — '
                '`metadata.user_id` is required for session-scoped commands.'
            )
        self.registry.set_session_model_routing(sess_key, arg)
        if arg is None:
            return (
                f'{_HEADER}'
                '**Auto model routing (this session):** following global setting'
            )
        state = 'on' if arg else 'off'
        return (
            f'{_HEADER}'
            f'**Auto model routing (this session):** {state}'
            f'\n\n_Use `proxy-set-model-routing:auto:session` to follow the global setting._'
        )

    def _status_markdown(self, sess_key: str | None = None) -> str:
        backend_name = self.registry.active_name()
        routing_on = bool(self.config.auto_model_routing)
        routing_label = 'on' if routing_on else 'off'
        if routing_on:
            mode = getattr(self.config, 'auto_model_routing_mode', 'classifier') or 'classifier'
            if mode == 'classifier':
                classifier = getattr(self.config, 'auto_model_routing_classifier_model', 'haiku')
                routing_label += f' (mode: classifier, model: `{classifier}`)'
            elif mode == 'rules':
                routing_label += ' (mode: rules — deterministic, no LLM call)'
            elif mode == 'tag':
                routing_label += ' (mode: tag — task header required)'
            else:
                routing_label += f' (mode: {mode})'
        lines = [
            '## anthproxy status\n',
            f'**Active backend:** `{backend_name}`\n',
            f'**Model routing:** {routing_label}\n',
        ]
        if sess_key:
            session_routing = self.registry.session_model_routing(sess_key)
            if session_routing is True:
                lines.append(
                    '\n_This session has model routing **on** '
                    '(use `proxy-set-model-routing:auto:session` to follow global)._\n'
                )
            elif session_routing is False:
                lines.append(
                    '\n_This session has model routing **off** '
                    '(use `proxy-set-model-routing:auto:session` to follow global)._\n'
                )
            override = self.registry.session_backend(sess_key)
            if override == SESSION_SUBSCRIPTION_SENTINEL:
                lines.append(
                    '\n_This session is locked to subscription backends '
                    '(use `proxy-set-backend:auto:session` to clear)._\n'
                )
            elif override is not None:
                lines.append(
                    f'\n_This session is pinned to `{override}` '
                    f'(use `proxy-set-backend:auto:session` to clear)._\n'
                )
        if self.selector is not None:
            lines.append(f'\n_{self.selector.status_line()}_\n')

        usage_backends = ['anthropic', 'codex']
        # OpenRouter is opt-in and constructs regardless of config, so only
        # surface its credit balance when an API key is actually configured.
        if getattr(self.config, 'openrouter_api_key', ''):
            usage_backends.append('openrouter')
        for name in usage_backends:
            try:
                backend = self.registry.instance(name)
            except Exception:
                continue
            if not hasattr(backend, 'get_usage_markdown'):
                continue
            lines.append('\n')
            try:
                usage_markdown = backend.get_usage_markdown(self.config)
                if name != backend_name:
                    usage_markdown = _qualify_first_heading(
                        usage_markdown, '*not the active backend*'
                    )
                lines.append('\n')
                lines.append(usage_markdown)
            except Exception as exc:
                logger.exception('Status usage fetch failed for %s: %s', name, exc)
                lines.append(f'_{name} usage information is temporarily unavailable._')

        return ''.join(lines)

    def _usage_markdown(self) -> str:
        snapshot = self.registry.snapshot()
        get_usage_markdown = getattr(snapshot.backend, 'get_usage_markdown', None)
        if get_usage_markdown is None:
            return (
                '## Subscription usage\n\n'
                'Usage reporting is available only while a subscription-based backend '
                '(`codex` or `anthropic`) is active. Switch with `proxy-set-backend:codex` '
                'or `proxy-set-backend:anthropic`, or start anthproxy with '
                '`--backend codex` / `--backend anthropic`.'
            )
        try:
            return get_usage_markdown(snapshot.config)
        except Exception as exc:
            logger.exception('Local usage command failed: %s', exc)
            return (
                '## Subscription usage\n\n'
                'Usage information is temporarily unavailable.'
            )

    def _stats_markdown(self, period) -> str:
        if self.stats_collector is None:
            return '## anthproxy stats\n\nStats collection is not enabled.'
        from .stats import format_stats_markdown
        try:
            records = self.stats_collector.read_records(period.start_ts, period.end_ts)
            return format_stats_markdown(records, period)
        except Exception as exc:
            logger.exception('Stats command failed: %s', exc)
            return '## anthproxy stats\n\nStats information is temporarily unavailable.'

    def _set_backend_markdown(self, arg: str | None) -> str:
        available = ', '.join(f'`{n}`' for n in _backend_names())

        # proxy-set-backend:auto → resume auto-selection
        if arg == 'auto':
            if self.selector is None:
                return (
                    f'{_BACKEND_HEADER}'
                    'Auto-selection is not enabled. Restart anthproxy with `--auto-backend` to use it.'
                )
            new_name = self.selector.resume()
            return (
                f'{_BACKEND_HEADER}'
                f'Auto-selection resumed. Active backend: `{new_name}`.'
            )

        # proxy-set-backend:subscription → restrict to subscription backends globally
        if arg == 'subscription':
            if self.selector is None:
                return (
                    f'{_BACKEND_HEADER}'
                    'Subscription-only mode requires auto-backend to be enabled. '
                    'Restart anthproxy with `--auto-backend` to use it.'
                )
            new_name = self.selector.restrict_subscription()
            return (
                f'{_BACKEND_HEADER}'
                f'Restricted to subscription backends ({"/".join(SUBSCRIPTION_BACKENDS)}). '
                f'Auto-selection continues among them; bedrock is never used as a fallback.\n\n'
                f'Active backend: `{new_name}`. '
                f'Send `proxy-set-backend:auto` to return to full auto-selection.'
            )

        if arg is None:
            return (
                f'{_BACKEND_HEADER}'
                'Backend was not changed: invalid command.\n\n'
                f'Use `proxy-set-backend:<name>` with one of: {available}.'
            )
        result = self.registry.switch(arg, reason='manual command')
        # Pause auto-selector when the user manually pins a backend
        if self.selector is not None and result.kind in ('changed', 'unchanged'):
            self.selector.pin(arg)
        if result.kind == 'unchanged':
            pin_note = ' Auto-selection is paused.' if self.selector is not None else ''
            return (
                f'{_BACKEND_HEADER}'
                f'Backend is already `{result.current}`. No change was made.{pin_note}'
            )
        if result.kind == 'invalid':
            return (
                f'{_BACKEND_HEADER}'
                f'Unknown backend: `{arg}`.\n\nAvailable backends: {available}.'
            )
        if result.kind == 'failed':
            return (
                f'{_BACKEND_HEADER}'
                f'Could not switch to `{arg}`. The active backend remains '
                f'`{result.current}`.\n\nReason: {result.error}'
            )
        auto_note = (
            ' Auto-selection is **paused** — send `proxy-set-backend:auto` to resume.'
            if self.selector is not None else ''
        )
        return (
            f'{_BACKEND_HEADER}'
            f'Switched from `{result.previous}` to `{result.current}`.\n\n'
            f'New requests use `{result.current}`. Requests already in progress '
            f'continue on the backend they started with.{auto_note}'
        )

    def _session_set_backend_markdown(self, arg: str | None, sess_key: str | None) -> str:
        available = ', '.join(f'`{n}`' for n in _backend_names())

        if sess_key is None:
            return (
                f'{_BACKEND_HEADER}'
                'Per-session backend selection requires a client that sends '
                '`metadata.user_id` (e.g. the Claude Code CLI). No session identity '
                'was found in this request.'
            )

        # proxy-set-backend:auto:session → clear this session's override
        if arg == 'auto':
            existed = self.registry.clear_session_backend(sess_key)
            global_name = self.registry.active_name()
            if existed:
                return (
                    f'{_BACKEND_HEADER}'
                    f'Session backend override cleared. This session now follows the '
                    f'global active backend (`{global_name}`).'
                )
            return (
                f'{_BACKEND_HEADER}'
                f'No session backend was set. This session already follows the '
                f'global active backend (`{global_name}`).'
            )

        # proxy-set-backend:subscription:session → lock this session to subscription backends
        if arg == 'subscription':
            result = self.registry.set_session_subscription(sess_key)
            if result.kind == 'unchanged':
                return (
                    f'{_BACKEND_HEADER}'
                    'This session is already locked to subscription backends. No change was made.'
                )
            if result.kind == 'failed':
                return (
                    f'{_BACKEND_HEADER}'
                    f'Could not lock this session to subscription backends.\n\nReason: {result.error}'
                )
            return (
                f'{_BACKEND_HEADER}'
                f'This session is now locked to subscription backends ({"/".join(SUBSCRIPTION_BACKENDS)}). '
                'It will never use bedrock even if the global default is bedrock.\n\n'
                'Send `proxy-set-backend:auto:session` to clear.'
            )

        if arg is None:
            return (
                f'{_BACKEND_HEADER}'
                'Backend was not changed: invalid command.\n\n'
                f'Use `proxy-set-backend:<name>:session` with one of: {available}.'
            )

        result = self.registry.set_session_backend(sess_key, arg)
        if result.kind == 'unchanged':
            return (
                f'{_BACKEND_HEADER}'
                f'This session is already pinned to `{result.current}`. No change was made.'
            )
        if result.kind == 'invalid':
            return (
                f'{_BACKEND_HEADER}'
                f'Unknown backend: `{arg}`.\n\nAvailable backends: {available}.'
            )
        if result.kind == 'failed':
            return (
                f'{_BACKEND_HEADER}'
                f'Could not pin this session to `{arg}`.\n\nReason: {result.error}'
            )
        return (
            f'{_BACKEND_HEADER}'
            f'This session is now pinned to `{result.current}` '
            f'(was `{result.previous}`).\n\n'
            f'Only requests from this session use `{result.current}`; the global '
            f'default (`{self.registry.active_name()}`) is unchanged. '
            f'Send `proxy-set-backend:auto:session` to revert.'
        )

    def _handle_count_tokens(self):
        try:
            self._validate_content_type()
            body = self._read_body()
            payload = self._parse_json(body)
            sess_key = _session_key(payload)
            self._session_prefix = _session_short_id(sess_key) if sess_key else None
            _blks, _chars, sys_hash, _head = _system_fingerprint(payload)
            self._session_hash = sys_hash
            oauth_credential = _oauth_credential(self)
            snapshot = self.registry.snapshot_for_request(
                sess_key,
                prefer_backend=getattr(self, '_prefer_backend', None),
                oauth_credential=oauth_credential,
            )
            logger.info('%s Routing request: operation=count_tokens backend=%s',
                        self._log_tag(), snapshot.name)
            if snapshot.name == 'oauth':
                credentials = {'oauth': snapshot.credentials}
            else:
                credentials = snapshot.backend.parse_credentials(
                    self.headers.get('x-api-key', '')
                )

            result = snapshot.backend.count_tokens(payload, credentials, snapshot.config)
            self._send_json(200, result)

        except AnthropicRequestError as exc:
            self._send_error(exc)
        except _CLIENT_DISCONNECT:
            logger.info('%s client disconnected before response', self._log_tag())
        except Exception as exc:
            logger.exception('%s Count tokens failure: %s', self._log_tag(), exc)
            self._send_json(502, anthropic_error_payload('api_error', 'Upstream request failed'))

    def _handle_cache(self):
        try:
            self._validate_content_type()
            body = self._read_body()
            data = self._parse_json(body)
            key = data.get('key')
            value = data.get('value')
            if not isinstance(key, str) or not key:
                self._send_json(400, anthropic_error_payload('invalid_request_error', 'key must be a non-empty string'))
                return
            if not isinstance(value, str) or not value:
                self._send_json(400, anthropic_error_payload('invalid_request_error', 'value must be a non-empty string'))
                return
            backend = self.registry.instance('bedrock')
            logger.info('%s Routing request: operation=cache backend=bedrock', self._log_tag())
            backend.store_cached_credential(key, value)
            self._send_json(200, {'status': 'ok'})
        except AnthropicRequestError as exc:
            self._send_error(exc)
        except _CLIENT_DISCONNECT:
            logger.info('%s client disconnected before response', self._log_tag())
        except Exception as exc:
            logger.exception('%s Cache store failure: %s', self._log_tag(), exc)
            self._send_json(502, anthropic_error_payload('api_error', 'Cache store failed'))

    # -- helpers --

    def _log_tag(self) -> str:
        """Per-request log prefix: ``[<sess8> <hash> +<elapsed>s]``.

        ``sess8`` is the first 8 chars of the first UUID found in the session
        key (``metadata.user_id``), or ``--------`` when unknown.  ``hash`` is
        the first 8 hex chars of sha256(system), or ``--------`` when there is
        no system prompt.  It is a display-only signal (the per-conversation
        routing key is keyed on the first user message, not this hash — see
        ``_context_key``): the system hash is unstable across a conversation's
        turns, but its turn-1 value still helps an operator spot agent types.
        Elapsed is seconds since ``do_POST`` began.  ``getattr`` guards cover
        paths that emit before stamping (e.g. an early 404) and reused handler
        instances.
        """
        sess = getattr(self, '_session_prefix', None) or '--------'
        shash = getattr(self, '_session_hash', None) or '--------'
        start = getattr(self, '_req_start', None)
        if start is not None:
            return f'[{sess} {shash} +{time.monotonic() - start:.2f}s]'
        return f'[{sess} {shash}]'

    def _send_error(self, exc: AnthropicRequestError) -> None:
        """Send an Anthropic-format error response for a known request error."""
        self._send_json(exc.status_code, anthropic_error_payload(exc.error_type, exc.message))

    def _validate_content_type(self):
        ct = self.headers.get('Content-Type', '')
        if 'application/json' not in ct:
            raise AnthropicRequestError(
                'content-type must be application/json',
                error_type='invalid_request_error',
                status_code=400,
            )

    def _read_body(self) -> bytes:
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length)

    def _parse_json(self, body: bytes) -> dict:
        try:
            return json.loads(body or b'{}')
        except (json.JSONDecodeError, ValueError):
            raise AnthropicRequestError('Malformed JSON body')

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode('utf-8')
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            if status == 429:
                self.send_header('Retry-After', '1')
            self.end_headers()
            self.wfile.write(body)
        except _CLIENT_DISCONNECT:
            # Client closed the socket before we could respond.  Swallow so a
            # dead-socket error response (e.g. a 502 emitted from an exception
            # handler) cannot double-fault up into socketserver.
            logger.info('%s client disconnected before response', self._log_tag())

    def _prime_sse_with_keepalive(self, generator, config, req_start):
        """Prime the first SSE chunk while optionally sending keepalive comments.

        Sends HTTP 200 headers *first*, then primes the generator. This keeps the
        client alive during long upstream TTFB instead of leaving the socket idle.

        If ``config.sse_keepalive_interval`` is a positive number, spawns a
        background thread to call ``next(generator)`` while the main thread
        periodically writes ``": keepalive\\n\\n"`` SSE comment lines.  Comment
        lines are valid SSE that the Anthropic SDK silently discards.

        Returns ``(first_chunk, priming_error)`` where:
        - ``first_chunk``: the primed chunk string, or ``None`` (empty stream).
        - ``priming_error``: ``None`` on success, or an ``AnthropicRequestError``
          raised inside the generator (e.g. a 429 from the upstream).

        Thread safety: the background thread only calls ``next(generator)``
        (reads from upstream); the main thread only writes keepalive comments to
        ``self.wfile``.  The two never touch the same object concurrently.  After
        ``primed.set()`` the background thread has exited; the main thread then
        owns ``generator`` and ``wfile`` exclusively for the remainder of the
        stream.
        """
        interval = getattr(config, 'sse_keepalive_interval', 0)
        if not isinstance(interval, (int, float)) or interval <= 0:
            # Keepalive disabled: block on priming in the main thread.
            try:
                return next(generator), None
            except StopIteration:
                return None, None
            except AnthropicRequestError as exc:
                return None, exc

        # Keepalive enabled: prime in a background thread while the main thread
        # sends keepalive comments to reset the client's read idle-timeout.
        first_chunk = [None]
        priming_error = [None]
        primed = threading.Event()

        def _prime():
            try:
                first_chunk[0] = next(generator)
            except StopIteration:
                pass
            except AnthropicRequestError as exc:
                priming_error[0] = exc
            finally:
                primed.set()

        thread = threading.Thread(target=_prime, daemon=True,
                                  name='sse-prime')
        thread.start()

        # Send keepalive comments until priming completes.
        keepalive_count = 0
        while not primed.wait(timeout=interval):
            try:
                self.wfile.write(b': keepalive\n\n')
                self.wfile.flush()
                keepalive_count += 1
                logger.debug('%s SSE keepalive #%d sent', self._log_tag(), keepalive_count)
            except _CLIENT_DISCONNECT:
                # Client gave up during a keepalive send; wait for priming to
                # finish (it will shortly) so the generator is in a clean state,
                # then bubble the disconnect up via the return value.
                logger.info('%s client disconnected during SSE keepalive (keepalives_sent=%d)',
                            self._log_tag(), keepalive_count)
                primed.wait()  # don't abandon the priming thread
                break
            except Exception as exc:
                logger.warning('%s SSE keepalive write error: %s', self._log_tag(), exc)
                primed.wait()
                break

        if keepalive_count:
            # Log the total count to make slow-upstream episodes diagnosable.
            elapsed = (time.monotonic() - req_start) if req_start is not None else 0.0
            logger.info('%s SSE keepalive complete (keepalives_sent=%d ttfb=%.2fs)',
                        self._log_tag(), keepalive_count, elapsed)

        return first_chunk[0], priming_error[0]

    def _send_sse_error(self, exc: AnthropicRequestError) -> None:
        """Emit a canonical Anthropic SSE error event for a post-header error.

        Used when priming fails (e.g. 429) after HTTP 200 has been committed, so
        the error must be delivered in-band as SSE rather than as an HTTP status.
        The ``sse_event``/``anthropic_error_payload`` helpers produce the same
        ``event: error\\ndata: {...}\\n\\n`` shape that the Anthropic SDK expects
        and the bedrock mapper emits on mid-stream errors.
        """
        frame = sse_event('error', anthropic_error_payload(exc.error_type, exc.message))
        try:
            self.wfile.write(frame.encode('utf-8'))
            self.wfile.flush()
        except _CLIENT_DISCONNECT:
            pass  # Client already gone — nothing to deliver.
        except Exception:
            logger.debug('%s error sending SSE error event', self._log_tag(), exc_info=True)

    def _send_sse(self, generator, *, config=None):
        """Send an SSE response, priming the first upstream chunk after headers.

        Sends HTTP 200 immediately so the client sees a response even during
        long upstream TTFB.  Priming (and optional keepalive) happens via
        ``_prime_sse_with_keepalive`` after ``end_headers``.  An
        ``AnthropicRequestError`` during priming (e.g. a 429) is delivered as an
        in-band SSE error event via ``_send_sse_error``.
        """
        req_start = getattr(self, '_req_start', None)
        chunks_sent = 0
        bytes_sent = 0
        try:
            # Header commit is inside the try: on a cancelled request the very
            # first write (end_headers) is where BrokenPipeError surfaces, and
            # it must be caught here rather than escalating to a 502 attempt.
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()

            # Prime the first chunk (with keepalive if configured).
            first_chunk, priming_error = self._prime_sse_with_keepalive(
                generator, config, req_start)
            if priming_error is not None:
                logger.warning('%s SSE priming error (post-header): %s', self._log_tag(), priming_error)
                self._send_sse_error(priming_error)
                return

            # TTFB: elapsed at the point the first chunk is ready to send.
            ttfb = (time.monotonic() - req_start) if req_start is not None else None
            ttfb_str = f'{ttfb:.2f}s' if ttfb is not None else 'unknown'

            if first_chunk is not None:
                data = first_chunk.encode('utf-8')
                self.wfile.write(data)
                self.wfile.flush()
                chunks_sent += 1
                bytes_sent += len(data)
            for chunk in generator:
                data = chunk.encode('utf-8')
                self.wfile.write(data)
                self.wfile.flush()
                chunks_sent += 1
                bytes_sent += len(data)
        except _CLIENT_DISCONNECT:
            # Client cancelled the request and closed the socket — benign.
            # chunks_sent=0 means the client disconnected before the first
            # real data byte (during keepalive or before priming finished).
            ttfb = (time.monotonic() - req_start) if req_start is not None else None
            ttfb_str = f'{ttfb:.2f}s' if ttfb is not None else 'unknown'
            logger.info('%s client disconnected mid-stream (ttfb=%s chunks_sent=%d bytes_sent=%d)',
                        self._log_tag(), ttfb_str, chunks_sent, bytes_sent)
        except AnthropicRequestError as exc:
            # Headers are committed, so relay the upstream failure in-band.
            ttfb = (time.monotonic() - req_start) if req_start is not None else None
            ttfb_str = f'{ttfb:.2f}s' if ttfb is not None else 'unknown'
            logger.warning('%s SSE stream request error after headers committed '
                           '(ttfb=%s chunks_sent=%d bytes_sent=%d): %s',
                           self._log_tag(), ttfb_str, chunks_sent, bytes_sent, exc)
            self._send_sse_error(exc)
        except Exception as exc:
            # Headers + data already sent; cannot send a new HTTP status.
            # Log the error and close the stream silently so the client
            # receives a clean EOF rather than a garbled second response.
            ttfb = (time.monotonic() - req_start) if req_start is not None else None
            ttfb_str = f'{ttfb:.2f}s' if ttfb is not None else 'unknown'
            logger.error('%s SSE stream error after headers committed '
                         '(ttfb=%s chunks_sent=%d bytes_sent=%d): %s',
                         self._log_tag(), ttfb_str, chunks_sent, bytes_sent, exc)
        finally:
            # Close the generator so the backend's `finally: conn.close()` and
            # the stats wrapper's `finally` run synchronously rather than waiting
            # for refcount finalization.  Guarded: non-generator iterables (e.g.
            # a plain list_iterator) have no `.close()`.
            close = getattr(generator, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.debug('%s error closing SSE generator',
                                 self._log_tag(), exc_info=True)

    def _serve_ui_file(self, path: str):
        """Serve static files from anthproxy/ui/dist/, fallback to index.html for SPA."""
        import mimetypes
        from pathlib import Path

        ui_dist = (Path(__file__).parent / 'ui' / 'dist').resolve()

        # Strip /ui prefix and resolve the file
        relative = path[4:].lstrip('/')  # Remove '/ui' prefix
        if not relative or relative == '':
            relative = 'index.html'

        file_path = (ui_dist / relative).resolve()

        # Security: ensure resolved path is within ui/dist/
        try:
            file_path.relative_to(ui_dist)
        except ValueError:
            self._send_json(404, anthropic_error_payload('not_found_error', 'Not found'))
            return

        # If it's a directory or doesn't exist, try index.html
        if not file_path.is_file():
            index_path = (ui_dist / 'index.html').resolve()
            if index_path.is_file():
                file_path = index_path
            else:
                self._send_json(404, anthropic_error_payload('not_found_error', 'Not found'))
                return

        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type is None:
                mime_type = 'application/octet-stream'

            with open(file_path, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-cache' if file_path.name == 'index.html' else 'public, max-age=31536000')
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            logger.exception('Error serving UI file %s: %s', file_path, exc)
            self._send_json(500, anthropic_error_payload('api_error', 'Failed to serve UI'))

    def log_message(self, format, *args):
        logger.info('%s ' + format, self._log_tag(), *args)
