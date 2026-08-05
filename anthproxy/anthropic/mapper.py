import json
import logging

from .. import model_config as _model_config
from .._shared.model_alias import resolve_alias as _resolve_alias
from ..mapper import strip_codex_thinking_blocks as _strip_codex_thinking_blocks

logger = logging.getLogger(__name__)

REQUIRED_BETAS = ('claude-code-20250219', 'oauth-2025-04-20')
_CLAUDE_CLI_VERSION = '2.1.88'
_CC_SYSTEM_PREFIX = 'You are Claude Code, Anthropic\'s official CLI for Claude.'
MAX_CACHE_CONTROL_BLOCKS = 4
def resolve_model(model: str) -> str:
    return _resolve_alias(model, _model_config.model_aliases('anthropic'))

_resolve_model = resolve_model


def _supports_effort(model_id: str) -> bool:
    """Return True iff the resolved Anthropic model ID accepts output_config.effort.

    Haiku models reject this parameter with HTTP 400; Sonnet, Opus, and Fable
    all accept it.  The substring check is robust for dated IDs such as
    ``claude-haiku-4-5-20251001`` and cannot false-positive on other tiers.
    """
    return 'haiku' not in model_id.lower()


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Return True iff the resolved Anthropic model ID accepts thinking.type='adaptive'.

    Haiku rejects adaptive thinking with HTTP 400 ("adaptive thinking is not
    supported on this model") but still supports manual extended thinking
    (``type='enabled'`` with ``budget_tokens``), so only the adaptive shape is
    stripped — see ``_build_body``.  Sonnet, Opus, and Fable accept adaptive.
    """
    return 'haiku' not in model_id.lower()


def _supports_disabled_thinking(model_id: str) -> bool:
    """Return True iff the resolved model accepts thinking.type='disabled'.

    Fable rejects explicitly-disabled thinking with HTTP 400 ("`thinking.type.disabled`
    is not supported for this model") — it only accepts an absent thinking field
    (defaults to adaptive) or manual `type='enabled'`.  Haiku, Sonnet, and Opus all
    accept `type='disabled'`.  When unsupported, ``_build_body`` drops the thinking
    field so the model falls back to its adaptive default.
    """
    return 'fable' not in model_id.lower()


_SAMPLING_CONTROL_KEYS = frozenset({'temperature', 'top_p', 'top_k'})


def _supports_sampling_controls(model_id: str) -> bool:
    """Return True iff the resolved model accepts temperature/top_p/top_k.

    Opus 4.7+, Opus 4.8, and Fable use fixed sampling and reject any non-default
    sampling parameter with HTTP 400 ("`temperature` is deprecated for this
    model").  Older Opus, Sonnet, and Haiku still accept them — those tiers only
    forbid sending ``temperature`` and ``top_p`` together, which is the caller's
    responsibility, not a per-model strip.
    """
    model = model_id.lower()
    return not ('opus-4-7' in model or 'opus-4-8' in model or 'opus-5' in model or 'fable' in model)


# Long-context (1m) beta tokens, e.g. 'context-1m-2025-08-07'.  Matched by
# prefix so the gate survives date-stamp revisions of the token.
_LONG_CONTEXT_BETA_PREFIX = 'context-1m'


def _supports_long_context(model_id: str) -> bool:
    """Return True iff the resolved Anthropic model ID may carry the 1m context beta.

    Long context (the ``context-1m*`` beta) is only granted to Opus on the
    subscription: Haiku has a 200k window and rejects it outright (HTTP 400),
    while Sonnet returns HTTP 429 "Usage credits are required for long context
    requests."  Keyed on the resolved model so model-tier routing to a non-opus
    tier never forwards an unusable long-context beta.  The substring check is
    robust for dated IDs such as ``claude-opus-4-8``.
    """
    return 'opus' in model_id.lower()


# Betas that require thinking to be enabled or adaptive; stripped when thinking
# is absent or was stripped by model-specific sanitization.
_THINKING_REQUIRED_BETAS = frozenset({'clear_thinking_20251015'})


def _thinking_active(payload: dict, resolved_model: str) -> bool:
    """Return True iff thinking will be active in the outbound request.

    Mirrors the stripping logic in ``_build_body``: adaptive thinking is removed
    for Haiku, so a payload with ``type='adaptive'`` on a Haiku model counts as
    inactive.
    """
    thinking = payload.get('thinking')
    if not isinstance(thinking, dict):
        return False
    t_type = thinking.get('type')
    if t_type == 'enabled':
        return True
    if t_type == 'adaptive':
        return _supports_adaptive_thinking(resolved_model)
    return False


def merge_betas(payload: dict) -> str:
    raw_model = payload.get('model') or ''
    resolved_model = resolve_model(raw_model) if raw_model else ''
    active = _thinking_active(payload, resolved_model)
    long_context_ok = _supports_long_context(resolved_model)
    betas: list[str] = []
    seen: set[str] = set()
    for beta in REQUIRED_BETAS:
        if beta not in seen:
            betas.append(beta)
            seen.add(beta)
    for beta in payload.get('_anthropic_beta') or []:
        if not active and beta in _THINKING_REQUIRED_BETAS:
            logger.debug(
                'Dropped thinking beta %s: thinking not active for model %s',
                beta, resolved_model,
            )
            continue
        if not long_context_ok and beta.startswith(_LONG_CONTEXT_BETA_PREFIX):
            logger.debug(
                'Dropped long-context beta %s: 1m context not supported for model %s',
                beta, resolved_model,
            )
            continue
        if beta not in seen:
            betas.append(beta)
            seen.add(beta)
    return ','.join(betas)


# Context-management edit/strategy types that require thinking to be enabled or
# adaptive.  When thinking is stripped (Haiku), the matching body strategy must
# also be removed or Anthropic rejects with HTTP 400 ("`clear_thinking_...`
# strategy requires `thinking` to be enabled or adaptive").  Prefix-matched to
# survive date-stamp revisions of the strategy name.
_THINKING_REQUIRED_EDIT_PREFIX = 'clear_thinking'


def _strip_thinking_edits(body: dict) -> None:
    """Drop ``clear_thinking*`` edits from ``context_management`` in place.

    Mutates the shallow-copied ``body`` only (never the caller's nested objects):
    rebuilds ``context_management`` / its ``edits`` list rather than editing
    them in place.  No-op when the structure is absent or unexpected.
    """
    cm = body.get('context_management')
    if not isinstance(cm, dict):
        return
    edits = cm.get('edits')
    if not isinstance(edits, list):
        return
    kept = [
        e for e in edits
        if not (isinstance(e, dict)
                and isinstance(e.get('type'), str)
                and e['type'].startswith(_THINKING_REQUIRED_EDIT_PREFIX))
    ]
    if len(kept) == len(edits):
        return
    if kept:
        new_cm = {k: v for k, v in cm.items() if k != 'edits'}
        new_cm['edits'] = kept
        body['context_management'] = new_cm
    else:
        # ``edits`` was the only content worth keeping; drop the whole block if
        # nothing else remains, otherwise keep the siblings without ``edits``.
        siblings = {k: v for k, v in cm.items() if k != 'edits'}
        if siblings:
            body['context_management'] = siblings
        else:
            body.pop('context_management', None)
    logger.debug(
        'Dropped clear_thinking context-management edit(s) for model %s',
        body.get('model'),
    )


def _stringify_content(content) -> str:
    """Flatten an Anthropic message ``content`` to plain text.

    A string is returned as-is; a list of blocks is reduced to its ``type=='text'``
    block texts joined by newlines.  Used to fold inline ``role:'system'`` turns
    into the top-level system field.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            blk['text'] for blk in content
            if isinstance(blk, dict) and blk.get('type') == 'text'
            and isinstance(blk.get('text'), str)
        ]
        return '\n'.join(parts)
    return ''


def _with_cache_breakpoint(block: dict) -> dict:
    if 'cache_control' in block:
        return block
    return {**block, 'cache_control': {'type': 'ephemeral'}}


def _ensure_cc_system_prefix(system) -> list[dict]:
    prefix_block = {
        'type': 'text',
        'text': _CC_SYSTEM_PREFIX,
        'cache_control': {'type': 'ephemeral'},
    }

    if system is None:
        return [prefix_block]
    if isinstance(system, str):
        if system.strip() == '':
            return [prefix_block]
        return [prefix_block, _with_cache_breakpoint({'type': 'text', 'text': system})]
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get('type') == 'text':
                if block.get('text', '').strip().startswith(_CC_SYSTEM_PREFIX):
                    return system
                break
        if not system:
            return [prefix_block]
        return [prefix_block] + list(system[:-1]) + [_with_cache_breakpoint(system[-1])]
    return [prefix_block]


def _is_cc_caller(system) -> bool:
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block.get('text', '').strip().startswith(_CC_SYSTEM_PREFIX)
            break
    elif isinstance(system, str):
        return system.strip().startswith(_CC_SYSTEM_PREFIX)
    return False


def _enforce_cache_control_limit(body: dict) -> dict:
    positions: list = []

    sys_blocks = body.get('system')
    if isinstance(sys_blocks, list):
        for i, blk in enumerate(sys_blocks):
            if isinstance(blk, dict) and 'cache_control' in blk:
                positions.append(('system', i))

    tools = body.get('tools')
    if isinstance(tools, list):
        for i, tool in enumerate(tools):
            if isinstance(tool, dict) and 'cache_control' in tool:
                positions.append(('tools', i))

    messages = body.get('messages')
    if isinstance(messages, list):
        for mi, msg in enumerate(messages):
            content = msg.get('content') if isinstance(msg, dict) else None
            if isinstance(content, list):
                for ci, blk in enumerate(content):
                    if isinstance(blk, dict) and 'cache_control' in blk:
                        positions.append(('messages', mi, ci))

    excess = len(positions) - MAX_CACHE_CONTROL_BLOCKS
    if excess <= 0:
        return body

    logger.debug(
        'Stripping %d excess cache_control block(s) (found %d, limit %d)',
        excess, len(positions), MAX_CACHE_CONTROL_BLOCKS,
    )
    body = dict(body)
    for pos in positions[:excess]:
        tag = pos[0]
        if tag == 'system':
            lst = list(body['system'])
            lst[pos[1]] = {k: v for k, v in lst[pos[1]].items() if k != 'cache_control'}
            body['system'] = lst
        elif tag == 'tools':
            lst = list(body['tools'])
            lst[pos[1]] = {k: v for k, v in lst[pos[1]].items() if k != 'cache_control'}
            body['tools'] = lst
        elif tag == 'messages':
            msgs = list(body['messages'])
            msg = dict(msgs[pos[1]])
            content = list(msg['content'])
            content[pos[2]] = {k: v for k, v in content[pos[2]].items() if k != 'cache_control'}
            msg['content'] = content
            msgs[pos[1]] = msg
            body['messages'] = msgs
    return body


def _cache_block_ttl(block) -> str:
    """Return '1h' if the block carries an explicit 1h breakpoint, else '5m'.

    An ``ephemeral`` cache_control without an explicit ``ttl`` defaults to 5m on
    the Anthropic API.
    """
    cc = block.get('cache_control') if isinstance(block, dict) else None
    if isinstance(cc, dict) and cc.get('ttl') == '1h':
        return '1h'
    return '5m'


def _block_with_1h(block: dict) -> dict:
    cc = block.get('cache_control')
    new_cc = dict(cc) if isinstance(cc, dict) else {'type': 'ephemeral'}
    new_cc['ttl'] = '1h'
    return {**block, 'cache_control': new_cc}


def _cache_control_positions(body: dict) -> list:
    """Collect cache_control block positions in API processing order.

    Order is ``tools`` -> ``system`` -> ``messages`` to match how the Anthropic
    API evaluates the cross-section ttl ordering rule.
    """
    positions: list = []

    tools = body.get('tools')
    if isinstance(tools, list):
        for i, tool in enumerate(tools):
            if isinstance(tool, dict) and 'cache_control' in tool:
                positions.append(('tools', i))

    sys_blocks = body.get('system')
    if isinstance(sys_blocks, list):
        for i, blk in enumerate(sys_blocks):
            if isinstance(blk, dict) and 'cache_control' in blk:
                positions.append(('system', i))

    messages = body.get('messages')
    if isinstance(messages, list):
        for mi, msg in enumerate(messages):
            content = msg.get('content') if isinstance(msg, dict) else None
            if isinstance(content, list):
                for ci, blk in enumerate(content):
                    if isinstance(blk, dict) and 'cache_control' in blk:
                        positions.append(('messages', mi, ci))

    return positions


def _block_at(body: dict, pos: tuple):
    if pos[0] == 'tools':
        return body['tools'][pos[1]]
    if pos[0] == 'system':
        return body['system'][pos[1]]
    return body['messages'][pos[1]]['content'][pos[2]]


def _promote_block_ttl(body: dict, pos: tuple) -> None:
    """Copy-on-write the block at ``pos`` so its breakpoint becomes ttl='1h'."""
    if pos[0] == 'tools':
        lst = list(body['tools'])
        lst[pos[1]] = _block_with_1h(lst[pos[1]])
        body['tools'] = lst
    elif pos[0] == 'system':
        lst = list(body['system'])
        lst[pos[1]] = _block_with_1h(lst[pos[1]])
        body['system'] = lst
    else:
        msgs = list(body['messages'])
        msg = dict(msgs[pos[1]])
        content = list(msg['content'])
        content[pos[2]] = _block_with_1h(content[pos[2]])
        msg['content'] = content
        msgs[pos[1]] = msg
        body['messages'] = msgs


def _normalize_cache_ttl_ordering(body: dict) -> dict:
    """Ensure no 5m cache breakpoint precedes a 1h one across tools/system/messages.

    The Anthropic API rejects a request where a ttl='1h' cache_control block
    comes after a ttl='5m' block in the concatenated tools -> system -> messages
    sequence. The proxy injects default (5m) breakpoints — e.g. the Claude Code
    system prefix — which can land ahead of client-supplied 1h breakpoints. We
    promote every offending earlier 5m breakpoint to 1h, preserving all caching
    while restoring a valid ordering.
    """
    positions = _cache_control_positions(body)
    ttls = [_cache_block_ttl(_block_at(body, pos)) for pos in positions]

    last_1h = -1
    for idx, ttl in enumerate(ttls):
        if ttl == '1h':
            last_1h = idx
    if last_1h < 0:
        return body

    to_promote = [positions[idx] for idx in range(last_1h) if ttls[idx] == '5m']
    if not to_promote:
        return body

    body = dict(body)
    for pos in to_promote:
        _promote_block_ttl(body, pos)
    return body


_INTERNAL_KEYS = frozenset({'_anthropic_beta', '_anthproxy_internal_classifier'})


def build_body(payload: dict) -> bytes:
    body = {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}
    body['model'] = resolve_model(payload.get('model', ''))

    # Strip output_config.effort for models that reject it (Haiku).  The
    # body dict is a shallow copy, so we must not mutate the nested object
    # in place — build a new one instead.
    if not _supports_effort(body['model']):
        oc = body.get('output_config')
        if isinstance(oc, dict) and 'effort' in oc:
            oc = {k: v for k, v in oc.items() if k != 'effort'}
            if oc:
                body['output_config'] = oc
            else:
                body.pop('output_config', None)
            logger.debug(
                'Dropped unsupported output_config.effort for model %s', body['model'],
            )

    # Strip adaptive thinking for models that reject it (Haiku).  Manual
    # extended thinking (type='enabled') and disabled thinking are preserved.
    if not _supports_adaptive_thinking(body['model']):
        thinking = body.get('thinking')
        if isinstance(thinking, dict) and thinking.get('type') == 'adaptive':
            body.pop('thinking', None)
            logger.debug(
                'Dropped unsupported adaptive thinking for model %s', body['model'],
            )

    # Strip explicitly-disabled thinking for models that reject it (Fable), which
    # returns HTTP 400 for thinking.type='disabled' and only accepts an absent
    # thinking field (defaults to adaptive) or manual type='enabled'.  Dropping the
    # field lets the model fall back to its adaptive default.
    if not _supports_disabled_thinking(body['model']):
        thinking = body.get('thinking')
        if isinstance(thinking, dict) and thinking.get('type') == 'disabled':
            body.pop('thinking', None)
            logger.debug(
                'Dropped unsupported disabled thinking for model %s', body['model'],
            )

    # Strip the clear_thinking context-management strategy whenever thinking will
    # not be active in the outbound request (e.g. adaptive thinking removed for
    # Haiku above).  Pairs with the clear_thinking *beta* strip in _merge_betas —
    # both keyed on _thinking_active — so the body strategy can never outlive the
    # thinking it depends on (Anthropic else rejects with HTTP 400 "clear_thinking
    # ... strategy requires thinking to be enabled or adaptive").
    if not _thinking_active(payload, body['model']):
        _strip_thinking_edits(body)

    # Strip sampling controls for fixed-sampling models (Opus 4.7+/4.8, Fable),
    # which reject any non-default temperature/top_p/top_k with HTTP 400.  Drop
    # on any presence (including None) — omission is safer than serializing null.
    if not _supports_sampling_controls(body['model']):
        dropped = [k for k in _SAMPLING_CONTROL_KEYS if k in body]
        for k in dropped:
            body.pop(k, None)
        if dropped:
            logger.debug(
                'Dropped unsupported sampling controls for model %s: %s',
                body['model'], ','.join(sorted(dropped)),
            )

    raw_system = payload.get('system')
    cc_caller = _is_cc_caller(raw_system)

    if 'system' in payload:
        body['system'] = _ensure_cc_system_prefix(raw_system)
    else:
        body['system'] = _ensure_cc_system_prefix(None)

    # Strip thinking blocks whose signature the Codex mapper minted (codexenc:
    # prefix).  On a Codex→Anthropic backend switch Claude Code resends history
    # with those synthetic signatures; Anthropic validates them cryptographically
    # and rejects them with HTTP 400 "Invalid signature in thinking block".
    messages = body.get('messages')
    if isinstance(messages, list):
        stripped = _strip_codex_thinking_blocks(messages)
        if stripped is not messages:
            body['messages'] = stripped
            messages = stripped
            logger.debug('Stripped Codex synthetic thinking block(s) from messages')

    # Fold inline role:'system' messages into the top-level system field.  Some
    # models (e.g. Sonnet 4.6) reject a 'system' role in messages[] with HTTP 400
    # ("role 'system' is not supported on this model"); the top-level system
    # field is universally accepted.  Mirrors the codex/bedrock mappers.  Folded
    # text is appended after the CC prefix + client system so the required Claude
    # Code system block stays first.
    messages = body.get('messages')
    if isinstance(messages, list) and any(
        isinstance(m, dict) and m.get('role') == 'system' for m in messages
    ):
        folded = [
            _stringify_content(m.get('content'))
            for m in messages
            if isinstance(m, dict) and m.get('role') == 'system'
        ]
        body['messages'] = [
            m for m in messages
            if not (isinstance(m, dict) and m.get('role') == 'system')
        ]
        extra = [{'type': 'text', 'text': t} for t in folded if t]
        if extra:
            body['system'] = list(body['system']) + extra
        logger.debug(
            'Folded %d inline system message(s) into top-level system', len(folded),
        )

    if not cc_caller:
        tools = body.get('tools')
        if tools and isinstance(tools, list):
            last = tools[-1]
            if isinstance(last, dict) and 'cache_control' not in last:
                body['tools'] = list(tools[:-1]) + [_with_cache_breakpoint(last)]

    body = _enforce_cache_control_limit(body)
    body = _normalize_cache_ttl_ordering(body)
    return json.dumps(body).encode('utf-8')


_merge_betas = merge_betas
_build_body = build_body
