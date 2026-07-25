"""OpenRouter mapper for anthproxy.

Builds outbound request bodies and headers for the OpenRouter Anthropic-compatible
Messages API endpoint (https://openrouter.ai/api/v1/messages).
"""

import json

from .. import model_config
from ..mapper import AnthropicRequestError
from .._shared.model_alias import CONTEXT_SUFFIXES

_HOST = 'openrouter.ai'
_MESSAGES_PATH = '/api/v1/messages'
_INTERNAL_KEYS = frozenset({'_anthropic_beta', '_anthproxy_internal_classifier'})


def _resolve_model(model: str) -> str:
    """Resolve a model alias or bare ID to the OpenRouter model slug.

    Resolution order:
    1. Empty model → raise ``AnthropicRequestError(400)``.
    2. Direct match in the openrouter alias dict.
    3. Strip each ``CONTEXT_SUFFIXES`` suffix (``:1m``, ``[1m]``) and retry.
    4. Fall back to the ``"default"`` alias key when present.
    5. Pass *model* through verbatim (no ``"default"`` configured).

    When ``"default"`` is configured, any model not matched by an explicit alias
    — including native OpenRouter slugs — is redirected to the default. This
    matches the ``local`` mapper semantics: ``"default"`` is a catch-all for
    operators who want a single target regardless of what the client requests.
    """
    if not model:
        raise AnthropicRequestError('model is required', status_code=400)

    aliases = model_config.model_aliases('openrouter')

    if model in aliases:
        return aliases[model]

    for suffix in CONTEXT_SUFFIXES:
        if model.endswith(suffix):
            base = model[: -len(suffix)]
            if base in aliases:
                return aliases[base]

    return aliases.get('default', model)


def _strip_thinking_blocks(messages: list) -> list:
    """Remove all thinking blocks from assistant messages in history.

    OpenRouter routes to upstream providers whose thinking-block signatures are
    opaque to each other and to the user's own subscription.  Stripping them on
    the request side means OpenRouter's upstream never receives a foreign
    signature (whether ``or:``-tagged from a prior OpenRouter turn or a genuine
    Anthropic signature from a prior subscription turn).  Reasoning continuity
    across turns is not guaranteed across OpenRouter's providers anyway.
    """
    if not isinstance(messages, list):
        return messages
    result = []
    changed = False
    for msg in messages:
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            result.append(msg)
            continue
        content = msg.get('content')
        if not isinstance(content, list):
            result.append(msg)
            continue
        filtered = [
            b for b in content
            if not (isinstance(b, dict) and b.get('type') in ('thinking', 'redacted_thinking'))
        ]
        if len(filtered) == len(content):
            result.append(msg)
            continue
        changed = True
        result.append({**msg, 'content': filtered or [{'type': 'text', 'text': ''}]})
    return result if changed else messages


def _build_body(payload: dict) -> bytes:
    """Strip internal sentinel keys, substitute the resolved model ID, serialize."""
    body = {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}
    body['model'] = _resolve_model(payload.get('model', ''))
    messages = body.get('messages')
    if isinstance(messages, list):
        stripped = _strip_thinking_blocks(messages)
        if stripped is not messages:
            body['messages'] = stripped
    return json.dumps(body).encode('utf-8')


def _request_headers(api_key: str, *, stream: bool) -> dict:
    """Build the HTTP request headers for an OpenRouter API call."""
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream' if stream else 'application/json',
        'HTTP-Referer': 'https://github.com/nj4x/anthproxy',
        'X-Title': 'anthproxy',
    }
