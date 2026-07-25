"""Generic Anthropic Messages API protocol primitives.

These helpers are backend-agnostic and used by all backend implementations.
Bedrock-specific mapping lives in ``anthproxy/bedrock/mapper.py``.
"""

import json


def anthropic_error_payload(type_, message):
    return {
        'type': 'error',
        'error': {
            'type': type_,
            'message': message,
        }
    }


class AnthropicRequestError(Exception):
    def __init__(self, message, error_type='invalid_request_error', status_code=400):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


def _count_chars(obj):
    """Recursively sum all string lengths in a nested structure."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, list):
        return sum(_count_chars(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_count_chars(v) for v in obj.values())
    return 0


# Tool-use scaffolding overhead, in tokens.  When a request declares tools the
# provider injects a sizeable tool-use system preamble plus per-tool framing that
# the ~4-chars/token text heuristic alone misses badly (measured ~5x undercount
# without these).  The constants were calibrated against the Anthropic
# /v1/messages/count_tokens endpoint: a one-time base of ~500 tokens when any
# tools are present, plus ~60 tokens of framing per tool, on top of each tool's
# serialized text.  With these, tool-bearing requests track the official count to
# within ~5% (vs. 4-5x off before).
_TOOL_USE_BASE_OVERHEAD = 500
_TOOL_FRAMING_OVERHEAD = 60


def estimate_input_tokens(payload):
    """Rough token estimate from the Anthropic request payload.

    Uses a ~4-chars/token heuristic over the meaningful content — messages,
    system prompt, and the full serialized tool definitions — plus a calibrated
    tool-use scaffolding overhead (see ``_TOOL_USE_BASE_OVERHEAD`` /
    ``_TOOL_FRAMING_OVERHEAD``) when tools are present.

    Accuracy: tracks the official count_tokens endpoint to within ~10% for
    natural-language prose and tool-bearing requests.  It systematically
    *undercounts* dense code (~1.5-1.8x: code tokenizes to ~2.2 chars/token, not
    4) and non-text blocks such as images count ~0 — both are intrinsic to a
    char-based heuristic and not recoverable without a real tokenizer.  Callers
    that gate on a window threshold should leave headroom for this skew.
    """
    total_chars = 0

    for msg in payload.get('messages') or []:
        total_chars += _count_chars(msg.get('content') or '')

    system = payload.get('system')
    if system:
        total_chars += _count_chars(system)

    tools = payload.get('tools') or []
    for tool in tools:
        # Count the whole tool structure (name, description, input_schema, and
        # any other fields) rather than three hand-picked keys.
        total_chars += _count_chars(tool)

    tokens = total_chars // 4
    if tools:
        tokens += _TOOL_USE_BASE_OVERHEAD + _TOOL_FRAMING_OVERHEAD * len(tools)

    return max(1, tokens)


# Sentinel prefix used by the Codex mapper to smuggle ``encrypted_content``
# (opaque reasoning state) back into the Responses API on the next turn.
# Non-Codex backends (Anthropic, Bedrock) must strip thinking blocks that
# carry this prefix before forwarding to their providers, which validate
# signatures cryptographically and reject foreign ones with HTTP 400.
CODEX_REASONING_SIG_PREFIX = 'codexenc:'

# Sentinel prefix the OpenRouter backend stamps onto thinking-block signatures
# in its responses.  OpenRouter routes to upstream providers (often a different
# Anthropic account, or a non-Anthropic reasoning model) whose signatures are
# foreign to the user's own Anthropic subscription.  On an OpenRouter→Anthropic
# or OpenRouter→Bedrock switch those providers reject the foreign signatures
# with HTTP 400 "Invalid signature in thinking block".  Stripping tagged blocks
# in the target mapper prevents the 400.
OPENROUTER_REASONING_SIG_PREFIX = 'or:'

# All signature prefixes that mark a thinking block as foreign to the Anthropic
# and Bedrock providers.  Used by ``strip_codex_thinking_blocks``.
_FOREIGN_SIG_PREFIXES = (CODEX_REASONING_SIG_PREFIX, OPENROUTER_REASONING_SIG_PREFIX)


_ALL_THINKING_TYPES = frozenset({'thinking', 'redacted_thinking'})


def strip_all_thinking_blocks(messages):
    """Remove ALL ``thinking`` and ``redacted_thinking`` blocks from history.

    Used as a recovery fallback when Anthropic rejects a thinking-block with
    HTTP 400 after a model-tier switch (opus signatures/data are invalid for
    sonnet/haiku and vice versa).  Both ``thinking`` (rejected as "Invalid
    `signature`") and ``redacted_thinking`` (rejected as "Invalid `data`")
    are model-specific and must be stripped together.  Conversation alternation
    is maintained by inserting an empty text block when all content blocks in
    a message were thinking blocks.

    Returns the original list unchanged when nothing was stripped.
    """
    if not messages:
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
            if not (isinstance(b, dict) and b.get('type') in _ALL_THINKING_TYPES)
        ]
        if len(filtered) == len(content):
            result.append(msg)
            continue
        changed = True
        result.append({**msg, 'content': filtered or [{'type': 'text', 'text': ''}]})
    return result if changed else messages


def strip_codex_thinking_blocks(messages):
    """Remove thinking blocks with foreign-prefixed signatures from message history.

    On a Codex→Anthropic, Codex→Bedrock, OpenRouter→Anthropic, or
    OpenRouter→Bedrock backend switch, Claude Code resends history containing
    thinking blocks whose ``signature`` holds a ``codexenc:`` or ``or:``
    sentinel.  Those providers validate signatures cryptographically and reject
    foreign ones (HTTP 400 "Invalid signature in thinking block").  Strip them
    so the conversation continues; foreign reasoning state is opaque to the
    target provider anyway.

    Returns the original list unchanged when nothing is stripped.
    """
    if not messages:
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
            if not (
                isinstance(b, dict)
                and b.get('type') == 'thinking'
                and isinstance(b.get('signature', ''), str)
                and b['signature'].startswith(_FOREIGN_SIG_PREFIXES)
            )
        ]
        if len(filtered) == len(content):
            result.append(msg)
            continue
        changed = True
        # Degenerate edge case: all blocks were synthetic thinking — insert a
        # minimal text block to preserve the assistant turn so conversation
        # alternation (user→assistant→user) stays valid.
        result.append({**msg, 'content': filtered or [{'type': 'text', 'text': ''}]})
    return result if changed else messages


def sse_event(event_type, payload):
    return 'event: %s\ndata: %s\n\n' % (event_type, json.dumps(payload))


def emit_block_start(index, content_block):
    return sse_event('content_block_start', {
        'type': 'content_block_start',
        'index': index,
        'content_block': content_block,
    })


def emit_block_stop(index):
    return sse_event('content_block_stop', {
        'type': 'content_block_stop',
        'index': index,
    })


def emit_text_delta(index, text):
    return sse_event('content_block_delta', {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'text_delta', 'text': text},
    })


def emit_thinking_delta(index, thinking):
    return sse_event('content_block_delta', {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'thinking_delta', 'thinking': thinking},
    })


def emit_signature_delta(index, signature):
    return sse_event('content_block_delta', {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'signature_delta', 'signature': signature},
    })


def emit_input_json_delta(index, partial_json):
    return sse_event('content_block_delta', {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'input_json_delta', 'partial_json': partial_json},
    })


def emit_message_delta_stop(stop_reason, usage):
    return (
        sse_event('message_delta', {
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
            'usage': usage,
        }),
        sse_event('message_stop', {'type': 'message_stop'}),
    )
