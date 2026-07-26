import base64
import binascii
import hashlib
import json
import logging
import uuid

from .. import model_config as _model_config
from .._shared.http_util import read_sse_lines as _read_sse_lines
from .._shared.model_alias import resolve_alias as _resolve_alias
from ..mapper import (
    CODEX_REASONING_SIG_PREFIX as _REASONING_SIG_PREFIX,
    AnthropicRequestError,
    emit_block_start,
    emit_block_stop,
    emit_input_json_delta,
    emit_message_delta_stop,
    emit_signature_delta,
    emit_text_delta,
    emit_thinking_delta,
    sse_event,
)

logger = logging.getLogger(__name__)


def _resolve_model(model: str) -> str:
    return _resolve_alias(model, _model_config.model_aliases('codex'), prefix_match=True)


def _encode_reasoning_signature(item_id: str, encrypted_content: str) -> str:
    """Pack a reasoning item's id + encrypted_content into a signature string."""
    raw = json.dumps({'id': item_id, 'enc': encrypted_content}).encode('utf-8')
    return _REASONING_SIG_PREFIX + base64.b64encode(raw).decode('ascii')


def _decode_reasoning_signature(signature: str) -> dict | None:
    """Reverse ``_encode_reasoning_signature``; return None if not one of ours."""
    if not isinstance(signature, str) or not signature.startswith(_REASONING_SIG_PREFIX):
        return None
    encoded = signature[len(_REASONING_SIG_PREFIX):]
    try:
        raw = base64.b64decode(encoded.encode('ascii'), validate=True)
        obj = json.loads(raw.decode('utf-8'))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict) or not obj.get('enc'):
        return None
    return obj


def _effort_from_budget(budget_tokens: int) -> str:
    if budget_tokens <= 4096:
        return 'low'
    if budget_tokens <= 16384:
        return 'medium'
    return 'high'


def _effort_from_output_config(payload: dict) -> str | None:
    output_config = payload.get('output_config')
    if not isinstance(output_config, dict):
        return None
    effort = output_config.get('effort')
    if effort in ('low', 'medium', 'high', 'xhigh'):
        return effort
    if effort == 'max':
        return 'xhigh'
    return None


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                parts.append(block.get('text', ''))
        return '\n'.join(parts)
    return str(content) if content else ''


def _reasoning_items_from_content(content: list) -> list[dict]:
    """Rebuild Codex ``reasoning`` input items from thinking-block signatures.

    Only thinking blocks whose ``signature`` we minted (sentinel-prefixed) yield
    an item; genuine Anthropic signatures and summary-only thinking are skipped,
    so reasoning state survives a Codex→Codex round-trip without leaking foreign
    signatures back to the Responses API.
    """
    items: list[dict] = []
    for block in content:
        if not isinstance(block, dict) or block.get('type') != 'thinking':
            continue
        decoded = _decode_reasoning_signature(block.get('signature', ''))
        if decoded is None:
            continue
        item: dict = {
            'type': 'reasoning',
            'encrypted_content': decoded['enc'],
            'summary': [],
        }
        if decoded.get('id'):
            item['id'] = decoded['id']
        items.append(item)
    return items


def _convert_message_to_input_item(
    msg: dict,
    include_reasoning: bool = True,
) -> dict | None:
    role = msg.get('role', 'user')
    if role == 'system':
        return None
    content = msg.get('content', [])

    if isinstance(content, str):
        content = [{'type': 'text', 'text': content}]

    if not content:
        return None

    tool_results = [b for b in content if isinstance(b, dict) and b.get('type') == 'tool_result']
    if tool_results:
        return {'_multi': [
            {
                'type': 'function_call_output',
                'call_id': r.get('tool_use_id', ''),
                'output': _stringify_content(r.get('content', '')),
            }
            for r in tool_results
        ]}

    reasoning_items = _reasoning_items_from_content(content) if include_reasoning else []

    tool_uses = [b for b in content if isinstance(b, dict) and b.get('type') == 'tool_use']
    if tool_uses:
        items = []
        for tu in tool_uses:
            items.append({
                'type': 'function_call',
                'call_id': tu.get('id', f'call_{uuid.uuid4().hex[:16]}'),
                'name': tu.get('name', ''),
                'arguments': json.dumps(tu.get('input', {})),
            })
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text' and block.get('text'):
                items.insert(0, {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': block['text']}],
                })
                break
        return {'_multi': reasoning_items + items}

    out_content = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get('type', '')
        if btype == 'text':
            text_type = 'output_text' if role == 'assistant' else 'input_text'
            out_content.append({'type': text_type, 'text': block.get('text', '')})
        elif btype == 'image':
            src = block.get('source', {})
            if src.get('type') == 'base64':
                data_url = f'data:{src["media_type"]};base64,{src["data"]}'
                out_content.append({'type': 'input_image', 'image_url': data_url})

    message_item = {
        'type': 'message',
        'role': role,
        'content': out_content,
    } if out_content else None

    if reasoning_items:
        items = list(reasoning_items)
        if message_item is not None:
            items.append(message_item)
        return {'_multi': items}

    return message_item


def _is_clean_user_turn(msg: dict) -> bool:
    """True when msg is a plain user turn — not an orphaned tool_result block."""
    if msg.get('role') != 'user':
        return False
    content = msg.get('content', [])
    if not isinstance(content, list):
        return True
    return not any(
        isinstance(b, dict) and b.get('type') == 'tool_result'
        for b in content
    )


def _truncate_messages_for_context(
    messages: list[dict], system, limit_tokens: int
) -> list[dict]:
    """Drop oldest messages until the conservative token estimate is within limit_tokens.

    Uses chars / 3 as a conservative estimate (accounts for code-heavy content that
    tokenizes at ~2-3 chars/token, not the 4-chars/token text heuristic).  After each
    drop, skips any leading message that is an orphaned tool_result or an assistant
    turn so the remaining list always starts with a clean user turn.
    """
    if limit_tokens <= 0 or not messages:
        return messages

    def _char_count(msgs):
        total = 0
        for m in msgs:
            c = m.get('content', '')
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        total += len(b.get('text') or b.get('output') or '')
        if isinstance(system, str):
            total += len(system)
        elif isinstance(system, list):
            for b in system:
                if isinstance(b, dict):
                    total += len(b.get('text') or '')
        return total

    def _estimate(msgs):
        return _char_count(msgs) // 3

    if _estimate(messages) <= limit_tokens:
        return messages

    msgs = list(messages)
    original_len = len(msgs)
    idx = 0
    while idx < len(msgs) - 1:
        if _estimate(msgs[idx:]) <= limit_tokens and _is_clean_user_turn(msgs[idx]):
            break
        idx += 1
    msgs = msgs[idx:]

    # Post-loop: if we landed on a non-clean message, skip past it.
    # This can occur when the loop exhausts (estimate still over limit or tail is non-clean).
    # Skip any leading assistant turns or orphaned tool_result blocks.
    while msgs and not _is_clean_user_turn(msgs[0]):
        msgs = msgs[1:]

    # Last-resort fallback: if cleanup emptied the list, walk back through the original
    # messages to find the latest adjacent (assistant w/ tool_use) + (user w/ tool_result)
    # pair.  This preserves a minimal valid exchange rather than returning an empty list
    # that would cause a 400 error from the upstream API.
    if not msgs:
        for i in range(len(messages) - 2, -1, -1):
            asst_msg = messages[i]
            user_msg = messages[i + 1]
            asst_content = asst_msg.get('content', [])
            user_content = user_msg.get('content', [])
            has_tool_use = (
                asst_msg.get('role') == 'assistant'
                and isinstance(asst_content, list)
                and any(
                    isinstance(b, dict) and b.get('type') == 'tool_use'
                    for b in asst_content
                )
            )
            has_tool_result = (
                user_msg.get('role') == 'user'
                and isinstance(user_content, list)
                and any(
                    isinstance(b, dict) and b.get('type') == 'tool_result'
                    for b in user_content
                )
            )
            if has_tool_use and has_tool_result:
                msgs = [asst_msg, user_msg]
                logger.warning(
                    'Codex: truncation left no clean user turn; falling back to last '
                    'tool-pair (messages[%d:%d]) to avoid empty input',
                    i, i + 2,
                )
                break

    dropped = original_len - len(msgs)
    if dropped:
        logger.warning(
            'Codex: dropped %d oldest message(s) to stay within ~%d token limit '
            '(conservative chars/3 estimate)',
            dropped, limit_tokens,
        )
    return msgs


def _map_request(payload: dict, context_limit: int = 0) -> dict:
    model = _resolve_model(payload.get('model', ''))
    system = payload.get('system', '')
    instruction_parts: list[str] = []
    if system:
        instruction_parts.append(_stringify_content(system))

    input_items: list[dict] = []
    messages = _truncate_messages_for_context(
        payload.get('messages', []) or [], system, context_limit
    )
    reasoning_index = None
    if len(messages) >= 2:
        previous_message = messages[-2]
        final_message = messages[-1]
        final_content = final_message.get('content', [])
        if (
            previous_message.get('role') == 'assistant'
            and final_message.get('role') == 'user'
            and isinstance(final_content, list)
            and any(
                isinstance(block, dict) and block.get('type') == 'tool_result'
                for block in final_content
            )
        ):
            reasoning_index = len(messages) - 2

    for index, msg in enumerate(messages):
        if msg.get('role') == 'system':
            text = _stringify_content(msg.get('content', ''))
            if text:
                instruction_parts.append(text)
            continue

        item = _convert_message_to_input_item(
            msg,
            include_reasoning=index == reasoning_index,
        )
        if item is None:
            continue
        if '_multi' in item:
            input_items.extend(item['_multi'])
        else:
            input_items.append(item)

    instructions = '\n\n'.join(p for p in instruction_parts if p)

    if not input_items:
        raise AnthropicRequestError(
            'Codex request requires at least one input item; '
            'messages are empty, system-only, or contained only unsupported content',
            error_type='invalid_request_error',
            status_code=400,
        )

    body: dict = {
        'model': model,
        'input': input_items,
        'instructions': instructions,
        'stream': True,
        'store': False,
    }

    tools = payload.get('tools')
    if tools:
        body['tools'] = [
            {
                'type': 'function',
                'name': t.get('name', ''),
                'description': t.get('description', ''),
                'parameters': t.get('input_schema', {}),
            }
            for t in tools
        ]

    tool_choice = payload.get('tool_choice')
    if tool_choice:
        tc_type = tool_choice.get('type', '') if isinstance(tool_choice, dict) else tool_choice
        if tc_type == 'auto':
            body['tool_choice'] = 'auto'
        elif tc_type == 'any':
            body['tool_choice'] = 'required'
        elif tc_type == 'tool':
            body['tool_choice'] = {
                'type': 'function',
                'name': tool_choice.get('name', ''),
            }

    effort = _effort_from_output_config(payload)
    thinking = payload.get('thinking')
    if effort is not None:
        body['reasoning'] = {'effort': effort}
        body['include'] = ['reasoning.encrypted_content']
    elif isinstance(thinking, dict) and thinking.get('type') == 'enabled':
        budget = thinking.get('budget_tokens', 8192)
        body['reasoning'] = {'effort': _effort_from_budget(budget)}
        body['include'] = ['reasoning.encrypted_content']

    if 'top_p' in payload:
        body['top_p'] = payload['top_p']

    cache_key = ''
    metadata = payload.get('metadata') or {}
    if isinstance(metadata, dict) and metadata.get('user_id'):
        try:
            parsed = json.loads(metadata['user_id'])
            cache_key = str(parsed['session_id'])
        except (json.JSONDecodeError, KeyError, TypeError):
            cache_key = str(metadata['user_id'])[:64]
    elif instructions:
        cache_key = hashlib.sha256(instructions.encode('utf-8')).hexdigest()[:32]
    if cache_key:
        body['prompt_cache_key'] = cache_key

    return body


def _map_response(output_items: list[dict], usage: dict, status: str, requested_model: str) -> dict:
    content = []
    has_tool_use = False

    for item in output_items:
        itype = item.get('type', '')
        if itype in ('reasoning_summary', 'reasoning'):
            summary_parts = item.get('summary', []) or []
            text = ''.join(
                p.get('text', '') for p in summary_parts if isinstance(p, dict)
            )
            if not text:
                text = item.get('text', '')
            encrypted = item.get('encrypted_content')
            if text or encrypted:
                block = {'type': 'thinking', 'thinking': text}
                if encrypted:
                    block['signature'] = _encode_reasoning_signature(
                        item.get('id', ''), encrypted,
                    )
                content.append(block)

        elif itype == 'message':
            for part in item.get('content', []):
                if isinstance(part, dict) and part.get('type') == 'output_text':
                    text = part.get('text', '')
                    if text:
                        content.append({'type': 'text', 'text': text})

        elif itype == 'function_call':
            try:
                input_data = json.loads(item.get('arguments', '{}') or '{}')
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            content.append({
                'type': 'tool_use',
                'id': item.get('call_id', f'toolu_{uuid.uuid4().hex[:24]}'),
                'name': item.get('name', ''),
                'input': input_data,
            })
            has_tool_use = True

    if has_tool_use:
        stop_reason = 'tool_use'
    elif status == 'incomplete':
        stop_reason = 'max_tokens'
    else:
        stop_reason = 'end_turn'

    _cached = (
        int(usage['input_tokens_details']['cached_tokens'])
        if isinstance(usage.get('input_tokens_details'), dict)
           and usage['input_tokens_details'].get('cached_tokens')
        else 0
    )
    return {
        'id': f'msg_{uuid.uuid4().hex}',
        'type': 'message',
        'role': 'assistant',
        'content': content or [{'type': 'text', 'text': ''}],
        'model': requested_model,
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {
            'input_tokens': max(0, int(usage.get('input_tokens', 0)) - _cached),
            'output_tokens': usage.get('output_tokens', 0),
            **({'cache_read_input_tokens': _cached} if _cached else {}),
        },
    }


def _raise_response_failed(event: dict) -> None:
    err = event.get('response', {}).get('error', {}) or {}
    message = err.get('message', 'Codex request failed')
    normalized = str(message).lower()
    context_overflow = any(phrase in normalized for phrase in (
        'prompt is too long',
        'maximum context length',
        'context length exceeded',
        'context window exceeded',
        'exceeds the context window',
        'context limit exceeded',
        'exceeded the context limit',
    ))
    raise AnthropicRequestError(
        message,
        error_type='invalid_request_error' if context_overflow else 'api_error',
        status_code=400 if context_overflow else 502,
    )


def _iter_stream_as_anthropic_sse(response, requested_model: str, estimated_input_tokens: int = 0):
    message_id = f'msg_{uuid.uuid4().hex}'
    block_index = 0
    text_block_open = False
    thinking_block_open = False
    has_text_opened = False
    tool_block_idx: dict[str, int] = {}
    tool_block_open: set[str] = set()
    stop_reason = 'end_turn'
    final_input_tokens = estimated_input_tokens
    final_output_tokens = 0
    final_cache_read: int | None = None

    yield sse_event('message_start', {
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'content': [],
            'model': requested_model,
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {
                'input_tokens': 0,
                'output_tokens': 0,
            },
        },
    })
    yield sse_event('ping', {'type': 'ping'})

    for raw_line in _read_sse_lines(response):
        line = raw_line.strip()
        if not line or not line.startswith('data: '):
            continue
        data = line[6:]
        if data.strip() in ('[DONE]', ''):
            continue

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        etype = event.get('type', '')

        if etype == 'response.output_text.delta':
            delta = event.get('delta', '')
            if not delta:
                continue
            if not text_block_open:
                if thinking_block_open:
                    thinking_block_open = False
                    yield emit_block_stop(block_index - 1)
                yield emit_block_start(block_index, {'type': 'text', 'text': ''})
                text_block_open = True
                has_text_opened = True
                block_index += 1
            yield emit_text_delta(block_index - 1, delta)

        elif etype in ('response.reasoning_summary_text.delta', 'response.reasoning_text.delta'):
            if has_text_opened:
                continue
            delta = event.get('delta', '')
            if not delta:
                continue
            if not thinking_block_open:
                yield emit_block_start(block_index, {'type': 'thinking', 'thinking': ''})
                thinking_block_open = True
                block_index += 1
            yield emit_thinking_delta(block_index - 1, delta)

        elif etype == 'response.reasoning_summary_part.added':
            if has_text_opened or not thinking_block_open:
                continue
            yield emit_thinking_delta(block_index - 1, '\n\n')

        elif etype == 'response.output_item.added':
            item = event.get('item', {})
            if not item:
                continue
            itype = item.get('type', '')
            item_id = item.get('id', '')

            if itype == 'function_call':
                if thinking_block_open:
                    thinking_block_open = False
                    yield emit_block_stop(block_index - 1)
                if text_block_open:
                    text_block_open = False
                    yield emit_block_stop(block_index - 1)
                tool_block_idx[item_id] = block_index
                tool_block_open.add(item_id)
                yield emit_block_start(block_index, {
                    'type': 'tool_use',
                    'id': item.get('call_id', f'toolu_{uuid.uuid4().hex[:24]}'),
                    'name': item.get('name', ''),
                    'input': {},
                })
                block_index += 1

        elif etype == 'response.function_call_arguments.delta':
            item_id = event.get('item_id', '')
            delta = event.get('delta', '')
            if not delta or item_id not in tool_block_idx:
                continue
            yield emit_input_json_delta(tool_block_idx[item_id], delta)

        elif etype == 'response.output_item.done':
            item = event.get('item', {})
            item_id = item.get('id', '')
            if item_id in tool_block_open:
                tool_block_open.discard(item_id)
                yield emit_block_stop(tool_block_idx[item_id])
            elif item.get('type') == 'reasoning' and item.get('encrypted_content') and not has_text_opened:
                # Attach the opaque reasoning state to the thinking block as a
                # signature so the next turn can replay it (reasoning continuity).
                if not thinking_block_open:
                    yield emit_block_start(block_index, {'type': 'thinking', 'thinking': ''})
                    thinking_block_open = True
                    block_index += 1
                yield emit_signature_delta(
                    block_index - 1,
                    _encode_reasoning_signature(item_id, item['encrypted_content']),
                )

        elif etype in ('response.completed', 'response.done', 'response.incomplete'):
            resp = event.get('response', {}) or {}
            usage = resp.get('usage', {}) or {}
            resp_status = resp.get('status', 'completed')
            if etype == 'response.incomplete':
                resp_status = 'incomplete'

            final_input_tokens = usage.get('input_tokens', estimated_input_tokens)
            final_output_tokens = usage.get('output_tokens', 0)
            cached = (usage.get('input_tokens_details') or {}).get('cached_tokens')
            if cached:
                final_cache_read = int(cached)
                final_input_tokens = max(0, final_input_tokens - final_cache_read)

            if resp_status == 'incomplete':
                stop_reason = 'max_tokens'
            elif tool_block_idx:
                stop_reason = 'tool_use'
            else:
                stop_reason = 'end_turn'

            break

        elif etype == 'response.failed':
            _raise_response_failed(event)

    if thinking_block_open:
        yield emit_block_stop(block_index - 1)
    if text_block_open:
        yield emit_block_stop(block_index - 1)
    for item_id in list(tool_block_open):
        yield emit_block_stop(tool_block_idx[item_id])

    usage_out: dict = {
        'input_tokens': final_input_tokens,
        'output_tokens': final_output_tokens,
    }
    if final_cache_read:
        usage_out['cache_read_input_tokens'] = final_cache_read
    msg_delta, msg_stop = emit_message_delta_stop(stop_reason, usage_out)
    yield msg_delta
    yield msg_stop
