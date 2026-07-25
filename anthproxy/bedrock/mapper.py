import base64
import logging
import re
import uuid

from .. import model_config as _model_config
from ..mapper import (
    AnthropicRequestError,
    emit_block_start,
    emit_block_stop,
    emit_input_json_delta,
    emit_message_delta_stop,
    emit_signature_delta,
    emit_text_delta,
    emit_thinking_delta,
    sse_event,
    strip_codex_thinking_blocks as _strip_codex_thinking_blocks,
)

logger = logging.getLogger(__name__)


# Model alias mapping: Anthropic API model names -> AWS Bedrock model IDs.
#
# This map was built by cross-referencing two sources in the Cline codebase
# (~/projects/cline/src/shared/api.ts):
#
#   1. `anthropicModels` / `claudeCodeModels` — the model IDs that Cline sends
#      in the Anthropic-format request body (e.g. "claude-opus-4-6", "sonnet").
#   2. `bedrockModels` — the corresponding AWS Bedrock model IDs
#      (e.g. "anthropic.claude-opus-4-6-v1").
#
# To update when new models are added:
#   a) Open ~/projects/cline/src/shared/api.ts
#   b) Find new keys in `anthropicModels` or `claudeCodeModels`
#   c) Find the matching Bedrock ID in `bedrockModels`
#   d) Add the mapping below:  '<anthropic_alias>': '<bedrock_model_id>'
#   e) If the Bedrock model requires a cross-region inference profile,
#      also add it to INFERENCE_PROFILE_REQUIRED_MODELS below.
#
# Model alias tables are now driven by ~/.anthproxy/config.json (or built-in defaults
# in anthproxy/model_config.py).  The names are kept for backward compatibility with
# any code/tests that import them directly.
MODEL_ALIASES = _model_config.model_aliases('bedrock')
INFERENCE_PROFILE_REQUIRED_MODELS = _model_config.inference_profile_models()


_DATE_SUFFIX_RE = re.compile(r'-\d{8}$')
_CONTEXT_SUFFIX_RE = re.compile(r'(:\d+m|\[\d+m\])$', re.IGNORECASE)
# Matches Claude 4+ models by name pattern: anthropic.claude-{role}-4...
# Distinguishes from claude-3-x (where the digit follows immediately after "claude-")
_CLAUDE4_RE = re.compile(r'^anthropic\.claude-[a-z]+-4')


def _needs_inference_profile(model):
    base = _CONTEXT_SUFFIX_RE.sub('', model)
    return base in INFERENCE_PROFILE_REQUIRED_MODELS or bool(_CLAUDE4_RE.match(base))


def _guess_bedrock_model_id(model):
    """Best-effort translation of an unknown claude-* alias to a Bedrock model ID.

    Strips context-window suffixes (:1m / [1m]), prepends 'anthropic.',
    and appends '-v1:0' when the name ends with a date (YYYYMMDD).
    Context-window variants get ':1m' appended to the result.
    """
    context_suffix = ''
    m = _CONTEXT_SUFFIX_RE.search(model)
    if m:
        context_suffix = ':1m'
        model = model[:m.start()]

    bedrock_id = 'anthropic.' + model
    if _DATE_SUFFIX_RE.search(model):
        bedrock_id += '-v1:0'

    return bedrock_id + context_suffix


def normalize_model_id(model):
    if not isinstance(model, str) or not model:
        raise AnthropicRequestError('model is required')

    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]

    bedrock_prefixes = ('anthropic.', 'amazon.', 'openai.', 'qwen.', 'deepseek.', 'meta.', 'arn:')
    inference_profile_prefixes = ('us.', 'eu.', 'apac.', 'jp.', 'global.')
    if model.startswith(bedrock_prefixes) or model.startswith(inference_profile_prefixes):
        return model

    if model.startswith('claude'):
        bedrock_id = _guess_bedrock_model_id(model)
        logger.warning(
            'Unknown Claude model alias %r — guessing Bedrock ID %r. '
            'Add to MODEL_ALIASES if incorrect.',
            model, bedrock_id,
        )
        return bedrock_id

    return model


def apply_inference_profile_model_id(model, region_name='us-east-1', use_inference_profile=True, use_global=False):
    if not use_inference_profile or not _needs_inference_profile(model):
        return model

    if model.startswith(('us.', 'eu.', 'apac.', 'jp.', 'global.')):
        return model

    if use_global:
        return 'global.%s' % model

    region_name = region_name or 'us-east-1'
    if region_name.startswith('us-'):
        return 'us.%s' % model
    if region_name.startswith('eu-'):
        return 'eu.%s' % model
    if region_name.startswith('ap-northeast-1') and _needs_inference_profile(model):
        return 'jp.%s' % model
    if region_name.startswith('ap-'):
        return 'apac.%s' % model

    return model


def _ensure_content_blocks(content):
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]

    if isinstance(content, list):
        return content

    raise AnthropicRequestError('Message content must be a string or a list of content blocks')


def _has_cache_control(block):
    """Return True if the block carries an Anthropic cache_control directive."""
    cc = block.get('cache_control') if isinstance(block, dict) else None
    return isinstance(cc, dict) and cc.get('type') in ('ephemeral',)


def _normalize_system_blocks(system):
    if system is None:
        return []

    if isinstance(system, str):
        return [{'text': system}]

    if isinstance(system, list):
        blocks = []
        for block in system:
            if not isinstance(block, dict) or block.get('type', 'text') != 'text':
                raise AnthropicRequestError('Only text system blocks are supported')
            blocks.append({'text': block.get('text', '')})
            # Translate Anthropic cache_control into a Bedrock cachePoint block
            if _has_cache_control(block):
                blocks.append({'cachePoint': {'type': 'default'}})
        return blocks

    raise AnthropicRequestError('System prompt must be a string or a list of text blocks')


def _decode_image_source(source):
    if not isinstance(source, dict):
        raise AnthropicRequestError('Image source must be an object')

    source_type = source.get('type')
    if source_type == 'url':
        raise AnthropicRequestError(
            'Image URL sources are not supported by the Bedrock proxy; '
            'encode the image as base64 instead'
        )
    if source_type != 'base64':
        raise AnthropicRequestError('Only base64 image sources are supported')

    media_type = source.get('media_type', '')
    if '/' not in media_type:
        raise AnthropicRequestError('Image media_type is required')

    _, image_format = media_type.split('/', 1)

    try:
        image_bytes = base64.b64decode(source.get('data', ''))
    except Exception:
        raise AnthropicRequestError('Invalid base64 image payload')

    return {
        'image': {
            'format': image_format,
            'source': {
                'bytes': image_bytes,
            }
        }
    }


def _map_tool_result_content(content):
    if isinstance(content, str):
        return [{'text': content}]

    if isinstance(content, list):
        mapped = []
        for block in content:
            if not isinstance(block, dict):
                raise AnthropicRequestError('tool_result content blocks must be objects')

            block_type = block.get('type', 'text')
            if block_type == 'text':
                mapped.append({'text': block.get('text', '')})
            elif block_type == 'image':
                mapped.append(_decode_image_source(block.get('source')))
            else:
                raise AnthropicRequestError('Unsupported tool_result content block type: %s' % block_type)
        return mapped

    raise AnthropicRequestError('tool_result content must be a string or a list')


def map_anthropic_content_to_bedrock(content):
    mapped = []

    for block in _ensure_content_blocks(content):
        if not isinstance(block, dict):
            raise AnthropicRequestError('Content blocks must be objects')

        block_type = block.get('type', 'text')
        if block_type == 'text':
            mapped.append({'text': block.get('text', '')})
        elif block_type == 'image':
            mapped.append(_decode_image_source(block.get('source')))
        elif block_type == 'tool_use':
            if not block.get('name'):
                raise AnthropicRequestError('tool_use blocks require a name')

            mapped.append({
                'toolUse': {
                    'toolUseId': block.get('id') or ('toolu_%s' % uuid.uuid4().hex),
                    'name': block.get('name'),
                    'input': block.get('input', {}),
                }
            })
        elif block_type == 'tool_result':
            if not block.get('tool_use_id'):
                raise AnthropicRequestError('tool_result blocks require tool_use_id')

            mapped.append({
                'toolResult': {
                    'toolUseId': block.get('tool_use_id'),
                    'content': _map_tool_result_content(block.get('content', '')),
                    'status': 'error' if block.get('is_error') else 'success',
                }
            })
        elif block_type == 'thinking':
            reasoning_text = {'text': block.get('thinking', '')}
            signature = block.get('signature')
            if signature:
                reasoning_text['signature'] = signature
            mapped.append({
                'reasoningContent': {
                    'reasoningText': reasoning_text,
                }
            })
        elif block_type == 'redacted_thinking':
            try:
                redacted_bytes = base64.b64decode(block.get('data', ''))
            except Exception:
                raise AnthropicRequestError('Invalid base64 data in redacted_thinking block')
            mapped.append({
                'reasoningContent': {
                    'redactedContent': redacted_bytes,
                }
            })
        else:
            raise AnthropicRequestError('Unsupported content block type: %s' % block_type)

        # Translate Anthropic cache_control on any content block into a Bedrock cachePoint
        if _has_cache_control(block):
            mapped.append({'cachePoint': {'type': 'default'}})

    return mapped


def map_anthropic_tools_to_bedrock(tools):
    if not tools:
        return None

    mapped_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get('name'):
            raise AnthropicRequestError('Each tool must include a name')

        mapped_tools.append({
            'toolSpec': {
                'name': tool['name'],
                'description': tool.get('description', ''),
                'inputSchema': {
                    'json': tool.get('input_schema', {'type': 'object', 'properties': {}})
                }
            }
        })

    return mapped_tools


def map_tool_choice_to_bedrock(tool_choice):
    if tool_choice in (None, 'auto'):
        return {'auto': {}}

    if tool_choice == 'any':
        return {'any': {}}

    if tool_choice == 'none':
        return None

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get('type')
        if choice_type == 'auto':
            return {'auto': {}}
        if choice_type == 'any':
            return {'any': {}}
        if choice_type == 'tool' and tool_choice.get('name'):
            return {'tool': {'name': tool_choice['name']}}

    raise AnthropicRequestError('Unsupported tool_choice value')


def _map_messages_and_system(payload):
    """Shared core for map_anthropic_request_to_bedrock and map_count_tokens_request_to_bedrock.

    Returns a request_kwargs dict with modelId, messages, and (if present) system.
    Does NOT include inferenceConfig, toolConfig, or additionalModelRequestFields.
    """
    if not isinstance(payload, dict):
        raise AnthropicRequestError('JSON body must be an object')

    model = normalize_model_id(payload.get('model'))
    messages = payload.get('messages')

    if not isinstance(messages, list) or not messages:
        raise AnthropicRequestError('messages must be a non-empty list')

    # Strip thinking blocks carrying Codex synthetic signatures (codexenc:
    # prefix).  Bedrock passes signatures through to its reasoningContent API;
    # foreign signatures cause provider-side rejections on Codex→Bedrock
    # backend switches.
    messages = _strip_codex_thinking_blocks(messages)

    bedrock_messages = []
    inline_system_texts = []
    for message in messages:
        if not isinstance(message, dict):
            raise AnthropicRequestError('Each message must be an object')

        role = message.get('role')
        if role == 'system':
            content = message.get('content', '')
            if isinstance(content, str):
                inline_system_texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        inline_system_texts.append(block.get('text', ''))
            continue

        if role not in ('user', 'assistant'):
            raise AnthropicRequestError(
                'Only user and assistant roles are supported (got: %r)' % role
            )

        bedrock_messages.append({
            'role': role,
            'content': map_anthropic_content_to_bedrock(message.get('content', '')),
        })

    request_kwargs = {
        'modelId': model,
        'messages': bedrock_messages,
    }

    system = _normalize_system_blocks(payload.get('system'))
    for text in inline_system_texts:
        system.append({'text': text})
    if system:
        request_kwargs['system'] = system

    return request_kwargs


def map_anthropic_request_to_bedrock(payload):
    request_kwargs = _map_messages_and_system(payload)

    inference_config = {}
    if payload.get('max_tokens') is not None:
        inference_config['maxTokens'] = payload.get('max_tokens')
    if payload.get('temperature') is not None:
        inference_config['temperature'] = payload.get('temperature')
    if payload.get('top_p') is not None:
        inference_config['topP'] = payload.get('top_p')
    if payload.get('stop_sequences') is not None:
        inference_config['stopSequences'] = payload.get('stop_sequences')
    if inference_config:
        request_kwargs['inferenceConfig'] = inference_config

    mapped_tools = map_anthropic_tools_to_bedrock(payload.get('tools'))
    mapped_tool_choice = map_tool_choice_to_bedrock(payload.get('tool_choice'))
    if mapped_tools:
        request_kwargs['toolConfig'] = {'tools': mapped_tools}
        if mapped_tool_choice:
            request_kwargs['toolConfig']['toolChoice'] = mapped_tool_choice

    # Note: Anthropic's 'metadata' field is intentionally NOT forwarded to
    # Bedrock's requestMetadata — Bedrock imposes strict constraints (values
    # ≤256 chars, limited character set) that Anthropic metadata values
    # routinely violate.

    additional_fields = {}
    # Note: anthropic_beta flags (prompt-caching, extended-thinking, etc.) are
    # Anthropic Messages API concepts and must NOT be forwarded to Bedrock's
    # additionalModelRequestFields — Bedrock handles these features natively
    # via cachePoint blocks and the thinking parameter respectively.
    if payload.get('thinking') is not None:
        additional_fields['thinking'] = payload.get('thinking')
    if payload.get('output_config') is not None:
        additional_fields['output_config'] = payload.get('output_config')
    if additional_fields:
        request_kwargs['additionalModelRequestFields'] = additional_fields

    return request_kwargs


def map_bedrock_stop_reason(reason):
    mapping = {
        'end_turn': 'end_turn',
        'max_tokens': 'max_tokens',
        'stop_sequence': 'stop_sequence',
        'tool_use': 'tool_use',
    }
    return mapping.get(reason, 'end_turn')


def map_bedrock_content_to_anthropic(content):
    mapped = []

    for block in content or []:
        if 'text' in block:
            mapped.append({'type': 'text', 'text': block.get('text', '')})
        elif 'toolUse' in block:
            tool_use = block.get('toolUse', {})
            mapped.append({
                'type': 'tool_use',
                'id': tool_use.get('toolUseId') or ('toolu_%s' % uuid.uuid4().hex),
                'name': tool_use.get('name'),
                'input': tool_use.get('input', {}),
            })
        elif 'reasoningContent' in block:
            reasoning = block.get('reasoningContent', {})
            if 'redactedContent' in reasoning:
                redacted_data = reasoning['redactedContent']
                if isinstance(redacted_data, bytes):
                    redacted_data = base64.b64encode(redacted_data).decode('ascii')
                mapped.append({
                    'type': 'redacted_thinking',
                    'data': redacted_data,
                })
            else:
                reasoning_text = reasoning.get('reasoningText', {})
                thinking_text = reasoning_text.get('text', '')
                thinking_signature = reasoning_text.get('signature', '')
                thinking_block = {'type': 'thinking', 'thinking': thinking_text}
                if thinking_signature:
                    thinking_block['signature'] = thinking_signature
                mapped.append(thinking_block)

    return mapped


def _map_bedrock_usage_to_anthropic(usage):
    """Translate Bedrock usage metadata into Anthropic usage shape.

    Bedrock fields:
      - inputTokens / outputTokens
      - cacheReadInputTokens / cacheWriteInputTokens

    Anthropic fields:
      - input_tokens / output_tokens
      - cache_read_input_tokens / cache_creation_input_tokens

    Cache fields are only included when present in the Bedrock response.
    Legacy ``*TokenCount`` spellings are accepted as fallbacks.
    """
    result = {
        'input_tokens': usage.get('inputTokens', 0),
        'output_tokens': usage.get('outputTokens', 0),
    }
    cache_read = usage.get('cacheReadInputTokens')
    if cache_read is None:
        cache_read = usage.get('cacheReadInputTokenCount')
    cache_write = usage.get('cacheWriteInputTokens')
    if cache_write is None:
        cache_write = usage.get('cacheWriteInputTokenCount')
    if cache_read is not None:
        result['cache_read_input_tokens'] = cache_read
    if cache_write is not None:
        result['cache_creation_input_tokens'] = cache_write
    return result


def map_bedrock_response_to_anthropic(response, requested_model):
    output = response.get('output', {}).get('message', {})
    usage = response.get('usage', {}) or {}

    return {
        'id': 'msg_%s' % uuid.uuid4().hex,
        'type': 'message',
        'role': output.get('role', 'assistant'),
        'content': map_bedrock_content_to_anthropic(output.get('content', [])),
        'model': requested_model,
        'stop_reason': map_bedrock_stop_reason(response.get('stopReason')),
        'stop_sequence': None,
        'usage': _map_bedrock_usage_to_anthropic(usage),
    }


def iter_bedrock_stream_as_anthropic_sse(stream_response, requested_model, estimated_usage=None, on_actual_usage=None):
    message_id = 'msg_%s' % uuid.uuid4().hex
    # NOTE: Bedrock only reports token usage in the final 'metadata' stream
    # event. Anthropic's native API knows input_tokens before streaming
    # begins and sends them in message_start. We therefore start with the
    # best available estimate and replace it with Bedrock's actual totals in
    # the final message_delta event.
    usage = dict(estimated_usage or {})
    usage.setdefault('input_tokens', 0)
    usage.setdefault('output_tokens', 0)
    usage['output_tokens'] = 0
    if usage['input_tokens'] < 0:
        usage['input_tokens'] = 0
    if usage['output_tokens'] < 0:
        usage['output_tokens'] = 0
    for key in ('cache_read_input_tokens', 'cache_creation_input_tokens'):
        value = usage.get(key)
        if value is not None and value < 0:
            usage[key] = 0
    if not usage['input_tokens'] and not usage.get('cache_read_input_tokens') and not usage.get('cache_creation_input_tokens'):
        usage['input_tokens'] = 1
    actual_usage = None

    def _record_actual_usage(mapped_usage):
        if on_actual_usage is None:
            return
        try:
            on_actual_usage(dict(mapped_usage))
        except Exception as exc:
            logger.warning('Bedrock usage callback failed: %s', exc)

    stop_reason = 'end_turn'

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
            'usage': usage,
        }
    })

    # Track which block indices have had content_block_start emitted.
    # Bedrock does not send contentBlockStart for reasoning blocks — only
    # contentBlockDelta events with reasoningContent.  We emit a synthetic
    # thinking content_block_start on the first delta for those indices.
    started_block_indices = set()

    try:
        for event in stream_response.get('stream') or []:
            if 'contentBlockStart' in event:
                block_start = event.get('contentBlockStart', {})
                content_block_index = block_start.get('contentBlockIndex', 0)
                start = block_start.get('start', {})

                if 'toolUse' in start:
                    tool_use = start.get('toolUse', {})
                    content_block = {
                        'type': 'tool_use',
                        'id': tool_use.get('toolUseId') or ('toolu_%s' % uuid.uuid4().hex),
                        'name': tool_use.get('name'),
                        'input': {},
                    }
                elif 'reasoningContent' in start:
                    reasoning_start = start.get('reasoningContent', {})
                    if 'redactedContent' in reasoning_start:
                        # Rare: redactedContent present in the start event itself.
                        redacted_data = reasoning_start['redactedContent']
                        if isinstance(redacted_data, bytes):
                            redacted_data = base64.b64encode(redacted_data).decode('ascii')
                        content_block = {
                            'type': 'redacted_thinking',
                            'data': redacted_data,
                        }
                    else:
                        # Bedrock's contentBlockStart for reasoning typically has
                        # an empty reasoningContent — we can't tell yet whether
                        # the block will be thinking or redacted_thinking.  Defer
                        # the content_block_start emission to the first delta,
                        # which carries the actual type (text vs redactedContent).
                        continue
                else:
                    content_block = {
                        'type': 'text',
                        'text': '',
                    }

                started_block_indices.add(content_block_index)
                yield emit_block_start(content_block_index, content_block)
                continue

            if 'contentBlockDelta' in event:
                block_delta = event.get('contentBlockDelta', {})
                content_block_index = block_delta.get('contentBlockIndex', 0)
                delta = block_delta.get('delta', {})

                if 'text' in delta:
                    # Adaptive-thinking models (e.g. Opus 4.6) can emit text
                    # deltas without a prior contentBlockStart.  Guard the same
                    # way we already do for reasoning blocks.
                    if content_block_index not in started_block_indices:
                        started_block_indices.add(content_block_index)
                        yield emit_block_start(content_block_index, {'type': 'text', 'text': ''})
                    yield emit_text_delta(content_block_index, delta.get('text', ''))
                elif 'toolUse' in delta and 'input' in delta.get('toolUse', {}):
                    yield emit_input_json_delta(content_block_index, delta.get('toolUse', {}).get('input', ''))
                elif 'reasoningContent' in delta:
                    reasoning = delta.get('reasoningContent', {})
                    # Bedrock doesn't always send contentBlockStart for reasoning
                    # blocks, so emit a synthetic one on the first delta.
                    if content_block_index not in started_block_indices:
                        started_block_indices.add(content_block_index)
                        if 'redactedContent' in reasoning:
                            redacted_data = reasoning['redactedContent']
                            if isinstance(redacted_data, bytes):
                                redacted_data = base64.b64encode(redacted_data).decode('ascii')
                            yield emit_block_start(content_block_index, {'type': 'redacted_thinking', 'data': redacted_data})
                            # redacted_thinking blocks are complete in the start event;
                            # no incremental deltas needed
                            continue
                        else:
                            yield emit_block_start(content_block_index, {'type': 'thinking', 'thinking': ''})
                    if 'text' in reasoning:
                        yield emit_thinking_delta(content_block_index, reasoning.get('text', ''))
                    elif 'signature' in reasoning:
                        yield emit_signature_delta(content_block_index, reasoning.get('signature', ''))
                    elif 'redactedContent' in reasoning:
                        # Additional redacted chunks after block already started —
                        # Anthropic SSE has no incremental delta type for
                        # redacted_thinking, so log and skip.
                        logger.debug('Ignoring additional redactedContent delta for block %d', content_block_index)
                continue

            if 'contentBlockStop' in event:
                yield emit_block_stop(event.get('contentBlockStop', {}).get('contentBlockIndex', 0))
                continue

            if 'messageStop' in event:
                stop_reason = map_bedrock_stop_reason(event.get('messageStop', {}).get('stopReason'))
                continue

            if 'metadata' in event:
                usage_payload = event.get('metadata', {}).get('usage', {})
                actual_usage = _map_bedrock_usage_to_anthropic(usage_payload)
                usage = actual_usage
                _record_actual_usage(actual_usage)

    except Exception as exc:
        logger.warning('Bedrock stream error mid-stream: %s', exc)
        yield sse_event('error', {
            'type': 'error',
            'error': {
                'type': 'api_error',
                'message': 'Upstream stream error: %s' % str(exc),
            }
        })
        return

    # Emit message_delta with complete usage from Bedrock's metadata event.
    # Per Anthropic spec message_delta.usage normally only contains
    # output_tokens, but because Bedrock doesn't provide input_tokens until
    # the stream ends, we include the full usage here so clients can obtain
    # accurate totals.  We also include input_tokens in this event (non-
    # standard but necessary) so that clients that missed it from
    # message_start can still report correct values.
    msg_delta, msg_stop = emit_message_delta_stop(stop_reason, usage)
    yield msg_delta
    yield msg_stop


def map_count_tokens_request_to_bedrock(payload):
    """Translate an Anthropic count_tokens request into Bedrock converse kwargs.

    The count_tokens request accepts a subset of the messages request fields:
    ``model``, ``messages``, ``system``, and ``tools``.  We reuse the existing
    translation helpers and strip any fields that are irrelevant to token
    counting (stream, temperature, max_tokens, etc.).
    """
    request_kwargs = _map_messages_and_system(payload)

    mapped_tools = map_anthropic_tools_to_bedrock(payload.get('tools'))
    if mapped_tools:
        request_kwargs['toolConfig'] = {'tools': mapped_tools}

    return request_kwargs


def map_bedrock_count_tokens_response(input_tokens):
    """Build an Anthropic-compatible count_tokens response."""
    return {'token_count': input_tokens}
