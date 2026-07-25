"""Request mapping for the local (LM Studio) backend.

The local backend is an Anthropic-compatible pass-through: the upstream
(LM Studio) accepts the same Messages API wire format, so no structural
translation is needed.  The only work done here is model-alias resolution
so that short aliases (``sonnet``, ``opus``, ``haiku``) and full Claude IDs
are mapped to whatever LM Studio model is configured in config.json.

Model resolution
----------------
Consults ``model_config.model_aliases('local')``.  If the incoming model
string is a direct key in that dict it is mapped to the value.  Otherwise
the special ``"default"`` key is used.  This means every model string —
known or unknown — resolves to the configured local model (default:
``google/gemma-4-12b``).
"""

import json

from .. import model_config as _model_config


def _resolve_model(model: str) -> str:
    """Resolve *model* to the LM Studio model ID.

    Uses the ``local`` section of the model config.  Falls back to the
    ``"default"`` key when the incoming model string is not an explicit alias.
    """
    aliases = _model_config.model_aliases('local')
    if model in aliases:
        return aliases[model]
    return aliases.get('default', model)


_INTERNAL_KEYS = frozenset({'_anthropic_beta', '_anthproxy_internal_classifier'})


def _build_body(payload: dict) -> bytes:
    """Build the outbound request body for LM Studio.

    Copies the payload as-is (LM Studio speaks native Anthropic wire format),
    strips internal proxy keys, and substitutes the resolved model ID.
    """
    body = {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}
    body['model'] = _resolve_model(payload.get('model', ''))
    return json.dumps(body).encode('utf-8')
