"""Shared model-alias resolution for Codex and Anthropic backends.

Each backend keeps its own alias **dict** (``CODEX_MODEL_ALIASES``,
``ANTHROPIC_MODEL_ALIASES``); only the resolution algorithm and the
``CONTEXT_SUFFIXES`` constant are shared here.

Not used by Gauss (frozen, keeps its own private copies).

Public API:
    CONTEXT_SUFFIXES
    resolve_alias(model, alias_dict, *, prefix_match=False)
"""

from ..mapper import AnthropicRequestError

# Context-window variant suffixes that should be stripped before alias lookup.
# Declared once here; codex.py and anthropic.py import this constant.
# Gauss's private copy (gauss.py line 365) stays untouched.
CONTEXT_SUFFIXES: tuple[str, ...] = (':1m', '[1m]')


def resolve_alias(model: str, alias_dict: dict[str, str], *, prefix_match: bool = False) -> str:
    """Resolve a model name using *alias_dict*.

    Resolution order:
    1. Raise ``AnthropicRequestError(400)`` when *model* is empty.
    2. Direct match in *alias_dict*.
    3. Strip each ``CONTEXT_SUFFIXES`` suffix and retry direct match.
    4. If *prefix_match* is ``True``: match any ``claude-`` prefixed key whose
       value would be the target, treating *model* as a full ID that starts with
       a known alias (e.g. ``claude-opus-4-8-20251201`` → ``claude-opus-4-8``).
    5. Pass *model* through verbatim (native backend ID or unknown).

    Args:
        model: the model string from the incoming request.
        alias_dict: the backend-specific alias dictionary.
        prefix_match: enable step 4 (needed by Codex; not needed by Anthropic).

    Returns:
        The resolved backend model ID.

    Raises:
        AnthropicRequestError: if *model* is empty/``None``.
    """
    if not model:
        raise AnthropicRequestError('model is required', status_code=400)

    # 1. Direct alias match
    if model in alias_dict:
        return alias_dict[model]

    # 2. Strip context-window suffixes and retry
    for suffix in CONTEXT_SUFFIXES:
        if model.endswith(suffix):
            base = model[:-len(suffix)]
            if base in alias_dict:
                return alias_dict[base]

    # 3. Prefix match on full Anthropic IDs (Codex only)
    if prefix_match:
        for alias, target in alias_dict.items():
            if alias.startswith('claude-') and model.startswith(alias):
                return target

    return model  # native backend ID or unknown — pass through
