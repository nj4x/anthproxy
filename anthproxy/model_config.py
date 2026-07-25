"""External model configuration for anthproxy.

Model-alias tables, pricing, and related data that used to live as module-level
constants in the per-backend mappers are all defaulted here and can be overridden
via ``~/.anthproxy/config.json`` (or the path in ``ANTHPROXY_CONFIG``).

Merge semantics
---------------
- ``model_aliases.<backend>`` and ``model_pricing`` / ``model_labels`` sections:
  per-key deep-merge — file entries override/extend the built-in defaults.
- ``bedrock_inference_profile_models`` list: **replaced** when present in the file
  (lists cannot be meaningfully merged key-by-key).

The config file is never read at import time; ``load()`` is cached on first call and
thread-safe.  ``ensure_file()`` writes the defaults once at server startup so users
get an editable template; it is never called implicitly at import.

``reset()`` clears the cache — used in tests only.

Public API
----------
    load()                     -> dict          (merged config, cached)
    model_aliases(backend)     -> dict[str,str]
    inference_profile_models() -> frozenset[str]
    model_pricing()            -> dict[str, tuple[float,...]]
    model_labels()             -> dict[str,str]
    ensure_file()              -> None
    reset()                    -> None
"""

import copy
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in defaults (verbatim from the per-backend mappers + stats.py)
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "model_aliases": {
        "bedrock": {
            # Claude Code short aliases
            "fable": "anthropic.claude-fable-5",
            "fable[1m]": "anthropic.claude-fable-5:1m",
            "sonnet": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "sonnet[1m]": "anthropic.claude-sonnet-4-5-20250929-v1:0:1m",
            "opus": "anthropic.claude-opus-4-8",
            "opus[1m]": "anthropic.claude-opus-4-8:1m",
            # Sonnet 4-5
            "claude-4-5-sonnet": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-sonnet-4-5-20250929": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-sonnet-4-5-20250929:1m": "anthropic.claude-sonnet-4-5-20250929-v1:0:1m",
            "claude-sonnet-4-5-20250929[1m]": "anthropic.claude-sonnet-4-5-20250929-v1:0:1m",
            # Sonnet 4-6
            "claude-4-6-sonnet": "anthropic.claude-sonnet-4-6",
            "claude-sonnet-4-6": "anthropic.claude-sonnet-4-6",
            "claude-sonnet-4-6:1m": "anthropic.claude-sonnet-4-6:1m",
            "claude-sonnet-4-6[1m]": "anthropic.claude-sonnet-4-6:1m",
            # Sonnet 4-20250514
            "claude-sonnet-4-20250514": "anthropic.claude-sonnet-4-20250514-v1:0",
            "claude-sonnet-4-20250514:1m": "anthropic.claude-sonnet-4-20250514-v1:0:1m",
            "claude-sonnet-4-20250514[1m]": "anthropic.claude-sonnet-4-20250514-v1:0:1m",
            # Haiku 4-5
            "haiku": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5-20251001-v1:0",
            # Fable 5
            "claude-fable-5": "anthropic.claude-fable-5",
            "claude-fable-5:1m": "anthropic.claude-fable-5:1m",
            "claude-fable-5[1m]": "anthropic.claude-fable-5:1m",
            # Opus 4-8
            "claude-opus-4-8": "anthropic.claude-opus-4-8",
            "claude-opus-4-8:1m": "anthropic.claude-opus-4-8:1m",
            "claude-opus-4-8[1m]": "anthropic.claude-opus-4-8:1m",
            # Opus 4-7
            "claude-opus-4-7": "anthropic.claude-opus-4-7",
            "claude-opus-4-7:1m": "anthropic.claude-opus-4-7:1m",
            "claude-opus-4-7[1m]": "anthropic.claude-opus-4-7:1m",
            # Opus 4-6
            "claude-opus-4-6": "anthropic.claude-opus-4-6-v1",
            "claude-opus-4-6:1m": "anthropic.claude-opus-4-6-v1:1m",
            "claude-opus-4-6[1m]": "anthropic.claude-opus-4-6-v1:1m",
            # Opus 4-5
            "claude-opus-4-5-20251101": "anthropic.claude-opus-4-5-20251101-v1:0",
            # Opus 4-1
            "claude-opus-4-1-20250805": "anthropic.claude-opus-4-1-20250805-v1:0",
            # Opus 4-20250514
            "claude-opus-4-20250514": "anthropic.claude-opus-4-20250514-v1:0",
            # Claude 3.x models
            "claude-3-7-sonnet-20250219": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
        },
        "codex": {
            "opus": "gpt-5.6-sol",
            "opus[1m]": "gpt-5.6-sol",
            "sonnet": "gpt-5.6-terra",
            "haiku": "gpt-5.6-luna",
            "claude-opus-5": "gpt-5.6-sol",
            "claude-opus-4-8": "gpt-5.6-sol",
            "claude-sonnet-4-6": "gpt-5.6-terra",
            "claude-haiku-4-5-20251001": "gpt-5.6-luna",
            "default": "gpt-5.6-terra",
        },
        "anthropic": {
            "fable": "claude-fable-5",
            "opus": "claude-opus-5",
            "sonnet": "claude-sonnet-4-6",
            "haiku": "claude-haiku-4-5-20251001",
        },
        # 'local' backend: use the 'default' key as a catch-all target.
        # All unrecognised models resolve to this value.
        "local": {
            "default": "lmstudio-community/gemma-4-12B-it-MLX-4bit",
        },
        "openrouter": {
            "haiku": "deepseek/deepseek-v4-flash",
            "claude-haiku-4-5-20251001": "deepseek/deepseek-v4-flash",
            "sonnet": "z-ai/glm-5.2",
            "opus": "moonshotai/kimi-k3",
            "claude-opus-5": "moonshotai/kimi-k3",
            "default": "z-ai/glm-5.2",
        },
    },
    "bedrock_inference_profile_models": [
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-sonnet-4-6:1m",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-8:1m",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-7:1m",
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-opus-4-6-v1:1m",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0:1m",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0:1m",
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-opus-4-1-20250805-v1:0",
    ],
    # tier -> [input, output, cache_read, cache_write] USD per million tokens
    "model_pricing": {
        "fable": [10.0, 30.0, 1.00, 12.50],
        "opus": [5.0, 25.0, 0.50, 6.25],
        "sonnet": [3.0, 15.0, 0.30, 3.75],
        "haiku": [1.0, 5.0, 0.10, 1.25],
    },
    "model_labels": {
        "fable": "Fable",
        "opus": "Opus",
        "sonnet": "Sonnet",
        "haiku": "Haiku",
        "other": "Other",
    },
}


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_cache: dict | None = None
_cache_lock = threading.Lock()


def _config_path() -> Path:
    override = os.environ.get("ANTHPROXY_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".anthproxy" / "config.json"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    For dict values: per-key merge.  For all other types (including lists):
    the override value replaces the base value entirely.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load() -> dict:
    """Return the merged model config (cached after first call)."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        path = _config_path()
        merged = copy.deepcopy(_DEFAULTS)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    merged = _deep_merge(merged, raw)
                else:
                    logger.warning(
                        "anthproxy config file %s: expected a JSON object, got %s — "
                        "using built-in defaults",
                        path,
                        type(raw).__name__,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "anthproxy config file %s could not be loaded (%s) — "
                    "using built-in defaults",
                    path,
                    exc,
                )
        _cache = merged
        return _cache


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------


def model_aliases(backend: str) -> dict:
    """Return the model-alias dict for *backend*."""
    return load()["model_aliases"].get(backend, {})


def inference_profile_models() -> frozenset:
    """Return the set of Bedrock model IDs that require a cross-region inference profile."""
    return frozenset(load().get("bedrock_inference_profile_models", []))


def model_pricing() -> dict:
    """Return tier -> (input, output, cache_read, cache_write) tuples (USD/M tokens)."""
    raw = load().get("model_pricing", {})
    # Convert lists from JSON to tuples to match the original dict format in stats.py.
    return {tier: tuple(vals) for tier, vals in raw.items()}


def model_labels() -> dict:
    """Return tier -> display-label string dict."""
    return dict(load().get("model_labels", {}))


# ---------------------------------------------------------------------------
# Startup helper — write defaults once so the user has an editable template
# ---------------------------------------------------------------------------


def ensure_file() -> None:
    """Write the default config to ``~/.anthproxy/config.json`` if it does not exist.

    Best-effort: logs a warning on failure but never raises.  Intended to be
    called once at server startup before the first ``load()``.  Never called at
    import time so tests and library users are not affected.
    """
    path = _config_path()
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_DEFAULTS, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote default model config to %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write default model config to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def reset() -> None:
    """Clear the config cache.  For use in tests only."""
    global _cache
    with _cache_lock:
        _cache = None
