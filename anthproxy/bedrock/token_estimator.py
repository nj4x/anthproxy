import hashlib
import json
import logging
import os
import pathlib
import tempfile
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_STATE_VERSION = 1
_MAX_PREFIXES_PER_MODEL = 128
_PREFIX_TTL_SECS = 6 * 60 * 60
_MIN_RATIO = 0.05
_MAX_RATIO = 20.0
_EWMA_ALPHA = 0.35


@dataclass(frozen=True)
class EstimatedUsage:
    input_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def as_anthropic(self):
        usage = {
            'input_tokens': max(0, int(self.input_tokens)),
            'output_tokens': 0,
        }
        if self.cache_read_input_tokens:
            usage['cache_read_input_tokens'] = max(0, int(self.cache_read_input_tokens))
        if self.cache_creation_input_tokens:
            usage['cache_creation_input_tokens'] = max(0, int(self.cache_creation_input_tokens))
        return usage


@dataclass(frozen=True)
class RequestEstimateContext:
    model_key: str
    prefix_hash: str | None
    prefix_chars: int
    suffix_chars: int
    has_cache_prefix: bool


class BedrockTokenEstimator:
    def __init__(self, home: pathlib.Path):
        self._home = home
        self._state_path = home / 'token-estimator.json'
        self._lock = threading.Lock()
        self._state = self._load_state()

    def build_context(self, request: dict) -> RequestEstimateContext:
        model_key = _normalize_model_key(request.get('modelId'))
        prefix_obj, suffix_obj = _split_cache_prefix(request)
        prefix_chars = _measure_chars(prefix_obj)
        suffix_chars = _measure_chars(suffix_obj)
        prefix_hash = _hash_json(prefix_obj) if prefix_obj is not None else None
        return RequestEstimateContext(
            model_key=model_key,
            prefix_hash=prefix_hash,
            prefix_chars=prefix_chars,
            suffix_chars=suffix_chars,
            has_cache_prefix=prefix_obj is not None,
        )

    def estimate(self, context: RequestEstimateContext) -> EstimatedUsage:
        with self._lock:
            model_state = self._state['models'].get(context.model_key, {})
            ratio = _clamp_ratio(model_state.get('ratio', 0.25))
            prefixes = model_state.get('prefixes', {})
            prefix_entry = prefixes.get(context.prefix_hash) if context.prefix_hash else None
            now = time.time()
            if prefix_entry and now - prefix_entry.get('observed_at', 0) > _PREFIX_TTL_SECS:
                prefix_entry = None

        prefix_tokens = max(0, round(context.prefix_chars * ratio))
        suffix_tokens = max(1 if context.suffix_chars or not context.has_cache_prefix else 0,
                            round(context.suffix_chars * ratio))

        if context.has_cache_prefix:
            if prefix_entry and prefix_entry.get('kind') == 'cache_read':
                return EstimatedUsage(
                    input_tokens=suffix_tokens,
                    cache_read_input_tokens=max(1, int(prefix_entry.get('tokens', prefix_tokens) or prefix_tokens)),
                )
            return EstimatedUsage(
                input_tokens=suffix_tokens,
                cache_creation_input_tokens=max(1, prefix_tokens),
            )

        return EstimatedUsage(input_tokens=max(1, prefix_tokens + suffix_tokens))

    def observe(self, context: RequestEstimateContext, actual_usage: dict) -> None:
        input_tokens = max(0, int(actual_usage.get('input_tokens', 0) or 0))
        cache_read = max(0, int(actual_usage.get('cache_read_input_tokens', 0) or 0))
        cache_create = max(0, int(actual_usage.get('cache_creation_input_tokens', 0) or 0))
        total_chars = context.prefix_chars + context.suffix_chars
        if total_chars <= 0:
            return

        total_tokens = input_tokens + cache_read + cache_create
        observed_ratio = _clamp_ratio(total_tokens / total_chars) if total_tokens else _MIN_RATIO

        with self._lock:
            models = self._state.setdefault('models', {})
            model_state = models.setdefault(context.model_key, {'ratio': 0.25, 'prefixes': {}})
            current_ratio = _clamp_ratio(model_state.get('ratio', 0.25))
            model_state['ratio'] = _clamp_ratio((1 - _EWMA_ALPHA) * current_ratio + _EWMA_ALPHA * observed_ratio)

            if context.prefix_hash and context.has_cache_prefix:
                kind = 'cache_read' if cache_read > 0 else 'cache_create'
                tokens = cache_read if cache_read > 0 else cache_create
                if tokens > 0:
                    prefixes = model_state.setdefault('prefixes', {})
                    prefixes[context.prefix_hash] = {
                        'kind': kind,
                        'tokens': int(tokens),
                        'observed_at': time.time(),
                    }
                    self._prune_prefixes(prefixes)

        try:
            self._save_state()
        except OSError as exc:
            logger.warning('Failed to persist Bedrock token estimator to %s: %s', self._state_path, exc)

    def _prune_prefixes(self, prefixes: dict) -> None:
        expired = [key for key, value in prefixes.items()
                   if time.time() - value.get('observed_at', 0) > _PREFIX_TTL_SECS]
        for key in expired:
            prefixes.pop(key, None)
        if len(prefixes) <= _MAX_PREFIXES_PER_MODEL:
            return
        for key, _ in sorted(prefixes.items(), key=lambda item: item[1].get('observed_at', 0))[:-_MAX_PREFIXES_PER_MODEL]:
            prefixes.pop(key, None)

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {'version': _STATE_VERSION, 'models': {}}
        try:
            data = json.loads(self._state_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {'version': _STATE_VERSION, 'models': {}}
        if not isinstance(data, dict) or data.get('version') != _STATE_VERSION:
            return {'version': _STATE_VERSION, 'models': {}}
        models = data.get('models')
        if not isinstance(models, dict):
            return {'version': _STATE_VERSION, 'models': {}}
        return {'version': _STATE_VERSION, 'models': models}

    def _save_state(self) -> None:
        self._home.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._home), prefix='.token-estimator.', suffix='.tmp')
        try:
            os.write(fd, payload.encode('utf-8'))
            os.close(fd)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._state_path)
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _normalize_model_key(model_id):
    if not isinstance(model_id, str) or not model_id:
        return 'unknown'
    if '.' in model_id:
        prefix, rest = model_id.split('.', 1)
        if prefix in ('us', 'eu', 'apac', 'global'):
            return rest
    return model_id


def _split_cache_prefix(request):
    system = list(request.get('system') or [])
    for index, block in enumerate(system):
        if isinstance(block, dict) and block.get('cachePoint'):
            return (
                {'system': system[:index + 1]},
                {
                    'system': system[index + 1:],
                    'messages': request.get('messages') or [],
                    'toolConfig': request.get('toolConfig') or {},
                },
            )
    return (None, {
        'system': system,
        'messages': request.get('messages') or [],
        'toolConfig': request.get('toolConfig') or {},
    })


def _hash_json(value):
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _measure_chars(value):
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, list):
        return sum(_measure_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(_measure_chars(k) + _measure_chars(v) for k, v in value.items())
    return len(str(value))


def _clamp_ratio(value):
    if value < _MIN_RATIO:
        return _MIN_RATIO
    if value > _MAX_RATIO:
        return _MAX_RATIO
    return value
