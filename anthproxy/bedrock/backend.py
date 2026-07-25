import json
import logging
import os
import pathlib
import tempfile
import threading

from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from botocore.session import Session

from .._shared import Backend
from ..config import Config
from ..mapper import AnthropicRequestError
from .auth import extract_aws_credentials
from .mapper import (
    apply_inference_profile_model_id,
    iter_bedrock_stream_as_anthropic_sse,
    map_anthropic_request_to_bedrock,
    map_bedrock_count_tokens_response,
    map_bedrock_response_to_anthropic,
    map_count_tokens_request_to_bedrock,
)
from .token_estimator import BedrockTokenEstimator

logger = logging.getLogger(__name__)


def _bedrock_home(config=None) -> pathlib.Path:
    """Priority: config.bedrock_home -> $BEDROCK_HOME -> ~/.bedrock."""
    if config is not None and getattr(config, 'bedrock_home', ''):
        return pathlib.Path(config.bedrock_home)
    env = os.environ.get('BEDROCK_HOME', '').strip()
    if env:
        p = pathlib.Path(env)
        if p.is_dir():
            return p
    return pathlib.Path.home() / '.bedrock'


def _cache_file(home: pathlib.Path) -> pathlib.Path:
    return home / 'credentials.json'


def _load_cache(home: pathlib.Path) -> dict[str, str]:
    """Load the cache file; return {} on missing/malformed (never raise)."""
    cache_file = _cache_file(home)
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _save_cache(home: pathlib.Path, cache: dict[str, str]) -> None:
    """Atomically write the whole cache as JSON with mode 0600."""
    home.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file(home)
    text = json.dumps(cache, indent=2, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(dir=str(home), prefix='.credentials.', suffix='.tmp')
    try:
        os.write(fd, text.encode('utf-8'))
        os.close(fd)
        os.chmod(tmp, 0o600)         # chmod before replace so the file is never world-readable
        os.replace(tmp, cache_file)  # atomic on same filesystem
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


class BedrockRuntimeAdapter:
    def __init__(self, region_name='us-east-1', aws_access_key_id=None,
                 aws_secret_access_key=None, aws_session_token=None):
        session = Session()
        client_kwargs = {
            'region_name': region_name,
            'config': BotoConfig(read_timeout=840, connect_timeout=10),
        }
        if aws_access_key_id is not None:
            client_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key is not None:
            client_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if aws_session_token is not None:
            client_kwargs['aws_session_token'] = aws_session_token

        self.client = session.create_client('bedrock-runtime', **client_kwargs)

    def converse(self, **kwargs):
        return self.client.converse(**kwargs)

    def converse_stream(self, **kwargs):
        return self.client.converse_stream(**kwargs)

    def count_tokens(self, **kwargs):
        kwargs.setdefault('inferenceConfig', {})
        kwargs['inferenceConfig']['maxTokens'] = 1
        response = self.client.converse(**kwargs)
        usage = response.get('usage', {}) or {}
        return usage.get('inputTokens', 0)


def _map_client_error(exc):
    """Translate botocore ClientError to AnthropicRequestError."""
    error = getattr(exc, 'response', {}).get('Error', {})
    error_code = error.get('Code', 'ClientError')
    error_message = error.get('Message') or str(exc)
    status_code = 502
    error_type = 'api_error'

    if error_code in ('ValidationException',):
        status_code = 400
        error_type = 'invalid_request_error'
    elif error_code in ('AccessDeniedException', 'UnrecognizedClientException'):
        status_code = 401
        error_type = 'authentication_error'
    elif error_code in ('ExpiredTokenException',):
        status_code = 401
        error_type = 'authentication_error'
    elif error_code in ('ThrottlingException', 'TooManyRequestsException'):
        status_code = 429
        error_type = 'rate_limit_error'

    logger.warning('Bedrock client error %s: %s', error_code, error_message)
    return AnthropicRequestError(error_message, error_type=error_type, status_code=status_code)


class BedrockBackend(Backend):
    def __init__(self, config: Config | None = None):
        self._home = _bedrock_home(config)
        self._lock = threading.Lock()
        self._credential_cache: dict[str, str] = _load_cache(self._home)
        self._token_estimator = BedrockTokenEstimator(self._home)
        if self._credential_cache:
            logger.info('Loaded %d cached Bedrock credential(s) from %s',
                        len(self._credential_cache), _cache_file(self._home))

    def store_cached_credential(self, key: str, value: str) -> None:
        logger.info('Caching credentials: %s', key)
        with self._lock:
            self._credential_cache[key] = value
            try:
                _save_cache(self._home, self._credential_cache)
            except OSError as exc:
                logger.warning('Failed to persist Bedrock credential cache to %s: %s',
                               _cache_file(self._home), exc)

    def parse_credentials(self, api_key: str) -> dict:
        if len(api_key) <= 250:
            with self._lock:
                cached = self._credential_cache.get(api_key)
            if cached:
                logger.info('Using cached credentials: %s', api_key)
                return extract_aws_credentials(cached)
        return extract_aws_credentials(api_key)

    def _make_adapter(self, credentials, config):
        return BedrockRuntimeAdapter(
            region_name=config.region,
            **credentials,
        )

    def _apply_profile(self, model_id, config):
        return apply_inference_profile_model_id(
            model_id,
            region_name=config.region,
            use_inference_profile=config.use_inference_profile,
            use_global=config.use_global_inference_profile,
        )

    def send_message(self, payload: dict, credentials: dict, config: Config) -> dict:
        try:
            bedrock_request = map_anthropic_request_to_bedrock(payload)
            bedrock_request['modelId'] = self._apply_profile(bedrock_request['modelId'], config)
            requested_model = payload.get('model')

            adapter = self._make_adapter(credentials, config)
            bedrock_resp = adapter.converse(**bedrock_request)
            return map_bedrock_response_to_anthropic(bedrock_resp, requested_model)
        except ClientError as exc:
            raise _map_client_error(exc)

    def send_message_stream(self, payload: dict, credentials: dict, config: Config):
        try:
            bedrock_request = map_anthropic_request_to_bedrock(payload)
            bedrock_request['modelId'] = self._apply_profile(bedrock_request['modelId'], config)
            requested_model = payload.get('model')
            estimate_context = self._token_estimator.build_context(bedrock_request)
            estimated_usage = self._token_estimator.estimate(estimate_context).as_anthropic()

            adapter = self._make_adapter(credentials, config)
            response = adapter.converse_stream(**bedrock_request)
            return iter_bedrock_stream_as_anthropic_sse(
                response,
                requested_model,
                estimated_usage=estimated_usage,
                on_actual_usage=lambda usage: self._token_estimator.observe(estimate_context, usage),
            )
        except ClientError as exc:
            raise _map_client_error(exc)

    def count_tokens(self, payload: dict, credentials: dict, config: Config) -> dict:
        try:
            bedrock_request = map_count_tokens_request_to_bedrock(payload)
            bedrock_request['modelId'] = self._apply_profile(bedrock_request['modelId'], config)

            adapter = self._make_adapter(credentials, config)
            input_tokens = adapter.count_tokens(**bedrock_request)
            return map_bedrock_count_tokens_response(input_tokens)
        except ClientError as exc:
            raise _map_client_error(exc)
