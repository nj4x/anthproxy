import argparse
import dataclasses
import json
import os
from pathlib import Path

from .backends_registry import backend_names as _backend_names


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from an env var.  '1'/'true'/'yes' → True; '0'/'false'/'no' → False."""
    val = os.environ.get(name, '').lower()
    if val in ('1', 'true', 'yes'):
        return True
    if val in ('0', 'false', 'no'):
        return False
    return default


def _validated_classifier_model(value: str, parser: argparse.ArgumentParser) -> str:
    model = (value or '').strip()
    if not model:
        parser.error('auto model routing classifier model must be a non-empty string')
    return model


_DEFAULT_CLASSIFICATION: dict[str, str] = {
    'trivial': 'haiku',
    'standard': 'sonnet',
    'deep': 'opus',
}

_VALID_CLASSIFICATION_LABELS: frozenset[str] = frozenset(_DEFAULT_CLASSIFICATION)


def _parse_classification_str(
    raw: str | None, p: argparse.ArgumentParser
) -> dict[str, str]:
    """Parse a comma-separated ``label:model`` string, overlay on defaults.

    Merges parsed pairs into ``_DEFAULT_CLASSIFICATION`` so unspecified labels
    keep their default targets.  Calls ``p.error()`` on any malformed input.
    """
    if not raw or not raw.strip():
        return dict(_DEFAULT_CLASSIFICATION)
    result = dict(_DEFAULT_CLASSIFICATION)
    for pair in raw.split(','):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(':', 1)
        if len(parts) != 2:
            p.error(
                f'--auto-model-routing-classification: malformed pair {pair!r}; '
                'expected label:model format'
            )
        label, model = parts[0].strip(), parts[1].strip()
        if label not in _VALID_CLASSIFICATION_LABELS:
            p.error(
                f'--auto-model-routing-classification: unknown label {label!r}; '
                'valid labels are: trivial, standard, deep'
            )
        if not model:
            p.error(
                f'--auto-model-routing-classification: model for label {label!r} '
                'must be a non-empty string'
            )
        result[label] = model
    return result


@dataclasses.dataclass
class Config:
    host: str = '127.0.0.1'
    port: int = 8082
    region: str = 'us-east-1'
    use_inference_profile: bool = True
    use_global_inference_profile: bool = False
    backend: str = 'bedrock'
    log_level: str = 'INFO'
    no_prompt_translate: bool = False
    request_history_size: int = 5
    log_file: str = '/tmp/anthproxy.log'
    codex_home: str = ''
    codex_unsupported_model_fallback: str = ''
    bedrock_home: str = ''
    anthropic_home: str = ''
    openrouter_api_key: str = ''
    local_base_url: str = 'http://127.0.0.1:1235'
    auto_backend: bool = True
    auto_backend_mode: str = 'subscription'
    auto_backend_interval: float = 60.0
    auto_backend_weekly_margin: float = 5.0
    auto_model_routing: bool = False
    auto_model_routing_classifier_model: str = 'haiku'
    auto_model_routing_long_context_threshold: int = 150_000
    auto_model_routing_affirmation_inherit: bool = True
    auto_model_routing_classification: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(_DEFAULT_CLASSIFICATION)
    )
    auto_model_routing_long: str = 'opus[1m]'
    auto_model_routing_confidence_bump: bool = False
    auto_model_routing_min_confidence: float = 0.0
    auto_model_routing_mode: str = 'classifier'
    auto_model_routing_task_tiers: dict[str, str] | None = None
    sse_keepalive_interval: float = 10.0
    db_path: str | None = None   # Path to SQLite DB file; None disables DB recording
    enable_ui: bool = False       # Whether /admin/* and /ui/* endpoints are active
    codex_context_limit: int = 100_000


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        prog='anthproxy',
        description='Standalone Anthropic-to-AWS Bedrock HTTP proxy',
    )
    p.add_argument('--host', default=os.environ.get('ANTHPROXY_HOST', '127.0.0.1'),
                   help='Bind address (default: 127.0.0.1, env: ANTHPROXY_HOST)')
    p.add_argument('--port', type=int,
                   default=int(os.environ.get('ANTHPROXY_PORT', '8082')),
                   help='Bind port (default: 8082, env: ANTHPROXY_PORT)')
    p.add_argument('--region',
                   default=os.environ.get('ANTHPROXY_REGION',
                           os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')),
                   help='AWS region (default: us-east-1, env: ANTHPROXY_REGION / AWS_DEFAULT_REGION)')
    p.add_argument('--no-inference-profile', dest='use_inference_profile',
                   action='store_false', default=True,
                   help='Disable cross-region inference profile prefixing')
    p.add_argument('--global-inference-profile', dest='use_global_inference_profile',
                   action='store_true', default=False,
                   help='Use global. prefix instead of region-based prefix')
    p.add_argument('--backend', default=os.environ.get('ANTHPROXY_BACKEND', 'bedrock'),
                   choices=list(_backend_names()),
                   help='LLM backend (default: bedrock)')
    p.add_argument('--codex-home',
                   default=os.environ.get('CODEX_HOME', ''),
                   help='Path to Codex home directory (default: ~/.codex,'
                        ' env: CODEX_HOME)')
    p.add_argument('--codex-unsupported-model-fallback',
                   dest='codex_unsupported_model_fallback',
                   default=os.environ.get('ANTHPROXY_CODEX_UNSUPPORTED_MODEL_FALLBACK', ''),
                   help='Opt-in model alias to retry once when Codex with a ChatGPT account'
                        ' returns HTTP 400 because the requested model is unsupported'
                        ' (default: disabled, env: ANTHPROXY_CODEX_UNSUPPORTED_MODEL_FALLBACK)')
    p.add_argument('--anthropic-home',
                   default=os.environ.get('ANTHROPIC_HOME', ''),
                   help='Path to Anthropic credential directory (default: ~/.anthropic,'
                        ' env: ANTHROPIC_HOME)')
    p.add_argument('--bedrock-home',
                   default=os.environ.get('BEDROCK_HOME', ''),
                   help='Path to Bedrock home directory for credential cache'
                        ' (default: ~/.bedrock, env: BEDROCK_HOME)')
    p.add_argument('--openrouter-api-key',
                   default=os.environ.get('OPENROUTER_API_KEY', ''),
                   help='OpenRouter API key (env: OPENROUTER_API_KEY)')
    p.add_argument('--local-base-url', dest='local_base_url',
                   default=os.environ.get('ANTHPROXY_LOCAL_BASE_URL', 'http://127.0.0.1:1235'),
                   help='Base URL for the local (LM Studio) backend'
                        ' (default: http://127.0.0.1:1235,'
                        ' env: ANTHPROXY_LOCAL_BASE_URL)')
    p.add_argument('--log-level',
                   default=os.environ.get('ANTHPROXY_LOG_LEVEL', 'INFO'),
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='Log level (default: INFO, env: ANTHPROXY_LOG_LEVEL)')
    p.add_argument('--no-prompt-translate', dest='no_prompt_translate',
                   action='store_true',
                   default=_env_bool('ANTHPROXY_NO_PROMPT_TRANSLATE', False),
                   help='Disable system-prompt and tool-name translation'
                        ' (env: ANTHPROXY_NO_PROMPT_TRANSLATE)')
    p.add_argument('--request-history-size', type=int,
                   default=int(os.environ.get('ANTHPROXY_REQUEST_HISTORY_SIZE', '5')),
                   help='Number of recent requests to keep in ring buffer'
                        ' (default: 5, env: ANTHPROXY_REQUEST_HISTORY_SIZE)')
    p.add_argument('--log-file',
                   default=os.environ.get('ANTHPROXY_LOG_FILE', '/tmp/anthproxy.log'),
                   help='Write log to this file at --log-level verbosity (default: /tmp/anthproxy.log, env: ANTHPROXY_LOG_FILE)')
    p.add_argument('--auto-backend', dest='auto_backend',
                   action=argparse.BooleanOptionalAction,
                   default=_env_bool('ANTHPROXY_AUTO_BACKEND', True),
                   help='Automatically select the best available backend (default: on;'
                        ' --no-auto-backend to disable, env: ANTHPROXY_AUTO_BACKEND=0)')
    p.add_argument('--auto-backend-mode', dest='auto_backend_mode',
                   default=os.environ.get('ANTHPROXY_AUTO_BACKEND_MODE', 'subscription'),
                   choices=['auto', 'subscription'],
                   help='Initial auto-selection routing mode at startup: "subscription" restricts'
                        ' selection to subscription backends (anthropic, codex, openrouter) and never falls'
                        ' back to bedrock; "auto" allows bedrock as a fallback. Overridable at'
                        ' runtime via proxy-set-backend (default: subscription,'
                        ' env: ANTHPROXY_AUTO_BACKEND_MODE)')
    p.add_argument('--auto-backend-interval', dest='auto_backend_interval', type=float,
                   default=float(os.environ.get('ANTHPROXY_AUTO_BACKEND_INTERVAL', '60')),
                   help='Seconds between selector ticks for token refresh and auto-backend'
                        ' re-evaluation; successful subscription usage lookups are cached'
                        ' separately for 5 minutes (default: 60, env: ANTHPROXY_AUTO_BACKEND_INTERVAL)')
    p.add_argument('--auto-backend-weekly-margin', dest='auto_backend_weekly_margin', type=float,
                   default=float(os.environ.get('ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN', '5')),
                   help='Hysteresis band in weekly-utilization percentage points: a backend must be'
                        ' at least this far below the current active backend before the selector'
                        ' switches away from a healthy available incumbent, preventing flapping'
                        ' (default: 5, env: ANTHPROXY_AUTO_BACKEND_WEEKLY_MARGIN)')
    p.add_argument('--auto-model-routing', dest='auto_model_routing',
                   action=argparse.BooleanOptionalAction,
                   default=_env_bool('ANTHPROXY_AUTO_MODEL_ROUTING', False),
                   help='Automatically route requests whose model is any non-empty string to a'
                        ' configured target. Routing failures preserve the original requested model.'
                        ' In classifier mode, the classifier call uses the model set by'
                        ' --auto-model-routing-classifier-model (default: off,'
                        ' env: ANTHPROXY_AUTO_MODEL_ROUTING=1)')
    p.add_argument('--auto-model-routing-classifier-model',
                   dest='auto_model_routing_classifier_model',
                   default=os.environ.get(
                       'ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFIER_MODEL', 'haiku'),
                   help='Model alias to use for the internal complexity-classifier call when'
                        ' --auto-model-routing is enabled (default: haiku,'
                        ' env: ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFIER_MODEL)')
    p.add_argument('--auto-model-routing-long-context-threshold',
                   dest='auto_model_routing_long_context_threshold', type=int,
                   default=int(os.environ.get(
                       'ANTHPROXY_AUTO_MODEL_ROUTING_LONG_CONTEXT_THRESHOLD', '150000')),
                   help='Estimated-input-token threshold at/above which --auto-model-routing'
                        ' deterministically forces the target configured by'
                        ' --auto-model-routing-long and injects the context-1m beta, bypassing the'
                        ' classifier. The estimate is a ~4-chars/token text heuristic over'
                        ' messages+system+tools plus a calibrated tool-use overhead; it still'
                        ' undercounts dense code (~1.5-1.8x) and images, so the default is'
                        ' intentionally below the 200K window for headroom. 0 disables the floor;'
                        ' setting --auto-model-routing-long=off also disables it. Only effective'
                        ' when --auto-model-routing is on (default: 150000,'
                        ' env: ANTHPROXY_AUTO_MODEL_ROUTING_LONG_CONTEXT_THRESHOLD)')
    p.add_argument('--auto-model-routing-affirmation-inherit',
                   dest='auto_model_routing_affirmation_inherit',
                   action=argparse.BooleanOptionalAction,
                   default=_env_bool('ANTHPROXY_AUTO_MODEL_ROUTING_AFFIRMATION_INHERIT', True),
                   help='When --auto-model-routing is on, treat a bare confirmation turn'
                        ' ("yes", "go ahead", "proceed") as a continuation: inherit the'
                        " conversation's established tier (or floor to sonnet when uncached)"
                        ' instead of classifying it as trivial→haiku and poisoning the session'
                        ' tier cache. When disabled, such turns are classified normally'
                        ' (default: on, env: ANTHPROXY_AUTO_MODEL_ROUTING_AFFIRMATION_INHERIT)')
    p.add_argument(
        '--auto-model-routing-classification',
        dest='auto_model_routing_classification',
        default=os.environ.get('ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFICATION', ''),
        help='Comma-separated label:model pairs overriding tier targets for '
             '--auto-model-routing. Valid labels: trivial, standard, deep. '
             'Unspecified labels keep their defaults (trivial→haiku, '
             'standard→sonnet, deep→opus). Example: standard:opus,deep:fable '
             '(env: ANTHPROXY_AUTO_MODEL_ROUTING_CLASSIFICATION)',
    )
    p.add_argument(
        '--auto-model-routing-long',
        dest='auto_model_routing_long',
        default=os.environ.get('ANTHPROXY_AUTO_MODEL_ROUTING_LONG', 'opus[1m]'),
        help='Model forced by the long-context size floor under --auto-model-routing '
             '(default: opus[1m]). Pass "off" to disable the floor entirely '
             'regardless of --auto-model-routing-long-context-threshold. '
             '(env: ANTHPROXY_AUTO_MODEL_ROUTING_LONG)',
    )
    p.add_argument('--auto-model-routing-confidence-bump',
                   dest='auto_model_routing_confidence_bump',
                   action=argparse.BooleanOptionalAction,
                   default=_env_bool('ANTHPROXY_AUTO_MODEL_ROUTING_CONFIDENCE_BUMP', False),
                   help='When enabled, the classifier uses a structured JSON output format '
                        'that includes a confidence score; turns classified below '
                        '--auto-model-routing-min-confidence are bumped to the next tier '
                        '(trivial→standard, standard→deep). Only effective when '
                        '--auto-model-routing is on. '
                        '(default: off, env: ANTHPROXY_AUTO_MODEL_ROUTING_CONFIDENCE_BUMP)')
    p.add_argument('--auto-model-routing-min-confidence',
                   dest='auto_model_routing_min_confidence', type=float,
                   default=float(os.environ.get(
                       'ANTHPROXY_AUTO_MODEL_ROUTING_MIN_CONFIDENCE', '0.0')),
                   help='Minimum confidence score (0.0–1.0) required before the '
                        'classifier\'s tier label is used as-is; turns below this threshold '
                        'are bumped up (trivial→standard, standard→deep). Only effective when '
                        '--auto-model-routing-confidence-bump is on. '
                        '(default: 0.0, env: ANTHPROXY_AUTO_MODEL_ROUTING_MIN_CONFIDENCE)')
    p.add_argument(
        '--auto-model-routing-mode',
        dest='auto_model_routing_mode',
        default=os.environ.get('ANTHPROXY_AUTO_MODEL_ROUTING_MODE', 'classifier'),
        choices=['classifier', 'rules', 'tag'],
        help='Classification mode for --auto-model-routing: "classifier" (default) calls a lightweight '
             'LLM classifier on the active backend; "rules" uses deterministic keyword rules with no LLM '
             'call; "tag" routes via the task name supplied by X-Anthproxy-Override: task:<name> against '
             '--auto-model-routing-task-tiers. Per-request override via '
             'X-Anthproxy-Override: route:<mode> takes precedence over this setting. '
             '(env: ANTHPROXY_AUTO_MODEL_ROUTING_MODE)',
    )
    p.add_argument(
        '--auto-model-routing-task-tiers',
        dest='auto_model_routing_task_tiers',
        default=os.environ.get('ANTHPROXY_AUTO_MODEL_ROUTING_TASK_TIERS', ''),
        help='JSON object mapping task names to model tier aliases for '
             '--auto-model-routing-mode=tag. Example: \'{"extraction":"haiku","analysis":"sonnet"}\'. '
             'Unknown task names fail-closed to the requested model. '
             '(env: ANTHPROXY_AUTO_MODEL_ROUTING_TASK_TIERS)',
    )
    p.add_argument('--sse-keepalive-interval', dest='sse_keepalive_interval', type=float,
                   default=float(os.environ.get('ANTHPROXY_SSE_KEEPALIVE_INTERVAL', '10.0')),
                   help='Seconds between SSE keepalive comment lines (": keepalive\\n\\n") sent to'
                        ' the client while waiting for the upstream first byte on streaming'
                        ' requests; 0 disables keepalive (default: 10.0,'
                        ' env: ANTHPROXY_SSE_KEEPALIVE_INTERVAL)')
    p.add_argument('--db-path', dest='db_path',
                   default=os.environ.get('ANTHPROXY_DB_PATH', None),
                   help='Path to SQLite DB for session tracing'
                        ' (default: ~/.anthproxy/anthproxy.db when --enable-ui is set,'
                        ' env: ANTHPROXY_DB_PATH)')
    p.add_argument('--enable-ui', dest='enable_ui',
                   action='store_true', default=False,
                   help='Enable admin API and web UI at /admin/* and /ui/*')
    p.add_argument('--codex-context-limit', dest='codex_context_limit', type=int,
                   default=int(os.environ.get('ANTHPROXY_CODEX_CONTEXT_LIMIT', '100000')),
                   help='Estimated-token ceiling for Codex requests; oldest messages are dropped'
                        ' when the conservative estimate exceeds this value. 0 disables truncation'
                        ' (default: 100000, env: ANTHPROXY_CODEX_CONTEXT_LIMIT)')

    args = p.parse_args(argv)

    args.codex_unsupported_model_fallback = (
        args.codex_unsupported_model_fallback or ''
    ).strip()
    args.auto_model_routing_classification = _parse_classification_str(
        args.auto_model_routing_classification, p
    )
    if not (args.auto_model_routing_long or '').strip():
        p.error('--auto-model-routing-long must be a non-empty string or "off"')
    # Clamp min_confidence to [0.0, 1.0] before constructing Config.
    args.auto_model_routing_min_confidence = max(
        0.0, min(1.0, args.auto_model_routing_min_confidence)
    )

    # Parse task-tiers JSON string into dict or None
    raw_task_tiers = (args.auto_model_routing_task_tiers or '').strip()
    if raw_task_tiers:
        try:
            tiers_parsed = json.loads(raw_task_tiers)
            if not isinstance(tiers_parsed, dict):
                p.error('--auto-model-routing-task-tiers must be a JSON object mapping task names to tier aliases')
            args.auto_model_routing_task_tiers = {str(k): str(v) for k, v in tiers_parsed.items()}
        except json.JSONDecodeError as exc:
            p.error(f'--auto-model-routing-task-tiers: invalid JSON: {exc}')
    else:
        args.auto_model_routing_task_tiers = None

    # Validate classifier model
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
    if not cfg.auto_model_routing_classifier_model.strip():
        p.error('auto model routing classifier model must be a non-empty string')
    if cfg.sse_keepalive_interval < 0:
        p.error('--sse-keepalive-interval must be >= 0')
    if cfg.auto_model_routing_long_context_threshold < 0:
        p.error('--auto-model-routing-long-context-threshold must be >= 0')
    if cfg.codex_context_limit < 0:
        p.error('--codex-context-limit must be >= 0')
    # Default db_path when enable_ui is set and no explicit path was given
    if cfg.enable_ui and cfg.db_path is None:
        cfg.db_path = str(Path.home() / '.anthproxy' / 'anthproxy.db')
    return cfg
