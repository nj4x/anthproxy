import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from .backends_registry import backend_names as _backend_names
from .backends_registry import set_enabled_backends as _set_enabled_backends
from .constants import VALID_BACKEND_MODES


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


def _parse_backends_str(
    raw: str | None, p: argparse.ArgumentParser, full_names: frozenset[str]
) -> frozenset[str] | None:
    """Parse the ``--backends`` allowlist. ``None`` input means no filter.

    Splits on commas, strips whitespace, drops empty tokens, de-duplicates
    (preserving first occurrence). Validates every token against *full_names*
    (the unfiltered discovered set). Calls ``p.error()`` on an unknown token
    or a resulting empty set — an allowlist with zero usable backends is
    always a configuration mistake, never a valid intent.
    """
    if raw is None:
        return None
    tokens: list[str] = []
    seen: set[str] = set()
    for tok in raw.split(','):
        tok = tok.strip()
        if not tok or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    if not tokens:
        p.error(
            '--backends: must name at least one backend, comma-separated '
            '(e.g. --backends anthropic,codex)'
        )
    unknown = [t for t in tokens if t not in full_names]
    if unknown:
        p.error(
            f'--backends: unknown backend(s) {unknown!r}; '
            f'valid backends: {sorted(full_names)}'
        )
    return frozenset(tokens)


def _apply_peer_gate(
    enabled: frozenset[str] | None,
    peer_base_url: str,
    p: argparse.ArgumentParser,
    full_names: frozenset[str],
) -> frozenset[str] | None:
    """Apply ADR-0021 §3's ``peer`` membership rule to the parsed allowlist.

    *peer_base_url* must already be stripped by the caller, so that "configured"
    means the same thing here as everywhere else that reads ``Config``.

    ``--peer-base-url`` is what makes ``peer`` a member. With no target and no
    allowlist, ``peer`` is filtered out and behaves as if not installed. With a
    target, it joins the default set implicitly — but only in the branch where
    the operator passed no ``--backends``; an allowlist is exhaustive. Naming
    ``peer`` in an allowlist with no target is a hard error rather than a silent
    narrowing, because the narrowed set feeds the default repair below.
    """
    if 'peer' not in full_names or peer_base_url:
        return enabled
    if enabled is None:
        return frozenset(full_names - {'peer'})
    if 'peer' in enabled:
        p.error(
            "--backends names 'peer' but --peer-base-url is unset; set "
            '--peer-base-url to the target anthproxy instance, or drop '
            "'peer' from --backends"
        )
    return enabled


def _resolve_home(home_override: str) -> str:
    """Resolve a home directory: explicit override > environment > default ~/.anthproxy."""
    if home_override and home_override.strip():
        return home_override.strip()
    return str(Path.home() / '.anthproxy')


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
    backends: tuple[str, ...] = ()   # Allowlist the operator stated; empty means none stated (peer is gated separately)
    log_level: str = 'INFO'
    no_prompt_translate: bool = False
    request_history_size: int = 5
    log_file: str = '/tmp/anthproxy.log'
    anthproxy_home: str = ''
    codex_home: str = ''
    codex_unsupported_model_fallback: str = ''
    bedrock_home: str = ''
    anthropic_home: str = ''
    openrouter_api_key: str = ''
    local_base_url: str = 'http://127.0.0.1:1235'
    peer_base_url: str = ''
    peer_api_key: str = ''
    auto_backend: bool = True
    auto_backend_mode: str = 'subscription'
    auto_backend_interval: float = 60.0
    auto_backend_weekly_margin: float = 5.0
    auto_backend_pace_delta: str = 'on'
    auto_backend_oauth_pace_deadband_pp: float = 3.0
    auto_model_routing: bool = False
    auto_model_routing_classifier_model: str = 'haiku'
    auto_model_routing_long_context_threshold: int = 150_000
    auto_model_routing_affirmation_inherit: bool = True
    auto_model_routing_classification: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(_DEFAULT_CLASSIFICATION)
    )
    auto_model_routing_long: str = 'off'
    auto_model_routing_confidence_bump: bool = False
    auto_model_routing_min_confidence: float = 0.0
    auto_model_routing_mode: str = 'classifier'
    auto_model_routing_task_tiers: dict[str, str] | None = None
    auto_model_routing_prior_response_summary_limit: int = 1000
    auto_model_routing_system_prompt_weight: float = 0.30
    auto_model_routing_user_prompt_weight: float = 0.70
    auto_model_routing_trivial_threshold: float = 38.0
    auto_model_routing_standard_threshold: float = 75.0
    auto_model_routing_system_prompt_cache_size: int = 256
    auto_model_routing_system_prompt_preview_limit: int = 500
    lock_requested_model: str = 'claude-sonnet-4-6'      # Model baseline lock for routing; 'off' disables
    sse_keepalive_interval: float = 10.0
    db_path: str | None = None   # Path to SQLite DB file; None disables DB recording
    stats_dir: str = ''           # Path to stats directory; empty uses default under anthproxy_home
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
    p.add_argument('--backend', default=None,
                   help='LLM backend (default: bedrock, env: ANTHPROXY_BACKEND). '
                        'Must be a member of the --backends allowlist if one is set; '
                        'an unchosen default is repaired to the first enabled backend.')
    p.add_argument('--backends', dest='backends',
                   default=os.environ.get('ANTHPROXY_BACKENDS'),
                   help='Comma-separated allowlist restricting which backends are '
                        'discoverable/selectable (e.g. --backends anthropic,codex). '
                        'Absent: all discovered backends are enabled except peer, '
                        'which --peer-base-url enables (default,'
                        ' env: ANTHPROXY_BACKENDS)')
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
    p.add_argument('--peer-base-url', dest='peer_base_url',
                   default=os.environ.get('ANTHPROXY_PEER_BASE_URL', ''),
                   help='Base URL of another anthproxy instance to dispatch to'
                        ' via the peer backend. Setting it is what enables the'
                        ' peer backend; when --backends is also passed, peer must'
                        ' still be listed there explicitly (default: unset,'
                        ' env: ANTHPROXY_PEER_BASE_URL)')
    p.add_argument('--peer-api-key', dest='peer_api_key',
                   default=os.environ.get('ANTHPROXY_PEER_API_KEY', ''),
                   help='Credential sent to the peer as X-Anthproxy-Peer-Key for a'
                        ' fronting access-control layer to consume; anthproxy itself'
                        ' never checks it (default: unset, env: ANTHPROXY_PEER_API_KEY)')
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
    p.add_argument('--anthproxy-home', dest='anthproxy_home',
                   default=os.environ.get('ANTHPROXY_HOME', ''),
                   help='Root directory for anthproxy config, state, and credentials'
                        ' (default: ~/.anthproxy, env: ANTHPROXY_HOME)')
    p.add_argument('--stats-dir', dest='stats_dir',
                   default=os.environ.get('ANTHPROXY_STATS_DIR', ''),
                   help='Directory for stats JSONL files (default: $ANTHPROXY_HOME/stats,'
                        ' env: ANTHPROXY_STATS_DIR)')
    p.add_argument('--auto-backend', dest='auto_backend',
                   action=argparse.BooleanOptionalAction,
                   default=_env_bool('ANTHPROXY_AUTO_BACKEND', True),
                   help='Automatically select the best available backend (default: on;'
                        ' --no-auto-backend to disable, env: ANTHPROXY_AUTO_BACKEND=0)')
    p.add_argument('--auto-backend-mode', dest='auto_backend_mode',
                   default=os.environ.get('ANTHPROXY_AUTO_BACKEND_MODE', 'subscription'),
                   choices=list(VALID_BACKEND_MODES),
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
    p.add_argument('--auto-backend-pace-delta', dest='auto_backend_pace_delta',
                   default=os.environ.get('ANTHPROXY_AUTO_BACKEND_PACE_DELTA', 'on'),
                   choices=['on', 'off'],
                   help='Rank backends by pace delta (burn%% minus elapsed%% of the quota'
                        ' window) instead of raw burn%%, so a backend ahead of calendar/'
                        ' time-to-reset pace loses to one behind pace regardless of window'
                        ' length. "off" reverts to the prior raw-%% comparison'
                        ' (default: on, env: ANTHPROXY_AUTO_BACKEND_PACE_DELTA)')
    p.add_argument('--auto-backend-oauth-pace-deadband-pp',
                   dest='auto_backend_oauth_pace_deadband_pp', type=float,
                   default=float(os.environ.get(
                       'ANTHPROXY_AUTO_BACKEND_OAUTH_PACE_DEADBAND_PP', '3')),
                   help='OAuth monthly pacing headroom in percentage points. Enterprise'
                        ' wins while its utilization is below UTC month elapsed plus this'
                        ' margin; once reached, an available personal subscription wins.'
                        ' Enterprise also wins when no personal subscription is confirmed'
                        ' available. Negative values are clamped to 0 (default: 3, env:'
                        ' ANTHPROXY_AUTO_BACKEND_OAUTH_PACE_DEADBAND_PP)')
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
        default=os.environ.get('ANTHPROXY_AUTO_MODEL_ROUTING_LONG', 'off'),
        help='Model forced by the long-context size floor under --auto-model-routing '
             '(default: off). Pass "opus[1m]" or other model to enable the floor. '
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
    p.add_argument(
        '--auto-model-routing-prior-response-summary-limit',
        dest='auto_model_routing_prior_response_summary_limit', type=int,
        default=int(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_PRIOR_RESPONSE_SUMMARY_LIMIT', '1000')),
        help='Maximum characters of the prior assistant response sent to the classifier '
             'during affirmation enrichment (30/70 head/tail split). '
             'Valid range: [50, 32000]. '
             '(default: 1000, env: ANTHPROXY_AUTO_MODEL_ROUTING_PRIOR_RESPONSE_SUMMARY_LIMIT)',
    )
    p.add_argument(
        '--auto-model-routing-system-prompt-weight',
        dest='auto_model_routing_system_prompt_weight', type=float,
        default=float(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_SYSTEM_PROMPT_WEIGHT', '0.20')),
        help='Weight applied to the system-prompt tier score in the weighted blend '
             '(must sum to 1.0 with --auto-model-routing-user-prompt-weight; both > 0). '
             '(default: 0.30, env: ANTHPROXY_AUTO_MODEL_ROUTING_SYSTEM_PROMPT_WEIGHT)',
    )
    p.add_argument(
        '--auto-model-routing-user-prompt-weight',
        dest='auto_model_routing_user_prompt_weight', type=float,
        default=float(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_USER_PROMPT_WEIGHT', '0.80')),
        help='Weight applied to the user-prompt tier score in the weighted blend '
             '(must sum to 1.0 with --auto-model-routing-system-prompt-weight; both > 0). '
             '(default: 0.70, env: ANTHPROXY_AUTO_MODEL_ROUTING_USER_PROMPT_WEIGHT)',
    )
    p.add_argument(
        '--auto-model-routing-trivial-threshold',
        dest='auto_model_routing_trivial_threshold', type=float,
        default=float(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_TRIVIAL_THRESHOLD', '30')),
        help='Weighted-score threshold below which the blended tier is "trivial". '
             'Must be strictly less than --auto-model-routing-standard-threshold. '
             'On the 0-100 numeric scale: default 35. '
             '(default: 38, env: ANTHPROXY_AUTO_MODEL_ROUTING_TRIVIAL_THRESHOLD)',
    )
    p.add_argument(
        '--auto-model-routing-standard-threshold',
        dest='auto_model_routing_standard_threshold', type=float,
        default=float(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_STANDARD_THRESHOLD', '60')),
        help='Weighted-score threshold at/above which the blended tier is "deep"; '
             'between trivial_threshold and this value is "standard". '
             'Must be strictly greater than --auto-model-routing-trivial-threshold. '
             'On the 0-100 numeric scale: default 65. '
             '(default: 65, env: ANTHPROXY_AUTO_MODEL_ROUTING_STANDARD_THRESHOLD)',
    )
    p.add_argument(
        '--auto-model-routing-system-prompt-cache-size',
        dest='auto_model_routing_system_prompt_cache_size', type=int,
        default=int(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_SYSTEM_PROMPT_CACHE_SIZE', '256')),
        help='Maximum number of system-prompt SHA256 → tier-score entries in the '
             'in-memory LRU cache (evicts oldest on overflow). Must be >= 1. '
             '(default: 256, env: ANTHPROXY_AUTO_MODEL_ROUTING_SYSTEM_PROMPT_CACHE_SIZE)',
    )
    p.add_argument(
        '--auto-model-routing-system-prompt-preview-limit',
        dest='auto_model_routing_system_prompt_preview_limit', type=int,
        default=int(os.environ.get(
            'ANTHPROXY_AUTO_MODEL_ROUTING_SYSTEM_PROMPT_PREVIEW_LIMIT', '500')),
        help='Maximum characters of the system prompt sent to the system-prompt '
             'classifier (head-capped). Must be >= 1. '
             '(default: 500, env: ANTHPROXY_AUTO_MODEL_ROUTING_SYSTEM_PROMPT_PREVIEW_LIMIT)',
    )
    p.add_argument('--lock-requested-model', dest='lock_requested_model',
                   default=os.environ.get('ANTHPROXY_LOCK_REQUESTED_MODEL', 'claude-sonnet-4-6'),
                   help='Override the incoming request model with a fixed baseline before'
                        ' auto-routing fires. The classifier still runs and routes relative'
                        ' to this baseline (trivial→haiku, deep→opus). "off" disables the'
                        ' lock and passes the client\'s model through unchanged (default: claude-sonnet-4-6,'
                        ' env: ANTHPROXY_LOCK_REQUESTED_MODEL)')
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

    # --backends: validate against the full discovered set, then install the
    # filter before any subsequent backend_names() call. This is a required
    # ordering — see ADR-0020 §4.
    full_backend_names = frozenset(_backend_names())
    enabled = _parse_backends_str(args.backends, p, full_backend_names)
    args.backends = tuple(sorted(enabled)) if enabled is not None else ()
    args.peer_base_url = (args.peer_base_url or '').strip()
    gated = _apply_peer_gate(enabled, args.peer_base_url, p, full_backend_names)
    _set_enabled_backends(gated)

    # --backend: distinguish an explicit choice (CLI flag or env var) from the
    # unset packaged default. An explicit value outside the enabled set is a
    # hard error; an unchosen default is silently repaired (ADR-0020 §5, §6).
    backend_env = os.environ.get('ANTHPROXY_BACKEND')
    backend_explicit = args.backend is not None or bool(backend_env)
    if args.backend is None:
        args.backend = backend_env or 'bedrock'
    filtered_backend_names = _backend_names()
    if args.backend not in filtered_backend_names:
        if backend_explicit:
            hint = ''
            if args.backend == 'peer' and not args.peer_base_url:
                hint = '; --peer-base-url is unset, which is what withholds peer'
            p.error(
                f'--backend {args.backend!r} is not in the enabled backend set '
                f'{list(filtered_backend_names)}; pass --backends to include it '
                f'or choose a different --backend{hint}'
            )
        if not filtered_backend_names:
            p.error('--backends: resulting enabled backend set is empty')
        repaired = filtered_backend_names[0]
        logger.warning('Backend default repaired: %s -> %s', args.backend, repaired)
        args.backend = repaired

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

    # Normalise lock_requested_model
    args.lock_requested_model = (args.lock_requested_model or '').strip() or 'off'

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
    prior_limit = cfg.auto_model_routing_prior_response_summary_limit
    if prior_limit < 50 or prior_limit > 32_000:
        raise ValueError(
            f'auto_model_routing_prior_response_summary_limit must be in [50, 32000], '
            f'got {prior_limit}'
        )
    # Weighted blend validation (ADR 0010)
    sys_w = cfg.auto_model_routing_system_prompt_weight
    usr_w = cfg.auto_model_routing_user_prompt_weight
    if abs(sys_w + usr_w - 1.0) >= 1e-9:
        raise ValueError(
            f'auto_model_routing_system_prompt_weight + auto_model_routing_user_prompt_weight '
            f'must equal 1.0, got {sys_w} + {usr_w} = {sys_w + usr_w}'
        )
    if sys_w <= 0:
        raise ValueError(
            f'auto_model_routing_system_prompt_weight must be > 0, got {sys_w}'
        )
    if usr_w <= 0:
        raise ValueError(
            f'auto_model_routing_user_prompt_weight must be > 0, got {usr_w}'
        )
    trivial_t = cfg.auto_model_routing_trivial_threshold
    standard_t = cfg.auto_model_routing_standard_threshold
    if trivial_t >= standard_t:
        raise ValueError(
            f'auto_model_routing_trivial_threshold must be < auto_model_routing_standard_threshold, '
            f'got {trivial_t} >= {standard_t}'
        )
    # Warn if thresholds look like old 0–2 scale values (e.g. 0.75/1.50). On the
    # new 0–100 scale, any threshold ≤ 2 would route almost everything as 'deep'.
    if trivial_t <= 2.0 or standard_t <= 2.0:
        logger.warning(
            'auto-model-routing thresholds look like un-migrated 0–2 scale values '
            '(trivial=%s standard=%s). On the new 0–100 scale this routes almost '
            'everything as "deep". Update to e.g. --auto-model-routing-trivial-threshold 38 '
            '--auto-model-routing-standard-threshold 75.',
            trivial_t, standard_t,
        )
    if cfg.auto_model_routing_system_prompt_cache_size < 1:
        raise ValueError(
            f'auto_model_routing_system_prompt_cache_size must be >= 1, '
            f'got {cfg.auto_model_routing_system_prompt_cache_size}'
        )
    if cfg.auto_model_routing_system_prompt_preview_limit < 1:
        raise ValueError(
            f'auto_model_routing_system_prompt_preview_limit must be >= 1, '
            f'got {cfg.auto_model_routing_system_prompt_preview_limit}'
        )
    # Resolve anthproxy_home and derive paths from it
    cfg.anthproxy_home = _resolve_home(cfg.anthproxy_home)

    # Default stats_dir to $ANTHPROXY_HOME/stats if not explicitly set
    if not cfg.stats_dir or not cfg.stats_dir.strip():
        cfg.stats_dir = str(Path(cfg.anthproxy_home) / 'stats')

    # Default db_path when enable_ui is set and no explicit path was given
    if cfg.enable_ui and cfg.db_path is None:
        cfg.db_path = str(Path(cfg.anthproxy_home) / 'anthproxy.db')

    return cfg
