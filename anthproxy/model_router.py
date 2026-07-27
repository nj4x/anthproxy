"""LLM-based request complexity classifier and model-tier router.

When ``auto_model_routing`` is enabled in the server config, incoming requests
are classified by a lightweight classifier call on the already-selected backend
and then rewritten to one of three tiers before the main backend dispatch:

  trivial  →  haiku   (simple lookup, formatting, tiny edit)
  standard →  sonnet  (normal coding/explanation/debug)
  deep     →  opus    (architecture, large refactor, complex reasoning)

Any non-empty string model value is eligible for routing — ``sonnet``,
``opus``, ``haiku``, ``fable``, full Anthropic IDs, Bedrock/Codex native
IDs, ARNs, inference-profile IDs, and 1m variants are all rewritten to the
corresponding short tier alias.  Missing, non-string, or whitespace-only model
values are forwarded unchanged.

The classifier call uses a synthetic, bounded summary of the request — never
the full system prompt, tool schemas, provider metadata, headers, or history.
Any classifier failure keeps the model as the original requested model
(fail-closed).

Long-context size floor
-----------------------
Before the classifier runs, a deterministic size floor checks an in-process
token estimate of the whole request (``estimate_input_tokens``), combined with
two session-aware signals supplied by the handler: the last response's measured
total context size and a per-session actual/estimate calibration ratio.  The
floor fires when ``max(round(estimate * ratio), session_context_tokens)`` meets
or exceeds ``config.auto_model_routing_long_context_threshold`` (0 disables); the
model is then forced to the long-context opus tier (``opus[1m]``) and a
``context-1m`` beta is injected, so a request near the 200K window is served by
the only model that can hold it.  The floor outranks the classifier and the
walk-back/session cache, and applies even to text-less continuation turns.  The
size estimate is a local computation and is never sent to the classifier, so the
bounded classifier-input invariant is preserved.

Internal sentinel key
---------------------
Classifier payloads carry ``_anthproxy_internal_classifier = True``.  Backends
and mappers must strip this key before sending to providers.  ``route_model()``
no-ops immediately on any payload that already carries the sentinel to prevent
recursive classification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .mapper import estimate_input_tokens
from .model_tier import model_tier_rank
from .request_text import (
    _TRANSCRIPT_FALLBACK_LIMIT,
    is_short_affirmation,
    is_title_generation,
    last_transcript_user_turn,
    strip_reminders,
)

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ELIGIBLE_MODEL = 'sonnet'
_SENTINEL_KEY = '_anthproxy_internal_classifier'

# Tiers routed to when the classifier fires
_TIER_TRIVIAL: Literal['haiku'] = 'haiku'  # superseded by config.auto_model_routing_classification; kept as default-value documentation
_TIER_STANDARD: Literal['sonnet'] = 'sonnet'  # superseded by config.auto_model_routing_classification; kept as default-value documentation
_TIER_DEEP: Literal['opus'] = 'opus'  # superseded by config.auto_model_routing_classification; kept as default-value documentation

# Long-context deep tier forced by the size floor (see route_model).  Resolves to
# the 1m-window opus on backends that have one (anthropic via the context-1m beta
# below; bedrock via the distinct ":1m" inference profile) and collapses to plain
# opus on backends without a 1m variant (e.g. codex), which is benign.
_TIER_DEEP_LONG_CONTEXT: Literal['opus[1m]'] = 'opus[1m]'  # superseded by config.auto_model_routing_long; kept for reference only

# Tier ranks (canonical, from model_tier.py): haiku=0, sonnet=1, opus=2, fable=3.
# Unknown / non-tier model IDs rank None → fail-open (no cap applied).
def _cap_cached_tier(
    cached_tier: str,
    requested: str,
    label_map: dict[str, str] | None = None,
) -> str:
    """Clamp a replayed cache tier so it never UPGRADES the requested model.

    When ``label_map`` is provided (the runtime ``config.auto_model_routing_classification``
    dict, e.g. ``{'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}``), a reverse
    map ``{model_str: rank}`` is built so custom targets like ``'fable'`` (mapped to the
    ``deep`` slot) are ranked correctly.  Exact lookup takes precedence for the *cached* tier;
    substring match is the fallback.  The *requested* model always uses substring match only —
    it is arbitrary client input and not necessarily one of the configured targets.

    Without ``label_map`` (or when it is ``None``), falls back to the existing
    ``_tier_rank()`` substring-match-only behaviour.

    Upgrade-only, asymmetric by design: a cached *lower* tier may still downgrade a
    higher-requested continuation.  Fails open when either rank is unknown.
    """
    _LABEL_ORDER = {'trivial': 0, 'standard': 1, 'deep': 2}
    if label_map is not None:
        reverse: dict[str, int] = {
            model: _LABEL_ORDER[label]
            for label, model in label_map.items()
            if label in _LABEL_ORDER
        }
        cached_rank: int | None = reverse.get(cached_tier, model_tier_rank(cached_tier))
        requested_rank = model_tier_rank(requested)
    else:
        cached_rank = model_tier_rank(cached_tier)
        requested_rank = model_tier_rank(requested)

    if cached_rank is not None and requested_rank is not None and cached_rank > requested_rank:
        return requested
    return cached_tier

# Long-context (1m) beta.  Injected by the size floor so the forced opus[1m] tier
# actually receives a 1m window on the Anthropic subscription even when the client
# never sent the beta.  Prefix-matched for dedupe so any dated client token counts
# as already-present; mirrors the anthropic mapper's _LONG_CONTEXT_BETA_PREFIX gate
# ("context-1m"), which keeps this beta only for resolved-opus models.
_LONG_CONTEXT_BETA_PREFIX = 'context-1m'
_LONG_CONTEXT_BETA = 'context-1m-2025-08-07'

# Per-session calibration ratio bounds for the size floor.  The handler pairs a
# response's measured input context with the estimate we computed for that
# request to form actual/estimate; the floor multiplies the next estimate by it
# to correct the estimator's code/tool undercount.  Only trusted when the
# baseline estimate is large enough that fixed overhead does not dominate, and
# clamped to [1.0, _RATIO_MAX] so it only ever inflates (never shrinks a
# conservative prose estimate below the raw value).
_RATIO_MIN_BASELINE = 2_000
_RATIO_MAX = 3.0

# Maximum final-user text length passed to the classifier (characters).
_TEXT_LIMIT = 4_000

# Head fraction for prior-response 30/70 truncation (30% head, 70% tail).
_PRIOR_RESPONSE_HEAD_FRAC = 0.30

# When text exceeds _TEXT_LIMIT, keep the first _TEXT_LIMIT_HEAD_FRAC fraction
# (opening context) and the last (1 - _TEXT_LIMIT_HEAD_FRAC) fraction (most-
# recent intent), joined by _TRUNCATION_MARKER.  The tail bias reflects that
# classifier accuracy depends more on what the user is asking NOW than on
# session preamble that appears at the head.
_TEXT_LIMIT_HEAD_FRAC = 0.20
_TRUNCATION_MARKER = '\n...[truncated]...\n'

# When the final message is text-less, recovered prior-turn text (transcript
# fallback or message walk-back) is tail-capped to this many trailing chars
# before _TEXT_LIMIT, so a large prior turn cannot bias routing toward a higher
# tier.  Shared with the transcript fallback's bound in request_text.py.
_WALKBACK_TAIL_LIMIT = _TRANSCRIPT_FALLBACK_LIMIT

# Maximum final-user text length echoed into the INFO classification log line.
_LOG_PROMPT_LIMIT = 200


# Numeric tier scores for the weighted blend (ADR 0010).
# Maps the three classifier labels to scalar scores; missing labels get 1.0 (standard).
_LABEL_SCORES: dict[str, float] = {'trivial': 0.0, 'standard': 1.0, 'deep': 2.0}

# Specialized system prompt for system-prompt tier inference (ADR 0012).
# Distinct from _CLASSIFIER_SYSTEM (which classifies user prompts) — teaches the
# classifier to judge role complexity, not task complexity.
_CLASSIFIER_SYSTEM_PROMPT_TIER = (
    'You are a one-word complexity classifier for AI agent system prompts. '
    'Read the system prompt preview and reply with EXACTLY one of these three labels and nothing else:\n'
    '\n'
    'trivial  — a narrow specialist role with low cognitive complexity: file browsing, '
    'simple retrieval, formatting, mechanical data tasks, or any role that mostly reads '
    'and summarises with minimal reasoning.\n'
    '\n'
    'standard — a general-purpose assistant, software engineer, or analyst role that '
    'handles a mix of coding, explanation, debugging, and planning tasks.\n'
    '\n'
    'deep     — a research architect, system designer, or expert-level specialist role '
    'that implies complex reasoning, architectural decisions, multi-domain synthesis, '
    'or security-sensitive work.\n'
    '\n'
    'Judge complexity from the ROLE implied by the system prompt, not from any specific '
    'user task. A "file search specialist" is trivial by role even for a hard task; '
    'a "research architect" is deep by role even for a simple task.\n'
    '\n'
    'Examples:\n'
    '"You are a file search and retrieval assistant." → trivial\n'
    '"You are a helpful assistant." → standard\n'
    '"You are Claude, an AI assistant." → standard\n'
    '"You are a senior software engineer helping with code review." → standard\n'
    '"You are a security researcher and penetration testing expert." → deep\n'
    '"You are a research architect responsible for designing distributed systems." → deep\n'
    '\n'
    'Reply with ONLY the single label word. No punctuation, no explanation.'
)

# Module-level LRU cache: system_prompt_sha256 → (tier_label, score_float).
# Thread-safe: _sys_prompt_cache_lock is held only for short dict operations,
# never across classifier network calls (satisfies concurrency.md invariant).
_sys_prompt_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
_sys_prompt_cache_lock = threading.Lock()

# In-flight sentinel for concurrent affirmation mitigation.  When two
# simultaneous affirmation turns on the same session both read an empty cache,
# only the first one that acquires ctx_key in this set calls the classifier;
# the second uses the floor tier instead.  The lock is held only while
# checking/inserting/removing; never held across the classifier network call.
_affirmation_inflight_lock = threading.Lock()
_affirmation_inflight: set[str] = set()


def _prompt_log_preview(text: str) -> str:
    """One-line, bounded preview of the classifier prompt for INFO logging."""
    preview = text.replace('\r', ' ').replace('\n', ' ')
    if len(preview) > _LOG_PROMPT_LIMIT:
        return preview[:_LOG_PROMPT_LIMIT] + '…'
    return preview

# Classifier response constraints
_CLASSIFIER_MAX_TOKENS = 4
_CLASSIFIER_TEMPERATURE = 0.0

_CLASSIFIER_SYSTEM = (
    'You are a one-word complexity classifier for AI coding assistant requests. '
    'Read the JSON summary of the user request and reply with EXACTLY one of '
    'these three labels and nothing else:\n'
    '\n'
    'trivial  — greeting, acknowledgement, simple lookup, short factual answer, '
    'formatting, tiny edit, or low-risk mechanical task that requires almost no '
    'reasoning.\n'
    '\n'
    'standard — normal coding, explanation, debugging, planning, or multi-step '
    'work that does not require extended reasoning or unusual depth.\n'
    '\n'
    'deep     — architecture, ambiguous high-stakes design, advanced mathematics, '
    'security-sensitive reasoning, '
    'or a task likely to need extended chain-of-thought thinking.\n'
    '\n'
    'Judge complexity ONLY from the user\'s intent in "final_user_text". The '
    'other fields (message counts, tool counts, conversation length) describe '
    'the harness setup and prior context, NOT the difficulty of THIS request — '
    'never escalate the label because of them. A short, simple request is '
    'trivial even inside a long session with many tools.\n'
    '\n'
    'Floor: any request that asks to plan, design, or outline an approach or '
    'implementation is AT LEAST "standard", never "trivial", even when the '
    'wording looks small or simple.\n'
    '\n'
    'Examples (final_user_text → label):\n'
    '"hi" → trivial\n'
    '"thanks, that works" → trivial\n'
    '"fix the typo in the README" → trivial\n'
    '"read the first 20 lines of config.py" → trivial\n'
    '"what is the signature of parse_config()?" → trivial\n'
    '"implement auth middleware for the Express router" → standard\n'
    '"plan how to add a logout button" → standard\n'
    '"why does this endpoint return 500?" → standard\n'
    '"refactor the payment module to use the strategy pattern" → standard\n'
    '"update 5 test files to use the new mock factory" → standard\n'
    '"add cursor-based pagination to the list users API" → standard\n'
    '"redesign the auth layer for SSO and multi-tenant isolation" → deep\n'
    '\n'
    'Reply with ONLY the single label word. No punctuation, no explanation.'
)

# Labels the classifier is allowed to produce
_VALID_LABELS = frozenset({'trivial', 'standard', 'deep'})

# ---------------------------------------------------------------------------
# Confidence-bump JSON classifier prompt (separate path, activated by config)
# ---------------------------------------------------------------------------

# Used ONLY when config.auto_model_routing_confidence_bump is True.  The
# existing _CLASSIFIER_SYSTEM / parse_classifier_label pair is never modified.
_CLASSIFIER_SYSTEM_JSON = (
    'You are a complexity classifier for AI coding assistant requests. '
    'Read the JSON summary of the user request and reply ONLY with valid JSON:\n'
    '{"label":"trivial"|"standard"|"deep","confidence":0.0-1.0}\n'
    '\n'
    'trivial  — greeting, acknowledgement, simple lookup, short factual answer, '
    'formatting, tiny edit, or low-risk mechanical task that requires almost no '
    'reasoning.\n'
    '\n'
    'standard — normal coding, explanation, debugging, planning, or multi-step '
    'work that does not require extended reasoning or unusual depth.\n'
    '\n'
    'deep     — architecture, ambiguous high-stakes design, advanced mathematics, '
    'security-sensitive reasoning, '
    'or a task likely to need extended chain-of-thought thinking.\n'
    '\n'
    'Judge complexity ONLY from the user\'s intent in "final_user_text". The '
    'other fields (message counts, tool counts, conversation length) describe '
    'the harness setup and prior context, NOT the difficulty of THIS request — '
    'never escalate the label because of them. A short, simple request is '
    'trivial even inside a long session with many tools.\n'
    '\n'
    'Floor: any request that asks to plan, design, or outline an approach or '
    'implementation is AT LEAST "standard", never "trivial", even when the '
    'wording looks small or simple.\n'
    '\n'
    'confidence must reflect how certain you are (0.0=uncertain, 1.0=certain).\n'
    '\n'
    '"implement auth middleware for the Express router" → standard\n'
    '"refactor the payment module to use the strategy pattern" → standard\n'
    '"update 5 test files to use the new mock factory" → standard\n'
    '"read the first 20 lines of config.py" → trivial\n'
    '"what is the signature of parse_config()?" → trivial\n'
    '\n'
    'Reply with ONLY the JSON object. No other text.'
)

# Larger token budget for the JSON-format response.  The JSON payload
# {"label":"standard","confidence":0.87} is ~14 tokens; 40 gives ample headroom.
_CLASSIFIER_MAX_TOKENS_JSON = 40

# Appended to _CLASSIFIER_SYSTEM_JSON when the classifier payload includes
# prior_response_summary so the classifier knows to weight that context.
_CLASSIFIER_SYSTEM_JSON_PRIOR_SUFFIX = (
    '\n\nWhen prior_response_summary is provided, use it as additional context '
    'about the ongoing task to inform your complexity assessment.'
)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

# Stable reason codes for logging / telemetry (never shown to users)
ReasonCode = Literal[
    'disabled',
    'model_not_eligible',
    'malformed_payload',
    'missing_final_user_text',
    'override_no_classifier',
    'size_forced_long_context',
    'session_cached_tier',
    'session_cached_tier_capped',
    'session_cached_walkback',
    'session_cached_walkback_capped',
    'session_cached_walkback_tool_result',
    'affirmation_inherited',
    'affirmation_floored_standard',  # retained for old stats rows; retired from affirmation code path
    'affirmation_classified',        # new: no cache, classifier called and succeeded
    'affirmation_classifier_failed', # new: no cache, classifier failed or no prior text
    'rule_title_generation',
    'classifier_trivial',
    'classifier_standard',
    'classifier_deep',
    'classifier_trivial_bumped',
    'classifier_standard_bumped',
    'classifier_failed',
    'classifier_invalid',
    # Rules-mode reason codes
    'classifier_rules_trivial',
    'classifier_rules_standard',
    'classifier_rules_deep',
    'rules_no_signal',
    # Tag-mode reason codes
    'task_tag_routed',
    'no_task_tag',
    'unknown_task_tag',
]


@dataclass(frozen=True)
class ModelRoutingDecision:
    requested_model: str
    routed_model: str
    classification: str | None  # raw label from classifier, or None
    applied: bool               # True if the model was actually rewritten
    reason_code: ReasonCode
    # Raw per-request size estimate (chars/4 + tool overhead) computed for the
    # long-context floor; 0 on early returns that short-circuit before it.  The
    # handler pairs it with the response's measured input context to refresh the
    # per-session calibration ratio.
    estimated_input_tokens: int = 0
    # Telemetry-only fields (Phase 1) — all have defaults so no existing callers
    # or dataclasses.replace() call-sites need updating.  Never used for routing
    # decisions; read only by the handler log line and external consumers.
    predicted_input_tokens: int = 0      # max(corrected_est, session_floor); 0 when floor inactive
    session_context_tokens: int = 0      # last-response measured context tokens fed to the floor
    session_estimate_ratio: float = 1.0  # per-session calibration ratio at decision time
    classifier_mode: str = 'classifier'  # 'classifier' | 'walkback_cache' | 'affirmation'
    classifier_confidence: float | None = None  # reserved; None until confidence is produced
    tier_bumped: bool = False            # True when confidence_bump mode promoted the tier (trivial→standard or standard→deep); False on all other paths
    task_tag: str | None = None          # reserved; None until task-tag routing is added
    classifier_input_tokens: int = 0   # estimated tokens sent to classifier; 0 when no classifier call made
    classifier_output_tokens: int = 0  # estimated tokens returned by classifier; 0 when no classifier call made
    # Classifier transparency fields — populated only when an actual LLM classifier
    # call succeeds (reason_code in classifier_trivial/standard/deep and variants).
    # None on all other paths (size floor, affirmation, walk-back, disabled, rules,
    # tag, failed, invalid, override_no_classifier).
    classifier_model: str | None = None        # model ID used for the classifier call
    classifier_summary_json: str | None = None # bounded JSON sent to the classifier
    classifier_raw_response: str | None = None # full concatenated text from classifier response blocks
    classifier_format: str | None = None       # 'standard' or 'json' response format
    # Uncapped resolved tier for the affirmation_classified path only.
    # The handler writes this (not routed_model) to the tier cache so subsequent
    # turns can apply their own cap.  None on all other paths.
    cache_tier: str | None = None
    # ADR 0010/0011: weighted system-prompt + user-prompt blend fields.
    # All four numeric fields are None when routing is disabled, when a cached
    # tier is inherited (affirmation_inherited path), or when the blend is not
    # applied (rules/tag mode, size floor, early returns).
    # system_prompt_classification_failed is True only when classification was
    # attempted and failed; False in all other cases (success, no-system-prompt,
    # disabled, cached).
    system_prompt_tier: str | None = None
    system_prompt_score: float | None = None
    user_prompt_score: float | None = None
    routing_weighted_score: float | None = None
    system_prompt_classification_failed: bool = False


@dataclass(frozen=True)
class RoutingSummary:
    """Bounded, provider-agnostic summary passed to the classifier.

    Fields here must never include system prompts, tool schemas/names/descs,
    full prior history text, provider metadata, API keys, or headers.
    """
    final_user_text: str        # stripped, truncated
    text_truncated: bool
    total_messages: int
    prior_user_messages: int
    prior_assistant_messages: int
    tool_use_count: int         # across all message content blocks
    tool_result_count: int
    final_non_text_blocks: int  # blocks in the final message that aren't text
    has_images: bool
    recovered_via_walkback: bool = False  # True if final_user_text came from walk-back over prior messages (not the final message itself); used by handlers for cache-first logic; routing-internal only — excluded from to_classifier_json()
    final_is_tool_result_only: bool = False  # True iff the final user message contains ONLY tool_result blocks; routing-internal only — excluded from to_classifier_json(). Used to bypass the no-upgrade cap for agentic continuations.
    # True iff the FINAL message's own text is a bare affirmation ("yes",
    # "proceed"); routing-internal continuation signal. Both is_short_affirmation
    # and recovered_via_walkback are excluded from to_classifier_json(): both are
    # routing-internal signals, not classifier complexity signals.
    is_short_affirmation: bool = False
    # SHA-256 of payload['system'] (or None when absent); used as the LRU cache
    # key for system-prompt tier classification (ADR 0010/0012).  Not a prompt
    # content field — just a stable hash for cache lookup and audit queries.
    # Excluded from to_classifier_json() (never sent to the user-prompt classifier).
    system_prompt_sha256: str | None = None

    def to_classifier_json(self, prior_response_summary: str | None = None) -> str:
        d: dict = {
            'final_user_text': self.final_user_text,
            'text_truncated': self.text_truncated,
            'total_messages': self.total_messages,
            'prior_user_messages': self.prior_user_messages,
            'prior_assistant_messages': self.prior_assistant_messages,
            'tool_use_count': self.tool_use_count,
            'tool_result_count': self.tool_result_count,
            'final_non_text_blocks': self.final_non_text_blocks,
            'has_images': self.has_images,
        }
        if prior_response_summary is not None:
            d['prior_response_summary'] = prior_response_summary
        return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def is_model_auto_routable(model: object) -> bool:
    """Return True iff ``model`` is a non-empty string.

    Any non-empty string model value is eligible for auto routing — including
    short aliases (``sonnet``, ``opus``, ``haiku``, ``fable``), full Anthropic
    IDs, backend-native IDs, ARNs, and 1m variants.  Non-string, empty, or
    whitespace-only values are ineligible and kept unchanged.
    """
    return isinstance(model, str) and bool(model.strip())


def _ensure_long_context_beta(payload: dict) -> None:
    """Ensure ``payload['_anthropic_beta']`` carries a long-context (1m) beta.

    Used by the size floor so the forced ``opus[1m]`` tier receives a real 1m
    window on the Anthropic subscription even when the client never sent the
    beta.  No-op if any ``context-1m*`` token is already present (dedupe by
    prefix, tolerant of dated revisions).  Copy-on-write: assigns a fresh list
    rather than mutating one the caller may share.  Harmless for backends that
    ignore ``_anthropic_beta`` (bedrock/codex/local).
    """
    raw = payload.get('_anthropic_beta')
    existing = [b for b in raw if isinstance(b, str)] if isinstance(raw, list) else []
    if any(b.startswith(_LONG_CONTEXT_BETA_PREFIX) for b in existing):
        return
    payload['_anthropic_beta'] = existing + [_LONG_CONTEXT_BETA]


def calibrated_ratio(
    measured_input_tokens: int, route_estimate: int, prior_ratio: float,
) -> float:
    """Return the per-session actual/estimate calibration ratio for the size floor.

    ``measured_input_tokens`` is the response's measured input context
    (``input + cache_read + cache_creation``, excluding output); ``route_estimate``
    is the raw estimate we computed for that same request.  When the baseline is
    too small to be meaningful (``route_estimate < _RATIO_MIN_BASELINE``) the prior
    ratio is kept, so a tiny turn cannot inject a noisy ratio dominated by fixed
    overhead.  The result is clamped to ``[1.0, _RATIO_MAX]`` — it only ever
    inflates the estimate, never shrinks an already-conservative one.
    """
    if route_estimate < _RATIO_MIN_BASELINE or measured_input_tokens <= 0:
        return _clamp_ratio(prior_ratio)
    return _clamp_ratio(measured_input_tokens / route_estimate)


def _clamp_ratio(ratio: float) -> float:
    if ratio < 1.0:
        return 1.0
    if ratio > _RATIO_MAX:
        return _RATIO_MAX
    return ratio


# ---------------------------------------------------------------------------
# Routing summary extraction
# ---------------------------------------------------------------------------

def _extract_user_text(content) -> tuple[str, str, int, bool] | None:
    """Extract text from one message's ``content``.

    Returns ``(stripped_text, raw_text, non_text_blocks, has_images)`` or
    ``None`` if the content is structurally malformed (not a str/list, a
    non-dict block, or a non-string ``text`` value).  ``stripped_text`` has
    Claude Code wrapper blocks removed via ``strip_reminders``; ``raw_text``
    preserves the pre-strip text for the transcript fallback.

    Privacy contract: reads only ``text``/``image`` block types — never tool
    schemas, tool names/descriptions, metadata, or provider keys.
    """
    if isinstance(content, str):
        return strip_reminders(content), content, 0, False
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    raw_parts: list[str] = []
    non_text = 0
    has_images = False
    for block in content:
        if not isinstance(block, dict):
            return None
        btype = block.get('type')
        if btype == 'text':
            t = block.get('text')
            if not isinstance(t, str):
                return None
            raw_parts.append(t)
            stripped = strip_reminders(t)
            if stripped:
                text_parts.append(stripped)
        else:
            non_text += 1
            if btype == 'image':
                has_images = True
    return ' '.join(text_parts), ' '.join(raw_parts), non_text, has_images


def build_routing_summary(payload: dict) -> RoutingSummary | None:
    """Extract a bounded routing summary from an Anthropic messages payload.

    Returns None if the payload is malformed, messages are missing/empty, or no
    usable user text can be recovered.  Any of those conditions causes the
    caller to fail-closed and keep the original requested model.

    Text source order (first non-empty wins): the final user message's text →
    the final message's embedded ``<transcript>`` last user turn → a walk-back
    over prior ``messages`` to the most recent ``role=='user'`` turn with usable
    text.  The walk-back keeps tool_result-only continuation turns freshly
    classified instead of reusing a stale cached tier.  Recovered prior-turn
    text is tail-capped to ``_WALKBACK_TAIL_LIMIT`` before ``_TEXT_LIMIT`` so a
    large prior turn cannot bias routing toward a higher tier.

    Privacy contract: this function never reads system, tool schemas, tool
    names, tool descriptions, metadata, _anthropic_beta, or any provider key.
    It reads only messages[].content text/image block types and counts.
    ``<transcript>`` blocks (prior conversation history embedded by skills) are
    stripped before classification; prior ``messages`` are consulted only as a
    text source for THIS turn.  Counts, ``has_images``, and
    ``final_non_text_blocks`` always describe the final message only.
    """
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        return None

    # Walk all messages to accumulate counts
    total_messages = len(messages)
    prior_user = 0
    prior_assistant = 0
    tool_use_count = 0
    tool_result_count = 0

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return None
        role = msg.get('role')
        is_last = (i == total_messages - 1)

        if not is_last:
            if role == 'user':
                prior_user += 1
            elif role == 'assistant':
                prior_assistant += 1

        content = msg.get('content')
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get('type')
                if btype == 'tool_use':
                    tool_use_count += 1
                elif btype == 'tool_result':
                    tool_result_count += 1

    # Inspect the final message — must be a user message
    final_msg = messages[-1]
    if not isinstance(final_msg, dict) or final_msg.get('role') != 'user':
        return None

    # final_non_text / has_images describe the FINAL message only — never the
    # walk-back source — since they characterise THIS turn's request shape.
    extracted = _extract_user_text(final_msg.get('content'))
    if extracted is None:
        return None
    final_user_text, raw_text, final_non_text, has_images = extracted

    # Detect tool_result-only final message for agentic continuations.
    final_content = final_msg.get('content')
    final_is_tool_result_only = (
        isinstance(final_content, list)
        and len(final_content) > 0
        and all(
            isinstance(b, dict) and b.get('type') == 'tool_result'
            for b in final_content
        )
    )

    recovered_via_walkback = False
    # True only when final_user_text came from the final message's OWN content
    # (Path 1).  Transcript fallback (Path 2) and walk-back (Path 3) both leave
    # recovered_via_walkback False for Path 2, so this explicit flag is required
    # to gate is_short_affirmation: an affirmation recovered from a <transcript>
    # block or a prior message is historical, not a live confirmation turn.
    text_from_final_message_directly = bool(final_user_text)

    if not final_user_text:
        # The final message may be transcript-only (e.g. a skill sub-turn with
        # no extra instruction).  Fall back to the last user turn inside the
        # final message's transcript so trivial tasks still route accurately
        # rather than fail-closing to the requested model.
        fallback = strip_reminders(last_transcript_user_turn(raw_text))
        if fallback:
            final_user_text = fallback

    if not final_user_text:
        # Still text-less — typically a tool_result-only continuation turn in an
        # agentic loop.  Walk prior messages backward to the most recent user
        # turn with usable text and classify on THAT, so the turn is freshly
        # classified instead of reusing a stale cached tier.  Head-cap the
        # recovered text so the recovered intent (not boilerplate) is visible
        # to the classifier. Prompts/instructions put their imperative at the
        # head and boilerplate at the tail; taking the head recovers intent.
        for prior in reversed(messages[:-1]):
            if not isinstance(prior, dict) or prior.get('role') != 'user':
                continue
            prior_extracted = _extract_user_text(prior.get('content'))
            if prior_extracted is None:
                continue
            prior_text, prior_raw, _, _ = prior_extracted
            if not prior_text:
                prior_text = strip_reminders(last_transcript_user_turn(prior_raw))
            if prior_text:
                final_user_text = prior_text[:_WALKBACK_TAIL_LIMIT]
                recovered_via_walkback = True
                break

    if not final_user_text:
        return None

    # Truncate to classifier input limit: keep first 20% (opening context) +
    # marker + last 80% (current intent).  The tail is more informative for
    # classification — recent requests outweigh session preamble.
    truncated = len(final_user_text) > _TEXT_LIMIT
    if truncated:
        head_len = int(_TEXT_LIMIT * _TEXT_LIMIT_HEAD_FRAC)
        tail_len = _TEXT_LIMIT - head_len - len(_TRUNCATION_MARKER)
        final_user_text = (
            final_user_text[:head_len]
            + _TRUNCATION_MARKER
            + final_user_text[-tail_len:]
        )

    # Compute system prompt SHA256 for the LRU cache key (ADR 0010).
    # Uses same serialization as handlers.py _extract_prompt_capture() for consistency.
    # The SHA is a cache key — not prompt content — so it does not violate the
    # classifier-input privacy contract.
    system_prompt_sha256: str | None = None
    system = payload.get('system')
    if system:
        try:
            if isinstance(system, str):
                sha_bytes = system.encode('utf-8')
            else:
                sha_bytes = json.dumps(system, sort_keys=True, ensure_ascii=False).encode('utf-8')
            system_prompt_sha256 = hashlib.sha256(sha_bytes).hexdigest()
        except Exception:
            pass  # fail-open: leave as None

    return RoutingSummary(
        final_user_text=final_user_text,
        text_truncated=truncated,
        total_messages=total_messages,
        prior_user_messages=prior_user,
        prior_assistant_messages=prior_assistant,
        tool_use_count=tool_use_count,
        tool_result_count=tool_result_count,
        final_non_text_blocks=final_non_text,
        has_images=has_images,
        recovered_via_walkback=recovered_via_walkback,
        final_is_tool_result_only=final_is_tool_result_only,
        is_short_affirmation=(
            text_from_final_message_directly and is_short_affirmation(final_user_text)
        ),
        system_prompt_sha256=system_prompt_sha256,
    )


# ---------------------------------------------------------------------------
# Prior-response extraction helpers (affirmation enrichment)
# ---------------------------------------------------------------------------

def _extract_assistant_text(content) -> str:
    """Extract text from an assistant message's content (str or list of blocks).

    Malformed items (non-dict elements, non-str text values) are silently skipped.
    Text blocks are concatenated with '\\n'.  Returns empty string if no text found.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts: list[str] = []
    for element in content:
        if not isinstance(element, dict):
            continue  # gracefully skip malformed items
        if element.get('type') == 'text':
            text = element.get('text') or ''
            if isinstance(text, str) and text:
                parts.append(text)
    return '\n'.join(parts)


def _truncate_prior_response(text: str, limit: int) -> str:
    """Apply 30/70 head/tail truncation when text exceeds the configured limit."""
    if len(text) <= limit:
        return text
    head_len = int(limit * _PRIOR_RESPONSE_HEAD_FRAC)
    tail_len = limit - head_len - len(_TRUNCATION_MARKER)
    return text[:head_len] + _TRUNCATION_MARKER + text[-tail_len:]


def _extract_prior_response_summary(
    messages: list,
    limit: int,
) -> str | None:
    """Walk backward through messages to find the most recent assistant message with text.

    Returns the (truncated) text or None if no text-bearing assistant message exists.
    The caller must not classify bare affirmation text when None is returned.
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            continue
        text = _extract_assistant_text(msg.get('content'))
        if text:
            return _truncate_prior_response(text, limit)
    return None


# ---------------------------------------------------------------------------
# System-prompt tier classification helpers (ADR 0010 / ADR 0012)
# ---------------------------------------------------------------------------

def _extract_system_prompt_preview(payload: dict, limit: int) -> str:
    """Extract and head-cap system prompt text for the system-prompt classifier.

    Returns empty string when no system prompt is present or text is empty.
    ADR 0012: if system is a str, use directly; if a list, collect 'text' blocks
    (with isinstance(dict) guard), concatenate with '\\n', then head-cap to limit.
    """
    system = payload.get('system')
    if not system:
        return ''
    if isinstance(system, str):
        return system[:limit]
    if isinstance(system, list):
        parts: list[str] = []
        for element in system:
            if not isinstance(element, dict):
                continue
            if element.get('type') == 'text':
                text = element.get('text', '')
                if isinstance(text, str) and text:
                    parts.append(text)
        return '\n'.join(parts)[:limit]
    return ''


def build_system_prompt_classifier_payload(system_preview: str, config: 'Config') -> dict:
    """Build a classifier payload for system-prompt role-complexity inference.

    Uses _CLASSIFIER_SYSTEM_PROMPT_TIER (role-focused prompt, not task-focused)
    and the standard one-word response format.  The same configured classifier
    model is reused; no separate model config.
    """
    return {
        _SENTINEL_KEY: True,
        'model': config.auto_model_routing_classifier_model,
        'max_tokens': _CLASSIFIER_MAX_TOKENS,
        'temperature': _CLASSIFIER_TEMPERATURE,
        'system': _CLASSIFIER_SYSTEM_PROMPT_TIER,
        'messages': [
            {
                'role': 'user',
                'content': system_preview,
            }
        ],
    }


def _classify_system_prompt(
    system_preview: str,
    system_prompt_sha256: str | None,
    cache_size: int,
    config: 'Config',
    snapshot,
    credentials: dict,
    log_tag: str,
) -> tuple[str, float, bool]:
    """Return (tier_label, score, classification_failed) for a system prompt.

    Cache hit path: reads from _sys_prompt_cache under a short lock, returns cached result.
    Cache miss path: calls the system-prompt classifier, writes to cache on success only.
    Failed classifications are not cached so future requests can retry.

    When system_preview is empty (no system prompt), returns ('standard', 1.0, False)
    per ADR 0010 default.

    Thread safety: _sys_prompt_cache_lock is held only for short dict operations,
    never across the classifier network call (satisfies concurrency.md invariant).
    """
    if not system_preview:
        return 'standard', 1.0, False

    # Cache lookup (short lock, no I/O)
    if system_prompt_sha256:
        with _sys_prompt_cache_lock:
            if system_prompt_sha256 in _sys_prompt_cache:
                _sys_prompt_cache.move_to_end(system_prompt_sha256)
                tier, score = _sys_prompt_cache[system_prompt_sha256]
                return tier, score, False

    # Cache miss — classify (no lock held across network call)
    clf_payload = build_system_prompt_classifier_payload(system_preview, config)
    try:
        send_fn = getattr(
            snapshot.backend, 'send_classifier_message', snapshot.backend.send_message
        )
        response = send_fn(clf_payload, credentials, config)
        label = parse_classifier_label(response)
        if label is None:
            logger.warning(
                '%s System-prompt classifier returned invalid label — using standard: raw=%r',
                log_tag,
                _classifier_raw_text_preview(response),
            )
            return 'standard', 1.0, True  # fail-open; do NOT cache
        score = _LABEL_SCORES.get(label, 1.0)
        # Write to cache on success only; do not cache failures.
        if system_prompt_sha256:
            with _sys_prompt_cache_lock:
                _sys_prompt_cache[system_prompt_sha256] = (label, score)
                _sys_prompt_cache.move_to_end(system_prompt_sha256)
                while len(_sys_prompt_cache) > cache_size:
                    _sys_prompt_cache.popitem(last=False)
        logger.debug(
            '%s System-prompt classifier: sha=%s → %s (score=%.1f)',
            log_tag, (system_prompt_sha256 or '')[:8], label, score,
        )
        return label, score, False
    except Exception as exc:
        logger.warning(
            '%s System-prompt classifier call failed — using standard: %s',
            log_tag, exc,
        )
        return 'standard', 1.0, True  # fail-open; do NOT cache


def _apply_weighted_blend(
    user_label: str,
    system_prompt_sha256: str | None,
    payload: dict,
    config: 'Config',
    snapshot,
    credentials: dict,
    log_tag: str,
) -> tuple[str, str, float, float, float, bool]:
    """Apply weighted system-prompt + user-prompt tier blend (ADR 0010).

    Returns (final_tier_label, sys_tier, sys_score, user_score, weighted_score, sys_failed).

    ``user_label`` is the post-bump classifier label (trivial/standard/deep).
    The system prompt is classified (with LRU caching) and blended with the
    user score using the configured weights and thresholds.

    Only called in 'classifier' mode after a successful user-prompt label is
    produced (including the affirmation_classified path).
    """
    cache_size = getattr(config, 'auto_model_routing_system_prompt_cache_size', 256)
    preview_limit = getattr(config, 'auto_model_routing_system_prompt_preview_limit', 500)
    sys_weight = getattr(config, 'auto_model_routing_system_prompt_weight', 0.30)
    user_weight = getattr(config, 'auto_model_routing_user_prompt_weight', 0.70)
    trivial_threshold = getattr(config, 'auto_model_routing_trivial_threshold', 0.75)
    standard_threshold = getattr(config, 'auto_model_routing_standard_threshold', 1.50)

    sys_preview = _extract_system_prompt_preview(payload, preview_limit)
    sys_tier, sys_score, sys_failed = _classify_system_prompt(
        sys_preview, system_prompt_sha256, cache_size, config, snapshot, credentials, log_tag,
    )
    user_score = _LABEL_SCORES.get(user_label, 1.0)
    weighted_score = sys_weight * sys_score + user_weight * user_score

    if weighted_score < trivial_threshold:
        final_label = 'trivial'
    elif weighted_score < standard_threshold:
        final_label = 'standard'
    else:
        final_label = 'deep'

    logger.debug(
        '%s Weighted blend: user=%s(%.1f) sys=%s(%.1f) score=%.3f → %s',
        log_tag, user_label, user_score, sys_tier, sys_score, weighted_score, final_label,
    )
    return final_label, sys_tier, sys_score, user_score, weighted_score, sys_failed


# ---------------------------------------------------------------------------
# Classifier payload construction
# ---------------------------------------------------------------------------

def build_classifier_payload(
    summary: RoutingSummary,
    config: 'Config',
    prior_response_summary: str | None = None,
) -> dict:
    """Build a synthetic, non-streaming classifier request.

    The payload uses the configured classifier model, tiny max_tokens,
    temperature 0, a fixed system prompt, and a single user message with the
    JSON routing summary.  It carries the internal sentinel key so backends can
    identify and isolate classifier calls.

    When ``config.auto_model_routing_confidence_bump`` is True, uses the JSON
    system prompt (``_CLASSIFIER_SYSTEM_JSON``) and a larger ``max_tokens``
    budget to accommodate the structured response.  The existing one-word prompt
    and four-token budget are untouched when confidence bump is off.

    When ``prior_response_summary`` is provided it is injected into the routing
    summary JSON and the system prompt is extended with the prior-context suffix.
    """
    use_json = getattr(config, 'auto_model_routing_confidence_bump', False)
    system = _CLASSIFIER_SYSTEM_JSON if use_json else _CLASSIFIER_SYSTEM
    if use_json and prior_response_summary is not None:
        system = system + _CLASSIFIER_SYSTEM_JSON_PRIOR_SUFFIX
    return {
        _SENTINEL_KEY: True,
        'model': config.auto_model_routing_classifier_model,
        'max_tokens': _CLASSIFIER_MAX_TOKENS_JSON if use_json else _CLASSIFIER_MAX_TOKENS,
        'temperature': _CLASSIFIER_TEMPERATURE,
        'system': system,
        'messages': [
            {
                'role': 'user',
                'content': summary.to_classifier_json(prior_response_summary=prior_response_summary),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Classifier response parsing
# ---------------------------------------------------------------------------

_ALPHA_TOKEN_RE = re.compile(r'[a-z]+')

# Max chars of raw classifier output echoed into the invalid-label warning.
_RAW_LABEL_LOG_LIMIT = 120


def _classifier_raw_text_preview(response: dict) -> str:
    """One-line bounded preview of a classifier response's text blocks.

    Used only in the ``classifier_invalid`` warning so a deterministic rejection
    (e.g. the model wrapping its answer in an unexpected token) is diagnosable
    from the log instead of being discarded.  Returns ``'<non-text>'`` when the
    response carries no text blocks.
    """
    content = response.get('content')
    if not isinstance(content, list):
        return '<no-content>'
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'text':
            t = block.get('text')
            if isinstance(t, str):
                parts.append(t)
    if not parts:
        return '<non-text>'
    preview = ' '.join(parts).replace('\r', ' ').replace('\n', ' ').strip()
    if len(preview) > _RAW_LABEL_LOG_LIMIT:
        return preview[:_RAW_LABEL_LOG_LIMIT] + '…'
    return preview


def parse_classifier_label(
    response: dict,
) -> Literal['trivial', 'standard', 'deep'] | None:
    """Extract a valid classification label from a non-streaming response dict.

    Accepts a response whose only alphabetic token is one of the three valid
    labels, tolerating surrounding punctuation/markdown (``deep.``, ``**deep**``,
    ``"standard"``, ``- trivial``).  Production traffic showed the classifier
    (haiku, temperature 0) deterministically wrapping its single-word answer in
    trailing punctuation or markdown; the previous exact-match parser rejected
    those as ``classifier_invalid`` and silently misrouted genuine deep tasks down
    to the requested tier.

    Still rejects sentences, JSON objects, negations, and multi-label output by
    requiring exactly one alphabetic token: ``The answer is deep``, ``not deep``,
    ``deep trivial``, and ``{"label": "deep"}`` all contain more than one and are
    rejected.  Returns None on any invalid or unexpected output (and the caller
    logs the raw text so the failure stays diagnosable).
    """
    content = response.get('content')
    if not isinstance(content, list) or not content:
        return None

    # Collect text from all text blocks.  Reasoning models (e.g. OpenRouter's
    # deepseek alias) emit ``thinking``/``redacted_thinking`` blocks alongside
    # the text label — skip those rather than invalidating the whole response.
    # Other non-text blocks (``tool_use``, ``image``) still invalidate it.
    _SKIP_BLOCK_TYPES = frozenset({'thinking', 'redacted_thinking'})
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        btype = block.get('type')
        if btype in _SKIP_BLOCK_TYPES:
            continue
        if btype != 'text':
            return None
        t = block.get('text')
        if not isinstance(t, str):
            return None
        text_parts.append(t)

    combined = ' '.join(text_parts).strip().lower()
    # Fast path: bare label, no wrapping at all.
    if combined in _VALID_LABELS:
        return combined  # type: ignore[return-value]
    # Tolerant path: accept iff the response contains exactly ONE alphabetic
    # token and it is a valid label.  This admits punctuation/markdown wrapping
    # (deep., **deep**, "standard") while rejecting sentences, negations, JSON,
    # and any output naming two words (whether or not both are labels).
    alpha_tokens = _ALPHA_TOKEN_RE.findall(combined)
    if len(alpha_tokens) == 1 and alpha_tokens[0] in _VALID_LABELS:
        return alpha_tokens[0]  # type: ignore[return-value]
    return None


def parse_classifier_label_json(
    response: dict,
) -> tuple[str, float] | None:
    """Extract a valid (label, confidence) pair from a non-streaming JSON response.

    Used ONLY when ``config.auto_model_routing_confidence_bump`` is True.  The
    existing ``parse_classifier_label`` function and one-word prompt are never
    modified.

    Accepts a response whose text blocks (after stripping thinking/
    redacted_thinking) contain a single JSON object with ``label`` and
    ``confidence`` keys.  Tolerates surrounding whitespace but rejects:

    - Malformed JSON or non-object values
    - Missing ``label`` or ``confidence`` keys
    - Unknown label (not in ``_VALID_LABELS``)
    - Non-numeric ``confidence`` or confidence outside ``[0.0, 1.0]``
    - Non-text content blocks (other than skipped thinking blocks)

    Returns ``(label, confidence)`` on success, or ``None`` on any error.
    The caller fails-closed to the original requested model on ``None``.
    """
    content = response.get('content')
    if not isinstance(content, list) or not content:
        return None

    _SKIP_BLOCK_TYPES = frozenset({'thinking', 'redacted_thinking'})
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        btype = block.get('type')
        if btype in _SKIP_BLOCK_TYPES:
            continue
        if btype != 'text':
            return None
        t = block.get('text')
        if not isinstance(t, str):
            return None
        text_parts.append(t)

    combined = ' '.join(text_parts).strip()
    if not combined:
        return None

    try:
        parsed = json.loads(combined)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    label = parsed.get('label')
    if label not in _VALID_LABELS:
        return None

    confidence = parsed.get('confidence')
    if not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        return None

    return (label, confidence)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rules-mode classifier
# ---------------------------------------------------------------------------

# Bare greeting/ack phrases that are unambiguously trivial.  Conservative:
# only exact matches (after lowercasing + stripping) that carry no task signal.
_RULES_TRIVIAL_PHRASES: frozenset[str] = frozenset({
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'ty', 'thx',
    'ok', 'okay', 'great', 'good', 'nice', 'cool', 'awesome', 'got it',
    'understood', 'noted', 'ack', 'acknowledged',
    'sounds good', 'makes sense', 'looks good', 'lgtm', 'done',
    'yes', 'no', 'correct', 'sure', 'right',
})

# Trivial task prefixes: short utterances that do not escalate to standard
# once certain safety keywords (e.g., plan, design) are not present.
# Note: prefixes do NOT have trailing spaces; word boundary is checked after.
_RULES_TRIVIAL_PREFIXES: tuple[str, ...] = (
    'read ', 'look at ', 'check ', 'open ', 'show ', 'list ', 'find ', 'search ',
    'get ', 'fetch ', 'see ', 'view ', 'watch ', 'browse ', 'look ', 'what ',
    'sure', 'ok', 'okay', 'alright', 'right',
    'sounds good', 'makes sense', 'looks good',
)

# Max length for prefix-based trivial detection.  Requests longer than this
# are not considered prefix-trivial, even if they start with a trivial prefix.
_TRIVIAL_PREFIX_MAX_CHARS = 120

# Deep complexity signals: security-sensitive, architectural, or high-stakes
# design concepts that warrant extended reasoning.
_RULES_DEEP_KEYWORDS: tuple[str, ...] = (
    'architecture', 'multi-tenant', 'multitenant', 'sso', 'saml',
    'csrf', 'xss', 'sql injection', 'vulnerability', 'exploit',
    'cryptograph', 'distributed system', 'microservice', 'microservices',
)

# Standard planning/implementation signals: any of these in the text
# indicates a real task that needs at least sonnet-level reasoning.
_RULES_STANDARD_KEYWORDS: tuple[str, ...] = (
    'plan', 'design', 'outline', 'implement', 'build', 'create', 'fix',
    'add', 'debug', 'refactor', 'write', 'develop', 'test', 'deploy',
    'integrate', 'migrate', 'update', 'explain', 'analyze', 'review',
    'optimize', 'improve', 'restructure', 'rewrite', 'generate', 'parse',
    'format', 'convert', 'extract', 'modify', 'delete', 'remove',
    'rename', 'move', 'search', 'query', 'fetch', 'load', 'render',
    'redesign', 'auth', 'security', 'api', 'database', 'schema',
    'endpoint', 'function', 'class', 'module', 'service', 'component',
    'feature', 'bug', 'error', 'issue', 'problem', 'how', 'what', 'why',
)


def classify_by_rules(summary: 'RoutingSummary') -> str | None:
    """Rule-based routing classifier using only RoutingSummary fields.

    Conservative; returns a classification label (``'trivial'``,
    ``'standard'``, or ``'deep'``) or ``None`` when no signal is present
    (caller fails closed to the requested model).

    Uses only ``final_user_text``, ``has_images``, and
    ``final_non_text_blocks`` from *summary* — never system, tools,
    tool schemas, or full history.
    """
    # Non-text content (images, tool blocks) → at least standard, even with no text.
    if summary.has_images or summary.final_non_text_blocks > 0:
        return 'standard'

    text = (summary.final_user_text or '').strip()
    if not text:
        return None

    lower = text.lower()

    # Exact phrase match → trivial (checked before keyword scanning so a
    # bare "hi" never accidentally hits a standard keyword).
    if lower in _RULES_TRIVIAL_PHRASES:
        return 'trivial'

    # Prefix-trivial detection: short requests starting with trivial prefixes
    # (e.g., "read the file", "look at the error") are trivial unless they
    # contain a safety keyword (e.g., "plan", "design").
    if len(text) <= _TRIVIAL_PREFIX_MAX_CHARS:
        for prefix in _RULES_TRIVIAL_PREFIXES:
            if lower.startswith(prefix):
                # Word boundary check: ensure the character after the prefix
                # is not alphanumeric (e.g., "rightclick" doesn't match "right ").
                end = len(prefix)
                if end < len(lower) and lower[end].isalpha():
                    continue  # False positive; keep looking.
                # Safety check: ensure no standard planning keywords are present.
                # Plan/design must never be trivial, even with a trivial prefix.
                has_safety_keyword = any(
                    kw in lower for kw in _RULES_STANDARD_KEYWORDS
                )
                if not has_safety_keyword:
                    return 'trivial'
                break  # Stop trying other prefixes once one matches.

    # Deep complexity signals outrank standard ones.
    for kw in _RULES_DEEP_KEYWORDS:
        if kw in lower:
            return 'deep'

    # Standard planning / implementation signals.
    for kw in _RULES_STANDARD_KEYWORDS:
        if kw in lower:
            return 'standard'

    # No signal detected.
    return None


# ---------------------------------------------------------------------------
# Multi-mode classifier dispatcher
# ---------------------------------------------------------------------------

def _dispatch_classifier_mode(
    mode: str,
    summary: 'RoutingSummary',
    config: 'Config',
    snapshot,
    credentials: dict,
    est_tokens: int,
    requested: str,
    log_tag: str,
    task_tag: str | None = None,
    prior_response_summary: str | None = None,
) -> tuple[str | None, str, float | None, int, int, str | None, str | None, str | None, str | None]:
    """Dispatch to the appropriate classification mode.

    Returns ``(label_or_tier, reason_code, classifier_confidence, clf_in, clf_out,
    clf_model, clf_summary_json, clf_raw_response, clf_format)`` where the first
    element is:

    * For ``'classifier'`` and ``'rules'`` modes: a classification label
      (``'trivial'``, ``'standard'``, or ``'deep'``), or ``None`` on failure.
    * For ``'tag'`` mode: the tier/model string from
      ``config.auto_model_routing_task_tiers``, or ``None`` on failure.

    ``classifier_confidence`` is a float when ``auto_model_routing_confidence_bump``
    is enabled and the JSON response parses successfully; ``None`` otherwise.
    ``None`` first element means the caller must fail-closed to the requested model.
    Unknown or empty mode values fall back to ``'classifier'``.

    The last four elements (clf_model, clf_summary_json, clf_raw_response,
    clf_format) are only populated on the successful LLM call path; they are
    ``None`` for rules, tag, failure, and invalid paths.
    """
    if mode == 'tag':
        # Tag mode: look up the task name in the configured task→tier map.
        if not task_tag:
            logger.info(
                '%s Model router: tag mode but no task_tag supplied — '
                'fail-closed to %s',
                log_tag, requested,
            )
            return None, 'no_task_tag', None, 0, 0, None, None, None, None
        task_tiers = getattr(config, 'auto_model_routing_task_tiers', None)
        if not task_tiers or task_tag not in task_tiers:
            logger.warning(
                '%s Model router: unknown task tag %r in tag mode '
                '(configured: %s) — fail-closed to %s',
                log_tag, task_tag,
                list(task_tiers) if task_tiers else '(none)',
                requested,
            )
            return None, 'unknown_task_tag', None, 0, 0, None, None, None, None
        tier = task_tiers[task_tag]
        logger.info(
            '%s Model router: tag mode task=%r → tier=%s '
            '(backend=%s requested=%s)',
            log_tag, task_tag, tier, snapshot.name, requested,
        )
        return tier, 'task_tag_routed', None, 0, 0, None, None, None, None

    if mode == 'rules':
        # Rules mode: deterministic keyword-based classification; no LLM call.
        label = classify_by_rules(summary)
        if label is None:
            logger.info(
                '%s Model router: rules mode — no signal, fail-closed to %s',
                log_tag, requested,
            )
            return None, 'rules_no_signal', None, 0, 0, None, None, None, None
        reason = f'classifier_rules_{label}'
        logger.info(
            '%s Model router: rules mode → %s (backend=%s requested=%s '
            'prompt_chars=%d)',
            log_tag, label, snapshot.name, requested,
            len(summary.final_user_text),
        )
        return label, reason, None, 0, 0, None, None, None, None

    # Default / 'classifier' mode: LLM-based classifier call.
    threshold = getattr(config, 'auto_model_routing_long_context_threshold', 0)
    _size_pct = (
        f'{round(est_tokens * 100 / threshold)}%' if threshold > 0 else 'n/a'
    )
    logger.info(
        '%s Model router: classifying (backend=%s requested=%s prompt_chars=%d '
        'est=%d/%d (%s)) prompt=%r',
        log_tag,
        snapshot.name,
        requested,
        len(summary.final_user_text),
        est_tokens,
        threshold if threshold > 0 else 0,
        _size_pct,
        _prompt_log_preview(summary.final_user_text),
    )
    # Full (bounded to _TEXT_LIMIT) prompt at DEBUG — dropped by the default
    # INFO console handler, but written in full to --log-file when configured.
    logger.debug(
        '%s Model router: classifier prompt (backend=%s requested=%s) full=%r',
        log_tag, snapshot.name, requested, summary.final_user_text,
    )
    classifier_payload = build_classifier_payload(summary, config, prior_response_summary)
    use_confidence_bump = getattr(config, 'auto_model_routing_confidence_bump', False)
    # Transparency fields captured before the call so they're available even when
    # the response is available (raw_response captured after).
    clf_model: str | None = config.auto_model_routing_classifier_model
    clf_summary_json: str | None = summary.to_classifier_json(prior_response_summary=prior_response_summary)
    clf_format: str | None = 'json' if use_confidence_bump else 'standard'
    parsed_confidence: float | None = None
    clf_raw_response: str | None = None
    try:
        send_fn = getattr(
            snapshot.backend, 'send_classifier_message', snapshot.backend.send_message
        )
        response = send_fn(classifier_payload, credentials, config)
        # Extract full concatenated text from all text-type content blocks.
        # Mirrors the extraction in parse_classifier_label (skips thinking blocks,
        # collects text blocks verbatim, never lowercases or truncates).
        _content = response.get('content') if isinstance(response, dict) else None
        if isinstance(_content, list):
            _parts: list[str] = []
            _SKIP = frozenset({'thinking', 'redacted_thinking'})
            for _block in _content:
                if isinstance(_block, dict):
                    _btype = _block.get('type')
                    if _btype in _SKIP:
                        continue
                    if _btype == 'text':
                        _t = _block.get('text')
                        if isinstance(_t, str):
                            _parts.append(_t)
            if _parts:
                clf_raw_response = ' '.join(_parts)
        if use_confidence_bump:
            parsed_json = parse_classifier_label_json(response)
            if parsed_json is None:
                label: str | None = None
            else:
                label, parsed_confidence = parsed_json
        else:
            label = parse_classifier_label(response)
    except Exception as exc:
        logger.warning(
            '%s Model router: classifier call failed (backend=%s) — keeping %s: %s',
            log_tag, snapshot.name, requested, exc,
        )
        return None, 'classifier_failed', None, 0, 0, None, None, None, None

    if label is None:
        logger.warning(
            '%s Model router: classifier returned invalid label (backend=%s) — '
            'keeping %s: raw=%r',
            log_tag, snapshot.name, requested,
            _classifier_raw_text_preview(response),
        )
        return None, 'classifier_invalid', None, 0, 0, None, None, None, None

    if label == 'trivial':
        reason = 'classifier_trivial'
    elif label == 'deep':
        reason = 'classifier_deep'
    else:
        reason = 'classifier_standard'

    clf_in = response.get('usage', {}).get('input_tokens', 0)
    clf_out = response.get('usage', {}).get('output_tokens', 0)
    return label, reason, parsed_confidence, clf_in, clf_out, clf_model, clf_summary_json, clf_raw_response, clf_format


# ---------------------------------------------------------------------------
# Main routing entry point
# ---------------------------------------------------------------------------

def route_model(
    payload: dict,
    snapshot,
    credentials: dict,
    cached_session_tier: str | None = None,
    session_context_tokens: int = 0,
    session_estimate_ratio: float = 1.0,
    log_tag: str = '',
    override_mode: str | None = None,
    task_tag: str | None = None,
    ctx_key: str | None = None,
    baseline_model: str | None = None,
) -> ModelRoutingDecision:
    """Classify the request and rewrite ``payload['model']`` in place.

    Called after local-command bypass and before the main backend dispatch.
    Any non-empty string model is eligible for routing.  Never holds registry or
    selector locks.  Routing failures preserve the original requested model.

    If ``cached_session_tier`` is provided and the request's final_user_text
    came from a walk-back over prior messages (a text-less continuation turn),
    reuses the cached tier without calling the classifier.  This avoids
    misclassifying recovered boilerplate and saves a classifier call.

    The long-context size floor combines two session-aware signals with the
    per-request estimate: ``session_context_tokens`` (the last response's measured
    total context size, a lower bound on this turn's input) and
    ``session_estimate_ratio`` (a calibration factor correcting the estimator's
    undercount).  The floor fires when
    ``max(round(estimate * ratio), session_context_tokens) >= threshold``.  Both
    default to the identity (``0`` / ``1.0``), reproducing the estimate-only floor.

    Returns a ``ModelRoutingDecision`` describing the outcome for logging.
    """
    # If this is itself a classifier payload, skip — prevents recursion
    if payload.get(_SENTINEL_KEY):
        raw_model = str(payload.get('model', ''))
        return ModelRoutingDecision(
            requested_model=raw_model,
            routed_model=raw_model,
            classification=None,
            applied=False,
            reason_code='disabled',
        )

    config = snapshot.config

    raw_model = payload.get('model')

    # Apply baseline_model lock at input: substitute the client's model with the
    # locked baseline so all downstream paths (classifier, routing summary, cache)
    # see a consistent input. Preserve the original in 'requested' for response echo.
    if baseline_model:
        payload['model'] = baseline_model

    if not config.auto_model_routing:
        return ModelRoutingDecision(
            requested_model=str(raw_model) if isinstance(raw_model, str) else repr(raw_model),
            routed_model=str(raw_model) if isinstance(raw_model, str) else repr(raw_model),
            classification=None,
            applied=False,
            reason_code='disabled',
        )

    # Any non-empty string model value is eligible
    if not is_model_auto_routable(raw_model):
        return ModelRoutingDecision(
            requested_model=str(raw_model) if isinstance(raw_model, str) else repr(raw_model),
            routed_model=str(raw_model) if isinstance(raw_model, str) else repr(raw_model),
            classification=None,
            applied=False,
            reason_code='model_not_eligible',
        )

    # Capture the actual incoming model string; fail-closed branches return this.
    requested = raw_model  # guaranteed str and non-empty by is_model_auto_routable

    # When a baseline_model lock is active, routing decisions (tier caps, classifier
    # log context) operate against the locked baseline rather than the client's
    # arbitrary model.  The client-sent model is preserved in `requested` for
    # response echoing, the `applied` flag, and ModelRoutingDecision.requested_model.
    routing_baseline = baseline_model if baseline_model else requested

    # --- Long-context size floor (deterministic, pre-classifier).  When the
    # estimated total request size is near the 200K window, force the long-context
    # opus tier and inject the context-1m beta so the only model that can serve the
    # request is selected.  This outranks the classifier and the walk-back/cache
    # paths and applies even to text-less continuation turns (a large tool_result-
    # only turn still needs 1m).  The estimate counts system+tools+history but is
    # NEVER sent to the classifier, so the bounded classifier-input invariant holds.
    threshold = config.auto_model_routing_long_context_threshold
    long_target = config.auto_model_routing_long  # model string or 'off'
    floor_active = threshold > 0 and long_target != 'off'
    try:
        est_tokens = estimate_input_tokens(payload) if threshold > 0 else 0
    except Exception:
        # Malformed payload (e.g. non-list messages) — let build_routing_summary
        # below handle it and fail-closed; never crash the request on the floor.
        est_tokens = 0
    if floor_active:
        # Combine the per-request estimate (calibration-corrected) with the last
        # measured session context size; the larger drives the decision.
        corrected = round(est_tokens * session_estimate_ratio)
        predicted = max(corrected, session_context_tokens)
        if predicted >= threshold:
            payload['model'] = long_target
            _ensure_long_context_beta(payload)  # harmless for non-opus: Anthropic
            # mapper's _supports_long_context gate strips context-1m for non-opus
            logger.info(
                '%s Model router: size floor forcing %s (backend=%s requested=%s '
                'predicted=%d threshold=%d est=%d ratio=%.2f session_floor=%d ctx=%s)',
                log_tag, long_target, snapshot.name, requested,
                predicted, threshold, est_tokens, session_estimate_ratio,
                session_context_tokens,
                ctx_key.split('\x00', 1)[-1] if ctx_key else '--------',
            )
            return ModelRoutingDecision(
                requested_model=requested,
                routed_model=long_target,
                classification=None,
                applied=(long_target != requested),
                reason_code='size_forced_long_context',
                estimated_input_tokens=est_tokens,
                predicted_input_tokens=predicted,
                session_context_tokens=session_context_tokens,
                session_estimate_ratio=session_estimate_ratio,
            )

    # --- Build routing summary
    summary = build_routing_summary(payload)
    if summary is None:
        # Malformed or no final user text — fail-closed: use baseline lock if active,
        # otherwise keep the originally requested model.
        fallback = routing_baseline if baseline_model else requested
        payload['model'] = fallback
        return ModelRoutingDecision(
            requested_model=requested,
            routed_model=fallback,
            classification=None,
            applied=(fallback != requested),
            reason_code='missing_final_user_text',
            estimated_input_tokens=est_tokens,
        )

    # --- Cache-first path: if this is a text-less continuation turn (walk-back
    # recovery) and we have a cached session tier, reuse it instead of
    # re-classifying recovered prose (which is typically boilerplate).
    if (summary.recovered_via_walkback and cached_session_tier is not None):
        if summary.final_is_tool_result_only:
            # Agentic tool_result continuation: the client sends the base model
            # on every turn unconditionally; 'requested' carries no downgrade intent.
            # Bypass the no-upgrade cap, but when baseline_model lock is active,
            # cap against the baseline to enforce the lock.
            if baseline_model:
                capped = _cap_cached_tier(
                    cached_session_tier, routing_baseline,
                    label_map=config.auto_model_routing_classification,
                )
            else:
                capped = cached_session_tier
            payload['model'] = capped
            return ModelRoutingDecision(
                requested_model=requested,
                routed_model=capped,
                classification=None,
                applied=(capped != requested),
                reason_code='session_cached_walkback_tool_result',
                estimated_input_tokens=est_tokens,
                classifier_mode='walkback_cache',
            )
        # Non-tool_result walkback (image-only, transcript-only, etc.):
        # Apply the no-upgrade cap so a prior expensive task cannot hijack
        # a new simpler request via walkback (the production session_cached_walkback bug).
        capped = _cap_cached_tier(
            cached_session_tier, routing_baseline,
            label_map=config.auto_model_routing_classification,
        )
        payload['model'] = capped
        walkback_reason: ReasonCode = (
            'session_cached_walkback_capped'
            if capped != cached_session_tier
            else 'session_cached_walkback'
        )
        return ModelRoutingDecision(
            requested_model=requested,
            routed_model=capped,
            classification=None,
            applied=(capped != requested),
            reason_code=walkback_reason,
            estimated_input_tokens=est_tokens,
            classifier_mode='walkback_cache',
        )

    # --- Extract prior assistant response (unconditional) for classifier enrichment.
    # Used both by the affirmation no-cache path and the main classifier dispatch.
    _prior_response_limit = getattr(
        config, 'auto_model_routing_prior_response_summary_limit', 1000
    )
    _messages_list = payload.get('messages') or []
    prior_response_summary = _extract_prior_response_summary(
        _messages_list, _prior_response_limit
    )

    # --- Short-affirmation continuation: a bare "yes"/"proceed"/"go ahead" as
    # THIS turn's own text carries no complexity signal — it greenlights work the
    # prior turns already established.
    #
    # Cached-tier path: inherit the tier immediately (no classifier call).
    # No-cache path: extract the prior assistant response and call the classifier
    # with enriched input (prior_response_summary) so the turn is routed at the
    # complexity of the work the user agreed to, not the bare "yes".  The result
    # is written to the tier cache (via non-None classification) so subsequent
    # tool-result turns inherit the established tier.  If no text-bearing
    # assistant message exists (critical invariant) or the classifier call fails,
    # fall back to the standard floor without writing the tier cache.
    if (config.auto_model_routing_affirmation_inherit
            and summary.is_short_affirmation):
        if cached_session_tier is not None:
            # Cached path: inherit immediately, no classifier call.
            if baseline_model:
                capped = _cap_cached_tier(
                    cached_session_tier, routing_baseline,
                    label_map=config.auto_model_routing_classification,
                )
            else:
                capped = cached_session_tier
            payload['model'] = capped
            return ModelRoutingDecision(
                requested_model=requested,
                routed_model=capped,
                classification=None,
                applied=(capped != requested),
                reason_code='affirmation_inherited',
                estimated_input_tokens=est_tokens,
                classifier_mode='affirmation',
            )

        # No-cache path: prior_response_summary already extracted above.
        def _affirmation_floor() -> ModelRoutingDecision:
            standard_tier = config.auto_model_routing_classification['standard']
            _capped = _cap_cached_tier(
                standard_tier, routing_baseline,
                label_map=config.auto_model_routing_classification,
            ) if baseline_model else standard_tier
            payload['model'] = _capped
            return ModelRoutingDecision(
                requested_model=requested,
                routed_model=_capped,
                classification=None,
                applied=(_capped != requested),
                reason_code='affirmation_classifier_failed',
                estimated_input_tokens=est_tokens,
                classifier_mode='affirmation',
            )

        # Critical invariant: do not classify bare affirmation text when no
        # text-bearing assistant message exists.  This prevents a session-opening
        # "yes" from writing a trivial tier to the cache before any real task is
        # established.
        if prior_response_summary is None:
            return _affirmation_floor()

        # Concurrent affirmation mitigation: only one classifier call per
        # context key; others use the floor tier.
        sentinel_acquired = False
        if ctx_key is not None:
            with _affirmation_inflight_lock:
                if ctx_key in _affirmation_inflight:
                    return _affirmation_floor()
                _affirmation_inflight.add(ctx_key)
                sentinel_acquired = True

        aff_label: str | None = None
        aff_routed: str | None = None
        aff_cache_tier: str | None = None
        aff_summary_json: str | None = None
        aff_sys_tier: str | None = None
        aff_sys_score: float | None = None
        aff_user_score: float | None = None
        aff_weighted_score: float | None = None
        aff_sys_failed: bool = False
        try:
            use_json = getattr(config, 'auto_model_routing_confidence_bump', False)
            aff_summary_json = summary.to_classifier_json(
                prior_response_summary=prior_response_summary
            )
            system = (
                _CLASSIFIER_SYSTEM_JSON + _CLASSIFIER_SYSTEM_JSON_PRIOR_SUFFIX
                if use_json
                else _CLASSIFIER_SYSTEM
            )
            clf_payload = {
                _SENTINEL_KEY: True,
                'model': config.auto_model_routing_classifier_model,
                'max_tokens': _CLASSIFIER_MAX_TOKENS_JSON if use_json else _CLASSIFIER_MAX_TOKENS,
                'temperature': _CLASSIFIER_TEMPERATURE,
                'system': system,
                'messages': [{'role': 'user', 'content': aff_summary_json}],
            }
            logger.info(
                '%s Model router: affirmation classifying with prior_response_summary '
                '(summary_len=%d backend=%s requested=%s)',
                log_tag, len(prior_response_summary), snapshot.name, requested,
            )
            send_fn = getattr(
                snapshot.backend, 'send_classifier_message', snapshot.backend.send_message
            )
            response = send_fn(clf_payload, credentials, config)
            if use_json:
                parsed_json = parse_classifier_label_json(response)
                label: str | None = parsed_json[0] if parsed_json else None
            else:
                label = parse_classifier_label(response)

            if label is None:
                logger.warning(
                    '%s Model router: affirmation classifier returned invalid label '
                    '(backend=%s) — using floor: raw=%r',
                    log_tag, snapshot.name,
                    _classifier_raw_text_preview(response),
                )
            else:
                # Apply weighted blend (ADR 0010) to produce the final tier.
                # Mode guard mirrors the main dispatch path: blend only in 'classifier'
                # mode; 'rules' avoids LLM calls by design, 'tag' uses direct strings.
                aff_effective_mode = (
                    override_mode
                    or getattr(config, 'auto_model_routing_mode', None)
                    or 'classifier'
                )
                if aff_effective_mode == 'classifier':
                    (blend_label, aff_sys_tier, aff_sys_score,
                     aff_user_score, aff_weighted_score, aff_sys_failed) = _apply_weighted_blend(
                        label, summary.system_prompt_sha256, payload, config,
                        snapshot, credentials, log_tag,
                    )
                else:
                    blend_label = label
                uncapped_tier = config.auto_model_routing_classification[blend_label]
                capped_tier = (
                    _cap_cached_tier(
                        uncapped_tier, routing_baseline,
                        label_map=config.auto_model_routing_classification,
                    )
                    if baseline_model
                    else uncapped_tier
                )
                aff_label = label
                aff_routed = capped_tier
                aff_cache_tier = uncapped_tier
        except Exception as exc:
            logger.warning(
                '%s Model router: affirmation classifier call failed (backend=%s) — '
                'using floor: %s',
                log_tag, snapshot.name, exc,
            )
        finally:
            if sentinel_acquired and ctx_key is not None:
                with _affirmation_inflight_lock:
                    _affirmation_inflight.discard(ctx_key)

        if aff_label is not None:
            payload['model'] = aff_routed
            return ModelRoutingDecision(
                requested_model=requested,
                routed_model=aff_routed,  # type: ignore[arg-type]
                cache_tier=aff_cache_tier,
                classification=aff_label,
                applied=(aff_routed != requested),
                reason_code='affirmation_classified',
                estimated_input_tokens=est_tokens,
                classifier_mode='affirmation',
                classifier_summary_json=aff_summary_json,
                system_prompt_tier=aff_sys_tier,
                system_prompt_score=aff_sys_score,
                user_prompt_score=aff_user_score,
                routing_weighted_score=aff_weighted_score,
                system_prompt_classification_failed=aff_sys_failed,
            )
        return _affirmation_floor()

    # --- Title-generation rule: prompts whose final paragraph is the
    # "Write the title in the predominant language of the session…" instruction
    # are always trivial — they ask for one short label regardless of the session
    # content wrapped inside <session>…</session>.  Bypasses the classifier and
    # does NOT write the tier cache (classification=None) so the parent session's
    # complexity signal is preserved.
    if summary.final_user_text and is_title_generation(summary.final_user_text):
        trivial_model = config.auto_model_routing_classification['trivial']
        if baseline_model:
            capped = _cap_cached_tier(
                trivial_model, routing_baseline,
                label_map=config.auto_model_routing_classification,
            )
        else:
            capped = trivial_model
        payload['model'] = capped
        return ModelRoutingDecision(
            requested_model=requested,
            routed_model=capped,
            classification=None,
            applied=(capped != requested),
            reason_code='rule_title_generation',
            estimated_input_tokens=est_tokens,
            classifier_mode='rules',
        )

    # --- Determine effective dispatch mode and classify
    # Priority: per-request override > global config > default ('classifier')
    effective_mode = (
        override_mode
        or getattr(config, 'auto_model_routing_mode', None)
        or 'classifier'
    )

    (label_or_tier, dispatch_reason, parsed_confidence, clf_in_tokens, clf_out_tokens,
     clf_model, clf_summary_json, clf_raw_response, clf_format) = _dispatch_classifier_mode(
        effective_mode, summary, config, snapshot, credentials,
        est_tokens, routing_baseline, log_tag, task_tag,
        prior_response_summary=prior_response_summary,
    )

    if label_or_tier is None:
        # Fail-closed: keep the originally requested model.
        # The 4 transparency fields remain None (their dataclass defaults) on all
        # non-successful-classifier paths (failed, invalid, rules, tag, etc.).
        return ModelRoutingDecision(
            requested_model=requested,
            routed_model=requested,
            classification=None,
            applied=False,
            reason_code=dispatch_reason,  # type: ignore[arg-type]
            estimated_input_tokens=est_tokens,
            classifier_mode=effective_mode,
        )

    # --- Confidence-based tier bump (only when confidence_bump mode is on).
    # When the classifier's confidence is below the configured threshold, promote
    # the label one tier to err on the side of more capable models:
    #   trivial → standard, standard → deep, deep → deep (no-op: already highest).
    # Deep stays deep with tier_bumped=False since no promotion is possible.
    # Both bumped and unbumped paths use the original classifier label as
    # `classification` (the raw signal) and set tier_bumped to distinguish.
    use_confidence_bump = getattr(config, 'auto_model_routing_confidence_bump', False)
    original_label = label_or_tier  # preserves the raw label for the classification field
    tier_bumped = False
    min_confidence = getattr(config, 'auto_model_routing_min_confidence', 0.0)
    if (effective_mode != 'tag'
            and use_confidence_bump
            and parsed_confidence is not None
            and parsed_confidence < min_confidence
            and label_or_tier in ('trivial', 'standard')):
        if label_or_tier == 'trivial':
            label_or_tier = 'standard'
            dispatch_reason = 'classifier_trivial_bumped'
        else:  # standard
            label_or_tier = 'deep'
            dispatch_reason = 'classifier_standard_bumped'
        tier_bumped = True

    # --- Weighted blend (ADR 0010): applies in 'classifier' mode after confidence bump.
    # For 'rules' and 'tag' modes the system-prompt classifier is not called —
    # 'rules' avoids LLM calls by design; 'tag' produces a direct model string.
    blend_sys_tier: str | None = None
    blend_sys_score: float | None = None
    blend_user_score: float | None = None
    blend_weighted_score: float | None = None
    blend_sys_failed: bool = False

    if effective_mode == 'classifier' and label_or_tier in _LABEL_SCORES:
        (label_or_tier, blend_sys_tier, blend_sys_score,
         blend_user_score, blend_weighted_score, blend_sys_failed) = _apply_weighted_blend(
            label_or_tier, summary.system_prompt_sha256, payload, config,
            snapshot, credentials, log_tag,
        )

    # Map result to the final model string
    if effective_mode == 'tag':
        # label_or_tier is already the model/tier string from the task map
        routed = label_or_tier
        classification = None
        reason: ReasonCode = dispatch_reason  # type: ignore[assignment]
    else:
        # classifier or rules mode: label_or_tier is 'trivial'/'standard'/'deep'
        # (possibly bumped); original_label is the pre-bump raw classifier label
        classification = original_label
        routed = config.auto_model_routing_classification[label_or_tier]
        reason = dispatch_reason  # type: ignore[assignment]

    payload['model'] = routed
    return ModelRoutingDecision(
        requested_model=requested,
        routed_model=routed,
        classification=classification,
        applied=(routed != requested),
        reason_code=reason,
        estimated_input_tokens=est_tokens,
        tier_bumped=tier_bumped,
        classifier_confidence=parsed_confidence,
        classifier_mode=effective_mode,
        classifier_input_tokens=clf_in_tokens,
        classifier_output_tokens=clf_out_tokens,
        classifier_model=clf_model,
        classifier_summary_json=clf_summary_json,
        classifier_raw_response=clf_raw_response,
        classifier_format=clf_format,
        system_prompt_tier=blend_sys_tier,
        system_prompt_score=blend_sys_score,
        user_prompt_score=blend_user_score,
        routing_weighted_score=blend_weighted_score,
        system_prompt_classification_failed=blend_sys_failed,
    )
