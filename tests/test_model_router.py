"""Tests for anthproxy.model_router — LLM-based request complexity router.

Coverage:
- Eligibility: any non-empty string model is routable
- Disabled config: payload untouched, no classifier call
- Malformed/degenerate payloads fail closed without crash
- Final user text extraction (string and block forms)
- system-reminder and transcript stripping parity with handlers._strip_reminders
- transcript-only messages: fallback to last user turn, not fail-closed
- Non-text final-user blocks counted but not serialized
- Compact classifier JSON excludes system, tool schemas, metadata, history text
- No hard overrides: thinking and effort are classifier signals, not bypasses
- Classifier payload: correct model, max_tokens=4, temperature=0, no tools/thinking
- Parser: only trivial/standard/deep accepted; everything else → None
- Routing decisions: trivial→haiku, standard→sonnet, deep→opus
- Classifier failure / invalid output keeps the original requested model
- Internal sentinel causes no-op and is never forwarded
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from anthproxy.model_router import (
    _CLASSIFIER_SYSTEM,
    RoutingSummary,
    _classify_system_prompt,
    _extract_system_prompt_preview,
    _sys_prompt_cache,
    _sys_prompt_cache_lock,
    build_classifier_payload,
    build_routing_summary,
    build_system_prompt_classifier_payload,
    classify_by_rules,
    is_model_auto_routable,
    parse_classifier_label,
    parse_classifier_score,
    parse_classifier_score_json,
    route_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(routing=True, classifier_model='haiku', long_context_threshold=190_000,
            affirmation_inherit=True, classification=None,
            auto_model_routing_long='opus[1m]', confidence_bump=False):
    cfg = MagicMock()
    cfg.auto_model_routing = routing
    cfg.auto_model_routing_classifier_model = classifier_model
    # Concrete int (not a MagicMock) so the size-floor comparison in route_model
    # is well-defined; the default is far above any small test payload.
    cfg.auto_model_routing_long_context_threshold = long_context_threshold
    # Concrete bool (not a MagicMock) so the affirmation-branch guard is
    # well-defined and individually toggleable per test.
    cfg.auto_model_routing_affirmation_inherit = affirmation_inherit
    # Concrete dict (not a MagicMock) so label->tier lookups and the no-upgrade
    # cap's reverse map are well-defined; defaults to the stock mapping.
    cfg.auto_model_routing_classification = (
        dict(classification) if classification is not None
        else {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'opus'}
    )
    # Concrete string (not a MagicMock) so the long-context floor's forced
    # model / 'off' sentinel comparison is well-defined.
    cfg.auto_model_routing_long = auto_model_routing_long
    # Concrete defaults for new routing-mode / confidence-bump fields so tests
    # that mock the classifier response as a plain string (not JSON) still work.
    cfg.auto_model_routing_confidence_bump = confidence_bump
    cfg.auto_model_routing_min_confidence = 0.0
    cfg.auto_model_routing_mode = 'classifier'
    cfg.auto_model_routing_task_tiers = None
    cfg.auto_model_routing_prior_response_summary_limit = 1000
    cfg.lock_requested_model = 'off'
    # ADR 0010/0012: weighted blend config — concrete values so comparisons work.
    cfg.auto_model_routing_system_prompt_weight = 0.30
    cfg.auto_model_routing_user_prompt_weight = 0.70
    cfg.auto_model_routing_trivial_threshold = 0.75
    cfg.auto_model_routing_standard_threshold = 1.50
    cfg.auto_model_routing_system_prompt_cache_size = 256
    cfg.auto_model_routing_system_prompt_preview_limit = 500
    return cfg


def _snapshot(backend=None, routing=True, classifier_model='haiku',
              long_context_threshold=190_000, affirmation_inherit=True,
              classification=None, auto_model_routing_long='opus[1m]',
              confidence_bump=False):
    snap = MagicMock()
    snap.config = _config(routing=routing, classifier_model=classifier_model,
                          long_context_threshold=long_context_threshold,
                          affirmation_inherit=affirmation_inherit,
                          classification=classification,
                          auto_model_routing_long=auto_model_routing_long,
                          confidence_bump=confidence_bump)
    snap.name = 'anthropic'
    if backend is None:
        backend = MagicMock()
        backend.send_message.return_value = _text_response('standard')
        if not hasattr(backend, 'send_classifier_message'):
            del backend.send_classifier_message  # force duck-typed fallback
    snap.backend = backend
    return snap


def _msg(content, role='user'):
    return {'role': role, 'content': content}


def _payload(model='sonnet', content='Hello', messages=None, **extra):
    if messages is None:
        messages = [_msg(content)]
    d = {'model': model, 'messages': messages}
    d.update(extra)
    return d


def _text_response(label: str) -> dict:
    return {
        'content': [{'type': 'text', 'text': label}],
        'stop_reason': 'end_turn',
    }


# ---------------------------------------------------------------------------
# 0. ReasonCode completeness
# ---------------------------------------------------------------------------

class TestReasonCode:
    """Verify that every expected reason code is a valid member of ReasonCode."""

    def test_session_cached_tier_is_valid_reason_code(self):
        """'session_cached_tier' must be in the ReasonCode Literal."""
        from anthproxy.model_router import ReasonCode
        import typing
        args = typing.get_args(ReasonCode)
        assert 'session_cached_tier' in args

    def test_session_cached_walkback_is_valid_reason_code(self):
        """'session_cached_walkback' must be in the ReasonCode Literal."""
        from anthproxy.model_router import ReasonCode
        import typing
        args = typing.get_args(ReasonCode)
        assert 'session_cached_walkback' in args

    def test_all_expected_reason_codes_present(self):
        """Guard against accidental removal of existing codes."""
        from anthproxy.model_router import ReasonCode
        import typing
        args = typing.get_args(ReasonCode)
        expected = {
            'disabled', 'model_not_eligible', 'malformed_payload',
            'missing_final_user_text', 'session_cached_tier', 'session_cached_walkback',
            'session_cached_tier_capped', 'session_cached_walkback_capped',
            'affirmation_inherited', 'affirmation_floored_standard',
            'affirmation_classified', 'affirmation_classifier_failed',
            'rule_title_generation',
            'classifier_trivial', 'classifier_standard', 'classifier_deep',
            'classifier_failed', 'classifier_invalid',
        }
        assert expected.issubset(set(args))

    def test_affirmation_reason_codes_are_valid(self):
        """All affirmation reason codes must be in the ReasonCode Literal."""
        from anthproxy.model_router import ReasonCode
        import typing
        args = typing.get_args(ReasonCode)
        assert 'affirmation_inherited' in args
        assert 'affirmation_floored_standard' in args  # retained for old stats rows
        assert 'affirmation_classified' in args
        assert 'affirmation_classifier_failed' in args

    def test_route_model_still_returns_missing_final_user_text_for_textless_payload(self):
        """route_model itself is unchanged: text-less payloads still fail-closed."""
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'done'},
                ]},
            ],
        }
        snap = _snapshot(routing=True)
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'missing_final_user_text'
        assert payload['model'] == 'sonnet'  # model unchanged by router
        assert decision.applied is False


# ---------------------------------------------------------------------------
# 1. Eligibility
# ---------------------------------------------------------------------------

class TestEligibility:
    @pytest.mark.parametrize('model', [
        # Short tier aliases
        'sonnet', 'opus', 'haiku', 'fable',
        # 1m / bracket variants
        'sonnet[1m]', 'sonnet:1m',
        # Full Anthropic model IDs
        'claude-sonnet-4-6', 'claude-opus-4-8',
        # Bedrock native IDs and inference profiles
        'anthropic.claude-sonnet-4-5-20250929-v1:0',
        'us.anthropic.claude-opus-4-8',
        # Mixed-case or leading-uppercase strings are still non-empty strings
        'SONNET',
        ' sonnet',  # leading space — still a non-empty string when stripped
    ])
    def test_eligible_models(self, model):
        assert is_model_auto_routable(model) is True

    @pytest.mark.parametrize('model', [
        '',     # empty string — not eligible
        None,   # not a string
        42,     # not a string
        '   ',  # whitespace-only — strips to empty
    ])
    def test_not_eligible_models(self, model):
        assert is_model_auto_routable(model) is False


# ---------------------------------------------------------------------------
# 2. Disabled config
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_disabled_leaves_model_unchanged(self):
        payload = _payload(model='sonnet')
        snap = _snapshot(routing=False)
        credentials = {}
        decision = route_model(payload, snap, credentials)
        assert payload['model'] == 'sonnet'
        assert decision.applied is False
        assert decision.reason_code == 'disabled'

    def test_disabled_no_classifier_call(self):
        backend = MagicMock()
        snap = _snapshot(backend=backend, routing=False)
        route_model(_payload(model='sonnet'), snap, {})
        backend.send_message.assert_not_called()
        backend.send_classifier_message.assert_not_called()

    def test_disabled_non_sonnet_model_preserved(self):
        payload = _payload(model='opus')
        snap = _snapshot(routing=False)
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'opus'
        assert decision.reason_code == 'disabled'


# ---------------------------------------------------------------------------
# 3. Model not eligible
# ---------------------------------------------------------------------------

class TestNotEligible:
    @pytest.mark.parametrize('model', [
        None, 42, '',
    ])
    def test_non_string_or_empty_skips_classifier(self, model):
        """Non-string or empty model values are ineligible and kept unchanged."""
        backend = MagicMock()
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model=model)
        decision = route_model(payload, snap, {})
        assert payload.get('model') == model
        assert decision.applied is False
        assert decision.reason_code == 'model_not_eligible'
        backend.send_message.assert_not_called()

    def test_opus_is_now_eligible_and_routed(self):
        """Any non-empty string model (incl. opus) is eligible for routing."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='opus', content='Hello')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'haiku'
        assert decision.applied is True
        assert decision.requested_model == 'opus'
        assert decision.reason_code == 'classifier_trivial'

    def test_full_model_id_is_eligible_and_routed(self):
        """Full Anthropic model IDs are eligible for routing."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('deep')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='claude-sonnet-4-6', content='Design a system')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'opus'
        assert decision.requested_model == 'claude-sonnet-4-6'
        assert decision.reason_code == 'classifier_deep'

    def test_standard_label_always_rewrites_to_sonnet_alias(self):
        """A 'standard' classifier label rewrites any model to the bare 'sonnet' alias."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='opus', content='Normal task')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.routed_model == 'sonnet'
        assert decision.requested_model == 'opus'
        # applied is True because sonnet != opus
        assert decision.applied is True


# ---------------------------------------------------------------------------
# 3b. Long-context size floor
# ---------------------------------------------------------------------------

def _big_text(n_chars: int) -> str:
    """A text blob of ``n_chars`` characters (≈ n_chars/4 estimated tokens)."""
    return 'x' * n_chars


def _classifier_backend(label='standard'):
    """A backend whose only classification path is the duck-typed send_message."""
    backend = MagicMock()
    del backend.send_classifier_message  # force send_message fallback
    backend.send_message.return_value = _text_response(label)
    return backend


class TestLongContextSizeFloor:
    """Deterministic size floor: huge requests are forced to opus[1m]."""

    def _ctx_token(self, betas):
        return [b for b in betas if isinstance(b, str) and b.startswith('context-1m')]

    def test_floor_forces_opus_1m_and_injects_beta(self):
        backend = _classifier_backend('trivial')  # would route trivial→haiku if reached
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'opus[1m]'
        assert decision.routed_model == 'opus[1m]'
        assert decision.classification is None
        assert decision.applied is True
        assert decision.reason_code == 'size_forced_long_context'
        # context-1m beta injected exactly once
        assert self._ctx_token(payload['_anthropic_beta']) == ['context-1m-2025-08-07']
        # classifier never consulted
        backend.send_message.assert_not_called()

    def test_floor_does_not_duplicate_existing_beta(self):
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000),
                           _anthropic_beta=['oauth-2025-04-20', 'context-1m-2025-08-07'])
        route_model(payload, snap, {})
        assert self._ctx_token(payload['_anthropic_beta']) == ['context-1m-2025-08-07']
        # pre-existing non-context beta preserved
        assert 'oauth-2025-04-20' in payload['_anthropic_beta']

    def test_floor_preserves_dated_client_beta_revision(self):
        """Any context-1m* token counts as present; the floor must not add another."""
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000),
                           _anthropic_beta=['context-1m-2099-12-31'])
        route_model(payload, snap, {})
        assert self._ctx_token(payload['_anthropic_beta']) == ['context-1m-2099-12-31']

    def test_floor_creates_beta_list_when_absent(self):
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        assert '_anthropic_beta' not in payload
        route_model(payload, snap, {})
        assert payload['_anthropic_beta'] == ['context-1m-2025-08-07']

    def test_floor_fires_on_text_alone_with_image_block_present(self):
        """The estimator counts text only; a big text block fires even alongside images."""
        snap = _snapshot(routing=True, long_context_threshold=10)
        content = [
            {'type': 'text', 'text': _big_text(4000)},
            {'type': 'image', 'source': {'type': 'base64', 'data': 'AAAA'}},
        ]
        payload = _payload(model='sonnet', messages=[_msg(content)])
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        assert payload['model'] == 'opus[1m]'

    def test_floor_fires_on_textless_continuation_turn(self):
        """A large tool_result-only final turn still needs 1m; floor must fire."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=10)
        content = [{'type': 'tool_result', 'tool_use_id': 't1',
                    'content': _big_text(4000)}]
        payload = _payload(model='sonnet', messages=[_msg(content)])
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        backend.send_message.assert_not_called()

    def test_below_threshold_falls_through_to_classifier(self):
        backend = _classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)  # default 190_000
        payload = _payload(model='opus', content='small request')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_standard'
        assert payload['model'] == 'sonnet'
        assert '_anthropic_beta' not in payload  # floor did not inject
        backend.send_message.assert_called_once()

    def test_threshold_zero_disables_floor(self):
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=0)
        payload = _payload(model='sonnet', content=_big_text(40000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_trivial'
        assert payload['model'] == 'haiku'
        backend.send_message.assert_called_once()

    def test_disabled_routing_keeps_floor_inert(self):
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=False, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'disabled'
        assert payload['model'] == 'sonnet'
        backend.send_message.assert_not_called()

    def test_sentinel_payload_skips_floor(self):
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        payload['_anthproxy_internal_classifier'] = True
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'disabled'
        assert payload['model'] == 'sonnet'

    def test_non_eligible_model_skips_floor(self):
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='', content=_big_text(4000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'model_not_eligible'

    def test_size_forced_long_context_is_valid_reason_code(self):
        from anthproxy.model_router import ReasonCode
        import typing
        assert 'size_forced_long_context' in typing.get_args(ReasonCode)

    def test_decision_carries_estimated_input_tokens(self):
        """Routed decisions expose the raw estimate for handler-side calibration."""
        backend = _classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)  # default 190_000
        payload = _payload(model='sonnet', content=_big_text(8000))  # 2000 est
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_standard'
        assert decision.estimated_input_tokens == 2000

    # --- session-aware signals -------------------------------------------------

    def test_session_floor_forces_even_when_estimate_small(self):
        """A large cached session floor trips the floor on a tiny request."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=190_000)
        payload = _payload(model='sonnet', content='continue')  # est ~2 tokens
        decision = route_model(payload, snap, {},
                               session_context_tokens=195_000)
        assert decision.reason_code == 'size_forced_long_context'
        assert payload['model'] == 'opus[1m]'
        backend.send_message.assert_not_called()

    def test_ratio_inflates_estimate_over_threshold(self):
        """A sub-threshold estimate is forced once scaled by the session ratio."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=190_000)
        # 600k chars → 150k raw est; *1.4 = 210k ≥ 190k.
        payload = _payload(model='sonnet', content=_big_text(600_000))
        decision = route_model(payload, snap, {}, session_estimate_ratio=1.4)
        assert decision.reason_code == 'size_forced_long_context'
        backend.send_message.assert_not_called()

    def test_max_picks_larger_of_estimate_and_floor(self):
        """When neither signal alone crosses, the larger still decides (here: neither)."""
        backend = _classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=190_000)
        payload = _payload(model='sonnet', content=_big_text(400_000))  # 100k est
        decision = route_model(payload, snap, {},
                               session_context_tokens=120_000,
                               session_estimate_ratio=1.0)
        # max(100k, 120k) = 120k < 190k → not forced.
        assert decision.reason_code == 'classifier_standard'

    def test_default_session_signals_reproduce_estimate_only(self):
        """Defaults (0, 1.0) behave exactly like the estimate-only floor."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        decision = route_model(payload, snap, {})  # no session args
        assert decision.reason_code == 'size_forced_long_context'


class TestCalibratedRatio:
    def test_normal_ratio(self):
        from anthproxy.model_router import calibrated_ratio
        assert round(calibrated_ratio(34_000, 20_000, 1.0), 3) == 1.7

    def test_clamped_to_max(self):
        from anthproxy.model_router import calibrated_ratio, _RATIO_MAX
        assert calibrated_ratio(200_000, 20_000, 1.0) == _RATIO_MAX

    def test_never_below_one(self):
        from anthproxy.model_router import calibrated_ratio
        # estimate over-counts (prose) → raw ratio < 1 → clamped to 1.0
        assert calibrated_ratio(10_000, 20_000, 1.0) == 1.0

    def test_small_baseline_keeps_prior(self):
        from anthproxy.model_router import calibrated_ratio
        # baseline below _RATIO_MIN_BASELINE → ignore noisy ratio, keep prior
        assert calibrated_ratio(30, 6, 1.4) == 1.4

    def test_zero_measured_keeps_prior(self):
        from anthproxy.model_router import calibrated_ratio
        assert calibrated_ratio(0, 50_000, 1.5) == 1.5


# ---------------------------------------------------------------------------
# 4. Internal sentinel prevents recursion
# ---------------------------------------------------------------------------

class TestSentinel:
    def test_sentinel_payload_is_noop(self):
        payload = _payload(model='sonnet')
        payload['_anthproxy_internal_classifier'] = True
        snap = _snapshot(routing=True)
        original_model = payload['model']
        decision = route_model(payload, snap, {})
        assert payload['model'] == original_model
        assert decision.applied is False
        assert decision.reason_code == 'disabled'

    def test_classifier_payload_carries_sentinel(self):
        summary = RoutingSummary(
            final_user_text='hello',
            text_truncated=False,
            total_messages=1,
            prior_user_messages=0,
            prior_assistant_messages=0,
            tool_use_count=0,
            tool_result_count=0,
            final_non_text_blocks=0,
            has_images=False,
        )
        cfg = _config()
        cp = build_classifier_payload(summary, cfg)
        assert cp.get('_anthproxy_internal_classifier') is True

    def test_sentinel_not_forwarded_to_anthropic_mapper(self):
        """Anthropic mapper strips _anthproxy_internal_classifier before building body."""
        from anthproxy.anthropic.mapper import _build_body
        payload = {
            '_anthproxy_internal_classifier': True,
            '_anthropic_beta': [],
            'model': 'haiku',
            'max_tokens': 4,
            'temperature': 0.0,
            'system': 'sys',
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert '_anthproxy_internal_classifier' not in body
        assert '_anthropic_beta' not in body

    def test_sentinel_not_forwarded_to_local_mapper(self):
        """Local mapper strips _anthproxy_internal_classifier before building body."""
        from anthproxy.local.mapper import _build_body
        payload = {
            '_anthproxy_internal_classifier': True,
            '_anthropic_beta': [],
            'model': 'haiku',
            'max_tokens': 4,
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
        body = json.loads(_build_body(payload))
        assert '_anthproxy_internal_classifier' not in body


# ---------------------------------------------------------------------------
# 5. Thinking / effort are classifier signals, not bypasses
# ---------------------------------------------------------------------------

class TestHardOverrides:
    def test_thinking_enabled_falls_through_to_classifier(self):
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response('trivial')
        payload = _payload(
            model='sonnet',
            **{'thinking': {'type': 'enabled', 'budget_tokens': 1000}},
        )
        snap = _snapshot(backend=backend, routing=True)
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'haiku'
        assert decision.reason_code == 'classifier_trivial'
        assert decision.classification == 'trivial'
        backend.send_classifier_message.assert_called_once()

    def test_thinking_adaptive_falls_through_to_classifier(self):
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response('standard')
        payload = _payload(model='sonnet', **{'thinking': {'type': 'adaptive'}})
        snap = _snapshot(backend=backend, routing=True)
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.reason_code == 'classifier_standard'
        assert decision.classification == 'standard'
        backend.send_classifier_message.assert_called_once()

    def test_thinking_disabled_falls_through_to_classifier(self):
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', **{'thinking': {'type': 'disabled'}})
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_standard'
        backend.send_classifier_message.assert_called_once()

    @pytest.mark.parametrize('effort', ['high', 'xhigh', 'max'])
    def test_high_effort_falls_through_to_classifier(self, effort):
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response('deep')
        payload = _payload(model='sonnet', **{'output_config': {'effort': effort}})
        snap = _snapshot(backend=backend, routing=True)
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'opus'
        assert decision.reason_code == 'classifier_deep'
        assert decision.classification == 'deep'
        backend.send_classifier_message.assert_called_once()

    @pytest.mark.parametrize('effort', ['low', 'medium', None])
    def test_low_effort_falls_through_to_classifier(self, effort):
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)
        if effort is not None:
            payload = _payload(model='sonnet', **{'output_config': {'effort': effort}})
        else:
            payload = _payload(model='sonnet')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_standard'
        backend.send_classifier_message.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Routing summary extraction
# ---------------------------------------------------------------------------

class TestBuildRoutingSummary:
    def test_simple_string_content(self):
        payload = _payload(content='What is 2+2?')
        s = build_routing_summary(payload)
        assert s is not None
        assert s.final_user_text == 'What is 2+2?'
        assert s.text_truncated is False
        assert s.total_messages == 1

    def test_list_content_text_blocks_concatenated(self):
        payload = _payload(messages=[_msg([
            {'type': 'text', 'text': 'Hello'},
            {'type': 'text', 'text': 'World'},
        ])])
        s = build_routing_summary(payload)
        assert s is not None
        assert 'Hello' in s.final_user_text
        assert 'World' in s.final_user_text

    def test_reminder_stripped_from_string(self):
        payload = _payload(content='<system-reminder>ctx</system-reminder>\nWrite tests')
        s = build_routing_summary(payload)
        assert s is not None
        assert 'system-reminder' not in s.final_user_text
        assert 'Write tests' in s.final_user_text

    def test_reminder_with_attributes_stripped_from_string(self):
        payload = _payload(
            content='<system-reminder data-source="cli">ctx</system-reminder>\nWrite tests'
        )
        s = build_routing_summary(payload)
        assert s is not None
        assert 'system-reminder' not in s.final_user_text
        assert 'ctx' not in s.final_user_text
        assert 'Write tests' in s.final_user_text

    def test_reminder_with_tag_whitespace_stripped(self):
        payload = _payload(content='<system-reminder >ctx</system-reminder>\nWrite tests')
        s = build_routing_summary(payload)
        assert s is not None
        assert 'system-reminder' not in s.final_user_text
        assert 'Write tests' in s.final_user_text

    def test_reminder_case_insensitive_stripped(self):
        payload = _payload(content='<System-Reminder>ctx</SYSTEM-REMINDER>\nWrite tests')
        s = build_routing_summary(payload)
        assert s is not None
        assert 'System-Reminder' not in s.final_user_text
        assert 'ctx' not in s.final_user_text
        assert 'Write tests' in s.final_user_text

    def test_unclosed_reminder_suffix_stripped(self):
        payload = _payload(
            content='Write tests\n<system-reminder data-source="cli">ctx that should not leak'
        )
        s = build_routing_summary(payload)
        assert s is not None
        assert s.final_user_text == 'Write tests'

    def test_reminder_stripped_from_blocks(self):
        payload = _payload(messages=[_msg([
            {'type': 'text', 'text': '<system-reminder>injected</system-reminder>'},
            {'type': 'text', 'text': 'Write tests'},
        ])])
        s = build_routing_summary(payload)
        assert s is not None
        assert 'system-reminder' not in s.final_user_text
        assert 'Write tests' in s.final_user_text

    def test_reminder_only_yields_none(self):
        payload = _payload(content='<system-reminder>only reminder</system-reminder>')
        assert build_routing_summary(payload) is None

    def test_unclosed_reminder_prefix_yields_none(self):
        payload = _payload(content='<system-reminder data-source="cli">ctx\nWrite tests')
        assert build_routing_summary(payload) is None

    def test_local_command_wrappers_stripped_from_blocks(self):
        # Reproduces the /clear-then-"hi" case: Claude Code injects local-command
        # wrapper blocks before the real user text.  Only "hi" should survive.
        payload = _payload(messages=[_msg([
            {'type': 'text', 'text':
                '<local-command-caveat>Caveat: messages below were '
                'generated…</local-command-caveat>'},
            {'type': 'text', 'text': '<command-name>/clear</command-name>'},
            {'type': 'text', 'text': '<command-message>clear</command-message>'},
            {'type': 'text', 'text': '<command-args></command-args>'},
            {'type': 'text', 'text': '<local-command-stdout></local-command-stdout>'},
            {'type': 'text', 'text': 'hi'},
        ])])
        s = build_routing_summary(payload)
        assert s is not None
        assert s.final_user_text == 'hi'
        for tag in ('local-command-caveat', 'command-name', 'command-message',
                    'command-args', 'local-command-stdout'):
            assert tag not in s.final_user_text

    def test_local_command_wrappers_stripped_from_string(self):
        payload = _payload(
            content='<command-name>/clear</command-name>\nWrite tests')
        s = build_routing_summary(payload)
        assert s is not None
        assert 'command-name' not in s.final_user_text
        assert 'Write tests' in s.final_user_text

    def test_unclosed_local_command_wrapper_tag_not_stripped(self):
        # Local-command wrapper tags intentionally keep closed-block-only
        # semantics so malformed wrappers still block command interception.
        payload = _payload(content='<command-name>/clear\nWrite tests')
        s = build_routing_summary(payload)
        assert s is not None
        assert '<command-name>/clear' in s.final_user_text

    def test_text_truncated_to_limit(self):
        long_text = 'a' * 5000
        payload = _payload(content=long_text)
        s = build_routing_summary(payload)
        assert s is not None
        assert s.text_truncated is True
        assert len(s.final_user_text) == 4000

    # -----------------------------------------------------------------------
    # Transcript stripping
    # -----------------------------------------------------------------------

    def test_transcript_stripped_from_string(self):
        # Instruction is outside (after) the transcript block.
        content = (
            '<transcript>User: ls\nAssistant: file.py</transcript>\n'
            'Run graphify'
        )
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert 'transcript' not in s.final_user_text
        assert 'Run graphify' in s.final_user_text

    def test_transcript_stripped_from_block_content(self):
        # Transcript in one block, instruction in another.
        payload = _payload(messages=[_msg([
            {'type': 'text', 'text': '<transcript>User: hi\nAssistant: hello</transcript>'},
            {'type': 'text', 'text': 'Run graphify'},
        ])])
        s = build_routing_summary(payload)
        assert s is not None
        assert 'transcript' not in s.final_user_text
        assert 'Run graphify' in s.final_user_text

    def test_unclosed_transcript_stripped(self):
        # No closing tag — regex strips to end of string; instruction precedes it.
        content = 'Run graphify\n<transcript>User: earlier turn'
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert 'transcript' not in s.final_user_text
        assert 'Run graphify' in s.final_user_text

    def test_transcript_only_string_fallback_to_last_user_turn(self):
        # Entire message is a transcript block → fallback extracts last User: turn.
        content = (
            '<transcript>'
            'User: do something complex\n'
            'Assistant: done\n'
            'User: run graphify\n'
            'Assistant: ok'
            '</transcript>'
        )
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert 'run graphify' in s.final_user_text.lower()
        assert 'transcript' not in s.final_user_text

    def test_transcript_only_block_fallback_to_last_user_turn(self):
        # Transcript in a single block, no other text blocks → same fallback.
        payload = _payload(messages=[_msg([
            {'type': 'text', 'text': (
                '<transcript>'
                'User: do something\n'
                'Assistant: done\n'
                'User: run graphify'
                '</transcript>'
            )},
        ])])
        s = build_routing_summary(payload)
        assert s is not None
        assert 'run graphify' in s.final_user_text.lower()

    def test_transcript_only_unparseable_markers_returns_tail(self):
        # No role markers inside the transcript → fallback returns tail content.
        content = '<transcript>some prior conversation blob without markers</transcript>'
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert 'conversation blob' in s.final_user_text

    def test_transcript_only_empty_inner_yields_none(self):
        # Empty transcript and no other text → None (fail-closed).
        content = '<transcript>   </transcript>'
        assert build_routing_summary(_payload(content=content)) is None

    def test_transcript_fallback_last_user_turn_truncated_to_tail(self):
        # Transcript-only message whose last User: turn is > 1000 chars → the
        # fallback keeps only its trailing 1000 chars (tail, not head).
        long_turn = 'HEADMARKER ' + ('a' * 1500) + ' TAILMARKER'
        content = (
            '<transcript>'
            'User: earlier short turn\n'
            'Assistant: ok\n'
            f'User: {long_turn}'
            '</transcript>'
        )
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert len(s.final_user_text) <= 1000
        assert 'TAILMARKER' in s.final_user_text
        assert 'HEADMARKER' not in s.final_user_text

    def test_transcript_fallback_unparseable_truncated_to_tail(self):
        # Transcript-only, no role markers, inner > 1000 chars → tail kept.
        inner = 'HEADMARKER ' + ('b' * 1500) + ' TAILMARKER'
        content = f'<transcript>{inner}</transcript>'
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert len(s.final_user_text) <= 1000
        assert 'TAILMARKER' in s.final_user_text
        assert 'HEADMARKER' not in s.final_user_text

    def test_transcript_does_not_inflate_prompt_chars(self):
        # A large transcript must not cause prompt_chars to reach 4000 when the
        # real instruction is short and outside the block.
        big_history = 'User: x\nAssistant: y\n' * 500  # >> 4000 chars of history
        content = f'<transcript>{big_history}</transcript>\nRun graphify'
        s = build_routing_summary(_payload(content=content))
        assert s is not None
        assert len(s.final_user_text) < 100  # only "Run graphify" remains
        assert 'Run graphify' in s.final_user_text

    def test_non_text_blocks_counted_not_serialized(self):
        payload = _payload(messages=[_msg([
            {'type': 'text', 'text': 'Look at this image'},
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAAA'}},
        ])])
        s = build_routing_summary(payload)
        assert s is not None
        assert s.final_non_text_blocks == 1
        assert s.has_images is True
        assert 'AAAA' not in s.final_user_text
        assert 'AAAA' not in s.to_classifier_json()

    def test_tool_use_counted_in_messages(self):
        messages = [
            {'role': 'user', 'content': 'Run a tool'},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {'cmd': 'ls'}}
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'file.py'},
                {'type': 'text', 'text': 'What next?'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert s.tool_use_count == 1
        assert s.tool_result_count == 1

    def test_empty_messages_returns_none(self):
        assert build_routing_summary({'model': 'sonnet', 'messages': []}) is None

    def test_missing_messages_returns_none(self):
        assert build_routing_summary({'model': 'sonnet'}) is None

    def test_non_list_messages_returns_none(self):
        assert build_routing_summary({'model': 'sonnet', 'messages': 'bad'}) is None

    def test_non_user_final_message_returns_none(self):
        payload = {'model': 'sonnet', 'messages': [
            {'role': 'assistant', 'content': 'I am the assistant'},
        ]}
        assert build_routing_summary(payload) is None

    def test_system_not_included_in_classifier_json(self):
        payload = _payload(content='Help me')
        payload['system'] = 'You are a helpful assistant with a long system prompt.'
        s = build_routing_summary(payload)
        assert s is not None
        j = s.to_classifier_json()
        assert 'You are a helpful assistant' not in j

    def test_tool_schemas_not_included_in_classifier_json(self):
        payload = _payload(content='Execute a command')
        payload['tools'] = [
            {'name': 'execute_shell', 'description': 'Run a shell command', 'input_schema': {'type': 'object'}}
        ]
        s = build_routing_summary(payload)
        assert s is not None
        j = s.to_classifier_json()
        # Tool names/descriptions/schemas are never serialized; tool availability
        # and count are no longer included at all (non-discriminative signals).
        assert 'execute_shell' not in j
        assert 'Run a shell command' not in j
        assert 'has_top_level_tools' not in j
        assert 'top_level_tool_count' not in j

    def test_prior_history_text_not_in_classifier_json(self):
        messages = [
            _msg('First user message with secret content'),
            {'role': 'assistant', 'content': 'Response with details'},
            _msg('Follow up question'),
        ]
        payload = {'model': 'sonnet', 'messages': messages}
        s = build_routing_summary(payload)
        assert s is not None
        j = s.to_classifier_json()
        assert 'First user message with secret content' not in j
        assert 'Response with details' not in j
        assert 'Follow up question' in j

    def test_metadata_not_in_classifier_json(self):
        payload = _payload(content='Help')
        payload['metadata'] = {'user_id': 'u123', 'session_id': 's456'}
        s = build_routing_summary(payload)
        assert s is not None
        j = s.to_classifier_json()
        assert 'u123' not in j
        assert 's456' not in j

    def test_thinking_and_effort_not_in_classifier_json(self):
        # thinking / effort are no longer serialized: they are effectively
        # always-on in the Claude Code envelope and only bias the classifier.
        payload = _payload(
            content='Solve this',
            **{
                'thinking': {'type': 'enabled', 'budget_tokens': 100},
                'output_config': {'effort': 'high'},
            },
        )
        s = build_routing_summary(payload)
        assert s is not None
        j = s.to_classifier_json()
        assert 'thinking_requested' not in j
        assert 'effort' not in j

    def test_prior_message_counts(self):
        messages = [
            _msg('Turn 1 user'),
            {'role': 'assistant', 'content': 'Turn 1 assistant'},
            _msg('Turn 2 user'),
            {'role': 'assistant', 'content': 'Turn 2 assistant'},
            _msg('Turn 3 user (final)'),
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert s.total_messages == 5
        assert s.prior_user_messages == 2
        assert s.prior_assistant_messages == 2

    # -----------------------------------------------------------------------
    # Walk-back over prior messages (text-less continuation turns)
    # -----------------------------------------------------------------------

    def test_walkback_recovers_prior_user_instruction(self):
        # Final message is tool_result-only → classify on the real instruction
        # a few turns back instead of fail-closing.
        messages = [
            _msg('redesign the auth layer for SSO and multi-tenant isolation'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {'cmd': 'ls'}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'auth.py'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'redesign the auth layer' in s.final_user_text

    def test_walkback_skips_intervening_tool_result_only_user_turn(self):
        # The most-recent user turn is tool_result-only; walk-back keeps going
        # to the earlier user turn that actually has text.
        messages = [
            _msg('add cursor-based pagination to the list users API'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'ok'},
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu2', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu2', 'content': 'ok'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'cursor-based pagination' in s.final_user_text

    def test_walkback_skips_assistant_messages(self):
        # Assistant text turns must never be used as the classifier input.
        messages = [
            _msg('fix the login bug'),
            {'role': 'assistant', 'content': 'assistant prose that must be ignored'},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'done'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'fix the login bug' in s.final_user_text
        assert 'assistant prose' not in s.final_user_text

    def test_walkback_head_capped_to_1000(self):
        # A large recovered prior turn keeps only its leading 1000 chars.
        # This preserves intent (which lives at the head of prompts) and avoids
        # boilerplate (which lives at the tail).
        long_turn = 'HEADMARKER ' + ('a' * 1500) + ' TAILMARKER'
        messages = [
            _msg(long_turn),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'ok'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert len(s.final_user_text) <= 1000
        assert 'HEADMARKER' in s.final_user_text
        assert 'TAILMARKER' not in s.final_user_text
        assert s.recovered_via_walkback is True

    def test_walkback_strips_reminders_from_recovered_text(self):
        messages = [
            _msg(
                '<system-reminder>internal note</system-reminder>\n'
                'refactor the payment module'
            ),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'ok'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'refactor the payment module' in s.final_user_text
        assert 'internal note' not in s.final_user_text
        assert 'system-reminder' not in s.final_user_text

    def test_final_text_wins_over_walkback(self):
        # When the final message has its own usable text, walk-back is not used.
        messages = [
            _msg('redesign the auth layer'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'ok'},
                {'type': 'text', 'text': 'now also write the tests'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'now also write the tests' in s.final_user_text
        assert 'redesign the auth layer' not in s.final_user_text

    def test_final_transcript_fallback_wins_over_walkback(self):
        # The final message's own transcript is more recent than a prior turn.
        messages = [
            _msg('prior plain instruction'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            _msg(
                '<transcript>User: do the recent thing\nAssistant: ok</transcript>'
            ),
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'do the recent thing' in s.final_user_text
        assert 'prior plain instruction' not in s.final_user_text

    def test_walkback_no_prior_user_text_returns_none(self):
        # Final tool_result-only and no prior user turn with text → fail-closed.
        messages = [
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
            ]},
        ]
        assert build_routing_summary({'model': 'sonnet', 'messages': messages}) is None

    def test_walkback_source_does_not_affect_final_shape_fields(self):
        # final_non_text_blocks / has_images describe the FINAL message only,
        # even when the classifier text came from a prior turn.
        messages = [
            _msg('fix the bug'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
                {'type': 'tool_result', 'tool_use_id': 'tu2', 'content': 'more'},
            ]},
        ]
        s = build_routing_summary({'model': 'sonnet', 'messages': messages})
        assert s is not None
        assert 'fix the bug' in s.final_user_text
        assert s.final_non_text_blocks == 2  # the two tool_result blocks
        assert s.has_images is False


# ---------------------------------------------------------------------------
# 6b. Classifier system prompt
# ---------------------------------------------------------------------------

class TestClassifierSystemPrompt:
    def test_planning_floored_to_standard(self):
        # Regression guard: the prompt must instruct that planning/design
        # requests are at least 38 (standard floor), never trivial.
        text = _CLASSIFIER_SYSTEM.lower()
        assert 'plan' in text
        assert 'never' in text
        assert '38' in text  # numeric floor instruction
        assert 'plan how to add a logout button" → 45' in _CLASSIFIER_SYSTEM


# ---------------------------------------------------------------------------
# 7. Classifier payload construction
# ---------------------------------------------------------------------------

class TestBuildClassifierPayload:
    def _make_summary(self, text='hello'):
        return RoutingSummary(
            final_user_text=text,
            text_truncated=False,
            total_messages=1,
            prior_user_messages=0,
            prior_assistant_messages=0,
            tool_use_count=0,
            tool_result_count=0,
            final_non_text_blocks=0,
            has_images=False,
        )

    def test_uses_configured_classifier_model(self):
        cfg = _config(classifier_model='haiku')
        cp = build_classifier_payload(self._make_summary(), cfg)
        assert cp['model'] == 'haiku'

    def test_max_tokens_is_tiny(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert cp['max_tokens'] == 8

    def test_temperature_is_zero(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert cp['temperature'] == 0.0

    def test_no_stream_key(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert 'stream' not in cp

    def test_no_tools_key(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert 'tools' not in cp

    def test_no_thinking_key(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert 'thinking' not in cp

    def test_no_output_config_key(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert 'output_config' not in cp

    def test_has_system_prompt(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert isinstance(cp.get('system'), str) and len(cp['system']) > 0

    def test_single_user_message(self):
        cp = build_classifier_payload(self._make_summary('hi'), _config())
        assert cp['messages'] == [{'role': 'user', 'content': cp['messages'][0]['content']}]
        content = cp['messages'][0]['content']
        # content must be valid JSON containing the routing summary
        parsed = json.loads(content)
        assert 'final_user_text' in parsed
        assert parsed['final_user_text'] == 'hi'

    def test_carries_sentinel(self):
        cp = build_classifier_payload(self._make_summary(), _config())
        assert cp.get('_anthproxy_internal_classifier') is True

    def test_always_on_signals_absent_from_classifier_json(self):
        """The four non-discriminative always-on signals are never serialized,
        but the discriminative final_user_text still is."""
        cp = build_classifier_payload(self._make_summary('hi'), _config())
        parsed = json.loads(cp['messages'][0]['content'])
        for key in ('has_top_level_tools', 'top_level_tool_count',
                    'thinking_requested', 'effort'):
            assert key not in parsed
        assert parsed['final_user_text'] == 'hi'

    def test_prior_response_summary_injected_into_json(self):
        cfg = _config(confidence_bump=True)
        cp = build_classifier_payload(
            self._make_summary('do the thing'),
            cfg,
            prior_response_summary='Here is my plan...',
        )
        parsed = json.loads(cp['messages'][0]['content'])
        assert parsed.get('prior_response_summary') == 'Here is my plan...'

    def test_prior_response_summary_extends_system_prompt(self):
        cfg = _config(confidence_bump=True)
        cp_without = build_classifier_payload(self._make_summary(), cfg)
        cp_with = build_classifier_payload(
            self._make_summary(), cfg, prior_response_summary='some context'
        )
        assert len(cp_with['system']) > len(cp_without['system'])
        assert 'prior_response_summary' in cp_with['system']

    def test_prior_response_summary_absent_without_confidence_bump(self):
        # suffix is only appended when use_json (confidence_bump) is True
        cfg = _config(confidence_bump=False)
        cp = build_classifier_payload(
            self._make_summary(), cfg, prior_response_summary='ctx'
        )
        # JSON still injected into message content
        parsed = json.loads(cp['messages'][0]['content'])
        assert parsed.get('prior_response_summary') == 'ctx'
        # but system prompt unchanged (non-JSON mode has no suffix)
        cp_without = build_classifier_payload(self._make_summary(), cfg)
        assert cp['system'] == cp_without['system']


# ---------------------------------------------------------------------------
# 8. Parser
# ---------------------------------------------------------------------------

class TestParseClassifierLabel:
    @pytest.mark.parametrize('label', ['trivial', 'standard', 'deep'])
    def test_valid_labels_accepted(self, label):
        resp = _text_response(label)
        assert parse_classifier_label(resp) == label

    @pytest.mark.parametrize('label', ['trivial', 'standard', 'deep'])
    def test_valid_labels_case_insensitive(self, label):
        resp = _text_response(label.upper())
        assert parse_classifier_label(resp) == label

    @pytest.mark.parametrize('label', ['trivial', 'standard', 'deep'])
    def test_valid_labels_with_surrounding_whitespace(self, label):
        resp = _text_response(f'  {label}  ')
        assert parse_classifier_label(resp) == label

    @pytest.mark.parametrize('bad', [
        'trivial standard',    # two words
        'yes',                 # unknown word
        '',                    # empty
        '{"label": "deep"}',  # JSON object (two alpha tokens)
        'The answer is deep',  # sentence
        'deep trivial',        # two valid labels
        'not deep',            # negation / two tokens
    ])
    def test_invalid_labels_rejected(self, bad):
        resp = _text_response(bad)
        assert parse_classifier_label(resp) is None

    @pytest.mark.parametrize('wrapped,label', [
        ('trivial.', 'trivial'),       # trailing period
        ('deep!', 'deep'),             # trailing exclamation
        ('**deep**', 'deep'),          # markdown bold
        ('"standard"', 'standard'),    # quoted
        ('`trivial`', 'trivial'),      # inline code
        ('deep\n', 'deep'),            # trailing newline
        ('- standard', 'standard'),    # markdown list marker
        ('deep,', 'deep'),             # trailing comma
        ('(trivial)', 'trivial'),      # parenthesized
    ])
    def test_single_label_with_surrounding_punctuation_accepted(self, wrapped, label):
        # Production: haiku at temp=0 deterministically emits a single valid label
        # wrapped in punctuation/markdown (e.g. "deep.", "**deep**").  The strict
        # exact-match parser rejected these as classifier_invalid and silently
        # misrouted a genuine deep task down to the requested (e.g. sonnet) tier.
        resp = _text_response(wrapped)
        assert parse_classifier_label(resp) == label

    def test_empty_content_returns_none(self):
        assert parse_classifier_label({'content': []}) is None

    def test_missing_content_returns_none(self):
        assert parse_classifier_label({}) is None

    def test_tool_use_only_response_returns_none(self):
        resp = {
            'content': [
                {'type': 'tool_use', 'id': 't1', 'name': 'bash', 'input': {}}
            ]
        }
        assert parse_classifier_label(resp) is None

    def test_thinking_block_skipped_label_accepted(self):
        # OpenRouter's default classifier alias routes to a reasoning model that
        # emits thinking blocks alongside the text label.  The parser must skip
        # thinking blocks rather than invalidating the whole response.
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'reasoning here', 'signature': 'sig'},
                {'type': 'text', 'text': 'standard'},
            ]
        }
        assert parse_classifier_label(resp) == 'standard'

    def test_redacted_thinking_block_skipped_label_accepted(self):
        resp = {
            'content': [
                {'type': 'redacted_thinking', 'data': 'opaque'},
                {'type': 'text', 'text': 'deep'},
            ]
        }
        assert parse_classifier_label(resp) == 'deep'

    def test_thinking_only_no_text_returns_none(self):
        # Reasoning model consumed the token budget on thinking; no text block.
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig'}
            ]
        }
        assert parse_classifier_label(resp) is None

    def test_thinking_then_tool_use_returns_none(self):
        # Thinking-skip must not weaken rejection of other non-text blocks.
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'r', 'signature': 's'},
                {'type': 'tool_use', 'id': 't1', 'name': 'bash', 'input': {}},
            ]
        }
        assert parse_classifier_label(resp) is None

    def test_redacted_thinking_then_punctuation_wrapped_label(self):
        resp = {
            'content': [
                {'type': 'redacted_thinking', 'data': 'x'},
                {'type': 'text', 'text': '**deep**'},
            ]
        }
        assert parse_classifier_label(resp) == 'deep'

    def test_non_list_content_returns_none(self):
        assert parse_classifier_label({'content': 'trivial'}) is None


# ---------------------------------------------------------------------------
# 8b. parse_classifier_score — numeric 0-100 parser
# ---------------------------------------------------------------------------

class TestParseClassifierScore:
    def _resp(self, text: str) -> dict:
        return {'content': [{'type': 'text', 'text': text}]}

    @pytest.mark.parametrize('text,expected', [
        ('0', 0),
        ('42', 42),
        ('100', 100),
        ('  37  ', 37),
        ('\n75\n', 75),
        ('5', 5),
    ])
    def test_valid_integers(self, text, expected):
        assert parse_classifier_score(self._resp(text)) == expected

    def test_multiple_digit_sequences_rejected_slash(self):
        # '42/100' → re.findall returns ['42', '100'] → rejected
        assert parse_classifier_score(self._resp('42/100')) is None

    def test_multiple_digit_sequences_rejected_space(self):
        # 'score: 42 (out of 100)' → re.findall returns ['42', '100'] → rejected
        assert parse_classifier_score(self._resp('score: 42 (out of 100)')) is None

    def test_multiple_digit_sequences_rejected_two_numbers(self):
        assert parse_classifier_score(self._resp('42 42')) is None

    def test_no_digits_rejected(self):
        assert parse_classifier_score(self._resp('standard')) is None

    def test_empty_text_rejected(self):
        assert parse_classifier_score(self._resp('')) is None

    def test_out_of_range_low_rejected(self):
        # -5 is handled by negative guard; 101 is out of range
        assert parse_classifier_score(self._resp('101')) is None

    def test_out_of_range_high_rejected(self):
        assert parse_classifier_score(self._resp('200')) is None

    def test_negative_number_guard(self):
        # re.findall(r'\d+', '-5') returns ['5'] and would silently pass without guard
        assert parse_classifier_score(self._resp('-5')) is None

    def test_negative_number_guard_no_false_positive(self):
        # '-' not immediately followed by a digit; guard doesn't fire; single digit sequence
        assert parse_classifier_score(self._resp('-abc 42')) == 42

    def test_thinking_block_skipped(self):
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'reasoning here', 'signature': 'sig'},
                {'type': 'text', 'text': '55'},
            ]
        }
        assert parse_classifier_score(resp) == 55

    def test_redacted_thinking_skipped(self):
        resp = {
            'content': [
                {'type': 'redacted_thinking', 'data': 'opaque'},
                {'type': 'text', 'text': '80'},
            ]
        }
        assert parse_classifier_score(resp) == 80

    def test_all_thinking_no_text_returns_none(self):
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig'}
            ]
        }
        assert parse_classifier_score(resp) is None

    def test_empty_content_list_returns_none(self):
        assert parse_classifier_score({'content': []}) is None

    def test_missing_content_returns_none(self):
        assert parse_classifier_score({}) is None

    def test_malformed_block_missing_type(self):
        # Block without 'type' key: get() returns None, not in skip set, not 'text' → None
        resp = {'content': [{'text': '42'}]}
        assert parse_classifier_score(resp) is None

    def test_non_text_block_rejected(self):
        resp = {'content': [{'type': 'tool_use', 'id': 't1', 'name': 'bash', 'input': {}}]}
        assert parse_classifier_score(resp) is None

    def test_non_list_content_returns_none(self):
        assert parse_classifier_score({'content': 'trivial'}) is None

    def test_boundary_zero(self):
        assert parse_classifier_score(self._resp('0')) == 0

    def test_boundary_hundred(self):
        assert parse_classifier_score(self._resp('100')) == 100


# ---------------------------------------------------------------------------
# 8c. parse_classifier_score_json — JSON {"score": N} parser
# ---------------------------------------------------------------------------

class TestParseClassifierScoreJson:
    def _resp(self, text: str) -> dict:
        return {'content': [{'type': 'text', 'text': text}]}

    def test_valid_score(self):
        assert parse_classifier_score_json(self._resp('{"score":55}')) == 55

    def test_boundary_zero(self):
        assert parse_classifier_score_json(self._resp('{"score":0}')) == 0

    def test_boundary_hundred(self):
        assert parse_classifier_score_json(self._resp('{"score":100}')) == 100

    def test_out_of_range_rejects(self):
        assert parse_classifier_score_json(self._resp('{"score":101}')) is None
        assert parse_classifier_score_json(self._resp('{"score":-1}')) is None

    def test_missing_score_key_rejects(self):
        assert parse_classifier_score_json(self._resp('{"label":"standard"}')) is None

    def test_float_score_rejects(self):
        # Only int is accepted; float is not
        assert parse_classifier_score_json(self._resp('{"score":55.5}')) is None

    def test_bool_score_rejects(self):
        # bool is subclass of int; must be explicitly rejected
        assert parse_classifier_score_json(self._resp('{"score":true}')) is None

    def test_string_score_rejects(self):
        assert parse_classifier_score_json(self._resp('{"score":"55"}')) is None

    def test_malformed_json_rejects(self):
        assert parse_classifier_score_json(self._resp('not json')) is None

    def test_non_object_json_rejects(self):
        assert parse_classifier_score_json(self._resp('[55]')) is None

    def test_empty_text_rejects(self):
        assert parse_classifier_score_json(self._resp('')) is None

    def test_empty_content_list_returns_none(self):
        assert parse_classifier_score_json({'content': []}) is None

    def test_missing_content_returns_none(self):
        assert parse_classifier_score_json({}) is None

    def test_thinking_block_skipped(self):
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'sig'},
                {'type': 'text', 'text': '{"score":42}'},
            ]
        }
        assert parse_classifier_score_json(resp) == 42

    def test_all_thinking_no_text_returns_none(self):
        resp = {
            'content': [
                {'type': 'thinking', 'thinking': 'r', 'signature': 's'}
            ]
        }
        assert parse_classifier_score_json(resp) is None

    def test_malformed_block_missing_type(self):
        resp = {'content': [{'text': '{"score":42}'}]}
        assert parse_classifier_score_json(resp) is None

    def test_non_text_block_rejected(self):
        resp = {'content': [{'type': 'tool_use', 'id': 't1', 'name': 'bash', 'input': {}}]}
        assert parse_classifier_score_json(resp) is None

    def test_whitespace_around_json(self):
        assert parse_classifier_score_json(self._resp('  {"score":75}  ')) == 75


# ---------------------------------------------------------------------------
# 9. End-to-end routing decisions via route_model
# ---------------------------------------------------------------------------

class TestRouteModel:
    def _run(self, label, classifier_model='haiku', routing=True):
        backend = MagicMock()
        # Ensure send_classifier_message doesn't exist so duck-typed fallback is used
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response(label)
        snap = _snapshot(backend=backend, routing=routing, classifier_model=classifier_model)
        payload = _payload(model='sonnet', content='Do something')
        credentials = {'token': 'abc'}
        decision = route_model(payload, snap, credentials)
        return payload, decision, backend

    def test_trivial_routes_to_haiku(self):
        payload, decision, _ = self._run('trivial')
        assert payload['model'] == 'haiku'
        assert decision.routed_model == 'haiku'
        assert decision.classification == 'trivial'
        assert decision.applied is True
        assert decision.reason_code == 'classifier_trivial'

    def test_standard_keeps_sonnet(self):
        payload, decision, _ = self._run('standard')
        assert payload['model'] == 'sonnet'
        assert decision.routed_model == 'sonnet'
        assert decision.classification == 'standard'
        assert decision.applied is False
        assert decision.reason_code == 'classifier_standard'

    def test_deep_routes_to_opus(self):
        payload, decision, _ = self._run('deep')
        assert payload['model'] == 'opus'
        assert decision.routed_model == 'opus'
        assert decision.classification == 'deep'
        assert decision.applied is True
        assert decision.reason_code == 'classifier_deep'

    def test_classifier_failure_keeps_sonnet(self):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.side_effect = RuntimeError('network error')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='Something')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.applied is False
        assert decision.reason_code == 'classifier_failed'

    def test_classifier_failure_keeps_original_model_non_sonnet(self):
        """Fail-closed keeps the *original* requested model, not 'sonnet'."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.side_effect = RuntimeError('network error')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='opus', content='Something')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'opus'  # original preserved, not 'sonnet'
        assert decision.routed_model == 'opus'
        assert decision.requested_model == 'opus'
        assert decision.applied is False
        assert decision.reason_code == 'classifier_failed'

    def test_invalid_classifier_output_keeps_sonnet(self):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('maybe deep?')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='Something')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.applied is False
        assert decision.reason_code == 'classifier_invalid'

    def test_invalid_classifier_output_keeps_original_model_non_sonnet(self):
        """Fail-closed on invalid classifier output keeps the original model, not 'sonnet'."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('maybe deep?')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='claude-opus-4-8', content='Something')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'claude-opus-4-8'  # original preserved
        assert decision.routed_model == 'claude-opus-4-8'
        assert decision.applied is False
        assert decision.reason_code == 'classifier_invalid'

    def test_duck_typed_send_classifier_message_used_when_available(self):
        """When backend has send_classifier_message, it is used instead of send_message."""
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='Hello')
        route_model(payload, snap, {})
        assert payload['model'] == 'haiku'
        backend.send_classifier_message.assert_called_once()
        backend.send_message.assert_not_called()

    def test_retry_does_not_call_route_model_again(self):
        """In the real retry flow, route_model is never called again after the first dispatch.

        The handler passes the already-mutated payload directly to _dispatch on retry,
        so route_model is only ever invoked once per original request.  This test verifies
        that the handler wires it that way — calling route_model twice would double-classify.
        """
        # Verify that the model mutation from the first route_model call persists
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='Simple query')
        decision1 = route_model(payload, snap, {})
        assert payload['model'] == 'haiku'
        assert decision1.reason_code == 'classifier_trivial'
        # The payload's model is now 'haiku'; the real retry handler reuses this
        # payload directly (route_model is not called again), so the model stays 'haiku'.

    def test_missing_final_user_text_fails_closed(self):
        backend = MagicMock()
        snap = _snapshot(backend=backend, routing=True)
        payload = {'model': 'sonnet', 'messages': [{'role': 'assistant', 'content': 'oops'}]}
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.reason_code == 'missing_final_user_text'
        backend.send_message.assert_not_called()

    def test_empty_messages_fails_closed(self):
        backend = MagicMock()
        snap = _snapshot(backend=backend, routing=True)
        payload = {'model': 'sonnet', 'messages': []}
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.reason_code == 'missing_final_user_text'

    def test_malformed_messages_value_fails_closed(self):
        backend = MagicMock()
        snap = _snapshot(backend=backend, routing=True)
        payload = {'model': 'sonnet', 'messages': 'not a list'}
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'sonnet'
        assert decision.reason_code == 'missing_final_user_text'

    def test_missing_final_user_text_keeps_original_non_sonnet_model(self):
        """Fail-closed on missing text keeps the original model, not 'sonnet'."""
        backend = MagicMock()
        snap = _snapshot(backend=backend, routing=True)
        payload = {'model': 'opus', 'messages': [{'role': 'assistant', 'content': 'oops'}]}
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'opus'  # original preserved
        assert decision.routed_model == 'opus'
        assert decision.reason_code == 'missing_final_user_text'

    def test_requested_model_preserved_in_decision(self):
        payload, decision, _ = self._run('deep')
        assert decision.requested_model == 'sonnet'

    def test_credentials_passed_to_classifier(self):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('trivial')
        snap = _snapshot(backend=backend, routing=True)
        creds = {'token': 'my_token'}
        payload = _payload(model='sonnet', content='Hi')
        route_model(payload, snap, creds)
        backend.send_message.assert_called_once()
        _call = backend.send_message.call_args
        assert _call[0][1] == creds or _call.args[1] == creds

    def test_cached_session_tier_used_for_walkback_only_continuation(self):
        """When final message is text-less (walk-back) and cached_session_tier is provided,
        skip classifier and reuse the cached tier instead of re-classifying boilerplate."""
        backend = MagicMock()
        del backend.send_classifier_message
        # This should NOT be called because we skip to cache
        backend.send_message.return_value = _text_response('trivial')
        snap = _snapshot(backend=backend, routing=True)

        # Simulate a text-less continuation turn (tool_result-only final message)
        # The walk-back will recover the prior instruction text
        payload = _payload(
            model='opus',
            messages=[
                _msg('Plan the refactor'),
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
                ]},
            ]
        )

        # Pass cached_session_tier='opus' (the tier from turn 1)
        decision = route_model(payload, snap, {}, cached_session_tier='opus')

        # Should reuse cached tier without classifying
        assert payload['model'] == 'opus'
        assert decision.routed_model == 'opus'
        assert decision.applied is False  # 'opus' was requested, 'opus' returned
        assert decision.classification is None  # classifier was not called
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        # Classifier should NOT have been called
        backend.send_message.assert_not_called()

    def test_cached_session_tier_only_for_walkback_not_direct_text(self):
        """cached_session_tier is only used when the text came from walk-back.
        If the final message has direct text, classifier is called normally."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)

        # Final message has direct text (not text-less continuation)
        payload = _payload(model='opus', content='New request with direct text')

        # Even if cached_session_tier is provided, it should be ignored
        # because the text came directly, not from walk-back
        decision = route_model(payload, snap, {}, cached_session_tier='opus')

        # Should call classifier and use its result (standard → sonnet), ignoring cache
        assert payload['model'] == 'sonnet'
        assert decision.routed_model == 'sonnet'
        assert decision.classification == 'standard'
        assert decision.reason_code == 'classifier_standard'
        backend.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# 9a-cap. No-upgrade cap on cache replay
# ---------------------------------------------------------------------------

def _walkback_payload(model):
    """A text-less continuation turn (tool_result-only final message) whose
    prior user turn supplies walk-back text — drives the session_cached_walkback
    path so the cached tier is replayed."""
    return _payload(
        model=model,
        messages=[
            _msg('Plan the refactor'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
            ]},
        ],
    )


def _image_walkback_payload(model):
    """A text-less continuation turn (image-only final message) whose prior user
    turn supplies walk-back text — drives the walkback path with cap still active
    (non-tool_result, so cap applies)."""
    return _payload(
        model=model,
        messages=[
            _msg('Analyze this image'),
            {'role': 'assistant', 'content': 'I can help.'},
            {'role': 'user', 'content': [
                {'type': 'image', 'source': {
                    'type': 'base64',
                    'media_type': 'image/jpeg',
                    'data': 'base64_image_data_stub',
                }},
            ]},
        ],
    )


class TestTierRank:
    """Tier ranks for the no-upgrade cap (from model_tier.py)."""

    def test_bare_aliases_rank(self):
        from anthproxy.model_tier import model_tier_rank
        assert model_tier_rank('haiku') == 0
        assert model_tier_rank('sonnet') == 1
        assert model_tier_rank('opus') == 2
        assert model_tier_rank('fable') == 3

    def test_full_anthropic_ids_rank(self):
        from anthproxy.model_tier import model_tier_rank
        assert model_tier_rank('claude-haiku-4-5-20251001') == 0
        assert model_tier_rank('claude-opus-4-8-20251201') == 2
        assert model_tier_rank('anthropic/claude-fable-3') == 3

    def test_context_suffix_stripped(self):
        from anthproxy.model_tier import model_tier_rank
        assert model_tier_rank('opus[1m]') == 2
        assert model_tier_rank('sonnet:1m') == 1
        assert model_tier_rank('fable[1m]') == 3

    def test_unknown_models_rank_none(self):
        from anthproxy.model_tier import model_tier_rank
        assert model_tier_rank('gpt-5.5') is None
        assert model_tier_rank('plugin-model-1') is None
        assert model_tier_rank('') is None

    def test_non_string_rank_none(self):
        from anthproxy.model_tier import model_tier_rank
        assert model_tier_rank(None) is None  # type: ignore[arg-type]
        assert model_tier_rank(42) is None  # type: ignore[arg-type]

    def test_uppercase_model_rank(self):
        from anthproxy.model_tier import model_tier_rank
        assert model_tier_rank('HAIKU') == 0
        assert model_tier_rank('OPUS') == 2
        assert model_tier_rank('FABLE') == 3
        assert model_tier_rank('Claude-Sonnet-4-6') == 1


class TestNoUpgradeCap:
    """Cache replay must never route above the tier this turn requested, except
    for tool_result-only continuations (agentic) where the bypass always replays
    the cached tier."""

    def _snap(self):
        backend = MagicMock()
        del backend.send_classifier_message
        # If the classifier were ever called the cap path would be wrong.
        backend.send_message.return_value = _text_response('deep')
        return _snapshot(backend=backend, routing=True), backend

    def test_walkback_upgrade_is_capped_to_requested(self):
        """Tool_result-only walkback now bypasses cap: haiku requested, sonnet
        cached → bypass cap, use sonnet (agentic continuation)."""
        snap, backend = self._snap()
        payload = _walkback_payload('claude-haiku-4-5-20251001')
        decision = route_model(payload, snap, {}, cached_session_tier='sonnet')
        assert payload['model'] == 'sonnet'
        assert decision.routed_model == 'sonnet'
        assert decision.applied is True
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        backend.send_message.assert_not_called()

    def test_walkback_opus_cached_capped_to_haiku(self):
        """Tool_result-only walkback bypasses cap: opus cached, haiku requested
        → use opus (bypass always replays cached tier for tool_result turns)."""
        snap, _ = self._snap()
        payload = _walkback_payload('haiku')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert payload['model'] == 'opus'
        assert decision.routed_model == 'opus'
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.applied is True

    def test_walkback_equal_tier_not_capped(self):
        """Tool_result-only bypass (equal tier): requested == cached → use cached.
        Bypass returns _tool_result code for all tool_result-only turns."""
        snap, _ = self._snap()
        payload = _walkback_payload('sonnet')
        decision = route_model(payload, snap, {}, cached_session_tier='sonnet')
        assert payload['model'] == 'sonnet'
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.applied is False

    def test_walkback_downgrade_replays_uncapped(self):
        """Tool_result-only bypass (downgrade): opus requested, haiku cached
        → use haiku (bypass replays cached for all tool_result turns)."""
        snap, _ = self._snap()
        payload = _walkback_payload('opus')
        decision = route_model(payload, snap, {}, cached_session_tier='haiku')
        assert payload['model'] == 'haiku'
        assert decision.routed_model == 'haiku'
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.applied is True

    def test_walkback_unknown_requested_fails_open(self):
        """Tool_result-only bypass with unknown requested model (fable → no tier rank).
        Bypass replays cached unconditionally for tool_result turns."""
        snap, _ = self._snap()
        payload = _walkback_payload('fable')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert payload['model'] == 'opus'
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.applied is True

    def test_affirmation_inherit_is_not_capped(self):
        """A bare 'yes' is content-free assent — inheriting opus on a haiku
        request is intended and must NOT be capped."""
        snap, backend = self._snap()
        payload = _payload(model='haiku', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert payload['model'] == 'opus'
        assert decision.routed_model == 'opus'
        assert decision.reason_code == 'affirmation_inherited'
        backend.send_message.assert_not_called()

    def test_cap_helper_upgrade_only(self):
        from anthproxy.model_router import _cap_cached_tier
        # Upgrade blocked
        assert _cap_cached_tier('sonnet', 'claude-haiku-4-5-20251001') == 'claude-haiku-4-5-20251001'
        assert _cap_cached_tier('opus', 'haiku') == 'haiku'
        # Equal / downgrade / unknown → replay cached
        assert _cap_cached_tier('sonnet', 'sonnet') == 'sonnet'
        assert _cap_cached_tier('haiku', 'opus') == 'haiku'
        assert _cap_cached_tier('opus', 'fable') == 'opus'


# ---------------------------------------------------------------------------
# 9a-label. _cap_cached_tier with a custom label_map (dynamic classification)
# ---------------------------------------------------------------------------

class TestCapCachedTierWithLabelMap:
    """The no-upgrade cap's reverse-lookup path when a custom label_map is given."""

    def test_fable_deep_cached_haiku_requested(self):
        from anthproxy.model_router import _cap_cached_tier
        label_map = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        # cached='fable' resolves to deep (rank=2) via the reverse map; requested
        # 'haiku' is rank=0 → cap fires (2 > 0) → capped down to requested.
        assert _cap_cached_tier('fable', 'haiku', label_map=label_map) == 'haiku'

    def test_fable_cached_opus_requested(self):
        from anthproxy.model_router import _cap_cached_tier
        label_map = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        # cached='fable' → deep (rank=2); requested='opus' is also rank=2 →
        # ranks equal, not an upgrade → replay cached verbatim.
        assert _cap_cached_tier('fable', 'opus', label_map=label_map) == 'fable'

    def test_no_label_map_fable_cached_caps_to_requested(self):
        from anthproxy.model_router import _cap_cached_tier
        # Without label_map, 'fable' has rank 3 (from model_tier_rank) →
        # cached_rank (3) > requested_rank → cap applies.
        assert _cap_cached_tier('fable', 'haiku', label_map=None) == 'haiku'
        assert _cap_cached_tier('fable', 'opus', label_map=None) == 'opus'

    def test_sonnet_special_deep_ranked_via_exact_lookup(self):
        from anthproxy.model_router import _cap_cached_tier
        # 'sonnet-special' is mapped to the deep slot; exact lookup in the
        # reverse map must win over the substring fallback (which would
        # otherwise match 'sonnet' → rank=1).
        label_map = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'sonnet-special'}
        assert _cap_cached_tier('sonnet-special', 'haiku', label_map=label_map) == 'haiku'


# ---------------------------------------------------------------------------
# 9a-custom. route_model with a custom auto_model_routing_classification map
# ---------------------------------------------------------------------------

class TestRouteModelCustomClassification:
    """route_model honors a runtime-configured label→tier classification map."""

    def test_classifier_returns_deep_routes_to_fable(self):
        backend = _classifier_backend('deep')
        classification = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        snap = _snapshot(backend=backend, routing=True, classification=classification)
        payload = _payload(model='sonnet', content='Redesign the auth layer for SSO')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'fable'
        assert decision.routed_model == 'fable'
        assert decision.classification == 'deep'
        assert decision.reason_code == 'classifier_deep'

    def test_classifier_returns_trivial_routes_to_haiku(self):
        backend = _classifier_backend('trivial')
        classification = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        snap = _snapshot(backend=backend, routing=True, classification=classification)
        payload = _payload(model='sonnet', content='hi')
        decision = route_model(payload, snap, {})
        assert payload['model'] == 'haiku'
        assert decision.routed_model == 'haiku'
        assert decision.classification == 'trivial'
        assert decision.reason_code == 'classifier_trivial'

    def test_session_walkback_with_fable_cached(self):
        """Tool_result-only bypass: cached 'fable' (deep) is replayed even though
        label_map defines fable > haiku, because tool_result-only turns bypass
        the cap."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('deep')
        classification = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        snap = _snapshot(backend=backend, routing=True, classification=classification)
        payload = _walkback_payload('haiku')
        decision = route_model(payload, snap, {}, cached_session_tier='fable')
        assert payload['model'] == 'fable'
        assert decision.routed_model == 'fable'
        assert decision.classification is None
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.applied is True
        backend.send_message.assert_not_called()

    def test_build_routing_summary_tool_result_only_sets_flag(self):
        """final_is_tool_result_only flag: True only when ALL blocks are tool_result."""
        from anthproxy.model_router import build_routing_summary

        # Tool_result-only final message → flag=True
        payload_tool_result_only = _payload(model='sonnet', messages=[
            _msg('plan'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
            ]},
        ])
        summary = build_routing_summary(payload_tool_result_only)
        assert summary.final_is_tool_result_only is True

        # Text + tool_result mix → flag=False
        payload_mixed = _payload(model='sonnet', messages=[
            _msg('plan'),
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            ]},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'result is done'},
                {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'exit 0'},
            ]},
        ])
        summary = build_routing_summary(payload_mixed)
        assert summary.final_is_tool_result_only is False

        # Image-only final message → flag=False
        payload_image_only = _payload(model='sonnet', messages=[
            _msg('analyze this'),
            {'role': 'assistant', 'content': 'I can help.'},
            {'role': 'user', 'content': [
                {'type': 'image', 'source': {
                    'type': 'base64',
                    'media_type': 'image/jpeg',
                    'data': 'data_stub',
                }},
            ]},
        ])
        summary = build_routing_summary(payload_image_only)
        assert summary.final_is_tool_result_only is False

    def test_walkback_image_only_upgrade_still_capped(self):
        """Image-only (not tool_result-only) walkback still hits the cap:
        opus cached, haiku requested → cap fires, model stays haiku."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('deep')
        classification = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        snap = _snapshot(backend=backend, routing=True, classification=classification)
        payload = _image_walkback_payload('haiku')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert payload['model'] == 'haiku'
        assert decision.routed_model == 'haiku'
        assert decision.reason_code == 'session_cached_walkback_capped'
        assert decision.applied is False
        backend.send_message.assert_not_called()

    def test_walkback_tool_result_only_replays_fable_over_sonnet(self):
        """Production bug scenario: agentic client requests sonnet on every turn,
        but first request was routed to fable (deep). Tool_result turns should
        stay on fable, not get capped to sonnet."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('deep')
        classification = {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'fable'}
        snap = _snapshot(backend=backend, routing=True, classification=classification)
        payload = _walkback_payload('claude-sonnet-4-6')
        decision = route_model(payload, snap, {}, cached_session_tier='fable')
        assert payload['model'] == 'fable'
        assert decision.routed_model == 'fable'
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.applied is True
        backend.send_message.assert_not_called()

    def test_affirmation_floored_to_configured_standard(self):
        # No prior assistant message → critical invariant fires; uses floor without calling classifier.
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('trivial')
        classification = {'trivial': 'haiku', 'standard': 'fable', 'deep': 'opus'}
        snap = _snapshot(backend=backend, routing=True, classification=classification)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert payload['model'] == 'fable'
        assert decision.routed_model == 'fable'
        assert decision.classification is None
        assert decision.reason_code == 'affirmation_classifier_failed'
        backend.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# 3c. Long-context floor: configurable target / 'off' switch
# ---------------------------------------------------------------------------

class TestLongContextFloorOff:
    """auto_model_routing_long controls the floor's forced model, or disables it."""

    def test_floor_disabled_off(self):
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=100,
                         auto_model_routing_long='off')
        payload = _payload(model='sonnet', content=_big_text(1000))
        decision = route_model(payload, snap, {})
        # Floor is inert ('off') so the classifier runs and decides normally.
        assert decision.reason_code == 'classifier_trivial'
        assert decision.classification == 'trivial'
        assert payload['model'] == 'haiku'
        backend.send_message.assert_called_once()

    def test_floor_enabled_fable(self):
        backend = _classifier_backend('trivial')  # would fire if floor didn't
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=100,
                         auto_model_routing_long='fable')
        payload = _payload(model='sonnet', content=_big_text(1000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        assert payload['model'] == 'fable'
        assert decision.routed_model == 'fable'
        backend.send_message.assert_not_called()

    def test_floor_default_opus_1m(self):
        """Regression guard: default auto_model_routing_long stays 'opus[1m]'."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=100)
        payload = _payload(model='sonnet', content=_big_text(1000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        assert payload['model'] == 'opus[1m]'
        assert decision.routed_model == 'opus[1m]'
        backend.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# 9b. Classification prompt logging
# ---------------------------------------------------------------------------

class TestClassificationLogging:
    """route_model emits an INFO prompt preview only when the classifier runs."""

    _LOGGER = 'anthproxy.model_router'

    def _classifier_backend(self, label='standard'):
        backend = MagicMock()
        backend.send_classifier_message.return_value = _text_response(label)
        return backend

    def test_logs_prompt_preview_when_classifier_runs(self, caplog):
        backend = self._classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='Refactor the parser module')
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            route_model(payload, snap, {})
        records = [r for r in caplog.records if r.name == self._LOGGER]
        msgs = [r.getMessage() for r in records]
        assert any('Model router: classifying' in m for m in msgs)
        assert any('requested=sonnet' in m for m in msgs)
        assert any('Refactor the parser module' in m for m in msgs)

    def test_long_prompt_truncated_in_log(self, caplog):
        backend = self._classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        head = 'START' + ('x' * 1000)
        tail = 'TAILMARKER'
        payload = _payload(model='sonnet', content=head + tail)
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            route_model(payload, snap, {})
        msgs = [r.getMessage() for r in caplog.records if r.name == self._LOGGER]
        classify = [m for m in msgs if 'Model router: classifying' in m]
        assert classify
        line = classify[0]
        assert '…' in line          # truncation marker present
        assert 'TAILMARKER' not in line  # tail beyond the limit is dropped

    def test_debug_log_carries_full_untruncated_prompt(self, caplog):
        # The untruncated prompt goes to DEBUG (→ --log-file); the INFO line
        # stays preview-truncated.
        backend = self._classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        head = 'START' + ('x' * 1000)
        tail = 'TAILMARKER'
        payload = _payload(model='sonnet', content=head + tail)
        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            route_model(payload, snap, {})
        msgs = [r.getMessage() for r in caplog.records if r.name == self._LOGGER]
        debug = [m for m in msgs if 'classifier prompt' in m]
        info = [m for m in msgs if 'Model router: classifying' in m]
        assert debug
        assert 'TAILMARKER' in debug[0]       # full text in DEBUG line
        assert info and 'TAILMARKER' not in info[0]  # INFO stays truncated

    def test_no_prompt_log_when_routing_disabled(self, caplog):
        backend = self._classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=False)
        payload = _payload(model='sonnet', content='Should not be logged')
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            route_model(payload, snap, {})
        msgs = [r.getMessage() for r in caplog.records if r.name == self._LOGGER]
        assert not any('Model router: classifying' in m for m in msgs)

    def test_no_prompt_log_when_model_not_eligible(self, caplog):
        backend = self._classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='', content='Should not be logged')
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            route_model(payload, snap, {})
        msgs = [r.getMessage() for r in caplog.records if r.name == self._LOGGER]
        assert not any('Model router: classifying' in m for m in msgs)

    def test_no_prompt_log_when_missing_final_user_text(self, caplog):
        backend = self._classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = {'model': 'sonnet', 'messages': [{'role': 'assistant', 'content': 'oops'}]}
        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            route_model(payload, snap, {})
        msgs = [r.getMessage() for r in caplog.records if r.name == self._LOGGER]
        assert not any('Model router: classifying' in m for m in msgs)


# ---------------------------------------------------------------------------
# 10. Handler integration: routing fires before dispatch
# ---------------------------------------------------------------------------

class TestHandlerRouting:
    def _handler(self, backend, routing=True):
        from anthproxy.handlers import ProxyRequestHandler
        handler = object.__new__(ProxyRequestHandler)
        snap = MagicMock()
        snap.name = 'anthropic'
        snap.backend = backend
        snap.config = _config(routing=routing)
        snap.session_pinned = False
        snap.session_subscription = False
        registry = MagicMock()
        registry.snapshot.return_value = snap
        handler.registry = registry
        handler.config = snap.config
        handler.selector = None
        handler.stats_collector = None
        handler._send_json = MagicMock()
        handler._send_sse = MagicMock()
        return handler

    def test_routing_applied_before_send_message(self):
        """Verify that when routing fires, send_message sees the rewritten model."""
        call_models = []

        def fake_send_message(payload, credentials, config):
            call_models.append(payload.get('model'))
            # First call is the classifier call (return valid label)
            if len(call_models) == 1:
                return _text_response('trivial')
            # Second call is the main dispatch
            return {'type': 'message', 'content': []}

        backend = MagicMock()
        del backend.send_classifier_message
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = fake_send_message

        handler = self._handler(backend, routing=True)
        payload = {'model': 'sonnet', 'messages': [_msg('Do a simple thing')]}
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        # Two calls: classifier first, then main dispatch
        assert len(call_models) == 2
        # The main dispatch (second call) should see 'haiku'
        assert call_models[1] == 'haiku'

    def test_routing_disabled_send_message_sees_original_model(self):
        captured = {}

        def fake_send_message(payload, credentials, config):
            captured['model'] = payload.get('model')
            return {'type': 'message', 'content': []}

        backend = MagicMock()
        backend.parse_credentials.return_value = {}
        backend.send_message.side_effect = fake_send_message

        handler = self._handler(backend, routing=False)
        payload = {'model': 'sonnet', 'messages': [_msg('Do something')]}
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        handler._handle_messages()

        assert captured['model'] == 'sonnet'

    def test_local_commands_bypass_routing(self):
        """proxy-get-backend must never reach route_model."""
        backend = MagicMock()
        backend.parse_credentials.return_value = {}

        handler = self._handler(backend, routing=True)
        handler.registry.active_name.return_value = 'anthropic'
        payload = {'messages': [_msg('proxy-get-backend')]}
        handler.headers = {}
        handler._validate_content_type = MagicMock()
        handler._read_body = MagicMock(return_value=b'{}')
        handler._parse_json = MagicMock(return_value=payload)

        with patch('anthproxy.handlers._route_model') as mock_route:
            handler._handle_messages()
            mock_route.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Short-affirmation detection and routing
# ---------------------------------------------------------------------------

class TestIsShortAffirmation:
    """Unit tests for the is_short_affirmation helper in request_text."""

    @pytest.mark.parametrize('text', [
        'yes', 'yep', 'yeah', 'yup', 'ok', 'okay', 'sure',
        'proceed', 'go', 'go ahead', 'go for it', 'please go ahead',
        'do it', 'please do', 'continue', 'please continue',
        'confirmed', 'sounds good', 'lgtm', 'ship it', 'make it so',
        'run it', 'please start',
    ])
    def test_curated_phrases_match(self, text):
        from anthproxy.request_text import is_short_affirmation
        assert is_short_affirmation(text) is True

    @pytest.mark.parametrize('text', [
        'Yes', 'YES', 'Yes!', 'yes.', 'OK.', 'Okay!', 'Proceed.',
        'Go ahead?', 'Please proceed', 'PLEASE GO AHEAD', '  yes  ',
        'go  ahead', 'sounds   good',
    ])
    def test_casing_punctuation_whitespace_normalized(self, text):
        from anthproxy.request_text import is_short_affirmation
        assert is_short_affirmation(text) is True

    @pytest.mark.parametrize('text', [
        'fix the bug', 'delete auth.py', 'add a test', 'what is 2+2',
        'retry the deploy', 'rename the class', 'yes, but also rename the file',
        'go to the next file and refactor it',
    ])
    def test_short_instructions_rejected(self, text):
        from anthproxy.request_text import is_short_affirmation
        assert is_short_affirmation(text) is False

    def test_bare_start_rejected_v1(self):
        """'start' alone is a real imperative; only 'please start' is an affirmation."""
        from anthproxy.request_text import is_short_affirmation
        assert is_short_affirmation('start') is False
        assert is_short_affirmation('please start') is True

    def test_over_max_chars_rejected(self):
        from anthproxy.request_text import is_short_affirmation
        # A string that starts with an affirmation word but is long is rejected.
        assert is_short_affirmation('yes ' + 'x' * 50) is False

    def test_empty_and_non_string_rejected(self):
        from anthproxy.request_text import is_short_affirmation
        assert is_short_affirmation('') is False
        assert is_short_affirmation('   ') is False
        assert is_short_affirmation(None) is False  # type: ignore[arg-type]


class TestIsTitleGeneration:
    """Unit tests for the is_title_generation helper in request_text."""

    _SUFFIX = (
        'Write the title in the predominant language of the session'
        ' — a stray word or code token in another language doesn\'t change it.'
        ' Ignore the language of the examples above.'
    )

    def _make_prompt(self, session_text: str) -> str:
        return f'<session>\n{session_text}\n</session>\n\n{self._SUFFIX}'

    def test_typical_title_prompt_detected(self):
        from anthproxy.request_text import is_title_generation
        text = self._make_prompt('Can claude code understand instruction to supply LLM API request with a user-defined header?')
        assert is_title_generation(text) is True

    def test_case_insensitive(self):
        from anthproxy.request_text import is_title_generation
        text = self._make_prompt('some session').replace('Write the title', 'WRITE THE TITLE')
        assert is_title_generation(text) is True

    def test_trailing_newline_detected(self):
        from anthproxy.request_text import is_title_generation
        text = self._make_prompt('some session') + '\n'
        assert is_title_generation(text) is True

    def test_no_suffix_not_detected(self):
        from anthproxy.request_text import is_title_generation
        assert is_title_generation('Explain how git rebase works') is False

    def test_empty_and_non_string(self):
        from anthproxy.request_text import is_title_generation
        assert is_title_generation('') is False
        assert is_title_generation(None) is False  # type: ignore[arg-type]

    def test_routing_short_circuits_to_trivial(self):
        """route_model assigns trivial tier without classifying title-gen prompts."""
        suffix = self._SUFFIX
        payload = {
            'model': 'claude-sonnet-5',
            'messages': [
                {'role': 'user', 'content': f'<session>\nHow do I use git?\n</session>\n\n{suffix}'},
            ],
        }
        snap = _snapshot()
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'rule_title_generation'
        assert decision.classification is None
        assert decision.applied is True
        assert payload['model'] == snap.config.auto_model_routing_classification['trivial']


class TestSummaryAffirmationFlag:
    """build_routing_summary sets is_short_affirmation only for Path-1 direct text."""

    def test_direct_affirmation_sets_flag(self):
        summary = build_routing_summary(_payload(model='sonnet', content='yes'))
        assert summary is not None
        assert summary.is_short_affirmation is True
        assert summary.recovered_via_walkback is False

    def test_direct_non_affirmation_flag_false(self):
        summary = build_routing_summary(_payload(model='sonnet', content='fix the bug'))
        assert summary is not None
        assert summary.is_short_affirmation is False

    def test_transcript_only_affirmation_flag_false(self):
        """An affirmation recovered from a <transcript> block is historical, not live."""
        content = '<transcript>\nUser: yes\n</transcript>'
        summary = build_routing_summary(_payload(model='sonnet', content=content))
        assert summary is not None
        # The transcript fallback recovered 'yes', but it did not come from the
        # final message's own text → flag must be False.
        assert summary.is_short_affirmation is False

    def test_walkback_affirmation_flag_false(self):
        """An affirmation recovered via walk-back over prior messages is not live."""
        payload = _payload(
            model='sonnet',
            messages=[
                _msg('yes'),
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
                ]},
            ],
        )
        summary = build_routing_summary(payload)
        assert summary is not None
        assert summary.recovered_via_walkback is True
        assert summary.is_short_affirmation is False

    def test_flag_excluded_from_classifier_json(self):
        """Both is_short_affirmation and recovered_via_walkback are routing-internal."""
        summary = build_routing_summary(_payload(model='sonnet', content='yes'))
        assert summary is not None
        as_json = json.loads(summary.to_classifier_json())
        assert 'is_short_affirmation' not in as_json
        assert 'recovered_via_walkback' not in as_json


class TestRouteAffirmation:
    """route_model treats a direct affirmation as a continuation."""

    def _backend(self, label='trivial'):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response(label)
        return backend

    def test_inherits_cached_tier_without_classifier(self):
        backend = self._backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert payload['model'] == 'opus'
        assert decision.routed_model == 'opus'
        assert decision.classification is None
        assert decision.reason_code == 'affirmation_inherited'
        assert decision.applied is True
        backend.send_message.assert_not_called()

    def test_inherited_model_equals_cached_tier_value(self):
        """The inherited model is exactly the cached_session_tier passed in."""
        backend = self._backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='please go ahead')
        decision = route_model(payload, snap, {}, cached_session_tier='sonnet')
        assert decision.routed_model == 'sonnet'
        assert decision.reason_code == 'affirmation_inherited'
        backend.send_message.assert_not_called()

    def test_floors_to_standard_when_no_cache_no_prior(self):
        # No prior assistant message → critical invariant fires; uses floor (affirmation_classifier_failed).
        backend = self._backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert payload['model'] == 'sonnet'
        assert decision.routed_model == 'sonnet'
        assert decision.classification is None
        assert decision.reason_code == 'affirmation_classifier_failed'
        backend.send_message.assert_not_called()

    def test_classification_none_so_handler_does_not_cache(self):
        """Affirmation paths that don't call the classifier return classification=None.

        - cached path (affirmation_inherited): classification=None (no cache write).
        - floor path (affirmation_classifier_failed, no prior assistant text):
          classification=None (no cache write).
        Note: the affirmation_classified path DOES return non-None classification.
        """
        backend = self._backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        for cached in ('opus', None):
            payload = _payload(model='sonnet', content='yes')  # single user msg, no prior assistant
            decision = route_model(payload, snap, {}, cached_session_tier=cached)
            assert decision.classification is None

    def test_size_floor_outranks_affirmation(self):
        """A large affirmation continuation still forces opus[1m] (floor wins)."""
        backend = self._backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=190_000)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {},
                               cached_session_tier='haiku',
                               session_context_tokens=195_000)
        assert decision.reason_code == 'size_forced_long_context'
        assert payload['model'] == 'opus[1m]'
        backend.send_message.assert_not_called()

    def test_knob_disabled_falls_through_to_classifier(self):
        backend = self._backend('trivial')
        snap = _snapshot(backend=backend, routing=True, affirmation_inherit=False)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        # Affirmation branch skipped → classifier runs and returns trivial → haiku.
        assert payload['model'] == 'haiku'
        assert decision.classification == 'trivial'
        assert decision.reason_code == 'classifier_trivial'
        backend.send_message.assert_called_once()

    def test_non_affirmation_direct_text_still_classified(self):
        backend = self._backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code == 'classifier_standard'
        backend.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# Prior-response affirmation enrichment
# ---------------------------------------------------------------------------

def _payload_with_prior(affirmation: str, prior_assistant_text: str, model: str = 'sonnet'):
    """Build a payload with a prior assistant message (text) followed by an affirmation."""
    return {
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'Plan the refactor'},
            {'role': 'assistant', 'content': prior_assistant_text},
            {'role': 'user', 'content': affirmation},
        ],
    }


def _payload_with_prior_blocks(affirmation: str, prior_blocks: list, model: str = 'sonnet'):
    """Build a payload where the prior assistant message uses block content."""
    return {
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'Plan the refactor'},
            {'role': 'assistant', 'content': prior_blocks},
            {'role': 'user', 'content': affirmation},
        ],
    }


class TestAffirmationEnrichment:
    """Tests for the prior-response affirmation enrichment feature."""

    def _snap(self, label: str = 'standard', **kw):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response(label)
        return _snapshot(backend=backend, routing=True, **kw), backend

    # ------------------------------------------------------------------
    # 1. Cached tier path: no classifier call
    # ------------------------------------------------------------------

    def test_cached_tier_no_classifier(self):
        snap, backend = self._snap()
        payload = _payload_with_prior('yes', 'I will implement X')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code == 'affirmation_inherited'
        assert decision.classification is None
        backend.send_message.assert_not_called()

    # ------------------------------------------------------------------
    # 2. No-cache with prior response: classifier called, result cached
    # ------------------------------------------------------------------

    def test_no_cache_with_prior_response_calls_classifier(self):
        snap, backend = self._snap('standard')
        payload = _payload_with_prior('yes', 'I will implement X by refactoring the module')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classified'
        assert decision.classification == 'standard'
        assert decision.routed_model == 'sonnet'
        assert decision.classifier_mode == 'affirmation'
        backend.send_message.assert_called_once()

    def test_no_cache_prior_response_prior_response_summary_in_json(self):
        """prior_response_summary field is injected into the classifier JSON."""
        snap, backend = self._snap('standard')
        payload = _payload_with_prior('yes', 'Implement auth middleware')
        route_model(payload, snap, {}, cached_session_tier=None)
        call_args = backend.send_message.call_args[0][0]  # first positional arg
        user_content = call_args['messages'][0]['content']
        parsed = json.loads(user_content)
        assert 'prior_response_summary' in parsed
        assert 'Implement auth middleware' in parsed['prior_response_summary']

    def test_no_cache_result_has_cache_tier(self):
        """affirmation_classified returns cache_tier (uncapped) and routed_model (capped)."""
        snap, backend = self._snap('deep')
        payload = _payload_with_prior('yes', 'Redesign the auth layer', model='haiku')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classified'
        assert decision.classification == 'deep'
        assert decision.cache_tier == 'opus'
        assert decision.routed_model == 'opus'

    # ------------------------------------------------------------------
    # 3. No prior text in last assistant message: backward walk
    # ------------------------------------------------------------------

    def test_backward_walk_finds_earlier_assistant_text(self):
        """Walk-back finds text in an earlier assistant message when the last has only tool_use."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': 'Plan refactor'},
                {'role': 'assistant', 'content': 'I will refactor the module in three steps'},
                {'role': 'user', 'content': 'ok'},
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}}
                ]},
                {'role': 'user', 'content': 'yes'},
            ],
        }
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classified'
        backend.send_message.assert_called_once()

    def test_no_text_in_any_assistant_message_uses_floor(self):
        """If no text-bearing assistant message exists anywhere, use floor (no cache write)."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'user', 'content': 'Plan refactor'},
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}}
                ]},
                {'role': 'user', 'content': 'yes'},
            ],
        }
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classifier_failed'
        assert decision.classification is None
        backend.send_message.assert_not_called()

    # ------------------------------------------------------------------
    # 4. Session-opening affirmation: no prior message at all
    # ------------------------------------------------------------------

    def test_session_opening_affirmation_uses_floor(self):
        """First turn is an affirmation: no prior assistant message → floor, no cache."""
        snap, backend = self._snap()
        payload = _payload(model='sonnet', content='yes')  # single user message
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classifier_failed'
        assert decision.classification is None
        backend.send_message.assert_not_called()

    # ------------------------------------------------------------------
    # 5. Text extraction (string content)
    # ------------------------------------------------------------------

    def test_prior_response_string_content_extracted(self):
        """Plain-string assistant content is extracted and sent as prior_response_summary."""
        snap, backend = self._snap('deep')
        payload = _payload_with_prior('proceed', 'Redesign the entire auth layer for SSO')
        route_model(payload, snap, {}, cached_session_tier=None)
        user_content = backend.send_message.call_args[0][0]['messages'][0]['content']
        parsed = json.loads(user_content)
        assert parsed['prior_response_summary'] == 'Redesign the entire auth layer for SSO'

    def test_prior_response_truncation_30_70(self):
        """Prior response longer than limit is truncated 30/70 head/tail."""
        from anthproxy.model_router import _TRUNCATION_MARKER
        snap, backend = self._snap('standard')
        snap.config.auto_model_routing_prior_response_summary_limit = 50
        long_text = 'A' * 30 + 'B' * 100
        payload = _payload_with_prior('yes', long_text)
        route_model(payload, snap, {}, cached_session_tier=None)
        user_content = backend.send_message.call_args[0][0]['messages'][0]['content']
        parsed = json.loads(user_content)
        summary = parsed['prior_response_summary']
        assert len(summary) == 50
        assert _TRUNCATION_MARKER in summary

    # ------------------------------------------------------------------
    # 6. Text extraction (list content)
    # ------------------------------------------------------------------

    def test_prior_response_list_content_text_blocks_collected(self):
        """Text blocks from list content are concatenated; non-text blocks skipped."""
        snap, backend = self._snap('standard')
        prior_blocks = [
            {'type': 'text', 'text': 'First step: refactor'},
            {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
            {'type': 'text', 'text': 'Second step: test'},
        ]
        payload = _payload_with_prior_blocks('yes', prior_blocks)
        route_model(payload, snap, {}, cached_session_tier=None)
        user_content = backend.send_message.call_args[0][0]['messages'][0]['content']
        parsed = json.loads(user_content)
        summary = parsed['prior_response_summary']
        assert 'First step: refactor' in summary
        assert 'Second step: test' in summary

    # ------------------------------------------------------------------
    # 7. Malformed content (non-dict items)
    # ------------------------------------------------------------------

    def test_malformed_content_non_dict_items_skipped(self):
        """Non-dict items in assistant content list are silently skipped (no crash)."""
        snap, backend = self._snap('standard')
        prior_blocks = [
            'not a dict',
            {'type': 'text', 'text': 'Valid text block'},
            42,
        ]
        payload = _payload_with_prior_blocks('yes', prior_blocks)
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        # The valid text block is found; classifier is called.
        assert decision.reason_code == 'affirmation_classified'
        assert decision.classification == 'standard'

    # ------------------------------------------------------------------
    # 8. Classifier failure (network error)
    # ------------------------------------------------------------------

    def test_classifier_failure_uses_floor_no_cache(self):
        """Classifier network error → floor tier; classification=None; no cache write."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.side_effect = RuntimeError('timeout')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload_with_prior('yes', 'Implement the feature')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classifier_failed'
        assert decision.classification is None
        assert decision.routed_model == 'sonnet'  # standard floor

    # ------------------------------------------------------------------
    # 10. Baseline lock: cache_tier uncapped, routed_model capped
    # ------------------------------------------------------------------

    def test_baseline_lock_cache_tier_uncapped_routed_capped(self):
        """With baseline_model=haiku, cache_tier is the raw opus, routed_model is capped."""
        snap, backend = self._snap('deep')
        payload = _payload_with_prior('yes', 'Redesign auth')
        # baseline_model=haiku means deep→opus gets capped to haiku
        decision = route_model(payload, snap, {}, cached_session_tier=None,
                               baseline_model='haiku')
        assert decision.reason_code == 'affirmation_classified'
        assert decision.cache_tier == 'opus'   # uncapped
        assert decision.routed_model == 'haiku'  # capped by baseline

    # ------------------------------------------------------------------
    # 11. No upgrade cap applies when no baseline_model
    # ------------------------------------------------------------------

    def test_no_upgrade_cap_without_baseline(self):
        """Without baseline_model, the tier from the classifier is used directly."""
        snap, backend = self._snap('trivial')
        payload = _payload_with_prior('yes', 'Format the file', model='opus')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classified'
        assert decision.routed_model == 'haiku'
        assert decision.cache_tier == 'haiku'

    # ------------------------------------------------------------------
    # 12. routed_model derivation
    # ------------------------------------------------------------------

    def test_routed_model_equals_capped_tier(self):
        """routed_model is the capped tier; cache_tier is the uncapped value."""
        snap, backend = self._snap('standard')
        snap.config.auto_model_routing_classification = {
            'trivial': 'haiku', 'standard': 'fable', 'deep': 'opus'
        }
        payload = _payload_with_prior('yes', 'Implement the feature')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classified'
        assert decision.classification == 'standard'
        assert decision.routed_model == 'fable'
        assert decision.cache_tier == 'fable'

    # ------------------------------------------------------------------
    # 13. ctx_key=None gate: classification returns non-None but no cache write
    # ------------------------------------------------------------------

    def test_ctx_key_none_classification_still_routes(self):
        """ctx_key=None: affirmation_classified routes correctly (cache write skipped by handler)."""
        snap, backend = self._snap('standard')
        payload = _payload_with_prior('yes', 'Implement auth')
        decision = route_model(payload, snap, {}, cached_session_tier=None,
                               ctx_key=None)
        assert decision.reason_code == 'affirmation_classified'
        assert decision.classification == 'standard'
        assert decision.routed_model == 'sonnet'

    # ------------------------------------------------------------------
    # 15. to_classifier_json with prior_response_summary
    # ------------------------------------------------------------------

    def test_to_classifier_json_with_prior_response_summary(self):
        """prior_response_summary is merged into the JSON dict when non-None."""
        summary = build_routing_summary({'model': 'sonnet', 'messages': [
            {'role': 'user', 'content': 'yes'}
        ]})
        assert summary is not None
        j = summary.to_classifier_json(prior_response_summary='do the thing')
        parsed = json.loads(j)
        assert parsed['prior_response_summary'] == 'do the thing'

    def test_to_classifier_json_without_prior_response_summary(self):
        """prior_response_summary is absent from JSON when None (default)."""
        summary = build_routing_summary({'model': 'sonnet', 'messages': [
            {'role': 'user', 'content': 'hi'}
        ]})
        assert summary is not None
        j = summary.to_classifier_json()
        parsed = json.loads(j)
        assert 'prior_response_summary' not in parsed

    # ------------------------------------------------------------------
    # confidence_bump=True in the affirmation no-cache path
    # ------------------------------------------------------------------

    def test_confidence_bump_true_uses_json_system_prompt_and_parser(self):
        """When confidence_bump=True, the affirmation classifier call uses
        _CLASSIFIER_SYSTEM_JSON + _CLASSIFIER_SYSTEM_JSON_PRIOR_SUFFIX and
        parse_classifier_score_json, not the one-word prompt/parser."""
        from anthproxy.model_router import _CLASSIFIER_SYSTEM_JSON, _CLASSIFIER_SYSTEM_JSON_PRIOR_SUFFIX
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': '{"score":55}'}],
            'usage': {'input_tokens': 10, 'output_tokens': 8},
        }
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_confidence_bump = True
        snap.config.auto_model_routing_min_confidence = 0.0

        payload = _payload_with_prior('yes', 'Implement auth middleware')
        decision = route_model(payload, snap, {}, cached_session_tier=None)

        assert decision.reason_code == 'affirmation_classified'
        assert decision.classification == 'standard'
        assert decision.routed_model == 'sonnet'

        call_args = backend.send_message.call_args[0][0]
        system_used = call_args['system']
        assert _CLASSIFIER_SYSTEM_JSON in system_used
        assert _CLASSIFIER_SYSTEM_JSON_PRIOR_SUFFIX in system_used

    def test_confidence_bump_true_invalid_json_response_falls_to_floor(self):
        """When confidence_bump=True and the classifier returns an invalid JSON
        response, the affirmation path falls back to floor (affirmation_classifier_failed)."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': 'standard'}],  # one-word: invalid for JSON parser
        }
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_confidence_bump = True

        payload = _payload_with_prior('proceed', 'Implement the feature')
        decision = route_model(payload, snap, {}, cached_session_tier=None)

        assert decision.reason_code == 'affirmation_classifier_failed'
        assert decision.classification is None


# ---------------------------------------------------------------------------
# Config validation for prior_response_summary_limit
# ---------------------------------------------------------------------------

class TestPriorResponseSummaryLimitConfig:
    """Validate the auto_model_routing_prior_response_summary_limit config option."""

    def test_limit_below_50_raises(self):
        """Limit < 50 raises ValueError."""
        from anthproxy.config import parse_args
        with pytest.raises((ValueError, SystemExit)):
            parse_args(['--auto-model-routing',
                        '--auto-model-routing-prior-response-summary-limit', '49'])

    def test_limit_above_32000_raises(self):
        """Limit > 32000 raises ValueError."""
        from anthproxy.config import parse_args
        with pytest.raises((ValueError, SystemExit)):
            parse_args(['--auto-model-routing',
                        '--auto-model-routing-prior-response-summary-limit', '32001'])

    def test_limit_at_50_valid(self):
        """Limit = 50 is valid (minimum boundary)."""
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing',
                          '--auto-model-routing-prior-response-summary-limit', '50'])
        assert cfg.auto_model_routing_prior_response_summary_limit == 50

    def test_limit_at_32000_valid(self):
        """Limit = 32000 is valid (maximum boundary)."""
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing',
                          '--auto-model-routing-prior-response-summary-limit', '32000'])
        assert cfg.auto_model_routing_prior_response_summary_limit == 32000

    def test_default_limit_is_1000(self):
        """Default limit is 1000."""
        from anthproxy.config import parse_args
        cfg = parse_args(['--auto-model-routing'])
        assert cfg.auto_model_routing_prior_response_summary_limit == 1000


# ---------------------------------------------------------------------------
# Unconditional prior-response enrichment (main classifier path)
# ---------------------------------------------------------------------------

class TestUnconditionalPriorResponseEnrichment:
    """Verify that prior_response_summary is injected on the main classifier
    dispatch path even when the user turn is NOT a short affirmation."""

    def _snap_capturing(self, label: str = 'standard', confidence_bump: bool = False):
        captured: list[dict] = []
        backend = MagicMock()
        del backend.send_classifier_message

        def fake_send(clf_payload, credentials, config):
            captured.append(clf_payload)
            return _text_response(label)

        backend.send_message.side_effect = fake_send
        snap = _snapshot(backend=backend, routing=True, confidence_bump=confidence_bump)
        return snap, captured

    def _make_payload(self, user_text: str, assistant_text: str) -> dict:
        return {
            'model': 'claude-haiku-20240307',
            'max_tokens': 100,
            'messages': [
                {'role': 'user', 'content': 'First question'},
                {'role': 'assistant', 'content': assistant_text},
                {'role': 'user', 'content': user_text},
            ],
        }

    def test_prior_response_injected_on_non_affirmation_turn(self):
        """A long non-affirmation user turn still receives prior_response_summary."""
        assistant_reply = 'Here is my detailed plan with many steps...'
        user_turn = 'Please now implement step 2 of your plan with full error handling and tests'
        payload = self._make_payload(user_turn, assistant_reply)
        snap, captured = self._snap_capturing(confidence_bump=True)

        route_model(payload, snap, {})

        assert captured, 'classifier was not called'
        msg_content = captured[0]['messages'][0]['content']
        parsed = json.loads(msg_content)
        assert 'prior_response_summary' in parsed
        assert parsed['prior_response_summary'] == assistant_reply

    def test_no_prior_response_when_no_assistant_message(self):
        """When there is no prior assistant message the field is absent."""
        payload = {
            'model': 'claude-haiku-20240307',
            'max_tokens': 100,
            'messages': [
                {'role': 'user', 'content': 'Write a full production app with tests'},
            ],
        }
        snap, captured = self._snap_capturing(confidence_bump=True)

        route_model(payload, snap, {})

        assert captured, 'classifier was not called'
        parsed = json.loads(captured[0]['messages'][0]['content'])
        assert 'prior_response_summary' not in parsed

    def test_prior_response_enriches_system_prompt_suffix(self):
        """System prompt gains the prior-context suffix when prior_response_summary present."""
        assistant_reply = 'I have analyzed the codebase and found three issues.'
        payload = self._make_payload(
            'Fix all three issues and add integration tests', assistant_reply
        )
        snap, captured = self._snap_capturing(label='deep', confidence_bump=True)

        route_model(payload, snap, {})

        assert captured
        system = captured[0]['system']
        assert 'prior_response_summary' in system


# ---------------------------------------------------------------------------
# Phase 0 — Routing ordering invariants
# ---------------------------------------------------------------------------

class TestRoutingOrderInvariants:
    """Explicit ordering invariant tests.

    Each test asserts that a specific pre-classifier branch fires BEFORE the
    classifier is consulted, preserving the documented priority order:
      1. size floor  >  2. walkback cache  >  3. affirmation  >  4. classifier

    These are regression guards: shifting any branch in priority would break
    the test that asserts it fires before the next stage.
    """

    # ------------------------------------------------------------------
    # Invariant 1: size floor fires BEFORE classifier
    # ------------------------------------------------------------------

    def test_size_floor_fires_before_classifier(self):
        """size_forced_long_context must trip before the classifier is consulted."""
        backend = _classifier_backend('trivial')  # would route trivial → haiku if reached
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        # Classifier must NOT have been called at all.
        backend.send_message.assert_not_called()

    def test_size_floor_via_session_context_fires_before_classifier(self):
        """A large cached session floor also trips before classifier."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=190_000)
        payload = _payload(model='sonnet', content='continue')
        decision = route_model(payload, snap, {}, session_context_tokens=195_000)
        assert decision.reason_code == 'size_forced_long_context'
        backend.send_message.assert_not_called()

    # ------------------------------------------------------------------
    # Invariant 3: walk-back cache before classifier
    # ------------------------------------------------------------------

    def test_walkback_cache_replay_before_classifier(self):
        """session_cached_walkback must short-circuit before the classifier call."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(
            model='opus',
            messages=[
                _msg('Plan the refactor'),
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
                ]},
            ],
        )
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        # Reason code must be a walkback cache code, not classifier_*.
        assert decision.reason_code in ('session_cached_walkback', 'session_cached_walkback_capped', 'session_cached_walkback_tool_result')
        # Classifier must NOT have been called.
        backend.send_message.assert_not_called()

    # ------------------------------------------------------------------
    # Invariant 4: short-affirmation before classifier
    # ------------------------------------------------------------------

    def test_affirmation_inherit_before_classifier(self):
        """affirmation_inherited fires before classifier when cached tier exists."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code == 'affirmation_inherited'
        # Classifier must NOT have been called.
        backend.send_message.assert_not_called()

    def test_affirmation_no_prior_text_does_not_call_classifier(self):
        """Critical invariant: no prior text-bearing assistant message → floor without classifier call."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='proceed')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classifier_failed'
        # Classifier must NOT have been called (no prior assistant text).
        backend.send_message.assert_not_called()

    # ------------------------------------------------------------------
    # Invariant 5: invalid classifier output keeps original model
    # ------------------------------------------------------------------

    def test_classifier_none_label_keeps_requested_model(self):
        """parse_classifier_label returns None → requested model unchanged, applied=False."""
        backend = MagicMock()
        del backend.send_classifier_message
        # Sentence with multiple words → parse_classifier_label returns None.
        backend.send_message.return_value = _text_response('The answer is definitely deep')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='claude-opus-4-8', content='Design the system')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_invalid'
        assert decision.applied is False
        assert payload['model'] == 'claude-opus-4-8'  # original preserved
        assert decision.routed_model == 'claude-opus-4-8'
        assert decision.classification is None

    def test_classifier_exception_keeps_requested_model(self):
        """Classifier network exception → fail-closed keeps original model."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.side_effect = RuntimeError('network failure')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='claude-haiku-4-5', content='Fix the test')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_failed'
        assert decision.applied is False
        assert payload['model'] == 'claude-haiku-4-5'
        assert decision.routed_model == 'claude-haiku-4-5'


# ---------------------------------------------------------------------------
# Phase 1 — ModelRoutingDecision telemetry fields
# ---------------------------------------------------------------------------

class TestModelRoutingDecisionTelemetry:
    """Verify that the new Phase 1 telemetry fields are populated correctly
    in each routing branch and present with defaults on others."""

    # ------------------------------------------------------------------
    # Field defaults on disabled/ineligible/sentinel paths
    # ------------------------------------------------------------------

    def test_disabled_has_default_telemetry(self):
        snap = _snapshot(routing=False)
        decision = route_model(_payload(model='sonnet'), snap, {})
        assert decision.reason_code == 'disabled'
        assert decision.predicted_input_tokens == 0
        assert decision.session_context_tokens == 0
        assert decision.session_estimate_ratio == 1.0
        assert decision.classifier_mode == 'classifier'
        assert decision.classifier_confidence is None
        assert decision.tier_bumped is False
        assert decision.task_tag is None

    def test_ineligible_has_default_telemetry(self):
        snap = _snapshot(routing=True)
        decision = route_model(_payload(model=''), snap, {})
        assert decision.reason_code == 'model_not_eligible'
        assert decision.predicted_input_tokens == 0
        assert decision.classifier_mode == 'classifier'

    # ------------------------------------------------------------------
    # Size floor telemetry
    # ------------------------------------------------------------------

    def test_size_floor_populates_predicted_and_session_tokens(self):
        """size_forced_long_context sets predicted_input_tokens, session_context_tokens,
        session_estimate_ratio from the values passed to route_model()."""
        snap = _snapshot(routing=True, long_context_threshold=190_000)
        payload = _payload(model='sonnet', content='continue')
        decision = route_model(
            payload, snap, {},
            session_context_tokens=200_000,
            session_estimate_ratio=1.5,
        )
        assert decision.reason_code == 'size_forced_long_context'
        # predicted = max(round(est * ratio), session_context_tokens)
        # est for tiny "continue" is a few tokens; session_context_tokens wins.
        assert decision.predicted_input_tokens >= 200_000
        assert decision.session_context_tokens == 200_000
        assert abs(decision.session_estimate_ratio - 1.5) < 1e-9

    def test_size_floor_classifier_mode_is_default(self):
        """Size floor does not set classifier_mode — it stays at the default 'classifier'."""
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        # classifier_mode stays at the default for the size floor path.
        assert decision.classifier_mode == 'classifier'

    # ------------------------------------------------------------------
    # Walk-back cache replay telemetry
    # ------------------------------------------------------------------

    def test_walkback_cache_sets_classifier_mode(self):
        """session_cached_walkback / _capped sets classifier_mode='walkback_cache'."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(
            model='opus',
            messages=[
                _msg('Redesign the auth layer'),
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
                ]},
            ],
        )
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code in ('session_cached_walkback', 'session_cached_walkback_capped', 'session_cached_walkback_tool_result')
        assert decision.classifier_mode == 'walkback_cache'
        assert decision.classifier_confidence is None

    # ------------------------------------------------------------------
    # Affirmation telemetry
    # ------------------------------------------------------------------

    def test_affirmation_inherited_sets_classifier_mode(self):
        """affirmation_inherited sets classifier_mode='affirmation'."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code == 'affirmation_inherited'
        assert decision.classifier_mode == 'affirmation'
        assert decision.classifier_confidence is None

    def test_affirmation_classifier_failed_sets_classifier_mode(self):
        """affirmation_classifier_failed (no prior text) sets classifier_mode='affirmation'."""
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='proceed')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classifier_failed'
        assert decision.classifier_mode == 'affirmation'

    # ------------------------------------------------------------------
    # Classifier success telemetry
    # ------------------------------------------------------------------

    def test_classifier_success_sets_classifier_mode(self):
        """Successful classification sets classifier_mode='classifier'."""
        backend = _classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the flaky test')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_standard'
        assert decision.classifier_mode == 'classifier'
        assert decision.classifier_confidence is None  # no confidence yet

    def test_classifier_deep_mode_and_applied(self):
        backend = _classifier_backend('deep')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='redesign the auth system')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_deep'
        assert decision.classifier_mode == 'classifier'
        assert decision.applied is True
        assert decision.classifier_confidence is None

    def test_classifier_trivial_mode(self):
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='hi')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_trivial'
        assert decision.classifier_mode == 'classifier'

    # ------------------------------------------------------------------
    # Classifier failure / invalid telemetry
    # ------------------------------------------------------------------

    def test_classifier_failed_sets_classifier_mode(self):
        """Classifier exception sets classifier_mode='classifier'."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.side_effect = RuntimeError('timeout')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_failed'
        assert decision.classifier_mode == 'classifier'
        assert decision.applied is False

    def test_classifier_invalid_sets_classifier_mode(self):
        """Invalid classifier label sets classifier_mode='classifier'."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('something entirely invalid')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_invalid'
        assert decision.classifier_mode == 'classifier'
        assert decision.applied is False

    # ------------------------------------------------------------------
    # Default values for reserved fields
    # ------------------------------------------------------------------

    def test_reserved_fields_default_on_all_paths(self):
        """tier_bumped is False on all paths that skip the confidence-bump promotion; task_tag is None until tag-routing populates it"""
        for content, cached_tier, label in [
            ('yes', 'opus', 'trivial'),          # affirmation path
            ('fix the bug', None, 'standard'),   # classifier path
        ]:
            backend = _classifier_backend(label)
            snap = _snapshot(backend=backend, routing=True)
            payload = _payload(model='sonnet', content=content)
            decision = route_model(payload, snap, {}, cached_session_tier=cached_tier)
            assert decision.tier_bumped is False, f'tier_bumped must be False for {decision.reason_code}'
            assert decision.task_tag is None, f'task_tag must be None for {decision.reason_code}'

    # ------------------------------------------------------------------
    # dataclasses.replace() preserves telemetry fields
    # ------------------------------------------------------------------

    def test_dataclasses_replace_preserves_telemetry(self):
        """dataclasses.replace() on a ModelRoutingDecision preserves telemetry fields."""
        from dataclasses import replace
        from anthproxy.model_router import ModelRoutingDecision
        original = ModelRoutingDecision(
            requested_model='sonnet',
            routed_model='opus',
            classification='deep',
            applied=True,
            reason_code='classifier_deep',
            estimated_input_tokens=1234,
            predicted_input_tokens=5678,
            session_context_tokens=9000,
            session_estimate_ratio=1.7,
            classifier_mode='classifier',
            classifier_confidence=None,
            tier_bumped=False,
            task_tag=None,
        )
        replaced = replace(original, reason_code='session_cached_tier', applied=False)
        assert replaced.predicted_input_tokens == 5678
        assert replaced.session_context_tokens == 9000
        assert abs(replaced.session_estimate_ratio - 1.7) < 1e-9
        assert replaced.classifier_mode == 'classifier'
        assert replaced.classifier_confidence is None
        assert replaced.tier_bumped is False
        assert replaced.task_tag is None


# ---------------------------------------------------------------------------
# Phase 1 classifier token tests
# ---------------------------------------------------------------------------

class TestClassifierTokens:
    """Verify that classifier_input_tokens and classifier_output_tokens are
    populated from the classifier response's usage field (Phase 1)."""

    def test_successful_classifier_call_populates_tokens(self):
        """A successful classifier call with a usage field populates token counts."""
        backend = MagicMock()
        del backend.send_classifier_message  # duck-typed fallback
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': 'standard'}],
            'stop_reason': 'end_turn',
            'usage': {'input_tokens': 42, 'output_tokens': 1},
        }
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the auth bug')
        decision = route_model(payload, snap, {})
        assert decision.classifier_input_tokens > 0, (
            'classifier_input_tokens must be > 0 after a successful classifier call'
        )
        assert decision.classifier_output_tokens >= 0

    def test_rules_mode_yields_zero_classifier_tokens(self):
        """Rules mode never calls the LLM; both token counts must be 0."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('standard')
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_mode = 'rules'
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.classifier_input_tokens == 0
        assert decision.classifier_output_tokens == 0


# ---------------------------------------------------------------------------
# Phase 3 ctx_key size-floor log test
# ---------------------------------------------------------------------------

class TestCtxKeyInSizeFloorLog:
    """Verify that ctx_key is logged by the size-floor branch of route_model."""

    def test_ctx_key_anchor_appears_in_size_floor_log(self, caplog):
        """When ctx_key is passed and the floor fires, 'ctx=<anchor>' is logged."""
        backend = _classifier_backend('trivial')  # would route trivial→haiku if reached
        snap = _snapshot(backend=backend, routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content=_big_text(4000))
        with caplog.at_level(logging.INFO, logger='anthproxy.model_router'):
            route_model(payload, snap, {}, ctx_key='session123\x00anchor456')
        assert 'ctx=anchor456' in caplog.text


# ---------------------------------------------------------------------------
# Confidence-bump promotion
# ---------------------------------------------------------------------------

class TestConfidenceBumpPromotion:
    """Verify tier_bumped when confidence_bump mode promotes a tier."""

    def test_confidence_bump_trivial_to_standard(self):
        """In numeric score mode, confidence_bump path uses {"score":N} JSON.

        The numeric score is parsed by parse_classifier_score_json; parsed_confidence
        is no longer available, so tier_bumped is False. tier_bumped removal is
        completed in ticket 02; this test verifies the numeric path classifies correctly.
        """
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': '{"score":10}'}],
            'stop_reason': 'end_turn',
        }
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_confidence_bump = True
        snap.config.auto_model_routing_min_confidence = 0.8
        payload = _payload(model='claude-sonnet-4-6', content='fix the typo')
        decision = route_model(payload, snap, {})
        assert decision.tier_bumped is False
        assert decision.reason_code == 'classifier_trivial'
        assert decision.classification == 'trivial'

    # ------------------------------------------------------------------
    # Default values for reserved fields
    # ------------------------------------------------------------------

    def test_reserved_fields_default_on_all_paths(self):
        """tier_bumped and task_tag are always False/None (reserved, not yet used)."""
        for content, cached_tier, label in [
            ('yes', 'opus', 'trivial'),          # affirmation path
            ('fix the bug', None, 'standard'),   # classifier path
        ]:
            backend = _classifier_backend(label)
            snap = _snapshot(backend=backend, routing=True)
            payload = _payload(model='sonnet', content=content)
            decision = route_model(payload, snap, {}, cached_session_tier=cached_tier)
            assert decision.tier_bumped is False, f'tier_bumped must be False for {decision.reason_code}'
            assert decision.task_tag is None, f'task_tag must be None for {decision.reason_code}'

    # ------------------------------------------------------------------
    # dataclasses.replace() preserves telemetry fields
    # ------------------------------------------------------------------

    def test_dataclasses_replace_preserves_telemetry(self):
        """dataclasses.replace() on a ModelRoutingDecision preserves telemetry fields."""
        from dataclasses import replace
        from anthproxy.model_router import ModelRoutingDecision
        original = ModelRoutingDecision(
            requested_model='sonnet',
            routed_model='opus',
            classification='deep',
            applied=True,
            reason_code='classifier_deep',
            estimated_input_tokens=1234,
            predicted_input_tokens=5678,
            session_context_tokens=9000,
            session_estimate_ratio=1.7,
            classifier_mode='classifier',
            classifier_confidence=None,
            tier_bumped=False,
            task_tag=None,
        )
        replaced = replace(original, reason_code='session_cached_tier', applied=False)
        assert replaced.predicted_input_tokens == 5678
        assert replaced.session_context_tokens == 9000
        assert abs(replaced.session_estimate_ratio - 1.7) < 1e-9
        assert replaced.classifier_mode == 'classifier'
        assert replaced.classifier_confidence is None
        assert replaced.tier_bumped is False
        assert replaced.task_tag is None


# ---------------------------------------------------------------------------
# TestClassifyByRules — unit tests for classify_by_rules()
# ---------------------------------------------------------------------------

class TestClassifyByRules:
    """Unit tests for classify_by_rules() — previously zero-coverage."""

    def _summary(self, text: str, has_images: bool = False, non_text: int = 0):
        return RoutingSummary(
            final_user_text=text, text_truncated=False,
            total_messages=1, prior_user_messages=0, prior_assistant_messages=0,
            tool_use_count=0, tool_result_count=0,
            final_non_text_blocks=non_text, has_images=has_images,
        )

    def test_empty_text_returns_none(self):
        assert classify_by_rules(self._summary('')) is None

    def test_whitespace_only_returns_none(self):
        assert classify_by_rules(self._summary('   \t\n')) is None

    def test_exact_trivial_phrase_hi(self):
        assert classify_by_rules(self._summary('hi')) == 'trivial'

    def test_exact_trivial_phrase_thanks(self):
        assert classify_by_rules(self._summary('thanks')) == 'trivial'

    def test_exact_trivial_new_phrase_lgtm(self):
        assert classify_by_rules(self._summary('lgtm')) == 'trivial'

    def test_exact_trivial_new_phrase_yes(self):
        assert classify_by_rules(self._summary('yes')) == 'trivial'

    def test_exact_trivial_new_phrase_sounds_good(self):
        assert classify_by_rules(self._summary('sounds good')) == 'trivial'

    def test_deep_keyword_architecture(self):
        assert classify_by_rules(self._summary('redesign the architecture')) == 'deep'

    def test_deep_beats_standard_keyword(self):
        """Deep scan runs before standard scan."""
        assert classify_by_rules(self._summary('add an architecture diagram')) == 'deep'

    def test_standard_keyword_fix(self):
        assert classify_by_rules(self._summary('fix the bug in auth.py')) == 'standard'

    def test_standard_keyword_add(self):
        assert classify_by_rules(self._summary('add a logout button')) == 'standard'

    def test_images_returns_standard(self):
        assert classify_by_rules(self._summary('', has_images=True)) == 'standard'

    def test_non_text_blocks_returns_standard(self):
        assert classify_by_rules(self._summary('', non_text=2)) == 'standard'

    def test_no_signal_returns_none(self):
        assert classify_by_rules(self._summary('blah blah blah')) is None

    # --- Prefix-trivial path ---

    def test_trivial_prefix_no_task_keyword_returns_trivial(self):
        """'sure, got it' — trivial prefix, word boundary, no standard keyword."""
        assert classify_by_rules(self._summary('sure, got it')) == 'trivial'

    def test_trivial_prefix_with_standard_keyword_returns_standard(self):
        """CRITICAL: 'sure, add cursor-based pagination' must NOT be trivial."""
        result = classify_by_rules(self._summary('sure, add cursor-based pagination'))
        assert result == 'standard', (
            f"Expected 'standard' (standard keyword 'add' present), got {result!r}. "
            "This is the Phase 7 regression case."
        )

    def test_trivial_prefix_with_deep_keyword_returns_deep(self):
        """Deep scan fires before prefix path."""
        assert classify_by_rules(self._summary('sure, redesign the architecture')) == 'deep'

    def test_trivial_prefix_text_over_120_chars_not_trivial(self):
        """120-char guard: text longer than limit skips prefix path."""
        long_text = 'okay, ' + 'x ' * 60  # well over 120 chars
        assert len(long_text) > 120
        result = classify_by_rules(self._summary(long_text))
        assert result != 'trivial'

    def test_word_boundary_rightclick_not_trivial(self):
        """'right' prefix must not match 'rightclick' — no word boundary."""
        result = classify_by_rules(self._summary('rightclick the button'))
        assert result != 'trivial', (
            "Word boundary check failed: 'rightclick' should not match prefix 'right'"
        )

    def test_word_boundary_nothing_not_trivial(self):
        """'no' prefix must not match 'nothing is wrong' — no word boundary."""
        result = classify_by_rules(self._summary('nothing is wrong'))
        assert result != 'trivial'

    def test_alright_prefix_no_task(self):
        assert classify_by_rules(self._summary('alright, carry on')) == 'trivial'

    def test_sounds_good_with_standard_keyword(self):
        """'sounds good, fix it' has standard keyword 'fix' → standard."""
        assert classify_by_rules(self._summary('sounds good, fix it')) == 'standard'


# ---------------------------------------------------------------------------
# ModelRoutingDecision — 4 new classifier transparency fields
# ---------------------------------------------------------------------------

class TestClassifierTransparencyFields:
    """New fields: classifier_model, classifier_summary_json,
    classifier_raw_response, classifier_format.

    All 4 remain None on non-classifier paths.  On a successful LLM call they
    are populated with the model ID, the bounded JSON sent, the full text
    concatenated from the response, and the format string ('standard'/'json').
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backend_with_response(self, label='standard', usage=None):
        """Backend whose send_message returns a text response for the given label."""
        backend = MagicMock()
        del backend.send_classifier_message  # force send_message fallback
        resp = {
            'content': [{'type': 'text', 'text': label}],
            'stop_reason': 'end_turn',
        }
        if usage is not None:
            resp['usage'] = usage
        backend.send_message.return_value = resp
        return backend

    def _run_classifier(self, label='standard', content='fix the auth bug',
                        classifier_model='haiku', **snap_kwargs):
        backend = self._backend_with_response(label)
        snap = _snapshot(backend=backend, routing=True,
                         classifier_model=classifier_model, **snap_kwargs)
        payload = _payload(model='sonnet', content=content)
        decision = route_model(payload, snap, {})
        return decision, snap

    # ------------------------------------------------------------------
    # Non-classifier paths: all 4 fields must be None
    # ------------------------------------------------------------------

    def test_disabled_routing_fields_are_none(self):
        snap = _snapshot(routing=False)
        decision = route_model(_payload(model='sonnet'), snap, {})
        assert decision.reason_code == 'disabled'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_model_not_eligible_fields_are_none(self):
        snap = _snapshot(routing=True)
        decision = route_model(_payload(model=''), snap, {})
        assert decision.reason_code == 'model_not_eligible'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_missing_user_text_fields_are_none(self):
        snap = _snapshot(routing=True)
        payload = {'model': 'sonnet', 'messages': [{'role': 'assistant', 'content': 'hi'}]}
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'missing_final_user_text'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_size_floor_fields_are_none(self):
        snap = _snapshot(routing=True, long_context_threshold=10)
        payload = _payload(model='sonnet', content='x' * 4000)
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'size_forced_long_context'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_affirmation_inherited_fields_are_none(self):
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='yes')
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code == 'affirmation_inherited'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_affirmation_classifier_failed_fields_are_none(self):
        # No prior text → affirmation_classifier_failed; transparency fields remain None.
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='proceed')
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classifier_failed'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_session_cached_walkback_fields_are_none(self):
        backend = _classifier_backend('trivial')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(
            model='opus',
            messages=[
                _msg('Plan the refactor'),
                {'role': 'assistant', 'content': [
                    {'type': 'tool_use', 'id': 'tu1', 'name': 'bash', 'input': {}},
                ]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'done'},
                ]},
            ],
        )
        decision = route_model(payload, snap, {}, cached_session_tier='opus')
        assert decision.reason_code == 'session_cached_walkback_tool_result'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_classifier_failed_fields_are_none(self):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.side_effect = RuntimeError('network error')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_failed'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_classifier_invalid_fields_are_none(self):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response('something invalid here')
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_invalid'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    def test_rules_mode_fields_are_none(self):
        backend = _classifier_backend('standard')
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_mode = 'rules'
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_rules_standard'
        assert decision.classifier_model is None
        assert decision.classifier_summary_json is None
        assert decision.classifier_raw_response is None
        assert decision.classifier_format is None

    # ------------------------------------------------------------------
    # Successful classifier path: all 4 fields populated
    # ------------------------------------------------------------------

    def test_classifier_model_matches_config(self):
        """classifier_model holds the model ID from config."""
        decision, _ = self._run_classifier(label='standard',
                                           classifier_model='claude-haiku-4-5')
        assert decision.classifier_model == 'claude-haiku-4-5'

    def test_classifier_model_with_different_config_model(self):
        """Different configured model IDs are reflected correctly."""
        decision, _ = self._run_classifier(label='trivial',
                                           classifier_model='claude-sonnet-4-6')
        assert decision.classifier_model == 'claude-sonnet-4-6'

    def test_classifier_summary_json_is_valid_json(self):
        """classifier_summary_json is a valid JSON string."""
        decision, _ = self._run_classifier(label='standard',
                                           content='implement auth middleware')
        assert decision.classifier_summary_json is not None
        parsed = json.loads(decision.classifier_summary_json)
        assert isinstance(parsed, dict)

    def test_classifier_summary_json_matches_to_classifier_json(self):
        """classifier_summary_json matches the output of RoutingSummary.to_classifier_json()."""
        decision, _ = self._run_classifier(label='standard',
                                           content='implement auth middleware')
        assert decision.classifier_summary_json is not None
        # Build the expected summary the same way route_model does.
        from anthproxy.model_router import build_routing_summary
        payload = _payload(model='sonnet', content='implement auth middleware')
        summary = build_routing_summary(payload)
        assert summary is not None
        assert decision.classifier_summary_json == summary.to_classifier_json()

    def test_classifier_summary_json_contains_final_user_text(self):
        """The JSON contains the final_user_text field with the request text."""
        decision, _ = self._run_classifier(label='deep',
                                           content='redesign the auth system')
        assert decision.classifier_summary_json is not None
        parsed = json.loads(decision.classifier_summary_json)
        assert 'final_user_text' in parsed
        assert 'redesign the auth system' in parsed['final_user_text']

    def test_classifier_summary_json_excludes_routing_internal_fields(self):
        """is_short_affirmation and recovered_via_walkback are never in the JSON."""
        decision, _ = self._run_classifier(label='standard',
                                           content='fix the bug')
        assert decision.classifier_summary_json is not None
        parsed = json.loads(decision.classifier_summary_json)
        assert 'is_short_affirmation' not in parsed
        assert 'recovered_via_walkback' not in parsed

    def test_classifier_raw_response_contains_response_text(self):
        """classifier_raw_response holds the text returned by the classifier."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': 'standard'}],
            'stop_reason': 'end_turn',
        }
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='implement auth middleware')
        decision = route_model(payload, snap, {})
        assert decision.classifier_raw_response == 'standard'

    def test_classifier_raw_response_concatenates_multiple_text_blocks(self):
        """Multiple text blocks are joined with a space into classifier_raw_response.

        The combined text must still parse as a valid label so the decision reaches
        the success path.  Using 'deep' followed by '!' (pure punctuation — no alpha
        token) gives combined = 'deep !' which has exactly one alpha token ('deep')
        and is accepted by parse_classifier_label, while exercising multi-block join.
        """
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [
                {'type': 'text', 'text': 'deep'},
                {'type': 'text', 'text': '!'},
            ],
            'stop_reason': 'end_turn',
        }
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='redesign the auth system')
        decision = route_model(payload, snap, {})
        assert decision.reason_code == 'classifier_deep'
        # Both blocks joined with a space
        assert decision.classifier_raw_response == 'deep !'

    def test_classifier_raw_response_skips_thinking_blocks(self):
        """thinking/redacted_thinking blocks are skipped; only text blocks matter."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [
                {'type': 'thinking', 'thinking': 'I should say standard', 'signature': 'sig'},
                {'type': 'text', 'text': 'standard'},
            ],
            'stop_reason': 'end_turn',
        }
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='implement auth middleware')
        decision = route_model(payload, snap, {})
        assert decision.classifier_raw_response == 'standard'
        assert 'thinking' not in (decision.classifier_raw_response or '')

    def test_classifier_raw_response_preserves_original_casing(self):
        """Raw response text is not lowercased or stripped."""
        backend = MagicMock()
        del backend.send_classifier_message
        # Uppercase response wrapping — tolerated by parse_classifier_label
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': 'STANDARD'}],
            'stop_reason': 'end_turn',
        }
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='implement auth middleware')
        decision = route_model(payload, snap, {})
        # Raw response preserves case even though the parser lowercases for matching
        assert decision.classifier_raw_response == 'STANDARD'

    def test_classifier_format_standard_when_confidence_bump_off(self):
        """classifier_format is 'standard' when confidence_bump is disabled."""
        decision, snap = self._run_classifier(label='standard',
                                              content='fix the bug')
        assert snap.config.auto_model_routing_confidence_bump is False
        assert decision.classifier_format == 'standard'

    def test_classifier_format_json_when_confidence_bump_on(self):
        """classifier_format is 'json' when confidence_bump mode is enabled."""
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {
            'content': [{'type': 'text', 'text': '{"score":55}'}],
            'stop_reason': 'end_turn',
        }
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_confidence_bump = True
        snap.config.auto_model_routing_min_confidence = 0.0
        payload = _payload(model='sonnet', content='fix the bug')
        decision = route_model(payload, snap, {})
        assert decision.classifier_format == 'json'
        assert decision.reason_code in (
            'classifier_standard', 'classifier_trivial', 'classifier_deep',
        )

    def test_all_four_fields_populated_on_trivial(self):
        """All 4 fields are set on classifier_trivial."""
        decision, _ = self._run_classifier(label='trivial', content='hi',
                                           classifier_model='haiku')
        assert decision.reason_code == 'classifier_trivial'
        assert decision.classifier_model == 'haiku'
        assert decision.classifier_summary_json is not None
        assert decision.classifier_raw_response == 'trivial'
        assert decision.classifier_format == 'standard'

    def test_all_four_fields_populated_on_deep(self):
        """All 4 fields are set on classifier_deep."""
        decision, _ = self._run_classifier(label='deep',
                                           content='redesign the auth system',
                                           classifier_model='haiku')
        assert decision.reason_code == 'classifier_deep'
        assert decision.classifier_model == 'haiku'
        assert decision.classifier_summary_json is not None
        assert decision.classifier_raw_response == 'deep'
        assert decision.classifier_format == 'standard'

    def test_dataclasses_replace_preserves_new_fields(self):
        """dataclasses.replace() on a decision preserves the 4 new fields."""
        from dataclasses import replace
        from anthproxy.model_router import ModelRoutingDecision
        original = ModelRoutingDecision(
            requested_model='sonnet',
            routed_model='opus',
            classification='deep',
            applied=True,
            reason_code='classifier_deep',
            classifier_model='haiku',
            classifier_summary_json='{"final_user_text":"redesign"}',
            classifier_raw_response='deep',
            classifier_format='standard',
        )
        replaced = replace(original, reason_code='classifier_deep', applied=True)
        assert replaced.classifier_model == 'haiku'
        assert replaced.classifier_summary_json == '{"final_user_text":"redesign"}'
        assert replaced.classifier_raw_response == 'deep'
        assert replaced.classifier_format == 'standard'


# ---------------------------------------------------------------------------
# ADR 0010/0011/0012: Weighted system-prompt + user-prompt tier blend
# ---------------------------------------------------------------------------


def _blend_snapshot(sys_label='standard', user_label='standard',
                    has_sys_classifier=True):
    """Snapshot where first send returns user label, second returns sys label."""
    backend = MagicMock()
    if has_sys_classifier:
        # send_classifier_message used for BOTH calls when available
        backend.send_classifier_message = MagicMock(
            side_effect=[_text_response(user_label), _text_response(sys_label)]
        )
    else:
        del backend.send_classifier_message
        backend.send_message = MagicMock(
            side_effect=[_text_response(user_label), _text_response(sys_label)]
        )
    return _snapshot(backend=backend, routing=True)


class TestWeightedBlendConfig:
    """Config validation for ADR 0010 weighted blend options."""

    def test_weights_must_sum_to_one(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c',
             'from anthproxy.config import parse_args; parse_args(['
             '"--auto-model-routing",'
             '"--auto-model-routing-system-prompt-weight=0.50",'
             '"--auto-model-routing-user-prompt-weight=0.60"'
             '])'],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_zero_system_weight_rejected(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c',
             'from anthproxy.config import parse_args; parse_args(['
             '"--auto-model-routing",'
             '"--auto-model-routing-system-prompt-weight=0.0",'
             '"--auto-model-routing-user-prompt-weight=1.0"'
             '])'],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_inverted_thresholds_rejected(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c',
             'from anthproxy.config import parse_args; parse_args(['
             '"--auto-model-routing",'
             '"--auto-model-routing-trivial-threshold=1.50",'
             '"--auto-model-routing-standard-threshold=0.75"'
             '])'],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_zero_cache_size_rejected(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c',
             'from anthproxy.config import parse_args; parse_args(['
             '"--auto-model-routing",'
             '"--auto-model-routing-system-prompt-cache-size=0"'
             '])'],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_zero_preview_limit_rejected(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-c',
             'from anthproxy.config import parse_args; parse_args(['
             '"--auto-model-routing",'
             '"--auto-model-routing-system-prompt-preview-limit=0"'
             '])'],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestExtractSystemPromptPreview:
    """_extract_system_prompt_preview extraction and head-capping."""

    def test_string_system_returned_directly(self):
        payload = {'system': 'You are a helpful assistant.'}
        assert _extract_system_prompt_preview(payload, 500) == 'You are a helpful assistant.'

    def test_string_system_head_capped(self):
        payload = {'system': 'A' * 600}
        assert _extract_system_prompt_preview(payload, 100) == 'A' * 100

    def test_list_system_text_blocks_concatenated(self):
        payload = {'system': [
            {'type': 'text', 'text': 'Block one.'},
            {'type': 'text', 'text': 'Block two.'},
        ]}
        assert _extract_system_prompt_preview(payload, 500) == 'Block one.\nBlock two.'

    def test_list_system_non_text_blocks_skipped(self):
        payload = {'system': [
            {'type': 'text', 'text': 'Role: assistant.'},
            {'type': 'image', 'source': {}},
        ]}
        assert _extract_system_prompt_preview(payload, 500) == 'Role: assistant.'

    def test_list_non_dict_items_skipped(self):
        payload = {'system': ['bare string', {'type': 'text', 'text': 'ok'}]}
        assert _extract_system_prompt_preview(payload, 500) == 'ok'

    def test_empty_system_returns_empty(self):
        assert _extract_system_prompt_preview({}, 500) == ''
        assert _extract_system_prompt_preview({'system': ''}, 500) == ''
        assert _extract_system_prompt_preview({'system': None}, 500) == ''

    def test_list_head_capped_after_concatenation(self):
        payload = {'system': [
            {'type': 'text', 'text': 'A' * 300},
            {'type': 'text', 'text': 'B' * 300},
        ]}
        result = _extract_system_prompt_preview(payload, 400)
        assert len(result) == 400


class TestSystemPromptClassifier:
    """_classify_system_prompt: LRU cache, classifier calls, failure handling."""

    def _make_config(self):
        cfg = MagicMock()
        cfg.auto_model_routing_classifier_model = 'haiku'
        cfg.auto_model_routing_confidence_bump = False
        return cfg

    def _make_snap(self, label='standard'):
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = _text_response(label)
        snap = MagicMock()
        snap.backend = backend
        snap.name = 'anthropic'
        return snap

    def setup_method(self):
        with _sys_prompt_cache_lock:
            _sys_prompt_cache.clear()

    def test_no_preview_returns_standard_default(self):
        cfg = self._make_config()
        snap = self._make_snap()
        tier, score, failed = _classify_system_prompt(
            '', None, 256, cfg, snap, {}, ''
        )
        assert tier == 'standard'
        assert score == 1.0
        assert failed is False
        snap.backend.send_message.assert_not_called()

    def test_cache_miss_calls_classifier(self):
        cfg = self._make_config()
        snap = self._make_snap('deep')
        tier, score, failed = _classify_system_prompt(
            'You are a research architect.', 'sha-abc', 256, cfg, snap, {}, ''
        )
        assert tier == 'deep'
        assert score == 2.0
        assert failed is False
        snap.backend.send_message.assert_called_once()

    def test_cache_hit_skips_classifier(self):
        cfg = self._make_config()
        snap = self._make_snap('deep')
        sha = 'sha-cached'
        with _sys_prompt_cache_lock:
            _sys_prompt_cache[sha] = ('trivial', 0.0)
        tier, score, failed = _classify_system_prompt(
            'You are a research architect.', sha, 256, cfg, snap, {}, ''
        )
        assert tier == 'trivial'
        assert score == 0.0
        assert failed is False
        snap.backend.send_message.assert_not_called()

    def test_successful_result_cached(self):
        cfg = self._make_config()
        snap = self._make_snap('deep')
        sha = 'sha-write-test'
        _classify_system_prompt('You are a research architect.', sha, 256, cfg, snap, {}, '')
        with _sys_prompt_cache_lock:
            assert sha in _sys_prompt_cache
            assert _sys_prompt_cache[sha] == ('deep', 2.0)

    def test_failed_classification_not_cached(self):
        cfg = self._make_config()
        snap = MagicMock()
        del snap.send_classifier_message
        snap.backend = MagicMock()
        del snap.backend.send_classifier_message
        snap.backend.send_message.side_effect = RuntimeError('network down')
        snap.name = 'anthropic'
        sha = 'sha-fail'
        tier, score, failed = _classify_system_prompt(
            'You are an agent.', sha, 256, cfg, snap, {}, ''
        )
        assert tier == 'standard'
        assert score == 1.0
        assert failed is True
        with _sys_prompt_cache_lock:
            assert sha not in _sys_prompt_cache

    def test_invalid_label_not_cached(self):
        cfg = self._make_config()
        backend = MagicMock()
        del backend.send_classifier_message
        backend.send_message.return_value = {'content': [{'type': 'text', 'text': 'definitely invalid multi word'}]}
        snap = MagicMock()
        snap.backend = backend
        snap.name = 'anthropic'
        sha = 'sha-invalid'
        tier, score, failed = _classify_system_prompt(
            'You are an agent.', sha, 256, cfg, snap, {}, ''
        )
        assert tier == 'standard'
        assert score == 1.0
        assert failed is True
        with _sys_prompt_cache_lock:
            assert sha not in _sys_prompt_cache

    def test_lru_eviction(self):
        cfg = self._make_config()
        snap = self._make_snap('standard')
        for i in range(3):
            sha = f'sha-{i}'
            with _sys_prompt_cache_lock:
                _sys_prompt_cache[sha] = ('standard', 1.0)
        # Cache size = 2; next write should evict oldest
        _classify_system_prompt('Some prompt.', 'sha-new', 2, cfg, snap, {}, '')
        with _sys_prompt_cache_lock:
            assert 'sha-new' in _sys_prompt_cache
            # sha-0 evicted (oldest), sha-1 and sha-2 or sha-new remain
            assert len(_sys_prompt_cache) <= 2

    def test_none_sha_skips_cache(self):
        cfg = self._make_config()
        snap = self._make_snap('deep')
        tier, score, failed = _classify_system_prompt(
            'You are a research architect.', None, 256, cfg, snap, {}, ''
        )
        assert tier == 'deep'
        # No sha → cache stays empty
        with _sys_prompt_cache_lock:
            assert None not in _sys_prompt_cache


class TestWeightedBlendRouting:
    """route_model() applies weighted blend in classifier mode."""

    def setup_method(self):
        with _sys_prompt_cache_lock:
            _sys_prompt_cache.clear()

    def _snap_with_two_calls(self, user_label: str, sys_label: str):
        """Snapshot whose classifier returns user_label first, sys_label second."""
        backend = MagicMock()
        backend.send_classifier_message = MagicMock(
            side_effect=[_text_response(user_label), _text_response(sys_label)]
        )
        return _snapshot(backend=backend, routing=True)

    def _snap_no_sys(self, user_label: str):
        """Snapshot with no system prompt — sys classifier never called."""
        backend = MagicMock()
        backend.send_classifier_message = MagicMock(
            return_value=_text_response(user_label)
        )
        return _snapshot(backend=backend, routing=True)

    def test_no_system_prompt_uses_standard_default(self):
        """No system prompt → sys_score=1.0; blend = 0.3*1 + 0.7*user_score."""
        snap = self._snap_no_sys('deep')
        payload = _payload(model='sonnet', content='redesign the auth system')
        decision = route_model(payload, snap, {})
        # user_score=2.0, sys_score=1.0 → weighted=0.3*1+0.7*2=1.70 → deep
        assert decision.routed_model == 'opus'
        assert decision.system_prompt_tier == 'standard'
        assert decision.system_prompt_score == 1.0
        assert decision.user_prompt_score == 2.0
        assert abs(decision.routing_weighted_score - 1.70) < 1e-9
        assert decision.system_prompt_classification_failed is False

    def test_trivial_user_trivial_sys_stays_trivial(self):
        """trivial user + trivial sys: 0.3*0 + 0.7*0 = 0.0 < 0.75 → trivial."""
        snap = self._snap_with_two_calls('trivial', 'trivial')
        payload = _payload(model='sonnet', content='hi', system='You are a file browser.')
        decision = route_model(payload, snap, {})
        assert decision.routed_model == 'haiku'
        assert decision.system_prompt_tier == 'trivial'
        assert abs(decision.routing_weighted_score - 0.0) < 1e-9

    def test_deep_user_trivial_sys_moderates_to_standard(self):
        """deep user(2) + trivial sys(0): 0.3*0 + 0.7*2 = 1.40 < 1.50 → standard."""
        snap = self._snap_with_two_calls('deep', 'trivial')
        payload = _payload(model='sonnet', content='redesign the auth system',
                           system='You are a file browser.')
        decision = route_model(payload, snap, {})
        assert decision.routed_model == 'sonnet'  # standard
        assert abs(decision.routing_weighted_score - 1.40) < 1e-9

    def test_trivial_user_deep_sys_elevates_to_standard(self):
        """trivial user(0) + deep sys(2): 0.3*2 + 0.7*0 = 0.60 < 0.75 → trivial."""
        snap = self._snap_with_two_calls('trivial', 'deep')
        payload = _payload(model='sonnet', content='hi',
                           system='You are a research architect.')
        decision = route_model(payload, snap, {})
        # 0.3*2 + 0.7*0 = 0.60 < 0.75 → still trivial
        assert decision.routed_model == 'haiku'
        assert abs(decision.routing_weighted_score - 0.60) < 1e-9

    def test_standard_user_standard_sys_stays_standard(self):
        """standard user(1) + standard sys(1): 0.3*1 + 0.7*1 = 1.0 → standard."""
        snap = self._snap_with_two_calls('standard', 'standard')
        payload = _payload(model='sonnet', content='implement auth',
                           system='You are a helpful assistant.')
        decision = route_model(payload, snap, {})
        assert decision.routed_model == 'sonnet'
        assert abs(decision.routing_weighted_score - 1.0) < 1e-9

    def test_deep_user_deep_sys_stays_deep(self):
        """deep user(2) + deep sys(2): 0.3*2 + 0.7*2 = 2.0 → deep."""
        snap = self._snap_with_two_calls('deep', 'deep')
        payload = _payload(model='sonnet', content='redesign the auth system',
                           system='You are a research architect.')
        decision = route_model(payload, snap, {})
        assert decision.routed_model == 'opus'
        assert abs(decision.routing_weighted_score - 2.0) < 1e-9

    def test_blend_fields_populated_on_success(self):
        """All 5 blend fields are populated on a successful classifier path."""
        snap = self._snap_no_sys('standard')
        payload = _payload(model='sonnet', content='implement auth')
        decision = route_model(payload, snap, {})
        assert decision.system_prompt_tier == 'standard'
        assert decision.system_prompt_score == 1.0
        assert decision.user_prompt_score == 1.0
        assert decision.routing_weighted_score is not None
        assert decision.system_prompt_classification_failed is False

    def test_blend_fields_none_on_disabled_routing(self):
        """Blend fields are None when routing is disabled."""
        snap = _snapshot(routing=False)
        payload = _payload(model='sonnet', content='implement auth')
        decision = route_model(payload, snap, {})
        assert decision.system_prompt_tier is None
        assert decision.system_prompt_score is None
        assert decision.user_prompt_score is None
        assert decision.routing_weighted_score is None
        assert decision.system_prompt_classification_failed is False

    def test_blend_fields_none_on_rules_mode(self):
        """Blend fields are None when effective mode is 'rules'."""
        snap = _snapshot(routing=True)
        snap.config.auto_model_routing_mode = 'rules'
        payload = _payload(model='sonnet', content='hi')
        decision = route_model(payload, snap, {})
        assert decision.system_prompt_tier is None
        assert decision.routing_weighted_score is None

    def test_sys_classif_failure_uses_standard_fallback(self):
        """System prompt classifier failure: sys_score=1.0, classification_failed=True."""
        backend = MagicMock()
        # First call = user prompt → standard
        # Second call (sys prompt) → exception
        backend.send_classifier_message = MagicMock(
            side_effect=[_text_response('standard'), RuntimeError('timeout')]
        )
        snap = _snapshot(backend=backend, routing=True)
        payload = _payload(model='sonnet', content='implement auth',
                           system='You are a helpful assistant.')
        decision = route_model(payload, snap, {})
        assert decision.system_prompt_classification_failed is True
        assert decision.system_prompt_score == 1.0
        assert decision.system_prompt_tier == 'standard'

    def test_sys_classif_result_cached_for_next_request(self):
        """A successful sys-prompt classification is cached so the next request skips it."""
        sys_content = 'You are a file browser.'
        with _sys_prompt_cache_lock:
            _sys_prompt_cache.clear()

        backend = MagicMock()
        backend.send_classifier_message = MagicMock(
            side_effect=[
                _text_response('standard'),  # user prompt 1st request
                _text_response('trivial'),   # sys prompt 1st request
            ]
        )
        snap = _snapshot(backend=backend, routing=True)
        snap.backend = backend

        payload = _payload(model='sonnet', content='implement auth', system=sys_content)
        route_model(payload, snap, {})
        assert backend.send_classifier_message.call_count == 2

        # Second request: sys prompt cached → only user prompt call
        backend.send_classifier_message.reset_mock()
        backend.send_classifier_message.side_effect = [_text_response('standard')]
        payload2 = _payload(model='sonnet', content='another task', system=sys_content)
        route_model(payload2, snap, {})
        assert backend.send_classifier_message.call_count == 1

    def test_affirmation_classified_populates_blend_fields(self):
        """affirmation_classified path also populates the 5 blend fields."""
        with _sys_prompt_cache_lock:
            _sys_prompt_cache.clear()

        # Two calls: first for affirmation user-prompt, second for sys-prompt
        backend = MagicMock()
        backend.send_classifier_message = MagicMock(
            side_effect=[
                _text_response('standard'),  # affirmation classifier
                _text_response('standard'),  # system prompt classifier
            ]
        )
        snap = _snapshot(backend=backend, routing=True)
        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'assistant', 'content': 'I will implement the auth module.'},
                {'role': 'user', 'content': 'yes'},
            ],
        }
        decision = route_model(payload, snap, {}, cached_session_tier=None)
        assert decision.reason_code == 'affirmation_classified'
        assert decision.system_prompt_tier is not None
        assert decision.routing_weighted_score is not None

    def test_affirmation_inherited_blend_fields_none(self):
        """affirmation_inherited path has no blend (cached tier, no classifier call)."""
        snap = _snapshot(routing=True)
        payload = {
            'model': 'sonnet',
            'messages': [{'role': 'user', 'content': 'yes'}],
        }
        decision = route_model(payload, snap, {}, cached_session_tier='sonnet')
        assert decision.reason_code == 'affirmation_inherited'
        assert decision.system_prompt_tier is None
        assert decision.routing_weighted_score is None
        assert decision.system_prompt_classification_failed is False

    def test_classification_raw_label_preserved_after_blend(self):
        """decision.classification holds the raw user-prompt label, not the blended tier."""
        snap = self._snap_with_two_calls('deep', 'trivial')
        payload = _payload(model='sonnet', content='redesign the auth system',
                           system='You are a file browser.')
        decision = route_model(payload, snap, {})
        # classification field must be the raw user label
        assert decision.classification == 'deep'

    def test_score_at_trivial_threshold_routes_standard(self):
        """weighted_score == trivial_threshold: < is False, so result is standard not trivial."""
        # With threshold=0.30: sys=standard(1), user=trivial(0) → 0.30*1 + 0.70*0 = 0.30 exactly
        snap = self._snap_with_two_calls('trivial', 'standard')
        snap.config.auto_model_routing_trivial_threshold = 0.30
        payload = _payload(model='sonnet', content='hi', system='You are a helpful assistant.')
        decision = route_model(payload, snap, {})
        import pytest
        assert decision.routing_weighted_score == pytest.approx(0.30)
        assert decision.routed_model == 'sonnet'  # standard (not trivial)

    def test_score_at_standard_threshold_routes_deep(self):
        """weighted_score == standard_threshold: < is False, so result is deep not standard."""
        # With threshold=1.40: deep(2)*0.70 + trivial(0)*0.30 = 1.40 exactly
        snap = self._snap_with_two_calls('deep', 'trivial')
        snap.config.auto_model_routing_standard_threshold = 1.40
        payload = _payload(model='sonnet', content='redesign', system='You are a file browser.')
        decision = route_model(payload, snap, {})
        import pytest
        assert decision.routing_weighted_score == pytest.approx(1.40)
        assert decision.routed_model == 'opus'  # deep (not standard)

    def test_affirmation_rules_mode_skips_sys_prompt_classifier(self):
        """In rules mode the affirmation path must not call the system-prompt classifier."""
        with _sys_prompt_cache_lock:
            _sys_prompt_cache.clear()

        backend = MagicMock()
        # Only one call allowed: the affirmation user-prompt classifier
        backend.send_classifier_message = MagicMock(
            side_effect=[_text_response('standard')]
        )
        snap = _snapshot(backend=backend, routing=True)
        snap.config.auto_model_routing_mode = 'rules'

        payload = {
            'model': 'sonnet',
            'messages': [
                {'role': 'assistant', 'content': 'Here is the plan.'},
                {'role': 'user', 'content': 'yes'},
            ],
            'system': 'You are a helpful assistant.',
        }
        route_model(payload, snap, {}, cached_session_tier=None)
        # Exactly one call (affirmation user-prompt classifier); system-prompt
        # classifier must not have been invoked.
        assert backend.send_classifier_message.call_count == 1


class TestSystemPromptShaInRoutingSummary:
    """build_routing_summary() computes system_prompt_sha256."""

    def test_sha_computed_for_string_system(self):
        import hashlib
        content = 'You are a helpful assistant.'
        payload = _payload(content='hello', system=content)
        summary = build_routing_summary(payload)
        assert summary is not None
        expected_sha = hashlib.sha256(content.encode()).hexdigest()
        assert summary.system_prompt_sha256 == expected_sha

    def test_sha_computed_for_list_system(self):
        import hashlib
        import json as _json
        system = [{'type': 'text', 'text': 'Role: assistant.'}]
        payload = _payload(content='hello', system=system)
        summary = build_routing_summary(payload)
        assert summary is not None
        expected_sha = hashlib.sha256(
            _json.dumps(system, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        assert summary.system_prompt_sha256 == expected_sha

    def test_sha_none_when_no_system(self):
        payload = _payload(content='hello')
        summary = build_routing_summary(payload)
        assert summary is not None
        assert summary.system_prompt_sha256 is None

    def test_sha_none_for_empty_system(self):
        payload = _payload(content='hello', system='')
        summary = build_routing_summary(payload)
        assert summary is not None
        assert summary.system_prompt_sha256 is None


class TestBuildSystemPromptClassifierPayload:
    """build_system_prompt_classifier_payload() structure."""

    def test_uses_sentinel_key(self):
        cfg = MagicMock()
        cfg.auto_model_routing_classifier_model = 'haiku'
        p = build_system_prompt_classifier_payload('You are a file browser.', cfg)
        assert p.get('_anthproxy_internal_classifier') is True

    def test_uses_correct_system_prompt(self):
        from anthproxy.model_router import _CLASSIFIER_SYSTEM_PROMPT_TIER
        cfg = MagicMock()
        cfg.auto_model_routing_classifier_model = 'haiku'
        p = build_system_prompt_classifier_payload('You are a file browser.', cfg)
        assert p['system'] == _CLASSIFIER_SYSTEM_PROMPT_TIER

    def test_system_preview_in_user_message(self):
        cfg = MagicMock()
        cfg.auto_model_routing_classifier_model = 'haiku'
        preview = 'You are a file browser.'
        p = build_system_prompt_classifier_payload(preview, cfg)
        assert p['messages'][0]['content'] == preview

    def test_max_tokens_is_small(self):
        from anthproxy.model_router import _CLASSIFIER_MAX_TOKENS
        cfg = MagicMock()
        cfg.auto_model_routing_classifier_model = 'haiku'
        p = build_system_prompt_classifier_payload('preview', cfg)
        assert p['max_tokens'] == _CLASSIFIER_MAX_TOKENS

    def test_temperature_zero(self):
        cfg = MagicMock()
        cfg.auto_model_routing_classifier_model = 'haiku'
        p = build_system_prompt_classifier_payload('preview', cfg)
        assert p['temperature'] == 0.0
