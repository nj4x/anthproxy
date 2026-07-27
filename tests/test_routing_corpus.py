"""Routing regression corpus tests.

Loads ``tests/fixtures/routing_cases.jsonl`` line by line and drives each case
through ``route_model()`` with a fake snapshot and an injected classifier backend,
then asserts ``reason_code``, ``applied``, and that the routed model contains the
expected tier substring.

Design rules (matching the constraint set for this module):
- No real model IDs, real credentials, or real API calls.
- No invented stubs; snapshot/config helpers mirror the patterns in
  ``test_model_router.py`` exactly.
- Corpus is fully deterministic and offline.
- Logging is file-only; no console output.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from anthproxy.model_router import ModelRoutingDecision, route_model

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_FIXTURES_DIR = pathlib.Path(__file__).parent / 'fixtures'
_CORPUS_PATH = _FIXTURES_DIR / 'routing_cases.jsonl'


def _load_cases() -> list[dict]:
    cases: list[dict] = []
    with _CORPUS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


_CASES: list[dict] = _load_cases()
_CASE_IDS: list[str] = [c['id'] for c in _CASES]


# ---------------------------------------------------------------------------
# Helpers — mirrors test_model_router._config / _snapshot exactly
# ---------------------------------------------------------------------------

def _config(
    routing: bool = True,
    classifier_model: str = 'haiku',
    long_context_threshold: int = 190_000,
    affirmation_inherit: bool = True,
    classification: dict | None = None,
    auto_model_routing_long: str = 'opus[1m]',
):
    cfg = MagicMock()
    cfg.auto_model_routing = routing
    cfg.auto_model_routing_classifier_model = classifier_model
    cfg.auto_model_routing_long_context_threshold = long_context_threshold
    cfg.auto_model_routing_affirmation_inherit = affirmation_inherit
    cfg.auto_model_routing_classification = (
        dict(classification) if classification is not None
        else {'trivial': 'haiku', 'standard': 'sonnet', 'deep': 'opus'}
    )
    cfg.auto_model_routing_long = auto_model_routing_long
    # Concrete defaults for routing-mode / confidence-bump fields so tests that
    # mock plain-text classifier responses work with the standard (non-JSON) path.
    cfg.auto_model_routing_confidence_bump = False
    cfg.auto_model_routing_min_confidence = 0.0
    cfg.auto_model_routing_mode = 'classifier'
    cfg.auto_model_routing_task_tiers = None
    cfg.auto_model_routing_prior_response_summary_limit = 1000
    # ADR 0010/0012: weighted blend config — concrete values so comparisons work.
    cfg.auto_model_routing_system_prompt_weight = 0.30
    cfg.auto_model_routing_user_prompt_weight = 0.70
    cfg.auto_model_routing_trivial_threshold = 0.75
    cfg.auto_model_routing_standard_threshold = 1.50
    cfg.auto_model_routing_system_prompt_cache_size = 256
    cfg.auto_model_routing_system_prompt_preview_limit = 500
    return cfg


def _config_from_overrides(overrides: dict):
    """Build a fake Config from default values merged with ``overrides``."""
    defaults: dict[str, Any] = {
        'routing': True,
        'classifier_model': 'haiku',
        'long_context_threshold': 190_000,
        'affirmation_inherit': True,
        'classification': None,
        'auto_model_routing_long': 'opus[1m]',
    }
    # Map fixture override keys → _config() parameter names
    _key_map = {
        'auto_model_routing': 'routing',
        'auto_model_routing_classifier_model': 'classifier_model',
        'auto_model_routing_long_context_threshold': 'long_context_threshold',
        'auto_model_routing_affirmation_inherit': 'affirmation_inherit',
        'auto_model_routing_classification': 'classification',
        'auto_model_routing_long': 'auto_model_routing_long',
    }
    for fixture_key, param in _key_map.items():
        if fixture_key in overrides:
            defaults[param] = overrides[fixture_key]
    return _config(**defaults)


def _text_response(label: str) -> dict:
    return {'content': [{'type': 'text', 'text': label}], 'stop_reason': 'end_turn'}


def _make_backend(
    fake_classifier_label: str | None,
    raw_classifier_response: str | None,
) -> MagicMock:
    """Create a backend mock that controls what the classifier call returns.

    Priority:
    1. If ``raw_classifier_response`` is set, the classifier returns that raw text
       (useful for invalid-label cases).
    2. If ``fake_classifier_label`` is set, the classifier returns a valid label.
    3. If both are None, calling the classifier raises ``AssertionError`` — the
       test will fail if the classifier is called unexpectedly.
    """
    backend = MagicMock()
    # Remove send_classifier_message so route_model falls back to send_message,
    # matching the duck-typed pattern used in test_model_router.py.
    del backend.send_classifier_message

    if raw_classifier_response is not None:
        backend.send_message.return_value = _text_response(raw_classifier_response)
    elif fake_classifier_label is not None:
        backend.send_message.return_value = _text_response(fake_classifier_label)
    else:
        # Classifier must NOT be called for this case.
        backend.send_message.side_effect = AssertionError(
            'classifier (send_message) was called unexpectedly for this corpus case'
        )

    return backend


def _snapshot(cfg, backend: MagicMock) -> MagicMock:
    snap = MagicMock()
    snap.config = cfg
    snap.name = 'anthropic'
    snap.backend = backend
    return snap


# ---------------------------------------------------------------------------
# Corpus parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', _CASES, ids=_CASE_IDS)
def test_routing_corpus(case: dict) -> None:
    """Drive one corpus case through route_model() and verify its decision."""
    # Handler-level bypass cases (e.g. override_no_classifier) are not exercised
    # via route_model(); they are handled entirely in handlers.py.  Skip them here
    # and rely on test_handlers_routing.py for coverage.
    if case.get('is_handler_level'):
        pytest.skip(
            f"Case '{case['id']}' is a handler-level bypass not testable via "
            'route_model(); see test_handlers_routing.py.'
        )

    # Build inputs
    payload: dict = json.loads(json.dumps(case['payload']))  # deep copy
    config_overrides: dict = case.get('config_overrides', {})
    cfg = _config_from_overrides(config_overrides)
    backend = _make_backend(
        fake_classifier_label=case.get('fake_classifier_label'),
        raw_classifier_response=case.get('raw_classifier_response'),
    )
    snap = _snapshot(cfg, backend)

    # Call route_model
    decision: ModelRoutingDecision = route_model(
        payload,
        snap,
        credentials={},
        cached_session_tier=case.get('cached_session_tier'),
        session_context_tokens=case.get('session_context_tokens', 0),
        session_estimate_ratio=case.get('session_estimate_ratio', 1.0),
    )

    # Assertions
    assert decision.reason_code == case['expected_reason_code'], (
        f"reason_code mismatch: got {decision.reason_code!r}, "
        f"expected {case['expected_reason_code']!r}"
    )

    assert decision.applied is case['expected_applied'], (
        f"applied mismatch: got {decision.applied!r}, "
        f"expected {case['expected_applied']!r}"
    )

    expected_contains: str = case['expected_routed_model_contains']
    assert expected_contains in decision.routed_model, (
        f"routed_model {decision.routed_model!r} does not contain "
        f"{expected_contains!r}"
    )

    # Also verify that payload['model'] was rewritten consistently with the decision
    assert payload['model'] == decision.routed_model, (
        f"payload['model'] ({payload['model']!r}) diverged from "
        f"decision.routed_model ({decision.routed_model!r})"
    )
