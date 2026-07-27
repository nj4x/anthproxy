# Use a specialized classifier system prompt for system-prompt tier inference

System prompt classification is a different task from user-prompt classification. The user-prompt classifier is instructed to "judge complexity ONLY from the user's intent in final_user_text." That instruction is wrong for system prompts, where the question is: "what complexity band does this agent role or instruction set imply?"

We use a separate classifier system prompt (a new constant in `model_router.py`) tuned for the system-prompt task. It teaches the classifier to look for role signals ("you are a file search specialist" → trivial/standard; "you are a research architect" → deep) rather than task signals. The same configured classifier model (`auto_model_routing_classifier_model`) is reused — no separate model config.

## Considered Options

**Reuse the existing user-prompt classifier system prompt:** The existing prompt explicitly tells the model to judge from `final_user_text`, which doesn't exist in the system-prompt classification request. The model would receive contradictory instructions and produce unreliable results. Rejected.

## Consequences

- One new string constant `_CLASSIFIER_SYSTEM_PROMPT_TIER` in `model_router.py`.
- A new `build_system_prompt_classifier_payload(system_preview, config)` function in `model_router.py`.
- System prompt preview is bounded to the first N chars, configured via `auto_model_routing_system_prompt_preview_limit` (default 500), to keep the classifier call cheap.
- **System prompt text extraction:** `payload['system']` may be a plain string or a list of typed content blocks (Anthropic messages format). Before preview-truncating, extract text as follows: if `system` is a `str`, use it directly; if it is a `list`, iterate over elements, check `isinstance(element, dict)` first to defend against non-dict items (bare strings, ints, None), and collect text from dict blocks where `type == 'text'` by reading `element.get('text', '')`, then concatenate with `'\n'` separator to preserve block boundaries. This defensive approach treats malformed or unexpected list elements as contributing empty strings, avoiding AttributeError. Head-cap the concatenated result to `auto_model_routing_system_prompt_preview_limit` chars. If the result is empty (no text blocks or empty string), skip system-prompt classification and apply the no-system-prompt default (score 1.0, tier 'standard') per ADR 0010.
- **Preview limit validation:** startup config validation must assert `auto_model_routing_system_prompt_preview_limit >= 1`; raise a `ConfigurationError` with readable text if the value is 0 or negative. A limit of 0 would silently treat every system prompt as empty and permanently skip system-prompt classification with no warning.
- **Classifier response parsing:** the classifier response for the system prompt is parsed using the existing `parse_classifier_label()` function in `model_router.py` — do not introduce a separate parser. The same label vocabulary (trivial / standard / deep) applies.
- **System-prompt classifier dispatch:** `route_model()` calls a new internal function `_classify_system_prompt(system_preview, system_prompt_sha256, cache, config)` (or extends an existing `_dispatch_classifier_mode` function with a `mode='system_prompt'` branch) to invoke the classifier via the isolated classifier call path. The dispatcher manages the LRU cache (check, miss, classify, write success only), handle failures (fall back to standard score 1.0, do not cache), and return `(tier_string, score_float, classification_failed_bool)` to the blend logic.
