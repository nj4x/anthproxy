# Backend-specific behavior

Per-provider transport and mapping quirks.

**Codex**: Accumulate item-level SSE events rather than trusting completed response envelopes. Normalize cached and non-cached input usage into disjoint Anthropic fields before handlers record context, cost, or stats. Streaming startup usage must not override final cumulative usage. Derive prompt cache keys from the nested session ID, with a safe fallback for non-Claude-Code callers.

**Anthropic**: Preserve the required Claude Code system block's ordering when merging beta headers. Keep model-aware payload sanitization in the mapper rather than the router, including capability gates, thinking/context-management compatibility, and folding inline system-role messages into top-level system content.

**Local**: Preserve plain HTTP for local base URLs; do not route them through HTTPS-only helpers.
