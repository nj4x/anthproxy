"""Shared Anthropic wire-protocol constants.

These constants are used by backends speaking the Anthropic protocol
(e.g., personal `anthropic` backend and enterprise `oauth` backend).
Placing them here allows backends to depend on `mapper/` (shared) rather
than a sibling backend package.
"""

# Anthropic wire-protocol constants.
ANTHROPIC_HOST = 'api.anthropic.com'
ANTHROPIC_VERSION = '2023-06-01'
MESSAGES_PATH = '/v1/messages?beta=true'
COUNT_TOKENS_PATH = '/v1/messages/count_tokens?beta=true'
CLAUDE_CLI_VERSION = '2.1.222'
USER_AGENT = f'claude-cli/{CLAUDE_CLI_VERSION} (external, cli)'
