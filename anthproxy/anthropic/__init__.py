from ..backends_registry import register_backend
from .backend import AnthropicBackend

register_backend('anthropic', AnthropicBackend)
