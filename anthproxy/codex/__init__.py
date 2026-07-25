from ..backends_registry import register_backend
from .backend import CodexBackend

register_backend('codex', CodexBackend)
