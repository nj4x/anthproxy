from ..backends_registry import register_backend
from .backend import OpenRouterBackend

register_backend('openrouter', OpenRouterBackend)
