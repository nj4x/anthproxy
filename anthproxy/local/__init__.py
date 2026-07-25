from ..backends_registry import register_backend
from .backend import LocalBackend

register_backend('local', LocalBackend)
