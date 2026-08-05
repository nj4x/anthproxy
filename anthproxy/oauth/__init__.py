from ..backends_registry import register_backend
from .backend import OAuthBackend

register_backend('oauth', OAuthBackend)
