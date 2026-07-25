from ..backends_registry import register_backend
from .backend import BedrockBackend

register_backend('bedrock', BedrockBackend)
