from ..backends_registry import register_backend
from .backend import PeerBackend

register_backend('peer', PeerBackend)
