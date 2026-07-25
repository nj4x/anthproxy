"""Backend plugin registry.

Backends register themselves at module import time. The registry is the single
source of truth for available backends at runtime.
"""

_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """Register a backend class by name."""
    _BACKENDS[name] = backend_class


def get_backend(name: str) -> type | None:
    """Retrieve a registered backend class by name, or None if not found."""
    return _BACKENDS.get(name)


def list_backends() -> tuple[str, ...]:
    """Return a sorted tuple of all registered backend names."""
    return tuple(sorted(_BACKENDS.keys()))


def clear_registry() -> None:
    """Clear all registered backends. Used by tests for isolation."""
    _BACKENDS.clear()
