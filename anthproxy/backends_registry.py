"""Backend plugin registry and discovery.

Backends register themselves by calling ``register_backend(name, BackendClass)``
from their package ``__init__.py``.  ``discover_backends()`` scans the
``anthproxy`` package directory, imports qualifying packages, and validates
the registration and hook contract.

Public API
----------
    BackendDiscoveryError
    RESERVED_NAMES
    register_backend(name, backend_class)
    get_backend(name) -> type | None
    list_backends() -> tuple[str, ...]   (sorted)
    backend_names() -> tuple[str, ...]   (declared order + discovered extras)
    internal_backend_names() -> frozenset[str]   (always-enabled, allowlist-exempt names)
    discover_backends() -> None
    class_hook(backend_class, name) -> callable | None
    set_enabled_backends(allowed) -> None   (allowlist filter; see ADR-0020)
    temporary_registry(mapping)          (context manager, test-only)

Allowlist filtering (ADR-0020)
-------------------------------
``_BACKENDS`` always holds every discovered backend; ``set_enabled_backends()``
installs an optional filter consulted by ``get_backend()``, ``list_backends()``,
and ``backend_names()``. Internal backends (``_INTERNAL_BACKENDS``) are exempt
from the filter and remain resolvable regardless of the allowlist. Call once,
from ``config.parse_args()``, before any server thread starts.
"""

import importlib
import inspect
import os
import re
import stat as _stat_module
from contextlib import contextmanager

from .constants import SESSION_SUBSCRIPTION_SENTINEL, VALID_BACKEND_MODES


class BackendDiscoveryError(Exception):
    """Raised when backend discovery or registration fails."""


# Names that may never be registered as backends.
RESERVED_NAMES: frozenset[str] = frozenset(VALID_BACKEND_MODES) | {SESSION_SUBSCRIPTION_SENTINEL}

# Canonical order for user-facing lists (CLI choices, help text, etc.).
_DECLARED_ORDER: tuple[str, ...] = (
    'bedrock', 'codex', 'anthropic', 'local', 'openrouter', 'peer',
)
_INTERNAL_BACKENDS: frozenset[str] = frozenset({'oauth'})

_BACKENDS: dict[str, type] = {}

# None means "no filter installed" (all discovered backends are enabled).
# Once installed by set_enabled_backends(), this holds the allowed name set;
# _INTERNAL_BACKENDS are always exempt from it (see module docstring).
_enabled_backends: frozenset[str] | None = None

_NAME_RE = re.compile(r'^[a-z][a-z0-9_-]*$')


# ---------------------------------------------------------------------------
# Core registry operations
# ---------------------------------------------------------------------------

def register_backend(name: str, backend_class: type) -> None:
    """Register a backend class under *name*.

    Called from a backend package's ``__init__.py``.  Validates name format,
    reserved-name constraints, and duplicate-class conflicts before writing.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise BackendDiscoveryError(f'invalid backend name: {name!r}')
    if name in RESERVED_NAMES:
        raise BackendDiscoveryError(
            f'backend name {name!r} is reserved; reserved names: {sorted(RESERVED_NAMES)}'
        )
    existing = _BACKENDS.get(name)
    if existing is not None and existing is not backend_class:
        raise BackendDiscoveryError(
            f'backend {name!r} already registered by '
            f'{existing.__module__}.{existing.__qualname__}'
        )
    _BACKENDS[name] = backend_class


def _is_enabled(name: str) -> bool:
    """True if *name* is usable under the current allowlist filter.

    Internal backends are always exempt from the filter. When no filter is
    installed (``_enabled_backends is None``), every registered name is enabled.
    """
    if name in _INTERNAL_BACKENDS:
        return True
    return _enabled_backends is None or name in _enabled_backends


def set_enabled_backends(allowed: 'frozenset[str] | None') -> None:
    """Install (or clear, with ``None``) the allowlist filter.

    Called exactly once, from ``config.parse_args()``, before any server
    thread starts — never from live server code. *allowed* must already be
    validated against the unfiltered ``backend_names()`` by the caller; this
    function does not re-validate. Internal backends are implicitly always
    enabled and need not be included in *allowed*.
    """
    global _enabled_backends
    _enabled_backends = allowed


def get_backend(name: str) -> type | None:
    """Return the registered backend class for *name*, or ``None``.

    Returns ``None`` for a name excluded by the allowlist filter, exactly as
    for an unregistered name — callers cannot distinguish the two cases.
    """
    if not _is_enabled(name):
        return None
    return _BACKENDS.get(name)


def list_backends() -> tuple[str, ...]:
    """Return all registered, allowlist-enabled backend names, sorted."""
    return tuple(sorted(n for n in _BACKENDS if _is_enabled(n)))


def backend_names() -> tuple[str, ...]:
    """Return enabled backend names in declared order, with discovered extras appended sorted.

    Raises ``BackendDiscoveryError`` when the registry is empty — a signal that
    ``discover_backends()`` has not been called.
    """
    if not _BACKENDS:
        raise BackendDiscoveryError(
            'backend registry is empty — did you forget to call discover_backends()?'
        )
    declared = [n for n in _DECLARED_ORDER if n in _BACKENDS and _is_enabled(n)]
    extras = sorted(
        n for n in _BACKENDS
        if n not in _DECLARED_ORDER and n not in _INTERNAL_BACKENDS and _is_enabled(n)
    )
    return tuple(declared + extras)


def internal_backend_names() -> frozenset[str]:
    """Return the always-enabled, allowlist-exempt internal backend names.

    Public accessor for ``_INTERNAL_BACKENDS`` — no other module may import
    the underscore-prefixed symbol directly (see module docstring).
    """
    return _INTERNAL_BACKENDS


def class_hook(backend_class: type, name: str):
    """Return the class-bound callable for hook *name*, or ``None``.

    Uses ``inspect.getattr_static`` to verify the attribute is a ``classmethod``
    or ``staticmethod`` (not a plain instance method), then returns the bound
    callable via ``getattr`` so descriptors resolve correctly.

    Returns ``None`` for absent hooks, non-class-bound hooks, and any hook
    declared as a plain function.
    """
    raw = inspect.getattr_static(backend_class, name, None)
    if not isinstance(raw, (classmethod, staticmethod)):
        return None
    return getattr(backend_class, name)


# ---------------------------------------------------------------------------
# Discovery internals
# ---------------------------------------------------------------------------

def _owning_package(cls: type) -> str:
    """Two-segment dotted package name of a class (e.g. 'anthproxy.codex')."""
    return '.'.join(cls.__module__.split('.')[:2])


def _assert_registered(package_name: str) -> None:
    """Raise if no class owned by *package_name* is in the registry."""
    for cls in _BACKENDS.values():
        if _owning_package(cls) == package_name:
            return
    raise BackendDiscoveryError(
        f'{package_name!r} imported but did not register a backend. '
        f"Add: register_backend('<name>', <BackendClass>) in "
        f'{package_name.split(".")[-1]}/__init__.py'
    )


def _assert_plugin_hooks(package_name: str) -> None:
    """Validate hook shapes for every class owned by *package_name*."""
    for cls in _BACKENDS.values():
        if _owning_package(cls) != package_name:
            continue
        label = f'{cls.__module__}.{cls.__qualname__}'
        for hook_name in ('from_config', 'model_aliases', 'summary_credentials'):
            raw = inspect.getattr_static(cls, hook_name, None)
            if raw is None:
                continue
            if not isinstance(raw, (classmethod, staticmethod)):
                raise BackendDiscoveryError(
                    f'{label}: {hook_name!r} must be a classmethod or staticmethod, '
                    f'not a plain function'
                )
        hook = class_hook(cls, 'model_aliases')
        if hook is not None:
            try:
                result = hook()
            except Exception as exc:
                raise BackendDiscoveryError(
                    f'{label}: model_aliases() raised during startup validation; '
                    f'it must return a dict[str, str] without raising'
                ) from exc
            if not isinstance(result, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in result.items()
            ):
                raise BackendDiscoveryError(
                    f'{label}: model_aliases() must return dict[str, str], '
                    f'got {type(result)!r}'
                )


def _stat_file(path: str) -> bool:
    """Return True if *path* is a regular file; False on FileNotFoundError.

    Any other ``OSError`` (e.g. ``PermissionError``) is re-raised as
    ``BackendDiscoveryError`` — "cannot access" is epistemically different from
    "absent".
    """
    try:
        st = os.stat(path, follow_symlinks=False)
        return _stat_module.S_ISREG(st.st_mode)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise BackendDiscoveryError(
            f'Cannot access {path!r}: check read permissions'
        ) from exc


def discover_backends() -> None:
    """Scan the ``anthproxy`` package directory and load qualifying plugins.

    A qualifying package is a *real* (non-symlinked) direct child directory of
    the ``anthproxy`` root that contains both ``__init__.py`` and ``backend.py``
    as regular files and whose directory name is a valid Python identifier that
    is not a keyword.

    Failure modes:
    - **Ignored:** non-identifier names, keyword names, symlinked directories,
      missing ``__init__.py``, missing ``backend.py``.
    - **Fatal:** filesystem errors, import failures, post-import identity
      mismatches, missing/invalid registrations, malformed hooks, built-in
      completeness failures.
    """
    import anthproxy
    import keyword

    roots = list(anthproxy.__path__)
    if len(roots) != 1:
        raise BackendDiscoveryError(
            f'anthproxy.__path__ has {len(roots)} entries {roots!r}; '
            f'multi-root package paths are not supported'
        )
    root = roots[0]

    candidates: list[tuple[str, str]] = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                child_name = entry.name
                if not child_name.isidentifier() or keyword.iskeyword(child_name):
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                candidates.append((child_name, entry.path))
    except OSError as exc:
        raise BackendDiscoveryError(
            f'Cannot enumerate {root!r}: check existence and read permissions'
        ) from exc

    candidates.sort()

    for child_name, child_path in candidates:
        init_py = os.path.join(child_path, '__init__.py')
        backend_py = os.path.join(child_path, 'backend.py')

        if not _stat_file(init_py) or not _stat_file(backend_py):
            continue

        fqname = f'anthproxy.{child_name}'

        try:
            module = importlib.import_module(fqname)
        except BackendDiscoveryError:
            raise
        except Exception as exc:
            raise BackendDiscoveryError(
                f'Failed to import {fqname!r}: fix the package or remove the '
                f'__init__.py + backend.py pair'
            ) from exc

        # Post-import identity: guard against same-named plain modules and stale
        # sys.modules entries planted by tests.
        module_file = getattr(module, '__file__', None)
        expected_real = os.path.realpath(init_py)
        actual_real = os.path.realpath(str(module_file)) if module_file is not None else None
        if actual_real != expected_real:
            raise BackendDiscoveryError(
                f'Package {fqname!r}: identity mismatch — '
                f'expected {expected_real!r}, got {actual_real!r}. '
                f'A same-named plain module may be shadowing the package.'
            )

        _assert_registered(fqname)
        _assert_plugin_hooks(fqname)

        # Name/directory agreement: the registered name must match the directory.
        for reg_name, cls in _BACKENDS.items():
            if _owning_package(cls) == fqname and reg_name != child_name:
                raise BackendDiscoveryError(
                    f'{fqname!r} registered under name {reg_name!r} '
                    f'but directory is {child_name!r}: registered name must match directory'
                )

    # Built-in completeness: every declared built-in must be registered.
    for name in _DECLARED_ORDER:
        if name not in _BACKENDS:
            raise BackendDiscoveryError(
                f'Declared built-in {name!r} is not registered. '
                f'Ensure anthproxy.{name} is importable and calls register_backend().'
            )


# ---------------------------------------------------------------------------
# Test-only overlay
# ---------------------------------------------------------------------------

@contextmanager
def temporary_registry(mapping: dict):
    """Atomically overlay ``_BACKENDS`` with *mapping* for the duration of the block.

    Validates name keys (format and reserved-name checks) but does not run
    discovery-time hook validation.  Raises ``BackendDiscoveryError`` on an
    empty replacement or invalid/reserved key.

    **Not thread-safe.**  Never use while live server/request threads are active.
    """
    if not mapping:
        raise BackendDiscoveryError('temporary_registry: mapping must not be empty')
    for key in mapping:
        if not isinstance(key, str) or not _NAME_RE.fullmatch(key):
            raise BackendDiscoveryError(f'invalid backend name: {key!r}')
        if key in RESERVED_NAMES:
            raise BackendDiscoveryError(
                f'backend name {key!r} is reserved; reserved names: {sorted(RESERVED_NAMES)}'
            )

    global _BACKENDS
    saved = _BACKENDS
    _BACKENDS = {**saved, **mapping}
    try:
        yield
    finally:
        _BACKENDS = saved
