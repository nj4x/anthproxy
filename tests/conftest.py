"""pytest configuration for the anthproxy test suite.

Isolates the ANTHPROXY_CONFIG env var so tests always use built-in defaults
and are never affected by a real ``~/.anthproxy/config.json`` on the
developer's machine.  Also resets the model_config cache between tests so
overrides in one test don't bleed into the next.
"""
import os
import tempfile
from pathlib import Path

import pytest


_ORIGINAL_CONFIG = os.environ.get('ANTHPROXY_CONFIG')
_COLLECTION_CONFIG = str(Path(tempfile.gettempdir()) / f'anthproxy-test-{os.getpid()}.json')
os.environ['ANTHPROXY_CONFIG'] = _COLLECTION_CONFIG

# Discover and register all backends once at collection time so the registry is populated
from anthproxy.server import discover_backends  # noqa: E402
discover_backends()


@pytest.fixture(autouse=True)
def _isolate_enabled_backends():
    """Reset the backend allowlist filter after each test.

    Prevents a test that installs ``--backends`` / calls
    ``set_enabled_backends()`` directly from leaking a restricted registry
    view into unrelated tests.
    """
    from anthproxy import backends_registry

    yield
    backends_registry.set_enabled_backends(None)


@pytest.fixture(autouse=True)
def _isolate_model_config(tmp_path):
    """Point ANTHPROXY_CONFIG at a non-existent path so defaults are used,
    and clear the module-level cache before and after each test."""
    from anthproxy import model_config

    os.environ['ANTHPROXY_CONFIG'] = str(tmp_path / 'test_config.json')
    model_config.reset()

    yield

    model_config.reset()
    os.environ['ANTHPROXY_CONFIG'] = _COLLECTION_CONFIG


def pytest_unconfigure(config):
    if _ORIGINAL_CONFIG is None:
        os.environ.pop('ANTHPROXY_CONFIG', None)
    else:
        os.environ['ANTHPROXY_CONFIG'] = _ORIGINAL_CONFIG
