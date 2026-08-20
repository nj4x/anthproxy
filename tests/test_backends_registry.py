"""Tests for backends_registry: discovery, registration, hooks, construction."""
from __future__ import annotations

import functools
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import anthproxy
from anthproxy import backends_registry
from anthproxy.backends_registry import (
    BackendDiscoveryError,
    backend_names,
    class_hook,
    get_backend,
    register_backend,
    temporary_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_registry(monkeypatch):
    """Rebind _BACKENDS to a fresh copy for the test; restore afterward.

    Does NOT evict sys.modules — use discovery_isolation for discovery tests.
    """
    saved = backends_registry._BACKENDS
    monkeypatch.setattr(backends_registry, '_BACKENDS', dict(saved))
    yield
    # monkeypatch restores the original reference automatically


@pytest.fixture()
def discovery_isolation(tmp_path, monkeypatch):
    """Discovery-specific isolation: save/rebind registry AND sys.modules.

    Returns a helper that the test uses to write package files under tmp_path.
    Teardown evicts every anthproxy.<name> entry created by the test.
    """
    before_modules = set(sys.modules.keys())
    saved = backends_registry._BACKENDS
    monkeypatch.setattr(backends_registry, '_BACKENDS', dict(saved))

    # Point anthproxy.__path__ at a temporary root containing real packages.
    pkg_root = tmp_path / 'anthproxy_pkgs'
    pkg_root.mkdir()
    original_path = list(anthproxy.__path__)
    monkeypatch.setattr(anthproxy, '__path__', [str(pkg_root)])

    def write_package(name: str, init_body: str = '', backend_body: str = '') -> Path:
        pkg = pkg_root / name
        pkg.mkdir(exist_ok=True)
        (pkg / '__init__.py').write_text(textwrap.dedent(init_body))
        (pkg / 'backend.py').write_text(textwrap.dedent(backend_body))
        return pkg

    yield write_package

    # Restore path
    anthproxy.__path__ = original_path

    # Evict any anthproxy.<foo> modules the test planted
    new_keys = set(sys.modules.keys()) - before_modules
    for key in new_keys:
        if key.startswith('anthproxy.'):
            child = key.split('.', 1)[1].split('.')[0]
            sys.modules.pop(key, None)
            if hasattr(anthproxy, child):
                try:
                    delattr(anthproxy, child)
                except AttributeError:
                    pass

    backends_registry._BACKENDS = saved


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_complete_package_imports_and_registers(self, discovery_isolation):
        write = discovery_isolation
        write(
            'myplugin',
            init_body="""\
                from anthproxy.backends_registry import register_backend
                from .backend import MyPlugin
                register_backend('myplugin', MyPlugin)
            """,
            backend_body="""\
                class MyPlugin:
                    @classmethod
                    def from_config(cls, config):
                        return cls()
                    @classmethod
                    def model_aliases(cls):
                        return {}
                    @classmethod
                    def summary_credentials(cls, snapshot):
                        return None
            """,
        )
        # Add myplugin to declared order so built-in completeness passes
        orig = backends_registry._DECLARED_ORDER
        backends_registry._DECLARED_ORDER = orig + ('myplugin',)
        try:
            # Patch out the real built-ins to avoid re-importing the real packages
            for name in orig:
                if name not in backends_registry._BACKENDS:
                    backends_registry._BACKENDS[name] = object  # type: ignore[assignment]
            backends_registry.discover_backends()
        finally:
            backends_registry._DECLARED_ORDER = orig

        assert get_backend('myplugin') is not None

    def test_backend_py_without_init_py_ignored(self, discovery_isolation):
        write = discovery_isolation
        pkg = write('orphanpkg', init_body='', backend_body='')
        (pkg / '__init__.py').unlink()  # remove init

        orig = backends_registry._DECLARED_ORDER
        # pre-populate built-ins so completeness passes without importing them
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()
        assert get_backend('orphanpkg') is None

    def test_init_py_without_backend_py_ignored(self, discovery_isolation):
        write = discovery_isolation
        pkg = write('halfpkg', init_body='', backend_body='')
        (pkg / 'backend.py').unlink()

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()
        assert get_backend('halfpkg') is None

    def test_non_identifier_name_skipped(self, discovery_isolation, tmp_path):
        pkg_root = Path(str(anthproxy.__path__[0]))
        for bad_name in ('plugin.bak', 'plugin-old', 'plugin copy'):
            bad = pkg_root / bad_name
            bad.mkdir()
            (bad / '__init__.py').write_text('')
            (bad / 'backend.py').write_text('')

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()
        for bad_name in ('plugin.bak', 'plugin-old', 'plugin copy'):
            assert get_backend(bad_name) is None

    def test_keyword_name_skipped(self, discovery_isolation, tmp_path):
        pkg_root = Path(str(anthproxy.__path__[0]))
        bad = pkg_root / 'class'
        bad.mkdir()
        (bad / '__init__.py').write_text('')
        (bad / 'backend.py').write_text('')

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()
        assert get_backend('class') is None

    def test_symlinked_directory_not_discovered(self, discovery_isolation, tmp_path):
        # Create a real package outside the root
        real_pkg = tmp_path / 'realplugin'
        real_pkg.mkdir()
        (real_pkg / '__init__.py').write_text('')
        (real_pkg / 'backend.py').write_text('')

        pkg_root = Path(str(anthproxy.__path__[0]))
        link = pkg_root / 'linkedplugin'
        link.symlink_to(real_pkg)

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()
        assert get_backend('linkedplugin') is None

    def test_plain_module_ignored(self, discovery_isolation):
        pkg_root = Path(str(anthproxy.__path__[0]))
        # A .py file (plain module) rather than a directory
        (pkg_root / 'plainmod.py').write_text('x = 1')

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()
        assert get_backend('plainmod') is None

    def test_shadowing_raises(self, discovery_isolation, tmp_path):
        write = discovery_isolation
        write('shadowpkg', init_body='', backend_body='')

        # Plant a same-named module in sys.modules pointing elsewhere
        fake_module = MagicMock()
        fake_module.__file__ = str(tmp_path / 'somewhere_else.py')
        sys.modules['anthproxy.shadowpkg'] = fake_module

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        with pytest.raises(BackendDiscoveryError, match='identity mismatch'):
            backends_registry.discover_backends()

    def test_zero_entry_path_fails(self, monkeypatch):
        monkeypatch.setattr(anthproxy, '__path__', [])
        with pytest.raises(BackendDiscoveryError, match='0 entries'):
            backends_registry.discover_backends()

    def test_multi_entry_path_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr(anthproxy, '__path__', [str(tmp_path), str(tmp_path)])
        with pytest.raises(BackendDiscoveryError, match='2 entries'):
            backends_registry.discover_backends()

    def test_root_oserror_raises(self, monkeypatch, tmp_path):
        nonexistent = str(tmp_path / 'doesnotexist')
        monkeypatch.setattr(anthproxy, '__path__', [nonexistent])
        with pytest.raises(BackendDiscoveryError, match='Cannot enumerate'):
            backends_registry.discover_backends()

    @pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses permissions')
    def test_child_permission_error_on_required_file_is_fatal(self, discovery_isolation, tmp_path):
        """PermissionError on os.stat(__init__.py) must abort discovery (not silently skip).

        The fatal path is the child-file stat: os.stat(pkg/__init__.py) raises
        PermissionError when the package directory has mode 000 (no execute bit).
        """
        write = discovery_isolation
        pkg = write('permdenied', init_body='', backend_body='')
        os.chmod(str(pkg), 0o000)
        try:
            orig = backends_registry._DECLARED_ORDER
            for name in orig:
                backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
            with pytest.raises(BackendDiscoveryError, match='Cannot access'):
                backends_registry.discover_backends()
        finally:
            os.chmod(str(pkg), 0o755)

    def test_missing_required_file_is_ignored_not_fatal(self, discovery_isolation):
        """Absence of backend.py silently skips — distinct from PermissionError."""
        write = discovery_isolation
        write('incomplete', init_body='', backend_body='')
        pkg = Path(str(anthproxy.__path__[0])) / 'incomplete'
        (pkg / 'backend.py').unlink()

        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        backends_registry.discover_backends()  # must not raise
        assert get_backend('incomplete') is None

    def test_import_failure_raises(self, discovery_isolation, monkeypatch):
        write = discovery_isolation
        write(
            'badimport',
            init_body='raise ImportError("deliberate")',
            backend_body='',
        )
        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        monkeypatch.setattr(backends_registry, '_DECLARED_ORDER', orig)
        with pytest.raises(BackendDiscoveryError, match='Failed to import'):
            backends_registry.discover_backends()

    def test_import_without_register_raises(self, discovery_isolation):
        write = discovery_isolation
        write('noreg', init_body='# no register_backend call', backend_body='')
        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        with pytest.raises(BackendDiscoveryError, match='imported but did not register'):
            backends_registry.discover_backends()

    def test_fail_fast_on_second_package(self, discovery_isolation, monkeypatch):
        write = discovery_isolation
        write(
            'aagood',
            init_body="""\
                from anthproxy.backends_registry import register_backend
                class _G:
                    @classmethod
                    def from_config(cls, c): return cls()
                    @classmethod
                    def model_aliases(cls): return {}
                    @classmethod
                    def summary_credentials(cls, s): return None
                register_backend('aagood', _G)
            """,
            backend_body='',
        )
        write('zzfail', init_body='raise RuntimeError("oops")', backend_body='')

        orig = backends_registry._DECLARED_ORDER
        monkeypatch.setattr(backends_registry, '_DECLARED_ORDER', orig + ('aagood',))
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        with pytest.raises(BackendDiscoveryError, match='Failed to import'):
            backends_registry.discover_backends()
        # 'aagood' was imported and registered before the failure
        assert get_backend('aagood') is not None

    def test_missing_declared_builtin_fails(self, discovery_isolation, monkeypatch):
        """Built-in completeness check fires when a declared backend isn't registered."""
        orig = backends_registry._DECLARED_ORDER
        monkeypatch.setattr(
            backends_registry, '_DECLARED_ORDER', orig + ('missingbuiltin',)
        )
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        with pytest.raises(BackendDiscoveryError, match="'missingbuiltin'.*not registered"):
            backends_registry.discover_backends()

    def test_repeat_discovery_is_harmless(self, monkeypatch):
        """Second discover_backends() call against intact registry must not raise."""
        backends_registry.discover_backends()

    def test_non_backend_package_guard_real_tree(self):
        """_shared and mapper must not be imported/registered by discover_backends()."""
        before_shared = sys.modules.get('anthproxy._shared')
        before_mapper = sys.modules.get('anthproxy.mapper')

        sys.modules.pop('anthproxy._shared', None)
        sys.modules.pop('anthproxy.mapper', None)
        try:
            backends_registry.discover_backends()
            assert 'anthproxy._shared' not in sys.modules, (
                'anthproxy._shared was (re-)imported by discover_backends(); '
                'the reserved-filename rule prohibits backend.py inside _shared/'
            )
            assert 'anthproxy.mapper' not in sys.modules, (
                'anthproxy.mapper was (re-)imported by discover_backends(); '
                'the reserved-filename rule prohibits backend.py inside mapper/'
            )
        finally:
            if before_shared is not None:
                sys.modules['anthproxy._shared'] = before_shared
            if before_mapper is not None:
                sys.modules['anthproxy.mapper'] = before_mapper

    def test_name_directory_mismatch_raises(self, discovery_isolation):
        write = discovery_isolation
        write(
            'mismatch',
            init_body="""\
                from anthproxy.backends_registry import register_backend
                class _M:
                    @classmethod
                    def from_config(cls, c): return cls()
                    @classmethod
                    def model_aliases(cls): return {}
                    @classmethod
                    def summary_credentials(cls, s): return None
                register_backend('othername', _M)
            """,
            backend_body='',
        )
        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        with pytest.raises(BackendDiscoveryError, match="registered under name 'othername'.*directory is 'mismatch'"):
            backends_registry.discover_backends()

    def test_negative_ownership(self, discovery_isolation):
        """A class defined in a different module fails _assert_registered."""
        write = discovery_isolation
        write(
            'foreignreg',
            init_body="""\
                from anthproxy.backends_registry import register_backend
                # Register a class whose module is not anthproxy.foreignreg
                import anthproxy.backends_registry as _r
                class _ForeignClass:
                    pass
                _ForeignClass.__module__ = 'anthproxy.someother'
                register_backend('foreignreg', _ForeignClass)
            """,
            backend_body='',
        )
        orig = backends_registry._DECLARED_ORDER
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        with pytest.raises(BackendDiscoveryError, match='imported but did not register'):
            backends_registry.discover_backends()


# ---------------------------------------------------------------------------
# Registration validation tests
# ---------------------------------------------------------------------------

class TestRegistrationValidation:
    def test_non_string_name_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='invalid backend name'):
            register_backend(123, object)  # type: ignore[arg-type]

    def test_empty_name_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='invalid backend name'):
            register_backend('', object)

    def test_colon_name_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='invalid backend name'):
            register_backend('a:b', object)

    def test_space_name_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='invalid backend name'):
            register_backend('my backend', object)

    def test_auto_is_reserved(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='reserved'):
            register_backend('auto', object)

    def test_subscription_is_reserved(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='reserved'):
            register_backend('subscription', object)

    def test_reserved_message_lists_reserved_set(self, clean_registry):
        with pytest.raises(BackendDiscoveryError) as exc_info:
            register_backend('auto', object)
        assert 'auto' in str(exc_info.value)
        assert 'subscription' in str(exc_info.value)

    def test_same_class_idempotent(self, clean_registry):
        class _C:
            pass
        register_backend('validname', _C)
        register_backend('validname', _C)  # must not raise
        assert get_backend('validname') is _C

    def test_different_class_raises(self, clean_registry):
        class _A:
            pass
        class _B:
            pass
        register_backend('dupname', _A)
        with pytest.raises(BackendDiscoveryError, match='already registered'):
            register_backend('dupname', _B)

    def test_distinct_messages(self, clean_registry):
        """Each guard produces a distinct diagnostic."""
        with pytest.raises(BackendDiscoveryError) as e1:
            register_backend('', object)
        with pytest.raises(BackendDiscoveryError) as e2:
            register_backend('auto', object)

        class _X:
            pass
        register_backend('x', _X)
        with pytest.raises(BackendDiscoveryError) as e3:
            register_backend('x', object)

        assert str(e1.value) != str(e2.value)
        assert str(e2.value) != str(e3.value)
        assert str(e1.value) != str(e3.value)

    def test_colon_name_stats_selector_corruption(self, clean_registry):
        """Pinning that ':'-containing names corrupt _parse_stats_selector."""
        from anthproxy.handlers import _parse_stats_selector
        # If 'a:b' were registered, parsing 'a:b' as stats input would
        # produce two parts and corrupt period-token parsing
        with pytest.raises(BackendDiscoveryError, match='invalid backend name'):
            register_backend('a:b', object)
        # The selector must not match a colon-containing token
        period, backend = _parse_stats_selector('a:b')
        assert backend != 'a:b'


# ---------------------------------------------------------------------------
# Configuration and construction tests
# ---------------------------------------------------------------------------

class TestBackendNames:
    def test_empty_registry_raises(self, monkeypatch):
        monkeypatch.setattr(backends_registry, '_BACKENDS', {})
        with pytest.raises(BackendDiscoveryError, match='empty'):
            backend_names()

    def test_registry_intact_after_empty_test(self):
        """The live registry is still populated after the empty-registry test."""
        names = backend_names()
        assert len(names) > 0

    def test_declared_order_is_prefix(self):
        names = backend_names()
        declared = [n for n in backends_registry._DECLARED_ORDER if n in backends_registry._BACKENDS]
        assert list(names[:len(declared)]) == declared

    def test_extras_are_sorted(self, clean_registry):
        orig_declared = backends_registry._DECLARED_ORDER
        class _P:
            pass
        class _Q:
            pass
        backends_registry._BACKENDS['zzplugin'] = _P
        backends_registry._BACKENDS['aaplugin'] = _Q
        names = backend_names()
        extras = [n for n in names if n not in orig_declared]
        assert extras == sorted(extras)

    def test_plugin_in_choices(self, monkeypatch):
        """A registered plugin must appear in --backend argparse choices."""
        from anthproxy.config import parse_args
        # Use existing backend name; just verify backend_names() drives choices
        names = backend_names()
        with pytest.raises(SystemExit):
            parse_args(['--backend', '__no_such_backend__'])
        # A valid name parses fine
        cfg = parse_args(['--backend', names[0]])
        assert cfg.backend == names[0]


class TestStatsBackendFilters:
    def test_plugin_registered_after_import_appears(self, clean_registry):
        """A plugin registered after handlers import appears in _stats_backend_filters()."""
        from anthproxy import handlers  # noqa: F401
        from anthproxy.handlers import _stats_backend_filters

        class _Fake:
            pass

        result_outside = _stats_backend_filters()
        assert 'fakestats' not in result_outside

        with temporary_registry({'fakestats': _Fake}):
            result_inside = _stats_backend_filters()
            assert 'fakestats' in result_inside

        result_after = _stats_backend_filters()
        assert 'fakestats' not in result_after

    def test_stats_selector_parsing_uses_backend_filter(self, clean_registry):
        """Verify the actual parse path, not just the accessor."""
        from anthproxy.handlers import _parse_stats_selector

        class _F:
            pass

        with temporary_registry({'statsplugin': _F}):
            period, backend = _parse_stats_selector('statsplugin')
            assert backend == 'statsplugin'


class TestClassHook:
    def test_classmethod_returns_callable(self):
        class _C:
            @classmethod
            def myhook(cls):
                return 'ok'
        result = class_hook(_C, 'myhook')
        assert result is not None
        assert result() == 'ok'

    def test_staticmethod_returns_callable(self):
        class _C:
            @staticmethod
            def myhook():
                return 42
        result = class_hook(_C, 'myhook')
        assert result is not None
        assert result() == 42

    def test_absent_returns_none(self):
        class _C:
            pass
        assert class_hook(_C, 'nonexistent') is None

    def test_plain_instance_method_returns_none(self):
        class _C:
            def myhook(self):
                return 'bad'
        assert class_hook(_C, 'myhook') is None

    def test_partial_attribute_returns_none(self):
        class _C:
            pass
        _C.myhook = functools.partial(lambda: None)  # type: ignore[attr-defined]
        assert class_hook(_C, 'myhook') is None

    def test_classmethod_result_is_invocable_not_just_nonnone(self):
        """Returning getattr_static directly would make classmethod non-callable on 3.10."""
        class _C:
            @classmethod
            def myhook(cls):
                return 'invoked'
        hook = class_hook(_C, 'myhook')
        # Must not raise TypeError: 'classmethod' object is not callable
        assert hook() == 'invoked'

    def test_inherited_classmethod_resolves(self):
        """class_hook walks the MRO (via getattr_static) and finds inherited hooks."""
        from anthproxy._shared import Backend
        class _Sub(Backend):
            pass
        result = class_hook(_Sub, 'model_aliases')
        assert result is not None
        assert result() == {}


# ---------------------------------------------------------------------------
# Hook contract tests
# ---------------------------------------------------------------------------

class TestHookContract:
    def test_missing_model_aliases_yields_empty(self, clean_registry):
        class _B:
            pass
        with temporary_registry({'nohooks': _B}):
            from anthproxy import model_config
            result = model_config.model_aliases('nohooks')
        assert result == {}

    def test_model_aliases_as_instance_method_yields_empty_no_typeerror(self, clean_registry):
        class _B:
            def model_aliases(self):
                return {'x': 'y'}
        with temporary_registry({'instmethod': _B}):
            from anthproxy import model_config
            result = model_config.model_aliases('instmethod')
        assert result == {}

    def test_summary_credentials_as_instance_method_returns_none_not_wrong_answer(self):
        class _B:
            def summary_credentials(self, snapshot):
                # If invoked on instance, snapshot binds to self and
                # 'snapshot' arg becomes the actual snapshot — wrong.
                return {'token': 'wrong'}
        snapshot = SimpleNamespace(backend=_B())
        from anthproxy.summary import SummaryDaemon
        daemon = SummaryDaemon(MagicMock(), MagicMock())
        result = daemon._get_credentials(snapshot)
        # class_hook must return None for instance methods, so _get_credentials → None
        assert result is None

    def test_assert_plugin_hooks_rejects_plain_function(self, clean_registry):
        class _B:
            def from_config(self, config):
                return self
        _B.__module__ = 'anthproxy.badhook'
        backends_registry._BACKENDS['badhook'] = _B
        with pytest.raises(BackendDiscoveryError, match="must be a classmethod or staticmethod"):
            backends_registry._assert_plugin_hooks('anthproxy.badhook')

    def test_model_aliases_raising_at_discovery_is_startup_failure(self, discovery_isolation):
        write = discovery_isolation
        write(
            'raisingaliases',
            init_body="""\
                from anthproxy.backends_registry import register_backend
                from .backend import _R
                register_backend('raisingaliases', _R)
            """,
            backend_body="""\
                class _R:
                    @classmethod
                    def from_config(cls, c): return cls()
                    @classmethod
                    def model_aliases(cls):
                        raise ValueError('deliberately bad')
                    @classmethod
                    def summary_credentials(cls, s): return None
            """,
        )
        orig = backends_registry._DECLARED_ORDER
        backends_registry._DECLARED_ORDER = orig + ('raisingaliases',)
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        try:
            with pytest.raises(BackendDiscoveryError, match='model_aliases\\(\\) raised'):
                backends_registry.discover_backends()
        finally:
            backends_registry._DECLARED_ORDER = orig

    def test_model_aliases_raising_in_temporary_registry_yields_empty(self, clean_registry, caplog):
        import logging
        class _B:
            @classmethod
            def model_aliases(cls):
                raise RuntimeError('fake error')
        with temporary_registry({'raiser': _B}):
            from anthproxy import model_config
            with caplog.at_level(logging.WARNING, logger='anthproxy.model_config'):
                result = model_config.model_aliases('raiser')
        assert result == {}
        assert any('model_aliases' in r.message for r in caplog.records)

    def test_model_aliases_non_dict_fails_discovery(self, discovery_isolation):
        write = discovery_isolation
        write(
            'badaliasreturn',
            init_body="""\
                from anthproxy.backends_registry import register_backend
                from .backend import _BR
                register_backend('badaliasreturn', _BR)
            """,
            backend_body="""\
                class _BR:
                    @classmethod
                    def from_config(cls, c): return cls()
                    @classmethod
                    def model_aliases(cls): return ['not', 'a', 'dict']
                    @classmethod
                    def summary_credentials(cls, s): return None
            """,
        )
        orig = backends_registry._DECLARED_ORDER
        backends_registry._DECLARED_ORDER = orig + ('badaliasreturn',)
        for name in orig:
            backends_registry._BACKENDS.setdefault(name, object)  # type: ignore[assignment]
        try:
            with pytest.raises(BackendDiscoveryError, match='model_aliases\\(\\) must return dict'):
                backends_registry.discover_backends()
        finally:
            backends_registry._DECLARED_ORDER = orig


# ---------------------------------------------------------------------------
# Alias merge tests
# ---------------------------------------------------------------------------

class TestAliasMerge:
    def test_file_config_wins_over_plugin(self, clean_registry):
        class _B:
            @classmethod
            def model_aliases(cls):
                return {'sonnet': 'plugin-sonnet'}
        with temporary_registry({'aliasplugin': _B}):
            from anthproxy import model_config
            # Inject a file-config override
            model_config.reset()
            original_load = model_config.load

            def _patched_load():
                data = {'model_aliases': {'aliasplugin': {'sonnet': 'file-sonnet'}}}
                return data

            model_config.load = _patched_load  # type: ignore[assignment]
            try:
                result = model_config.model_aliases('aliasplugin')
                assert result['sonnet'] == 'file-sonnet'
            finally:
                model_config.load = original_load
                model_config.reset()

    def test_plugin_only_uses_plugin_value(self, clean_registry):
        class _B:
            @classmethod
            def model_aliases(cls):
                return {'opus': 'plugin-opus'}
        with temporary_registry({'pluginonly': _B}):
            from anthproxy import model_config
            result = model_config.model_aliases('pluginonly')
        assert result.get('opus') == 'plugin-opus'

    def test_no_mutation_of_plugin_return(self, clean_registry):
        """Merging must not mutate the plugin's original dict."""
        original_aliases = {'opus': 'plugin-opus-original'}

        class _B:
            @classmethod
            def model_aliases(cls):
                return original_aliases

        with temporary_registry({'mutationtest': _B}):
            from anthproxy import model_config

            original_load = model_config.load

            def _patched_load():
                return {'model_aliases': {'mutationtest': {'opus': 'file-override'}}}

            model_config.load = _patched_load  # type: ignore[assignment]
            try:
                model_config.model_aliases('mutationtest')
                assert original_aliases == {'opus': 'plugin-opus-original'}
            finally:
                model_config.load = original_load

    def test_call_time_resolution(self, clean_registry):
        """Plugin registered after model_config import changes resolved aliases."""
        from anthproxy import model_config
        before = model_config.model_aliases('dynamicplugin')
        assert before == {}  # not registered yet

        class _B:
            @classmethod
            def model_aliases(cls):
                return {'haiku': 'dynamic-haiku'}

        with temporary_registry({'dynamicplugin': _B}):
            result = model_config.model_aliases('dynamicplugin')
        assert result.get('haiku') == 'dynamic-haiku'


# ---------------------------------------------------------------------------
# Temporary registry tests
# ---------------------------------------------------------------------------

class TestTemporaryRegistry:
    def test_preserves_builtins(self, clean_registry):
        before = dict(backends_registry._BACKENDS)

        class _F:
            pass

        with temporary_registry({'tempfake': _F}):
            for name in before:
                assert get_backend(name) is before[name]

    def test_never_empty(self, clean_registry):
        class _F:
            pass
        with temporary_registry({'tempne': _F}):
            names = backend_names()
            assert len(names) > 0

    def test_restores_on_exception(self, clean_registry):
        before_ref = backends_registry._BACKENDS

        class _F:
            pass

        try:
            with temporary_registry({'tempexc': _F}):
                raise ValueError('oops')
        except ValueError:
            pass

        assert backends_registry._BACKENDS is before_ref

    def test_restores_exact_reference(self, clean_registry):
        original_ref = backends_registry._BACKENDS

        class _F:
            pass

        with temporary_registry({'tempref': _F}):
            pass

        assert backends_registry._BACKENDS is original_ref

    def test_empty_mapping_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='must not be empty'):
            with temporary_registry({}):
                pass

    def test_invalid_key_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='invalid backend name'):
            with temporary_registry({'a:b': object}):
                pass

    def test_auto_key_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='reserved'):
            with temporary_registry({'auto': object}):
                pass

    def test_subscription_key_raises(self, clean_registry):
        with pytest.raises(BackendDiscoveryError, match='reserved'):
            with temporary_registry({'subscription': object}):
                pass


# ---------------------------------------------------------------------------
# Build backend construction tests
# ---------------------------------------------------------------------------

class TestBuildBackend:
    def test_from_config_receives_config(self, clean_registry):
        """from_config must receive the Config instance — regression against else: cls()."""
        received: list = []

        class _FakeBackend:
            @classmethod
            def from_config(cls, config):
                received.append(config)
                return cls()

        from anthproxy.config import Config
        from anthproxy.server import build_backend

        with temporary_registry({'fakebuild': _FakeBackend}):
            cfg = Config()
            build_backend('fakebuild', cfg)

        assert len(received) == 1 and received[0] is cfg

    def test_missing_from_config_raises_backend_error(self, clean_registry):
        class _NoCfg:
            pass

        from anthproxy.server import BackendError, build_backend

        with temporary_registry({'nocfg': _NoCfg}):
            with pytest.raises(BackendError, match='from_config'):
                from anthproxy.config import Config
                build_backend('nocfg', Config())

    def test_instance_method_from_config_raises_backend_error(self, clean_registry):
        """A plain instance method for from_config must not silently construct a wrong object."""
        class _InstCfg:
            def from_config(self, config):
                return self

        from anthproxy.server import BackendError, build_backend

        with temporary_registry({'instcfg': _InstCfg}):
            with pytest.raises(BackendError, match='from_config'):
                from anthproxy.config import Config
                build_backend('instcfg', Config())

    def test_bedrock_from_config_receives_config(self):
        from anthproxy.bedrock.backend import BedrockBackend
        from anthproxy.config import Config
        cfg = Config()
        b = BedrockBackend.from_config(cfg)
        assert isinstance(b, BedrockBackend)

    def test_zero_arg_backends_use_inherited_default(self):
        from anthproxy.anthropic.backend import AnthropicBackend
        from anthproxy.config import Config
        b = AnthropicBackend.from_config(Config())
        assert isinstance(b, AnthropicBackend)


# ---------------------------------------------------------------------------
# Summary credentials regression tests
# ---------------------------------------------------------------------------

class TestSummaryCredentialsRegression:
    def test_missing_backend_yields_none_not_attribute_error(self):
        """The deleted gauss arm raised AttributeError on Config.gauss_ums_token."""
        from anthproxy.summary import SummaryDaemon

        class _NoCredBackend:
            pass

        daemon = SummaryDaemon(MagicMock(), MagicMock())
        snap = SimpleNamespace(backend=_NoCredBackend(), config=MagicMock())
        result = daemon._get_credentials(snap)
        assert result is None

    def test_anthropic_returns_dict(self):
        from anthproxy.anthropic.backend import AnthropicBackend
        from anthproxy.config import Config
        from anthproxy.summary import SummaryDaemon

        daemon = SummaryDaemon(MagicMock(), MagicMock())
        backend_inst = AnthropicBackend()
        snap = SimpleNamespace(backend=backend_inst, config=Config())
        result = daemon._get_credentials(snap)
        assert isinstance(result, dict)

    def test_local_returns_empty_dict(self):
        from anthproxy.local.backend import LocalBackend
        from anthproxy.config import Config
        from anthproxy.summary import SummaryDaemon

        daemon = SummaryDaemon(MagicMock(), MagicMock())
        backend_inst = LocalBackend()
        snap = SimpleNamespace(backend=backend_inst, config=Config())
        result = daemon._get_credentials(snap)
        assert result == {}

    def test_openrouter_no_key_returns_none(self):
        from anthproxy.config import Config
        from anthproxy.openrouter.backend import OpenRouterBackend
        from anthproxy.summary import SummaryDaemon

        daemon = SummaryDaemon(MagicMock(), MagicMock())
        backend_inst = OpenRouterBackend()
        cfg = Config()
        cfg.openrouter_api_key = ''
        snap = SimpleNamespace(backend=backend_inst, config=cfg)
        result = daemon._get_credentials(snap)
        assert result is None

    def test_config_has_no_plugin_named_attribute(self):
        from anthproxy.config import Config
        assert not hasattr(Config(), 'gauss_ums_token')
        assert not hasattr(Config(), 'plugin_ums_token')


class TestEnabledBackendsFilter:
    """ADR-0020: --backends allowlist filtering at the accessor level."""

    def test_no_filter_means_all_enabled(self):
        assert backends_registry._enabled_backends is None
        names = backend_names()
        assert 'bedrock' in names and 'anthropic' in names

    def test_filter_restricts_backend_names(self):
        backends_registry.set_enabled_backends(frozenset({'anthropic', 'codex'}))
        names = backend_names()
        assert set(names) == {'anthropic', 'codex'}

    def test_filter_restricts_list_backends(self):
        backends_registry.set_enabled_backends(frozenset({'anthropic'}))
        # oauth is internal and always exempt, so it survives the filter.
        assert set(backends_registry.list_backends()) == {'anthropic', 'oauth'}

    def test_filter_restricts_get_backend(self):
        backends_registry.set_enabled_backends(frozenset({'anthropic'}))
        assert get_backend('anthropic') is not None
        assert get_backend('bedrock') is None

    def test_internal_backend_exempt_from_filter(self):
        backends_registry.set_enabled_backends(frozenset({'anthropic'}))
        assert get_backend('oauth') is not None
        assert 'oauth' not in backend_names()  # never listed, filter or not

    def test_reset_to_none_restores_full_set(self):
        backends_registry.set_enabled_backends(frozenset({'anthropic'}))
        assert set(backend_names()) == {'anthropic'}
        backends_registry.set_enabled_backends(None)
        assert set(backend_names()) == set(
            n for n in backends_registry._BACKENDS
            if n not in backends_registry._INTERNAL_BACKENDS
        )

    def test_rediscovery_is_idempotent_under_filter(self):
        """A second discover_backends() call must not choke on a filtered view."""
        backends_registry.set_enabled_backends(frozenset({'anthropic'}))
        backends_registry.discover_backends()  # must not raise
        assert get_backend('bedrock') is None  # filter still installed
        assert 'bedrock' in backends_registry._BACKENDS  # but never removed internally


class TestBackendsCliFlag:
    """ADR-0020: --backends / ANTHPROXY_BACKENDS CLI and env parsing."""

    def test_absent_flag_means_unrestricted(self):
        from anthproxy.config import parse_args
        cfg = parse_args([])
        assert cfg.backends == ()
        assert backends_registry._enabled_backends is None

    def test_valid_allowlist_installs_filter(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--backends', 'anthropic,codex'])
        assert cfg.backends == ('anthropic', 'codex')
        assert set(backend_names()) == {'anthropic', 'codex'}

    def test_whitespace_and_dedup(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--backends', ' anthropic , codex ,anthropic'])
        assert cfg.backends == ('anthropic', 'codex')

    def test_unknown_token_errors(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args(['--backends', 'anthropic,not_a_backend'])

    def test_empty_value_errors(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args(['--backends', ''])

    def test_oauth_token_rejected_as_unknown(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args(['--backends', 'oauth'])

    def test_explicit_backend_outside_allowlist_errors(self):
        from anthproxy.config import parse_args
        with pytest.raises(SystemExit):
            parse_args(['--backends', 'anthropic,codex', '--backend', 'bedrock'])

    def test_explicit_backend_inside_allowlist_succeeds(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--backends', 'anthropic,codex', '--backend', 'codex'])
        assert cfg.backend == 'codex'

    def test_unset_backend_default_is_repaired(self):
        from anthproxy.config import parse_args
        cfg = parse_args(['--backends', 'anthropic,codex'])
        assert cfg.backend in ('anthropic', 'codex')

    def test_env_backend_is_treated_as_explicit(self, monkeypatch):
        from anthproxy.config import parse_args
        monkeypatch.setenv('ANTHPROXY_BACKEND', 'bedrock')
        with pytest.raises(SystemExit):
            parse_args(['--backends', 'anthropic,codex'])
