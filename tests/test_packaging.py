"""Standing packaging assertion: every tracked anthproxy package is covered by
the setuptools include allow-list in pyproject.toml.

This catches the case where a new anthproxy subpackage is added (e.g. a nested
anthproxy/_shared/base/) without updating pyproject.toml, which would cause it
to be absent from the wheel while passing all tests on a source checkout.
"""

import fnmatch
import subprocess
import tomllib
from pathlib import Path


def _tracked_anthproxy_packages() -> list[str]:
    result = subprocess.run(
        ['git', 'ls-files', '--', 'anthproxy/'],
        capture_output=True,
        text=True,
        check=True,
    )
    packages = set()
    for path in result.stdout.splitlines():
        p = Path(path)
        if p.name == '__init__.py':
            dotted = '.'.join(p.parent.parts)
            packages.add(dotted)
    return sorted(packages)


def _include_patterns() -> list[str]:
    pyproject = Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject, 'rb') as f:
        data = tomllib.load(f)
    return data['tool']['setuptools']['packages']['find']['include']


def test_all_tracked_anthproxy_packages_match_include_list():
    packages = _tracked_anthproxy_packages()
    patterns = _include_patterns()

    unmatched = []
    for pkg in packages:
        if not any(fnmatch.fnmatchcase(pkg, pat) for pat in patterns):
            unmatched.append(pkg)

    assert not unmatched, (
        "These tracked anthproxy packages are not covered by the setuptools "
        "include allow-list in pyproject.toml and will be absent from the wheel:\n"
        + '\n'.join(f'  {p}' for p in unmatched)
        + "\nAdd a matching include pattern or update the package path. "
        "Note: the reserved-filename rule prohibits 'backend.py' inside "
        "anthproxy/_shared/ or anthproxy/mapper/ — use a different filename "
        "or a nested subpackage (e.g. anthproxy/_shared/base/)."
    )
