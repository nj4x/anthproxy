"""Configuration migration: move ~/.anthropic and ~/.bedrock into ANTHPROXY_HOME."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _migrate_home(src: Path, dst: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Migrate a credential home directory from src to dst.

    Returns (success, message) where success is True if migration completed or was a no-op.
    """
    if not src.exists():
        return True, f"{src} does not exist (no-op)"

    if dst.exists():
        return False, f"{dst} already exists; migration aborted"

    if dry_run:
        return True, f"would move {src} → {dst}"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True, f"moved {src} → {dst}"
    except Exception as exc:
        return False, f"failed to move {src} → {dst}: {exc}"


def migrate(anthproxy_home: str = '', dry_run: bool = False) -> int:
    """Migrate configuration from ~/.anthropic and ~/.bedrock into ANTHPROXY_HOME.

    Returns exit code (0 = success, 1 = failure).
    """
    # Resolve anthproxy_home
    if not anthproxy_home or not anthproxy_home.strip():
        anthproxy_home = str(Path.home() / '.anthproxy')
    else:
        anthproxy_home = anthproxy_home.strip()

    home_path = Path(anthproxy_home)

    # Paths to migrate
    old_anthropic = Path.home() / '.anthropic'
    old_bedrock = Path.home() / '.bedrock'

    new_anthropic = home_path / 'anthropic'
    new_bedrock = home_path / 'bedrock'

    print(f"ANTHPROXY_HOME: {anthproxy_home}")
    print(f"Dry run: {'yes' if dry_run else 'no'}")
    print()

    # Check each migration
    results = []

    if old_anthropic.exists() or old_bedrock.exists():
        if old_anthropic.exists():
            success, msg = _migrate_home(old_anthropic, new_anthropic, dry_run)
            results.append((success, msg))
            print(f"  anthropic: {msg}")
        else:
            results.append((True, "~/.anthropic does not exist (no-op)"))
            print(f"  anthropic: ~/.anthropic does not exist (no-op)")

        if old_bedrock.exists():
            success, msg = _migrate_home(old_bedrock, new_bedrock, dry_run)
            results.append((success, msg))
            print(f"  bedrock:   {msg}")
        else:
            results.append((True, "~/.bedrock does not exist (no-op)"))
            print(f"  bedrock:   ~/.bedrock does not exist (no-op)")
    else:
        print("No old credential directories found (~/.anthropic, ~/.bedrock)")
        print("Nothing to migrate.")
        return 0

    # Check for stray files to delete
    print()
    stray_files = [
        Path.home() / '.anthropic' / 'anthproxy.db',
        Path.home() / '.anthropic' / 'athproxy.db',
        Path.home() / '.anthropic' / 'claude.sqlite',
        Path.home() / '.anthproxy' / 'watchdog-latest.json',
        Path.home() / '.anthproxy' / 'watchdog-prior.json',
    ]

    stray_to_delete = [f for f in stray_files if f.exists()]
    if stray_to_delete:
        print("Stray files (not tracked by anthproxy code, will be deleted):")
        for f in stray_to_delete:
            print(f"  - {f}")
        if not dry_run:
            for f in stray_to_delete:
                try:
                    f.unlink()
                    print(f"    deleted {f}")
                except Exception as exc:
                    print(f"    failed to delete {f}: {exc}")
                    results.append((False, f"failed to delete {f}"))
        print()

    # Report
    if all(success for success, _ in results):
        if dry_run:
            print("Dry run completed; no changes made. Run again without --dry-run to commit.")
        else:
            print("Migration completed successfully.")
        return 0
    else:
        print("Migration failed:")
        for success, msg in results:
            if not success:
                print(f"  ERROR: {msg}")
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='anthproxy migrate',
        description='Migrate anthproxy credentials from ~/.anthropic and ~/.bedrock '
                    'into $ANTHPROXY_HOME (default ~/.anthproxy)',
    )
    p.add_argument('--dry-run', action='store_true',
                   help='Preview migration without making changes')
    p.add_argument('--anthproxy-home', default='',
                   help='Target ANTHPROXY_HOME directory (default: ~/.anthproxy)')

    args = p.parse_args(argv)

    return migrate(anthproxy_home=args.anthproxy_home, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
