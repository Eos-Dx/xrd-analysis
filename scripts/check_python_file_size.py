#!/usr/bin/env python3
"""Enforce the staged file-size budget for the analysis package."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "xrdanalysis"
DEFAULT_MAX_LINES = 1_000

# Keep empty unless a documented, independently tested removal stage is active.
TEMPORARY_EXEMPTIONS = {}


def line_count(path: Path) -> int:
    """Return physical source lines, including comments and blank lines."""
    return len(path.read_text(encoding="utf-8").splitlines())


def oversized_files(package_root: Path) -> list[tuple[Path, int, int]]:
    """Return files exceeding their default or explicitly grandfathered budget."""
    failures = []
    for path in sorted(package_root.rglob("*.py")):
        relative_path = path.relative_to(package_root).as_posix()
        limit = TEMPORARY_EXEMPTIONS.get(relative_path, DEFAULT_MAX_LINES)
        lines = line_count(path)
        if lines > limit:
            failures.append((path, lines, limit))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=PACKAGE_ROOT,
        help="package directory to check (default: %(default)s)",
    )
    args = parser.parse_args()

    failures = oversized_files(args.package_root)
    if not failures:
        print("PASS: Python source files are within their configured line budgets.")
        return 0

    print("FAIL: Python source files exceed the staged size budget:")
    for path, lines, limit in failures:
        print(f"  {path}: {lines} lines (limit {limit})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
