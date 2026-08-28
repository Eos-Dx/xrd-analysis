#!/usr/bin/env python3
"""Build the package-local native Metal direct Monte Carlo library."""

from xrdanalysis._native.build_metal import main


if __name__ == "__main__":
    raise SystemExit(main())
