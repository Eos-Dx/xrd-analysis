#!/usr/bin/env python3
"""CLI tool for validating technical HDF5 containers against schema v0.2.

Usage:
    python validate_technical_h5.py <path_to_h5_file>
    python validate_technical_h5.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hardware.container.v0_2.technical_validator import (
    validate_technical_container,
    print_validation_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate technical HDF5 containers against DIFRA schema v0.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s container.h5
  %(prog)s --strict container.h5
  %(prog)s path/to/technical_*.nxs.h5

Exit codes:
  0 - Valid container
  1 - Invalid container (schema violations)
  2 - Error (file not found, etc.)
        """,
    )
    
    parser.add_argument(
        "file",
        type=str,
        help="Path to technical HDF5 container file",
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop validation on first error (default: report all errors)",
    )
    
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed report, only show pass/fail",
    )
    
    args = parser.parse_args()
    
    # Check file exists
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 2
    
    # Validate
    try:
        is_valid, errors, warnings = validate_technical_container(
            str(file_path), strict=args.strict
        )
    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        return 2
    
    # Report
    if args.quiet:
        if is_valid:
            print(f"✅ VALID: {file_path}")
        else:
            print(f"❌ INVALID: {file_path} ({len(errors)} errors)")
    else:
        print_validation_report(str(file_path), is_valid, errors, warnings)
    
    # Exit code
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
