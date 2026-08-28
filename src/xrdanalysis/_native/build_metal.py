#!/usr/bin/env python3
"""Build the native Apple Metal direct detector Monte Carlo backend."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


NATIVE_DIR = Path(__file__).resolve().parent
SOURCE = NATIVE_DIR / "direct_monte_carlo_metal.mm"
SHADER = NATIVE_DIR / "direct_monte_carlo_metal.metal"


def _default_output() -> Path:
    return NATIVE_DIR / "libxrdanalysis_direct_monte_carlo_metal.dylib"


def _find_compiler(configured: str) -> str:
    candidate = shutil.which(configured)
    if candidate:
        return candidate
    result = subprocess.run(
        ["xcrun", "--find", configured],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    raise RuntimeError(f"Objective-C++ compiler is unavailable: {configured}")


def _compiler_command(compiler: str, output: Path, *, debug: bool) -> list[str]:
    if platform.system() != "Darwin":
        raise RuntimeError("native Metal build is supported only on macOS")
    optimization = ["-O0", "-g"] if debug else ["-O3", "-DNDEBUG"]
    return [
        compiler,
        "-std=c++17",
        *optimization,
        "-fPIC",
        "-fobjc-arc",
        "-fno-fast-math",
        "-Wall",
        "-Wextra",
        "-dynamiclib",
        str(SOURCE),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-Wl,-install_name,@rpath/libxrdanalysis_direct_monte_carlo_metal.dylib",
        "-o",
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler",
        default=os.environ.get("OBJCXX", "clang++"),
        help="Objective-C++ compiler command (default: OBJCXX or clang++)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="output dynamic-library path",
    )
    parser.add_argument(
        "--debug", action="store_true", help="build without optimization"
    )
    args = parser.parse_args()

    if not SOURCE.is_file():
        raise RuntimeError(f"native Metal host source is missing: {SOURCE}")
    if not SHADER.is_file():
        raise RuntimeError(f"native Metal shader source is missing: {SHADER}")
    if platform.system() != "Darwin":
        raise RuntimeError("Metal is available only on macOS")
    compiler = _find_compiler(args.compiler)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="xrdmc-metal-build-", dir=output.parent
    ) as temporary:
        temporary_output = Path(temporary) / output.name
        command = _compiler_command(compiler, temporary_output, debug=args.debug)
        print(" ".join(command))
        subprocess.run(command, check=True)
        os.replace(temporary_output, output)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Metal build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
