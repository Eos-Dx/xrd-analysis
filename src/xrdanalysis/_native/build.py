#!/usr/bin/env python3
"""Build the optional C++17/OpenMP direct detector Monte Carlo backend."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "direct_monte_carlo.cpp"
NATIVE_DIR = SOURCE.parent


def _default_output() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return NATIVE_DIR / f"libxrdanalysis_direct_monte_carlo{suffix}"


def _libomp_prefix() -> Path:
    configured = os.environ.get("LIBOMP_PREFIX")
    if configured:
        prefix = Path(configured).expanduser().resolve()
        if (prefix / "include/omp.h").is_file() and (prefix / "lib").is_dir():
            return prefix
        raise RuntimeError(
            f"LIBOMP_PREFIX does not contain libomp headers/libraries: {prefix}"
        )

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser().resolve()
        if (prefix / "include/omp.h").is_file() and (
            prefix / "lib/libomp.dylib"
        ).is_file():
            return prefix

    brew = shutil.which("brew")
    if brew:
        result = subprocess.run(
            [brew, "--prefix", "libomp"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            prefix = Path(result.stdout.strip())
            if (prefix / "include/omp.h").is_file() and (prefix / "lib").is_dir():
                return prefix

    for candidate in (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")):
        if (candidate / "include/omp.h").is_file() and (candidate / "lib").is_dir():
            return candidate
    raise RuntimeError(
        "macOS libomp was not found; install it with 'brew install libomp' or set "
        "LIBOMP_PREFIX"
    )


def _compiler_command(
    compiler: str,
    output: Path,
    *,
    debug: bool,
) -> list[str]:
    system = platform.system()
    optimization = ["-O0", "-g"] if debug else ["-O3", "-DNDEBUG"]
    common = [compiler, "-std=c++17", *optimization, "-fPIC", "-Wall", "-Wextra"]
    if system == "Darwin":
        libomp = _libomp_prefix()
        return [
            *common,
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{libomp / 'include'}",
            "-dynamiclib",
            str(SOURCE),
            f"-L{libomp / 'lib'}",
            "-lomp",
            f"-Wl,-rpath,{libomp / 'lib'}",
            "-o",
            str(output),
        ]
    if system == "Linux":
        return [
            *common,
            "-fopenmp",
            "-shared",
            str(SOURCE),
            "-o",
            str(output),
        ]
    raise RuntimeError(f"unsupported operating system: {system}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler",
        default=os.environ.get("CXX", "c++"),
        help="C++ compiler command (default: CXX or c++)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="output shared-library path",
    )
    parser.add_argument(
        "--debug", action="store_true", help="build without optimization"
    )
    args = parser.parse_args()

    if not SOURCE.is_file():
        raise RuntimeError(f"native source is missing: {SOURCE}")
    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise RuntimeError(f"C++ compiler is unavailable: {args.compiler}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="xrdmc-build-", dir=output.parent
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
        print(f"build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
