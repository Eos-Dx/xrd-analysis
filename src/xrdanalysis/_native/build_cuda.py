#!/usr/bin/env python3
"""Build the optional native CUDA direct detector Monte Carlo backend."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "direct_monte_carlo_cuda.cu"
NATIVE_DIR = SOURCE.parent
MINIMUM_COMPUTE_CAPABILITY = 60


def _default_output() -> Path:
    return NATIVE_DIR / "libxrdanalysis_direct_monte_carlo_cuda.so"


def _find_nvcc(configured: str) -> str:
    candidate = shutil.which(configured)
    if candidate:
        return candidate

    for variable in ("CUDA_HOME", "CUDA_PATH"):
        prefix = os.environ.get(variable)
        if prefix:
            executable = Path(prefix).expanduser() / "bin" / "nvcc"
            if executable.is_file():
                return str(executable.resolve())
    raise RuntimeError(
        f"CUDA compiler is unavailable: {configured}; install the CUDA toolkit or set "
        "CUDA_HOME"
    )


def _compiler_command(
    compiler: str,
    output: Path,
    *,
    architecture: int,
    debug: bool,
) -> list[str]:
    if platform.system() != "Linux":
        raise RuntimeError(
            "native CUDA build is supported only on Linux with an NVIDIA CUDA toolkit"
        )
    if architecture < MINIMUM_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "CUDA compute capability must be 60 or newer because the kernel uses "
            "double atomicAdd"
        )

    optimization = ["-O0", "-g", "-G"] if debug else ["-O3", "-DNDEBUG"]
    compute = f"compute_{architecture}"
    sm = f"sm_{architecture}"
    return [
        compiler,
        "-std=c++17",
        *optimization,
        "--shared",
        "--compiler-options=-fPIC,-Wall,-Wextra",
        f"--generate-code=arch={compute},code={sm}",
        f"--generate-code=arch={compute},code={compute}",
        str(SOURCE),
        "-lcurand",
        "-o",
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvcc",
        default=os.environ.get("NVCC", "nvcc"),
        help="CUDA compiler command (default: NVCC or nvcc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output(),
        help="output shared-library path",
    )
    parser.add_argument(
        "--architecture",
        type=int,
        default=MINIMUM_COMPUTE_CAPABILITY,
        metavar="XY",
        help="minimum CUDA compute capability without a decimal (default: 60)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="build without optimization"
    )
    args = parser.parse_args()

    if not SOURCE.is_file():
        raise RuntimeError(f"native CUDA source is missing: {SOURCE}")
    if platform.system() == "Darwin":
        raise RuntimeError(
            "CUDA is not supported on macOS; build on Linux with an NVIDIA GPU/toolkit"
        )
    compiler = _find_nvcc(args.nvcc)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="xrdmc-cuda-build-", dir=output.parent
    ) as temporary:
        temporary_output = Path(temporary) / output.name
        command = _compiler_command(
            compiler,
            temporary_output,
            architecture=args.architecture,
            debug=args.debug,
        )
        print(" ".join(command))
        subprocess.run(command, check=True)
        os.replace(temporary_output, output)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"CUDA build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
