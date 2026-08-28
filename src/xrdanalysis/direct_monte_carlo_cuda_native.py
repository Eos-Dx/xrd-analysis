"""Native CUDA direct detector Monte Carlo integration.

The optional backend uses a fused CUDA kernel. It samples detector pixels and
accumulates them through a pixel-major CSC integration plan without allocating
sampled detector frames. It never falls back to CuPy or CPU execution.
"""

from __future__ import annotations

import ctypes
import os
import platform
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import Sequence

import numpy as np

from xrdanalysis.direct_monte_carlo import NativeDirectMonteCarloPlan


_ABI_VERSION = 1
_ERROR_BUFFER_SIZE = 4096
_LIBRARY_ENV = "XRDANALYSIS_DIRECT_MONTE_CARLO_CUDA_LIBRARY"
_UINT64_MASK = (1 << 64) - 1
_MAX_IN_MEMORY_OUTPUT_BYTES = 1 << 30


class NativeCudaBackendUnavailableError(RuntimeError):
    """Raised when the native CUDA library or device is unavailable."""


class NativeCudaBackendError(RuntimeError):
    """Raised when native CUDA execution rejects a request."""


def _double_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _library_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        candidates.append(Path(configured).expanduser())
    if platform.system() == "Linux":
        candidates.append(
            Path(__file__).resolve().parent
            / "_native"
            / "libxrdanalysis_direct_monte_carlo_cuda.so"
        )
    return tuple(candidates)


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int32_pointer = ctypes.POINTER(ctypes.c_int32)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    uint64_pointer = ctypes.POINTER(ctypes.c_uint64)

    library.xrdmc_cuda_abi_version.argtypes = []
    library.xrdmc_cuda_abi_version.restype = ctypes.c_int
    library.xrdmc_cuda_device_count.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.xrdmc_cuda_device_count.restype = ctypes.c_int
    library.xrdmc_cuda_run.argtypes = [
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        uint64_pointer,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        int64_pointer,
        int32_pointer,
        double_pointer,
        ctypes.c_size_t,
        double_pointer,
        ctypes.c_size_t,
        int32_pointer,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        double_pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.xrdmc_cuda_run.restype = ctypes.c_int
    library.xrdmc_cuda_integrate.argtypes = [
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        int64_pointer,
        int32_pointer,
        double_pointer,
        ctypes.c_size_t,
        double_pointer,
        ctypes.c_size_t,
        int32_pointer,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        double_pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.xrdmc_cuda_integrate.restype = ctypes.c_int

    version = int(library.xrdmc_cuda_abi_version())
    if version != _ABI_VERSION:
        raise NativeCudaBackendUnavailableError(
            f"native CUDA ABI {version} does not match required ABI {_ABI_VERSION}"
        )
    return library


@lru_cache(maxsize=1)
def _load_native_cuda_library() -> ctypes.CDLL:
    failures: list[str] = []
    for candidate in _library_candidates():
        if not candidate.is_file():
            failures.append(f"{candidate}: file not found")
            continue
        try:
            return _configure_library(ctypes.CDLL(str(candidate)))
        except (OSError, AttributeError, NativeCudaBackendUnavailableError) as error:
            failures.append(f"{candidate}: {error}")
    detail = "; ".join(failures) if failures else "native CUDA requires Linux"
    raise NativeCudaBackendUnavailableError(
        "native CUDA direct-Monte-Carlo backend is unavailable; run "
        f"'python -m xrdanalysis._native.build_cuda' on Linux/NVIDIA ({detail})"
    )


def native_cuda_device_count() -> int:
    """Return usable CUDA device count reported by the native library."""
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    count = int(
        _load_native_cuda_library().xrdmc_cuda_device_count(
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    )
    if count < 0:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise NativeCudaBackendUnavailableError(
            f"native CUDA device query failed: {detail or 'no error detail'}"
        )
    return count


def native_cuda_backend_available() -> bool:
    """Return whether a compatible library can access a CUDA device."""
    try:
        return native_cuda_device_count() > 0
    except NativeCudaBackendUnavailableError:
        return False


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= _UINT64_MASK:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return int(seed)


def _validate_images(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
) -> np.ndarray:
    if not isinstance(plan, NativeDirectMonteCarloPlan):
        raise TypeError("plan must be a NativeDirectMonteCarloPlan")
    detector_images = np.asarray(images, dtype=np.float64)
    if detector_images.shape == plan.image_shape:
        detector_images = detector_images[np.newaxis, ...]
    if (
        detector_images.ndim != len(plan.image_shape) + 1
        or detector_images.shape[1:] != plan.image_shape
    ):
        raise ValueError(
            f"images must have shape {plan.image_shape} or "
            f"(measurements, {', '.join(str(value) for value in plan.image_shape)})"
        )
    if detector_images.shape[0] == 0:
        raise ValueError("images must contain at least one measurement")
    if not np.all(np.isfinite(detector_images)):
        raise ValueError("images must contain only finite values")
    return np.ascontiguousarray(detector_images, dtype=np.float64)


def _validate_scales(scales: Sequence[float], maximum_pixel: float) -> np.ndarray:
    noise_scales = np.asarray(scales, dtype=np.float64)
    if noise_scales.ndim != 1 or noise_scales.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(noise_scales)) or np.any(noise_scales <= 0.0):
        raise ValueError("scales must be finite and positive")
    squared = np.square(noise_scales)
    if not np.all(np.isfinite(squared)) or np.any(squared == 0.0):
        raise ValueError("squared scales must be finite and positive")
    maximum_lambda = max(0.0, maximum_pixel) / float(np.min(squared))
    if not np.isfinite(maximum_lambda) or maximum_lambda > np.iinfo(np.uint32).max:
        raise ValueError("Poisson rate exceeds the native CUDA uint32 range")
    return np.ascontiguousarray(noise_scales, dtype=np.float64)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _measurement_seed_array(count: int, values: Sequence[int] | None) -> np.ndarray:
    if values is None:
        seeds = [_splitmix64(index + 1) for index in range(count)]
    else:
        if len(values) != count:
            raise ValueError("measurement_seeds must match images")
        seeds = [_validate_seed(value) for value in values]
    return np.ascontiguousarray(seeds, dtype=np.uint64)


def _execution_controls(
    device: int,
    threads_per_block: int,
    profile_batch_size: int,
) -> tuple[int, int, int]:
    if isinstance(device, bool) or int(device) != device or device < 0:
        raise ValueError("device must be a non-negative integer")
    threads = _positive_integer(threads_per_block, "threads_per_block")
    if threads > 1024:
        raise ValueError("threads_per_block must not exceed 1024")
    batch_size = _positive_integer(profile_batch_size, "profile_batch_size")
    return int(device), threads, batch_size


def _normalization_indices(plan: NativeDirectMonteCarloPlan) -> np.ndarray:
    q_min, q_max = plan.q_normalization_band
    return np.ascontiguousarray(
        np.flatnonzero((plan.q_grid >= q_min) & (plan.q_grid <= q_max)),
        dtype=np.int32,
    )


def _plans_equivalent(
    left: NativeDirectMonteCarloPlan,
    right: NativeDirectMonteCarloPlan,
) -> bool:
    return (
        left.image_shape == right.image_shape
        and left.q_normalization_band == right.q_normalization_band
        and np.array_equal(left.csc_indptr, right.csc_indptr)
        and np.array_equal(left.csc_indices, right.csc_indices)
        and np.array_equal(left.csc_weights, right.csc_weights)
        and np.array_equal(
            left.normalization_denominators,
            right.normalization_denominators,
        )
        and np.array_equal(left.q_grid, right.q_grid)
    )


def _prepare_output(
    shape: tuple[int, ...],
    output: np.ndarray | None,
) -> np.ndarray:
    value_count = prod(shape)
    required_bytes = value_count * np.dtype(np.float64).itemsize
    if output is None:
        if required_bytes > _MAX_IN_MEMORY_OUTPUT_BYTES:
            gib = required_bytes / (1 << 30)
            raise ValueError(
                f"CUDA output requires {gib:.2f} GiB; process patient-sized batches "
                "or provide a writable C-contiguous float64 output such as np.memmap"
            )
        return np.empty(shape, dtype=np.float64, order="C")

    result = np.asarray(output)
    if result.shape != shape:
        raise ValueError(f"output must have shape {shape}")
    if result.dtype != np.float64:
        raise ValueError("output must use float64 dtype")
    if not result.flags.c_contiguous or not result.flags.writeable:
        raise ValueError("output must be writable and C-contiguous")
    return result


def direct_detector_monte_carlo_cuda_native(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    measurement_seeds: Sequence[int] | None = None,
    device: int = 0,
    threads_per_block: int = 256,
    profile_batch_size: int = 4096,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Run fused direct detector Monte Carlo with the native CUDA kernel.

    Output axes are ``(scale, draw, measurement, radial_bin)``. Random samples
    are independent of ``profile_batch_size``. No CPU or CuPy fallback exists.
    """
    detector_images = _validate_images(plan, images)
    noise_scales = _validate_scales(scales, float(np.max(detector_images)))
    draw_count = _positive_integer(draws, "draws")
    validated_seed = _validate_seed(seed)
    validated_device, threads, batch_size = _execution_controls(
        device,
        threads_per_block,
        profile_batch_size,
    )

    measurements = int(detector_images.shape[0])
    stable_seeds = _measurement_seed_array(measurements, measurement_seeds)
    normalization_indices = _normalization_indices(plan)
    result = _prepare_output(
        (int(noise_scales.size), draw_count, measurements, plan.bins),
        output,
    )

    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    library = _load_native_cuda_library()
    status = library.xrdmc_cuda_run(
        _double_pointer(detector_images),
        ctypes.c_size_t(measurements),
        ctypes.c_size_t(plan.pixels),
        stable_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        _double_pointer(noise_scales),
        ctypes.c_size_t(noise_scales.size),
        ctypes.c_size_t(draw_count),
        plan.csc_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        plan.csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _double_pointer(plan.csc_weights),
        ctypes.c_size_t(plan.csc_weights.size),
        _double_pointer(plan.normalization_denominators),
        ctypes.c_size_t(plan.bins),
        normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(normalization_indices.size),
        ctypes.c_uint64(validated_seed),
        ctypes.c_int(validated_device),
        ctypes.c_int(threads),
        ctypes.c_size_t(batch_size),
        _double_pointer(result),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise NativeCudaBackendError(
            f"native CUDA execution failed with status {status}: "
            f"{detail or 'no error detail'}"
        )
    return result


def integrate_detector_frames_cuda_native(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
    *,
    device: int = 0,
    threads_per_block: int = 256,
    profile_batch_size: int = 4096,
) -> np.ndarray:
    """Integrate fixed detector frames on CUDA for deterministic parity tests."""
    detector_images = _validate_images(plan, images)
    validated_device, threads, batch_size = _execution_controls(
        device,
        threads_per_block,
        profile_batch_size,
    )
    measurements = int(detector_images.shape[0])
    normalization_indices = _normalization_indices(plan)
    output = np.empty((measurements, plan.bins), dtype=np.float64, order="C")
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    status = _load_native_cuda_library().xrdmc_cuda_integrate(
        _double_pointer(detector_images),
        ctypes.c_size_t(measurements),
        ctypes.c_size_t(plan.pixels),
        plan.csc_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        plan.csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _double_pointer(plan.csc_weights),
        ctypes.c_size_t(plan.csc_weights.size),
        _double_pointer(plan.normalization_denominators),
        ctypes.c_size_t(plan.bins),
        normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(normalization_indices.size),
        ctypes.c_int(validated_device),
        ctypes.c_int(threads),
        ctypes.c_size_t(batch_size),
        _double_pointer(output),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise NativeCudaBackendError(
            f"native CUDA integration failed with status {status}: "
            f"{detail or 'no error detail'}"
        )
    return output


def direct_detector_monte_carlo_cuda_native_measurements(
    plans: Sequence[NativeDirectMonteCarloPlan],
    images: Sequence[np.ndarray],
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    device: int = 0,
    threads_per_block: int = 256,
    profile_batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Run distinct per-measurement plans on one common q grid."""
    if len(plans) == 0 or len(plans) != len(images):
        raise ValueError("plans and images must have equal non-zero length")
    if any(not isinstance(plan, NativeDirectMonteCarloPlan) for plan in plans):
        raise TypeError("every plan must be a NativeDirectMonteCarloPlan")
    reference_q = plans[0].q_grid
    for plan in plans[1:]:
        if plan.q_grid.shape != reference_q.shape or not np.allclose(
            plan.q_grid,
            reference_q,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("all measurement plans must use the same q grid")

    noise_scales = tuple(scales)
    draw_count = _positive_integer(draws, "draws")
    validated_seed = _validate_seed(seed)
    detector_images = [
        _validate_images(plan, image) for plan, image in zip(plans, images, strict=True)
    ]
    if any(image.shape[0] != 1 for image in detector_images):
        raise ValueError("each image entry must contain exactly one measurement")
    if all(_plans_equivalent(plans[0], plan) for plan in plans[1:]):
        combined = np.concatenate(detector_images, axis=0)
        profiles = direct_detector_monte_carlo_cuda_native(
            plans[0],
            combined,
            noise_scales,
            draw_count,
            seed=validated_seed,
            measurement_seeds=tuple(
                _splitmix64(index + 1) for index in range(len(plans))
            ),
            device=device,
            threads_per_block=threads_per_block,
            profile_batch_size=profile_batch_size,
        )
        return reference_q.copy(), profiles

    profiles = np.empty(
        (len(noise_scales), draw_count, len(plans), plans[0].bins),
        dtype=np.float64,
    )
    for index, (plan, image) in enumerate(zip(plans, detector_images, strict=True)):
        profiles[:, :, index] = direct_detector_monte_carlo_cuda_native(
            plan,
            image,
            noise_scales,
            draw_count,
            seed=validated_seed,
            measurement_seeds=(_splitmix64(index + 1),),
            device=device,
            threads_per_block=threads_per_block,
            profile_batch_size=profile_batch_size,
        )[:, :, 0]
    return reference_q.copy(), profiles


__all__ = [
    "NativeCudaBackendError",
    "NativeCudaBackendUnavailableError",
    "direct_detector_monte_carlo_cuda_native",
    "direct_detector_monte_carlo_cuda_native_measurements",
    "integrate_detector_frames_cuda_native",
    "native_cuda_backend_available",
    "native_cuda_device_count",
]
