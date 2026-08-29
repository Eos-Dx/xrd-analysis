"""Native Apple Metal direct detector Monte Carlo integration.

The optional backend uses a deterministic bin-major CSR kernel. Detector
pixels are sampled directly and no sampled detector frame or covariance
approximation is created. The API never falls back to CPU execution.
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csc_matrix

from xrdanalysis.direct_monte_carlo import NativeDirectMonteCarloPlan


_ABI_VERSION = 4
_ERROR_BUFFER_SIZE = 4096
_LIBRARY_ENV = "XRDANALYSIS_DIRECT_MONTE_CARLO_METAL_LIBRARY"
_SOURCE_ENV = "XRDANALYSIS_DIRECT_MONTE_CARLO_METAL_SOURCE"
_UINT64_MASK = (1 << 64) - 1
_MAX_IN_MEMORY_OUTPUT_BYTES = 1 << 30
_MAX_EXACT_FLOAT32_POISSON_RATE = 1 << 24


class MetalBackendUnavailableError(RuntimeError):
    """Raised when the native Metal library or Apple GPU is unavailable."""


class MetalBackendError(RuntimeError):
    """Raised when native Metal execution rejects a request."""


def _readonly_contiguous(values: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class NativeMetalMonteCarloPlan:
    """Immutable bin-major Metal integration plan."""

    image_shape: tuple[int, ...]
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    csr_weights: np.ndarray
    normalization_denominators: np.ndarray
    q_grid: np.ndarray
    q_normalization_band: tuple[float, float]

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.image_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("image_shape must contain positive dimensions")
        pixels = prod(shape)
        indptr = _readonly_contiguous(self.csr_indptr, np.dtype(np.int64))
        indices = _readonly_contiguous(self.csr_indices, np.dtype(np.int32))
        weights = _readonly_contiguous(self.csr_weights, np.dtype(np.float64))
        denominators = _readonly_contiguous(
            self.normalization_denominators,
            np.dtype(np.float64),
        )
        q_grid = _readonly_contiguous(self.q_grid, np.dtype(np.float64))
        if any(
            values.ndim != 1
            for values in (indptr, indices, weights, denominators, q_grid)
        ):
            raise ValueError("Metal plan arrays must be one-dimensional")
        if q_grid.size == 0 or denominators.shape != q_grid.shape:
            raise ValueError("normalization_denominators must match non-empty q_grid")
        if indptr.size != q_grid.size + 1:
            raise ValueError("CSR indptr length must equal radial bins plus one")
        if indices.size != weights.size:
            raise ValueError("CSR indices and weights must have equal length")
        if indptr[0] != 0 or indptr[-1] != weights.size or np.any(np.diff(indptr) < 0):
            raise ValueError("CSR indptr endpoints or ordering are invalid")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= pixels):
            raise ValueError("CSR detector-pixel index is out of range")
        if not np.all(np.isfinite(weights)):
            raise ValueError("CSR weights must be finite")
        if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
            raise ValueError("normalization_denominators must be finite and positive")
        if not np.all(np.isfinite(q_grid)) or np.any(np.diff(q_grid) <= 0.0):
            raise ValueError("q_grid must be finite and strictly increasing")
        if len(self.q_normalization_band) != 2:
            raise ValueError("q_normalization_band must contain two values")
        q_min, q_max = (float(value) for value in self.q_normalization_band)
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min > q_max:
            raise ValueError("q_normalization_band must be finite and ordered")
        if not np.any((q_grid >= q_min) & (q_grid <= q_max)):
            raise ValueError("q_normalization_band contains no q-grid bins")

        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "csr_indptr", indptr)
        object.__setattr__(self, "csr_indices", indices)
        object.__setattr__(self, "csr_weights", weights)
        object.__setattr__(self, "normalization_denominators", denominators)
        object.__setattr__(self, "q_grid", q_grid)
        object.__setattr__(self, "q_normalization_band", (q_min, q_max))

    @property
    def pixels(self) -> int:
        """Number of detector pixels expected by the plan."""
        return prod(self.image_shape)

    @property
    def bins(self) -> int:
        """Number of integrated radial bins."""
        return int(self.q_grid.size)

    def run(
        self,
        images: np.ndarray,
        scales: Sequence[float],
        draws: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run direct detector Monte Carlo on the Apple GPU."""
        return direct_detector_monte_carlo_metal(
            self,
            images,
            scales,
            draws,
            **kwargs,
        )


def prepare_metal_plan(
    plan: NativeDirectMonteCarloPlan,
) -> NativeMetalMonteCarloPlan:
    """Convert a validated pixel-major native plan to bin-major CSR."""
    if not isinstance(plan, NativeDirectMonteCarloPlan):
        raise TypeError("plan must be a NativeDirectMonteCarloPlan")
    csc = csc_matrix(
        (plan.csc_weights, plan.csc_indices, plan.csc_indptr),
        shape=(plan.bins, plan.pixels),
        dtype=np.float64,
    )
    csr = csc.tocsr(copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    return NativeMetalMonteCarloPlan(
        image_shape=plan.image_shape,
        csr_indptr=csr.indptr,
        csr_indices=csr.indices,
        csr_weights=csr.data,
        normalization_denominators=plan.normalization_denominators,
        q_grid=plan.q_grid,
        q_normalization_band=plan.q_normalization_band,
    )


def _library_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        candidates.append(Path(configured).expanduser())
    if platform.system() == "Darwin":
        candidates.append(
            Path(__file__).resolve().parent
            / "_native"
            / "libxrdanalysis_direct_monte_carlo_metal.dylib"
        )
    return tuple(candidates)


def _metal_source_path() -> Path:
    configured = os.environ.get(_SOURCE_ENV)
    source = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parent
        / "_native"
        / "direct_monte_carlo_metal.metal"
    )
    if not source.is_file():
        raise MetalBackendUnavailableError(
            f"Metal shader source is unavailable: {source}"
        )
    return source.resolve()


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    char_pointer = ctypes.c_char_p
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int32_pointer = ctypes.POINTER(ctypes.c_int32)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
    uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
    error_pointer = ctypes.POINTER(ctypes.c_char)

    library.xrdmc_metal_abi_version.argtypes = []
    library.xrdmc_metal_abi_version.restype = ctypes.c_int
    library.xrdmc_metal_device_count.argtypes = [error_pointer, ctypes.c_size_t]
    library.xrdmc_metal_device_count.restype = ctypes.c_int
    library.xrdmc_metal_session_create.argtypes = [
        char_pointer,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        uint64_pointer,
        int64_pointer,
        int32_pointer,
        double_pointer,
        ctypes.c_size_t,
        double_pointer,
        ctypes.c_size_t,
        int32_pointer,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_session_create.restype = ctypes.c_void_p
    library.xrdmc_metal_multi_session_create.argtypes = [
        char_pointer,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        uint64_pointer,
        ctypes.c_size_t,
        int32_pointer,
        int64_pointer,
        int32_pointer,
        double_pointer,
        ctypes.c_size_t,
        double_pointer,
        ctypes.c_size_t,
        int32_pointer,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_multi_session_create.restype = ctypes.c_void_p
    library.xrdmc_metal_session_destroy.argtypes = [ctypes.c_void_p]
    library.xrdmc_metal_session_destroy.restype = None
    library.xrdmc_metal_session_run.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_session_run.restype = ctypes.c_int
    library.xrdmc_metal_session_integrate.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_session_integrate.restype = ctypes.c_int
    library.xrdmc_metal_run.argtypes = [
        char_pointer,
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
        ctypes.c_size_t,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_run.restype = ctypes.c_int
    library.xrdmc_metal_integrate.argtypes = [
        char_pointer,
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
        ctypes.c_size_t,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_integrate.restype = ctypes.c_int
    library.xrdmc_metal_geometry_session_create.argtypes = [
        char_pointer,
        double_pointer,
        uint8_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        uint64_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        int32_pointer,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        int32_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_geometry_session_create.restype = ctypes.c_void_p
    library.xrdmc_metal_geometry_session_destroy.argtypes = [ctypes.c_void_p]
    library.xrdmc_metal_geometry_session_destroy.restype = None
    library.xrdmc_metal_geometry_session_run.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        double_pointer,
        double_pointer,
        double_pointer,
        ctypes.c_uint64,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_geometry_session_run.restype = ctypes.c_int
    library.xrdmc_metal_geometry_session_run_nested.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        double_pointer,
        double_pointer,
        double_pointer,
        ctypes.c_uint64,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_geometry_session_run_nested.restype = ctypes.c_int
    library.xrdmc_metal_geometry_session_integrate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        error_pointer,
        ctypes.c_size_t,
    ]
    library.xrdmc_metal_geometry_session_integrate.restype = ctypes.c_int
    version = int(library.xrdmc_metal_abi_version())
    if version != _ABI_VERSION:
        raise MetalBackendUnavailableError(
            f"native Metal ABI {version} does not match required ABI {_ABI_VERSION}"
        )
    return library


@lru_cache(maxsize=1)
def _load_metal_library() -> ctypes.CDLL:
    failures: list[str] = []
    for candidate in _library_candidates():
        if not candidate.is_file():
            failures.append(f"{candidate}: file not found")
            continue
        try:
            return _configure_library(ctypes.CDLL(str(candidate)))
        except (OSError, AttributeError, MetalBackendUnavailableError) as error:
            failures.append(f"{candidate}: {error}")
    detail = "; ".join(failures) if failures else "Metal requires macOS"
    raise MetalBackendUnavailableError(
        "native Metal backend is unavailable; run "
        f"'python -m xrdanalysis._native.build_metal' ({detail})"
    )


def metal_device_count() -> int:
    """Return Metal devices reported by the native runtime."""
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    count = int(
        _load_metal_library().xrdmc_metal_device_count(
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    )
    if count < 0:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise MetalBackendUnavailableError(
            f"Metal device query failed: {detail or 'no error detail'}"
        )
    return count


def metal_backend_available() -> bool:
    """Return whether a compatible runtime can access an Apple GPU."""
    try:
        _metal_source_path()
        return metal_device_count() > 0
    except MetalBackendUnavailableError:
        return False


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= _UINT64_MASK:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return int(seed)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _validate_images(plan: NativeMetalMonteCarloPlan, images: np.ndarray) -> np.ndarray:
    if not isinstance(plan, NativeMetalMonteCarloPlan):
        raise TypeError("plan must be a NativeMetalMonteCarloPlan")
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
    if np.max(np.abs(detector_images)) > np.finfo(np.float32).max:
        raise ValueError("images exceed Metal float32 range")
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
    maximum_rate = max(0.0, maximum_pixel) / float(np.min(squared))
    if not np.isfinite(maximum_rate) or maximum_rate > _MAX_EXACT_FLOAT32_POISSON_RATE:
        raise ValueError("Poisson rate exceeds the exact Metal float32 integer range")
    return np.ascontiguousarray(noise_scales, dtype=np.float64)


def _measurement_seeds(count: int, values: Sequence[int] | None) -> np.ndarray:
    seeds = (
        [_splitmix64(index + 1) for index in range(count)]
        if values is None
        else [_validate_seed(value) for value in values]
    )
    if len(seeds) != count:
        raise ValueError("measurement_seeds must match images")
    return np.ascontiguousarray(seeds, dtype=np.uint64)


def _normalization_indices(plan: NativeMetalMonteCarloPlan) -> np.ndarray:
    q_min, q_max = plan.q_normalization_band
    return np.ascontiguousarray(
        np.flatnonzero((plan.q_grid >= q_min) & (plan.q_grid <= q_max)),
        dtype=np.int32,
    )


def _prepare_output(
    shape: tuple[int, ...],
    output: np.ndarray | None,
) -> np.ndarray:
    required_bytes = prod(shape) * np.dtype(np.float64).itemsize
    if output is None:
        if required_bytes > _MAX_IN_MEMORY_OUTPUT_BYTES:
            gib = required_bytes / (1 << 30)
            raise ValueError(
                f"Metal output requires {gib:.2f} GiB; process patient-sized batches "
                "or provide a writable C-contiguous float64 output such as np.memmap"
            )
        return np.empty(shape, dtype=np.float64, order="C")
    result = output if isinstance(output, np.ndarray) else np.asarray(output)
    if result.shape != shape:
        raise ValueError(f"output must have shape {shape}")
    if result.dtype != np.float64:
        raise ValueError("output must use float64 dtype")
    if not result.flags.c_contiguous or not result.flags.writeable:
        raise ValueError("output must be writable and C-contiguous")
    return result


def _double_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def direct_detector_monte_carlo_metal(
    plan: NativeMetalMonteCarloPlan,
    images: np.ndarray,
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    measurement_seeds: Sequence[int] | None = None,
    device: int = 0,
    profile_batch_size: int = 4096,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Run deterministic-stream direct detector Monte Carlo on Metal."""
    detector_images = _validate_images(plan, images)
    noise_scales = _validate_scales(scales, float(np.max(detector_images)))
    draw_count = _positive_integer(draws, "draws")
    validated_seed = _validate_seed(seed)
    if isinstance(device, bool) or int(device) != device or device < 0:
        raise ValueError("device must be a non-negative integer")
    batch_size = _positive_integer(profile_batch_size, "profile_batch_size")
    measurements = int(detector_images.shape[0])
    stable_seeds = _measurement_seeds(measurements, measurement_seeds)
    normalization_indices = _normalization_indices(plan)
    result = _prepare_output(
        (int(noise_scales.size), draw_count, measurements, plan.bins),
        output,
    )
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    status = _load_metal_library().xrdmc_metal_run(
        os.fsencode(_metal_source_path()),
        _double_pointer(detector_images),
        ctypes.c_size_t(measurements),
        ctypes.c_size_t(plan.pixels),
        stable_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        _double_pointer(noise_scales),
        ctypes.c_size_t(noise_scales.size),
        ctypes.c_size_t(draw_count),
        plan.csr_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        plan.csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _double_pointer(plan.csr_weights),
        ctypes.c_size_t(plan.csr_weights.size),
        _double_pointer(plan.normalization_denominators),
        ctypes.c_size_t(plan.bins),
        normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(normalization_indices.size),
        ctypes.c_uint64(validated_seed),
        ctypes.c_int(device),
        ctypes.c_size_t(batch_size),
        _double_pointer(result),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise MetalBackendError(
            f"native Metal execution failed with status {status}: "
            f"{detail or 'no error detail'}"
        )
    return result


def integrate_detector_frames_metal(
    plan: NativeMetalMonteCarloPlan,
    images: np.ndarray,
    *,
    device: int = 0,
    profile_batch_size: int = 4096,
) -> np.ndarray:
    """Integrate fixed detector frames on Metal for numerical parity tests."""
    detector_images = _validate_images(plan, images)
    if isinstance(device, bool) or int(device) != device or device < 0:
        raise ValueError("device must be a non-negative integer")
    batch_size = _positive_integer(profile_batch_size, "profile_batch_size")
    measurements = int(detector_images.shape[0])
    normalization_indices = _normalization_indices(plan)
    output = np.empty((measurements, plan.bins), dtype=np.float64, order="C")
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    status = _load_metal_library().xrdmc_metal_integrate(
        os.fsencode(_metal_source_path()),
        _double_pointer(detector_images),
        ctypes.c_size_t(measurements),
        ctypes.c_size_t(plan.pixels),
        plan.csr_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        plan.csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _double_pointer(plan.csr_weights),
        ctypes.c_size_t(plan.csr_weights.size),
        _double_pointer(plan.normalization_denominators),
        ctypes.c_size_t(plan.bins),
        normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(normalization_indices.size),
        ctypes.c_int(device),
        ctypes.c_size_t(batch_size),
        _double_pointer(output),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        detail = error_buffer.value.decode("utf-8", errors="replace")
        raise MetalBackendError(
            f"native Metal integration failed with status {status}: "
            f"{detail or 'no error detail'}"
        )
    return output


__all__ = [
    "MetalBackendError",
    "MetalBackendUnavailableError",
    "NativeMetalMonteCarloPlan",
    "direct_detector_monte_carlo_metal",
    "integrate_detector_frames_metal",
    "metal_backend_available",
    "metal_device_count",
    "prepare_metal_plan",
]
