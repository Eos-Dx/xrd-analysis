"""Optional native direct detector Monte Carlo integration.

The backend samples detector pixels directly and accumulates sampled values
through a warmed pyFAI bbox/CSR lookup table. It does not construct sampled
detector frames and does not implement a covariance approximation.
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix


_ABI_VERSION = 1
_ERROR_BUFFER_SIZE = 4096
_LIBRARY_ENV = "XRDANALYSIS_DIRECT_MONTE_CARLO_LIBRARY"


class NativeBackendUnavailableError(RuntimeError):
    """Raised when the optional native direct-Monte-Carlo library is unavailable."""


class NativeBackendError(RuntimeError):
    """Raised when the native direct-Monte-Carlo library rejects a request."""


def _readonly_contiguous(values: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class NativeDirectMonteCarloPlan:
    """Immutable pixel-major integration plan for the native backend."""

    image_shape: tuple[int, ...]
    csc_indptr: np.ndarray
    csc_indices: np.ndarray
    csc_weights: np.ndarray
    normalization_denominators: np.ndarray
    q_grid: np.ndarray
    q_normalization_band: tuple[float, float]

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.image_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("image_shape must contain positive dimensions")
        pixels = int(np.prod(shape, dtype=np.int64))
        if pixels <= 0:
            raise ValueError("image_shape pixel count must be positive")

        indptr = _readonly_contiguous(self.csc_indptr, np.dtype(np.int64))
        indices = _readonly_contiguous(self.csc_indices, np.dtype(np.int32))
        weights = _readonly_contiguous(self.csc_weights, np.dtype(np.float64))
        denominators = _readonly_contiguous(
            self.normalization_denominators, np.dtype(np.float64)
        )
        q_grid = _readonly_contiguous(self.q_grid, np.dtype(np.float64))
        if any(
            array.ndim != 1
            for array in (indptr, indices, weights, denominators, q_grid)
        ):
            raise ValueError("native plan arrays must be one-dimensional")
        if q_grid.size == 0 or denominators.shape != q_grid.shape:
            raise ValueError("normalization_denominators must match non-empty q_grid")
        if indptr.size != pixels + 1:
            raise ValueError("CSC indptr length must equal detector pixels plus one")
        if indices.size != weights.size:
            raise ValueError("CSC indices and weights must have equal length")
        if indptr[0] != 0 or indptr[-1] != weights.size or np.any(np.diff(indptr) < 0):
            raise ValueError("CSC indptr endpoints or ordering are invalid")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= q_grid.size):
            raise ValueError("CSC bin index out of range")
        if not np.all(np.isfinite(weights)):
            raise ValueError("CSC weights must be finite")
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
        object.__setattr__(self, "csc_indptr", indptr)
        object.__setattr__(self, "csc_indices", indices)
        object.__setattr__(self, "csc_weights", weights)
        object.__setattr__(self, "normalization_denominators", denominators)
        object.__setattr__(self, "q_grid", q_grid)
        object.__setattr__(self, "q_normalization_band", (q_min, q_max))

    @property
    def pixels(self) -> int:
        """Number of detector pixels expected by the plan."""
        return int(np.prod(self.image_shape, dtype=np.int64))

    @property
    def bins(self) -> int:
        """Number of integrated radial bins produced by the plan."""
        return int(self.q_grid.size)

    def run(
        self,
        images: np.ndarray,
        scales: Sequence[float],
        draws: int,
        *,
        seed: int = 0,
        threads: int | None = None,
        backend: str = "native",
    ) -> np.ndarray:
        """Run direct detector Monte Carlo and return normalized profiles."""
        return direct_detector_monte_carlo(
            self,
            images,
            scales,
            draws,
            seed=seed,
            threads=threads,
            backend=backend,
        )


def _library_names() -> tuple[str, ...]:
    system = platform.system()
    if system == "Darwin":
        return (
            "libxrdanalysis_direct_monte_carlo.dylib",
            "libxrdanalysis_direct_monte_carlo.so",
        )
    if system == "Linux":
        return ("libxrdanalysis_direct_monte_carlo.so",)
    return ()


def _library_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        candidates.append(Path(configured).expanduser())
    native_dir = Path(__file__).resolve().parent / "_native"
    candidates.extend(native_dir / name for name in _library_names())
    return tuple(candidates)


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int32_pointer = ctypes.POINTER(ctypes.c_int32)
    int64_pointer = ctypes.POINTER(ctypes.c_int64)

    library.xrdmc_abi_version.argtypes = []
    library.xrdmc_abi_version.restype = ctypes.c_int
    library.xrdmc_run.argtypes = [
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        int64_pointer,
        int32_pointer,
        double_pointer,
        ctypes.c_size_t,
        double_pointer,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_int,
        double_pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.xrdmc_run.restype = ctypes.c_int
    library.xrdmc_integrate.argtypes = [
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        int64_pointer,
        int32_pointer,
        double_pointer,
        ctypes.c_size_t,
        double_pointer,
        double_pointer,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        double_pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.xrdmc_integrate.restype = ctypes.c_int

    version = int(library.xrdmc_abi_version())
    if version != _ABI_VERSION:
        raise NativeBackendUnavailableError(
            f"native direct-Monte-Carlo ABI {version} does not match required "
            f"ABI {_ABI_VERSION}"
        )
    return library


@lru_cache(maxsize=1)
def _load_native_library() -> ctypes.CDLL:
    failures: list[str] = []
    for candidate in _library_candidates():
        if not candidate.is_file():
            failures.append(f"{candidate}: file not found")
            continue
        try:
            return _configure_library(ctypes.CDLL(str(candidate)))
        except (OSError, AttributeError, NativeBackendUnavailableError) as error:
            failures.append(f"{candidate}: {error}")
    detail = "; ".join(failures) if failures else "unsupported operating system"
    raise NativeBackendUnavailableError(
        "native direct-Monte-Carlo backend is unavailable; run "
        f"'python -m xrdanalysis._native.build' ({detail})"
    )


def native_backend_available() -> bool:
    """Return whether a compatible native backend can be loaded."""
    try:
        _load_native_library()
    except NativeBackendUnavailableError:
        return False
    return True


def default_native_threads() -> int:
    """Reserve two logical CPUs and cap native execution at 12 threads."""
    logical_cpus = os.cpu_count() or 1
    return max(1, min(12, logical_cpus - 2))


def _unwrap_engine(candidate: Any) -> Any:
    engine = getattr(candidate, "engine", None)
    return engine if engine is not None else candidate


def _is_bbox_csr_method(method: Any) -> bool:
    split = str(getattr(method, "split", "")).lower()
    algorithm = str(getattr(method, "algo", "")).lower()
    return split == "bbox" and algorithm == "csr"


def _has_csr_arrays(engine: Any) -> bool:
    return all(hasattr(engine, name) for name in ("data", "indices", "indptr"))


def _select_warmed_bbox_csr_engine(
    integrator: Any,
    *,
    engine: Any | None,
    pixels: int,
    bins: int,
) -> Any:
    if engine is not None:
        selected = _unwrap_engine(engine)
        module_name = type(selected).__module__.lower()
        if (
            not _has_csr_arrays(selected)
            or "bbox" not in module_name
            or "csr" not in module_name
        ):
            raise ValueError("engine must be a warmed pyFAI bbox/CSR engine")
        candidates = [selected]
    else:
        engines = getattr(integrator, "engines", None)
        if not engines:
            raise ValueError(
                "integrator has no warmed engines; run pyFAI integrate1d with "
                "method=('bbox', 'csr', 'cython') first"
            )
        candidates = [
            _unwrap_engine(wrapper)
            for method, wrapper in engines.items()
            if _is_bbox_csr_method(method) and _has_csr_arrays(_unwrap_engine(wrapper))
        ]

    matching: list[Any] = []
    for candidate in candidates:
        indptr = np.asarray(candidate.indptr)
        indices = np.asarray(candidate.indices)
        candidate_bins = indptr.size - 1 if indptr.ndim == 1 else -1
        indices_fit = indices.size == 0 or (
            np.min(indices) >= 0 and np.max(indices) < pixels
        )
        if candidate_bins == bins and indices_fit:
            matching.append(candidate)

    if not matching:
        raise ValueError(
            "no warmed bbox/CSR engine matches image size and q-grid bin count"
        )
    if len(matching) > 1:
        raise ValueError(
            "multiple warmed bbox/CSR engines match; pass engine explicitly"
        )
    return matching[0]


def prepare_native_plan(
    integrator: Any,
    image_shape: Sequence[int],
    *,
    normalization_denominators: Sequence[float],
    q_grid: Sequence[float],
    q_normalization_band: tuple[float, float],
    engine: Any | None = None,
) -> NativeDirectMonteCarloPlan:
    """Convert a warmed pyFAI bbox/CSR LUT into a pixel-major native plan.

    ``normalization_denominators`` should be the fixed ``sum_normalization``
    returned by the warmed pyFAI integration configured with the corrections
    used for production profiles.
    """
    shape = tuple(int(value) for value in image_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("image_shape must contain positive dimensions")
    pixels = int(np.prod(shape, dtype=np.int64))
    if pixels <= 0:
        raise ValueError("image_shape pixel count must be positive")

    q_values = np.asarray(q_grid, dtype=np.float64)
    denominators = np.asarray(normalization_denominators, dtype=np.float64)
    if q_values.ndim != 1 or q_values.size == 0:
        raise ValueError("q_grid must be a non-empty one-dimensional array")
    if denominators.shape != q_values.shape:
        raise ValueError("normalization_denominators must match q_grid shape")
    if not np.all(np.isfinite(q_values)) or np.any(np.diff(q_values) <= 0.0):
        raise ValueError("q_grid must be finite and strictly increasing")
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        raise ValueError("normalization_denominators must be finite and positive")

    if len(q_normalization_band) != 2:
        raise ValueError("q_normalization_band must contain two values")
    q_min, q_max = (float(value) for value in q_normalization_band)
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min > q_max:
        raise ValueError("q_normalization_band must be finite and ordered")
    if not np.any((q_values >= q_min) & (q_values <= q_max)):
        raise ValueError("q_normalization_band contains no q-grid bins")

    selected = _select_warmed_bbox_csr_engine(
        integrator,
        engine=engine,
        pixels=pixels,
        bins=int(q_values.size),
    )
    csr_data = np.asarray(selected.data, dtype=np.float64)
    csr_indices = np.asarray(selected.indices)
    csr_indptr = np.asarray(selected.indptr)
    if csr_data.ndim != 1 or csr_indices.ndim != 1 or csr_indptr.ndim != 1:
        raise ValueError("pyFAI CSR LUT arrays must be one-dimensional")
    if csr_data.size != csr_indices.size or csr_indptr.size != q_values.size + 1:
        raise ValueError("pyFAI CSR LUT arrays have inconsistent dimensions")
    if not np.all(np.isfinite(csr_data)):
        raise ValueError("pyFAI CSR LUT weights must be finite")

    csr = csr_matrix(
        (csr_data, csr_indices, csr_indptr),
        shape=(int(q_values.size), pixels),
        dtype=np.float64,
    )
    csr.sum_duplicates()
    csc = csr.tocsc(copy=True)
    csc.sum_duplicates()
    csc.sort_indices()

    return NativeDirectMonteCarloPlan(
        image_shape=shape,
        csc_indptr=_readonly_contiguous(csc.indptr, np.dtype(np.int64)),
        csc_indices=_readonly_contiguous(csc.indices, np.dtype(np.int32)),
        csc_weights=_readonly_contiguous(csc.data, np.dtype(np.float64)),
        normalization_denominators=_readonly_contiguous(
            denominators,
            np.dtype(np.float64),
        ),
        q_grid=_readonly_contiguous(q_values, np.dtype(np.float64)),
        q_normalization_band=(q_min, q_max),
    )


def _double_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def direct_detector_monte_carlo(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    threads: int | None = None,
    backend: str = "native",
) -> np.ndarray:
    """Run fused pixel-level centered-Poisson Monte Carlo integration.

    Output axes are ``(scale, draw, measurement, radial_bin)``. Requesting the
    native backend never falls back to a Python or covariance implementation.
    """
    if backend != "native":
        raise ValueError("backend must be 'native'")
    if not isinstance(plan, NativeDirectMonteCarloPlan):
        raise TypeError("plan must be a NativeDirectMonteCarloPlan")
    if isinstance(draws, bool) or int(draws) != draws or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0 or seed > 2**64 - 1:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if threads is None:
        threads = default_native_threads()
    if isinstance(threads, bool) or int(threads) != threads or not 1 <= threads <= 1024:
        raise ValueError("threads must be an integer in [1, 1024]")

    detector_images = np.asarray(images, dtype=np.float64)
    if detector_images.shape == plan.image_shape:
        detector_images = detector_images[np.newaxis, ...]
    expected_ndim = len(plan.image_shape) + 1
    if (
        detector_images.ndim != expected_ndim
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
    detector_images = np.ascontiguousarray(detector_images, dtype=np.float64)

    noise_scales = np.asarray(scales, dtype=np.float64)
    if noise_scales.ndim != 1 or noise_scales.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(noise_scales)) or np.any(noise_scales <= 0.0):
        raise ValueError("scales must be finite and positive")
    noise_scales = np.ascontiguousarray(noise_scales, dtype=np.float64)

    measurements = int(detector_images.shape[0])
    output = np.empty(
        (int(noise_scales.size), int(draws), measurements, plan.bins),
        dtype=np.float64,
        order="C",
    )
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    library = _load_native_library()
    status = library.xrdmc_run(
        _double_pointer(detector_images),
        ctypes.c_size_t(measurements),
        ctypes.c_size_t(plan.pixels),
        _double_pointer(noise_scales),
        ctypes.c_size_t(noise_scales.size),
        ctypes.c_size_t(draws),
        plan.csc_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        plan.csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _double_pointer(plan.csc_weights),
        ctypes.c_size_t(plan.csc_weights.size),
        _double_pointer(plan.normalization_denominators),
        _double_pointer(plan.q_grid),
        ctypes.c_size_t(plan.bins),
        ctypes.c_double(plan.q_normalization_band[0]),
        ctypes.c_double(plan.q_normalization_band[1]),
        ctypes.c_uint64(seed),
        ctypes.c_int(threads),
        _double_pointer(output),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise NativeBackendError(
            f"native direct-Monte-Carlo execution failed with status {status}: "
            f"{message or 'no error detail'}"
        )
    return output


def direct_detector_monte_carlo_measurements(
    plans: Sequence[NativeDirectMonteCarloPlan],
    images: Sequence[np.ndarray],
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run measurements with distinct masks or geometries on one common q grid."""
    if len(plans) == 0 or len(plans) != len(images):
        raise ValueError("plans and images must have equal non-zero length")
    reference_q = plans[0].q_grid
    for plan in plans[1:]:
        if plan.q_grid.shape != reference_q.shape or not np.allclose(
            plan.q_grid,
            reference_q,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("all measurement plans must use the same q grid")

    profiles = np.empty(
        (len(scales), int(draws), len(plans), plans[0].bins),
        dtype=np.float64,
    )
    mask = (1 << 64) - 1
    for measurement_index, (plan, image) in enumerate(zip(plans, images, strict=True)):
        measurement_seed = (
            int(seed) + 0x9E3779B97F4A7C15 * (measurement_index + 1)
        ) & mask
        profiles[:, :, measurement_index] = plan.run(
            image,
            scales,
            draws,
            seed=measurement_seed,
            threads=threads,
        )[:, :, 0]
    return reference_q.copy(), profiles


def integrate_detector_frames(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
    *,
    threads: int | None = None,
) -> np.ndarray:
    """Integrate fixed detector frames through the native plan for parity tests."""
    if not isinstance(plan, NativeDirectMonteCarloPlan):
        raise TypeError("plan must be a NativeDirectMonteCarloPlan")
    if threads is None:
        threads = default_native_threads()
    if isinstance(threads, bool) or int(threads) != threads or not 1 <= threads <= 1024:
        raise ValueError("threads must be an integer in [1, 1024]")

    detector_images = np.asarray(images, dtype=np.float64)
    if detector_images.shape == plan.image_shape:
        detector_images = detector_images[np.newaxis, ...]
    expected_ndim = len(plan.image_shape) + 1
    if (
        detector_images.ndim != expected_ndim
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
    detector_images = np.ascontiguousarray(detector_images, dtype=np.float64)

    measurements = int(detector_images.shape[0])
    output = np.empty((measurements, plan.bins), dtype=np.float64, order="C")
    error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
    library = _load_native_library()
    status = library.xrdmc_integrate(
        _double_pointer(detector_images),
        ctypes.c_size_t(measurements),
        ctypes.c_size_t(plan.pixels),
        plan.csc_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        plan.csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _double_pointer(plan.csc_weights),
        ctypes.c_size_t(plan.csc_weights.size),
        _double_pointer(plan.normalization_denominators),
        _double_pointer(plan.q_grid),
        ctypes.c_size_t(plan.bins),
        ctypes.c_double(plan.q_normalization_band[0]),
        ctypes.c_double(plan.q_normalization_band[1]),
        ctypes.c_int(threads),
        _double_pointer(output),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise NativeBackendError(
            f"native detector integration failed with status {status}: "
            f"{message or 'no error detail'}"
        )
    return output


__all__ = [
    "NativeBackendError",
    "NativeBackendUnavailableError",
    "NativeDirectMonteCarloPlan",
    "default_native_threads",
    "direct_detector_monte_carlo",
    "direct_detector_monte_carlo_measurements",
    "integrate_detector_frames",
    "native_backend_available",
    "prepare_native_plan",
]
