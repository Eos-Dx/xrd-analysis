"""Optional CUDA direct detector Monte Carlo integration.

CuPy and cupyx are imported only when this backend is requested. The backend
samples detector pixels directly, applies the pyFAI sparse integration plan,
and never uses covariance or profile-space noise approximations.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np

from xrdanalysis.direct_monte_carlo import NativeDirectMonteCarloPlan


_UINT64_MASK = (1 << 64) - 1
_SCALE_STREAM_MIX = 0x9E3779B97F4A7C15
_MEASUREMENT_STREAM_MIX = 0xD1B54A32D192ED03


class CudaBackendUnavailableError(RuntimeError):
    """Raised when the optional CuPy CUDA stack cannot be used."""


class CudaBackendError(RuntimeError):
    """Raised when CUDA direct Monte Carlo execution fails."""


def _load_cuda_stack() -> tuple[Any, Any, int]:
    try:
        cupy = importlib.import_module("cupy")
        cupyx_sparse = importlib.import_module("cupyx.scipy.sparse")
    except Exception as error:
        raise CudaBackendUnavailableError(
            "CUDA direct Monte Carlo requires an optional CuPy installation "
            "matching the installed CUDA runtime"
        ) from error

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as error:
        raise CudaBackendUnavailableError(
            "CuPy is installed but no usable CUDA runtime is available"
        ) from error
    if device_count <= 0:
        raise CudaBackendUnavailableError("no CUDA devices are available")
    return cupy, cupyx_sparse, device_count


def cuda_backend_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""
    try:
        _load_cuda_stack()
    except CudaBackendUnavailableError:
        return False
    return True


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= _UINT64_MASK:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return int(seed)


def _validate_device(device: int | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, bool) or int(device) != device or device < 0:
        raise ValueError("device must be a non-negative integer or None")
    return int(device)


def _validate_images(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
) -> np.ndarray:
    if not isinstance(plan, NativeDirectMonteCarloPlan):
        raise TypeError("plan must be a NativeDirectMonteCarloPlan")
    detector_images = np.asarray(images, dtype=np.float64)
    if detector_images.shape == plan.image_shape:
        detector_images = detector_images[np.newaxis, ...]
    expected_ndim = len(plan.image_shape) + 1
    if (
        detector_images.ndim != expected_ndim
        or detector_images.shape[1:] != plan.image_shape
    ):
        shape_text = ", ".join(str(value) for value in plan.image_shape)
        raise ValueError(
            f"images must have shape {plan.image_shape} or (measurements, {shape_text})"
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
    scale_squared = np.square(noise_scales)
    if not np.all(np.isfinite(scale_squared)) or np.any(scale_squared == 0.0):
        raise ValueError("squared scales must be finite and positive")
    maximum_lambda = max(0.0, maximum_pixel) / float(np.min(scale_squared))
    if not np.isfinite(maximum_lambda) or maximum_lambda > np.iinfo(np.int64).max:
        raise ValueError("Poisson rate exceeds the supported CUDA int64 range")
    return np.ascontiguousarray(noise_scales, dtype=np.float64)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _stream_seed(seed: int, scale_index: int, measurement_index: int) -> int:
    mixed = seed
    mixed ^= ((scale_index + 1) * _SCALE_STREAM_MIX) & _UINT64_MASK
    mixed ^= ((measurement_index + 1) * _MEASUREMENT_STREAM_MIX) & _UINT64_MASK
    return _splitmix64(mixed)


def _gpu_csr(plan: NativeDirectMonteCarloPlan, cupy: Any, cupyx_sparse: Any) -> Any:
    int32_limit = np.iinfo(np.int32).max
    if plan.csc_weights.size > int32_limit or plan.csc_indptr[-1] > int32_limit:
        raise ValueError("integration plan exceeds cupyx sparse int32 index limits")
    csc = cupyx_sparse.csc_matrix(
        (
            cupy.asarray(plan.csc_weights, dtype=cupy.float64),
            cupy.asarray(plan.csc_indices, dtype=cupy.int32),
            cupy.asarray(plan.csc_indptr, dtype=cupy.int32),
        ),
        shape=(plan.bins, plan.pixels),
    )
    csr = csc.tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def _run_plan(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
    scales: np.ndarray,
    draws: int,
    *,
    seed: int,
    batch_draws: int,
    measurement_indices: Sequence[int],
    cupy: Any,
    cupyx_sparse: Any,
) -> np.ndarray:
    measurements = int(images.shape[0])
    if len(measurement_indices) != measurements:
        raise ValueError("measurement_indices must match images")

    output = np.empty(
        (int(scales.size), draws, measurements, plan.bins),
        dtype=np.float64,
    )
    integration = _gpu_csr(plan, cupy, cupyx_sparse)
    gpu_images = cupy.asarray(
        images.reshape(measurements, plan.pixels), dtype=cupy.float64
    )
    positive = cupy.maximum(gpu_images, 0.0)
    negative = gpu_images - positive
    denominators = cupy.asarray(plan.normalization_denominators, dtype=cupy.float64)
    q_min, q_max = plan.q_normalization_band
    normalization_bins = np.flatnonzero((plan.q_grid >= q_min) & (plan.q_grid <= q_max))
    gpu_normalization_bins = cupy.asarray(normalization_bins, dtype=cupy.int32)

    for scale_index, scale in enumerate(scales):
        scale_squared = float(scale * scale)
        for local_index, measurement_index in enumerate(measurement_indices):
            generator = cupy.random.default_rng(
                _stream_seed(seed, scale_index, int(measurement_index))
            )
            poisson_rate = positive[local_index] / scale_squared
            for draw_start in range(0, draws, batch_draws):
                draw_stop = min(draw_start + batch_draws, draws)
                current_draws = draw_stop - draw_start
                counts = generator.poisson(
                    lam=poisson_rate,
                    size=(current_draws, plan.pixels),
                )
                sampled = negative[local_index] + scale_squared * counts
                profiles = integration.dot(sampled.T).T / denominators
                normalizers = cupy.median(
                    profiles[:, gpu_normalization_bins],
                    axis=1,
                )
                valid = (
                    cupy.all(cupy.isfinite(profiles), axis=1)
                    & cupy.isfinite(normalizers)
                    & (normalizers > 1e-12)
                )
                if not bool(cupy.all(valid).item()):
                    invalid_offset = int(cupy.flatnonzero(~valid)[0].item())
                    raise CudaBackendError(
                        "non-finite or zero profile normalization at "
                        f"scale {scale_index}, draw {draw_start + invalid_offset}, "
                        f"measurement {measurement_index}"
                    )
                profiles /= normalizers[:, None]
                output[
                    scale_index,
                    draw_start:draw_stop,
                    local_index,
                    :,
                ] = cupy.asnumpy(profiles)
    return output


def _device_context(cupy: Any, device: int | None, device_count: int) -> Any:
    if device is None:
        return nullcontext()
    if device >= device_count:
        raise ValueError(
            f"device {device} is unavailable; detected CUDA device count is {device_count}"
        )
    return cupy.cuda.Device(device)


def direct_detector_monte_carlo_cuda(
    plan: NativeDirectMonteCarloPlan,
    images: np.ndarray,
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    batch_draws: int = 16,
    device: int | None = None,
) -> np.ndarray:
    """Run direct pixel-level centered-Poisson Monte Carlo on CUDA.

    Output axes are ``(scale, draw, measurement, radial_bin)``. GPU memory is
    bounded by ``batch_draws`` detector samples per measurement. Each
    scale/measurement RNG stream uses::

        splitmix64(seed ^ C_scale*(scale_index+1)
                         ^ C_measurement*(measurement_index+1))

    Results are deterministic for fixed inputs, ordering, ``batch_draws``, and
    CUDA/CuPy versions. This backend never falls back to CPU execution.
    """
    detector_images = _validate_images(plan, images)
    noise_scales = _validate_scales(scales, float(np.max(detector_images)))
    draw_count = _positive_integer(draws, "draws")
    batch_size = _positive_integer(batch_draws, "batch_draws")
    validated_seed = _validate_seed(seed)
    validated_device = _validate_device(device)
    cupy, cupyx_sparse, device_count = _load_cuda_stack()

    try:
        with _device_context(cupy, validated_device, device_count):
            return _run_plan(
                plan,
                detector_images,
                noise_scales,
                draw_count,
                seed=validated_seed,
                batch_draws=batch_size,
                measurement_indices=range(detector_images.shape[0]),
                cupy=cupy,
                cupyx_sparse=cupyx_sparse,
            )
    except (CudaBackendError, ValueError):
        raise
    except Exception as error:
        raise CudaBackendError(
            "CUDA direct Monte Carlo execution failed; reduce batch_draws if "
            "the failure is caused by GPU memory exhaustion"
        ) from error


def direct_detector_monte_carlo_cuda_measurements(
    plans: Sequence[NativeDirectMonteCarloPlan],
    images: Sequence[np.ndarray],
    scales: Sequence[float],
    draws: int,
    *,
    seed: int = 0,
    batch_draws: int = 16,
    device: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run distinct per-measurement plans and images on one common q grid."""
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

    detector_images = [
        _validate_images(plan, image) for plan, image in zip(plans, images, strict=True)
    ]
    if any(image.shape[0] != 1 for image in detector_images):
        raise ValueError("each image entry must contain exactly one measurement")
    maximum_pixel = max(float(np.max(image)) for image in detector_images)
    noise_scales = _validate_scales(scales, maximum_pixel)
    draw_count = _positive_integer(draws, "draws")
    batch_size = _positive_integer(batch_draws, "batch_draws")
    validated_seed = _validate_seed(seed)
    validated_device = _validate_device(device)
    cupy, cupyx_sparse, device_count = _load_cuda_stack()

    profiles = np.empty(
        (int(noise_scales.size), draw_count, len(plans), plans[0].bins),
        dtype=np.float64,
    )
    try:
        with _device_context(cupy, validated_device, device_count):
            for measurement_index, (plan, image) in enumerate(
                zip(plans, detector_images, strict=True)
            ):
                profiles[:, :, measurement_index, :] = _run_plan(
                    plan,
                    image,
                    noise_scales,
                    draw_count,
                    seed=validated_seed,
                    batch_draws=batch_size,
                    measurement_indices=(measurement_index,),
                    cupy=cupy,
                    cupyx_sparse=cupyx_sparse,
                )[:, :, 0, :]
    except (CudaBackendError, ValueError):
        raise
    except Exception as error:
        raise CudaBackendError(
            "CUDA direct Monte Carlo execution failed; reduce batch_draws if "
            "the failure is caused by GPU memory exhaustion"
        ) from error
    return reference_q.copy(), profiles


__all__ = [
    "CudaBackendError",
    "CudaBackendUnavailableError",
    "cuda_backend_available",
    "direct_detector_monte_carlo_cuda",
    "direct_detector_monte_carlo_cuda_measurements",
]
