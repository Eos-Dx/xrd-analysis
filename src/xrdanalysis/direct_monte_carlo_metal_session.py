"""Persistent Apple Metal sessions for repeated detector Monte Carlo runs."""

from __future__ import annotations

import ctypes
from hashlib import sha256
from threading import Lock
from dataclasses import dataclass
from typing import Sequence
from weakref import finalize

import numpy as np

from xrdanalysis.direct_monte_carlo_metal import (
    _ERROR_BUFFER_SIZE,
    MetalBackendError,
    NativeMetalMonteCarloPlan,
    _double_pointer,
    _load_metal_library,
    _measurement_seeds,
    _metal_source_path,
    _normalization_indices,
    _positive_integer,
    _prepare_output,
    _validate_images,
    _validate_scales,
    _validate_seed,
)


def _destroy_native_session(library: ctypes.CDLL, handle: int) -> None:
    library.xrdmc_metal_session_destroy(ctypes.c_void_p(handle))


def metal_plan_fingerprint(plan: NativeMetalMonteCarloPlan) -> str:
    """Return an exact content fingerprint for safe measurement grouping."""
    if not isinstance(plan, NativeMetalMonteCarloPlan):
        raise TypeError("plan must be a NativeMetalMonteCarloPlan")
    digest = sha256()
    digest.update(np.asarray(plan.image_shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(plan.q_normalization_band, dtype=np.float64).tobytes())
    for values in (
        plan.csr_indptr,
        plan.csr_indices,
        plan.csr_weights,
        plan.normalization_denominators,
        plan.q_grid,
    ):
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_frame_masked_inputs(
    plan: NativeMetalMonteCarloPlan,
    images: Sequence[np.ndarray],
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    detector_images = np.asarray(images, dtype=np.float64)
    if detector_images.shape == plan.image_shape:
        detector_images = detector_images[np.newaxis, ...]
    expected_ndim = len(plan.image_shape) + 1
    if (
        detector_images.ndim != expected_ndim
        or detector_images.shape[1:] != plan.image_shape
        or detector_images.shape[0] == 0
    ):
        raise ValueError(
            f"images must have shape {plan.image_shape} or "
            f"(measurements, {', '.join(str(value) for value in plan.image_shape)})"
        )

    frame_masks = np.asarray(masks)
    if frame_masks.shape == plan.image_shape and detector_images.shape[0] == 1:
        frame_masks = frame_masks[np.newaxis, ...]
    if frame_masks.shape != detector_images.shape:
        raise ValueError("masks must contain one frame-local mask per image")
    if frame_masks.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
        raise ValueError("masks must use uint8 or bool dtype")
    if np.any((frame_masks != 0) & (frame_masks != 1)):
        raise ValueError("masks must contain only binary values 0 and 1")
    frame_masks = np.ascontiguousarray(frame_masks, dtype=np.uint8)

    nonfinite = ~np.isfinite(detector_images)
    if np.any(nonfinite & (frame_masks == 0)):
        raise ValueError("unmasked detector pixels must contain only finite values")
    detector_images = np.array(detector_images, dtype=np.float64, order="C", copy=True)
    detector_images[frame_masks != 0] = 0.0
    detector_images = _validate_images(plan, detector_images)
    return detector_images, frame_masks


def _frame_masked_denominators(
    plan: NativeMetalMonteCarloPlan,
    masks: np.ndarray,
    pixel_normalization: np.ndarray | None,
) -> np.ndarray:
    normalization = (
        np.ones(plan.image_shape, dtype=np.float64)
        if pixel_normalization is None
        else np.asarray(pixel_normalization, dtype=np.float64)
    )
    if normalization.shape != plan.image_shape:
        raise ValueError("pixel_normalization must match plan.image_shape")
    if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise ValueError("pixel_normalization must be finite and positive")

    flat_normalization = normalization.ravel()
    flat_masks = masks.reshape(masks.shape[0], plan.pixels)
    unmasked = np.empty(plan.bins, dtype=np.float64)
    denominators = np.empty((masks.shape[0], plan.bins), dtype=np.float64)
    for bin_index in range(plan.bins):
        start = int(plan.csr_indptr[bin_index])
        stop = int(plan.csr_indptr[bin_index + 1])
        pixels = plan.csr_indices[start:stop]
        weighted_normalization = (
            plan.csr_weights[start:stop] * flat_normalization[pixels]
        )
        unmasked[bin_index] = np.sum(weighted_normalization, dtype=np.float64)
        denominators[:, bin_index] = np.sum(
            np.where(flat_masks[:, pixels] == 0, weighted_normalization, 0.0),
            axis=1,
            dtype=np.float64,
        )

    if not np.allclose(
        unmasked,
        plan.normalization_denominators,
        rtol=1e-6,
        atol=1e-8,
    ):
        maximum_error = float(
            np.max(np.abs(unmasked - plan.normalization_denominators))
        )
        raise ValueError(
            "pixel_normalization does not reproduce the unmasked pyFAI "
            f"normalization denominators (maximum absolute error {maximum_error:.6g})"
        )
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        raise ValueError("frame-local mask leaves an empty integration bin")
    return np.ascontiguousarray(denominators, dtype=np.float64)


class PersistentMetalMonteCarlo:
    """Own one immutable integration plan and its reusable GPU buffers."""

    def __init__(
        self,
        plan: NativeMetalMonteCarloPlan,
        images: np.ndarray,
        *,
        measurement_seeds: Sequence[int] | None = None,
        device: int = 0,
        scale_capacity: int = 8,
        profile_batch_size: int = 4096,
    ) -> None:
        detector_images = _validate_images(plan, images)
        if isinstance(device, bool) or int(device) != device or device < 0:
            raise ValueError("device must be a non-negative integer")
        scale_limit = _positive_integer(scale_capacity, "scale_capacity")
        batch_size = _positive_integer(profile_batch_size, "profile_batch_size")
        measurements = int(detector_images.shape[0])
        stable_seeds = _measurement_seeds(measurements, measurement_seeds)
        normalization_indices = _normalization_indices(plan)
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        library = _load_metal_library()
        handle = library.xrdmc_metal_session_create(
            str(_metal_source_path()).encode(),
            _double_pointer(detector_images),
            ctypes.c_size_t(measurements),
            ctypes.c_size_t(plan.pixels),
            stable_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            plan.csr_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            plan.csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            _double_pointer(plan.csr_weights),
            ctypes.c_size_t(plan.csr_weights.size),
            _double_pointer(plan.normalization_denominators),
            ctypes.c_size_t(plan.bins),
            normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(normalization_indices.size),
            ctypes.c_int(device),
            ctypes.c_size_t(scale_limit),
            ctypes.c_size_t(batch_size),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if not handle:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise MetalBackendError(
                f"persistent Metal session creation failed: "
                f"{detail or 'no error detail'}"
            )
        self._attach_native_session(
            library,
            int(handle),
            maximum_pixel=float(np.max(detector_images)),
            measurements=measurements,
            bins=plan.bins,
            scale_capacity=scale_limit,
            profile_batch_size=batch_size,
            plan_fingerprint=metal_plan_fingerprint(plan),
        )

    def _attach_native_session(
        self,
        library: ctypes.CDLL,
        handle: int,
        *,
        maximum_pixel: float,
        measurements: int,
        bins: int,
        scale_capacity: int,
        profile_batch_size: int,
        plan_fingerprint: str,
    ) -> None:
        self._library = library
        self._handle = handle
        self._finalizer = finalize(
            self,
            _destroy_native_session,
            library,
            handle,
        )
        self._lock = Lock()
        self._maximum_pixel = maximum_pixel
        self.measurements = measurements
        self.bins = bins
        self.scale_capacity = scale_capacity
        self.profile_batch_size = profile_batch_size
        self.plan_fingerprint = plan_fingerprint

    @property
    def closed(self) -> bool:
        """Return whether native GPU resources have been released."""
        return not self._finalizer.alive

    def close(self) -> None:
        """Release command queue and all persistent Metal buffers."""
        with self._lock:
            self._finalizer()

    def __enter__(self) -> PersistentMetalMonteCarlo:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _open_handle(self) -> ctypes.c_void_p:
        if self.closed:
            raise RuntimeError("persistent Metal session is closed")
        return ctypes.c_void_p(self._handle)

    def run(
        self,
        scales: Sequence[float],
        draws: int,
        *,
        seed: int = 0,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run all draws while reusing uploaded images and integration buffers."""
        noise_scales = _validate_scales(scales, self._maximum_pixel)
        if noise_scales.size > self.scale_capacity:
            raise ValueError("scales exceed persistent scale_capacity")
        draw_count = _positive_integer(draws, "draws")
        validated_seed = _validate_seed(seed)
        result = _prepare_output(
            (int(noise_scales.size), draw_count, self.measurements, self.bins),
            output,
        )
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        with self._lock:
            status = self._library.xrdmc_metal_session_run(
                self._open_handle(),
                _double_pointer(noise_scales),
                ctypes.c_size_t(noise_scales.size),
                ctypes.c_size_t(draw_count),
                ctypes.c_uint64(validated_seed),
                _double_pointer(result),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise MetalBackendError(
                f"persistent Metal execution failed with status {status}: "
                f"{detail or 'no error detail'}"
            )
        return result

    def integrate(self) -> np.ndarray:
        """Integrate the uploaded detector frames without random sampling."""
        result = np.empty((self.measurements, self.bins), dtype=np.float64, order="C")
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        with self._lock:
            status = self._library.xrdmc_metal_session_integrate(
                self._open_handle(),
                _double_pointer(result),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
        if status != 0:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise MetalBackendError(
                f"persistent Metal integration failed with status {status}: "
                f"{detail or 'no error detail'}"
            )
        return result


class GroupedPersistentMetalMonteCarlo:
    """Upload deduplicated plans and dispatch all measurements in one session."""

    def __init__(
        self,
        plans: Sequence[NativeMetalMonteCarloPlan],
        images: Sequence[np.ndarray],
        *,
        measurement_seeds: Sequence[int] | None = None,
        device: int = 0,
        scale_capacity: int = 8,
        profile_batch_size: int = 4096,
    ) -> None:
        if not plans or len(plans) != len(images):
            raise ValueError("plans and images must have equal non-zero length")
        reference = plans[0]
        for plan in plans[1:]:
            if (
                plan.image_shape != reference.image_shape
                or plan.bins != reference.bins
                or plan.q_normalization_band != reference.q_normalization_band
                or not np.array_equal(plan.q_grid, reference.q_grid)
            ):
                raise ValueError(
                    "all grouped plans must use the same image shape, q grid, and "
                    "normalization band"
                )
        detector_images = _validate_images(reference, np.stack(images))
        stable_seeds = _measurement_seeds(len(plans), measurement_seeds)
        if isinstance(device, bool) or int(device) != device or device < 0:
            raise ValueError("device must be a non-negative integer")
        scale_limit = _positive_integer(scale_capacity, "scale_capacity")
        batch_size = _positive_integer(profile_batch_size, "profile_batch_size")

        unique_plans: list[NativeMetalMonteCarloPlan] = []
        fingerprint_to_index: dict[str, int] = {}
        measurement_plan_indices: list[int] = []
        for plan in plans:
            fingerprint = metal_plan_fingerprint(plan)
            plan_index = fingerprint_to_index.get(fingerprint)
            if plan_index is None:
                plan_index = len(unique_plans)
                fingerprint_to_index[fingerprint] = plan_index
                unique_plans.append(plan)
            measurement_plan_indices.append(plan_index)

        indptr_parts: list[np.ndarray] = []
        indices_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        denominator_parts: list[np.ndarray] = []
        nonzero_offset = 0
        for plan in unique_plans:
            indptr_parts.append(plan.csr_indptr + nonzero_offset)
            indices_parts.append(plan.csr_indices)
            weight_parts.append(plan.csr_weights)
            denominator_parts.append(plan.normalization_denominators)
            nonzero_offset += int(plan.csr_weights.size)
        combined_indptr = np.ascontiguousarray(
            np.concatenate(indptr_parts),
            dtype=np.int64,
        )
        combined_indices = np.ascontiguousarray(
            np.concatenate(indices_parts),
            dtype=np.int32,
        )
        combined_weights = np.ascontiguousarray(
            np.concatenate(weight_parts),
            dtype=np.float64,
        )
        combined_denominators = np.ascontiguousarray(
            np.concatenate(denominator_parts),
            dtype=np.float64,
        )
        plan_mapping = np.ascontiguousarray(measurement_plan_indices, dtype=np.int32)
        normalization_indices = _normalization_indices(reference)
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        library = _load_metal_library()
        handle = library.xrdmc_metal_multi_session_create(
            str(_metal_source_path()).encode(),
            _double_pointer(detector_images),
            ctypes.c_size_t(len(plans)),
            ctypes.c_size_t(reference.pixels),
            stable_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_size_t(len(unique_plans)),
            plan_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            combined_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            combined_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            _double_pointer(combined_weights),
            ctypes.c_size_t(combined_weights.size),
            _double_pointer(combined_denominators),
            ctypes.c_size_t(reference.bins),
            normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(normalization_indices.size),
            ctypes.c_int(device),
            ctypes.c_size_t(scale_limit),
            ctypes.c_size_t(batch_size),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if not handle:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise MetalBackendError(
                f"grouped persistent Metal session creation failed: "
                f"{detail or 'no error detail'}"
            )
        session = PersistentMetalMonteCarlo.__new__(PersistentMetalMonteCarlo)
        combined_fingerprint = sha256(
            "".join(fingerprint_to_index).encode("ascii")
        ).hexdigest()
        session._attach_native_session(
            library,
            int(handle),
            maximum_pixel=float(np.max(detector_images)),
            measurements=len(plans),
            bins=reference.bins,
            scale_capacity=scale_limit,
            profile_batch_size=batch_size,
            plan_fingerprint=combined_fingerprint,
        )
        self._session = session
        self._group_count = len(unique_plans)
        self.measurements = len(plans)
        self.bins = reference.bins
        self.scale_capacity = scale_limit

    @property
    def group_count(self) -> int:
        """Number of distinct integration plans uploaded to Metal."""
        return self._group_count

    @property
    def closed(self) -> bool:
        return self._session.closed

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> GroupedPersistentMetalMonteCarlo:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(
        self,
        scales: Sequence[float],
        draws: int,
        *,
        seed: int = 0,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run each exact-plan group and restore original measurement order."""
        return self._session.run(scales, draws, seed=seed, output=output)

    def integrate(self) -> np.ndarray:
        """Integrate all uploaded frames and restore measurement order."""
        return self._session.integrate()


class FrameMaskedPreparedGeometryMetalMonteCarlo:
    """Share one exact pyFAI geometry LUT across frame-local detector masks.

    The plan must be warmed without a mask. ``pixel_normalization`` is the
    per-pixel normalization factor passed through the corresponding pyFAI
    correction path, for example ``integrator.solidAngleArray(image_shape)``
    when ``correctSolidAngle=True``. Each mask receives an independently
    recomputed denominator from the shared LUT weights.

    A plan is valid for one exact geometry only. Any center, distance, detector,
    wavelength, radial-range, or bin-count change requires a new pyFAI plan.
    """

    def __init__(
        self,
        plan: NativeMetalMonteCarloPlan,
        images: Sequence[np.ndarray],
        masks: np.ndarray,
        *,
        pixel_normalization: np.ndarray | None = None,
        measurement_seeds: Sequence[int] | None = None,
        device: int = 0,
        scale_capacity: int = 8,
        profile_batch_size: int = 4096,
    ) -> None:
        if not isinstance(plan, NativeMetalMonteCarloPlan):
            raise TypeError("plan must be a NativeMetalMonteCarloPlan")
        detector_images, frame_masks = _validate_frame_masked_inputs(
            plan,
            images,
            masks,
        )
        denominators = _frame_masked_denominators(
            plan,
            frame_masks,
            pixel_normalization,
        )
        if isinstance(device, bool) or int(device) != device or device < 0:
            raise ValueError("device must be a non-negative integer")
        scale_limit = _positive_integer(scale_capacity, "scale_capacity")
        batch_size = _positive_integer(profile_batch_size, "profile_batch_size")
        measurements = int(detector_images.shape[0])
        stable_seeds = _measurement_seeds(measurements, measurement_seeds)
        normalization_indices = _normalization_indices(plan)
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        library = _load_metal_library()
        handle = library.xrdmc_metal_frame_masked_session_create(
            str(_metal_source_path()).encode(),
            _double_pointer(detector_images),
            frame_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(measurements),
            ctypes.c_size_t(plan.pixels),
            stable_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            plan.csr_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            plan.csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            _double_pointer(plan.csr_weights),
            ctypes.c_size_t(plan.csr_weights.size),
            _double_pointer(denominators),
            ctypes.c_size_t(plan.bins),
            normalization_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(normalization_indices.size),
            ctypes.c_int(device),
            ctypes.c_size_t(scale_limit),
            ctypes.c_size_t(batch_size),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if not handle:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise MetalBackendError(
                "frame-masked prepared-geometry Metal session creation failed: "
                f"{detail or 'no error detail'}"
            )

        session = PersistentMetalMonteCarlo.__new__(PersistentMetalMonteCarlo)
        fingerprint = sha256()
        fingerprint.update(metal_plan_fingerprint(plan).encode("ascii"))
        fingerprint.update(frame_masks.tobytes(order="C"))
        fingerprint.update(denominators.tobytes(order="C"))
        session._attach_native_session(
            library,
            int(handle),
            maximum_pixel=float(np.max(detector_images)),
            measurements=measurements,
            bins=plan.bins,
            scale_capacity=scale_limit,
            profile_batch_size=batch_size,
            plan_fingerprint=fingerprint.hexdigest(),
        )
        denominators.setflags(write=False)
        self._session = session
        self.normalization_denominators = denominators
        self.measurements = measurements
        self.bins = plan.bins
        self.scale_capacity = scale_limit

    @property
    def closed(self) -> bool:
        return self._session.closed

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> FrameMaskedPreparedGeometryMetalMonteCarlo:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(
        self,
        scales: Sequence[float],
        draws: int,
        *,
        seed: int = 0,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run centered-Poisson draws with frame-local masks."""
        return self._session.run(scales, draws, seed=seed, output=output)

    def integrate(self) -> np.ndarray:
        """Integrate all frames deterministically with frame-local masks."""
        return self._session.integrate()


@dataclass(frozen=True)
class PreparedGeometryMetalResult:
    """Photon draws evaluated with one immutable set of pyFAI CSR plans."""

    profiles: np.ndarray
    deterministic_profiles: np.ndarray | None
    unique_plan_count: int


class PreparedGeometryMetalMonteCarlo:
    """Run photon Monte Carlo with geometry plans prepared outside Metal.

    The caller owns geometry construction. In particular, pyFAI may rebuild a
    bbox/CSR plan for every perturbed geometry. Metal only samples detector
    counts and applies those immutable weights.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        *,
        measurement_seeds: Sequence[int] | None = None,
        device: int = 0,
        profile_batch_size: int = 4096,
    ) -> None:
        if not images:
            raise ValueError("images must be non-empty")
        detector_images = [np.asarray(image, dtype=np.float64) for image in images]
        reference_shape = detector_images[0].shape
        if not reference_shape or any(image.shape != reference_shape for image in detector_images):
            raise ValueError("all images must have the same non-empty shape")
        if any(not np.isfinite(image).all() for image in detector_images):
            raise ValueError("images must contain only finite values")
        self._images = tuple(detector_images)
        self._measurement_seeds = measurement_seeds
        self._device = device
        self._profile_batch_size = profile_batch_size

    @property
    def measurements(self) -> int:
        return len(self._images)

    def run_geometry(
        self,
        plans: Sequence[NativeMetalMonteCarloPlan],
        photon_replicates: int,
        *,
        seed: int,
        include_deterministic: bool = False,
    ) -> PreparedGeometryMetalResult:
        """Evaluate photon replicates for one externally prepared geometry."""
        if len(plans) != self.measurements:
            raise ValueError("plans must contain one plan per measurement")
        replicate_count = _positive_integer(photon_replicates, "photon_replicates")
        with GroupedPersistentMetalMonteCarlo(
            plans,
            self._images,
            measurement_seeds=self._measurement_seeds,
            device=self._device,
            scale_capacity=1,
            profile_batch_size=self._profile_batch_size,
        ) as session:
            profiles = session.run((1.0,), replicate_count, seed=seed)[0]
            deterministic = session.integrate() if include_deterministic else None
            unique_plan_count = session.group_count
        return PreparedGeometryMetalResult(
            profiles=profiles,
            deterministic_profiles=deterministic,
            unique_plan_count=unique_plan_count,
        )


__all__ = [
    "FrameMaskedPreparedGeometryMetalMonteCarlo",
    "GroupedPersistentMetalMonteCarlo",
    "PreparedGeometryMetalMonteCarlo",
    "PreparedGeometryMetalResult",
    "PersistentMetalMonteCarlo",
    "metal_plan_fingerprint",
]
