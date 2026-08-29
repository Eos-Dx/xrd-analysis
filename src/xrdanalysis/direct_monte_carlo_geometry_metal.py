"""Persistent geometry-aware Apple Metal direct Monte Carlo integration.

The engine implements pyFAI radial bounding-box pixel splitting for flat,
rectangular detectors with exactly zero PONI rotations. Detector images,
masks, measurement seeds, and nominal geometry stay resident in Metal shared
buffers. Draw-specific effective distance, Poni1, and Poni2 are uploaded in
bounded chunks; no per-draw pyFAI sparse plan is constructed.
"""

from __future__ import annotations

import ctypes
import os
import weakref
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from xrdanalysis.direct_monte_carlo_metal import (
    MetalBackendError,
    _double_pointer,
    _ERROR_BUFFER_SIZE,
    _load_metal_library,
    _measurement_seeds,
    _metal_source_path,
    _positive_integer,
    _prepare_output,
    _validate_scales,
    _validate_seed,
)


_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class MetalDetectorGeometry:
    """Flat-detector PONI geometry accepted by the geometry-aware engine."""

    distance_m: float
    poni1_m: float
    poni2_m: float
    pixel1_m: float
    pixel2_m: float
    wavelength_m: float
    rot1_rad: float = 0.0
    rot2_rad: float = 0.0
    rot3_rad: float = 0.0
    orientation: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.distance_m,
            self.pixel1_m,
            self.pixel2_m,
            self.wavelength_m,
        )
        finite = (
            *positive,
            self.poni1_m,
            self.poni2_m,
            self.rot1_rad,
            self.rot2_rad,
            self.rot3_rad,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("detector geometry must contain only finite values")
        if any(value <= 0.0 for value in positive):
            raise ValueError(
                "distance, pixel sizes, and wavelength must be positive"
            )
        if (self.rot1_rad, self.rot2_rad, self.rot3_rad) != (0.0, 0.0, 0.0):
            raise ValueError(
                "geometry-aware Metal supports exactly zero PONI rotations"
            )
        if isinstance(self.orientation, bool) or self.orientation not in (1, 2, 3, 4):
            raise ValueError("detector orientation must be one of 1, 2, 3, or 4")

    @classmethod
    def from_pyfai(cls, integrator: Any) -> MetalDetectorGeometry:
        """Extract supported geometry from one pyFAI integrator."""
        detector = getattr(integrator, "detector", None)
        if detector is None:
            raise ValueError("integrator has no detector")
        if not bool(getattr(detector, "IS_FLAT", False)):
            raise ValueError("geometry-aware Metal requires a flat detector")
        if getattr(detector, "spline", None) is not None:
            raise ValueError("geometry-aware Metal does not support spline distortion")
        orientation = getattr(detector, "orientation", 3)
        orientation_value = int(getattr(orientation, "value", orientation) or 3)
        return cls(
            distance_m=float(integrator.dist),
            poni1_m=float(integrator.poni1),
            poni2_m=float(integrator.poni2),
            pixel1_m=float(detector.pixel1),
            pixel2_m=float(detector.pixel2),
            wavelength_m=float(integrator.wavelength),
            rot1_rad=float(integrator.rot1),
            rot2_rad=float(integrator.rot2),
            rot3_rad=float(integrator.rot3),
            orientation=orientation_value,
        )


def _destroy_geometry_session(library: ctypes.CDLL, handle: int) -> None:
    library.xrdmc_metal_geometry_session_destroy(ctypes.c_void_p(handle))


def _readonly_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _validate_q_grid(q_grid: Sequence[float]) -> tuple[np.ndarray, float, float]:
    values = _readonly_vector(q_grid, "q_grid")
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("q_grid must contain at least two increasing values")
    spacing = np.diff(values)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("geometry-aware Metal requires a uniform q_grid")
    q_delta = float(np.mean(spacing))
    q_min = float(values[0] - 0.5 * q_delta)
    q_max = float(values[-1] + 0.5 * q_delta)
    if q_min < -1e-10 or not np.isfinite((q_min, q_max)).all():
        raise ValueError("q_grid edges must be finite and non-negative")
    q_min = max(0.0, q_min)
    return values, q_min, q_max


def _normalization_indices(
    q_grid: np.ndarray,
    q_normalization_band: tuple[float, float],
) -> np.ndarray:
    if len(q_normalization_band) != 2:
        raise ValueError("q_normalization_band must contain two values")
    q_min, q_max = (float(value) for value in q_normalization_band)
    if not np.isfinite((q_min, q_max)).all() or q_min > q_max:
        raise ValueError("q_normalization_band must be finite and ordered")
    selected = np.ascontiguousarray(
        np.flatnonzero((q_grid >= q_min) & (q_grid <= q_max)),
        dtype=np.int32,
    )
    if selected.size == 0:
        raise ValueError("q_normalization_band contains no q-grid bins")
    return selected


def _validate_detector_images(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float64)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("images must have shape (rows, columns) or (measurements, rows, columns)")
    if any(dimension <= 0 for dimension in values.shape):
        raise ValueError("image dimensions must be positive")
    return np.ascontiguousarray(values, dtype=np.float64)


def _sanitize_masked_detector_pixels(
    images: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    nonfinite = ~np.isfinite(images)
    if np.any(nonfinite & (masks == 0)):
        raise ValueError("unmasked detector pixels must contain only finite values")
    if np.any(nonfinite):
        images = images.copy()
        images[nonfinite] = 0.0
    if np.max(np.abs(images)) > np.finfo(np.float32).max:
        raise ValueError("images exceed Metal float32 range")
    return images


def _validate_masks(masks: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if masks is None:
        return np.zeros(shape, dtype=np.uint8)
    values = np.asarray(masks)
    if values.shape == shape[1:]:
        values = np.broadcast_to(values, shape)
    if values.shape != shape:
        raise ValueError(f"masks must have shape {shape[1:]} or {shape}")
    return np.ascontiguousarray(values != 0, dtype=np.uint8)


def _geometry_vectors(
    geometries: Sequence[MetalDetectorGeometry],
    measurements: int,
) -> dict[str, np.ndarray]:
    if len(geometries) != measurements:
        raise ValueError("geometries must match detector images")
    if not all(isinstance(value, MetalDetectorGeometry) for value in geometries):
        raise TypeError("geometries must contain MetalDetectorGeometry values")
    return {
        "distance": np.ascontiguousarray(
            [value.distance_m for value in geometries], dtype=np.float64
        ),
        "poni1": np.ascontiguousarray(
            [value.poni1_m for value in geometries], dtype=np.float64
        ),
        "poni2": np.ascontiguousarray(
            [value.poni2_m for value in geometries], dtype=np.float64
        ),
        "pixel1": np.ascontiguousarray(
            [value.pixel1_m for value in geometries], dtype=np.float64
        ),
        "pixel2": np.ascontiguousarray(
            [value.pixel2_m for value in geometries], dtype=np.float64
        ),
        "wavelength": np.ascontiguousarray(
            [value.wavelength_m for value in geometries], dtype=np.float64
        ),
        "rotations": np.ascontiguousarray(
            [
                (value.rot1_rad, value.rot2_rad, value.rot3_rad)
                for value in geometries
            ],
            dtype=np.float64,
        ),
        "orientation": np.ascontiguousarray(
            [value.orientation for value in geometries], dtype=np.int32
        ),
    }


def _uint8_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _int32_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _uint64_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint64):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


class GeometryAwareMetalMonteCarlo:
    """Own persistent detector data and integrate draw-specific PONI geometry."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray | None,
        geometries: Sequence[MetalDetectorGeometry],
        q_grid: Sequence[float],
        q_normalization_band: tuple[float, float],
        *,
        measurement_seeds: Sequence[int] | None = None,
        scale_capacity: int = 5,
        draw_capacity: int = 32,
        profile_batch_size: int = 16,
        device: int = 0,
    ) -> None:
        detector_images = _validate_detector_images(images)
        detector_masks = _validate_masks(masks, detector_images.shape)
        detector_images = _sanitize_masked_detector_pixels(
            detector_images,
            detector_masks,
        )
        measurements, rows, columns = detector_images.shape
        geometry = _geometry_vectors(geometries, measurements)
        q_values, q_min, q_max = _validate_q_grid(q_grid)
        normalization = _normalization_indices(q_values, q_normalization_band)
        seeds = _measurement_seeds(measurements, measurement_seeds)
        scale_count = _positive_integer(scale_capacity, "scale_capacity")
        draw_count = _positive_integer(draw_capacity, "draw_capacity")
        batch_size = _positive_integer(profile_batch_size, "profile_batch_size")
        if isinstance(device, bool) or int(device) != device or device < 0:
            raise ValueError("device must be a non-negative integer")

        library = _load_metal_library()
        error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        handle = library.xrdmc_metal_geometry_session_create(
            os.fsencode(_metal_source_path()),
            _double_pointer(detector_images),
            _uint8_pointer(detector_masks),
            ctypes.c_size_t(measurements),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(columns),
            _uint64_pointer(seeds),
            _double_pointer(geometry["distance"]),
            _double_pointer(geometry["poni1"]),
            _double_pointer(geometry["poni2"]),
            _double_pointer(geometry["pixel1"]),
            _double_pointer(geometry["pixel2"]),
            _double_pointer(geometry["wavelength"]),
            _double_pointer(geometry["rotations"]),
            _int32_pointer(geometry["orientation"]),
            ctypes.c_size_t(q_values.size),
            ctypes.c_double(q_min),
            ctypes.c_double(q_max),
            _int32_pointer(normalization),
            ctypes.c_size_t(normalization.size),
            ctypes.c_size_t(scale_count),
            ctypes.c_size_t(draw_count),
            ctypes.c_int(int(device)),
            ctypes.c_size_t(batch_size),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if not handle:
            detail = error_buffer.value.decode("utf-8", errors="replace")
            raise MetalBackendError(
                "geometry-aware Metal session creation failed: "
                f"{detail or 'no error detail'}"
            )

        self._library = library
        self._handle = int(handle)
        self._finalizer = weakref.finalize(
            self,
            _destroy_geometry_session,
            library,
            int(handle),
        )
        self.measurements = measurements
        self.image_shape = (rows, columns)
        self.bins = int(q_values.size)
        self.q_grid = q_values
        self.q_normalization_band = tuple(
            float(value) for value in q_normalization_band
        )
        self.scale_capacity = scale_count
        self.draw_capacity = draw_count
        self.profile_batch_size = batch_size
        self._nominal_distance = geometry["distance"]
        self._nominal_poni1 = geometry["poni1"]
        self._nominal_poni2 = geometry["poni2"]
        self.maximum_image_value = float(np.max(detector_images))
        self.persistent_bytes = int(
            detector_images.nbytes
            + detector_masks.nbytes
            + seeds.nbytes
            + sum(values.nbytes for values in geometry.values())
        )

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Release persistent Metal buffers and command queue."""
        self._finalizer()

    def __enter__(self) -> GeometryAwareMetalMonteCarlo:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _open_handle(self) -> ctypes.c_void_p:
        if self.closed:
            raise RuntimeError("geometry-aware Metal session is closed")
        return ctypes.c_void_p(self._handle)

    def _draw_geometry(
        self,
        draws: int,
        effective_distance_m: np.ndarray | None,
        poni1_m: np.ndarray | None,
        poni2_m: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (draws, self.measurements)

        def prepare(
            values: np.ndarray | None,
            nominal: np.ndarray,
            name: str,
            *,
            positive: bool,
        ) -> np.ndarray:
            if values is None:
                result = np.broadcast_to(nominal, shape).copy()
            else:
                result = np.asarray(values, dtype=np.float64)
                if result.shape == (self.measurements,) and draws == 1:
                    result = result[np.newaxis, :]
                if result.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")
                result = np.ascontiguousarray(result, dtype=np.float64)
            if not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must contain only finite values")
            if positive and np.any(result <= 0.0):
                raise ValueError(f"{name} must contain positive values")
            return result

        return (
            prepare(
                effective_distance_m,
                self._nominal_distance,
                "effective_distance_m",
                positive=True,
            ),
            prepare(poni1_m, self._nominal_poni1, "poni1_m", positive=False),
            prepare(poni2_m, self._nominal_poni2, "poni2_m", positive=False),
        )

    @staticmethod
    def _draw_offset(value: int, draws: int) -> int:
        offset = _validate_seed(value)
        if offset + draws - 1 > _UINT64_MASK:
            raise ValueError("draw_offset plus draws exceeds unsigned 64-bit range")
        return offset

    def integrate(
        self,
        draws: int = 1,
        *,
        effective_distance_m: np.ndarray | None = None,
        poni1_m: np.ndarray | None = None,
        poni2_m: np.ndarray | None = None,
        draw_offset: int = 0,
        draw_chunk_size: int | None = None,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate fixed detector images for draw-specific geometry."""
        draw_count = _positive_integer(draws, "draws")
        offset = self._draw_offset(draw_offset, draw_count)
        distances, draw_poni1, draw_poni2 = self._draw_geometry(
            draw_count,
            effective_distance_m,
            poni1_m,
            poni2_m,
        )
        chunk_size = min(
            self.draw_capacity,
            draw_count,
            _positive_integer(
                self.draw_capacity if draw_chunk_size is None else draw_chunk_size,
                "draw_chunk_size",
            ),
        )
        result = _prepare_output(
            (draw_count, self.measurements, self.bins),
            output,
        )
        for start in range(0, draw_count, chunk_size):
            stop = min(start + chunk_size, draw_count)
            chunk_draws = stop - start
            chunk = np.empty(
                (chunk_draws, self.measurements, self.bins),
                dtype=np.float64,
                order="C",
            )
            error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
            status = self._library.xrdmc_metal_geometry_session_integrate(
                self._open_handle(),
                ctypes.c_size_t(chunk_draws),
                ctypes.c_uint64(offset + start),
                _double_pointer(distances[start:stop]),
                _double_pointer(draw_poni1[start:stop]),
                _double_pointer(draw_poni2[start:stop]),
                _double_pointer(chunk),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
            if status != 0:
                detail = error_buffer.value.decode("utf-8", errors="replace")
                raise MetalBackendError(
                    f"geometry-aware Metal integration failed with status {status}: "
                    f"{detail or 'no error detail'}"
                )
            result[start:stop] = chunk
        return result

    def run(
        self,
        scales: Sequence[float],
        draws: int,
        *,
        effective_distance_m: np.ndarray | None = None,
        poni1_m: np.ndarray | None = None,
        poni2_m: np.ndarray | None = None,
        seed: int = 0,
        draw_offset: int = 0,
        draw_chunk_size: int | None = None,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run centered-Poisson direct Monte Carlo with dynamic geometry."""
        draw_count = _positive_integer(draws, "draws")
        offset = self._draw_offset(draw_offset, draw_count)
        validated_seed = _validate_seed(seed)
        noise_scales = _validate_scales(scales, self.maximum_image_value)
        if noise_scales.size > self.scale_capacity:
            raise ValueError("scales exceed geometry session scale_capacity")
        distances, draw_poni1, draw_poni2 = self._draw_geometry(
            draw_count,
            effective_distance_m,
            poni1_m,
            poni2_m,
        )
        chunk_size = min(
            self.draw_capacity,
            draw_count,
            _positive_integer(
                self.draw_capacity if draw_chunk_size is None else draw_chunk_size,
                "draw_chunk_size",
            ),
        )
        result = _prepare_output(
            (int(noise_scales.size), draw_count, self.measurements, self.bins),
            output,
        )
        for start in range(0, draw_count, chunk_size):
            stop = min(start + chunk_size, draw_count)
            chunk_draws = stop - start
            chunk = np.empty(
                (
                    int(noise_scales.size),
                    chunk_draws,
                    self.measurements,
                    self.bins,
                ),
                dtype=np.float64,
                order="C",
            )
            error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
            status = self._library.xrdmc_metal_geometry_session_run(
                self._open_handle(),
                _double_pointer(noise_scales),
                ctypes.c_size_t(noise_scales.size),
                ctypes.c_size_t(chunk_draws),
                ctypes.c_uint64(offset + start),
                _double_pointer(distances[start:stop]),
                _double_pointer(draw_poni1[start:stop]),
                _double_pointer(draw_poni2[start:stop]),
                ctypes.c_uint64(validated_seed),
                _double_pointer(chunk),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
            if status != 0:
                detail = error_buffer.value.decode("utf-8", errors="replace")
                raise MetalBackendError(
                    f"geometry-aware Metal execution failed with status {status}: "
                    f"{detail or 'no error detail'}"
                )
            result[:, start:stop] = chunk
        return result

    def run_nested(
        self,
        scales: Sequence[float],
        geometry_draws: int,
        photon_replicates: int,
        *,
        effective_distance_m: np.ndarray | None = None,
        poni1_m: np.ndarray | None = None,
        poni2_m: np.ndarray | None = None,
        seed: int = 0,
        geometry_draw_offset: int = 0,
        photon_draw_offset: int = 0,
        geometry_chunk_size: int | None = None,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reuse each geometry realization across photon replicates.

        Output shape is ``(scale, geometry, photon, measurement, q)``. Geometry
        draws are independent; photon replicates are conditionally independent
        given one geometry draw.
        """
        geometry_count = _positive_integer(geometry_draws, "geometry_draws")
        replicate_count = _positive_integer(
            photon_replicates,
            "photon_replicates",
        )
        validated_seed = _validate_seed(seed)
        geometry_offset = self._draw_offset(geometry_draw_offset, geometry_count)
        total_photon_draws = geometry_count * replicate_count
        photon_offset = self._draw_offset(photon_draw_offset, total_photon_draws)
        noise_scales = _validate_scales(scales, self.maximum_image_value)
        if noise_scales.size > self.scale_capacity:
            raise ValueError("scales exceed geometry session scale_capacity")
        profiles_per_geometry = int(noise_scales.size) * replicate_count
        if profiles_per_geometry > self.profile_batch_size:
            raise ValueError(
                "scale count * photon_replicates exceeds profile_batch_size"
            )
        distances, draw_poni1, draw_poni2 = self._draw_geometry(
            geometry_count,
            effective_distance_m,
            poni1_m,
            poni2_m,
        )
        chunk_size = min(
            self.draw_capacity,
            geometry_count,
            _positive_integer(
                self.draw_capacity
                if geometry_chunk_size is None
                else geometry_chunk_size,
                "geometry_chunk_size",
            ),
        )
        result = _prepare_output(
            (
                int(noise_scales.size),
                geometry_count,
                replicate_count,
                self.measurements,
                self.bins,
            ),
            output,
        )
        for start in range(0, geometry_count, chunk_size):
            stop = min(start + chunk_size, geometry_count)
            chunk_draws = stop - start
            chunk = np.empty(
                (
                    int(noise_scales.size),
                    chunk_draws,
                    replicate_count,
                    self.measurements,
                    self.bins,
                ),
                dtype=np.float64,
                order="C",
            )
            error_buffer = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
            status = self._library.xrdmc_metal_geometry_session_run_nested(
                self._open_handle(),
                _double_pointer(noise_scales),
                ctypes.c_size_t(noise_scales.size),
                ctypes.c_size_t(chunk_draws),
                ctypes.c_size_t(replicate_count),
                ctypes.c_uint64(geometry_offset + start),
                ctypes.c_uint64(photon_offset + start * replicate_count),
                _double_pointer(distances[start:stop]),
                _double_pointer(draw_poni1[start:stop]),
                _double_pointer(draw_poni2[start:stop]),
                ctypes.c_uint64(validated_seed),
                _double_pointer(chunk),
                error_buffer,
                ctypes.c_size_t(len(error_buffer)),
            )
            if status != 0:
                detail = error_buffer.value.decode("utf-8", errors="replace")
                raise MetalBackendError(
                    "nested geometry-aware Metal execution failed with status "
                    f"{status}: {detail or 'no error detail'}"
                )
            result[:, start:stop] = chunk
        return result


__all__ = [
    "GeometryAwareMetalMonteCarlo",
    "MetalDetectorGeometry",
]
