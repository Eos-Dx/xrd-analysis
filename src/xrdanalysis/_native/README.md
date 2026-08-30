# Direct detector Monte Carlo native backend

Optional C++17/OpenMP backend for exact pixel-level detector Monte Carlo. It
implements only the centered Poisson model:

```text
positive = max(pixel, 0)
negative_component = pixel - positive
sampled = negative_component + scale^2 * Poisson(positive / scale^2)
```

Each sampled pixel is accumulated directly into a pixel-major CSC copy of a
warmed pyFAI bbox/CSR lookup table. No sampled 2D detector frame is allocated.
No covariance approximation or Python fallback is provided.

## Build

macOS requires Homebrew `libomp`:

```bash
brew install libomp
python -m xrdanalysis._native.build
```

Set `LIBOMP_PREFIX` when `libomp` is installed outside the Homebrew prefix.
Linux requires a C++17 compiler with OpenMP support:

```bash
python -m xrdanalysis._native.build
```

The default output is placed beside this README. A different library can be
loaded through `XRDANALYSIS_DIRECT_MONTE_CARLO_LIBRARY`.

## Python use

Warm pyFAI using the exact production geometry, mask, radial range,
normalization corrections, and method before creating a plan. Additive signal
corrections such as dark subtraction must be applied to input images before
calling this backend:

```python
result = integrator.integrate1d_ng(
    reference_image,
    npt,
    radial_range=radial_range,
    mask=mask,
    unit="q_nm^-1",
    method=("bbox", "csr", "cython"),
)

from xrdanalysis.direct_monte_carlo import prepare_native_plan

plan = prepare_native_plan(
    integrator,
    reference_image.shape,
    normalization_denominators=result.sum_normalization,
    q_grid=result.radial,
    q_normalization_band=(6.7, 7.1),
)
profiles = plan.run(
    measurement_images,
    scales=(0.25, 0.5, 1.0, 1.25, 1.5),
    draws=100,
    seed=20260826,
    threads=12,
)
```

Output shape is `(scales, draws, measurements, bins)`. Process large cohorts in
patient-sized batches to bound output memory and checkpoint completed patients.

Seeds are derived independently for every `(scale, draw, measurement)` tuple.
Results are identical across OpenMP thread counts for the same native binary.
The C++ standard does not require identical `std::poisson_distribution` output
across different standard-library implementations.

## Native CUDA build

The CUDA backend implements the same direct detector-space centered Poisson
model. One CUDA block produces one output profile: detector pixels are sampled
once and accumulated directly through the pixel-major CSC integration plan.
No sampled detector image, covariance approximation, or CPU fallback is used.

Build on Linux with an NVIDIA CUDA toolkit:

```bash
python -m xrdanalysis._native.build_cuda
```

The default build includes compute capability 6.0 machine code and PTX. Select
a newer minimum architecture when the deployment GPU is known:

```bash
python -m xrdanalysis._native.build_cuda --architecture 80
```

CUDA is not supported on macOS. The default output is
`libxrdanalysis_direct_monte_carlo_cuda.so` beside this README. The shared
library exports ABI version 1 through `xrdmc_cuda_abi_version`, reports device
availability through `xrdmc_cuda_device_count`, and runs the fused kernel
through `xrdmc_cuda_run`. `xrdmc_cuda_integrate` uses the same accumulation and
normalization path without random sampling for deterministic parity checks.

`xrdmc_cuda_run` accepts detector images, stable per-measurement seeds, noise
scales, draws, the CSC plan, normalization denominators and normalization-bin
indices. Output is flattened in `(scale, draw, measurement, bin)` order.
Random draws are keyed by `(base seed, scale, draw, measurement, pixel)`, so
changing profile batches or CUDA launch geometry does not change the random
number assigned to a detector pixel. Shared-memory atomic accumulation may
change final floating-point roundoff when block geometry changes; compare
profiles with an explicit numerical tolerance. Devices older than compute
capability 6.0 are rejected because the fused accumulation requires
double-precision atomic addition.

## Native Metal build

The Metal backend runs direct detector-space centered Poisson Monte Carlo on
Apple GPUs. It uses a bin-major deterministic CSR plan: one threadgroup emits
one profile and one thread integrates one radial bin in fixed CSR order. A
counter-based random stream is keyed by `(base seed, measurement seed, scale,
draw, pixel)`, so grouping or ordering does not change draws when stable
measurement seeds are preserved. A pixel contributing to multiple bins receives
the same sampled value. No atomics, sampled detector frame, covariance
approximation, CPU fallback, or CuPy fallback is used.

Build the Objective-C++ runtime on macOS:

```bash
python -m xrdanalysis._native.build_metal
```

The default output is
`libxrdanalysis_direct_monte_carlo_metal.dylib` beside this README. The build
requires only the macOS Objective-C++ compiler, Foundation, and Metal runtime.
The packaged `direct_monte_carlo_metal.metal` source is compiled at runtime
through `newLibraryWithSource` and cached in-process, so the standalone `metal`
CLI compiler is not required.

The library exports ABI version 5 through `xrdmc_metal_abi_version`, reports
device availability through `xrdmc_metal_device_count`, and exposes
`xrdmc_metal_run`, `xrdmc_metal_integrate`, and persistent-session functions.
`xrdmc_metal_session_create` retains one shared plan. The multi-plan variant
deduplicates exact plans, concatenates distinct CSR arrays, and stores a
measurement-to-plan mapping on the GPU. Session run and integration calls reuse
the command queue, images, seeds, plans, and bounded output/status buffers.

`FrameMaskedPreparedGeometryMetalMonteCarlo` shares one unmasked pyFAI
bbox/CSR geometry plan across measurement frames with different binary masks.
The caller supplies the per-pixel pyFAI normalization correction, for example
`integrator.solidAngleArray(image_shape)` when `correctSolidAngle=True`. The
session recomputes every measurement/bin denominator from the common LUT,
pixel correction, and frame-local `uint8` mask. The Metal kernels skip masked
pixels before deterministic accumulation or Poisson sampling. This preserves
the counter-based photon stream while avoiding one warmed pyFAI plan per mask.

For geometry-uncertainty experiments, geometry remains a pyFAI responsibility.
Rebuild and warm the exact bbox/CSR engine after each PONI, detector-distance,
detector, wavelength, radial-range, or bin-count perturbation. Convert the
unmasked engine with `prepare_native_plan` and `prepare_metal_plan`, then use
`FrameMaskedPreparedGeometryMetalMonteCarlo` for all measurement frames that
share that exact geometry. A static CSR plan cannot be reused after a center or
distance change because q-bin membership and bbox pixel-splitting weights have
changed. Use a distinct deterministic seed per geometry draw so checkpoint
boundaries do not duplicate photon streams.

All creation functions receive the packaged `.metal` source path as their first
argument. Host images, plan weights, denominators, scales, and output use
float64 ABI arrays; the Metal
backend validates and converts them to float32 for GPU execution. Output is
flattened in `(scale, draw, measurement, bin)` order. Profile batches bound GPU
output memory. Poisson sampling uses exact inversion for small rates and PTRS
transformed rejection for larger rates; rates above the exact float32 integer
range are rejected.
