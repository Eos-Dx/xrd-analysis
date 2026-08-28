Direct Detector Monte Carlo
===========================

Purpose
-------

``xrdanalysis.direct_monte_carlo`` provides a reusable direct detector-level
Monte Carlo engine. It is independent of product labels, model scores, patient
selection, and report generation.

The implemented noise model is the centered Poisson model:

.. math::

   x^+ = \max(x, 0), \qquad
   x' = x - x^+ + s^2\,\operatorname{Poisson}(x^+ / s^2).

The backend does not use covariance propagation or a profile-space noise
approximation.

Native execution
----------------

The C++17/OpenMP backend converts the warmed pyFAI bbox/CSR lookup table to a
pixel-major CSC representation. Each sampled pixel is accumulated directly
into the radial bins. A temporary sampled detector image is not allocated.

The caller must warm pyFAI with the exact production settings before creating
the native plan. These settings include geometry, thickness correction, mask,
radial range, unit, solid-angle correction, pixel-splitting method, and the
normalization denominator. The native backend must not silently select or
modify them.

Build the optional backend:

.. code-block:: bash

   python -m xrdanalysis._native.build

macOS requires Homebrew ``libomp``. Linux requires a C++17 compiler with
OpenMP support.

Parallel execution
------------------

Use only one parallel level:

* One process can use 10--12 OpenMP threads for one bounded batch.
* A patient-level process pool can use 10--12 processes with one native thread
  per process.

Combining 10--12 processes with 10--12 OpenMP threads oversubscribes the host
and is not supported by the benchmark protocol. Patient-sized batches and
patient-level checkpoints bound memory and preserve restartability.

Use the ``spawn`` multiprocessing context on macOS. Forking after pyFAI or an
Objective-C runtime has initialized is unsafe and can terminate workers before
the calculation starts.

Validation gates
----------------

Before a consumer replaces its existing integration loop, validate:

* identical q grids for the same warmed pyFAI context;
* deterministic native integration of the same detector frame against pyFAI;
* centered-Poisson mean and variance;
* invariance to OpenMP thread count for the same native binary;
* profile mean, variance, and quantiles on 10 real patients, 100 draws, and
  noise scales 0.25, 0.5, 1.0, 1.25, and 1.5;
* consumer-level score quantiles and threshold-crossing frequencies.

The C++ standard does not require ``std::poisson_distribution`` to produce the
same draw sequence across standard-library implementations. Reproducibility is
guaranteed for the same native binary and seed. NumPy-to-C++ validation is
statistical rather than draw-by-draw.

CUDA decision gate
------------------

CUDA development begins only when the 10-patient native benchmark shows that
the projected 175-patient or 4000-patient runtime remains unacceptable. A CUDA
backend must preserve the same direct pixel-level noise model, integration
weights, normalization, seeds, output axes, and validation gates. It must not
introduce a profile-space approximation.

Reference benchmark
-------------------

The acceptance benchmark used 10 real patients, 59 retained measurements, 100
draws, five noise scales, and 10 patient processes. Each process used one
native thread. The existing Python/NumPy/pyFAI implementation and the native
implementation were measured under the same concurrent host load.

.. list-table::
   :header-rows: 1

   * - Implementation
     - Wall time
     - Profiles/s
   * - Python + pyFAI reference
     - 59.20 s
     - 498.29
   * - C++17/OpenMP direct MC
     - 39.92 s
     - 738.97

Native speedup was 1.48x. The maximum deterministic normalized-profile
difference from pyFAI was 2.33e-7. Across independent 100-draw RNG streams,
100% of profile means were within five pooled standard errors; the median
native/reference variance ratio was 1.0006. Its 5th--95th percentile range was
0.717--1.399, consistent with unstable per-bin variance estimates at only 100
draws.

At the measured native rate, a 4000-patient cohort with approximately 5.9
measurements per patient, 5000 draws, and five scales would require roughly
nine days on this host. The CPU acceptance benchmark therefore opens the CUDA
development gate. It does not authorize a 175-patient or 4000-patient run until
the CUDA backend passes the same deterministic and statistical checks.

CUDA backend status
-------------------

``xrdanalysis.direct_monte_carlo_cuda_native`` implements the fused native
CUDA backend. One CUDA block produces one normalized profile. Each detector
pixel is sampled once, then accumulated directly through the pixel-major CSC
plan in shared memory. The backend does not allocate sampled detector images.
``profile_batch_size`` bounds device output memory without changing random
samples. Shared-memory atomic accumulation can change final floating-point
roundoff at approximately machine precision when block geometry changes;
reproducibility is numerical, not bitwise.

Build on Linux with an NVIDIA CUDA toolkit:

.. code-block:: bash

   python -m xrdanalysis._native.build_cuda --architecture 80

Compute capability 6.0 or newer is required for double-precision atomic
accumulation. The deployment architecture should be supplied explicitly when
known. The native CUDA API never falls back to CuPy or CPU execution.

``integrate_detector_frames_cuda_native`` bypasses random sampling and uses
the same CUDA accumulation and normalization path. It provides the strict
integration-parity gate, with an explicit numerical tolerance, before
statistical Poisson validation.

The Python API rejects implicit host output allocations above 1 GiB. Process
large cohorts in patient-sized batches or pass a writable C-contiguous
``float64`` output such as ``numpy.memmap``. ``profile_batch_size`` alone does
not bound host output. Equivalent per-measurement plans are grouped into one
CUDA call; genuinely different plans require separate uploads and should be
grouped by acquisition geometry before execution.

``xrdanalysis.direct_monte_carlo_cuda`` remains a separate optional CuPy
reference implementation. ``batch_draws`` bounds its temporary detector
samples. Selecting either CUDA API is explicit; no dispatcher silently changes
the execution engine.

The current development host has no NVIDIA CUDA device. Import, validation,
and unavailable-device contracts are tested locally; numerical CPU/CUDA
parity and the real-patient CUDA benchmark remain mandatory on an NVIDIA host.
Until deterministic integration parity, Poisson statistics, the 10-patient
benchmark, and consumer-level score parity pass on the deployment NVIDIA host,
the native CUDA backend is experimental and must not be used for the
175-patient or 4000-patient calculation.

An NVIDIA CI or deployment-host validation job must require CUDA execution:

.. code-block:: bash

   XRDANALYSIS_REQUIRE_CUDA_TESTS=1 PYTHONPATH=src python -m pytest -q \
     src/tests/data_processing/test_direct_monte_carlo_cuda_native.py

With this flag, missing library or device is a test failure rather than a skip.

Metal backend status
--------------------

``xrdanalysis.direct_monte_carlo_metal`` implements direct detector-space
centered-Poisson Monte Carlo for Apple silicon. The Objective-C++ bridge uses
the system Metal runtime. It compiles the packaged Metal Shading Language
source with ``newLibraryWithSource`` and caches the compute pipelines in the
process. A full Xcode installation and the standalone ``metal`` compiler are
not required.

Build the optional bridge on macOS:

.. code-block:: bash

   python -m xrdanalysis._native.build_metal

The Metal plan is a deterministic bin-major CSR conversion of the same warmed
pyFAI bbox/CSR plan used by the CPU backend. One thread integrates one radial
bin in fixed CSR order. A counter-based random stream assigns one sample to
each ``(seed, measurement_seed, scale, draw, pixel)`` tuple, including pixels
that contribute to more than one radial bin. Stable measurement seeds make
results independent of local grouping and ordering. Changing
``profile_batch_size`` does not change the sampled values.

Metal execution is explicit and never falls back to CPU or CUDA. The GPU path
uses float32 values and rejects Poisson rates above ``2**24``. Host inputs and
outputs retain the float64 Python ABI. Deterministic real-measurement parity
with pyFAI must therefore use a numerical tolerance rather than bitwise
equality. Implicit host outputs above 1 GiB are rejected; callers must process
patient-sized batches or provide a writable C-contiguous float64 output such
as ``numpy.memmap``.

``PersistentMetalMonteCarlo`` keeps one image set, integration plan, command
queue, and all working buffers on the GPU between calls. Its ``run`` method
executes every requested draw through one Python-to-native call. Resources are
released explicitly by ``close`` or by the context-manager protocol.

``GroupedPersistentMetalMonteCarlo`` accepts one plan per measurement. Exact
duplicate plans are stored once. Distinct plans are concatenated into one GPU
buffer and selected by a measurement-to-plan mapping in the shader. This still
uses one native session and one call when every measurement has a different
mask, thickness correction, or geometry.

.. code-block:: python

   from xrdanalysis.direct_monte_carlo_metal_session import (
       GroupedPersistentMetalMonteCarlo,
   )

   with GroupedPersistentMetalMonteCarlo(
       plans,
       images,
       measurement_seeds=stable_measurement_seeds,
       profile_batch_size=16384,
   ) as session:
       profiles = session.run(
           scales=(0.25, 0.5, 1.0, 1.25, 1.5),
           draws=5000,
           seed=20260826,
           output=output_memmap,
       )

The local Apple M4 Pro benchmark used 10 real patients and 56 measurements.
All 56 exact integration plans were different. The persistent multi-plan path
therefore exercised the worst-case plan mapping rather than duplicate-plan
reuse. Maximum deterministic normalized-profile difference from pyFAI remained
``3.41e-5``. Session construction and initial upload took 0.34 s and are not
included in the run times below.

.. list-table::
   :header-rows: 1

   * - Execution
     - Draws
     - Profiles
     - Wall time
     - Profiles/s
   * - Original Metal one-shot
     - 100
     - 28,000
     - 4.24 s
     - 6,597
   * - Persistent multi-plan
     - 100
     - 28,000
     - 3.40 s
     - 8,240
   * - Persistent multi-plan
     - 5,000
     - 1,400,000
     - 174.48 s
     - 8,024

Persistent execution produced exactly the same 28,000 values as the paired
one-shot run, improved its runtime by 1.25x, and was approximately 11.2x faster
than the prior 739 profiles/s C++/OpenMP reference.
The 5,000-draw validation used one Python-to-Metal call and a 1.043 GiB float64
memmap. GPU output workspace remained bounded to 16,384 profiles. The CPU and
Metal reference cohorts contained 59 and 56 measurements respectively, so the
CPU comparison is throughput-based rather than a paired wall-time claim.
At the measured 5,000-draw rate, integration alone projects to approximately
54 minutes for 175 patients and 20.4 hours for 4,000 patients, assuming 5.9
measurements per patient and five noise scales. These projections exclude
preprocessing, model scoring, checkpoint I/O, and contention from other jobs.

Run the macOS validation contract with:

.. code-block:: bash

   PYTHONPATH=src python -m pytest -q \
     src/tests/data_processing/test_direct_monte_carlo_metal.py

API
---

.. automodule:: xrdanalysis.direct_monte_carlo
   :members:
   :show-inheritance:

.. automodule:: xrdanalysis.direct_monte_carlo_cuda
   :members:
   :show-inheritance:

.. automodule:: xrdanalysis.direct_monte_carlo_cuda_native
   :members:
   :show-inheritance:

.. automodule:: xrdanalysis.direct_monte_carlo_metal
   :members:
   :show-inheritance:

.. automodule:: xrdanalysis.direct_monte_carlo_metal_session
   :members:
   :show-inheritance:
