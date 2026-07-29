Refactoring guardrails
======================

Scope
-----

The current refactoring scope is analysis-only: ``xrd-analysis`` remains the
owner of analysis and azimuthal-integration code. The first stages do not move,
edit, or reorganize
``XRD-preprocessing``. It is an external, read-only parity reference only.

Cross-repository SNR reference
------------------------------

``XRD-preprocessing`` is a read-only local reference for the common Poisson
SNR scalar contract. Normal ``xrd-analysis`` CI must neither import it nor
require a sibling checkout. A maintainer may opt in locally with
``XRD_PREPROCESSING_ROOT=/absolute/path/to/XRD-preprocessing`` and
``pytest -m cross_repo``. These checks compare only valid, aligned intensity
and sigma profiles and only the Poisson scalar metrics
``noise_std``, ``snr_linear``, ``snr_db``, and the ``poisson`` method label.

Parity is compatibility evidence, not authority to edit ``XRD-preprocessing``
from this repository. ``xrd-analysis`` intentionally retains residual and
``auto`` SNR modes, the legacy ``snr`` alias, smoothed/residual output arrays,
and tolerant batch handling; they are outside the cross-repository contract.
In Poisson mode, new SNR transformer objects default to native aligned
calculation.
``regrid_poisson=True`` is the explicit legacy opt-in, and restored old
pickles preserve their historical regridded Poisson scalar calculation.

Deferred FaultyPixel work
-------------------------

``FaultyPixelDetector`` refactoring is explicitly deferred. Current stages
must not move, split, or change its public imports, serialized behavior, or
pipeline behavior.

Difra compatibility
-------------------

Difra may load released ``xrdanalysis`` wheels and use current analysis import
paths. Keep azimuthal-integration behavior, the four-value 1D result contract
``(radial, intensity, sigma, distance)``, and accepted PONI/mask/error-model
inputs compatible until a separately validated Difra release proves otherwise.

Public and serialized APIs
--------------------------

Refactoring must preserve public module paths, exported names, call signatures,
and serialized ``joblib``/scikit-learn pipeline loading. New internal modules
are allowed only behind compatibility imports. Do not relocate a persisted class
without a compatibility shim and a stored-artifact loading test.

Stages and test gates
---------------------

#. Stabilize current tests and define numerical contract fixtures.
#. Add focused integration, HDF5, import, and serialization regression tests.
#. Extract one cohesive responsibility per change while preserving APIs.
#. Remove an adapter only after a release cycle and explicit approval.

Every behavior change requires focused unit tests and the complete supported
analysis test suite. Integration changes additionally require golden diffraction
fixtures; public API changes require import and artifact-loading tests.

Rollback
--------

Keep each extraction in a small, independently revertible commit. Retain the
previous implementation and public re-export during the compatibility window.
Before any deletion, create a preservation tag or archive and verify that the
test gate passes after restoration.

Python file-size budget
-----------------------

Run ``python scripts/check_python_file_size.py`` or ``make check-python-file-size``.
The analysis package defaults to a 1,000 physical-line maximum per Python file.
Three pre-existing modules have temporary, explicit ceilings in
``scripts/check_python_file_size.py``. They may not grow; each extraction should
reduce them until the default budget applies. Add an exemption only with a
documented removal stage and a dedicated test gate.

Current private extraction boundaries
-------------------------------------

Public classes and functions remain at their historical import paths. Internal
implementations are currently split into:

* ``_snr_math.py`` for SNR numerical kernels;
* ``_goodness.py`` for goodness scoring and filtering;
* ``_pipeline_diagnostics.py`` for non-fatal train/test split summaries;
* ``_evaluation_utils.py`` for ROC and threshold calculations;
* ``_spectrokinetic_math.py`` for private SVD/MCR-ALS numerical kernels;
* ``_spectrokinetic_mcr_support.py`` for stateless matrix, mask, fixed-spectra,
  initialization, and group-wavelength preparation.

``spectrokinetic_transformers.py`` retains the canonical public transformer
classes, ALS configuration/result types, ``run_als_iteration``, and
signature-preserving numerical helper wrappers, plus ALS orchestration and
payload/metadata assembly. The package and ``transformers`` facades retain
their class identity re-exports so historical imports and ``joblib`` artifacts
keep resolving. The private support modules must not become public import
paths. Its SK-Ana-inspired fixture data and frozen small matrices are
deterministic regression contracts. The optional parity fixture is absent, and
its generator is not a provenance-backed external SK-Ana reference.

``utility_functions.py`` retains signature-preserving wrappers for evaluation
functions. It also temporarily re-exports the historical scikit-learn metric
names used by wildcard-import notebooks. Remove those compatibility exports
only after consumer migration and an explicit deprecation period.
