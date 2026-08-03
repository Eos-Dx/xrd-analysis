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
All Python files in the analysis package are within the 1,000 physical-line
maximum. ``scripts/check_python_file_size.py`` has zero temporary exemptions.
Do not add an exemption without a documented removal stage and a dedicated test
gate.

Current private extraction boundaries
-------------------------------------

Public classes and functions remain at their historical import paths. Internal
implementations are currently split into:

* ``_snr_math.py`` for SNR numerical kernels;
* ``_goodness.py`` for goodness scoring and filtering;
* ``_pipeline_diagnostics.py`` for non-fatal train/test split summaries;
* ``_pipeline_experimentation.py`` for Optuna tuning and repeated-run
  variability evaluation;
* ``_pipeline_export.py`` for fitted pipeline serialization and CSV exports;
* ``_pipeline_validation.py`` for binary and multiclass validation support;
* ``_evaluation_utils.py`` for ROC and threshold calculations;
* ``_utility_hdf.py`` for HDF5 table loading;
* ``_utility_statistics.py`` for grouped profile statistics and plotting;
* ``_utility_angular.py`` for angular-range and weighted-integration helpers;
* ``_utility_image.py`` for image and PONI geometry helpers;
* ``_calibration_profiles.py`` for private calibration-profile numerical
  kernels;
* ``_spectrokinetic_math.py`` for private SVD/MCR-ALS numerical kernels;
* ``_spectrokinetic_mcr_support.py`` for stateless matrix, mask, fixed-spectra,
  initialization, and group-wavelength preparation.

``utility_functions.py`` remains the canonical compatibility facade for all
historical utility import paths. Its public wrappers preserve call signatures,
module-qualified names, public monkeypatch seams, and the historical
``from xrdanalysis.data_processing.utility_functions import *`` metric names
(``RocCurveDisplay``, ``auc``, ``f1_score``, ``precision_score``, and
``roc_curve``). Private utility modules are implementation details and must not
become required consumer imports.

Transformer implementation support is divided into five private modules:

* ``_transformer_dataframe.py`` for dataframe-oriented transformer support;
* ``_transformer_goodness.py`` for goodness transformation and filtering;
* ``_transformer_profile.py`` for profile and range transformation support;
* ``_transformer_signal.py`` for signal and Fourier transformation support;
* ``_transformer_soft_labels.py`` for soft-label and weighted-sample support.

``_transformer_compat.py`` centralizes compatibility dispatch used by canonical
wrappers. ``transformers.py`` remains the only canonical module for historical
transformer imports and class identity. It must preserve public signatures,
direct imports, public monkeypatch seams, sklearn/joblib deserialization, and
the spectrokinetic re-exports. No private transformer module is a public API.

``spectrokinetic_transformers.py`` retains the canonical public transformer
classes, ALS configuration/result types, ``run_als_iteration``, and
signature-preserving numerical helper wrappers, plus ALS orchestration and
payload/metadata assembly. The package and ``transformers`` facades retain
their class identity re-exports so historical imports and ``joblib`` artifacts
keep resolving. The private support modules must not become public import
paths. Its SK-Ana-inspired fixture data and frozen small matrices are
deterministic regression contracts. The optional parity fixture is absent, and
its generator is not a provenance-backed external SK-Ana reference.

``calibration_corrections.py`` remains the canonical public module for
``compute_calib_correction_profiles`` and ``correct_with_calib_profiles``.
Their signatures, package-facade identity, pickle paths, copy/in-place modes,
and tolerant/strict lookup behavior are compatibility contracts. Private
profile kernels remain importable from the canonical module for historical
internal callers, but ``_calibration_profiles.py`` is not a public API.

``fitting_functions.py`` retains all historical public fitting classes,
constructors, direct base class, attributes, module paths, and serialized
identities. A private helper centralizes only ordered parameter metadata.
Numerical formulas, producer parameter slicing, integer-array behavior, the
legacy skew-amplitude scaling, and Gamma edge behavior remain explicit
regression contracts; changing them requires a separate scientific decision
rather than a structural refactor.

Untouched compatibility boundaries
----------------------------------

``AzimuthalIntegration``, ``DeviationTransformer``, ``ColumnNormalizer``,
``DetectorJoiner``, and ``SNRTransformer`` remain canonical public classes in
``transformers.py``. Their direct imports, behavior, class identity, and saved
``joblib`` artifacts are outside these extraction changes. ``FaultyPixelDetector``
remains deferred as stated above. ``XRD-preprocessing`` remains an external,
read-only parity reference: this repository does not move, edit, or refactor
it as part of utility or transformer work.

Pipeline contracts
------------------

``pipeline.py`` retains the public ``MLPipeline`` and ``MLPipelineMulti``
methods, signatures, and serialized class paths. The private pipeline helpers
must not import these canonical classes. Public wrappers remain the dispatch
seams for ``predict``, ``predict_proba``, ``validate``, export, and maintained
test monkeypatches.
Historical validation dependency aliases imported from ``pipeline.py`` remain
available during the compatibility window; remove them only through an explicit
deprecation stage.

Training records the target column and optional sample-weight column. Both are
excluded from estimator features during prediction when present. Sample weights
are passed to estimator fitting only; they are never prediction features.

Binary validation finds an optimal threshold subject to both requested minimum
sensitivity and minimum specificity constraints. A score equal to that
threshold is positive (``score >= threshold``), consistently in validation and
binary CSV export. Exported binary pipelines carry ``optimal_threshold`` only
when it exists. Multiclass exported pipelines do not receive a binary
threshold. Multiclass probability columns stay in estimator ``classes_`` order.
