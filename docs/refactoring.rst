Refactoring guardrails
======================

Scope
-----

The current refactoring scope is analysis-only: ``xrd-analysis`` remains the
owner of analysis and azimuthal-integration code. The first stages do not move,
edit, or reorganize
``XRD-preprocessing``. It is an external, read-only parity reference only.

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
Four pre-existing modules have temporary, explicit ceilings in
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
* ``_evaluation_utils.py`` for ROC and threshold calculations.

``utility_functions.py`` retains signature-preserving wrappers for evaluation
functions. It also temporarily re-exports the historical scikit-learn metric
names used by wildcard-import notebooks. Remove those compatibility exports
only after consumer migration and an explicit deprecation period.
