Spectrokinetic Transformers
====================================

Overview
------------------------------------

This page documents the SK-Ana-inspired spectrokinetic transformers added for
matrix-per-row decomposition workflows.

Compatibility and extraction boundary
-------------------------------------

``xrdanalysis.data_processing.spectrokinetic_transformers`` is the canonical
public module. It retains ``SpectroSVDTransformer``, ``MCRALSTransformer``,
``ALSConfig``, ``ALSResult``, ``run_als_iteration``, and the public numerical
helper call signatures. Pure numerical implementation lives in the private
``_spectrokinetic_math`` module behind canonical wrappers.
``_spectrokinetic_mcr_support`` is a second private boundary for stateless
matrix/axis/mask preparation, fixed spectra, initialization, and group
wavelength alignment. Neither private module is a public import path.

The following class import paths remain supported:

* canonical: ``xrdanalysis.data_processing.spectrokinetic_transformers``;
* package facade: ``xrdanalysis.data_processing``;
* legacy facade: ``xrdanalysis.data_processing.transformers``.

The two facades are identity re-exports, not replacement classes. This keeps
existing scikit-learn pipelines and ``joblib`` artifacts resolvable at their
canonical class module paths. Do not move either transformer class without an
explicit serialized-artifact migration and compatibility window.

The canonical module also retains ALS orchestration and DataFrame payload and
metadata assembly. Private support functions must not acquire ownership of
public classes, their serialized identities, or public numerical helper
signatures.

Test gate and reference status
------------------------------

Private extraction must preserve the canonical import surface, helper
signatures, transformer qualified class names, ``joblib`` round trips, and the
focused SVD, ALS, coupled-constraint, unimodality, and broadening tests. The
existing Keele fixture data and frozen small matrices are deterministic
regression contracts. The optional parity fixture is absent, and its generator
is not a provenance-backed external SK-Ana reference. The suite therefore
protects the current implementation contract; it does not claim independent
SK-Ana numerical parity.

Classes
------------------------------------

.. automodule:: xrdanalysis.data_processing.spectrokinetic_transformers
   :members:
   :show-inheritance:

Usage Notes
------------------------------------

- Canonical input orientation is ``(delay, wavelength)`` per matrix.
- ``SpectroSVDTransformer`` provides SVD diagnostics and reconstruction curves.
- ``MCRALSTransformer`` supports:

  - row-wise decomposition
  - grouped decomposition with ``tile_delay`` or ``mean`` strategy
  - fixed spectra (hard/soft)
  - correction spectra mode
  - broadening mode with three-stage sigma refinement

- Correction-spectra and broadening are mutually exclusive in v1.
