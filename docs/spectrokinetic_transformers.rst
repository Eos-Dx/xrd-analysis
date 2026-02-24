Spectrokinetic Transformers
====================================

Overview
------------------------------------

This page documents the SK-Ana-inspired spectrokinetic transformers added for
matrix-per-row decomposition workflows.

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
