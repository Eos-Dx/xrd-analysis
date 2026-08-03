Legacy candidate audit
======================

Audit snapshot
--------------

This read-only audit was recorded on 2026-08-03 from ``dev_sad``. It covers
the named Difra, h5grpc, legacy, MCR-ALS, and d2xc candidates plus known local
consumers. It does not authorize deletion, branch removal, API deprecation, or
changes to ``XRD-preprocessing``.

Branch evidence
---------------

Counts below are ``dev_sad``-only / candidate-only commits at the audit
snapshot.

* ``difra_h5_grpc_legacy``: 32 / 0 locally and 33 / 0 for the remote-tracking
  ref. Both are ancestors of ``dev_sad``. The branch refs contain no unique
  commits, but removing either ref still requires explicit approval.
* ``mcr_als``: 88 / 0 locally and remotely. Both refs are ancestors of
  ``dev_sad``. The branch refs contain no unique commits; this does not make
  the current MCR-ALS API unused.
* ``origin/legacy``: 704 / 0 and an ancestor of ``dev_sad``. No code recovery
  from this ref is required before possible archival.
* ``origin/LEGACY_ULSTER``: 246 / 0 and an ancestor of ``dev_sad``. No code
  recovery from this ref is required before possible archival.
* ``d2xc_dev`` and ``origin/d2xc_dev``: 181 / 1 and not ancestors of
  ``dev_sad``. Unique commit ``514ff6b3`` adds editable manual points and
  stage stability/logging work under the removed embedded Difra hardware
  tree. Preserve this branch until that commit is compared semantically with
  the current standalone Difra repository.

Verified live surfaces
----------------------

* Standalone Difra's ``analysis_compat.py`` imports
  ``initialize_azimuthal_integrator_df``,
  ``initialize_azimuthal_integrator_poni_text``, ``FaultyPixelDetector``, and
  ``create_mask``. These are active compatibility surfaces. Faulty-pixel work
  remains deferred.
* ``XRD-preprocessing`` parity tests import the azimuthal-integration helpers,
  ``ColumnNormalizer``, and ``SNRTransformer``. It remains a read-only
  consumer and parity reference.
* ``eos_play`` scripts and notebooks use ``MCRALSTransformer`` extensively,
  including Clinical Trials, Keele, and Ulster workflows. MCR-ALS code and
  public imports are therefore live even though the old branch ref is fully
  merged.

Verified unused or redundant candidates
---------------------------------------

Only branch reachability is verified here. The fully merged branch refs listed
above contain no commits absent from ``dev_sad``. Exact current-source searches
found no ``h5grpc`` or ``d2xcdev`` module or import; those names principally
identify historical branches. This evidence is sufficient to classify the
fully merged local refs as technically redundant, not to remove remote archive
refs or related compatibility APIs.

Uncertain candidates
--------------------

``d2xc_dev`` is not redundant because commit ``514ff6b3`` is unique. The
standalone Difra repository has analogous editable-zone-point, continuous
movement, stage-limit, and telemetry handling, but equivalent behavior has not
been proven. A focused static comparison found:

* manual point editing is present in a newer, broader implementation supporting
  pixel and millimetre coordinates, measured-point protection, and allowed-zone
  validation;
* stage-controller ``RLock`` protection and Marlin position requests are
  present, with additional current position-cache handling;
* manual stage movement is now asynchronous, so the old warning about a
  synchronous move blocking real-time display is not directly applicable;
* the unique commit's explicit real-time active-state guard, callback lock,
  plot-tick recovery, and start/stop exception recovery are not present in the
  current ``technical/realtime_mixin.py``;
* its structured continuous-movement logging is not present in the current
  controller; and
* current logging installs ``sys.excepthook`` but not the unique commit's
  ``threading.excepthook`` or persistent ``faulthandler`` crash log.

The unique commit is therefore partially superseded, not safely discardable.
Any port belongs in standalone Difra with focused real-time, background-thread,
and hardware-controller tests. It is outside ``xrd-analysis`` refactoring.

``src/hardware/xystages.py`` is a compatibility shim whose target implementation
was removed. The only in-repository direct consumer is a legacy hardware test,
which now skips when that implementation is absent. Standalone Difra production
code uses ``difra.hardware.xystages``; its upstream-snapshot tests still mention
``hardware.xystages``. Removing or repairing the shim is a separate
compatibility decision, not a verified-safe cleanup in this phase.

Required decisions
------------------

#. Delete only fully merged local branch refs, or retain them as local archives?
#. Keep fully merged remote branches as historical archives, or schedule remote
   cleanup separately?
#. Compare and port any missing behavior from ``514ff6b3`` into standalone
   Difra, or preserve ``d2xc_dev`` only as an archive?
#. Remove, repair, or explicitly deprecate the broken legacy
   ``src/hardware/xystages.py`` shim in a separately tested change?

Test baseline
-------------

Before this documentation change, the supported full suite completed with 340
passed, 3 skipped, and 4 known warnings. The skips include the removed legacy
stage subsystem. No candidate code or branch was changed by this audit.
