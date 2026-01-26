"""Compatibility shim for legacy imports.

Exposes stage controller classes from hardware.difra.hardware.xystages
under the legacy top-level module path 'hardware.xystages'.
"""

from hardware.difra.hardware.xystages import *  # noqa: F401,F403
