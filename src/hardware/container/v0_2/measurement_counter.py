"""Measurement Counter Manager

Manages global, monotonic measurement counter for session containers.
Counter is stored as root attribute in HDF5 and survives crashes/interruptions.

The counter is:
- Initialized to 0 when session container is created
- Incremented atomically on each measurement (regular or analytical)
- Preserved even if measurement fails or is aborted
- Shared between regular and analytical measurements
- Never reset during session lifecycle
"""

import logging
import time
from pathlib import Path
from typing import Union

import h5py

from . import schema, utils

logger = logging.getLogger(__name__)


class MeasurementCounter:
    """Atomic measurement counter for session containers."""

    LOCK_TIMEOUT_SECONDS = 10
    LOCK_POLL_INTERVAL_MS = 50

    def __init__(self, session_file: Union[str, Path]):
        """Initialize counter for a session container.

        Args:
            session_file: Path to session HDF5 container
        """
        self.session_file = Path(session_file)
        self.lock_file = self.session_file.parent / f".{self.session_file.name}.lock"

    def _acquire_lock(self, timeout_seconds: float = LOCK_TIMEOUT_SECONDS) -> None:
        """Acquire exclusive lock on counter.

        Args:
            timeout_seconds: Lock acquisition timeout

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        start_time = time.time()
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # Try to create lock file exclusively
                with open(self.lock_file, "w") as f:
                    f.write(str(time.time()))
                break
            except FileExistsError:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        f"Could not acquire counter lock for {self.session_file} "
                        f"after {timeout_seconds}s"
                    )
                time.sleep(self.LOCK_POLL_INTERVAL_MS / 1000.0)

    def _release_lock(self) -> None:
        """Release counter lock."""
        try:
            self.lock_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to release counter lock: {e}")

    def get_current(self) -> int:
        """Get current counter value without incrementing.

        Args:
            Returns: Current counter value
        """
        try:
            with h5py.File(self.session_file, "r") as f:
                runtime = f.get(schema.GROUP_RUNTIME)
                if runtime is not None:
                    return int(runtime.attrs.get("measurement_counter", f.attrs.get("measurement_counter", 0)))
                return int(f.attrs.get("measurement_counter", 0))
        except Exception as e:
            logger.warning(f"Failed to read counter from {self.session_file}: {e}")
            return 0

    def get_next(self) -> int:
        """Get next counter value and increment atomically.

        Uses file-level locking to ensure atomic increment in multi-threaded scenarios.

        Args:
            Returns: Next measurement counter value

        Raises:
            TimeoutError: If counter lock cannot be acquired
            Exception: If HDF5 write fails
        """
        self._acquire_lock()
        try:
            with utils.open_h5_append(self.session_file) as f:
                runtime = f.get(schema.GROUP_RUNTIME)
                if runtime is not None:
                    current = int(runtime.attrs.get("measurement_counter", f.attrs.get("measurement_counter", 0)))
                else:
                    current = int(f.attrs.get("measurement_counter", 0))
                next_value = current + 1
                f.attrs["measurement_counter"] = next_value
                if runtime is not None:
                    runtime.attrs["measurement_counter"] = next_value
                logger.debug(
                    f"Counter incremented: {current} -> {next_value} for {self.session_file.name}"
                )
                return next_value
        finally:
            self._release_lock()

    def reset(self) -> None:
        """Reset counter to 0 (only for testing/recovery).

        Should not be used during normal operation.
        """
        self._acquire_lock()
        try:
            with utils.open_h5_append(self.session_file) as f:
                f.attrs["measurement_counter"] = 0
                runtime = f.get(schema.GROUP_RUNTIME)
                if runtime is not None:
                    runtime.attrs["measurement_counter"] = 0
                logger.warning(f"Counter reset to 0 for {self.session_file.name}")
        finally:
            self._release_lock()

    def get_metadata(self) -> dict:
        """Get counter metadata including timestamp and lock status.

        Returns:
            Dict with counter info
        """
        try:
            with h5py.File(self.session_file, "r") as f:
                runtime = f.get(schema.GROUP_RUNTIME)
                if runtime is not None:
                    counter = int(runtime.attrs.get("measurement_counter", f.attrs.get("measurement_counter", 0)))
                else:
                    counter = int(f.attrs.get("measurement_counter", 0))
                creation_time = f.attrs.get("creation_timestamp", "unknown")
        except Exception as e:
            logger.warning(f"Failed to read counter metadata: {e}")
            return {"counter": 0, "creation_timestamp": "unknown", "error": str(e)}

        return {
            "counter": counter,
            "creation_timestamp": creation_time,
            "file": str(self.session_file),
            "lock_file_exists": self.lock_file.exists(),
        }


def get_next_measurement_counter(session_file: Union[str, Path]) -> int:
    """Convenience function to get next counter value.

    Args:
        session_file: Path to session container

    Returns:
        Next measurement counter value
    """
    counter = MeasurementCounter(session_file)
    return counter.get_next()


def get_current_measurement_counter(session_file: Union[str, Path]) -> int:
    """Convenience function to get current counter value.

    Args:
        session_file: Path to session container

    Returns:
        Current counter value
    """
    counter = MeasurementCounter(session_file)
    return counter.get_current()


def reset_measurement_counter(session_file: Union[str, Path]) -> None:
    """Convenience function to reset counter (testing only).

    Args:
        session_file: Path to session container
    """
    counter = MeasurementCounter(session_file)
    counter.reset()
