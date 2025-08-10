"""Standardized logger utility for Ulster modules."""

import logging
from functools import wraps
from typing import Any, Dict, Optional

from .setup import get_logger, log_context, log_exceptions, log_performance


class UlsterLogger:
    """Enhanced logger wrapper for Ulster modules."""

    def __init__(self, name: str):
        self._logger = get_logger(name)
        self.name = name

    def debug(self, msg: str, **kwargs):
        """Log debug message with context."""
        self._logger.debug(msg, extra=kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message with context."""
        self._logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message with context."""
        self._logger.warning(msg, extra=kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """Log error message with context."""
        self._logger.error(msg, exc_info=exc_info, extra=kwargs)

    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(msg, extra=kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message with context."""
        self._logger.critical(msg, extra=kwargs)

    def hardware_state(self, state: str, msg: str, **kwargs):
        """Log message with hardware state context."""
        with log_context(hardware_state=state):
            self._logger.info(msg, extra=kwargs)

    def measurement(self, measurement_id: str, msg: str, level: str = "info", **kwargs):
        """Log message with measurement context."""
        with log_context(measurement_id=measurement_id):
            getattr(self._logger, level)(msg, extra=kwargs)

    def operation_start(self, operation: str, **kwargs):
        """Log start of an operation."""
        self._logger.debug(
            f"Starting {operation}", extra={**kwargs, "operation": f"{operation}_start"}
        )

    def operation_end(self, operation: str, success: bool = True, **kwargs):
        """Log end of an operation."""
        level = "info" if success else "error"
        status = "completed" if success else "failed"
        getattr(self._logger, level)(
            f"Operation {operation} {status}",
            extra={**kwargs, "operation": f"{operation}_end", "success": success},
        )

    def timing(self, operation: str, duration: float, **kwargs):
        """Log operation timing."""
        self._logger.info(
            f"Operation {operation} took {duration:.3f}s",
            extra={**kwargs, "operation": operation, "duration": duration},
        )

    def file_operation(
        self, action: str, file_path: str, success: bool = True, **kwargs
    ):
        """Log file operations."""
        level = "debug" if success else "error"
        status = "successful" if success else "failed"
        getattr(self._logger, level)(
            f"File {action} {status}: {file_path}",
            extra={
                **kwargs,
                "file_path": file_path,
                "file_action": action,
                "success": success,
            },
        )

    def detector_event(self, detector: str, event: str, **kwargs):
        """Log detector-specific events."""
        self._logger.info(
            f"Detector {detector}: {event}",
            extra={**kwargs, "detector": detector, "event": event},
        )

    def stage_event(self, position: tuple, event: str, **kwargs):
        """Log stage movement events."""
        x, y = position
        self._logger.info(
            f"Stage {event} at ({x:.3f}, {y:.3f})",
            extra={**kwargs, "stage_x": x, "stage_y": y, "event": event},
        )


def get_module_logger(module_name: str) -> UlsterLogger:
    """Get a standardized logger for a module."""
    return UlsterLogger(module_name)


def with_logging(
    operation: str = None, log_args: bool = False, log_result: bool = False
):
    """Decorator to add automatic logging to functions.

    Args:
        operation: Name of the operation (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """

    def decorator(func):
        op_name = operation or func.__name__
        logger = get_module_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log start
            log_data = {}
            if log_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)

            logger.operation_start(op_name, **log_data)

            try:
                result = func(*args, **kwargs)

                # Log success
                success_data = {}
                if log_result and result is not None:
                    success_data["result"] = str(result)[:100]  # Truncate long results

                logger.operation_end(op_name, success=True, **success_data)
                return result

            except Exception as e:
                # Log failure
                logger.operation_end(
                    op_name,
                    success=False,
                    error=str(e),
                    exception_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


def log_hardware_state(state: str):
    """Decorator to set hardware state context for a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with log_context(hardware_state=state):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_measurement(measurement_id: str = None):
    """Decorator to set measurement context for a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract measurement_id from args if not provided
            m_id = measurement_id
            if not m_id and len(args) > 0 and hasattr(args[0], "measurement_id"):
                m_id = getattr(args[0], "measurement_id", None)

            if m_id:
                with log_context(measurement_id=m_id):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
