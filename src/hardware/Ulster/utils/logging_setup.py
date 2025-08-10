"""Enhanced logging setup for Ulster application."""

import json
import logging
import logging.handlers
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # Optional: capture Qt messages into Python logging if PyQt5 is present
    from PyQt5.QtCore import QtMsgType, qInstallMessageHandler
except ImportError:  # pragma: no cover
    qInstallMessageHandler = None
    QtMsgType = None


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""

    def __init__(self):
        super().__init__()
        self._context = threading.local()

    def filter(self, record):
        # Add session context if available
        session_id = getattr(self._context, "session_id", None)
        if session_id:
            record.session_id = session_id

        # Add hardware state context
        hw_state = getattr(self._context, "hardware_state", "unknown")
        record.hardware_state = hw_state

        # Add measurement context
        measurement_id = getattr(self._context, "measurement_id", None)
        if measurement_id:
            record.measurement_id = measurement_id

        return True

    def set_context(self, **kwargs):
        """Set context for current thread."""
        for key, value in kwargs.items():
            setattr(self._context, key, value)

    def clear_context(self):
        """Clear context for current thread."""
        self._context = threading.local()


class PerformanceFilter(logging.Filter):
    """Filter to track performance metrics."""

    def filter(self, record):
        # Add timing information for specific operations
        if hasattr(record, "operation"):
            if not hasattr(self, "_timings"):
                self._timings = {}

            operation = record.operation
            if operation.endswith("_start"):
                self._timings[operation] = record.created
            elif operation.endswith("_end"):
                start_key = operation.replace("_end", "_start")
                if start_key in self._timings:
                    duration = record.created - self._timings[start_key]
                    record.duration = duration
                    del self._timings[start_key]

        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add contextual information if available
        for attr in ["session_id", "hardware_state", "measurement_id", "duration"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)

        # Add exception information
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def _default_log_path(app_name: str = "Ulster") -> Path:
    """Return a platform-appropriate log file path."""
    base = Path.home()
    if sys.platform.startswith("win"):
        # e.g. C:\\Users\\<user>\\AppData\\Local\\Ulster\\ulster.log
        appdata = base / "AppData" / "Local" / app_name
        appdata.mkdir(parents=True, exist_ok=True)
        return appdata / "ulster.log"
    else:
        # e.g. ~/.local/state/Ulster/ulster.log (XDG_STATE_HOME) or ~/.Ulster
        state_home = Path(os.environ.get("XDG_STATE_HOME", base / ".local" / "state"))
        folder = state_home / app_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "ulster.log"


# Global context filter instance
_context_filter = ContextFilter()


def setup_logging(
    log_path: Optional[Path] = None,
    level: int = logging.INFO,
    config: Optional[Dict[str, Any]] = None,
    capture_stdio: bool = True,
    structured: bool = False,
) -> Path:
    """Configure advanced logging with multiple handlers and filters.

    Args:
        log_path: Custom log file path
        level: Base logging level
        config: Configuration dict with logging settings
        capture_stdio: Whether to redirect stdout/stderr to logging
        structured: Whether to use structured (JSON) logging for file handler

    Returns:
        Path: The resolved log file path used
    """
    if log_path is None:
        log_path = _default_log_path()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load config defaults
    if config is None:
        config = {}

    console_level = config.get("console_level", level)
    file_level = config.get("file_level", logging.DEBUG)

    # Create formatters
    if structured:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(hardware_state)-10s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create handlers
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=config.get("max_bytes", 10 * 1024 * 1024),
        backupCount=config.get("backup_count", 5),
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)

    # Add filters
    performance_filter = PerformanceFilter()
    file_handler.addFilter(_context_filter)
    file_handler.addFilter(performance_filter)
    console_handler.addFilter(_context_filter)

    # Configure root logger
    root = logging.getLogger()

    # Avoid duplicate handlers if called twice
    existing_handlers = [type(h).__name__ for h in root.handlers]
    if "RotatingFileHandler" not in existing_handlers:
        root.addHandler(file_handler)
    if "StreamHandler" not in existing_handlers:
        root.addHandler(console_handler)

    root.setLevel(logging.DEBUG)  # Let handlers control their own levels

    # Install exception hook
    def _excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let the default handler print a clean message for Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger = logging.getLogger("uncaught")
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
            extra={"operation": "uncaught_exception"},
        )

    sys.excepthook = _excepthook

    # Redirect stdout/stderr if requested
    if capture_stdio:
        _setup_stdio_capture()

    # Route Qt messages into logging, if available
    if qInstallMessageHandler is not None and QtMsgType is not None:
        _setup_qt_logging()

    # Log startup
    logger = logging.getLogger("setup")
    logger.info(
        "Logging initialized",
        extra={
            "operation": "logging_setup",
            "log_path": str(log_path),
            "console_level": logging.getLevelName(console_level),
            "file_level": logging.getLevelName(file_level),
            "structured": structured,
        },
    )

    return log_path.resolve()


def _setup_stdio_capture():
    """Setup stdout/stderr capture to logging."""

    class _StreamToLogger:
        def __init__(self, logger: logging.Logger, level: int):
            self.logger = logger
            self.level = level
            self._buffer = ""

        def write(self, message):
            if message and message != "\n":
                # Handle partial lines/buffers
                self._buffer += message
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    if line.strip():
                        self.logger.log(self.level, line.strip())

        def flush(self):
            if self._buffer.strip():
                self.logger.log(self.level, self._buffer.strip())
                self._buffer = ""

    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)


def _setup_qt_logging():
    """Setup Qt message handling."""

    def qt_message_handler(mode, context, message):
        # Map Qt message types to logging levels
        if mode == QtMsgType.QtDebugMsg:
            level = logging.DEBUG
        elif mode == QtMsgType.QtInfoMsg:
            level = logging.INFO
        elif mode == QtMsgType.QtWarningMsg:
            level = logging.WARNING
        elif mode == QtMsgType.QtCriticalMsg:
            level = logging.ERROR
        elif mode == QtMsgType.QtFatalMsg:
            level = logging.CRITICAL
        else:
            level = logging.INFO

        logging.getLogger("qt").log(level, message)

    try:
        qInstallMessageHandler(qt_message_handler)
    except Exception:
        # If this fails, continue without Qt message redirection
        logging.getLogger(__name__).debug(
            "Qt message handler install failed", exc_info=True
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger. Call after setup_logging()."""
    return logging.getLogger(name or __name__)


@contextmanager
def log_context(**kwargs):
    """Context manager to set logging context for current thread."""
    _context_filter.set_context(**kwargs)
    try:
        yield
    finally:
        _context_filter.clear_context()


def log_performance(operation: str):
    """Decorator to log performance of operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)

            # Log start
            logger.debug(
                f"Starting {operation}", extra={"operation": f"{operation}_start"}
            )

            try:
                result = func(*args, **kwargs)

                # Log success
                logger.debug(
                    f"Completed {operation}", extra={"operation": f"{operation}_end"}
                )

                return result

            except Exception as e:
                # Log failure
                logger.error(
                    f"Failed {operation}: {e}",
                    exc_info=True,
                    extra={"operation": f"{operation}_error"},
                )
                raise

        return wrapper

    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None, *, reraise: bool = True):
    """Decorator to catch and log exceptions uniformly."""

    def _decorator(fn):
        log = logger or logging.getLogger(fn.__module__)

        def _wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.exception(
                    "Unhandled exception in %s",
                    fn.__name__,
                    extra={"operation": f"exception_{fn.__name__}"},
                )
                if reraise:
                    raise
                return None

        return _wrapped

    return _decorator


def configure_third_party_logging():
    """Configure logging for third party libraries."""
    # Reduce noise from common third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Enable debug for our modules only
    logging.getLogger("hardware.Ulster").setLevel(logging.DEBUG)


def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics."""
    stats = {"handlers": [], "loggers": [], "total_records": 0}

    root = logging.getLogger()
    for handler in root.handlers:
        handler_info = {
            "type": type(handler).__name__,
            "level": logging.getLevelName(handler.level),
            "formatter": (
                type(handler.formatter).__name__ if handler.formatter else None
            ),
        }
        stats["handlers"].append(handler_info)

    # Count active loggers
    for name in logging.getLogger().manager.loggerDict:
        if (
            logging.getLogger(name).handlers
            or logging.getLogger(name).level != logging.NOTSET
        ):
            stats["loggers"].append(name)

    return stats
