"""Ulster utilities package.

Logging utilities for the Ulster application. Use setup_logging() or
init_logging_from_env() early in your program (the GUI entrypoint does this
already) to enable logging to both console and a rotating file. Modules may
then use either the stdlib "logging" or hardware.Ulster.utils.logger helpers.

To override defaults, set environment variables before import. See
utils.logging_setup.init_logging_from_env for details.
"""

from .logging_setup import (
    configure_third_party_logging,
    get_log_stats,
    get_logger,
    init_logging_from_env,
    log_context,
    log_exceptions,
    log_performance,
    setup_logging,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "log_context",
    "log_performance",
    "log_exceptions",
    "configure_third_party_logging",
    "get_log_stats",
    "init_logging_from_env",
]
