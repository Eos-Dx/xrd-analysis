"""
Error Notifier for Omniscan Orchestrator

Displays user-friendly error messages using rich library
with color-coded severity levels.
"""

from enum import Enum
from rich.console import Console
from rich.panel import Panel


class ErrorSeverity(Enum):
    """Error severity levels with display colors."""
    INFO = "blue"
    WARNING = "yellow"
    ERROR = "red"
    CRITICAL = "bold red"


class ErrorNotifier:
    """Displays formatted error messages to the user.
    
    Uses rich library for professional console output with
    color-coded panels based on severity level.
    """
    
    def __init__(self):
        self.console = Console()
    
    def show_error(self, title: str, message: str, severity: ErrorSeverity):
        """Display error message with formatted panel.
        
        Args:
            title: Error title
            message: Error message details
            severity: ErrorSeverity level for color coding
        """
        color = severity.value
        
        panel = Panel(
            message,
            title=f"[{color}]{title}[/{color}]",
            border_style=color,
            expand=False
        )
        
        self.console.print(panel)
    
    # Specific error handlers for common workflow errors
    
    def key_switch_off_error(self):
        """Display error when key switch is OFF during login."""
        self.show_error(
            "Device Locked",
            "The device key switch is OFF. Please turn the key to the ON position to proceed.",
            ErrorSeverity.WARNING
        )
    
    def calibration_expired_error(self, hours_since: int = 24):
        """Display error when calibration has expired.
        
        Args:
            hours_since: Hours since last calibration
        """
        self.show_error(
            "Calibration Required",
            f"The daily calibration has expired (>{hours_since} hours old). "
            "Please perform a calibration measurement before proceeding with patient measurements.",
            ErrorSeverity.WARNING
        )
    
    def data_integrity_critical_error(self):
        """Display critical error for data integrity violation."""
        self.show_error(
            "CRITICAL: Data Integrity Violation",
            "The server has detected missing or corrupted data. Operations are blocked. "
            "Please contact service immediately.",
            ErrorSeverity.CRITICAL
        )
    
    def warmup_in_progress_info(self, remaining_seconds: int):
        """Display info message for warmup in progress.
        
        Args:
            remaining_seconds: Seconds remaining in warmup
        """
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        
        self.show_error(
            "Warmup In Progress",
            f"X-ray source is warming up. Time remaining: {minutes}m {seconds}s",
            ErrorSeverity.INFO
        )
    
    def connection_error(self, details: str):
        """Display connection error to hardware server.
        
        Args:
            details: Error details
        """
        self.show_error(
            "Connection Error",
            f"Cannot connect to hardware server: {details}",
            ErrorSeverity.ERROR
        )
    
    def session_expired_error(self):
        """Display error when user session has expired."""
        self.show_error(
            "Session Expired",
            "Your session has expired. Please log in again.",
            ErrorSeverity.WARNING
        )
    
    def measurement_blocked_error(self, reason: str):
        """Display error when measurement is blocked.
        
        Args:
            reason: Reason why measurement is blocked
        """
        self.show_error(
            "Measurement Blocked",
            f"Cannot start measurement: {reason}",
            ErrorSeverity.WARNING
        )
    
    def success_message(self, title: str, message: str):
        """Display success message.
        
        Args:
            title: Success title
            message: Success message
        """
        panel = Panel(
            message,
            title=f"[bold green]✓ {title}[/bold green]",
            border_style="green",
            expand=False
        )
        
        self.console.print(panel)
    
    def info_message(self, title: str, message: str):
        """Display informational message.
        
        Args:
            title: Info title
            message: Info message
        """
        panel = Panel(
            message,
            title=f"[bold blue]ℹ {title}[/bold blue]",
            border_style="blue",
            expand=False
        )
        
        self.console.print(panel)
