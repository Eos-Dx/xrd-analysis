from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .grpc_client import OmniscanGrpcClient
from .security.usb_stub import MaintenanceUSBStub
from .audit import get_audit_logger, initialize_audit_logger
from .rbac import User, Role, Permission, require_permission, AccessControlError
from .errors import OmniscanError, get_error


class InteractiveSession:
    """Interactive session for Omniscan Orchestrator CLI with FDA compliance."""
    
    COMMANDS = [
        "status", "enter-maintenance", "exit-maintenance", "renew",
        "get-config", "set-config", "patch-config", "validate-config",
        "device-state", "device-power", "measure-start", "measure-stop",
        "measure-status", "measure-result",
        "whoami", "help", "quit", "exit", "clear"
    ]
    
    def __init__(
        self,
        client: OmniscanGrpcClient,
        server_address: str,
        user: Optional[User] = None,
        device_id: Optional[str] = None,
    ):
        self.client = client
        self.server_address = server_address
        self.console = Console()
        
        # User context with RBAC
        self.user = user or User(
            user_id="unknown",
            username="unknown",
            role=Role.CLINICAL_OPERATOR,
        )
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now(timezone.utc)
        self.device_id = device_id or "UNKNOWN_DEVICE"
        
        # Initialize audit logger if not already done
        if not get_audit_logger():
            audit_dir = Path.home() / ".omniscan" / "audit"
            initialize_audit_logger(audit_dir, self.device_id)
            
        self.audit = get_audit_logger()
        if self.audit:
            self.audit.log_session_event(
                self.user.user_id,
                "started",
                self.session_id,
                details={"server_address": server_address, "role": self.user.role.value}
            )
        
        # Setup prompt with completion
        completer = WordCompleter(self.COMMANDS, ignore_case=True)
        style = Style.from_dict({
            'prompt': '#00aa00 bold',
        })
        
        history_file = Path.home() / ".omniscan_history"
        self.session = PromptSession(
            completer=completer,
            style=style,
            history=FileHistory(str(history_file)),
        )
        
    def run(self):
        """Start the interactive session."""
        self.print_welcome()
        
        while True:
            try:
                command = self.session.prompt("omniscan> ")
                command = command.strip()
                
                if not command:
                    continue
                    
                if command.lower() in ("quit", "exit"):
                    print("[yellow]Goodbye![/yellow]")
                    break
                    
                if command.lower() == "clear":
                    self.console.clear()
                    continue
                    
                if command.lower() == "help":
                    self.print_help()
                    continue
                
                self.execute_command(command)
                
            except KeyboardInterrupt:
                print("\n[yellow]Use 'quit' or 'exit' to leave the session[/yellow]")
                continue
            except EOFError:
                print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                print(f"[red]Error: {e}[/red]")
                
    def print_welcome(self):
        """Print welcome message with user context."""
        self.console.print("\n[bold cyan]╔═══════════════════════════════════════╗[/bold cyan]")
        self.console.print("[bold cyan]║  Omniscan Orchestrator CLI Session   ║[/bold cyan]")
        self.console.print("[bold cyan]╚═══════════════════════════════════════╝[/bold cyan]\n")
        
        # User info panel
        user_info = f"""[bold]User:[/bold] {self.user.username}
[bold]Role:[/bold] {self.user.get_role_name()}
[bold]Session ID:[/bold] {self.session_id[:8]}...
[bold]Device:[/bold] {self.device_id}"""
        
        panel = Panel(user_info, title="Session Info", border_style="green")
        self.console.print(panel)
        
        self.console.print(f"\n[dim]Connected to gRPC server: {self.server_address}[/dim]")
        self.console.print("[dim]Type 'help' for available commands, 'quit' to exit[/dim]\n")
        
    def print_help(self):
        """Print help message."""
        table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        
        commands = [
            ("status", "Check server maintenance status"),
            ("enter-maintenance [ttl]", "Enter maintenance mode (default: 900s)"),
            ("exit-maintenance", "Exit maintenance mode"),
            ("renew [ttl]", "Renew maintenance lease (default: 900s)"),
            ("get-config", "Get current configuration"),
            ("set-config <file>", "Set configuration from JSON file"),
            ("patch-config <json>", "Patch configuration with JSON"),
            ("validate-config <file>", "Validate configuration file"),
            ("device-state", "Get current device state"),
            ("device-power on|off", "Control device power"),
            ("measure-start [duration] [mode]", "Start measurement"),
            ("measure-stop", "Stop current measurement"),
            ("measure-status", "Get measurement status"),
            ("measure-result", "Get last measurement result"),
            ("whoami", "Display current user and permissions"),
            ("help", "Show this help message"),
            ("clear", "Clear screen"),
            ("quit/exit", "Exit session"),
        ]
        
        for cmd, desc in commands:
            table.add_row(cmd, desc)
            
        self.console.print(table)
        print()
        
    def execute_command(self, command: str):
        """Execute a command with audit logging and access control."""
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        try:
            # Special commands
            if cmd == "whoami":
                self._show_user_info()
                return
                
            # Maintenance commands
            if cmd == "status":
                result = self.client.get_status()
                self.print_json(result)
                
            elif cmd == "enter-maintenance":
                # Check permission
                try:
                    require_permission(self.user, Permission.ENTER_MAINTENANCE_MODE)
                except AccessControlError as e:
                    print(f"[red]{e}[/red]")
                    if self.audit:
                        self.audit.log_access_denied(
                            self.user.user_id,
                            "enter_maintenance",
                            "insufficient_permissions"
                        )
                    return
                    
                ttl = int(args[0]) if args else 900
                result = self.client.enter_maintenance(ttl)
                
                # Audit log
                if self.audit:
                    self.audit.log_maintenance_mode(
                        self.user.user_id,
                        "entered",
                        self.session_id,
                        details={"ttl_seconds": ttl}
                    )
                    
                self.print_json(result)
                
            elif cmd == "exit-maintenance":
                # Check permission
                try:
                    require_permission(self.user, Permission.EXIT_MAINTENANCE_MODE)
                except AccessControlError as e:
                    print(f"[red]{e}[/red]")
                    return
                    
                result = self.client.exit_maintenance()
                
                # Audit log
                if self.audit:
                    self.audit.log_maintenance_mode(
                        self.user.user_id,
                        "exited",
                        self.session_id
                    )
                    
                self.print_json(result)
                
            elif cmd == "renew":
                ttl = int(args[0]) if args else 900
                result = self.client.renew_maintenance(ttl)
                self.print_json(result)
                
            # Config commands
            elif cmd == "get-config":
                result = self.client.get_config()
                self.print_json(result)
                
            elif cmd == "set-config":
                # Check permission
                try:
                    require_permission(self.user, Permission.MODIFY_CONFIG)
                except AccessControlError as e:
                    print(f"[red]{e}[/red]")
                    return
                    
                if not args:
                    print("[red]Error: Missing file path[/red]")
                    return
                file_path = Path(args[0])
                if not file_path.exists():
                    print(f"[red]Error: File not found: {file_path}[/red]")
                    return
                    
                # Get old config for audit
                old_config = self.client.get_config()
                data = json.loads(file_path.read_text(encoding="utf-8"))
                result = self.client.set_config(data)
                
                # Audit log config changes
                if self.audit:
                    self.audit.log_config_change(
                        self.user.user_id,
                        "full_config",
                        old_config,
                        data,
                        self.session_id
                    )
                    
                self.print_json(result)
                
            elif cmd == "patch-config":
                if not args:
                    print("[red]Error: Missing JSON patch[/red]")
                    return
                patch = json.loads(" ".join(args))
                result = self.client.patch_config(patch)
                self.print_json(result)
                
            elif cmd == "validate-config":
                if not args:
                    print("[red]Error: Missing file path[/red]")
                    return
                file_path = Path(args[0])
                if not file_path.exists():
                    print(f"[red]Error: File not found: {file_path}[/red]")
                    return
                data = json.loads(file_path.read_text(encoding="utf-8"))
                result = self.client.validate_config(data)
                self.print_json(result)
                
            # Device commands
            elif cmd == "device-state":
                result = self.client.get_device_state()
                self.print_json(result)
                
            elif cmd == "device-power":
                if not args or args[0].lower() not in ("on", "off"):
                    print("[red]Error: Specify 'on' or 'off'[/red]")
                    return
                power_on = args[0].lower() == "on"
                result = self.client.power_device("detector", power_on, self.user.user_id)
                self.print_json(result)
                
            elif cmd == "measure-start":
                duration_s = int(args[0]) if args else 60
                result = self.client.start_measurement(duration_s * 1000, self.user.user_id)
                self.print_json(result)
                
            elif cmd == "measure-stop":
                result = self.client.stop_measurement(self.user.user_id)
                self.print_json(result)
                
            elif cmd == "measure-status":
                result = self.client.get_measurement_status()
                self.print_json(result)
                
            elif cmd == "measure-result":
                result = self.client.get_measurement_result()
                self.print_json(result)
                
            else:
                print(f"[red]Unknown command: {cmd}[/red]")
                print("[dim]Type 'help' for available commands[/dim]")
                
        except OmniscanError as e:
            # Handle Omniscan-specific errors with proper formatting
            print(f"[red]{e.user_message}[/red]")
            if self.audit:
                self.audit.log_error(
                    self.user.user_id,
                    e.get_error_ref(),
                    str(e),
                    details=e.technical_details,
                    session_id=self.session_id
                )
        except AccessControlError as e:
            print(f"[red]{e}[/red]")
        except Exception as e:
            print(f"[red]Error executing command: {e}[/red]")
            if self.audit:
                self.audit.log_error(
                    self.user.user_id,
                    "UNKNOWN",
                    str(e),
                    session_id=self.session_id
                )
            
        
    def _show_user_info(self):
        """Display current user information and permissions."""
        from .rbac import get_user_permissions, format_permissions
        
        permissions = get_user_permissions(self.user)
        
        info = f"""[bold]Username:[/bold] {self.user.username}
[bold]User ID:[/bold] {self.user.user_id}
[bold]Role:[/bold] {self.user.get_role_name()}
[bold]Session ID:[/bold] {self.session_id}
[bold]Session Started:[/bold] {self.session_start.strftime('%Y-%m-%d %H:%M:%S UTC')}

[bold]Permissions:[/bold]
{format_permissions(permissions)}"""
        
        panel = Panel(info, title="User Information", border_style="cyan")
        self.console.print(panel)
        print()
        
    def print_json(self, data):
        """Pretty print JSON data."""
        from rich.syntax import Syntax
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        self.console.print(syntax)
        print()
