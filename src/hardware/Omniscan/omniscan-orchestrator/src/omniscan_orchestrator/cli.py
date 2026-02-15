from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .grpc_client import OmniscanGrpcClient
from .security.usb_stub import MaintenanceUSBStub
from .cert_manager import CertificateManager
from .interactive import InteractiveSession


app = typer.Typer(add_completion=False, help="Omniscan Orchestrator CLI")


def _build_client(
    server_address: str,
    usb_path: Optional[Path],
    cert: Optional[Path],
    key: Optional[Path],
    ca_cert: Optional[Path],
) -> OmniscanGrpcClient:
    client_cert_path = None
    client_key_path = None
    ca_cert_path = None
    
    if usb_path:
        usb = MaintenanceUSBStub(usb_path)
        ident = usb.load()
        # For USB stub, would need to extract cert/key paths
        # For now, use insecure connection
        pass
    elif cert and key:
        client_cert_path = str(cert)
        client_key_path = str(key)
        ca_cert_path = str(ca_cert) if ca_cert else None
        
    return OmniscanGrpcClient(
        server_address=server_address,
        client_cert_path=client_cert_path,
        client_key_path=client_key_path,
        ca_cert_path=ca_cert_path
    )


@app.command()
def interactive(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Start an interactive session with the orchestrator server"""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    session = InteractiveSession(client, server)
    session.run()


@app.command()
def status(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    client = _build_client(server, usb_path, cert, key, ca_cert)
    print(client.get_status())


@app.command("health")
def health(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Get full aggregate health from Rust server (Health/GetAggregateHealth)."""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.get_aggregate_health()
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("interlocks")
def interlocks(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Get safety interlock status (Safety/GetInterlockStatus)."""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.get_interlocks()
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("state")
def server_state(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Get server state (Acquisition/GetState)."""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.get_server_state()
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("detector-health")
def detector_health(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Get detector health (DeviceControl/GetDetectorHealth)."""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.get_detector_health()
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("motion-health")
def motion_health(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Get motion controller health (DeviceControl/GetMotionHealth)."""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.get_motion_health()
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("enter-maintenance")
def enter_maintenance(
    ttl: int = typer.Option(900, help="Lease TTL seconds"),
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    print(client.enter_maintenance(ttl))


@app.command("renew")
def renew(
    ttl: int = typer.Option(900, help="New TTL seconds"),
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    print(client.renew_maintenance(ttl))


@app.command("exit-maintenance")
def exit_maintenance(
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    print(client.exit_maintenance())


@app.command("get-config")
def get_config(
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    print(client.get_config())


@app.command("set-config")
def set_config(
    file: Path = typer.Option(..., exists=True, readable=True, help="JSON config file"),
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    data = json.loads(file.read_text(encoding="utf-8"))
    print(client.set_config(data))


@app.command("patch-config")
def patch_config(
    patch: str = typer.Option(..., help="Inline JSON patch object"),
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    print(client.patch_config(json.loads(patch)))


@app.command("validate-config")
def validate_config(
    file: Path = typer.Option(..., exists=True, readable=True, help="JSON config file"),
    base_url: str = typer.Option("https://localhost:8443/api/v1"),
    usb_path: Optional[Path] = typer.Option(None),
    cert: Optional[Path] = typer.Option(None),
    key: Optional[Path] = typer.Option(None),
    ca_cert: Optional[Path] = typer.Option(None),
):
    client = _build_client(base_url, usb_path, cert, key, ca_cert)
    data = json.loads(file.read_text(encoding="utf-8"))
    print(client.validate_config(data))


# Device control commands
@app.command("device-state")
def device_state(
    base_url: str = typer.Option("http://127.0.0.1:8000/api/v1"),
):
    """Get current device state"""
    import requests
    resp = requests.get(f"{base_url}/device/state")
    resp.raise_for_status()
    print(resp.json())


@app.command("device-power")
def device_power(
    on: bool = typer.Option(..., "--on/--off", help="Turn device power on or off"),
    base_url: str = typer.Option("http://127.0.0.1:8000/api/v1"),
):
    """Control device power"""
    import requests
    resp = requests.post(f"{base_url}/device/power", json={"on": on})
    resp.raise_for_status()
    print(resp.json())


@app.command("measure-start")
def measure_start(
    duration: int = typer.Option(60, help="Measurement duration in seconds"),
    file_name: Optional[str] = typer.Option(None, help="Output file name (without extension)"),
    mode: str = typer.Option("calibrant", help="Measurement mode"),
    base_url: str = typer.Option("http://127.0.0.1:8000/api/v1"),
):
    """Start a measurement"""
    import requests
    payload = {"duration_s": duration, "mode": mode}
    if file_name:
        payload["file_name"] = file_name
    resp = requests.post(f"{base_url}/measure/start", json=payload)
    resp.raise_for_status()
    print(resp.json())


@app.command("measure-stop")
def measure_stop(
    base_url: str = typer.Option("http://127.0.0.1:8000/api/v1"),
):
    """Stop current measurement"""
    import requests
    resp = requests.post(f"{base_url}/measure/stop")
    resp.raise_for_status()
    print(resp.json())


@app.command("measure-status")
def measure_status(
    base_url: str = typer.Option("http://127.0.0.1:8000/api/v1"),
):
    """Get measurement status"""
    import requests
    resp = requests.get(f"{base_url}/measure/status")
    resp.raise_for_status()
    print(resp.json())


@app.command("measure-result")
def measure_result(
    base_url: str = typer.Option("http://127.0.0.1:8000/api/v1"),
):
    """Get last measurement result"""
    import requests
    resp = requests.get(f"{base_url}/measure/result")
    resp.raise_for_status()
    print(resp.json())


@app.command("calibrate-detector")
def calibrate_detector(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
    user: str = typer.Option("operator", help="User ID for audit trail"),
):
    """Calibrate the detector (required after power-on or when system is locked)"""
    from rich import print
    client = _build_client(server, usb_path, cert, key, ca_cert)
    try:
        print("[yellow]Calibrating detector...[/yellow]")
        result = client.calibrate_detector(user)
        if "error" in result:
            print(f"[red]✗ Calibration failed: {result['error']}[/red]")
            raise typer.Exit(1)
        print("[green]✓ Detector calibrated successfully[/green]")
    finally:
        client.close()


@app.command("measure-workflow")
def measure_workflow(
    duration: int = typer.Option(60, help="Measurement duration in seconds"),
    mode: str = typer.Option("calibrant", help="Measurement mode"),
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
    user: str = typer.Option("operator", help="User ID for audit trail"),
):
    """Execute complete measurement workflow: connect, activate interlocks, power on, and measure"""
    import time
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    console = Console()
    client = _build_client(server, usb_path, cert, key, ca_cert)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Step 1: Check connection and status
            task = progress.add_task("Connecting to server...", total=None)
            status = client.get_status()
            if "error" in status:
                console.print(f"[red]✗ Connection failed: {status['error']}[/red]")
                raise typer.Exit(1)
            console.print(f"[green]✓ Connected to {server}[/green]")
            console.print(f"  State: {status.get('state', 'unknown')}")
            progress.remove_task(task)
            
            # Step 2: Check interlocks
            task = progress.add_task("Checking safety interlocks...", total=None)
            interlocks = status.get('interlocks', {})
            if not interlocks.get('overall_safe', False):
                console.print("[red]✗ Safety interlocks not satisfied:[/red]")
                console.print(f"  Emergency stop: {interlocks.get('emergency_stop', 'unknown')}")
                console.print(f"  Door closed: {interlocks.get('door_closed', 'unknown')}")
                console.print(f"  Radiation safe: {interlocks.get('radiation_safe', 'unknown')}")
                console.print(f"  Cooling OK: {interlocks.get('cooling_ok', 'unknown')}")
                console.print(f"  Power OK: {interlocks.get('power_ok', 'unknown')}")
                raise typer.Exit(1)
            console.print("[green]✓ All safety interlocks satisfied[/green]")
            progress.remove_task(task)
            
            # Step 2b: Check if system is locked and needs calibration
            current_state = status.get('state', '')
            if current_state == 'SAFE':
                # Check if it's actually LOCKED from components
                for comp in status.get('components', []):
                    if comp.get('name') == 'Safety State Machine' and 'LOCKED' in comp.get('detail', ''):
                        task = progress.add_task("System locked - calibrating detector...", total=None)
                        calib_result = client.calibrate_detector(user)
                        if "error" in calib_result:
                            console.print(f"[red]✗ Calibration failed: {calib_result['error']}[/red]")
                            raise typer.Exit(1)
                        console.print("[green]✓ Detector calibrated[/green]")
                        progress.remove_task(task)
                        time.sleep(1)  # Brief wait after calibration
                        break
            
            # Step 3: Get device state
            task = progress.add_task("Checking device state...", total=None)
            device_state = client.get_device_state()
            if "error" in device_state:
                console.print(f"[yellow]⚠ Could not get device state: {device_state['error']}[/yellow]")
            else:
                detector = device_state.get('detector', {})
                console.print(f"[green]✓ Device state retrieved[/green]")
                console.print(f"  Detector powered: {detector.get('powered', False)}")
                console.print(f"  Detector status: {detector.get('status', 'unknown')}")
            progress.remove_task(task)
            
            # Step 4: Power on detector if needed
            if not device_state.get('detector', {}).get('powered', False):
                task = progress.add_task("Powering on detector...", total=None)
                power_result = client.power_device("detector", True, user)
                if "error" in power_result:
                    console.print(f"[red]✗ Failed to power on detector: {power_result['error']}[/red]")
                    raise typer.Exit(1)
                console.print("[green]✓ Detector powered on[/green]")
                progress.remove_task(task)
                
                # Wait for detector to stabilize
                task = progress.add_task("Waiting for detector to stabilize...", total=None)
                time.sleep(2)
                progress.remove_task(task)
            else:
                console.print("[green]✓ Detector already powered[/green]")
            
            # Step 5: Start measurement
            task = progress.add_task(f"Starting {duration}s measurement...", total=None)
            exposure_time_ms = duration * 1000
            measure_result = client.start_measurement(exposure_time_ms, user)
            if "error" in measure_result:
                console.print(f"[red]✗ Failed to start measurement: {measure_result['error']}[/red]")
                raise typer.Exit(1)
            console.print(f"[green]✓ Measurement started ({duration}s)[/green]")
            progress.remove_task(task)
            
            # Step 6: Monitor measurement progress
            task = progress.add_task(f"Acquiring data...", total=duration)
            for i in range(duration):
                time.sleep(1)
                progress.update(task, advance=1)
            progress.remove_task(task)
            
            # Step 7: Get measurement result
            task = progress.add_task("Retrieving measurement result...", total=None)
            result = client.get_measurement_result()
            if "error" in result:
                console.print(f"[yellow]⚠ Could not retrieve result: {result['error']}[/yellow]")
            elif result.get('has_result', False):
                console.print("[green]✓ Measurement completed successfully[/green]")
                console.print(f"  Exposure time: {result.get('exposure_time_ms', 0)}ms")
                console.print(f"  Data size: {result.get('data_size', 0)} bytes")
                if result.get('data_path'):
                    console.print(f"  Data path: {result['data_path']}")
                console.print(f"  Detector temp: {result.get('detector_temp', 0):.1f}°C")
            else:
                console.print("[yellow]⚠ No measurement result available yet[/yellow]")
            progress.remove_task(task)
            
        console.print("\n[bold green]✓ Measurement workflow completed[/bold green]")
        
    except Exception as e:
        console.print(f"\n[red]✗ Workflow failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


# Engineer certificate management
cert_app = typer.Typer(help="Engineer certificate management")
app.add_typer(cert_app, name="cert")


@cert_app.command("generate")
def cert_generate(
    engineer_id: str = typer.Option(..., help="Engineer ID (e.g., ENG001)"),
    device_uuid: str = typer.Option(..., help="Target device UUID"),
    validity_days: int = typer.Option(1, help="Certificate validity in days"),
    key_type: str = typer.Option("ecdsa", help="Key type: ecdsa or rsa"),
    cert_center_dir: Path = typer.Option(
        "C:\\dev\\Omniscan\\omniscan-certificate-center",
        help="Path to omniscan-certificate-center directory"
    ),
):
    """Generate a new engineer certificate for device access"""
    try:
        manager = CertificateManager(cert_center_dir)
        cert_path, key_path = manager.create_engineer_certificate(
            engineer_id=engineer_id,
            device_uuid=device_uuid,
            validity_days=validity_days,
            key_type=key_type,
        )
        
        print(f"\n✅ Engineer certificate created successfully!")
        print(f"\n📋 Certificate Details:")
        info = manager.get_certificate_info(cert_path)
        print(f"  Engineer: {info['common_name']}")
        print(f"  Device Scope: {', '.join(info['san_uris'])}")
        print(f"  Valid From: {info['not_valid_before']}")
        print(f"  Valid Until: {info['not_valid_after']}")
        print(f"  Status: {'✅ Valid' if info['is_valid'] else '❌ Invalid'}")
        
        print(f"\n📁 Files:")
        print(f"  Certificate: {cert_path}")
        print(f"  Private Key: {key_path}")
        
        print(f"\n🔐 Usage:")
        print(f"  omni-orch --cert {cert_path} --key {key_path} status")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise typer.Exit(1)


@cert_app.command("list")
def cert_list(
    cert_center_dir: Path = typer.Option(
        "C:\\dev\\Omniscan\\omniscan-certificate-center",
        help="Path to omniscan-certificate-center directory"
    ),
):
    """List all engineer certificates"""
    try:
        manager = CertificateManager(cert_center_dir)
        certs = manager.list_engineer_certificates()
        
        if not certs:
            print("No engineer certificates found.")
            return
        
        print(f"\n📋 Engineer Certificates ({len(certs)} found):\n")
        for cert in certs:
            status = "✅ Valid" if cert['is_valid'] else "❌ Expired"
            print(f"  {status} {cert['common_name']}")
            print(f"    Valid: {cert['not_valid_before']} to {cert['not_valid_after']}")
            print(f"    Scope: {', '.join(cert['san_uris'])}")
            print(f"    File: {cert['file_path']}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise typer.Exit(1)


@cert_app.command("info")
def cert_info(
    cert_path: Path = typer.Option(..., exists=True, help="Path to certificate file"),
):
    """Show detailed information about a certificate"""
    try:
        manager = CertificateManager(Path("."))
        info = manager.get_certificate_info(cert_path)
        
        print(f"\n📋 Certificate Information:")
        print(f"  Common Name: {info['common_name']}")
        print(f"  Subject: {info['subject']}")
        print(f"  Issuer: {info['issuer']}")
        print(f"  Serial Number: {info['serial_number']}")
        print(f"  Valid From: {info['not_valid_before']}")
        print(f"  Valid Until: {info['not_valid_after']}")
        print(f"  Status: {'✅ Valid' if info['is_valid'] else '❌ Invalid/Expired'}")
        
        if info['san_uris']:
            print(f"  Device Scope (SAN URIs):")
            for uri in info['san_uris']:
                print(f"    - {uri}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise typer.Exit(1)


# Dev utilities for the USB stub
usb_app = typer.Typer(help="Dev utilities for maintenance USB stub")
app.add_typer(usb_app, name="usb-dev")


@usb_app.command("init")
def usb_dev_init(
    path: Path = typer.Option(..., help="Directory representing the USB"),
    server_uuid: str = typer.Option(..., help="Target server UUID"),
    common_name: Optional[str] = typer.Option(None, help="Certificate CN override"),
    days: int = typer.Option(30, help="Certificate validity days"),
):
    stub = MaintenanceUSBStub.create_dev_stub(path, server_uuid=server_uuid, common_name=common_name, days_valid=days)
    ident = stub.load()
    print({
        "path": str(path),
        "fingerprint_sha256": ident.fingerprint_sha256,
        "metadata": ident.metadata,
    })


# Command Discovery commands
@app.command("server-capabilities")
def server_capabilities(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Get server capabilities and version information."""
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.get_server_capabilities()
    print(json.dumps(data, indent=2, ensure_ascii=False))


@app.command("list-commands")
def list_server_commands(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
    service: Optional[str] = typer.Option(None, help="Filter by service name"),
):
    """List all available commands from the server."""
    from rich.console import Console
    from rich.table import Table
    
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.list_commands()
    
    if "error" in data:
        print(f"Error: {data['error']}", file=sys.stderr)
        raise typer.Exit(1)
    
    # Filter by service if specified
    commands = data.get("commands", [])
    if service:
        commands = [cmd for cmd in commands if cmd["service_name"] == service]
    
    # Pretty print
    console = Console()
    table = Table(title=f"Available Commands{f' in {service}' if service else ''}")
    
    table.add_column("Service", style="cyan")
    table.add_column("Command", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("Safety", style="red")
    
    for cmd in commands:
        safety = ", ".join(cmd.get("safety_requirements", [])) or "None"
        table.add_row(
            cmd["service_name"],
            cmd["command_name"],
            cmd["description"][:60] + "..." if len(cmd["description"]) > 60 else cmd["description"],
            safety[:30] + "..." if len(safety) > 30 else safety,
        )
    
    console.print(table)
    console.print(f"\nTotal commands: {len(commands)}")
    console.print(f"Server version: {data['server_info']['server_version']}")
    console.print(f"Protocol version: {data['server_info']['protocol_version']}")


@app.command("validate-compatibility")
def validate_server_compatibility(
    server: str = typer.Option("localhost:50051", help="gRPC server address"),
    usb_path: Optional[Path] = typer.Option(None, help="Path to maintenance USB stub directory"),
    cert: Optional[Path] = typer.Option(None, help="Path to client certificate PEM"),
    key: Optional[Path] = typer.Option(None, help="Path to client private key PEM"),
    ca_cert: Optional[Path] = typer.Option(None, help="Path to server CA bundle (PEM)"),
):
    """Validate compatibility between orchestrator and server."""
    from rich.console import Console
    console = Console()
    
    client = _build_client(server, usb_path, cert, key, ca_cert)
    data = client.validate_compatibility()
    
    if "error" in data:
        print(f"❌ Error: {data['error']}", file=sys.stderr)
        raise typer.Exit(1)
    
    if data["compatible"]:
        console.print("✅ [green]Compatible[/green]")
        console.print(f"Message: {data['message']}")
    else:
        console.print("❌ [red]Incompatible[/red]")
        console.print(f"Message: {data['message']}")
        
        if data["missing_commands"]:
            console.print("\n[yellow]Missing commands:[/yellow]")
            for cmd in data["missing_commands"]:
                console.print(f"  - {cmd}")
        
        if data["version_warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in data["version_warnings"]:
                console.print(f"  - {warning}")
    
    raise typer.Exit(0 if data["compatible"] else 1)
