from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import typer
import httpx
from rich import print

app = typer.Typer(help=(
    "CLI client for Omniscan REST API (duplicates server endpoints)\n\n"
    "Quick usage: readiness\n"
    "  omni-rest server readiness --base-url http://localhost:8081 [--service Acquisition]\n\n"
    "Full reference: see CLI_REFERENCE.md in the repository root.\n"
))

# Default base URL (can override with --base-url)
DEFAULT_BASE = "http://localhost:8081"
SESSION_CACHE = Path.home() / ".omniscan_rest_session"


def load_session() -> Optional[str]:
    try:
        if SESSION_CACHE.exists():
            sid = SESSION_CACHE.read_text(encoding="utf-8").strip()
            return sid or None
    except Exception:
        return None
    return None


def save_session(session_id: str):
    try:
        SESSION_CACHE.write_text(session_id, encoding="utf-8")
    except Exception as e:
        print(f"[yellow]Warning: failed to cache session: {e}[/yellow]")


def auth_headers(session: Optional[str]) -> Dict[str, str]:
    sid = session or load_session()
    return {"x-session-id": sid} if sid else {}


def client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=15.0)


# --------------------------- Auth ---------------------------
auth_app = typer.Typer(help="Authentication endpoints")
app.add_typer(auth_app, name="auth")


@auth_app.command("login")
def login(
    username: str = typer.Option(..., "-u", "--username"),
    password: str = typer.Option(..., "-p", "--password"),
    base_url: str = typer.Option(DEFAULT_BASE, "-b", "--base-url"),
    cache: bool = typer.Option(True, help="Cache session locally for subsequent commands"),
):
    with client(base_url) as c:
        r = c.post("/api/auth/login", json={"username": username, "password": password})
        if r.status_code != 200:
            typer.echo(r.text)
            raise typer.Exit(code=1)
        data = r.json()
        print(json.dumps(data, indent=2))
        if data.get("success") and data.get("session_id") and cache:
            save_session(data["session_id"])


@auth_app.command("logout")
def logout(
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = typer.Option(None, help="Override session ID"),
):
    hdrs = auth_headers(session)
    if not hdrs:
        print("[red]No session available. Login first or pass --session[/red]")
        raise typer.Exit(1)
    with client(base_url) as c:
        r = c.post("/api/auth/logout", headers=hdrs)
        print(r.text if r.headers.get("content-type","" ).startswith("text/") else json.dumps(r.json(), indent=2))


# --------------------------- System ---------------------------
system_app = typer.Typer(help="System/health endpoints")
app.add_typer(system_app, name="system")


@system_app.command("health")
def system_health(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.get("/api/health", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@system_app.command("state")
def system_state(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.get("/api/state", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


# --------------------------- Patients ---------------------------
patients_app = typer.Typer(help="Patient management")
app.add_typer(patients_app, name="patients")


@patients_app.command("create")
def patients_create(
    first_name: str = typer.Option(...),
    last_name: str = typer.Option(...),
    date_of_birth: str = typer.Option(..., help="ISO date"),
    mrn: str = typer.Option(..., "--medical-record-number"),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    payload = {
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": date_of_birth,
        "medical_record_number": mrn,
    }
    with client(base_url) as c:
        r = c.post("/api/patients", headers=auth_headers(session), json=payload)
        print(json.dumps(r.json(), indent=2))


@patients_app.command("search")
def patients_search(
    mrn: str = typer.Option(...),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    with client(base_url) as c:
        r = c.get("/api/patients/search", headers=auth_headers(session), params={"mrn": mrn})
        print(json.dumps(r.json(), indent=2))


@patients_app.command("get")
def patients_get(
    patient_id: str = typer.Option(...),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    with client(base_url) as c:
        r = c.get(f"/api/patients/{patient_id}", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


# --------------------------- Measurements ---------------------------
meas_app = typer.Typer(help="Measurement endpoints")
app.add_typer(meas_app, name="measurements")


@meas_app.command("start")
def meas_start(
    patient_id: str = typer.Option(...),
    exposure_ms: int = typer.Option(..., help="Exposure time in ms"),
    sample_id: Optional[str] = typer.Option(None),
    notes: Optional[str] = typer.Option(None),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    payload: Dict[str, Any] = {
        "patient_id": patient_id,
        "exposure_duration": exposure_ms,
    }
    if sample_id:
        payload["sample_id"] = sample_id
    if notes:
        payload["notes"] = notes
    with client(base_url) as c:
        r = c.post("/api/measurements/start", headers=auth_headers(session), json=payload)
        print(json.dumps(r.json(), indent=2))


@meas_app.command("stop")
def meas_stop(
    run_id: str = typer.Option(...),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    with client(base_url) as c:
        r = c.post(f"/api/measurements/{run_id}/stop", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@meas_app.command("abort")
def meas_abort(
    run_id: str = typer.Option(...),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    with client(base_url) as c:
        r = c.post(f"/api/measurements/{run_id}/abort", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@meas_app.command("history")
def meas_history(
    limit: int = typer.Option(50),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    with client(base_url) as c:
        r = c.get("/api/measurements", headers=auth_headers(session), params={"limit": limit})
        print(json.dumps(r.json(), indent=2))


# --------------------------- Calibration ---------------------------
cal_app = typer.Typer(help="Calibration endpoints")
app.add_typer(cal_app, name="calibration")


@cal_app.command("start")
def cal_start(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.post("/api/calibration/start", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@cal_app.command("status")
def cal_status(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.get("/api/calibration/status", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@cal_app.command("latest")
def cal_latest(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.get("/api/calibration/latest", headers=auth_headers(session))
        # latest may return text JSON via JSONResponse
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text)

@cal_app.command("history")
def cal_history(
    hours: int = typer.Option(24, help="Look back window in hours"),
    limit: int = typer.Option(50, help="Max records to return"),
    base_url: str = typer.Option(DEFAULT_BASE),
    session: Optional[str] = None,
):
    with client(base_url) as c:
        r = c.get("/api/calibration/history", headers=auth_headers(session), params={"hours": hours, "limit": limit})
        print(json.dumps(r.json(), indent=2))


# --------------------------- GPIO ---------------------------
gpio_app = typer.Typer(help="GPIO & safety endpoints")
app.add_typer(gpio_app, name="gpio")


@gpio_app.command("state")
def gpio_state(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.get("/api/gpio/state", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@gpio_app.command("enable-button")
def gpio_enable_button(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.get("/api/gpio/enable-button", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


# --------------------------- Hardware ---------------------------
hw_app = typer.Typer(help="Hardware control")
app.add_typer(hw_app, name="hardware")


@hw_app.command("init")
def hw_init(device: str = typer.Option(..., help="detector or motion"), base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.post(f"/api/hardware/{device}/init", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


@hw_app.command("stop")
def hw_stop(device: str = typer.Option(..., help="detector or motion"), base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.post(f"/api/hardware/{device}/stop", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


# --------------------------- Motion ---------------------------
motion_app = typer.Typer(help="Motion control")
app.add_typer(motion_app, name="motion")


@motion_app.command("stop")
def motion_stop(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None):
    with client(base_url) as c:
        r = c.post("/api/motion/stop", headers=auth_headers(session))
        print(json.dumps(r.json(), indent=2))


# --------------------------- Server discovery ---------------------------
server_app = typer.Typer(help="Server discovery & readiness")
app.add_typer(server_app, name="server")


@server_app.command("capabilities")
def server_capabilities(base_url: str = typer.Option(DEFAULT_BASE)):
    with client(base_url) as c:
        r = c.get("/api/v1/server/capabilities")
        print(json.dumps(r.json(), indent=2))


@server_app.command("commands")
def server_commands(base_url: str = typer.Option(DEFAULT_BASE)):
    with client(base_url) as c:
        r = c.get("/api/v1/server/commands")
        print(json.dumps(r.json(), indent=2))


@server_app.command("readiness")
def server_commands_readiness(base_url: str = typer.Option(DEFAULT_BASE), session: Optional[str] = None, service: Optional[str] = typer.Option(None)):
    params = {"service": service} if service else None
    with client(base_url) as c:
        r = c.get("/api/v1/server/commands/readiness", headers=auth_headers(session), params=params)
        print(json.dumps(r.json(), indent=2))


@server_app.command("validate-compatibility")
def validate_compat(base_url: str = typer.Option(DEFAULT_BASE)):
    with client(base_url) as c:
        r = c.post("/api/v1/server/validate-compatibility")
        print(json.dumps(r.json(), indent=2))


# --------------------------- Debug ---------------------------
debug_app = typer.Typer(help="Debug endpoints (no auth)")
app.add_typer(debug_app, name="debug")


@debug_app.command("status")
def debug_status(base_url: str = typer.Option(DEFAULT_BASE)):
    with client(base_url) as c:
        r = c.get("/api/debug/status")
        print(json.dumps(r.json(), indent=2))


@debug_app.command("connection")
def debug_connection(base_url: str = typer.Option(DEFAULT_BASE)):
    with client(base_url) as c:
        r = c.get("/api/connection/status")
        print(json.dumps(r.json(), indent=2))


# --------------------------- Validation suite ---------------------------
@app.command("smoke")
def smoke(
    username: str = typer.Option(..., "-u"),
    password: str = typer.Option(..., "-p"),
    base_url: str = typer.Option(DEFAULT_BASE, "-b", "--base-url"),
):
    """Run a safe smoke test of key endpoints (non-destructive)."""
    results = []
    with client(base_url) as c:
        # Debug endpoints
        for path, method in [
            ("/api/debug/status", "GET"),
            ("/api/v1/server/capabilities", "GET"),
            ("/api/v1/server/commands", "GET"),
        ]:
            r = c.get(path) if method == "GET" else c.post(path)
            results.append({"path": path, "status": r.status_code})
        # Login
        r = c.post("/api/auth/login", json={"username": username, "password": password})
        if r.status_code == 200 and r.json().get("success"):
            sid = r.json().get("session_id")
        else:
            print(json.dumps({"login": r.text}, indent=2))
            raise typer.Exit(1)
        hdrs = {"x-session-id": sid}
        # Auth-required GETs
        for path in [
            "/api/state",
            "/api/health",
            "/api/gpio/state",
            "/api/gpio/enable-button",
            "/api/calibration/status",
            "/api/v1/server/commands/readiness",
        ]:
            r = c.get(path, headers=hdrs)
            results.append({"path": path, "status": r.status_code})
        # Logout
        r = c.post("/api/auth/logout", headers=hdrs)
        results.append({"path": "/api/auth/logout", "status": r.status_code})
    print(json.dumps({"results": results}, indent=2))
