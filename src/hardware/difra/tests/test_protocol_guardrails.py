from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO_ROOT / script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


def test_protocol_sync_script_passes() -> None:
    result = _run("scripts/check_protocol_sync.sh")
    assert result.returncode == 0, result.stdout + result.stderr


def test_difra_branch_parity_script_passes() -> None:
    result = _run("scripts/check_difra_branch_parity.sh")
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_command_schema_files_exist() -> None:
    cmd_dir = REPO_ROOT / "src/hardware/protocol/commands/v1"
    expected = {
        "initialize_detector.toml",
        "initialize_motion.toml",
        "get_state.toml",
        "move_to.toml",
        "home.toml",
        "start_exposure.toml",
        "pause.toml",
        "resume.toml",
        "stop.toml",
        "abort.toml",
    }
    found = {p.name for p in cmd_dir.glob("*.toml")}
    assert expected.issubset(found)
