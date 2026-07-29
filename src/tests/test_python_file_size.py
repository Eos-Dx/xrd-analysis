import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_analysis_python_files_stay_within_configured_size_budget():
    result = subprocess.run(
        [sys.executable, "scripts/check_python_file_size.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
