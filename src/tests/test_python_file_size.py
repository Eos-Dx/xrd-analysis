import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "scripts" / "check_python_file_size.py"


def _load_checker_module():
    """Load the script as a module so boundary behavior needs no subprocess."""
    spec = importlib.util.spec_from_file_location(
        "python_file_size_checker", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_python_files_stay_within_configured_size_budget():
    result = subprocess.run(
        [sys.executable, "scripts/check_python_file_size.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_oversized_files_rejects_default_limit(tmp_path):
    checker = _load_checker_module()
    source = tmp_path / "new_module.py"
    source.write_text("pass\n" * (checker.DEFAULT_MAX_LINES + 1), encoding="utf-8")

    assert checker.oversized_files(tmp_path) == [
        (source, checker.DEFAULT_MAX_LINES + 1, checker.DEFAULT_MAX_LINES)
    ]


def test_temporary_exemption_accepts_boundary_and_rejects_growth(tmp_path, monkeypatch):
    checker = _load_checker_module()
    monkeypatch.setattr(checker, "TEMPORARY_EXEMPTIONS", {"legacy.py": 3})
    source = tmp_path / "legacy.py"

    source.write_text("pass\n" * 3, encoding="utf-8")
    assert checker.oversized_files(tmp_path) == []

    source.write_text("pass\n" * 4, encoding="utf-8")
    assert checker.oversized_files(tmp_path) == [(source, 4, 3)]


def test_size_checker_reports_actionable_path_count_and_limit(
    tmp_path, monkeypatch, capsys
):
    checker = _load_checker_module()
    source = tmp_path / "too_large.py"
    source.write_text("pass\n" * (checker.DEFAULT_MAX_LINES + 1), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["check_python_file_size.py", "--package-root", str(tmp_path)]
    )

    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "FAIL: Python source files exceed the staged size budget:" in output
    assert f"{source}: {checker.DEFAULT_MAX_LINES + 1} lines" in output
    assert f"limit {checker.DEFAULT_MAX_LINES}" in output
