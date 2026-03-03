from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_primary_pyproject_supports_python_313():
    pyproject = tomllib.loads(_read_text(ROOT / "pyproject.toml"))
    python_spec = pyproject["tool"]["poetry"]["dependencies"]["python"]
    assert python_spec == ">=3.11,<3.14"


def test_modern_pyproject_template_supports_python_313():
    pyproject = tomllib.loads(_read_text(ROOT / "pyproject-py311.toml"))
    python_spec = pyproject["tool"]["poetry"]["dependencies"]["python"]
    assert python_spec == ">=3.11,<3.14"


def test_setup_metadata_declares_same_python_range():
    setup_text = _read_text(ROOT / "setup.py")
    assert 'python_requires=">=3.11,<3.14"' in setup_text


def test_environment_file_targets_eosdx13_with_python_313():
    env_text = _read_text(ROOT / "environment.yml")
    assert "name: eosdx13" in env_text
    assert "  - python=3.13" in env_text


def test_docs_workflow_uses_python_313_environment():
    workflow_text = _read_text(ROOT / ".github" / "workflows" / "documentation.yml")
    assert "activate-environment: eosdx13" in workflow_text
    assert 'python-version: "3.13"' in workflow_text


def test_helper_scripts_point_modern_setup_to_python_313():
    switch_text = _read_text(ROOT / "switch-python-version.ps1")
    assert '[ValidateSet("37", "311", "313")]' in switch_text
    assert "conda activate eosdx13" in switch_text

    setup_text = _read_text(ROOT / "setup-project.ps1")
    assert 'Write-Host "2) Python 3.13 (Modern)"' in setup_text
    assert '$pyVersion = "3.13"' in setup_text


def test_modern_pyproject_versions_match_eosdx13_runtime():
    pyproject = tomllib.loads(_read_text(ROOT / "pyproject.toml"))
    deps = pyproject["tool"]["poetry"]["dependencies"]
    dev_deps = pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]

    expected_runtime = {
        "numpy": "1.26.4",
        "scipy": "1.17.0",
        "pandas": "3.0.0",
        "matplotlib": "3.10.8",
        "seaborn": "0.13.2",
        "joblib": "1.5.3",
        "h5py": "3.15.1",
        "scikit-image": "0.26.0",
        "scikit-learn": "1.8.0",
        "pyFAI": "2025.12.1",
        "requests": "2.32.5",
        "PyQt5": "5.15.11",
        "opencv-python": "4.13.0",
        "click": "8.3.1",
    }
    for name, version in expected_runtime.items():
        assert deps[name] == version

    expected_dev = {
        "pytest": "9.0.2",
        "black": "26.1.0",
        "flake8": "7.3.0",
        "isort": "7.0.0",
        "pre-commit": "4.5.1",
        "coverage": "7.13.4",
    }
    for name, version in expected_dev.items():
        assert dev_deps[name] == version


def test_environment_file_pins_match_installed_eosdx13_versions():
    env_text = _read_text(ROOT / "environment.yml")
    expected_lines = [
        "  - click=8.3.1",
        "  - coverage=7.13.4",
        "  - flake8=7.3.0",
        "  - h5py=3.15.1",
        "  - isort=7.0.0",
        "  - matplotlib=3.10.8",
        "  - pandas=3.0.0",
        "  - pre-commit=4.5.1",
        "  - pyFAI=2025.12.1",
        "  - pytest=9.0.2",
        "  - scikit-image=0.26.0",
        "  - scikit-learn=1.8.0",
        "  - scipy=1.17.0",
    ]
    for line in expected_lines:
        assert line in env_text
