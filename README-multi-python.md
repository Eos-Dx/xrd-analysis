# Multi-Python Version Support

This repository supports both Python 3.7 and Python 3.11 with separate Poetry lock files for each version.

## Files Structure

```
├── pyproject-py37.toml     # Poetry config for Python 3.7
├── poetry-py37.lock        # Lock file for Python 3.7
├── pyproject-py311.toml    # Poetry config for Python 3.11
├── poetry-py311.lock       # Lock file for Python 3.11
├── switch-python-version.ps1  # Helper script to switch versions
├── pyproject.toml          # Active config (symlinked/copied)
└── poetry.lock            # Active lock (symlinked/copied)
```

## Installation on New Machine

### Method 1: Auto-Setup (Easiest) 🚀

After cloning the repository:

```powershell
# Clone and enter directory
git clone <your-repo-url>
cd xrd-analysis

# Activate your desired Python environment
conda activate ulster37    # for Python 3.7
# OR
conda activate eosdx       # for Python 3.11

# Run auto-setup (detects Python version and configures automatically)
.\setup-project.ps1
```

This script will:
- ✅ Detect your Python version
- ✅ Select the appropriate lock file automatically
- ✅ Install Poetry if needed
- ✅ Install all dependencies
- ✅ Verify installation

### Method 2: Manual Selection

```powershell
# Clone repository
git clone <your-repo-url>
cd xrd-analysis

# Choose your configuration:

# For Python 3.7:
.\switch-python-version.ps1 37
poetry install

# OR for Python 3.11:
.\switch-python-version.ps1 311
poetry install
```

### Method 3: Direct Copy (Advanced)

```powershell
# For Python 3.7:
Copy-Item "pyproject-py37.toml" "pyproject.toml" -Force
Copy-Item "poetry-py37.lock" "poetry.lock" -Force
poetry install

# For Python 3.11:
Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
Copy-Item "poetry-py311.lock" "poetry.lock" -Force
poetry install
```

## Quick Setup

### For Python 3.7 (d2xc_dev compatibility):
```powershell
# Create and activate conda environment
conda create -n ulster37 python=3.7.16
conda activate ulster37

# Switch to Python 3.7 configuration
.\switch-python-version.ps1 37

# Install dependencies
poetry env use python
poetry install
```

### For Python 3.11 (modern development):
```powershell
# Activate your Python 3.11 environment
conda activate eosdx

# Switch to Python 3.11 configuration
.\switch-python-version.ps1 311

# Install dependencies
poetry env use python
poetry install
```

## Manual Switching (Alternative)

If you prefer to switch manually:

```powershell
# For Python 3.7:
Copy-Item "pyproject-py37.toml" "pyproject.toml" -Force
Copy-Item "poetry-py37.lock" "poetry.lock" -Force

# For Python 3.11:
Copy-Item "pyproject-py311.toml" "pyproject.toml" -Force
Copy-Item "poetry-py311.lock" "poetry.lock" -Force
```

## Key Differences

### Python 3.7 (Legacy/d2xc compatibility):
- Pinned to older package versions with guaranteed wheels
- Compatible with legacy hardware control systems
- Optimized for stability on older systems

### Python 3.11 (Modern):
- Latest package versions with performance improvements
- Modern development tools (better black, pytest, etc.)
- Enhanced typing and debugging support

## Updating Lock Files

When adding new dependencies:

1. Switch to the target Python version
2. Activate the corresponding conda environment
3. Edit the appropriate `pyproject-pyXXX.toml` file
4. Copy it to `pyproject.toml`
5. Run `poetry lock` to update the lock file
6. Copy the updated `poetry.lock` to `poetry-pyXXX.lock`

## CI/CD Support

You can test both Python versions in CI by using matrix builds:

```yaml
strategy:
  matrix:
    python-version: ["3.7", "3.11"]
    include:
      - python-version: "3.7"
        config-file: "pyproject-py37.toml"
      - python-version: "3.11"
        config-file: "pyproject-py311.toml"
```
