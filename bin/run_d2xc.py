#!/usr/bin/env python3
"""
D2XC Software Launcher
Launches the D2XC GUI application with proper environment setup.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Main launcher function."""
    # Determine repository root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Path to configuration file
    config_path = (
        repo_root
        / "src"
        / "hardware"
        / "difra"
        / "resources"
        / "config"
        / "global.json"
    )

    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        return 1

    # Read conda environment from config
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        conda_env = config.get("conda")
        if not conda_env:
            print(f"[ERROR] 'conda' field not found in {config_path}")
            return 1
    except Exception as e:
        print(f"[ERROR] Failed to read config: {e}")
        return 1

    # Check if conda is available
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "[ERROR] 'conda' was not found. Please run from an Anaconda/Miniconda environment."
        )
        return 1

    # Path to main application
    app_path = repo_root / "src" / "hardware" / "difra" / "gui" / "main_app.py"

    if not app_path.exists():
        print(f"[ERROR] Application file not found: {app_path}")
        return 1

    print("Starting D2XC software...")
    print(f"Using conda environment: {conda_env}")
    print(f"Repository root: {repo_root}")
    print(f"Application path: {app_path}")

    # Launch the application
    try:
        cmd = ["conda", "run", "-n", conda_env, "python", str(app_path)] + sys.argv[1:]
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to launch application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
