#!/usr/bin/env python3
"""
Simple D2XC/EOSDxDc Launcher
This script launches the D2XC GUI application.
Can be compiled to an executable using PyInstaller or similar tools.
"""

import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


def show_error(title, message):
    """Show error message in a GUI dialog."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror(title, message)
        root.destroy()
    except:
        print(f"ERROR: {message}")


def main():
    """Main launcher function."""
    try:
        # Determine repository root (3 levels up: bin -> hardware -> src -> root)
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent.parent

        # Path to configuration file
        config_path = (
            repo_root
            / "src"
            / "hardware"
            / "eosdxdc"
            / "resources"
            / "config"
            / "global.json"
        )

        if not config_path.exists():
            show_error(
                "Configuration Error", f"Configuration file not found:\n{config_path}"
            )
            return 1

        # Read conda environment from config
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            conda_env = config.get("conda")
            if not conda_env:
                show_error(
                    "Configuration Error", f"'conda' field not found in {config_path}"
                )
                return 1
        except Exception as e:
            show_error("Configuration Error", f"Failed to read config: {e}")
            return 1

        # Check if conda is available
        try:
            subprocess.run(
                ["conda", "--version"], check=True, capture_output=True, timeout=10
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            show_error(
                "Conda Error",
                "'conda' was not found or not working properly.\n\n"
                "Please ensure:\n"
                "• Anaconda/Miniconda is installed\n"
                "• Conda is available in your PATH\n"
                "• Or run this from an Anaconda Prompt",
            )
            return 1

        # Path to main application
        app_path = repo_root / "src" / "hardware" / "eosdxdc" / "gui" / "main_app.py"

        if not app_path.exists():
            show_error("Application Error", f"Application file not found:\n{app_path}")
            return 1

        # Launch the application
        try:
            cmd = ["conda", "run", "-n", conda_env, "python", str(app_path)] + sys.argv[
                1:
            ]
            # Launch without waiting (detached process)
            if os.name == "nt":  # Windows
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd)
            return 0
        except Exception as e:
            show_error("Launch Error", f"Failed to launch application: {e}")
            return 1

    except Exception as e:
        show_error("Unexpected Error", f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
