#!/usr/bin/env python3
"""
Simple D2XC/DiFRA Launcher
This script launches the D2XC GUI application.
Can be compiled to an executable using PyInstaller or similar tools.
"""

import json
import logging
import logging.handlers
import os
import subprocess
import sys
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox


def setup_launcher_logging():
    """Setup comprehensive logging for launcher with both file and console output."""
    # Determine log file location
    if sys.platform.startswith("win"):
        log_dir = Path.home() / "AppData" / "Local" / "Ulster" / "launcher_logs"
    else:
        log_dir = Path.home() / ".local" / "state" / "Ulster" / "launcher_logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"launcher_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler with detailed formatting
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Launcher logging initialized. Log file: {log_file}")
    logger.info(f"Log directory: {log_dir}")
    
    return logger, log_file


def show_error(title, message, logger=None):
    """Show error message in a GUI dialog and log it."""
    if logger:
        logger.error(f"{title}: {message}")
    
    # Always print to console as well
    print(f"\n{'='*80}")
    print(f"ERROR: {title}")
    print(f"{'-'*80}")
    print(message)
    print(f"{'='*80}\n")
    
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror(title, message)
        root.destroy()
    except Exception as e:
        if logger:
            logger.error(f"Failed to show GUI error dialog: {e}")
        print(f"Failed to show GUI error dialog: {e}")


def main():
    """Main launcher function."""
    logger = None
    try:
        # Setup logging first
        logger, log_file = setup_launcher_logging()
        logger.info("="*80)
        logger.info("DiFRA Launcher Starting")
        logger.info("="*80)
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Determine repository root (3 levels up: bin -> hardware -> src -> root)
        script_dir = Path(__file__).resolve().parent
        logger.info(f"Script directory: {script_dir}")
        
        repo_root = script_dir.parent.parent.parent
        logger.info(f"Repository root: {repo_root}")

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
        logger.info(f"Config path: {config_path}")

        if not config_path.exists():
            error_msg = (
                f"Configuration file not found:\n{config_path}\n\n"
                f"Expected location: {config_path}\n"
                f"Repository root: {repo_root}\n\n"
                f"Please ensure the application is properly installed."
            )
            logger.error(f"Configuration file not found: {config_path}")
            logger.error(f"Checked path exists: {config_path.exists()}")
            logger.error(f"Parent directory exists: {config_path.parent.exists()}")
            if config_path.parent.exists():
                logger.error(f"Contents of config directory: {list(config_path.parent.glob('*'))}")
            show_error("Configuration Error", error_msg, logger)
            return 1

        # Read conda environment from config
        logger.info("Reading configuration file...")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.debug(f"Configuration loaded: {json.dumps(config, indent=2)}")
            
            conda_env = config.get("conda")
            if not conda_env:
                error_msg = (
                    f"'conda' field not found in configuration file:\n{config_path}\n\n"
                    f"Available fields: {list(config.keys())}\n\n"
                    f"Please ensure the configuration file contains a 'conda' field "
                    f"specifying the conda environment name."
                )
                logger.error(f"'conda' field not found in config")
                logger.error(f"Available config fields: {list(config.keys())}")
                show_error("Configuration Error", error_msg, logger)
                return 1
            
            logger.info(f"Conda environment from config: {conda_env}")
        except json.JSONDecodeError as e:
            error_msg = (
                f"Failed to parse configuration file (invalid JSON):\n{config_path}\n\n"
                f"Error: {e}\n\n"
                f"Line {e.lineno}, Column {e.colno}\n\n"
                f"Please check the configuration file format."
            )
            logger.error(f"JSON decode error: {e}", exc_info=True)
            show_error("Configuration Error", error_msg, logger)
            return 1
        except Exception as e:
            error_msg = (
                f"Failed to read configuration file:\n{config_path}\n\n"
                f"Error: {type(e).__name__}: {e}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            logger.error(f"Failed to read config: {e}", exc_info=True)
            show_error("Configuration Error", error_msg, logger)
            return 1

        # Check if conda is available
        logger.info("Checking conda availability...")
        try:
            result = subprocess.run(
                ["conda", "--version"], 
                check=True, 
                capture_output=True, 
                timeout=10,
                text=True
            )
            logger.info(f"Conda version: {result.stdout.strip()}")
            
            # Check if the specified conda environment exists
            logger.info(f"Checking if conda environment '{conda_env}' exists...")
            env_list = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                timeout=10,
                text=True
            )
            logger.debug(f"Available conda environments:\n{env_list.stdout}")
            
            if conda_env not in env_list.stdout:
                error_msg = (
                    f"Conda environment '{conda_env}' not found.\n\n"
                    f"Available environments:\n{env_list.stdout}\n\n"
                    f"Please create the environment or update the configuration file."
                )
                logger.error(f"Conda environment '{conda_env}' not found")
                show_error("Conda Environment Error", error_msg, logger)
                return 1
            
            logger.info(f"Conda environment '{conda_env}' found")
            
        except FileNotFoundError as e:
            error_msg = (
                "'conda' command was not found.\n\n"
                "Please ensure:\n"
                "• Anaconda/Miniconda is installed\n"
                "• Conda is available in your PATH\n"
                "• Or run this from an Anaconda Prompt\n\n"
                f"System PATH:\n{os.environ.get('PATH', 'Not set')}"
            )
            logger.error(f"Conda not found: {e}", exc_info=True)
            logger.error(f"PATH: {os.environ.get('PATH', 'Not set')}")
            show_error("Conda Error", error_msg, logger)
            return 1
        except subprocess.TimeoutExpired as e:
            error_msg = (
                "Conda command timed out (10 seconds).\n\n"
                "Conda may be unresponsive or not working properly.\n"
                "Please check your conda installation."
            )
            logger.error(f"Conda check timed out: {e}", exc_info=True)
            show_error("Conda Error", error_msg, logger)
            return 1
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Conda command failed with error code {e.returncode}.\n\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}\n\n"
                "Conda may not be working properly. Please check your installation."
            )
            logger.error(f"Conda check failed: {e}", exc_info=True)
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            show_error("Conda Error", error_msg, logger)
            return 1

        # Path to main application
        app_path = repo_root / "src" / "hardware" / "difra" / "gui" / "main_app.py"
        logger.info(f"Application path: {app_path}")

        if not app_path.exists():
            error_msg = (
                f"Application file not found:\n{app_path}\n\n"
                f"Expected location: {app_path}\n"
                f"Repository root: {repo_root}\n\n"
                f"Please ensure the application is properly installed."
            )
            logger.error(f"Application file not found: {app_path}")
            logger.error(f"Checked path exists: {app_path.exists()}")
            logger.error(f"Parent directory exists: {app_path.parent.exists()}")
            if app_path.parent.exists():
                logger.error(f"Contents of gui directory: {list(app_path.parent.glob('*'))}")
            show_error("Application Error", error_msg, logger)
            return 1

        # Launch the application
        logger.info("Launching application...")
        try:
            cmd = ["conda", "run", "-n", conda_env, "python", str(app_path)] + sys.argv[
                1:
            ]
            logger.info(f"Launch command: {' '.join(cmd)}")
            logger.info(f"Additional arguments: {sys.argv[1:]}")
            
            # Launch without waiting (detached process)
            if os.name == "nt":  # Windows
                logger.info("Launching on Windows with CREATE_NEW_CONSOLE flag")
                process = subprocess.Popen(
                    cmd, 
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"Process started with PID: {process.pid}")
            else:
                logger.info("Launching on Unix-like system")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"Process started with PID: {process.pid}")
            
            logger.info("="*80)
            logger.info("Application launched successfully")
            logger.info(f"Log file location: {log_file}")
            logger.info("="*80)
            return 0
            
        except Exception as e:
            error_msg = (
                f"Failed to launch application.\n\n"
                f"Error: {type(e).__name__}: {e}\n\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            logger.error(f"Failed to launch application: {e}", exc_info=True)
            show_error("Launch Error", error_msg, logger)
            return 1

    except Exception as e:
        error_msg = (
            f"An unexpected error occurred in the launcher.\n\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        if logger:
            logger.critical(f"Unexpected error: {e}", exc_info=True)
        show_error("Unexpected Error", error_msg, logger)
        return 1


if __name__ == "__main__":
    sys.exit(main())
