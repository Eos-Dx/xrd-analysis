# Deprecated entrypoint, kept for backward compatibility.
# Please use main_app.py instead.
import sys
from pathlib import Path

# Compute path to the new entrypoint
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_app = project_root / "src" / "hardware" / "Ulster" / "gui" / "main_app.py"

if __name__ == "__main__":
    print("[WARNING] 'Ulster.py' is deprecated. Launching main_app.py...")
    # Execute main_app.py in the current interpreter
    code = compile(main_app.read_text(encoding="utf-8"), str(main_app), "exec")
    globals_dict = {"__name__": "__main__"}
    exec(code, globals_dict)
