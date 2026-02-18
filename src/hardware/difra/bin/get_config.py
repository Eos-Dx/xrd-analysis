import json
import sys
from pathlib import Path

global_cfg = Path(sys.argv[1])
main_cfg = Path(sys.argv[2])
chosen = main_cfg

try:
    if global_cfg.exists():
        data = json.loads(global_cfg.read_text())
        setup = str(data.get("default_setup") or "").strip()
        if setup:
            setup_path = global_cfg.parent / "setups" / f"{setup}.json"
            if setup_path.exists():
                chosen = setup_path
except Exception:
    pass

print(chosen)
