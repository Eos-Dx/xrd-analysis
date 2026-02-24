import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hardware.difra.gui.operator_manager import (  # noqa: E402
    DEFAULT_MODIFICATION_PASSWORD_HASH,
    OperatorManager,
)


def test_operator_manager_bootstraps_hash_in_config(tmp_path):
    config_path = Path(tmp_path) / "operators.json"

    manager = OperatorManager(config_path=config_path)

    assert manager.operator_modify_password_hash == DEFAULT_MODIFICATION_PASSWORD_HASH
    assert manager.verify_modify_password("Ulster2026!") is True
    assert manager.verify_modify_password("wrong-password") is False

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "operator_modify_password_hash" in data
    assert data["operator_modify_password_hash"] == DEFAULT_MODIFICATION_PASSWORD_HASH
    assert data["operator_modify_password_hash"] != "Ulster2026!"


def test_operator_manager_uses_configured_hash(tmp_path):
    config_path = Path(tmp_path) / "operators.json"
    config_path.write_text(
        json.dumps(
            {
                "operators": {
                    "sad1": {
                        "name": "Sergey",
                        "surname": "DENISOV",
                        "email": "s@m.c",
                        "phone": "",
                        "institution": "",
                    }
                },
                "current_operator_id": "sad1",
                "operator_modify_password_hash": "2bb80d537b1da3e38bd30361aa855686bde0eacd"
                "7162fef6a25fe97bf527a25b",  # sha256("secret")
            }
        ),
        encoding="utf-8",
    )

    manager = OperatorManager(config_path=config_path)
    assert manager.verify_modify_password("secret") is True
    assert manager.verify_modify_password("Ulster2026!") is False
