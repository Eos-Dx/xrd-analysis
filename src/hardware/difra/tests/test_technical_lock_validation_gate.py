"""GUI lock workflow tests: enforce validation before technical lock."""

import sys
import tempfile
from pathlib import Path

import h5py

# Add project src to path
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import order matters because technical mixins reference each other.
import hardware.difra.gui.main_window_ext.technical.h5_management_mixin as _h5_mixin  # noqa: F401
import hardware.difra.gui.main_window_ext.technical.h5_management_locking_mixin as locking_mod
from hardware.difra.gui.main_window_ext.technical.h5_management_locking_mixin import (
    H5ManagementLockingMixin,
)


class _MessageBoxStub:
    Yes = 1
    No = 0
    critical_calls = []
    question_calls = []
    information_calls = []

    @classmethod
    def reset(cls):
        cls.critical_calls = []
        cls.question_calls = []
        cls.information_calls = []

    @staticmethod
    def question(*args, **kwargs):
        _MessageBoxStub.question_calls.append((args, kwargs))
        return _MessageBoxStub.Yes

    @staticmethod
    def information(*args, **kwargs):
        _MessageBoxStub.information_calls.append((args, kwargs))
        return None

    @staticmethod
    def warning(*args, **kwargs):
        return _MessageBoxStub.Yes

    @staticmethod
    def critical(_parent, title, text):
        _MessageBoxStub.critical_calls.append((title, text))
        return None


class _ContainerManagerStub:
    def __init__(self, *, locked=False):
        self._locked = bool(locked)

    def is_container_locked(self, _path):
        return self._locked


class _ValidatorStub:
    def __init__(self, result):
        self._result = result

    def validate_technical_container(self, *_args, **_kwargs):
        return self._result


class _Harness(H5ManagementLockingMixin):
    def __init__(self, active_path: Path):
        self.config = {"container_version": "0.2"}
        self._active_technical_container_path = str(active_path)
        self._active_technical_container_locked = False
        self.lock_calls = []
        self.events = []

    def _lock_container(self, container_path: str, container_id: str):
        self.lock_calls.append((container_path, container_id))

    def _log_technical_event(self, msg: str):
        self.events.append(msg)

    def _get_active_detector_aliases(self):
        return ["SAXS"]


def _create_container(path: Path, *, schema_version: str = "0.2"):
    with h5py.File(path, "w") as h5f:
        h5f.attrs["container_id"] = "tech_test_001"
        h5f.attrs["schema_version"] = schema_version


def test_lock_is_blocked_when_validation_fails(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        container_path = Path(tmp) / "technical_test.nxs.h5"
        _create_container(container_path)
        harness = _Harness(container_path)
        _MessageBoxStub.reset()

        monkeypatch.setattr(
            locking_mod,
            "get_container_manager",
            lambda _cfg: _ContainerManagerStub(locked=False),
        )
        monkeypatch.setattr(
            locking_mod,
            "get_technical_validator",
            lambda _cfg: _ValidatorStub((False, ["Missing required group: /entry"], [])),
        )
        monkeypatch.setattr(locking_mod, "QMessageBox", _MessageBoxStub)
        monkeypatch.setattr(harness, "_ensure_poni_before_lock", lambda *_args, **_kwargs: True)

        harness.lock_active_technical_container()

        assert harness.lock_calls == []
        assert harness._active_technical_container_locked is False
        assert len(_MessageBoxStub.critical_calls) == 1
        assert "Validation Failed" in _MessageBoxStub.critical_calls[0][0]


def test_lock_proceeds_when_validation_passes(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        container_path = Path(tmp) / "technical_test.nxs.h5"
        _create_container(container_path, schema_version="0.2")
        harness = _Harness(container_path)
        _MessageBoxStub.reset()

        monkeypatch.setattr(
            locking_mod,
            "get_container_manager",
            lambda _cfg: _ContainerManagerStub(locked=False),
        )
        monkeypatch.setattr(
            locking_mod,
            "get_technical_validator",
            lambda _cfg: _ValidatorStub((True, [], [])),
        )
        monkeypatch.setattr(locking_mod, "QMessageBox", _MessageBoxStub)
        monkeypatch.setattr(harness, "_ensure_poni_before_lock", lambda *_args, **_kwargs: True)

        harness.lock_active_technical_container()

        assert len(harness.lock_calls) == 1
        assert harness._active_technical_container_locked is True
        assert _MessageBoxStub.critical_calls == []


def test_lock_is_blocked_when_user_declines_poni_selection(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        container_path = Path(tmp) / "technical_test.nxs.h5"
        _create_container(container_path, schema_version="0.2")
        harness = _Harness(container_path)
        _MessageBoxStub.reset()

        monkeypatch.setattr(
            locking_mod,
            "get_container_manager",
            lambda _cfg: _ContainerManagerStub(locked=False),
        )
        monkeypatch.setattr(
            locking_mod,
            "get_technical_validator",
            lambda _cfg: _ValidatorStub((True, [], [])),
        )
        monkeypatch.setattr(locking_mod, "QMessageBox", _MessageBoxStub)

        original_question = _MessageBoxStub.question
        _MessageBoxStub.question = staticmethod(lambda *args, **kwargs: _MessageBoxStub.No)
        try:
            harness.lock_active_technical_container()
        finally:
            _MessageBoxStub.question = original_question

        assert harness.lock_calls == []
        assert harness._active_technical_container_locked is False


def test_demo_mode_auto_provisions_poni_then_locks(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        container_path = Path(tmp) / "technical_test.nxs.h5"
        _create_container(container_path, schema_version="0.2")
        harness = _Harness(container_path)
        harness.config["DEV"] = True
        _MessageBoxStub.reset()

        monkeypatch.setattr(
            locking_mod,
            "get_container_manager",
            lambda _cfg: _ContainerManagerStub(locked=False),
        )
        monkeypatch.setattr(
            locking_mod,
            "get_technical_validator",
            lambda _cfg: _ValidatorStub((True, [], [])),
        )
        monkeypatch.setattr(locking_mod, "QMessageBox", _MessageBoxStub)
        monkeypatch.setattr(
            harness,
            "_collect_lock_detector_aliases",
            lambda *_args, **_kwargs: ["SAXS"],
        )
        monkeypatch.setattr(
            harness,
            "_auto_provision_demo_poni_files",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            harness,
            "_sync_active_technical_container_from_table",
            lambda **_kwargs: True,
            raising=False,
        )
        states = iter([False, True])
        monkeypatch.setattr(
            harness,
            "_container_has_poni_datasets",
            lambda *_args, **_kwargs: next(states),
        )

        harness.lock_active_technical_container()

        assert len(harness.lock_calls) == 1
        assert harness._active_technical_container_locked is True
        assert len(_MessageBoxStub.information_calls) >= 1
