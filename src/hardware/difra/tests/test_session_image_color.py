"""Tests for session image color channel handling."""

import sys
from types import SimpleNamespace

import numpy as np

from hardware.difra.gui.main_window_ext.session_mixin import SessionMixin


class _ImageViewStub:
    def __init__(self, image_path):
        self.current_image_path = str(image_path)


class _WorkspaceHarness(SessionMixin):
    def __init__(self, image_path):
        self.image_view = _ImageViewStub(image_path)


def test_extract_current_image_array_converts_bgr_to_rgb(monkeypatch):
    fake_cv2 = SimpleNamespace()
    fake_cv2.IMREAD_UNCHANGED = -1
    fake_cv2.COLOR_BGR2RGB = 1
    fake_cv2.COLOR_BGRA2RGBA = 2
    fake_cv2.imread = lambda _path, _flag: np.array([[[0, 0, 255]]], dtype=np.uint8)
    fake_cv2.cvtColor = lambda arr, code: (
        arr[:, :, ::-1]
        if code == fake_cv2.COLOR_BGR2RGB
        else arr[:, :, [2, 1, 0, 3]]
    )

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    harness = _WorkspaceHarness("/tmp/dummy.png")
    image_array = harness._extract_current_image_array()

    assert image_array.shape == (1, 1, 3)
    assert image_array[0, 0].tolist() == [255, 0, 0]


def test_extract_current_image_array_converts_bgra_to_rgba(monkeypatch):
    fake_cv2 = SimpleNamespace()
    fake_cv2.IMREAD_UNCHANGED = -1
    fake_cv2.COLOR_BGR2RGB = 1
    fake_cv2.COLOR_BGRA2RGBA = 2
    fake_cv2.imread = lambda _path, _flag: np.array([[[10, 20, 30, 40]]], dtype=np.uint8)
    fake_cv2.cvtColor = lambda arr, code: (
        arr[:, :, ::-1]
        if code == fake_cv2.COLOR_BGR2RGB
        else arr[:, :, [2, 1, 0, 3]]
    )

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    harness = _WorkspaceHarness("/tmp/dummy.png")
    image_array = harness._extract_current_image_array()

    assert image_array.shape == (1, 1, 4)
    assert image_array[0, 0].tolist() == [30, 20, 10, 40]
