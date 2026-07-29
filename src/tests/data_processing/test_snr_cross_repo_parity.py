"""Opt-in SNR parity check against a local XRD-preprocessing checkout."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.cross_repo

# This module must be invisible to ordinary CI: importing xrd-analysis itself
# may pull optional scientific dependencies that a cross-repository check alone
# does not need.
if not os.environ.get("XRD_PREPROCESSING_ROOT"):
    pytest.skip(
        "set XRD_PREPROCESSING_ROOT to run cross-repository SNR parity",
        allow_module_level=True,
    )

from xrdanalysis.data_processing.transformers import SNRTransformer  # noqa: E402


def _preprocessing_root() -> Path:
    """Return explicitly requested sibling source, or skip outside local parity runs."""
    configured_root = os.environ.get("XRD_PREPROCESSING_ROOT")
    assert configured_root is not None
    root = Path(configured_root).expanduser().resolve()
    if not (root / "src" / "xrd_preprocessing" / "snr.py").is_file():
        pytest.skip(
            f"XRD_PREPROCESSING_ROOT is not an xrd-preprocessing checkout: {root}"
        )
    return root


@pytest.mark.parametrize(
    ("q", "intensity", "sigma"),
    [
        (
            np.linspace(0.1, 1.1, 5),
            np.array([2.0, 3.0, 5.0, 7.0, 11.0]),
            np.array([0.5, 1.0, 1.0, 2.0, 2.0]),
        ),
        (
            np.array([0.1, 0.2, 0.7, 1.8, 3.0]),
            np.array([3.0, 4.0, 9.0, 10.0, 14.0]),
            np.array([1.0, 1.5, 2.0, 2.5, 4.0]),
        ),
    ],
    ids=["uniform", "nonuniform"],
)
def test_native_poisson_scalars_match_xrd_preprocessing(
    monkeypatch, q, intensity, sigma
):
    """Compare only the documented common native Poisson scalar contract."""
    root = _preprocessing_root()
    monkeypatch.syspath_prepend(str(root / "src"))
    from xrd_preprocessing import snr as preprocessing_snr  # noqa: PLC0415

    module_path = Path(preprocessing_snr.__file__).resolve()
    expected_source = (root / "src").resolve()
    assert module_path.is_relative_to(
        expected_source
    ), f"loaded xrd_preprocessing from {module_path}, expected {expected_source}"

    reference = preprocessing_snr.calculate_snr(
        q, intensity, sigma=sigma, snr_method="poisson"
    )
    frame = pd.DataFrame(
        {
            "q_range": [q],
            "radial_profile_data": [intensity],
            "radial_profile_sigma": [sigma],
        }
    )
    result = SNRTransformer(snr_method="poisson").transform(frame).iloc[0]

    assert result["snr_method_used"] == "poisson"
    for name in ("noise_std", "snr_linear", "snr_db"):
        assert result[name] == pytest.approx(reference[name])
