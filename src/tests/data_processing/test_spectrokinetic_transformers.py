"""Tests for SK-Ana-inspired spectrokinetic transformers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing.pipeline import MLPipeline
from xrdanalysis.data_processing.spectrokinetic_transformers import (
    MCRALSTransformer,
    SpectroSVDTransformer,
    convolve_spectrum,
    enforce_unimodal,
    optimize_broadening_single,
    solve_C,
    solve_C_coupled,
    solve_S,
    solve_S_coupled,
)


def _build_synthetic_matrix(
    n_delay: int = 30,
    n_wav: int = 50,
    n_comp: int = 2,
    noise: float = 0.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    delay = np.linspace(0.0, 5.0, n_delay)
    wav = np.linspace(300.0, 700.0, n_wav)

    c = np.abs(rng.normal(size=(n_delay, n_comp)))
    centers = np.linspace(350.0, 650.0, n_comp)
    widths = np.linspace(25.0, 40.0, n_comp)

    s = np.zeros((n_wav, n_comp), dtype=float)
    for i in range(n_comp):
        s[:, i] = np.exp(-((wav - centers[i]) ** 2) / (2 * widths[i] ** 2))
    s = s / np.maximum(np.max(s, axis=0, keepdims=True), 1e-12)

    mat = c @ s.T
    if noise > 0:
        mat = mat + noise * rng.normal(size=mat.shape)

    return delay, wav, c, s, mat


def _load_keele_i_vs_q_standard_df() -> pd.DataFrame:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "skana"
        / "I_vs_q_100samples.dat"
    )
    raw = pd.read_csv(fixture, sep=r"\s+")

    q_axis = raw["q"].to_numpy(dtype=float)
    delay_axis = np.asarray([float(c) for c in raw.columns if c != "q"], dtype=float)
    spectro_matrix = raw.drop(columns=["q"]).to_numpy(dtype=float).T

    return pd.DataFrame(
        {
            "spectro_matrix": [spectro_matrix],
            "delay_axis": [delay_axis],
            "wavelength_axis": [q_axis],
        }
    )


def _load_keele_component_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "skana"
    fat = pd.read_csv(fixture_dir / "xrd_component_fat.txt", sep=r"\s+")
    water = pd.read_csv(fixture_dir / "xrd_component_water.txt", sep=r"\s+")

    x_fat = fat.iloc[:, 0].to_numpy(dtype=float)
    x_water = water.iloc[:, 0].to_numpy(dtype=float)
    if not np.allclose(x_fat, x_water):
        raise ValueError("Fat and water component profiles must share the same x-axis.")

    fat_profile = fat.iloc[:, 1].to_numpy(dtype=float)
    water_profile = water.iloc[:, 1].to_numpy(dtype=float)
    return x_fat, fat_profile, water_profile


def test_svd_lof_and_residual_curve_known_low_rank():
    delay, wav, _c, _s, mat = _build_synthetic_matrix(n_comp=2, noise=0.0)

    df = pd.DataFrame(
        {
            "spectro_matrix": [mat],
            "delay_axis": [delay],
            "wavelength_axis": [wav],
        }
    )

    tr = SpectroSVDTransformer(max_rank=6)
    out = tr.transform(df)

    lof_curve = out.iloc[0]["svd_lof_curve"]
    rec_rank = out.iloc[0]["svd_recommended_rank"]

    assert len(lof_curve) == 6
    # True rank is 2, so rank-2 LOF should be near machine precision.
    assert lof_curve[1] < 1e-8
    assert 1 <= rec_rank <= 6


def test_solve_c_and_s_unconstrained_identity_case():
    s = np.eye(2)
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    c0 = np.zeros((2, 2))
    c_est = solve_C(s=s, data=data, c=c0, nonneg_c=False)
    assert np.allclose(c_est, data)

    c_eye = np.eye(2)
    s0 = np.zeros((2, 2))
    s_est = solve_S(
        c=c_eye,
        data=np.eye(2),
        s=s0,
        x_s=np.array([1.0, 2.0]),
        nonneg_s=False,
        uni_s=False,
        s0=None,
        norm_s=False,
        smooth=0.0,
        sum_s=False,
        hard_s0=False,
        w_hard_s0=1.0,
    )
    assert np.allclose(s_est, np.eye(2), atol=1e-10)


def test_mixed_per_component_nonneg_s_constraints():
    delay, wav, c_true, _s, _mat = _build_synthetic_matrix(n_delay=40, n_wav=80, n_comp=2)

    s_target = np.zeros((wav.size, 2), dtype=float)
    s_target[:, 0] = np.exp(-((wav - 450) ** 2) / (2 * 30**2))
    s_target[:, 1] = np.sin(np.linspace(0, 3 * np.pi, wav.size))
    mat = c_true @ s_target.T

    s_init = np.abs(np.random.default_rng(0).normal(size=s_target.shape))
    s_est = solve_S(
        c=c_true,
        data=mat,
        s=s_init,
        x_s=wav,
        nonneg_s=[True, False],
        uni_s=False,
        s0=None,
        norm_s=False,
        smooth=0.0,
        sum_s=False,
        hard_s0=False,
        w_hard_s0=1.0,
    )

    assert np.min(s_est[:, 0]) >= -1e-10
    assert np.min(s_est[:, 1]) < -1e-3


def test_unimodality_projection_returns_single_mode():
    y = np.array([0.0, 3.0, 1.0, 4.0, 2.0, 0.5, 1.0], dtype=float)
    proj = enforce_unimodal(y)

    # A discrete unimodal sequence should have at most one switch in sign
    # of first differences from positive to negative.
    diff = np.diff(proj)
    sign = np.sign(diff)
    sign = sign[sign != 0]
    switches = np.sum((sign[:-1] > 0) & (sign[1:] < 0))
    assert switches <= 1


def test_broadening_refinement_improves_reconstruction_error():
    wav = np.linspace(300.0, 700.0, 120)
    s = np.zeros((wav.size, 2))
    s[:, 0] = np.exp(-((wav - 420) ** 2) / (2 * 20**2))
    s[:, 1] = np.exp(-((wav - 560) ** 2) / (2 * 30**2))

    c_row = np.array([0.6, 0.4])
    g_true = np.array([3.0, 1.5])

    data_row = np.zeros(wav.size)
    for k in range(2):
        data_row += c_row[k] * convolve_spectrum(s[:, k], g_true[k])

    g_init = np.zeros(2)

    def err(gvec):
        recon = np.zeros_like(data_row)
        for j in range(2):
            recon += c_row[j] * convolve_spectrum(s[:, j], gvec[j])
        return float(np.sum((data_row - recon) ** 2))

    e0 = err(g_init)
    g_opt = optimize_broadening_single(
        data_row=data_row,
        c_row=c_row,
        s=s,
        g_init=g_init,
        sigma_max=10.0,
        broadening_vec=np.array([True, True]),
    )
    e1 = err(g_opt)

    assert np.all(g_opt >= 0)
    assert e1 < e0


def test_fixed_spectra_hard_and_soft_constraints():
    delay, wav, c_true, s_true, mat = _build_synthetic_matrix(n_delay=30, n_wav=70, n_comp=2)
    s_init = np.abs(np.random.default_rng(1).normal(size=s_true.shape))
    s0 = s_true[:, [0]]

    hard = solve_S(
        c=c_true,
        data=mat,
        s=s_init,
        x_s=wav,
        nonneg_s=True,
        uni_s=False,
        s0=s0,
        norm_s=False,
        smooth=0.0,
        sum_s=False,
        hard_s0=True,
        w_hard_s0=1.0,
    )
    assert np.allclose(hard[:, 0], s0[:, 0], atol=1e-8)

    no_soft = solve_S(
        c=c_true,
        data=mat,
        s=s_init,
        x_s=wav,
        nonneg_s=True,
        uni_s=False,
        s0=None,
        norm_s=False,
        smooth=0.0,
        sum_s=False,
        hard_s0=False,
        w_hard_s0=1.0,
    )
    soft = solve_S(
        c=c_true,
        data=mat,
        s=s_init,
        x_s=wav,
        nonneg_s=True,
        uni_s=False,
        s0=s0,
        norm_s=False,
        smooth=0.0,
        sum_s=False,
        hard_s0=False,
        w_hard_s0=10.0,
    )

    dist_soft = np.linalg.norm(soft[:, 0] - s0[:, 0])
    dist_no = np.linalg.norm(no_soft[:, 0] - s0[:, 0])
    assert dist_soft <= dist_no + 1e-10


def test_correction_mode_enforces_coupling_and_orthogonality():
    delay, wav, _c, _s, _mat = _build_synthetic_matrix(n_delay=25, n_wav=60, n_comp=3)

    c_true = np.abs(np.random.default_rng(2).normal(size=(delay.size, 3)))
    c_true[:, 1] = 0.4 * c_true[:, 0]  # coupled correction column

    s_true = np.zeros((wav.size, 3), dtype=float)
    s_true[:, 0] = np.exp(-((wav - 440) ** 2) / (2 * 25**2))
    s_true[:, 1] = np.sin(np.linspace(0, 4 * np.pi, wav.size))
    s_true[:, 2] = np.exp(-((wav - 610) ** 2) / (2 * 35**2))

    data = c_true @ s_true.T
    c_init = np.abs(np.random.default_rng(3).normal(size=c_true.shape))
    s_init = np.abs(np.random.default_rng(4).normal(size=s_true.shape))

    c_coupled = solve_C_coupled(
        s=s_init,
        data=data,
        c=c_init,
        nonneg_c=True,
        null_c=None,
        close_c=False,
        w_close_c=0.0,
        n_fixed=1,
    )

    ratio = c_coupled[:, 1] / np.maximum(c_coupled[:, 0], 1e-12)
    assert float(np.std(ratio)) < 1e-6

    s_coupled = solve_S_coupled(
        c=c_true,
        data=data,
        s=s_init,
        x_s=wav,
        nonneg_s=True,
        uni_s=False,
        s0=None,
        norm_s=False,
        smooth=0.0,
        sum_s=False,
        hard_s0=False,
        w_hard_s0=1.0,
        n_fixed=1,
        lambda_corr=1e-2,
    )

    dot = float(np.dot(s_coupled[:, 0], s_coupled[:, 1]))
    denom = float(np.linalg.norm(s_coupled[:, 0]) * np.linalg.norm(s_coupled[:, 1]))
    if denom > 0:
        assert abs(dot / denom) < 1e-6
    assert abs(float(np.mean(s_coupled[:, 1]))) < 1e-10


def test_mcrals_row_transformer_outputs_columns_and_shapes():
    delay, wav, _c, _s, mat = _build_synthetic_matrix(n_delay=24, n_wav=48, n_comp=2, noise=0.001)

    df = pd.DataFrame(
        {
            "spectro_matrix": [mat],
            "delay_axis": [delay],
            "wavelength_axis": [wav],
        }
    )

    tr = MCRALSTransformer(n_components=2, maxiter=25, thresh=1e-5, init_method="svd")
    out = tr.transform(df)

    assert "als_C" in out.columns
    assert "als_S" in out.columns
    assert "als_meta" in out.columns

    c_est = out.iloc[0]["als_C"]
    s_est = out.iloc[0]["als_S"]
    model = out.iloc[0]["als_model"]

    assert isinstance(c_est, np.ndarray)
    assert isinstance(s_est, np.ndarray)
    assert c_est.shape == (delay.size, 2)
    assert s_est.shape == (wav.size, 2)
    assert model.shape == mat.shape


def test_mcrals_group_tile_delay_splits_c_back_to_each_row():
    delay1, wav, _c1, _s1, mat1 = _build_synthetic_matrix(n_delay=16, n_wav=40, n_comp=2, seed=10)
    delay2, wav2, _c2, _s2, mat2 = _build_synthetic_matrix(n_delay=12, n_wav=40, n_comp=2, seed=11)

    assert np.allclose(wav, wav2)

    df = pd.DataFrame(
        {
            "group_id": ["g1", "g1"],
            "spectro_matrix": [mat1, mat2],
            "delay_axis": [delay1, delay2],
            "wavelength_axis": [wav, wav2],
        }
    )

    tr = MCRALSTransformer(
        decomposition_mode="group",
        group_col="group_id",
        group_strategy="tile_delay",
        n_components=2,
        maxiter=20,
        thresh=1e-4,
    )
    out = tr.transform(df)

    c0 = out.iloc[0]["als_C"]
    c1 = out.iloc[1]["als_C"]
    s0 = out.iloc[0]["als_S"]
    s1 = out.iloc[1]["als_S"]

    assert c0.shape[0] == delay1.size
    assert c1.shape[0] == delay2.size
    assert c0.shape[1] == 2
    assert c1.shape[1] == 2
    assert np.allclose(s0, s1)


@pytest.mark.parametrize("n_components", [2, 3, 4])
def test_keele_standard_data_svd_then_mcrals_components(n_components: int):
    df = _load_keele_i_vs_q_standard_df()
    row_in = df.iloc[0]
    mat_in = row_in["spectro_matrix"]
    delay = row_in["delay_axis"]
    wav = row_in["wavelength_axis"]

    pipe = MLPipeline(
        data_wrangling_steps=[
            ("svd", SpectroSVDTransformer(max_rank=10)),
            (
                "als",
                MCRALSTransformer(
                    n_components=n_components,
                    init_method="svd",
                    maxiter=200,
                    thresh=1e-5,
                    nonneg_s=True,
                    nonneg_c=True,
                ),
            ),
        ],
        preprocessing_steps=[],
        estimator=None,
    )

    out = pipe.transform(df)
    row = out.iloc[0]

    assert row["svd_u"].shape[0] == delay.size
    assert row["svd_vt"].shape[1] == wav.size
    assert len(row["svd_lof_curve"]) == 10
    assert 1 <= int(row["svd_recommended_rank"]) <= 10

    assert row["als_C"].shape == (delay.size, n_components)
    assert row["als_S"].shape == (wav.size, n_components)
    assert row["als_model"].shape == mat_in.shape
    assert row["als_residual"].shape == mat_in.shape
    assert bool(row["als_converged"])
    assert np.isfinite(float(row["als_lof_pct"]))
    assert np.isfinite(float(row["als_rss"]))


@pytest.mark.parametrize("init_method", ["svd", "pca", "nmf", "seq"])
@pytest.mark.parametrize("norm_mode", ["intensity", "l1"])
def test_keele_fixed_profiles_three_components_init_and_constraints(
    init_method: str,
    norm_mode: str,
):
    df = _load_keele_i_vs_q_standard_df()
    row_in = df.iloc[0]
    wav = row_in["wavelength_axis"]

    x_ref, fat_profile, water_profile = _load_keele_component_profiles()
    fixed = np.column_stack([fat_profile, water_profile])
    expected_fixed_interp = np.column_stack(
        [
            np.interp(wav, x_ref, fat_profile),
            np.interp(wav, x_ref, water_profile),
        ]
    )

    tr = MCRALSTransformer(
        n_components=3,
        init_method=init_method,
        maxiter=180,
        thresh=1e-5,
        fixed_spectra=fixed,
        fixed_wavelength_axis=x_ref,
        interpolate_fixed=True,
        hard_s0=True,
        nonneg_c=True,
        nonneg_s=[True, True, False],
        norm_s=True,
        norm_mode=norm_mode,
        sum_norm=False,
        random_state=42,
    )
    out = tr.transform(df)
    row = out.iloc[0]
    s = np.asarray(row["als_S"], dtype=float)

    assert bool(row["als_converged"])
    assert row["als_C"].shape[1] == 3
    assert s.shape[1] == 3

    # Hard fixed spectra should be preserved exactly after interpolation.
    assert np.allclose(s[:, :2], expected_fixed_interp, atol=1e-12)
    assert float(np.min(s[:, 0])) >= -1e-12
    assert float(np.min(s[:, 1])) >= -1e-12

    assert row["als_meta"]["init_method"] == init_method
    assert row["als_meta"]["constraints"]["nonneg_s"] == [True, True, False]
    assert row["als_meta"]["constraints"]["norm_mode"] == norm_mode

    third = s[:, 2]
    assert np.isfinite(third).all()
    assert float(np.max(np.abs(third))) > 0.0
    if norm_mode == "l1":
        assert np.isclose(float(np.sum(np.abs(third))), 1.0, atol=1e-8)
    else:
        assert np.isclose(float(np.max(np.abs(third))), 1.0, atol=1e-8)


@pytest.mark.parametrize("norm_mode", ["intensity", "l1"])
def test_keele_fixed_profiles_restart_initialization(norm_mode: str):
    df = _load_keele_i_vs_q_standard_df()
    wav = np.asarray(df.iloc[0]["wavelength_axis"], dtype=float)

    x_ref, fat_profile, water_profile = _load_keele_component_profiles()
    fixed = np.column_stack([fat_profile, water_profile])
    expected_fixed_interp = np.column_stack(
        [
            np.interp(wav, x_ref, fat_profile),
            np.interp(wav, x_ref, water_profile),
        ]
    )

    base = MCRALSTransformer(
        n_components=3,
        init_method="svd",
        maxiter=180,
        thresh=1e-5,
        fixed_spectra=fixed,
        fixed_wavelength_axis=x_ref,
        interpolate_fixed=True,
        hard_s0=True,
        nonneg_c=True,
        nonneg_s=[True, True, False],
        norm_s=True,
        norm_mode=norm_mode,
        sum_norm=False,
        random_state=42,
    )
    base_row = base.transform(df).iloc[0]

    restart = MCRALSTransformer(
        n_components=3,
        init_method="restart",
        restart_result=base._last_result,
        maxiter=60,
        thresh=1e-5,
        fixed_spectra=fixed,
        fixed_wavelength_axis=x_ref,
        interpolate_fixed=True,
        hard_s0=True,
        nonneg_c=True,
        nonneg_s=[True, True, False],
        norm_s=True,
        norm_mode=norm_mode,
        sum_norm=False,
        random_state=42,
    )
    restart_row = restart.transform(df).iloc[0]
    s_restart = np.asarray(restart_row["als_S"], dtype=float)

    assert bool(restart_row["als_converged"])
    assert restart_row["als_meta"]["init_method"] == "restart"
    assert np.allclose(s_restart[:, :2], expected_fixed_interp, atol=1e-12)
    assert float(restart_row["als_lof_pct"]) <= float(base_row["als_lof_pct"]) + 1e-3


def test_optional_parity_fixture_if_available():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "skana"
        / "parity_fixture.npz"
    )

    if not fixture.exists():
        pytest.skip("No SK-Ana parity fixture found. Generate via build_skana_reference_fixture.R")

    npz = np.load(fixture, allow_pickle=True)

    mat = npz["matrix"]
    delay = npz["delay"]
    wav = npz["wavelength"]
    expected_svd_lof = npz["svd_lof_first10"]
    expected_als_lof = float(npz["als_lof"])
    expected_model = npz["als_model"]

    df = pd.DataFrame(
        {
            "spectro_matrix": [mat],
            "delay_axis": [delay],
            "wavelength_axis": [wav],
        }
    )

    svd = SpectroSVDTransformer(max_rank=10)
    svd_out = svd.transform(df)
    lof_curve = np.asarray(svd_out.iloc[0]["svd_lof_curve"], dtype=float)
    assert np.allclose(lof_curve[:10], expected_svd_lof, atol=0.2)

    als = MCRALSTransformer(
        n_components=2,
        init_method="svd",
        maxiter=80,
        thresh=1e-5,
        nonneg_s=True,
        nonneg_c=True,
    )
    als_out = als.transform(df)

    lof = float(als_out.iloc[0]["als_lof_pct"])
    model = np.asarray(als_out.iloc[0]["als_model"], dtype=float)

    assert abs(lof - expected_als_lof) <= 1.0

    rel_frob = np.linalg.norm(model - expected_model) / max(
        np.linalg.norm(expected_model), 1e-12
    )
    assert rel_frob <= 1e-2


def test_pipeline_integration_with_mlpipeline_wrangle():
    delay, wav, _c, _s, mat = _build_synthetic_matrix(n_delay=20, n_wav=36, n_comp=2)

    df = pd.DataFrame(
        {
            "spectro_matrix": [mat, mat * 1.01],
            "delay_axis": [delay, delay],
            "wavelength_axis": [wav, wav],
            "target": [0, 1],
        }
    )

    pipe = MLPipeline(
        data_wrangling_steps=[
            (
                "als",
                MCRALSTransformer(
                    n_components=2,
                    maxiter=15,
                    thresh=1e-4,
                ),
            )
        ],
        preprocessing_steps=[],
        estimator=None,
    )

    transformed = pipe.transform(df)
    assert isinstance(transformed, pd.DataFrame)
    assert "als_model" in transformed.columns
    assert transformed["als_model"].notna().all()
