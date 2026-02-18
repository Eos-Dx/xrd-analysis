import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Keele ALS Variability Analysis (No SKA Input)

    This notebook does full analysis directly from raw curves:
    1. Load `I_vs_q` cancer and non-cancer `.dat` files.
    2. Run MCR-ALS with fixed spectral shapes (`3 fixed + 2 free` by default).
    3. Evaluate multiple initialization modes (`svd`, `pca`, `nmf`, `seq`, `restart`) and normalization modes (`intensity`, `l1`).
    4. Compare coefficient variability between cancer/non-cancer via:
       - covariance volume proxy: `log(det(cov))`
       - Mahalanobis distance distributions
       - permutation test on `delta_logdet`
    """)
    return


@app.cell
def _(mo):
    cancer_dat = mo.ui.text(
        value="/Users/sad/dev/eos_play/jupyter_notebooks/Keele/I_vs_q_Cancer_L1.dat",
        label="Cancer I_vs_q .dat path",
    )
    noncancer_dat = mo.ui.text(
        value="/Users/sad/dev/eos_play/jupyter_notebooks/Keele/I_vs_q_NonCancer_L1.dat",
        label="Non-Cancer I_vs_q .dat path",
    )

    fixed_1 = mo.ui.text(
        value="/Users/sad/dev/eos_play/jupyter_notebooks/Keele/xrd_component_water.txt",
        label="Fixed profile #1 path (water)",
    )
    fixed_2 = mo.ui.text(
        value="/Users/sad/dev/eos_play/jupyter_notebooks/Keele/xrd_component_fat.txt",
        label="Fixed profile #2 path (fat)",
    )
    fixed_3 = mo.ui.text(
        value="/Users/sad/dev/eos_play/jupyter_notebooks/Keele/xrd_component_collagen.txt",
        label="Fixed profile #3 path (collagen)",
    )

    init_methods = mo.ui.text(
        value="ls",
        label="Init methods (comma-separated; ls alias maps to svd)",
    )
    norm_modes = mo.ui.text(
        value="l1",
        label="Norm modes (comma-separated)",
    )

    n_components = mo.ui.number(value=5, start=1, step=1, label="Total components")
    n_fixed = mo.ui.number(value=3, start=0, step=1, label="Fixed components")
    nonneg_s = mo.ui.text(
        value="1,1,1,0,0",
        label="Per-component nonneg_s vector (1/0, comma-separated)",
    )

    maxiter = mo.ui.number(value=200, start=10, step=10, label="ALS maxiter")
    thresh = mo.ui.number(value=1e-5, start=1e-8, step=1e-5, label="ALS thresh")
    random_state = mo.ui.number(value=42, start=0, step=1, label="Random state")

    run_perm_test = mo.ui.checkbox(value=True, label="Run permutation test")
    n_perm = mo.ui.number(value=300, start=50, step=50, label="Permutation count")

    mo.vstack(
        [
            mo.hstack([cancer_dat, noncancer_dat]),
            mo.hstack([fixed_1, fixed_2]),
            fixed_3,
            mo.hstack([init_methods, norm_modes]),
            mo.hstack([n_components, n_fixed, nonneg_s]),
            mo.hstack([maxiter, thresh, random_state]),
            mo.hstack([run_perm_test, n_perm]),
        ]
    )
    return (
        cancer_dat,
        fixed_1,
        fixed_2,
        fixed_3,
        init_methods,
        maxiter,
        n_components,
        n_fixed,
        n_perm,
        noncancer_dat,
        nonneg_s,
        norm_modes,
        random_state,
        run_perm_test,
        thresh,
    )


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    from sklearn.covariance import LedoitWolf

    sys.path.insert(0, "/Users/sad/dev/xrd-analysis/src")
    from xrdanalysis.data_processing.spectrokinetic_transformers import MCRALSTransformer

    def _parse_csv_tokens(raw: str) -> list[str]:
        return [x.strip() for x in raw.split(",") if x.strip()]

    def _normalize_init_method(raw: str) -> str:
        m = raw.strip().lower()
        aliases = {
            "ls": "svd",
            "l-s": "svd",
            "l_s": "svd",
        }
        m = aliases.get(m, m)
        allowed = {"svd", "pca", "nmf", "seq", "restart"}
        if m not in allowed:
            raise ValueError(
                f"Unsupported init method '{raw}'. Allowed: {sorted(allowed)} (plus ls alias)."
            )
        return m

    def _parse_nonneg_vector(raw: str, n: int) -> list[bool]:
        tokens = _parse_csv_tokens(raw)
        if len(tokens) != n:
            raise ValueError(f"nonneg_s length {len(tokens)} must equal n_components={n}.")
        out = []
        for t in tokens:
            tl = t.lower()
            if tl in {"1", "true", "t", "yes", "y"}:
                out.append(True)
            elif tl in {"0", "false", "f", "no", "n"}:
                out.append(False)
            else:
                raise ValueError(f"Invalid nonneg token: '{t}'. Use 1/0 or true/false.")
        return out

    def _load_i_vs_q_dat(path: str) -> tuple[np.ndarray, np.ndarray]:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        df = pd.read_csv(p, sep=r"\s+")
        q_col = "q" if "q" in df.columns else df.columns[0]
        q = df[q_col].to_numpy(dtype=float)
        curves = df.drop(columns=[q_col]).to_numpy(dtype=float).T

        if curves.ndim != 2 or curves.shape[1] != q.size:
            raise ValueError(f"Unexpected dat shape from {p}: q={q.shape}, curves={curves.shape}")
        return q, curves

    def _load_profile(path: str) -> tuple[np.ndarray, np.ndarray]:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Profile file not found: {p}")

        df = pd.read_csv(p, sep=r"\s+|,", engine="python")
        if df.shape[1] < 2:
            raise ValueError(f"Profile file must have at least 2 columns: {p}")

        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x.size < 3:
            raise ValueError(f"Profile has too few numeric points: {p}")
        return x, y

    def _interp_rows(curves: np.ndarray, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        out = np.zeros((curves.shape[0], q_to.size), dtype=float)
        for i in range(curves.shape[0]):
            out[i, :] = np.interp(q_to, q_from, curves[i, :])
        return out

    def _group_cov_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        lw = LedoitWolf().fit(x)
        cov = lw.covariance_
        prec = lw.precision_
        mu = lw.location_
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = np.nan
        return mu, cov, prec, float(logdet)

    def _mahal_sq(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
        d = x - mu
        return np.einsum("ij,jk,ik->i", d, prec, d)

    def _perm_test_delta_logdet(
        coeffs: np.ndarray,
        labels: np.ndarray,
        n_perm: int,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        x_c = coeffs[labels == 1]
        x_n = coeffs[labels == 0]
        obs = _group_cov_stats(x_c)[3] - _group_cov_stats(x_n)[3]

        perm_stats = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            y_perm = rng.permutation(labels)
            xp_c = coeffs[y_perm == 1]
            xp_n = coeffs[y_perm == 0]
            perm_stats[i] = _group_cov_stats(xp_c)[3] - _group_cov_stats(xp_n)[3]

        p_one_sided = float((1 + np.sum(perm_stats >= obs)) / (n_perm + 1))
        return float(obs), p_one_sided

    return (
        MCRALSTransformer,
        _group_cov_stats,
        _interp_rows,
        _load_i_vs_q_dat,
        _load_profile,
        _mahal_sq,
        _normalize_init_method,
        _parse_csv_tokens,
        _parse_nonneg_vector,
        _perm_test_delta_logdet,
        mannwhitneyu,
        np,
        pd,
    )


@app.cell(hide_code=True)
def _(
    MCRALSTransformer,
    _group_cov_stats,
    _interp_rows,
    _load_i_vs_q_dat,
    _load_profile,
    _mahal_sq,
    _normalize_init_method,
    _parse_csv_tokens,
    _parse_nonneg_vector,
    _perm_test_delta_logdet,
    cancer_dat,
    fixed_1,
    fixed_2,
    fixed_3,
    init_methods,
    mannwhitneyu,
    maxiter,
    n_components,
    n_fixed,
    n_perm,
    noncancer_dat,
    nonneg_s,
    norm_modes,
    np,
    pd,
    random_state,
    run_perm_test,
    thresh,
):
    metrics_df = pd.DataFrame()
    coeffs_long_df = pd.DataFrame()
    analysis_error = None

    try:
        n_comp = int(n_components.value)
        n_fix = int(n_fixed.value)
        if n_fix > n_comp:
            raise ValueError("n_fixed cannot exceed n_components.")

        nonneg_vec = _parse_nonneg_vector(nonneg_s.value, n_comp)
        methods = [_normalize_init_method(m) for m in _parse_csv_tokens(init_methods.value)]
        norms = [m.lower() for m in _parse_csv_tokens(norm_modes.value)]

        q_c, curves_c = _load_i_vs_q_dat(cancer_dat.value)
        q_n, curves_n = _load_i_vs_q_dat(noncancer_dat.value)

        if q_c.shape != q_n.shape or not np.allclose(q_c, q_n):
            curves_n = _interp_rows(curves_n, q_n, q_c)

        q = q_c
        curves_all = np.vstack([curves_c, curves_n])
        labels = np.concatenate(
            [np.ones(curves_c.shape[0], dtype=int), np.zeros(curves_n.shape[0], dtype=int)]
        )

        fixed_paths = [p for p in [fixed_1.value, fixed_2.value, fixed_3.value] if p.strip()]
        if n_fix > 0 and len(fixed_paths) < n_fix:
            raise ValueError(f"Need at least {n_fix} fixed profile paths, got {len(fixed_paths)}.")

        fixed_mat = None
        if n_fix > 0:
            fixed_cols = []
            for p in fixed_paths[:n_fix]:
                x_ref, y_ref = _load_profile(p)
                fixed_cols.append(np.interp(q, x_ref, y_ref))
            fixed_mat = np.column_stack(fixed_cols)

        df_input = pd.DataFrame(
            {
                "spectro_matrix": [curves_all],
                "delay_axis": [np.arange(curves_all.shape[0], dtype=float)],
                "wavelength_axis": [q],
            }
        )

        rng = np.random.default_rng(int(random_state.value))
        rows = []
        coeffs_rows = []

        for norm_mode in norms:
            restart_result = None

            for init in methods:
                init_for_run = init
                if init == "restart" and restart_result is None:
                    warm = MCRALSTransformer(
                        n_components=n_comp,
                        init_method="svd",
                        maxiter=int(maxiter.value),
                        thresh=float(thresh.value),
                        nonneg_c=True,
                        nonneg_s=nonneg_vec,
                        norm_s=True,
                        norm_mode=norm_mode,
                        sum_norm=False,
                        fixed_spectra=fixed_mat,
                        fixed_wavelength_axis=q if fixed_mat is not None else None,
                        interpolate_fixed=False,
                        hard_s0=bool(n_fix > 0),
                        random_state=int(random_state.value),
                    )
                    warm.transform(df_input)
                    restart_result = warm._last_result

                tr = MCRALSTransformer(
                    n_components=n_comp,
                    init_method=init_for_run,
                    restart_result=restart_result if init_for_run == "restart" else None,
                    maxiter=int(maxiter.value),
                    thresh=float(thresh.value),
                    nonneg_c=True,
                    nonneg_s=nonneg_vec,
                    norm_s=True,
                    norm_mode=norm_mode,
                    sum_norm=False,
                    fixed_spectra=fixed_mat,
                    fixed_wavelength_axis=q if fixed_mat is not None else None,
                    interpolate_fixed=False,
                    hard_s0=bool(n_fix > 0),
                    random_state=int(random_state.value),
                )

                out = tr.transform(df_input)
                row = out.iloc[0]
                coeffs = np.asarray(row["als_C"], dtype=float)

                restart_result = tr._last_result

                x_c = coeffs[labels == 1]
                x_n = coeffs[labels == 0]

                mu_c, cov_c, prec_c, logdet_c = _group_cov_stats(x_c)
                mu_n, cov_n, prec_n, logdet_n = _group_cov_stats(x_n)

                d2_c = _mahal_sq(x_c, mu_c, prec_c)
                d2_n = _mahal_sq(x_n, mu_n, prec_n)
                mw = mannwhitneyu(d2_c, d2_n, alternative="greater")

                delta_logdet = float(logdet_c - logdet_n)
                if run_perm_test.value:
                    _obs, perm_p = _perm_test_delta_logdet(
                        coeffs,
                        labels,
                        int(n_perm.value),
                        rng,
                    )
                else:
                    perm_p = np.nan

                run_id = f"{norm_mode}:{init_for_run}"
                rows.append(
                    {
                        "run_id": run_id,
                        "norm_mode": norm_mode,
                        "init_method": init_for_run,
                        "converged": bool(row["als_converged"]),
                        "iterations": int(row["als_iter"]),
                        "lof_pct": float(row["als_lof_pct"]),
                        "logdet_cov_cancer": logdet_c,
                        "logdet_cov_noncancer": logdet_n,
                        "delta_logdet": delta_logdet,
                        "perm_p_one_sided": perm_p,
                        "mah_median_cancer": float(np.median(d2_c)),
                        "mah_median_noncancer": float(np.median(d2_n)),
                        "mah_mw_p_one_sided": float(mw.pvalue),
                        "supports_higher_cancer_variability": bool(
                            (delta_logdet > 0) and (np.median(d2_c) > np.median(d2_n))
                        ),
                    }
                )

                coeff_cols = {f"coef_{i+1}": coeffs[:, i] for i in range(coeffs.shape[1])}
                coeff_frame = pd.DataFrame(coeff_cols)
                coeff_frame["label"] = np.where(labels == 1, "cancer", "noncancer")
                coeff_frame["run_id"] = run_id
                coeffs_rows.append(coeff_frame)

        metrics_df = pd.DataFrame(rows).sort_values(["norm_mode", "init_method"]).reset_index(drop=True)
        coeffs_long_df = pd.concat(coeffs_rows, ignore_index=True) if coeffs_rows else pd.DataFrame()

    except Exception as exc:
        analysis_error = str(exc)
    return analysis_error, coeffs_long_df, metrics_df


@app.cell(hide_code=True)
def _(analysis_error, metrics_df, mo):
    if analysis_error is not None:
        mo.md(f"❌ Analysis failed: `{analysis_error}`")
    elif metrics_df.empty:
        mo.md("⚠️ No results yet.")
    else:
        mo.md("## Run Metrics")
        mo.ui.table(metrics_df)
    return


@app.cell(hide_code=True)
def _(coeffs_long_df, metrics_df, mo, np, plt):
    if metrics_df.empty or coeffs_long_df.empty:
        mo.md("ℹ️ No coefficient summary yet.")
    else:
        best = metrics_df.sort_values(
            ["supports_higher_cancer_variability", "delta_logdet", "mah_median_cancer"],
            ascending=[False, False, False],
        ).iloc[0]

        run_id = best["run_id"]
        sub = coeffs_long_df[coeffs_long_df["run_id"] == run_id].copy()

        coef_cols = [c for c in sub.columns if c.startswith("coef_")]
        std_tbl = (
            sub.groupby("label")[coef_cols]
            .std()
            .T.rename(columns={"cancer": "std_cancer", "noncancer": "std_noncancer"})
            .reset_index(names="component")
        )

        mo.md(f"## Best Run Snapshot: `{run_id}`")
        mo.ui.table(std_tbl)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(std_tbl))
        w = 0.35
        ax.bar(x - w / 2, std_tbl["std_cancer"], width=w, label="cancer", color="#d62728", alpha=0.8)
        ax.bar(x + w / 2, std_tbl["std_noncancer"], width=w, label="noncancer", color="#1f77b4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(std_tbl["component"])
        ax.set_ylabel("Coefficient std")
        ax.set_title("Coefficient Variability by Group")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        mo.as_html(fig)
        plt.close(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
