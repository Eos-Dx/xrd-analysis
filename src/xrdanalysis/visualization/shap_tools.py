from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def plot_shap_vs_q(
    pipeline,
    df: pd.DataFrame,
    dataset_name: str,
    q_tuple: Optional[Tuple[float, float]] = None,
    max_samples: int = 600,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim_profile: Optional[Tuple[float, float]] = None,
    ylim_scatter: Optional[Tuple[float, float]] = None,
    save_dir: str | Path = "./figures",
    # Optional split controls
    split: bool = False,
    splitter=None,
    y_column: Optional[str] = None,
    y_value=None,
    y_data: Optional[pd.Series] = None,
    # Aggregation/plot controls
    aggregate_abs: bool = False,
    violin_points: int = 50,
    violin_abs: bool = False,
    use_tqdm: bool = True,
    **split_args,
) -> dict:
    """
    Plot SHAP values vs q for a fitted MLPipeline (tree-based estimator) with
    flattener + scaler preprocessing.

    Parameters
    ----------
    pipeline : xrdanalysis.data_processing.pipeline.MLPipeline
        A fitted pipeline. Must have trained_preprocessor and trained_estimator.
    df : pandas.DataFrame
        DataFrame to compute SHAP on (should include 'radial_profile_data'; if it
        includes per-row 'q_range', it's used to determine q-vector length).
    dataset_name : str
        Used in figure titles and filenames.
    q_tuple : (float, float), optional
        Range (q_min, q_max) if df does not store per-row q_range vectors.
    max_samples : int, optional
        Maximum number of samples to include in the SHAP scatter.
    xlim : (float, float), optional
        x-axis limits for both profile and scatter (q-axis).
    ylim_profile : (float, float), optional
        y-axis limits for the mean |SHAP| profile.
    ylim_scatter : (float, float), optional
        y-axis limits for the SHAP scatter.
    save_dir : Path-like, optional
        Directory to save the output figures (created if missing).
    split : bool, optional
        If True, split the wrangled DataFrame and compute SHAP on the test split.
        Splitting is done AFTER wrangling and BEFORE preprocessing (to match pipeline semantics).
    splitter : callable, optional
        Split function with signature (X, y, **split_args). Defaults to pipeline.splitter if None.
    y_column, y_value, y_data : optional
        Target specification for splitting. If y_data is None, y is inferred via pipeline.infer_y(Xw, y_column, y_value).
    **split_args : dict
        Arguments forwarded to the splitter (e.g., test_size, random_state, group_col, stratify_cols).
        NOTE: You can pass random_state as an int (single run) or as an iterable of ints
        (e.g., random_state=(20, 40, 1)) to aggregate SHAP over multiple splits.
    aggregate_abs : bool, optional
        If True, aggregate profiles as mean(|SHAP|) per q. If False (default), use mean(SHAP).
    violin_points : int, optional
        Number of q-positions to show in the violin plot (downsampling along q). Defaults to 50.
    violin_abs : bool, optional
        If True, violin uses |SHAP| values; if False (default), violin uses signed SHAP values.
    use_tqdm : bool, optional
        If True, show a tqdm progress bar over seeds when aggregating.

    Returns
    -------
    dict
        A dictionary with keys: 'q_vec', 'mean_abs_shap', 'shap_matrix',
        'profile_path', 'scatter_path'.
    """
    print("=== SHAP vs q: start ===")

    # 0) Validate fitted pieces
    try:
        trained_pre = pipeline.trained_preprocessor
        trained_est = pipeline.trained_estimator
    except Exception:
        trained_pre = getattr(pipeline, "trained_preprocessor", None)
        trained_est = getattr(pipeline, "trained_estimator", None)

    if trained_pre is None:
        raise RuntimeError(
            "Pipeline.trained_preprocessor is None; call pipeline.train first."
        )
    if trained_est is None:
        raise RuntimeError(
            "Pipeline.trained_estimator is None; call pipeline.train first."
        )

    try:
        # The final fitted model is typically the last step of the estimator pipeline
        model = trained_est.steps[-1][1]
    except Exception as e:
        raise RuntimeError(f"Could not access fitted model from trained_estimator: {e}")

    # 1) Build input DataFrame (wrangle), optionally split to test
    try:
        Xw = pipeline.wrangle(df)
    except Exception as e:
        raise RuntimeError(f"Failed to wrangle data: {e}")

    # Determine seeds for splitting (single or multiple)
    seeds = None
    if split:
        # Prepare y for splitting
        if y_data is None:
            if y_column is None and y_value is None:
                raise RuntimeError(
                    "When split=True, provide y_data or y_column (and optional y_value) to infer y."
                )
            try:
                y = pipeline.infer_y(Xw, y_column, y_value)
            except Exception as e:
                raise RuntimeError(f"Failed to infer y for splitting: {e}")
        else:
            y = y_data
        split_fn = (
            splitter if splitter is not None else getattr(pipeline, "splitter", None)
        )
        if split_fn is None:
            raise RuntimeError("No splitter provided and pipeline has no splitter set.")

        # Seeds extraction
        rs = split_args.get("random_state", None)
        if isinstance(rs, (list, tuple)):
            seeds = list(rs)
        else:
            seeds = [rs]

    else:
        seeds = [None]

    # Prepare accumulation for multi-seed profiles
    profiles = []  # per-seed aggregated profile (per current aggregate_abs setting)
    abs_profiles = (
        []
    )  # per-seed mean(|SHAP|) profile (always computed for violin/overlay)
    last_outputs = None
    q_vec_ref = None
    # For real violin: collect absolute SHAP values at selected q across all seeds
    violin_buffers = None  # will become a list of lists, one per q index
    q_idx_array = None

    multi_seed = len(seeds) > 1
    iterator = seeds
    bar = None
    if use_tqdm and multi_seed and _tqdm is not None:
        bar = _tqdm(seeds, desc="SHAP seeds")
        iterator = bar

    for i_seed, seed in enumerate(iterator):
        if bar is not None:
            try:
                bar.set_postfix(seed=seed)
            except Exception:
                pass
        X_for_shap = Xw
        y_for_shap = None
        X_train = X_test = y_train = y_test = None

        if split:
            local_args = dict(split_args)
            local_args["random_state"] = seed
            try:
                X_train, X_test, y_train, y_test = split_fn(Xw, y, **local_args)
                X_for_shap = X_test
                y_for_shap = y_test
            except Exception as e:
                raise RuntimeError(f"Splitting failed (seed={seed}): {e}")

            # Print split statistics only for single-seed runs
            if not multi_seed:
                print(f"Split statistics (seed={seed}):")
                print(f"  Total samples: {len(Xw)}, Test samples: {len(X_test)}")
                try:
                    print(
                        f"  y_test counts: {y_test.value_counts(dropna=False).to_dict()}"
                    )
                except Exception:
                    import pandas as _pd

                    print(
                        f"  y_test counts: {_pd.Series(y_test).value_counts(dropna=False).to_dict()}"
                    )
                group_col = split_args.get("group_col")
                if group_col and group_col in Xw.columns:
                    total_groups = Xw[group_col].nunique()
                    test_groups = (
                        X_test[group_col].nunique()
                        if group_col in X_test.columns
                        else 0
                    )
                    print(f"  Groups - Total: {total_groups}, Test: {test_groups}")

        # Preprocess using fitted preprocessor
        try:
            X_scaled = pipeline.preprocess(X_for_shap)
        except Exception as e:
            raise RuntimeError(
                f"Failed to preprocess data via fitted preprocessor: {e}"
            )

        # Ensure estimator sees consistent feature names as during training
        feat_names = getattr(pipeline, "feature_names_", None)
        if feat_names is not None and not isinstance(X_scaled, pd.DataFrame):
            X_scaled = pd.DataFrame(
                np.asarray(X_scaled),
                index=getattr(X_for_shap, "index", None),
                columns=feat_names,
            )

        # Ensure ndarray for SHAP if needed later
        if hasattr(X_scaled, "values"):
            X_np = np.asarray(X_scaled.values)
        else:
            X_np = np.asarray(X_scaled)

        if not multi_seed:
            print(f"Feature matrix shape (seed={seed}): {X_np.shape}")
            # Diagnostic stats for SHAP troubleshooting
            try:
                X_for_pred = X_scaled if hasattr(X_scaled, "values") else X_scaled
                proba = pipeline.trained_estimator.predict_proba(X_for_pred)[:, 1]
                print(
                    f"  proba min/max/std: {proba.min():.4f}/{proba.max():.4f}/{proba.std():.4f}"
                )
                col_var = np.var(X_np, axis=0)
                nonzero_var_cols = (col_var > 1e-12).sum()
                print(
                    f"  feature var median: {np.median(col_var):.6f}, nonzero-cols: {nonzero_var_cols}/{X_np.shape[1]}"
                )
                model = trained_est.steps[-1][1]
                fi = getattr(model, "feature_importances_", None)
                if fi is not None:
                    print(
                        f"  FI sum/max/nonzero: {np.sum(fi):.4f}/{np.max(fi):.4f}/{(fi > 1e-12).sum()}"
                    )
            except Exception as e:
                print(f"  Diagnostics failed: {e}")

        # 2) Derive q-vector
        q_vec = None
        try:
            q_candidate = np.asarray(
                X_for_shap.iloc[0]["q_range"]
            )  # per-row q-range from integration
            if q_candidate.ndim == 1 and q_candidate.size == X_np.shape[1]:
                q_vec = q_candidate
                if i_seed == 0:
                    print("q_vec from df.iloc[0]['q_range'].")
        except Exception:
            pass

        if q_vec is None:
            if q_tuple is None:
                raise RuntimeError(
                    "Need q_tuple=(q_min, q_max) if df has no per-row 'q_range' or length mismatch."
                )
            q_vec = np.linspace(float(q_tuple[0]), float(q_tuple[1]), X_np.shape[1])
            if i_seed == 0:
                print(f"q_vec built from q_tuple {q_tuple} with {q_vec.size} points.")

        # Fix any mismatches
        if q_vec.shape[0] != X_np.shape[1]:
            print(
                f"q length {q_vec.shape[0]} != n_features {X_np.shape[1]}, resampling q."
            )
            q_vec = np.linspace(float(q_vec.min()), float(q_vec.max()), X_np.shape[1])

        # 3) Compute SHAP values (tree models)
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_np)
        except Exception as e:
            raise RuntimeError(
                "Failed to compute SHAP values. Ensure a tree-based model (LightGBM/XGBoost) "
                f"and SHAP installed. Original error: {e}"
            )

        # Normalize shap_vals to a 2D matrix S (n_samples, n_features)
        if isinstance(shap_vals, list):
            if len(shap_vals) == 2:
                S = shap_vals[1]  # positive class
                target_desc = "positive class"
            else:
                S = np.mean(np.abs(np.stack(shap_vals, axis=0)), axis=0)
                target_desc = "mean over classes"
        else:
            S = shap_vals
            target_desc = "binary/one-output"

        print(f"SHAP matrix shape (seed={seed}): {S.shape} ({target_desc})")

        # Global q-profiles for this seed
        profile_seed = (
            np.mean(np.abs(S), axis=0) if aggregate_abs else np.mean(S, axis=0)
        )
        profiles.append(profile_seed)
        abs_profiles.append(np.mean(np.abs(S), axis=0))

        # Prepare violin buffers (absolute SHAP distributions across seeds)
        if multi_seed:
            if q_idx_array is None:
                K = int(max(1, min(violin_points, S.shape[1])))
                q_idx_array = np.unique(
                    np.linspace(0, S.shape[1] - 1, K).round().astype(int)
                )
                violin_buffers = [[] for _ in range(len(q_idx_array))]
            S_for_violin = np.abs(S) if violin_abs else S
            # Append SHAP values per selected q (all samples)
            for b, j in enumerate(q_idx_array):
                violin_buffers[b].append(S_for_violin[:, j])

        # Keep last-run details for return
        last_outputs = (q_vec, S, profile_seed)
        if q_vec_ref is None:
            q_vec_ref = q_vec

    # Aggregate across seeds
    profiles_arr = np.vstack(profiles) if len(profiles) > 0 else np.empty((0, 0))
    abs_profiles_arr = (
        np.vstack(abs_profiles) if len(abs_profiles) > 0 else np.empty((0, 0))
    )
    if profiles_arr.size == 0:
        raise RuntimeError("No SHAP profiles computed.")

    mean_profile = np.mean(profiles_arr, axis=0)
    std_profile = (
        np.std(profiles_arr, axis=0, ddof=1)
        if profiles_arr.shape[0] > 1
        else np.zeros_like(mean_profile)
    )
    # Absolute overlay for violin
    mean_abs_profile = (
        np.mean(abs_profiles_arr, axis=0) if abs_profiles_arr.size else mean_profile
    )

    # 5) Plots
    outdir = Path(save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    title = (
        f"SHAP vs q — {dataset_name} | q=[{q_vec_ref.min():.3f}, {q_vec_ref.max():.3f}]"
    )

    # (a) Profile plot with mean ± sigma band (use abs overlay for multi-seed)
    plt.figure(figsize=(9, 4.2))
    if multi_seed:
        plt.plot(q_vec_ref, mean_abs_profile, label="mean |SHAP| (across seeds)")
        # Optional: omit band to keep focus on violin; keep commented if needed
        # plt.fill_between(q_vec_ref, mean_abs_profile - np.std(abs_profiles_arr, axis=0, ddof=1),
        #                  mean_abs_profile + np.std(abs_profiles_arr, axis=0, ddof=1), alpha=0.2, label="±1σ |SHAP|")
        _y_label = "mean |SHAP|"
    else:
        _line_label = "mean |SHAP|" if aggregate_abs else "mean SHAP"
        _y_label = _line_label
        plt.plot(q_vec_ref, mean_profile, label=_line_label)
        plt.fill_between(
            q_vec_ref,
            mean_profile - std_profile,
            mean_profile + std_profile,
            alpha=0.25,
            label="±1σ",
        )
    plt.xlabel("q (nm$^{-1}$)")
    plt.ylabel(_y_label)
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim_profile is not None:
        plt.ylim(ylim_profile)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path_profile = (
        outdir
        / f"shap_q_profile_avg_{dataset_name}_{q_vec_ref.min():.3f}-{q_vec_ref.max():.3f}.png"
    )
    plt.savefig(out_path_profile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved profile with band: {out_path_profile}")

    # (b) Violin plot across seeds (distribution of SHAP per q across all seeds)
    out_path_violin = None
    try:
        if multi_seed and violin_buffers is not None and q_idx_array is not None:
            data = [
                np.concatenate(chunk) for chunk in violin_buffers
            ]  # each chunk: list per seed
            positions = q_vec_ref[q_idx_array]
            K = len(positions)
            plt.figure(figsize=(9, 4.2))
            parts = plt.violinplot(
                data,
                positions=positions,
                widths=(positions.max() - positions.min()) / max(K, 2),
                showmeans=False,
                showextrema=False,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_alpha(0.3)
            plt.plot(
                q_vec_ref,
                mean_abs_profile,
                color="C0",
                linewidth=2,
                label="mean |SHAP|",
            )
            plt.xlabel("q (nm$^{-1}$)")
            plt.ylabel(
                "|SHAP| distribution across seeds"
                if violin_abs
                else "SHAP distribution across seeds"
            )
            plt.title(f"SHAP violin across seeds — {dataset_name}")
            if xlim is not None:
                plt.xlim(xlim)
            if ylim_scatter is not None:
                plt.ylim(ylim_scatter)
            plt.legend(loc="best")
            plt.tight_layout()
            out_path_violin = (
                outdir
                / f"shap_q_violin_{dataset_name}_{q_vec_ref.min():.3f}-{q_vec_ref.max():.3f}.png"
            )
            plt.savefig(out_path_violin, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"Saved violin: {out_path_violin}")
    except Exception as e:
        print(f"Violin skipped: {e}")
        out_path_violin = None

    print("=== SHAP vs q: done ===")

    # Final summary for multi-seed: only print compact info
    if multi_seed:
        print(f"Seeds used: {seeds}")

    return {
        "q_vec": q_vec_ref,
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "profiles_per_seed": profiles,
        "seeds_used": seeds,
        "profile_path": str(out_path_profile),
        "violin_path": str(out_path_violin) if out_path_violin is not None else None,
    }
