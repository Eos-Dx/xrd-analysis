"""Private Optuna tuning and repeated-run evaluation helpers."""

from __future__ import annotations

import numpy as np


def tune_estimator_optuna(
    pipeline,
    X,
    y_column,
    y_value=None,
    y_data=None,
    *,
    wrangle=True,
    split=True,
    preprocess=True,
    n_trials: int = 30,
    timeout=None,
    direction: str = "minimize",
    param_space=None,
    study_name: str = None,
    sampler=None,
    pruner=None,
    show_flag: bool = False,
    print_flag: bool = False,
    threshold_range: tuple = None,
    penalty_weight: float = 0.0,
    variability_n_runs: int = 0,
    variability_seed: int = 0,
    variability_vary_split: bool = True,
    variability_vary_estimator: bool = True,
    **split_args,
):
    """Run the established in-place Optuna tuning workflow for ``pipeline``."""
    try:
        import optuna  # type: ignore
    except Exception as e:
        raise ImportError(
            "Optuna is required for tuning. Please install it with `pip install optuna`."
        ) from e
    if not pipeline.estimator or len(pipeline.estimator) == 0:
        raise RuntimeError("Estimator is not set in the pipeline.")
    est_name, est_obj = pipeline.estimator[-1]

    def default_lgbm_space(trial):
        params = {}
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-3, 3e-1, log=True
        )
        params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        max_depth = params["max_depth"]
        params["num_leaves"] = trial.suggest_int(
            "num_leaves",
            16,
            min(512, (1 << max_depth) if max_depth and max_depth > 0 else 512),
        )
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
        params["min_split_gain"] = trial.suggest_float("min_split_gain", 0.0, 0.3)
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 2.0)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 2.0)
        return params

    def sample_params(trial):
        if param_space is not None:
            return param_space(trial)
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            if isinstance(est_obj, LGBMClassifier):
                return default_lgbm_space(trial)
        except Exception:
            pass
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        if hasattr(est_obj, "n_estimators"):
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
        if hasattr(est_obj, "learning_rate"):
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 1e-3, 3e-1, log=True
            )
        return params

    base_params = getattr(est_obj, "get_params", lambda **k: {})(deep=False)

    def threshold_penalty(thr, rng):
        if rng is None or thr is None:
            return 0.0
        lower, upper = rng
        try:
            thr = float(thr)
        except Exception:
            return 0.0
        return (
            0.0
            if lower <= thr <= upper
            else lower - thr if thr < lower else thr - upper
        )

    def objective(trial):
        params = sample_params(trial)
        split_args_trial = dict(split_args)
        rs_split = params.pop("random_state_split", None)
        if rs_split is not None:
            split_args_trial["random_state"] = rs_split
        rs_est = params.pop("random_state_est", None)
        if rs_est is not None:
            params["random_state"] = rs_est
        for key in ["random_state", "class_weight", "verbose"]:
            if key in base_params and key not in params:
                params[key] = base_params[key]
        if hasattr(est_obj, "set_params"):
            est_obj.set_params(**params)
        else:
            pipeline.estimator[-1] = (est_name, est_obj)
        results = pipeline.train(
            X,
            y_column,
            y_value=y_value,
            y_data=y_data,
            wrangle=wrangle,
            split=split,
            preprocess=preprocess,
            print_flag=False,
            show_flag=False,
            print_split_summary=False,
            **split_args_trial,
        )
        try:
            roc = (
                float(results.get("roc_auc")) / 100.0
                if results.get("roc_auc") is not None
                else 0.0
            )
            sen = (
                float(results.get("sensitivity")) / 100.0
                if results.get("sensitivity") is not None
                else 0.0
            )
            spe = (
                float(results.get("specificity")) / 100.0
                if results.get("specificity") is not None
                else 0.0
            )
        except Exception:
            roc, sen, spe = 0.0, 0.0, 0.0
        return float(
            3.0
            - roc
            - spe
            - sen
            + float(penalty_weight)
            * float(
                threshold_penalty(
                    getattr(pipeline, "optimal_threshold", None), threshold_range
                )
            )
        )

    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner, study_name=study_name
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_trial.params
    for key in ["random_state", "class_weight", "verbose"]:
        if key in base_params and key not in best_params:
            best_params[key] = base_params[key]
    if hasattr(est_obj, "set_params"):
        est_obj.set_params(**best_params)
    best_results = pipeline.train(
        X,
        y_column,
        y_value=y_value,
        y_data=y_data,
        wrangle=wrangle,
        split=split,
        preprocess=preprocess,
        print_flag=print_flag,
        show_flag=show_flag,
        print_split_summary=False,
        **split_args,
    )
    variability = None
    if int(variability_n_runs) > 0:
        try:
            variability = pipeline.evaluate_variability(
                X,
                y_column,
                y_value=y_value,
                y_data=y_data,
                wrangle=wrangle,
                split=split,
                preprocess=preprocess,
                n_runs=int(variability_n_runs),
                seed=int(variability_seed),
                vary_split=bool(variability_vary_split),
                vary_estimator=bool(variability_vary_estimator),
                **split_args,
            )
        except Exception:
            variability = None
    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "best_results": best_results,
        "variability": variability,
        "study": study,
    }


def evaluate_variability(
    pipeline,
    X,
    y_column,
    y_value=None,
    y_data=None,
    *,
    wrangle=True,
    split=True,
    preprocess=True,
    n_runs: int = 20,
    seed: int = 0,
    vary_split: bool = True,
    vary_estimator: bool = True,
    show_flag: bool = False,
    print_flag: bool = False,
    **split_args,
):
    """Run repeated public ``pipeline.train`` calls and aggregate their metrics."""
    if not pipeline.estimator or len(pipeline.estimator) == 0:
        raise RuntimeError("Estimator is not set in the pipeline.")
    _est_name, est_obj = pipeline.estimator[-1]
    try:
        base_params = est_obj.get_params(deep=False)
    except Exception:
        base_params = {}
    base_split_rs, base_est_rs = split_args.get("random_state", None), base_params.get(
        "random_state", None
    )
    roc_list, sen_list, spe_list, thr_list = [], [], [], []
    for i in range(int(n_runs)):
        run_split_rs = (
            (seed + i if base_split_rs is None else int(base_split_rs) + i)
            if vary_split
            else base_split_rs
        )
        run_est_rs = (
            (seed + 1000 + i if base_est_rs is None else int(base_est_rs) + i)
            if vary_estimator
            else base_est_rs
        )
        try:
            if run_est_rs is not None and hasattr(est_obj, "set_params"):
                est_obj.set_params(random_state=run_est_rs)
        except Exception:
            pass
        split_args_trial = dict(split_args)
        if run_split_rs is not None:
            split_args_trial["random_state"] = run_split_rs
        try:
            results = pipeline.train(
                X,
                y_column,
                y_value=y_value,
                y_data=y_data,
                wrangle=wrangle,
                split=split,
                preprocess=preprocess,
                print_flag=False,
                show_flag=False,
                print_split_summary=False,
                **split_args_trial,
            )
        except Exception:
            continue
        for values, key in [
            (roc_list, "roc_auc"),
            (sen_list, "sensitivity"),
            (spe_list, "specificity"),
        ]:
            if results.get(key) is not None:
                values.append(float(results[key]))
        threshold = results.get(
            "threshold", getattr(pipeline, "optimal_threshold", None)
        )
        if threshold is not None:
            try:
                thr_list.append(float(threshold))
            except Exception:
                pass

    def stats(values):
        if not values:
            return {"mean": None, "std": None}
        arr = np.array(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }

    return {
        "runs": len(roc_list),
        "roc_auc_values": roc_list,
        "sensitivity_values": sen_list,
        "specificity_values": spe_list,
        "threshold_values": thr_list,
        "roc_auc": stats(roc_list),
        "sensitivity": stats(sen_list),
        "specificity": stats(spe_list),
        "threshold": stats(thr_list),
    }
