from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xrdanalysis.data_processing import (
    FaultyPixelDetector,
    MeasurementTypeClassifier,
    _pipeline_experimentation,
    _pipeline_export,
    _pipeline_validation,
)
from xrdanalysis.data_processing._pipeline_diagnostics import emit_split_summary

# Retain historical validation dependency aliases during the compatibility window.
from xrdanalysis.data_processing._pipeline_validation import *  # noqa: F401,F403
from xrdanalysis.data_processing.detector_joining import join_detectors
from xrdanalysis.data_processing.utility_functions import (
    calculate_optimal_threshold,
    generate_roc_based_metrics,
)


class MLPipeline:
    """Builds a modular pipeline for data wrangling, preprocessing,
    and estimator training."""

    def __init__(
        self,
        data_wrangling_steps=None,
        splitter=train_test_split,
        preprocessing_steps=None,
        estimator=None,
    ):
        """
        Initializes the MLPipeline with specified steps for data wrangling,
        splitting, preprocessing, and estimator training.

        :param data_wrangling_steps: Steps for data wrangling. \
        Defaults to None.
        :type data_wrangling_steps: list
        :param splitter: The method for splitting the dataset. \
        Defaults to train_test_split.
        :type splitter: callable
        :param preprocessing_steps: Steps for data preprocessing. \
        Defaults to None.
        :type preprocessing_steps: list
        :param estimator: The estimator for training. Defaults to None.
        :type estimator: object
        """
        self.data_wrangling_steps = (
            data_wrangling_steps if data_wrangling_steps is not None else []
        )
        self.splitter = splitter
        self.preprocessing_steps = (
            preprocessing_steps if preprocessing_steps is not None else []
        )
        self.estimator = [estimator]
        self.trained_preprocessor = None
        self.trained_estimator = None
        self.feature_names_ = None
        self.target_column_ = None
        self.sample_weight_col_ = None

    def add_data_wrangling_step(self, name, transformer, position=None):
        """
        Adds a data wrangling step to the pipeline.

        :param name: The name of the data wrangling step.
        :type name: str
        :param transformer: The transformer function to apply in the step.
        :type transformer: callable
        :param position: The position to insert the step. \
        If None, adds to the end.
        :type position: int, optional
        """
        if position is not None:
            self.data_wrangling_steps.insert(position, (name, transformer))
        else:
            self.data_wrangling_steps.append((name, transformer))

    def add_preprocessing_step(self, name, transformer, position=None):
        """
        Adds a preprocessing step to the pipeline.

        :param name: The name of the preprocessing step.
        :type name: str
        :param transformer: The transformer function to apply in the step.
        :type transformer: callable
        :param position: The position to insert the step. \
        If None, adds to the end.
        :type position: int, optional
        """
        if position is not None:
            self.preprocessing_steps.insert(position, (name, transformer))
        else:
            self.preprocessing_steps.append((name, transformer))

    def set_estimator(self, estimator):
        """
        Sets the estimator for the pipeline.

        :param estimator: The estimator to be used for model training.
        :type estimator: object
        """
        self.estimator = [estimator]

    def set_splitter(self, splitter):
        """
        Sets the dataset splitter for the pipeline.

        :param splitter: The method to use for splitting the dataset.
        :type splitter: callable
        """
        self.splitter = splitter

    def wrangle(self, data, stat=False):
        """
        Applies data wrangling steps to the entire dataset.

        :param data: The dataset to apply the wrangling steps to.
        :type data: DataFrame
        :param stat: If True, returns tuple (data, stats_dict). Defaults to False.
        :type stat: bool
        :return: The wrangled dataset, or tuple (dataset, stats) if stat=True.
        :rtype: DataFrame or tuple
        """
        # If no wrangling steps, return data as-is
        if not self.data_wrangling_steps:
            return (data, {}) if stat else data

        data_wrangling_pipeline = Pipeline(self.data_wrangling_steps)

        # Prefer fit_transform for sklearn>=1.8 compatibility.
        # Calling Pipeline.transform() after fit() now triggers stricter
        # fitted/tag checks on the final transformer, which breaks older
        # stateless custom transformers. Wrangling steps are designed to run
        # on the full dataset, so a single fit_transform is the correct path.
        try:
            data_wrangled = data_wrangling_pipeline.fit_transform(data)
        except AttributeError:
            # Fallback for transformers that only implement transform.
            data_wrangled = data
            for _step_name, _transformer in self.data_wrangling_steps:
                if hasattr(_transformer, "fit_transform"):
                    try:
                        data_wrangled = _transformer.fit_transform(data_wrangled)
                        continue
                    except TypeError:
                        data_wrangled = _transformer.fit_transform(data_wrangled, None)
                        continue

                if hasattr(_transformer, "fit"):
                    _transformer.fit(data_wrangled)
                data_wrangled = _transformer.transform(data_wrangled)

        # Collect stats from transformers if requested
        if stat:
            stats = {}
            for step_name, transformer in self.data_wrangling_steps:
                # Try to get stats from transformer
                if hasattr(transformer, "stats_"):
                    stats[step_name] = transformer.stats_
                elif hasattr(transformer, "get_stats"):
                    try:
                        stats[step_name] = transformer.get_stats()
                    except Exception:
                        pass
            return data_wrangled, stats

        return data_wrangled

    def preprocess(self, data):
        """Apply the processing steps to the entire dataset.

        :param data: The dataset to preprocess.
        :type data: DataFrame
        :return: The preprocessed dataset.
        :rtype: DataFrame
        """
        return self.trained_preprocessor.transform(data)

    def train_preprocess(self, data):
        """Train preprocessor and apply the processing steps to the entire \
        dataset.

        :param data: The dataset to preprocess.
        :type data: DataFrame
        :return: The preprocessed dataset.
        :rtype: DataFrame
        """

        self.train_preprocessor(data)

        preprocessed_data = self.preprocess(data)

        return preprocessed_data

    def transform(self, data, stat=False):
        """
        Applies wrangling steps to the dataset.

        :param data: The dataset to transform.
        :type data: DataFrame
        :param stat: If True, returns tuple (data, stats_dict). Defaults to False.
        :type stat: bool
        :return: The transformed dataset, or tuple (dataset, stats) if stat=True.
        :rtype: DataFrame or tuple
        """
        return self.wrangle(data, stat=stat)

    def wrangle_preprocess_transform(self, data, train=True):
        """
        Applies both wrangling and preprocessing steps to the dataset.

        :param data: The dataset to transform.
        :type data: DataFrame
        :param train: Whether to train the preprocessor on the data.
        :type train: bool
        :return: The transformed dataset.
        :rtype: DataFrame
        """
        wrangled_data = self.wrangle(data)

        if train:
            self.train_preprocessor(wrangled_data)

        preprocessed_data = self.preprocess(wrangled_data)

        return preprocessed_data

    def infer_y(self, X, y_column, y_value=None):
        """
        Infers the y values from the dataset based on a column and an optional
        filter value.

        :param X: The dataset.
        :type X: DataFrame
        :param y_column: The column containing the target variable.
        :type y_column: str
        :param y_value: The value to filter y values by. Defaults to None.
        :type y_value: object, optional
        :return: The inferred y values.
        :rtype: Series
        """
        if y_value is not None:
            # Return a boolean series where y equals y_value
            return X[y_column] == y_value
        return X[y_column]  # Return the entire series if no filtering is needed

    def train_preprocessor(self, data):
        """
        Trains the preprocessing pipeline on the dataset.

        :param data: The dataset to train the preprocessor on.
        :type data: DataFrame
        :return: The trained preprocessor pipeline.
        :rtype: Pipeline
        """
        data_preprocessing_pipeline = Pipeline(self.preprocessing_steps)

        # Apply wrangling pipeline to the full dataset
        data_preprocessing_pipeline.fit(data)

        self.trained_preprocessor = data_preprocessing_pipeline

        return self.trained_preprocessor

    def train_estimator(self, X, y, sample_weight=None):
        """
        Trains the estimator using the provided features and target variable.

        :param X: The feature matrix.
        :type X: DataFrame
        :param y: The target variable.
        :type y: Series
        :param sample_weight: Sample weights for training.
        :type sample_weight: array-like, optional
        :return: The trained estimator pipeline.
        :rtype: Pipeline
        """
        # Initialize the pipeline of preprocessing steps and estimator
        estimator_pipeline = Pipeline(self.estimator)

        # Fit the pipeline with optional sample weights
        if sample_weight is not None:
            estimator_pipeline.fit(
                X, y, **{f"{self.estimator[0][0]}__sample_weight": sample_weight}
            )
        else:
            estimator_pipeline.fit(X, y)

        # Store the fitted pipeline
        self.trained_estimator = estimator_pipeline

        return self.trained_estimator

    def predict(self, X, wrangle=False, preprocess=True):
        """
        Predicts outcomes using the trained estimator.

        :param X: The dataset to predict on.
        :type X: DataFrame
        :param wrangle: Whether to apply wrangling steps to the data.
        :type wrangle: bool
        :param preprocess: Whether to apply preprocessing steps to the data.
        :type preprocess: bool
        :return: The predicted values.
        :rtype: Series
        """
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")
        X = X.copy()
        if wrangle:
            X = self.wrangle(X)
        for column in (
            getattr(self, "target_column_", None),
            getattr(self, "sample_weight_col_", None),
        ):
            if column is not None and column in X.columns:
                X = X.drop(columns=[column])
        if preprocess:
            if not self.trained_preprocessor:
                raise RuntimeError("Preprocessing has not been fitted yet.")
            X = self.preprocess(X)
        # Ensure consistent feature names if estimator was fitted with names
        if not isinstance(X, pd.DataFrame) and getattr(self, "feature_names_", None):
            import numpy as _np

            X = pd.DataFrame(
                _np.asarray(X),
                index=getattr(X, "index", None),
                columns=self.feature_names_,
            )
        # Use the trained pipeline for prediction (preprocessing + estimator)
        return self.trained_estimator.predict(X)

    def predict_proba(self, X, wrangle=False, preprocess=True):
        """
        Predicts class probabilities using the trained estimator.

        :param X: The dataset to predict on.
        :type X: DataFrame
        :param wrangle: Whether to apply wrangling steps to the data.
        :type wrangle: bool
        :param preprocess: Whether to apply preprocessing steps to the data.
        :type preprocess: bool
        :return: The predicted probabilities.
        :rtype: ndarray
        """
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")
        X = X.copy()
        if wrangle:
            X = self.wrangle(X)
        for column in (
            getattr(self, "target_column_", None),
            getattr(self, "sample_weight_col_", None),
        ):
            if column is not None and column in X.columns:
                X = X.drop(columns=[column])
        if preprocess:
            if not self.trained_preprocessor:
                raise RuntimeError("Preprocessing has not been fitted yet.")
            X = self.preprocess(X)
        # Ensure consistent feature names if estimator was fitted with names
        if not isinstance(X, pd.DataFrame) and getattr(self, "feature_names_", None):
            import numpy as _np

            X = pd.DataFrame(
                _np.asarray(X),
                index=getattr(X, "index", None),
                columns=self.feature_names_,
            )
        # Use the trained pipeline for prediction (preprocessing + estimator)
        return self.trained_estimator.predict_proba(X)

    def validate(
        self,
        y_true,
        y_score,
        metrics=["accuracy", "roc_auc"],
        show_flag=False,
        print_flag=False,
        min_sensitivity=None,
        min_specificity=None,
    ):
        """Validate binary or multiclass probability scores."""
        return _pipeline_validation.validate(
            self,
            y_true,
            y_score,
            metrics,
            show_flag,
            print_flag,
            min_sensitivity,
            min_specificity,
            calculate_optimal_threshold=calculate_optimal_threshold,
            generate_roc_based_metrics=generate_roc_based_metrics,
        )

    def train(
        self,
        X,
        y_column,
        y_value=None,
        y_data=None,
        wrangle=True,
        split=True,
        preprocess=True,
        print_flag=True,
        show_flag=False,
        min_sensitivity=None,
        min_specificity=None,
        print_split_summary: bool = False,
        sample_weight_col: Optional[str] = None,
        **split_args,
    ):
        """
        Execute the full training pipeline, including data wrangling, \
        splitting, preprocessing, fitting, and validation.

        :param X: The input dataset.
        :type X: pandas.DataFrame
        :param y_column: Name of the target variable column.
        :type y_column: str
        :param y_value: Target variable value to filter on. Defaults to None.
        :type y_value: object, optional
        :param y_data: Predefined target values. Defaults to None.
        :type y_data: pandas.Series, optional
        :param wrangle: If True, apply data wrangling steps. Defaults to True.
        :type wrangle: bool
        :param split: If True, split the dataset into training and test sets. \
        Defaults to True.
        :type split: bool
        :param preprocess: If True, apply preprocessing steps. \
        Defaults to True.
        :type preprocess: bool
        :param print_flag: If True, print validation results. Defaults to True.
        :type print_flag: bool
        :param show_flag: If True, display the ROC curve. Defaults to False.
        :type show_flag: bool
        :param min_sensitivity: Minimum sensitivity threshold. \
        Defaults to None.
        :type min_sensitivity: float, optional
        :param min_specificity: Minimum specificity threshold. \
        Defaults to None.
        :type min_specificity: float, optional
        :param print_split_summary: If True (or if 'print_debug' is passed in \
        split_args), print a summary of the split AFTER splitting and BEFORE \
        preprocessing, including rows, unique groups (if group_col provided), \
        and optional stratification bin counts.
        :type print_split_summary: bool
        :param split_args: Additional arguments for the dataset\
        splitter function (e.g., test_size, random_state, group_col, \
        stratify_cols, print_debug).
        :type split_args: dict
        """
        X = X.copy()
        self.target_column_ = y_column
        self.sample_weight_col_ = sample_weight_col
        # Wrangle the data
        if wrangle:
            X = self.wrangle(X)

        if y_data is None:
            y = self.infer_y(X, y_column, y_value)
        else:
            y = y_data
        if y_column is not None and y_column in X.columns:
            X = X.drop(columns=[y_column])

        if split:
            # Split the data (with optional arguments for custom splits)
            X_train, X_test, y_train, y_test = self.splitter(X, y, **split_args)
        else:
            X_train = X
            y_train = y
            X_test = X
            y_test = y

        if split and print_split_summary:
            emit_split_summary(
                X,
                X_train,
                X_test,
                y_train,
                y_test,
                split_args,
                sample_weight_col,
            )

        # Extract sample weights if specified
        sample_weight_train = None
        if sample_weight_col is not None:
            if sample_weight_col not in X_train.columns:
                raise KeyError(
                    f"Sample weight column '{sample_weight_col}' not found in training data"
                )
            sample_weight_train = X_train[sample_weight_col].values
            # Remove weight column from features before preprocessing
            X_train = X_train.drop(columns=[sample_weight_col])
            X_test = (
                X_test.drop(columns=[sample_weight_col])
                if sample_weight_col in X_test.columns
                else X_test
            )

        # Fit the pipeline on training data
        if preprocess:
            self.train_preprocessor(X_train)
            _train_index = X_train.index if hasattr(X_train, "index") else None
            _test_index = X_test.index if hasattr(X_test, "index") else None
            X_train = self.preprocess(X_train)
            X_test = self.preprocess(X_test)
            # Ensure estimator sees consistent feature names to avoid sklearn warnings
            if not isinstance(X_train, pd.DataFrame):
                import numpy as _np

                n_features = X_train.shape[1]
                if not self.feature_names_:
                    self.feature_names_ = [f"f{i}" for i in range(n_features)]
                X_train = pd.DataFrame(
                    _np.asarray(X_train),
                    index=_train_index,
                    columns=self.feature_names_,
                )
                X_test = pd.DataFrame(
                    _np.asarray(X_test), index=_test_index, columns=self.feature_names_
                )
            else:
                # If a DataFrame, capture and reuse its columns as feature names
                self.feature_names_ = list(X_train.columns)

        estimator = self.train_estimator(
            X_train, y_train, sample_weight=sample_weight_train
        )

        # For multiclass, pass the full matrix; for binary it will still be (n, 2)
        y_score = estimator.predict_proba(X_test)

        # Validate the training results
        results = self.validate(
            y_test,
            y_score,
            print_flag=print_flag,
            show_flag=show_flag,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
        )

        # Attach split summary info for programmatic access
        try:
            split_summary = {
                "n_total": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_label_counts": pd.Series(y_train).value_counts().to_dict(),
                "test_label_counts": pd.Series(y_test).value_counts().to_dict(),
            }
            results["split_summary"] = split_summary
        except Exception:
            pass

        return results

    def export_pipeline(self, wrangle=False, preprocess=True, save_path=None):
        """
        Exports the full pipeline including wrangling, preprocessing, and
        estimator steps.

        :param wrangle: Whether to include wrangling steps in the \
        exported pipeline.
        :type wrangle: bool
        :param preprocess: Whether to include preprocessing steps in the \
        exported pipeline.
        :type preprocess: bool
        :param save_path: The file path to save the exported pipeline. \
        Defaults to None.
        :type save_path: str, optional
        :return: The full pipeline.
        :rtype: Pipeline
        """
        return _pipeline_export.export_binary_pipeline(
            self, wrangle, preprocess, save_path
        )

    def export_predictions(self, data, save_path, wrangle=False, preprocess=True):
        """
        Exports predictions for the given dataset to a CSV file.
        For multiclass models, exports probability matrix; for binary, exports predictions.

        :param data: The dataset to predict on.
        :type data: DataFrame
        :param save_path: The file path to save the predictions.
        :type save_path: str
        :param wrangle: Whether to apply wrangling steps to the data.
        :type wrangle: bool
        :param preprocess: Whether to apply preprocessing steps to the data.
        :type preprocess: bool
        """
        return _pipeline_export.export_binary_predictions(
            self, data, save_path, wrangle, preprocess
        )

    def validate_dataset(
        self,
        data,
        y_column=None,
        y_value=None,
        y_data=None,
        wrangle=False,
        preprocess=False,
        metrics=["accuracy", "roc_auc"],
        show_flag=False,
        print_flag=False,
        min_sensitivity=None,
        min_specificity=None,
    ):
        """Validate a dataset through public prediction and validation seams."""
        return _pipeline_validation.validate_dataset(
            self,
            data,
            y_column,
            y_value,
            y_data,
            wrangle,
            preprocess,
            metrics,
            show_flag,
            print_flag,
            min_sensitivity,
            min_specificity,
        )

    def tune_estimator_optuna(
        self,
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
        """
        Hyperparameter tuning for the current estimator using Optuna.

        - Updates the existing estimator instance in-place via set_params (no new estimator is created)
        - Re-trains the pipeline each trial using the provided split arguments
        - Minimizes objective: 3 - roc_auc - specificity - sensitivity (all in [0,1])

        Parameters
        ----------
        X : DataFrame
            Input dataset.
        y_column : str
            Name of the target column (ignored if y_data is provided).
        y_value : object, optional
            Optional filter value for y via infer_y.
        y_data : Series, optional
            Predefined target values aligned with X.
        wrangle, split, preprocess : bool
            Flags passed to train() controlling pipeline stages.
        n_trials : int
            Number of Optuna trials.
        timeout : int, optional
            Study timeout in seconds.
        direction : str
            Optuna study direction ('minimize' or 'maximize').
        param_space : callable | None
            Callable taking (trial) -> dict of estimator params. If None and
            estimator is LGBMClassifier, a reasonable default search space is used.
            You can include keys 'random_state_est' and 'random_state_split' in the
            returned dict to control estimator.random_state and splitter random_state
            per trial.
        study_name : str, optional
            Name for the study.
        sampler, pruner : optuna.samplers.BaseSampler, optuna.pruners.BasePruner, optional
            Custom sampler/pruner.
        show_flag, print_flag : bool
            Flags passed to validation/plotting.
        threshold_range : tuple(lower, upper), optional
            If provided, add a penalty when the optimal threshold lies outside this range.
            The penalty equals the distance to the nearest bound.
        penalty_weight : float
            Multiplier for the threshold penalty (default 0.0 = disabled).
        variability_n_runs : int
            If > 0, compute variability stats after tuning using the best params.
        variability_seed : int
            Base seed for variability runs.
        variability_vary_split : bool
            If True, vary splitter random_state per variability run.
        variability_vary_estimator : bool
            If True, vary estimator random_state per variability run.
        **split_args : dict
            Passed through to the splitter via train().

        Returns
        -------
        dict with keys: best_params, best_value, best_results, study
        """
        return _pipeline_experimentation.tune_estimator_optuna(
            self,
            X,
            y_column,
            y_value,
            y_data,
            wrangle=wrangle,
            split=split,
            preprocess=preprocess,
            n_trials=n_trials,
            timeout=timeout,
            direction=direction,
            param_space=param_space,
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            show_flag=show_flag,
            print_flag=print_flag,
            threshold_range=threshold_range,
            penalty_weight=penalty_weight,
            variability_n_runs=variability_n_runs,
            variability_seed=variability_seed,
            variability_vary_split=variability_vary_split,
            variability_vary_estimator=variability_vary_estimator,
            **split_args,
        )

    def evaluate_variability(
        self,
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
        """
        Estimate variability of key metrics over repeated runs with different seeds.

        Runs train() n_runs times (re-splitting and/or re-seeding the estimator) and
        aggregates mean and std for ROC AUC, sensitivity, specificity, and threshold.

        Parameters
        ----------
        X : DataFrame
            Input dataset.
        y_column : str
            Target column (ignored if y_data is provided).
        y_value : object, optional
            Optional filter value for infer_y.
        y_data : Series, optional
            Predefined target values aligned with X.
        wrangle, split, preprocess : bool
            Pipeline stage flags.
        n_runs : int
            Number of repeated evaluations.
        seed : int
            Base seed used when split_args/random_state or estimator random_state
            are not provided.
        vary_split : bool
            If True, use a different splitter random_state per run.
        vary_estimator : bool
            If True, use a different estimator random_state per run.
        show_flag, print_flag : bool
            Flags forwarded to validation.
        **split_args : dict
            Splitter arguments (e.g., test_size, random_state, group_col, stratify_cols).

        Returns
        -------
        dict with keys:
            - 'runs'
            - 'roc_auc_values', 'sensitivity_values', 'specificity_values', 'threshold_values'
            - 'roc_auc', 'sensitivity', 'specificity', 'threshold' (each a dict with 'mean', 'std')
        """
        return _pipeline_experimentation.evaluate_variability(
            self,
            X,
            y_column,
            y_value,
            y_data,
            wrangle=wrangle,
            split=split,
            preprocess=preprocess,
            n_runs=n_runs,
            seed=seed,
            vary_split=vary_split,
            vary_estimator=vary_estimator,
            show_flag=show_flag,
            print_flag=print_flag,
            **split_args,
        )


class MLPipelineMulti(MLPipeline):
    """
    Multiclass variant of MLPipeline.

    - Uses estimator.predict_proba to compute class probabilities for all classes
    - Computes multiclass ROC AUC (macro/weighted) and accuracy
    - Does not compute binary thresholds
    """

    def validate_multiclass(
        self,
        y_true,
        y_proba,
        y_pred,
        metrics=None,
        multi_class: str = "ovr",
        average: str = "macro",
        show_flag: bool = False,
        print_flag: bool = False,
    ):
        """Validate multiclass probabilities and predictions."""
        return _pipeline_validation.validate_multiclass(
            self,
            y_true,
            y_proba,
            y_pred,
            metrics,
            multi_class,
            average,
            show_flag,
            print_flag,
        )

    def train(
        self,
        X,
        y_column,
        y_value=None,
        y_data=None,
        wrangle=True,
        split=True,
        preprocess=True,
        print_flag=True,
        show_flag=False,
        metrics=None,
        multi_class: str = "ovr",
        average: str = "macro",
        print_split_summary: bool = False,
        sample_weight_col: Optional[str] = None,
        **split_args,
    ):
        X = X.copy()
        self.target_column_ = y_column
        self.sample_weight_col_ = sample_weight_col
        if wrangle:
            X = self.wrangle(X)

        if y_data is None:
            y = self.infer_y(X, y_column, y_value)
        else:
            y = y_data
        if y_column is not None and y_column in X.columns:
            X = X.drop(columns=[y_column])

        if split:
            X_train, X_test, y_train, y_test = self.splitter(X, y, **split_args)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        if split and print_split_summary:
            emit_split_summary(
                X,
                X_train,
                X_test,
                y_train,
                y_test,
                split_args,
                sample_weight_col,
            )

        # Extract sample weights if specified
        sample_weight_train = None
        if sample_weight_col is not None:
            if sample_weight_col not in X_train.columns:
                raise KeyError(
                    f"Sample weight column '{sample_weight_col}' not found in training data"
                )
            sample_weight_train = X_train[sample_weight_col].values
            # Remove weight column from features before preprocessing
            X_train = X_train.drop(columns=[sample_weight_col])
            X_test = (
                X_test.drop(columns=[sample_weight_col])
                if sample_weight_col in X_test.columns
                else X_test
            )

        if preprocess:
            self.train_preprocessor(X_train)
            X_train = self.preprocess(X_train)
            X_test = self.preprocess(X_test)

        estimator = self.train_estimator(
            X_train, y_train, sample_weight=sample_weight_train
        )
        y_proba = estimator.predict_proba(X_test)
        y_pred = estimator.predict(X_test)

        results = self.validate_multiclass(
            y_true=y_test,
            y_proba=y_proba,
            y_pred=y_pred,
            metrics=metrics,
            multi_class=multi_class,
            average=average,
            show_flag=show_flag,
            print_flag=print_flag,
        )

        # Attach split summary info for programmatic access
        try:
            split_summary = {
                "n_total": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_label_counts": pd.Series(y_train).value_counts().to_dict(),
                "test_label_counts": pd.Series(y_test).value_counts().to_dict(),
            }
            results["split_summary"] = split_summary
        except Exception:
            pass

        return results

    def export_pipeline(self, wrangle=False, preprocess=True, save_path=None):
        """Export full pipeline (no threshold stored for multiclass)."""
        return _pipeline_export.export_multiclass_pipeline(
            self, wrangle, preprocess, save_path
        )

    def export_predictions(self, data, save_path, wrangle=False, preprocess=True):
        return _pipeline_export.export_multiclass_predictions(
            self, data, save_path, wrangle, preprocess
        )

    def validate_dataset(
        self,
        data,
        y_column=None,
        y_value=None,
        y_data=None,
        wrangle=False,
        preprocess=False,
        metrics=None,
        show_flag=False,
        print_flag=False,
        multi_class: str = "ovr",
        average: str = "macro",
    ):
        # Calculate metrics on an arbitrary dataset after training
        """Validate a multiclass dataset through public prediction seams."""
        return _pipeline_validation.validate_multiclass_dataset(
            self,
            data,
            y_column,
            y_value,
            y_data,
            wrangle,
            preprocess,
            metrics,
            show_flag,
            print_flag,
            multi_class,
            average,
        )
