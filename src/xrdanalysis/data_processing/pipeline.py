from joblib import dump

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xrdanalysis.data_processing.utility_functions import (
    calculate_optimal_threshold,
    generate_roc_curve,
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

    def add_data_wrangling_step(self, name, transformer, position=None):
        if position is not None:
            self.data_wrangling_steps.insert(position, (name, transformer))
        else:
            self.data_wrangling_steps.append((name, transformer))

    def add_preprocessing_step(self, name, transformer, position=None):
        if position is not None:
            self.preprocessing_steps.insert(position, (name, transformer))
        else:
            self.preprocessing_steps.append((name, transformer))

    def set_estimator(self, estimator):
        self.estimator = [estimator]

    def set_splitter(self, splitter):
        self.splitter = splitter

    def wrangle(self, data):
        """Apply the wrangling steps to the entire dataset."""
        data_wrangling_pipeline = Pipeline(self.data_wrangling_steps)

        # Apply wrangling pipeline to the full dataset
        data_wrangled = data_wrangling_pipeline.transform(data)

        return data_wrangled

    def preprocess(self, data):
        """Apply the prossecing steps to the entire dataset."""
        return self.trained_preprocessor.transform(data)

    def wrangle_transform(self, data):
        """Apply the wrangling steps to the entire dataset."""
        data_all = Pipeline(self.data_wrangling_steps + self.preprocessing_steps)

        self.train_preprocessor(self.wrangle(data))

        return data_all.transform(data)

    def wrangle_preprocess_train_transform(self, data):
        """Apply the wrangling steps to the entire dataset."""
        data_wrangling_preprocessing_pipeline = Pipeline(
            self.data_wrangling_steps + self.preprocessing_steps
        )

        # Apply wrangling pipeline to the full dataset
        data_wrangled_preprocessed = (
            data_wrangling_preprocessing_pipeline.fit_transform(data)
        )

        return data_wrangled_preprocessed

    def infer_y(self, X, y_column, y_value=None):
        """Infer the y values from the wrangled dataset."""
        if y_value is not None:
            # Return a boolean series where y equals y_value
            return X[y_column] == y_value
        return X[
            y_column
        ]  # Return the entire series if no filtering is needed

    def train_preprocessor(self, data):
        """Train preprocessor."""
        data_preprocessing_pipeline = Pipeline(self.preprocessing_steps)

        # Apply wrangling pipeline to the full dataset
        data_preprocessing_pipeline.fit(data)

        self.trained_preprocessor = data_preprocessing_pipeline

        return self.trained_preprocessor

    def train_estimator(self, X, y):
        """Initialize and fit the preprocessing and estimator pipeline."""
        # Initialize the pipeline of preprocessing steps and estimator
        estimator_pipeline = Pipeline(self.estimator)

        # Fit the pipeline
        estimator_pipeline.fit(X, y)

        # Store the fitted pipeline
        self.trained_estimator = estimator_pipeline

        return self.trained_estimator

    def predict(self, X, wrangle=False, preprocess=True):
        """Predict using the trained pipeline."""
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")
        X = X.copy()
        if wrangle:
            X = self.wrangle(X)
        if preprocess:
            if not self.trained_preprocessor:
                raise RuntimeError("Preprocessing has not been fitted yet.")
            X = self.trained_preprocessor.transform(X)
        # Use the trained pipeline for prediction (preprocessing + estimator)
        return self.trained_estimator.predict(X)

    def predict_proba(self, X, wrangle=False, preprocess=True):
        """Predict using the trained pipeline."""
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")
        X = X.copy()
        if wrangle:
            X = self.wrangle(X)
        if preprocess:
            if not self.trained_preprocessor:
                raise RuntimeError("Preprocessing has not been fitted yet.")
            X = self.preprocess(X)
        # Use the trained pipeline for prediction (preprocessing + estimator)
        return self.trained_estimator.predict_proba(X)

    def validate(self, y_true, y_pred, y_score, metrics=["accuracy", "roc_auc"],
                 show_flag=False, print_flag=False
                 ):
        """Validate the trained estimator on test data using
        specified metrics."""
        # Calculate and return the desired metrics
        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)

        if "roc_auc" in metrics:
            if show_flag:
                generate_roc_curve(y_true, y_score)
            results["roc_auc"] = roc_auc_score(y_true, y_score)

        if "precision" in metrics:
            results["precision"] = precision_score(y_true, y_pred)
        if print_flag:
            print(results)

    def train(self,
              X,
              y_column,
              y_value=None,
              y_data=None,
              wrangle=True,
              split=True,
              preprocess=True,
              print_flag=True,
              show_flag=False,
              **split_args
              ):
        """Run the full pipeline: wrangle, split, fit, predict
        and validate on test data."""
        X = X.copy()
        # Wrangle the data
        if wrangle:
            X = self.wrangle(X)

        if y_data is None:
            y = self.infer_y(X, y_column, y_value)
        else:
            y = y_data

        if split:
            # Split the data (with optional arguments for custom splits)
            X_train, X_test, y_train, y_test = self.splitter(
                X, y, **split_args
            )
        else:
            X_train = X
            y_train = y
            X_test = X
            y_test = y

        # Fit the pipeline on training data
        if preprocess:
            self.train_preprocessor(X_train)
            X_train = self.trained_preprocessor.transform(X_train)
            X_test = self.trained_preprocessor.transform(X_test)

        estimator = self.train_estimator(X_train, y_train)

        y_pred = estimator.predict(X_test)
        y_score = estimator.predict_proba(X_test)[:, 1]

        # Validate the training results
        self.validate(y_test, y_pred, y_score,
                      print_flag=print_flag, show_flag=show_flag)
        _, _, _, self.optimal_threshold = calculate_optimal_threshold(
            y_test, y_score
        )

    def export_pipeline(self, wrangle=False, preprocess=True, save_path=None):
        """Export the pipeline"""
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")

        if wrangle and preprocess:
            full_pipeline = Pipeline(
                steps=[
                    *self.data_wrangling_steps,
                    *self.trained_preprocessor.steps,
                    *self.trained_estimator.steps,
                ]
            )
        elif wrangle:
            full_pipeline = Pipeline(
                steps=[
                    *self.data_wrangling_steps,
                    *self.trained_estimator.steps,
                ]
            )
        elif preprocess:
            full_pipeline = Pipeline(
                steps=[
                    *self.trained_preprocessor.steps,
                    *self.trained_estimator.steps,
                ]
            )
        else:
            full_pipeline = self.trained_estimator

        full_pipeline.optimal_threshold = self.optimal_threshold

        if save_path:
            dump(full_pipeline, save_path)

        return full_pipeline

    def export_predictions(
        self, data, save_path, wrangle=False, preprocess=True
    ):
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")

        model = self.export_pipeline(wrangle, preprocess)

        y_score = model.predict_proba(data)[:, 1]
        y_pred = y_score > model.optimal_threshold
        df_saxs_pred = pd.DataFrame(
            data=y_pred, index=data.index, columns=["cancer_diagnosis"]
        )

        df_saxs_pred.to_csv(save_path)
