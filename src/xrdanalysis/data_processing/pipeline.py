from pickle import dump

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
        self.trained_pipeline = None

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

    def infer_y(self, X, y_column, y_value=None):
        """Infer the y values from the wrangled dataset."""
        if y_value is not None:
            # Return a boolean series where y equals y_value
            return X[y_column] == y_value
        return X[
            y_column
        ]  # Return the entire series if no filtering is needed

    def fit(self, X, y):
        """Initialize and fit the preprocessing and estimator pipeline."""
        # Initialize the pipeline of preprocessing steps and estimator
        pipeline = Pipeline(self.preprocessing_steps + self.estimator)

        # Fit the pipeline
        pipeline.fit(X, y)

        # Store the fitted pipeline
        self.trained_pipeline = pipeline

        return pipeline

    def predict(self, X, wrangle=False):
        """Predict using the trained pipeline."""
        if not self.trained_pipeline:
            raise RuntimeError("Pipeline has not been fitted yet.")

        if wrangle:
            self.wrangle(X)
        # Use the trained pipeline for prediction (preprocessing + estimator)
        return self.trained_pipeline.predict(X)

    def validate(
        self, y_true, y_pred, y_score, metrics=["accuracy", "roc_auc"]
    ):
        """Validate the trained estimator on test data using
        specified metrics."""
        # Calculate and return the desired metrics
        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)

        if "roc_auc" in metrics:
            generate_roc_curve(y_true, y_score)
            results["roc_auc"] = roc_auc_score(y_true, y_score)

        if "precision" in metrics:
            results["precision"] = precision_score(y_true, y_pred)

    def train(self, X, y_column, y_value=None, split=True, **split_args):
        """Run the full pipeline: wrangle, split, fit, predict
        and validate on test data."""
        # Wrangle the data
        X_wrangled = self.wrangle(X)

        y = self.infer_y(X_wrangled, y_column, y_value)

        if split:
            # Split the data (with optional arguments for custom splits)
            X_train, X_test, y_train, y_test = self.splitter(
                X_wrangled, y, **split_args
            )
        else:
            X_train = X_wrangled
            y_train = y
            X_test = X_wrangled
            y_test = y

        # Fit the pipeline on training data
        self.fit(X_train, y_train)

        y_pred = self.trained_pipeline.predict(X_test)
        y_score = self.trained_pipeline.predict_proba(X_test)[:, 1]

        # Validate the training results
        self.validate(y_test, y_pred, y_score)
        _, _, _, self.optimal_threshold = calculate_optimal_threshold(
            y_test, y_score
        )

    def export_pipeline(self, wrangle=False, save_path=None):
        "Export the pipeline"
        if not self.trained_pipeline:
            raise RuntimeError("Pipeline has not been fitted yet.")

        if wrangle:
            full_pipeline = Pipeline(
                steps=[
                    *self.data_wrangling_steps,
                    *self.trained_pipeline.steps,
                ]
            )
        else:
            full_pipeline = self.trained_pipeline

        full_pipeline.optimal_threshold = self.optimal_threshold

        if save_path:
            dump(full_pipeline, open(save_path, "wb"))

        return full_pipeline

    def export_predictions(self, data, save_path, wrangle=True):
        if not self.trained_pipeline:
            raise RuntimeError("Pipeline has not been fitted yet.")

        model = self.export_pipeline(wrangle)

        y_score = model.predict_proba(data)[:, 1]
        y_pred = y_score > model.optimal_threshold
        df_saxs_pred = pd.DataFrame(
            data=y_pred, index=data.index, columns=["cancer_diagnosis"]
        )

        df_saxs_pred.to_csv(save_path)
