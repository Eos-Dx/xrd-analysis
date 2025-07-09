import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

    def wrangle(self, data):
        """
        Applies data wrangling steps to the entire dataset.

        :param data: The dataset to apply the wrangling steps to.
        :type data: DataFrame
        :return: The wrangled dataset.
        :rtype: DataFrame
        """
        data_wrangling_pipeline = Pipeline(self.data_wrangling_steps)

        # Apply wrangling pipeline to the full dataset
        data_wrangled = data_wrangling_pipeline.transform(data)

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
        return X[
            y_column
        ]  # Return the entire series if no filtering is needed

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

    def train_estimator(self, X, y):
        """
        Trains the estimator using the provided features and target variable.

        :param X: The feature matrix.
        :type X: DataFrame
        :param y: The target variable.
        :type y: Series
        :return: The trained estimator pipeline.
        :rtype: Pipeline
        """
        # Initialize the pipeline of preprocessing steps and estimator
        estimator_pipeline = Pipeline(self.estimator)

        # Fit the pipeline
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
        if preprocess:
            if not self.trained_preprocessor:
                raise RuntimeError("Preprocessing has not been fitted yet.")
            X = self.preprocess(X)
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
        if preprocess:
            if not self.trained_preprocessor:
                raise RuntimeError("Preprocessing has not been fitted yet.")
            X = self.preprocess(X)
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
        """
        Validate the performance of the trained estimator on test data.

        :param y_true: The true target values.
        :type y_true: pandas.Series
        :param y_score: Predicted probabilities for the positive class.
        :type y_score: numpy.ndarray
        :param metrics: Metrics to compute, e.g., ["accuracy", "roc_auc"].
        :type metrics: list
        :param show_flag: If True, displays the ROC curve. Defaults to False.
        :type show_flag: bool
        :param print_flag: If True, prints the validation results. \
        Defaults to False.
        :type print_flag: bool
        :param min_sensitivity: Minimum sensitivity threshold. \
        Defaults to None.
        :type min_sensitivity: float, optional
        :param min_specificity: Minimum specificity threshold. \
        Defaults to None.
        :type min_specificity: float, optional
        :return: A dictionary containing the computed metric results.
        :rtype: dict
        """
        # Calculate and return the desired metrics
        results = {}
        _, _, _, self.optimal_threshold = calculate_optimal_threshold(
            y_true, y_score, print_flag=print_flag
        )
        y_pred = y_score > self.optimal_threshold
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)

        if "roc_auc" in metrics:
            (sensitivity, specificity, precision, ba_accuracy, threshold) = (
                generate_roc_based_metrics(
                    y_true,
                    y_score,
                    show_flag,
                    min_sensitivity=min_sensitivity,
                    min_specificity=min_specificity,
                )
            )
            results["roc_auc"] = round(roc_auc_score(y_true, y_score) * 100, 1)
            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
            results["precision"] = precision
            results["ba_accuracy"] = ba_accuracy
            results["threshold"] = threshold

        if print_flag:
            print(results)

        self.last_results = results

        return results

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
        **split_args
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
        :param split_args: Additional arguments for the dataset\
        splitter function.
        :type split_args: dict
        """
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
            X_train = self.preprocess(X_train)
            X_test = self.preprocess(X_test)

        estimator = self.train_estimator(X_train, y_train)

        y_score = estimator.predict_proba(X_test)[:, 1]

        # Validate the training results
        self.validate(
            y_test,
            y_score,
            print_flag=print_flag,
            show_flag=show_flag,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
        )

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
        """
        Exports predictions for the given dataset to a CSV file.

        :param data: The dataset to predict on.
        :type data: DataFrame
        :param save_path: The file path to save the predictions.
        :type save_path: str
        :param wrangle: Whether to apply wrangling steps to the data.
        :type wrangle: bool
        :param preprocess: Whether to apply preprocessing steps to the data.
        :type preprocess: bool
        """
        if not self.trained_estimator:
            raise RuntimeError("Estimator has not been fitted yet.")

        model = self.export_pipeline(wrangle, preprocess)

        y_score = self.predict_proba(data, wrangle, preprocess)[:, 1]
        y_pred = y_score > model.optimal_threshold
        df_saxs_pred = pd.DataFrame(
            data=y_pred, index=data.index, columns=["cancer_diagnosis"]
        )

        df_saxs_pred.to_csv(save_path)

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
        """
        Validate the trained estimator on a dataset using specified metrics.

        :param data: The dataset for validation.
        :type data: pandas.DataFrame
        :param y_column: Name of the target variable column. Defaults to None.
        :type y_column: str, optional
        :param y_value: Target variable value to filter on. Defaults to None.
        :type y_value: object, optional
        :param y_data: Predefined target values. Defaults to None.
        :type y_data: pandas.Series, optional
        :param wrangle: If True, apply data wrangling steps. Defaults to False.
        :type wrangle: bool
        :param preprocess: If True, apply preprocessing steps. \
        Defaults to False.
        :type preprocess: bool
        :param metrics: Metrics to compute, e.g., ["accuracy", "roc_auc"].
        :type metrics: list
        :param show_flag: If True, displays the ROC curve. Defaults to False.
        :type show_flag: bool
        :param print_flag: If True, prints the validation results. \
        Defaults to False.
        :type print_flag: bool
        :param min_sensitivity: Minimum sensitivity threshold. \
        Defaults to None.
        :type min_sensitivity: float, optional
        :param min_specificity: Minimum specificity threshold. \
        Defaults to None.
        :type min_specificity: float, optional
        :return: A dictionary containing the computed metric results.
        :rtype: dict
        """
        # Calculate and return the desired metrics
        if y_data is None:
            y_true = self.infer_y(data, y_column, y_value)
        else:
            y_true = y_data
        y_score = self.predict_proba(data, wrangle, preprocess)[:, 1]
        return self.validate(
            y_true,
            y_score,
            metrics,
            show_flag,
            print_flag,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
        )
