==========================================
MLPipeline: Philosophy and Structure
==========================================

The **MLPipeline** is designed to build a modular and flexible machine learning pipeline for data wrangling, preprocessing, training estimators, and validating results. Its primary goals include:

- **One pipeline trains one model**: This ensures the pipeline maintains a single focus, avoiding unnecessary complexity.
- **Each transformer and estimator performs one function**: Keeping each step's purpose singular simplifies debugging and improves maintainability.
- **Reusability of transformers and estimators**: Components of the pipeline (such as transformers or estimators) are designed to be reusable across different pipelines and datasets.
- **Consistency of input/output formats**: All pipeline steps should maintain consistent input/output formats, simplifying integration between steps.

The pipeline is broken down into several key phases: **data wrangling**, **splitting**, **fitting** (including **data processing** and **estimator** steps), **training**, **predicting**, and **exporting**. These phases ensure proper data preparation, training, and evaluation.

.. contents:: Table of Contents
   :local:
   :depth: 3

######################
Pipeline Structure
######################

The `MLPipeline` class is structured around several important components that guide the flow of data from raw form to predictions and validation.

---------------------
Data Wrangling
---------------------

**Data wrangling** refers to transformations that change the values of observations based on information specific to those observations. These transformations can be applied to the entire dataset, regardless of whether the data is intended for training or testing, since each transformation affects only the observation in question.

This step is critical to clean and prepare the raw dataset, ensuring that it is free from noise or inconsistencies. Typical transformations might include:

- Imputing missing values
- Removing outliers
- Converting categorical values to numerical form
- Feature engineering that depends only on the observation

Wrangling transformations should not depend on the global data distribution, making it safe to apply them before splitting the data into training and test sets.

.. code-block:: python

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

---------------------
Data Splitting
---------------------

**Splitting** is performed after the data wrangling phase. The reason for this sequencing is that any data wrangling transformations operate independently on each observation. After wrangling, the dataset can be split into training and test sets, as subsequent data processing steps might depend on global features of the training data.

The `MLPipeline` supports both default splitting methods (such as `train_test_split`) and custom splitting functions for more complex scenarios (e.g., stratified splitting or time series splitting).

.. code-block:: python

    def set_splitter(self, splitter):
        """
        Sets the dataset splitter for the pipeline.

        :param splitter: The method to use for splitting the dataset.
        :type splitter: callable
        """
        self.splitter = splitter

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
        **split_args
    ):

    ...

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

    ...

---------------------
Fitting
---------------------

The **fitting** process consists of two steps: **Data Processing** and **Estimator**.

+++++++++++++++++++++
Data Processing
+++++++++++++++++++++

**Data processing** involves transformations on features that require fitting on training data (e.g., scaling). These transformations are retained and applied to test data during prediction, avoiding data leakage. This phase ensures that the model is trained with proper preprocessing while preventing overfitting.

Transformations included in data processing could be:

- Standardization (e.g., scaling)
- One-hot encoding
- Principal Component Analysis (PCA)
- Feature selection techniques

.. code-block:: python

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


+++++++++++++++++++++
Estimator
+++++++++++++++++++++

The **estimator** is the machine learning model that is trained on the preprocessed data. This step follows preprocessing and forms the final stage of the pipeline. The estimator can be any model from `scikit-learn`, such as logistic regression, decision trees, or ensemble methods.

After the preprocessing steps are applied to the training data, the estimator is fitted to the processed data.

.. code-block:: python

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


############################################
Working with MLPipeline
############################################

------
Train
------

The `train` function is responsible for running the full pipeline, from wrangling to fitting and validation. It performs the following steps:

1. **Wrangle** the raw dataset using the defined wrangling transformations.
2. **Infer the y values** from the cleaned dataset based on the `y_column` and optional `y_value`. This step helps to isolate the target variable from the feature set, which may involve filtering rows or extracting a specific column.
    - *Side note*: The `infer_y` function allows flexibility in how the target variable is derived from the dataset.
3. **Split** the data into training and testing subsets using the chosen splitting strategy.
4. **Fit** the pipeline (preprocessing and estimator) on the training data.
5. **Validate** the model performance on the test data.

.. code-block:: python

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
        **split_args
    ):
        """
        Runs the full pipeline: wrangles, splits, fits, predicts, and validates
        on test data.

        :param X: The dataset to train on.
        :type X: DataFrame
        :param y_column: The column containing the target variable.
        :type y_column: str
        :param y_value: The value of the target variable to filter by. \
        Defaults to None.
        :type y_value: object, optional
        :param y_data: Predefined y values for the dataset. Defaults to None.
        :type y_data: Series, optional
        :param wrangle: Whether to apply wrangling steps to the data.
        :type wrangle: bool
        :param split: Whether to split the dataset into training and test sets.
        :type split: bool
        :param preprocess: Whether to apply preprocessing steps.
        :type preprocess: bool
        :param print_flag: Whether to print validation results. \
        Defaults to True.
        :type print_flag: bool
        :param show_flag: Whether to display the ROC curve. Defaults to False.
        :type show_flag: bool
        :param split_args: Additional arguments for the splitter function.
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

        y_pred = estimator.predict(X_test)
        y_score = estimator.predict_proba(X_test)[:, 1]

        # Validate the training results
        self.validate(
            y_test, y_pred, y_score, print_flag=print_flag, show_flag=show_flag
        )
        _, _, _, self.optimal_threshold = calculate_optimal_threshold(
            y_test, y_score
        )

------------------
Predictions
------------------

The `predict` and `predict_proba` functions utilize the trained pipeline to make predictions on new or test data. If specified, the function can also wrangle the input data before making predictions.

Steps include:

1. Ensures the pipeline has been fitted before prediction.
2. Optionally **wrangles** the input data if `wrangle=True`.
3. Uses the trained pipeline to **predict** the target values.

.. code-block:: python

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

####################
Validation
####################

The `validate` function evaluates the trained model's performance on a test dataset using specified evaluation metrics. Key steps include:

1. **Calculating accuracy**: If the `"accuracy"` metric is selected, the accuracy score is computed using `accuracy_score`.
2. **Generating ROC curve and calculating AUC**: For the `"roc_auc"` metric, the function generates a ROC curve and calculates the area under the curve (AUC) using `generate_roc_curve` and `roc_auc_score`.
3. **Calculating precision**: If the `"precision"` metric is chosen, the precision score is computed using `precision_score`.

The results for the chosen metrics are returned in a dictionary.

.. code-block:: python

    def validate(
        self,
        y_true,
        y_pred,
        y_score,
        metrics=["accuracy", "roc_auc"],
        show_flag=False,
        print_flag=False,
    ):
        """
        Validates the trained estimator on test data using specified metrics.

        :param y_true: The true target values.
        :type y_true: Series
        :param y_pred: The predicted target values.
        :type y_pred: Series
        :param y_score: The predicted probabilities for the positive class.
        :type y_score: ndarray
        :param metrics: The metrics to compute. \
        Defaults to ["accuracy", "roc_auc"].
        :type metrics: list
        :param show_flag: Whether to display the ROC curve. Defaults to False.
        :type show_flag: bool
        :param print_flag: Whether to print the validation results. \
        Defaults to False.
        :type print_flag: bool
        """
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


############################################
Export
############################################

--------------------
Exporting Pipeline
--------------------

The `export_pipeline` function allows users to export the fitted pipeline, either including the wrangling steps or just the trained parts (preprocessing + estimator). The pipeline can be serialized and saved to a file for later use.

Steps include:

1. Ensuring the pipeline has been fitted.
2. Wrangling the input data if needed.
3. Serializing the full pipeline (or parts of it) using Python’s `pickle` module.

.. code-block:: python

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

---------------------------------------
Exporting Predictions
---------------------------------------

The `export_predictions` function exports the model's predictions for a given dataset to a CSV file. The process involves:

1. **Ensuring the pipeline has been trained**: It first checks whether the pipeline has been fitted, raising an error if it has not.
2. **Optionally wrangling the input data**: If `wrangle=True`, the data wrangling steps are applied before prediction.
3. **Generating predictions**: The function computes the probability scores for the input data using the pipeline's `predict_proba` method, then applies a threshold to convert these scores into binary predictions (e.g., `cancer_diagnosis`).
4. **Exporting to CSV**: The predictions are saved as a CSV file at the specified `save_path`.

.. code-block:: python

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

        y_score = model.predict_proba(data)[:, 1]
        y_pred = y_score > model.optimal_threshold
        df_saxs_pred = pd.DataFrame(
            data=y_pred, index=data.index, columns=["cancer_diagnosis"]
        )

        df_saxs_pred.to_csv(save_path)
